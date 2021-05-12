import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset

from .dgl_gcn import GCN
from .dgl_gcn_v1 import GCN_V1
from .dgl_gcn_v2 import GCN_V2
from .dgl_gat import GAT

class EarlyStopping:
    def __init__(self, patience=10, save='modelname'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save = save

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.save + 'es_checkpoint.pt')

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train_dgl_model(model_name, n_hid, n_classes, dropout, features, adj, labels, idx_train, idx_val, idx_test, \
                    n_epochs=200, aggr="mean", n_neigh_threshold=3, trim_ratio=0.45, trim_compensate=True, cuda=False, save='modelname'):
    # convert to dgl compatible dataset
    features = torch.tensor(features.toarray())
    in_feats = features.shape[1]
    labels = torch.tensor(labels, dtype=torch.long)
    n_nodes = adj.shape[0]
    train_mask = torch.zeros((n_nodes,), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask = torch.zeros((n_nodes,), dtype=torch.bool)
    val_mask[idx_val] = True
    test_mask = torch.zeros((n_nodes,), dtype=torch.bool)
    test_mask[idx_test] = True
    g = dgl.from_scipy(adj)

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # # debug
    # print("features", features.shape, type(features))
    # print(features)
    # print("adj", adj.shape, type(adj))
    # print(adj)
    # print("labels", labels.shape, type(labels))
    # print(labels)
    # print("idx_train", idx_train.shape, type(idx_train))
    # print(idx_train)
    # print("idx_val", idx_val.shape, type(idx_val))
    # print(idx_val)
    # print("idx_test", idx_test.shape, type(idx_test))
    # print(idx_test)

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    support_models = ['GCN', 'GCN_V1', 'GCN_V2', 'GAT']
    if model_name == "GCN" or model_name not in support_models:
        if model_name not in support_models: print("using the default gcn model")
        model = GCN(g, in_feats, n_hid, n_classes,
                    F.relu, n_layers=1, dropout=dropout,
                    aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate)
        
    elif model_name == "GCN_V1":
        print("using GCN_V1")
        model = GCN_V1(g, in_feats, n_hid, n_classes,
                    F.relu, n_layers=1, dropout=dropout,
                    aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate)
    elif model_name == "GCN_V2":
        print("using GCN_V2")
        model = GCN_V2(g, in_feats, n_hid, n_classes,
                    F.relu, n_layers=1, dropout=dropout,
                    aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate)
    elif model_name == "GAT":
        print("using GAT")
        model = GAT(g, in_feats, n_hid, n_classes, dropout=dropout,
                    aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    lr = 0.01
    weight_decay = 5e-4
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    
    stopper = EarlyStopping(save=save)
    
    for epoch in range(n_epochs):
        model.train()
        # forward
        logits = model(features)
        # print(logits[train_mask])
        # print(labels[train_mask])
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, features, labels, val_mask)

        stopper.step(val_acc, model)

    model.load_state_dict(torch.load(save + 'es_checkpoint.pt'))
    test_acc = evaluate(model, features, labels, test_mask)

    return model, test_acc
