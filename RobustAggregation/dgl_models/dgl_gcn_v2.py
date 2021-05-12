"""GCN using basic message passing

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset


class NodeApplyModule_V2(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, bias=True):
        super(NodeApplyModule_V2, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        h = torch.mm(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer_V2(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True, 
                 aggr="mean",
                 n_neigh_threshold=3,
                 trim_ratio=0.45,
                 trim_compensate=True):
        super(GCNLayer_V2, self).__init__()
        self.g = g
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.aggr = aggr
        self.n_neigh_threshold = n_neigh_threshold
        self.trim_ratio = trim_ratio
        self.trim_compensate =trim_compensate
        self.node_update = NodeApplyModule_V2(in_feats, out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        pass
    
    def message_func(self, edge):
        msg = edge.src['h'] * edge.src['norm']
        if self.aggr == "median":
            return {'m': msg, 'self': edge.dst['h'] * edge.dst['norm'], 'input_ft': edge.src['input_ft']}
        else:
            return {'m': msg}

    def reduce_func(self, node):
        if self.aggr == "median":
            n_neigh = node.mailbox['input_ft'].shape[1]
            if n_neigh <= self.n_neigh_threshold: 
                # accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
                accum = torch.sum(node.mailbox['self'], 1) * node.data['norm']
                # accum = (torch.sum(node.mailbox['self'], 1) + torch.sum(node.mailbox['m'], 1)) / 2 * node.data['norm']
            else:
                b = n_neigh // 2 - (1 - n_neigh % 2)
                b = max(min(b, int(n_neigh * self.trim_ratio)), 1)
                # b = max(b, 1)

                # print(n_neigh, b)

                sorted_indices = torch.argsort(node.mailbox['input_ft'], dim=1)
                msg = node.mailbox['m']
                # trim mask
                trimmed_msg = torch.gather(msg, 1, sorted_indices[:, b:-b])

                n_selected = n_neigh - b * 2
                if self.trim_compensate:
                    # accum = torch.sum(sorted_neigh[:, b:-b], 1) * node.data['norm'] / n_selected * n_neigh
                    alpha = b * 2
                    accum = (torch.sum(trimmed_msg, 1) + node.mailbox['self'][:,0] * alpha) * node.data['norm']
                    # accum = (torch.sum(sorted_neigh[:, b:-b, ...], 1) + node.mailbox['self'][:,0] * alpha) * node.data['norm'] / (n_selected + alpha) * n_neigh
                else:
                    accum = torch.sum(trimmed_msg, 1) * node.data['norm'] / n_selected * n_neigh
        else:
            accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
        return {'h': accum}

    def forward(self, h):
        
        dropped_h = self.dropout(h) if self.dropout else h
        self.g.ndata['input_ft'] = h
        self.g.ndata['h']  = dropped_h
        
        self.g.update_all(self.message_func, self.reduce_func, self.node_update)
        h = self.g.ndata.pop('h')
        return h

class GCN_V2(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 n_layers=1,
                 dropout=0.5, 
                 aggr="mean",
                 n_neigh_threshold=3,
                 trim_ratio=0.45,
                 trim_compensate=True):
        super(GCN_V2, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer_V2(g, in_feats, n_hidden, activation, dropout, \
                                    aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer_V2(g, n_hidden, n_hidden, activation, dropout, \
                                        aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate))
        # output layer
        self.layers.append(GCNLayer_V2(g, n_hidden, n_classes, None, dropout, \
                                    aggr=aggr, n_neigh_threshold=n_neigh_threshold, trim_ratio=trim_ratio, trim_compensate=trim_compensate))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

