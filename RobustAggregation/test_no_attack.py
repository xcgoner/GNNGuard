import logging
import torch
# import sys
# sys.path.insert(0, '/n/scratch2/xz204/Dr37/lib/python3.7/site-packages')
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from deeprobust.graph.defense import * # GCN, GAT, GIN, JK, GCN_attack,accuracy_1
from tqdm import tqdm
import scipy
import numpy as np
from sklearn.preprocessing import normalize
import pickle

from dgl_models.train_dgl import train_dgl_model


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
# cora and citeseer are binary, pubmed has not binary features
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GCN_V2', 'GAT', 'GIN', 'JK'])
parser.add_argument('--GNNGuard', action='store_true')
parser.add_argument('--aggr', type=str, default='mean',  choices=['mean', 'median'])
parser.add_argument('--n-neigh-threshold', type=int, default=3,  help='threshold for self-isolation')
parser.add_argument('--trim-ratio', type=float, default=0.45,  help='trimmed mean')
parser.add_argument('--trim-compensate', action='store_true',  help='wether to use self embedding to compensate trimmed mean')
parser.add_argument("--log", type=str, help="dir of the log file", default='test_no_attack_results')
parser.add_argument("--save", type=str, help="name of temp model", default='nettack_di_results')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# logging
filehandler = logging.FileHandler(args.log + '.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(args)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels

if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)

"""set the number of training/val/testing nodes"""
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following 
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1
    

def test(adj, features):
    'ALL the baselines'


    # train dgl
    _, test_acc = train_dgl_model(args.modelname, 16, labels.max().item() + 1, 0.5 , \
                    features, adj, labels, idx_train, idx_val, idx_test, n_epochs=200, \
                    aggr=args.aggr, n_neigh_threshold=args.n_neigh_threshold, trim_ratio=args.trim_ratio, trim_compensate=args.trim_compensate,
                    cuda=torch.cuda.is_available(), 
                    save=args.save)

    return test_acc


if __name__ == '__main__':
    # main()
    test_acc = test(adj, features)

    logger.info('Testing accuracy: %f' % (test_acc))

    # pass
