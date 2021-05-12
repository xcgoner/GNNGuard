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
from dgl.nn import GATConv
from dgl.base import DGLError
from dgl import function as fn
from dgl.ops import edge_softmax
import torch.nn.functional as F


class GATLayer(GATConv):
    def __init__(self, in_feats, out_feats, num_heads,
                 feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=None, 
                 aggr="mean", n_neigh_threshold=3, trim_ratio=0.45, trim_compensate=True):
        super(GATLayer, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope,
                                        residual, activation)
        self.aggr = aggr
        self.n_neigh_threshold= n_neigh_threshold
        self.trim_ratio= trim_ratio
        self.trim_compensate = trim_compensate

    def message_func(self, edge):
        msg = edge.src['ft']
        if self.aggr == "median":
            return {'m': msg, 'e_a': edge.data['e_a'], 'self': edge.dst['ft'], 'self_a': edge.dst['self_a']}
        else:
            return {'m': msg, 'e_a': edge.data['e_a']}

    def reduce_func(self, node):
        if self.aggr == "median":
            n_neigh = node.mailbox['m'].shape[1]
            if n_neigh <= self.n_neigh_threshold: 
                accum = node.mailbox['self'][:,1,:,:]
            else:
                b = n_neigh // 2 - (1 - n_neigh % 2)
                b = max(min(b, int(n_neigh * self.trim_ratio)), 1)
                # b = max(b, 1)

                # print(n_neigh, b)

                sorted_indices = torch.argsort(node.mailbox['m'], dim=1)
                msg = node.mailbox['m']
                alpha = node.mailbox['e_a']
                # print(msg.shape, alpha.shape)
                trimmed_msg = torch.gather(msg, 1, sorted_indices[:, b:-b, :])
                trimmed_alpha = torch.gather(alpha, 1, sorted_indices[:, b:-b, :, :1])

                if self.trim_compensate:
                    self_msg = node.mailbox['self'][:,:1,:]
                    self_alpha = node.mailbox['self_a'][:,:1,:]
                    trimmed_msg = torch.cat([trimmed_msg] + (2 * b) * [self_msg], 1)
                    trimmed_alpha = torch.cat([trimmed_alpha] + (2 * b) * [self_alpha], 1)
                alpha = self.attn_drop(F.softmax(trimmed_alpha, dim=1))
                accum = torch.sum(trimmed_msg * trimmed_alpha, 1)
        else:
            alpha = self.attn_drop(F.softmax(node.mailbox['e_a'], dim=1))
            accum = torch.sum(node.mailbox['m'] * alpha, 1)
        
        return {'ft': accum}

    def forward(self, graph, feat):
        r"""
        Description
        -----------
        Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # here we suppose feat is not tuple
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'ft': feat_dst, 'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['e_a'] = e
            # compute softmax
            # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            # graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
            #                  fn.sum('m', 'ft'))

            # self-attention
            self_a = self.leaky_relu((feat_dst * (self.attn_l+self.attn_r)).sum(dim=-1).unsqueeze(-1))
            graph.dstdata.update({'self_a': self_a})

            graph.update_all(self.message_func, self.reduce_func)
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst

class GAT(nn.Module):
    def __init__(self,
                g,
                in_feats,
                n_hidden,
                n_classes,
                dropout=0.5, 
                aggr="mean",
                n_neigh_threshold=3,
                trim_ratio=0.45,
                trim_compensate=True):
        super(GAT, self).__init__()

        attn_drop = 0.6
        negative_slope = 0.2
        residual = False
        activation = F.relu
        heads = [8, 1]
        n_layers = 1

        self.g = g
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(GATLayer(
            in_feats, n_hidden, heads[0],
            dropout, attn_drop, negative_slope, False, activation,
            aggr, n_neigh_threshold, trim_ratio, trim_compensate))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_feats = n_hidden * num_heads
            self.gat_layers.append(GATLayer(
                n_hidden * heads[l-1], n_hidden, heads[l],
                dropout, attn_drop, negative_slope, residual, activation,
            aggr, n_neigh_threshold, trim_ratio, trim_compensate))
        # output projection
        self.gat_layers.append(GATLayer(
            n_hidden * heads[-2], n_classes, heads[-1],
            dropout, attn_drop, negative_slope, residual, None,
            aggr, n_neigh_threshold, trim_ratio, trim_compensate))

    def forward(self, inputs):
        h = inputs
        for l in range(self.n_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

