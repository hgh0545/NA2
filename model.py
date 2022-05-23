import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import utils
from copy import deepcopy
from sklearn.metrics import f1_score
import numpy as np
import scipy.sparse as sp
from copy import copy
from torch.autograd import Function
from torch.autograd import Variable
# from dgl.nn.pytorch.conv import SAGEConv,SGConv,GATConv,GINConv,GraphConv
from utils import *

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight.data = torch.from_numpy(np.random.uniform(-stdv, stdv, size=self.weight.shape)).to(torch.float)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            # self.bias.data = torch.from_numpy(np.random.uniform(-stdv, stdv, size=self.bias.shape)).to(torch.float)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):
        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.inner_features=None
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.activate_value = []
        # self.initialize()
    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, input_data):
        '''
            adj: normalized adjacency matrix
        '''
        x, adj = input_data
        self.activate_value = []
        if torch.is_tensor(adj) == False:
            adj = sparse_mx_to_torch_sparse_tensor((normalize_adj(adj))).to(self.device)
        if torch.is_tensor(x) == False:
            x = sparse_mx_to_torch_sparse_tensor(x).to(self.device)

        if self.with_relu:
            x = self.gc1(x, adj)
            self.activate_value.append(x)
            x = F.relu(x)
            self.inner_features = x
            # x.retain_grad()
        else:
            x = self.gc1(x, adj)
            # x.retain_grad()
            self.activate_value.append(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x.retain_grad()
        self.output = x
        self.activate_value.append(x)
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1)
        return x

    def test(self, x, adj, id, labels, select_node=False):
        self.eval()
        prob = F.log_softmax(self.forward([x, adj]), dim=-1)
        print('test acc: %f'% accuracy(prob[id], labels[id]))
        if select_node == True:
            correct_node = []
            wrong_node = []
            for i in id:
                if prob[i].argmax() == labels[i]:
                    correct_node.append(i.item())
                else:
                    wrong_node.append(i.item())
            correct_node = np.array(correct_node)
            wrong_node = np.array(wrong_node)
            np.save("save_model/nodes/correct_nodes.npy", correct_node)
            np.save("save_model/nodes/wrong_nodes.npy", wrong_node)