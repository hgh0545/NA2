from torch.nn.modules.module import Module
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import utils


class BaseAttack(Module):
    def __init__(self, model, nnodes):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnode = nnodes

    def attack(self):
        pass

    def check_adj(self, adj):
        '''
            check if the modified adjacency is symmetric and unweighted
        '''
        if type(adj) is torch.Tensor:
            adj = adj.cpu().numpy()
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        if sp.issparse(adj):
            assert adj.tocsr().max() == 1, "Max value should be 1!"
            assert adj.tocsr().min() == 0, "Min value should be 0!"
        else:
            assert adj.max() == 1, "Max value should be 1!"
            assert adj.min() == 0, "Min value should be 0!"




