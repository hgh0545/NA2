import numpy as np
import torch
import scipy.sparse as sp

def topk(emb, k):
    return emb*(torch.argsort(torch.argsort(emb))>=emb.shape[1]-k)

def feature_adj(adj, idx_train, sim, k):
    sim = sim - np.eye(adj.shape[0])
    for id in idx_train:
        for kk in range(k):
            neighbor = sim[id].argmax()
            if adj[id][neighbor] == 0:
                adj[id][neighbor] = 1
                adj[neighbor][id] = 1
            else:
                adj[id][neighbor] = 0
                adj[neighbor][id] = 0
            sim[id][neighbor] = -1
            sim[neighbor][id] = -1
    return adj