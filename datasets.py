import networkx as nx
import numpy as np
import os
import pickle
import torch
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import random
import pickle as pkl
import sys
def process_features(features):
    row_sum_diag = np.sum(features, axis=1)
    row_sum_diag_inv = np.power(row_sum_diag, -1)
    row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
    row_sum_inv = np.diag(row_sum_diag_inv)
    return np.dot(row_sum_inv, features)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# def sample(idx,l):
#     mask = np.zeros(l)
#     mask[idx] = idx
#     return np.array(mask)
# def sample1(idx,l):
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask)


def save_sparse_csr(filename,array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
    )

def load_data1(dataset, model='GCN'):
    ## get data
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        data_path = 'data'
        suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
        objects = []
        for suffix in suffixs:
            file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
            objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
        x, y, allx, ally, tx, ty, graph = objects
        x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

        # test indices
        test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
        with open(test_index_file, 'r') as f:
            lines = f.readlines()
        indices = [int(line.strip()) for line in lines]
        min_index, max_index = min(indices), max(indices)

        # preprocess test indices and combine all data
        tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
        features = np.vstack([allx, tx_extend])
        features[indices] = tx
        ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
        labels = np.vstack([ally, ty_extend])
        labels[indices] = ty
        labels1 = []
        for i in range(len(labels)):
            labels1.append(labels[i].argmax())
        labels1 = np.array(labels1)
        # get adjacency matrix
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
        # adj = torch.from_numpy(adj)
        adj = np.array(adj)
        #修改1
        # adj = adj.type(torch.float)
        # print('adj',adj.dtype)

        # idx_t = range(len(y))
        # idx_v = range(len(y), len(y) + 500)
        idx_train = np.arange(0, len(y), 1)
        idx_val = np.arange(len(y), len(y) + 500, 1)
        idx_test = np.array(indices)

        # idx_train = sample(idx_t, len(y))
        # idx_val = sample1(idx_v, 500)
        # train_mask = sample_mask(idx_train, labels.shape[0])
        # val_mask = sample_mask(idx_val, labels.shape[0])
        # test_mask = sample_mask(idx_test, labels.shape[0])
        # zeros = np.zeros(labels.shape)
        # y_train = zeros.copy()
        # y_val = zeros.copy()
        # y_test = zeros.copy()
        # y_train[train_mask, :] = labels[train_mask, :]
        # y_val[val_mask, :] = labels[val_mask, :]
        # y_test[test_mask, :] = labels[test_mask, :]
        # features = torch.from_numpy(process_features(features))
        #修改2
        # features = features.type(torch.float32)

        # y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        #     torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
        #     torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)

        # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels
    # elif dataset == 'nell.0.1':
    elif 'nell' in dataset:
        # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        # objects = []
        # for i in range(len(names)):
        #     with open("data/{}.{}".format(dataset, names[i]), 'rb') as f:
        #         if sys.version_info > (3, 0):
        #             objects.append(pkl.load(f, encoding='latin1'))
        #         else:
        #             objects.append(pkl.load(f))
        #
        # x, y, tx, ty, allx, ally, graph = tuple(objects)
        #
        # test_idx_reorder = parse_index_file("data/{}.test.index".format(dataset))
        # features = allx.tolil()
        # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        # labels1 = np.argmax(ally, axis=-1)
        # features = preprocess_features(features, False)
        # # support = preprocess_adj(adj)
        # idx_test = test_idx_reorder
        # idx_train = range(len(y))
        # idx_val = range(len(y), len(y)+969)
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/nell/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/nell/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/nell/{}.features.npz".format(dataset)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack(
                (features, sp.lil_matrix(
                    (features.shape[0], len(isolated_node_idx))
                )), dtype=np.float32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(
                len(isolated_node_idx), dtype=np.float32)
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/nell/{}.features".format(dataset),
                            features)
        else:
            features = load_sparse_csr(
                "data/nell/{}.features.npz".format(dataset))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels1 = np.vstack((ally, ty))
        labels1[test_idx_reorder, :] = labels1[test_idx_range, :]
        labels1 = np.argmax(labels1, axis=-1)
        print("num of class: ", int(labels1.max()) + 1)
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

    elif dataset == 'polblogs':
        adj = np.zeros((1222, 1222))
        with open('data/'+str(dataset) + '.txt')as f:
            for j in f:
                entry = [float(x) for x in j.split(" ")]
                adj[int(entry[0]), int(entry[1])] = 1
                adj[int(entry[1]), int(entry[0])] = 1
        labels1 = np.loadtxt('data/'+str(dataset) + '_label.txt')
        labels1 = labels1.astype(int)
        labels1 = labels1[:,1:].flatten()
        idx_train = np.loadtxt('data/'+str(dataset) + '_train_node.txt')
        idx_train = idx_train.astype(int)
        idx_val = np.loadtxt('data/'+str(dataset) + '_validation_node.txt')
        idx_val = idx_val.astype(int)
        idx_test = np.loadtxt('data/'+str(dataset) + '_test_node.txt')
        idx_test = idx_test.astype(int)

        features = np.eye(adj.shape[0])

    elif dataset == 'cora_ml':
        filename = 'data/' + str(dataset) + '_adj' + '.npz'
        adj = sp.load_npz(filename)
        filename = 'data/' + str(dataset) + '_features' + '.npz'
        features = sp.load_npz(filename)
        filename = 'data/' + str(dataset) + '_label' + '.npy'
        labels1 = np.load(filename)
        filename = 'data/' + str(dataset) + '_train_node' + '.npy'
        idx_train = np.load(filename)
        filename = 'data/' + str(dataset) + '_val_node' + '.npy'
        idx_val = np.load(filename)
        filename = 'data/' + str(dataset) + '_test_node' + '.npy'
        idx_test = np.load(filename)

    else:
        # filename = 'data/' +'amazon_electronics_photo' + '.npz'
        # # data = np.load(filename)
        # # adj, features, labels1 = load_npz(filename)
        # adj, features, labels1= get_adj(filename)
        # print(adj.sum())
        # idx_train, idx_val, idx_test = get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels1, seed=15)
        # filename = 'data/' + 'amazon_electronics_photo' + '_adj' + '.npz'
        # sp.save_npz(filename, adj)
        # filename = 'data/' + 'amazon_electronics_photo' + '_features' + '.npz'
        # sp.save_npz(filename, features)
        # filename = 'data/' + 'amazon_electronics_photo' + '_train_node' + '.npy'
        # np.save(filename, idx_train)
        # filename = 'data/' + 'amazon_electronics_photo' + '_val_node' + '.npy'
        # np.save(filename, idx_val)
        # filename = 'data/' + 'amazon_electronics_photo' + '_test_node' + '.npy'
        # np.save(filename, idx_test)
        # filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        # np.save(filename, labels1)

        filename = 'data/' + 'amazon_electronics_photo' + '_adj' + '.npz'
        adj = sp.load_npz(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_features' + '.npz'
        features = sp.load_npz(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        labels1 = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo'+ '_train_node' + '.npy'
        idx_train = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_val_node' + '.npy'
        idx_val = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_test_node' + '.npy'
        idx_test = np.load(filename)
        # filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        # labels1 = np.load(filename)
    if model == 'sage':
        return sp.csr_matrix(adj), sp.csr_matrix(features), idx_train, idx_val, idx_test, labels1, graph
    else:
        return sp.csr_matrix(adj), sp.csr_matrix(features), idx_train, idx_val, idx_test, labels1


def get_adj( filename, require_lcc=True):
    adj, features, labels = load_npz(filename)
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    if require_lcc:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels

def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                             loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels

def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    '''
        This setting follows nettack/mettack, where we split the nodes
        into 10% training, 10% validation and 80% testing data
    '''
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def largest_connected_components(adj, n_components=1):
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def get_train_validation_test(nnodes, labels, n=20,seed=15):
    random.seed = seed
    idx_node = np.arange(nnodes)
    idx_dic_same_label = {}
    label_node = labels
    dd = defaultdict(list)
    for k, va in [(v, i) for i, v in enumerate(label_node)]:
        dd[k].append(va)
    for key in dd:
        idx = dd[key]
        a = idx_node[idx]
        idx_dic_same_label[key] = a

    train_val = []
    idx_train = []
    idx_val = []
    idx_node = list(idx_node)
    for k, v in idx_dic_same_label.items():
        tmp = random.sample(list(v), 2*n)
        train_val.append(tmp)
        idx_train.append(tmp[:n])
        idx_val.append(tmp[n:])

    idx_train = sum(idx_train, [])
    idx_val = sum(idx_val, [])
    for i in train_val:
        for m in i:
            idx_node.remove(m)

    idx_test = random.sample(list(idx_node), 1000)
    # for k, v in idx_dic_same_label.items():
    #     idx_train.append(random.sample(list(v), n))

    a = [x for x in idx_train if x in idx_val]
    b = [x for x in idx_train if x in idx_test]
    c = [x for x in idx_val if x in idx_test]

    print(a)
    print(b)
    print(c)
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features, sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if sparse:
        return sparse_to_tuple(features)
    else:
        return features.toarray()
    # return features

def load_new_data():
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    graph_node_features_and_labels_file_path = "/public/hgh/Graph-VFL-attack/data/cornell/out1_node_feature_label.txt"
    graph_adjacency_list_file_path = "/public/hgh/Graph-VFL-attack/data/cornell/out1_graph_edges.txt"
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
            graph_labels_dict[int(line[0])] = int(line[2])
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    return adj, features, labels

if __name__ == '__main__':
    adj, features, labels = load_npz("/public/hgh/Graph-VFL-attack/data/cora_ml.npz")
    adj = sp.csr_matrix(adj)
    # # features = sp.csr_matrix(features)
    # adj = sp.load_npz("/public/hgh/Graph-VFL-attack/data/cora_ml_adj.npz")
    # labels = np.load("/public/hgh/Graph-VFL-attack/data/cora_ml_label.npy")
    # idx_train, idx_val, idx_test = get_train_validation_test(adj.shape[0], labels=labels, n=20, seed=30)
    # # np.savez("/public/hgh/Graph-VFL-attack/data/cora_ml_adj.npz", adj)
    # # np.savez("/public/hgh/Graph-VFL-attack/data/cora_ml_features.npz", features)
    # # np.save("/public/hgh/Graph-VFL-attack/data/cora_ml_label.npy", labels)
    # np.save("/public/hgh/Graph-VFL-attack/data/cora_ml_train_node.npy", idx_train)
    # np.save("/public/hgh/Graph-VFL-attack/data/cora_ml_val_node.npy", idx_val)
    # np.save("/public/hgh/Graph-VFL-attack/data/cora_ml_test_node.npy", idx_test)



    # adj, features, labels = load_new_data()
    # features = sp.csr_matrix(features)
    # idx_train, idx_val, idx_test = get_train_validation_test(adj.shape[0], labels=labels, n=10, seed=30)
    # np.savez("/public/hgh/Graph-VFL-attack/data/cornell/cornell_adj.npz", adj)
    # np.savez("/public/hgh/Graph-VFL-attack/data/cornell/cornell_features.npz", features)
    # np.save("/public/hgh/Graph-VFL-attack/data/cornell/cornell_label.npy", labels)
    # np.save("/public/hgh/Graph-VFL-attack/data/cornell/cornell_train_node.npy", idx_train)
    # np.save("/public/hgh/Graph-VFL-attack/data/cornell/cornell_val_node.npy", idx_val)
    # np.save("/public/hgh/Graph-VFL-attack/data/cornell/cornell_test_node.npy", idx_test)