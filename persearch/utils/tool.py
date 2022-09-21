import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

import time
from functools import wraps


def save_dict(my_dict, prefix, filename):
    path = Path(prefix, filename)
    with path.open(mode='w', encoding='utf-8') as f:
        json.dump(my_dict, f, indent=4, ensure_ascii=False)


def print_title(title='', token='-', length=32):
    print('{0} {1:^20} {0}'.format(token*length, title))


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print(' # {}.{} : {:4f} s'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


def unique_list(seq):
    """ remove duplicated items in lst while preserve the order
    https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-\
    from-a-list-whilst-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def flatten(list_of_list):
    """flatten list of list"""
    return [item for sublist in list_of_list for item in sublist]


def split_array(arr, n_split=3, p=None):
    """
    split 1-d array into n splits with wight p: [float], return a: [np.array],
    each split will has at least one data
    """
    p = np.ones(n_split)/n_split if p is None else p
    n = len(arr)

    assert min(p) > 0
    assert sum(p) == 1
    assert n >= n_split
    assert n_split == len(p)

    size_split = np.array([np.ceil(p_i * n) for p_i in p])
    while sum(size_split) > n:
        size_split[np.argmax(size_split)] -= 1
    size_split = size_split.astype(int)
    point_split = [sum(size_split[:i]) for i in range(1, len(size_split))]
    arr_shuffle = np.random.permutation(arr)
    arr_split = np.split(arr_shuffle, point_split)
    return arr_split


def batch_generator(ind, batch_size=128):
    """ind: list alike index"""
    start = 0
    while start < len(ind):
        end = start + batch_size if start + batch_size < len(ind) else len(ind)
        yield ind[start: end]
        start += batch_size


def batch_generator_n(n, batch_size=128):
    """generate batch range with maximum n"""
    start = 0
    while start < n:
        end = start + batch_size if start + batch_size < n else n
        yield range(start, end)
        start += batch_size


def dist_cos(inp1, inp2, bias=1e-08, is_fully=False):
    """
    :param inp1: tensor[n1, dim]
    :param inp2: tensor[n2, dim]
    :param bias: small value to avoid division by zero
    :param is_fully: binary, if False, n1=n2 (should), perform similarity by row
    :return: cos: tensor[n1, n2] OR tensor[n]
    """
    if is_fully:
        dot = torch.matmul(inp1, inp2.t())
        norm1 = inp1.norm(dim=-1).view(-1, 1)
        norm2 = inp2.norm(dim=-1).view(1, -1)
        norm1x2 = norm1.matmul(norm2) + bias
    else:
        inp1 = inp1.expand_as(inp2)
        dot = torch.mul(inp1, inp2).sum(dim=-1)
        norm1 = inp1.norm(dim=-1)
        norm2 = inp2.norm(dim=-1)
        norm1x2 = norm1.mul(norm2) + bias
    return dot / norm1x2


def dist_dot(inp1, inp2, is_fully=False):
    """
    :param inp1: tensor[n1, dim]
    :param inp2: tensor[n2, dim]
    :param is_fully: binary, if False, n1=n2 (should), perform similarity by row
    :return: dot: tensor[n1, n2] OR tensor[n]
    """
    if is_fully:
        return torch.matmul(inp1, inp2.t())
    else:
        return torch.mul(inp1, inp2).sum(dim=-1)


def dist_euclidean(x, y):
    # x: B x N x D
    # y: B x M x D
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    b = x.size(0)
    assert d == y.size(2)
    x = x.unsqueeze(2).expand(b,n, m, d)
    y = y.unsqueeze(1).expand(b,n, m, d)
    return torch.pow(x - y, 2).sum(3)


def entropy(labels, base=None):
    """labels: a list of values like [1, 2, 1, 3, 5]"""
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def smart_sort(x, permutation):
    """sort x by given order permutation (row-by-row)
    where x and permutation are 2-d tensor with same shape
    https://discuss.pytorch.org/t/how-to-sort-tensor-by-given-order/61625/2
    """
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


def neg_sampling(pad_sequence, n_vocab, n_neg=2, pool=None, weight=None,
                 idx_pad=0, subtract=True, replace=True,
                 batch_size=256, device=None):
    """
    for a given padded ids [n, pad_len], return negative samples [n, n_neg] from
    pool with probability weight
    :param pad_sequence: tensor[n, pad_len]
    :param n_vocab: int, size of vocabulary corresponding to id in pad_sequence
            and pool (assume n_vocab contains pad token, and the pad token will
            not present in the pool)
    :param n_neg: int, number of n_neg for each of units
    :param pool: T[n_pool]
    :param weight: T[n_pool]
    :param idx_pad: int
    :param subtract: bool, whether subtract pad_seq from sampling pool
    :param replace: bool, if True, samples are drawn with replacement.
    :param batch_size: int
    :param device: torch.device, just for intermediate accelerate
        output will follow the device of pad_sequence
    :return: negs T[n, n_neg]
    """
    device_available = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    device = device_available if device is None else device

    pool_ = torch.tensor(range(n_vocab)) if pool is None else pool.clone()
    weight_ = torch.ones_like(pool_, dtype=torch.float, device=pad_sequence.device) if weight is None else weight.clone()
    assert pool_.shape == weight_.shape

    distribution = torch.zeros(n_vocab, device=pad_sequence.device)
    distribution.scatter_(0, pool_.to(distribution.device), weight_.to(distribution.device))
    distribution[idx_pad] = 0
    distribution = distribution.to(device)
    # expand weight and create mask
    data_iter = DataLoader(pad_sequence, batch_size=batch_size)
    collect = []
    for i, data_batch in enumerate(data_iter):
        n = data_batch.shape[0]
        data_batch = data_batch.to(device)
        distribution_expand = distribution.expand((n, -1))
        if subtract:
            mask = torch.ones_like(distribution_expand)
            mask.scatter_(1, data_batch, 0)
            distribution_batch = distribution_expand.mul(mask)
        else:
            distribution_batch = distribution_expand
        # todo: when non-zero element less than n_neg, replace should be True
        assert distribution_batch.sum(1).min() > 0
        sample_batch = torch.multinomial(distribution_batch, n_neg, replace)
        collect.append(sample_batch)
    ind_negs = torch.cat(collect)
    return ind_negs.to(pad_sequence.device)


def bipartite_to_full(ind, value, n, m):
    """
    :param ind: [2, n_edge]
    :param value: [n_edge]
    :param n: int
    :param m: int
    :return: graph dense tensor
    """
    ui_matrix = torch.sparse.FloatTensor(ind, value, torch.Size([n, m]))
    ui_matrix_t = ui_matrix.transpose(0, 1).to_dense()
    uu_graph = torch.sparse.mm(ui_matrix, ui_matrix_t)
    return uu_graph
