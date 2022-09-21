import numpy as np
import torch
from sklearn.metrics import ndcg_score

from persearch.utils.tool import batch_generator


metric_map = dict(
    hr='topk', mrr='topk', idcg='topk',
    tp='ctr', fn='ctr', fp='ctr', tn='ctr',
    recall='ctr', precision='ctr', f_score='ctr',
    roc='ctr', auc='ctr_score',
)


""" top k evaluation with ground truth as only one dimension
ref
https://www.cnblogs.com/shenxiaolin/p/9309749.html
input: 
    :param pred: tensor[n_samples, topk], recommended item id
    :param truth: tensor[n_samples, padded], real item id
    :param k=100: will set to the length of pred if pred_len less than k
"""


def _get_hit_pos(pred, truth, k=50):
    n, pred_len = pred.shape
    pad_len = truth.shape[1]
    k = k if pred_len >= k else pred_len
    pred_k = pred[:, :k]  # (n, k)
    pred_k_expand = pred_k.unsqueeze(-1).expand((n, k, pad_len))
    hit_pos = []
    for i in range(pad_len):
        pred_i = pred_k_expand[:, :, i]
        truth_i = truth[:, i].view(-1, 1)
        diff_i = (pred_i - truth_i)  # (n, k)
        hit_pos_i = (diff_i == 0).nonzero()  # (n_hit, 2)
        hit_pos.append(hit_pos_i)
    return hit_pos


def _get_hit_pos_all(pred, truth, k=50):
    """
    return T[n, pad_len], the position of truth in the pred list
    if it is not in the pred, place (k+1)
    # the rank is set to 1-based to avoid zero division error
    """
    max_pos = (pred[:, 0] == pred[:, 0]).long() * (k + 1)
    hit_pos = _get_hit_pos(pred, truth, k)
    poss_hit = []
    for pos in hit_pos:
        pos_hit = max_pos.scatter(0, pos[:, 0], pos[:, 1] + 1)
        poss_hit.append(pos_hit.view(-1, 1))
    hit_pos_all = torch.cat(poss_hit, dim=1)
    return hit_pos_all


def hr(pred, truth, k=50, **kwargs):
    """assume there will not be zero(pad) in pred"""
    hit_pos = _get_hit_pos(pred, truth, k)
    hit_pos_stack = torch.cat(hit_pos)
    units_hit = hit_pos_stack[:, 0].unique()
    return units_hit.shape[0] / pred.shape[0]


def mrr(pred, truth, k=50, **kwargs):
    """assume there will not be zero(pad) in pred"""
    collect_pos_hit = _get_hit_pos_all(pred, truth, k)
    min_pos = collect_pos_hit.min(dim=1)[0]
    hit_unit = (min_pos != (k + 1)).nonzero()
    reciprocal_rank = torch.div(1.0, min_pos[hit_unit])
    return reciprocal_rank.sum().item() / pred.shape[0]


def ndcg(pred, truth, k=50, **kwargs):
    """

    :param pred: T[n, mix_len(100/1000, etc.)]
    :param truth: T[n, pad_len(real list)]
    :param k: int
    :param kwargs:
    :return: ndcg score
    """
    n, mix_len = pred.shape
    pred_np, truth_np = pred.cpu().numpy(), truth.cpu().numpy()

    predk_np = pred_np[:, :k]
    pred_score = np.array(range(k)).reshape(1, -1).repeat(n, axis=0)
    pred_score = (pred_score + 1) / k
    pred_score = np.flip(pred_score, axis=1)

    truth_ = np.zeros_like(predk_np)
    for i in range(n):
        truth_[i] = np.isin(predk_np[i], truth_np[i]).astype(np.int64)

    score = ndcg_score(truth_, pred_score)
    return score


""" ctr evaluation
ref when encountering division by zero
https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
input:
    pred, tensor[n_samples, ], binary prediction
    truth, tensor[n_samples, ], binary ground truth
    pos_v, int, positive label value
    neg_v, int, negative label value
return:
    metric, float tensor
"""


def tp(pred, truth, pos_v=1, **kwargs):
    pred = pred.flatten()
    truth = truth.flatten()
    return ((pred == truth) & (pred == pos_v)).sum(-1).float()


def fp(pred, truth, pos_v=1, **kwargs):
    pred = pred.flatten()
    truth = truth.flatten()
    return ((pred != truth) & (pred == pos_v)).sum(-1).float()


def tn(pred, truth, neg_v=0, **kwargs):
    pred = pred.flatten()
    truth = truth.flatten()
    return ((pred == truth) & (pred == neg_v)).sum(-1).float()


def fn(pred, truth, neg_v=0, **kwargs):
    pred = pred.flatten()
    truth = truth.flatten()
    return ((pred != truth) & (pred == neg_v)).sum(-1).float()


def recall(pred, truth, pos_v=1, neg_v=0, **kwargs):
    tp_ = tp(pred, truth, pos_v)
    fn_ = fn(pred, truth, neg_v)
    if tp_ + fn_ == 0:
        print('warning: Recall is ill-defined and being set to 0.0 due to no positive truth.')
        return 0.0
    return tp_.true_divide(tp_ + fn_)


def precision(pred, truth, pos_v=1, **kwargs):
    tp_ = tp(pred, truth, pos_v)
    fp_ = fp(pred, truth, pos_v)
    if tp_ + fp_ == 0:
        print('warning: Precision is ill-defined and being set to 0.0 due to no positive prediction.')
        return 0.0
    return tp_ / (tp_ + fp_)


def f_score(pred, truth, pos_v=1, neg_v=0, beta=1, **kwargs):
    """beta should be positive"""
    recall_ = recall(pred, truth, pos_v, neg_v)
    precision_ = precision(pred, truth, pos_v)
    if recall_ + precision_ == 0:
        fp_ = fp(pred, truth, pos_v)
        fn_ = fn(pred, truth, neg_v)
        if fp_ + fn_ == 0:
            print('F1=1 because all the samples are negative and all the prediction are correct.')
            return 1.0
        else:
            return 0.0
    f_score_ = (1 + beta * beta) * (precision_ * recall_) / (beta * beta * precision_ + recall_)
    return f_score_.item()


""" ctr roc and auc
https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
input:
    pred, tensor[n_samples] (float), prediction score in [0, 1]
    truth, tensor[n_samples], actual binary label {0, 1}
    batch_size, int
"""


def roc(pred, truth, batch_size=64, **kwargs):
    """return tpr: FloatTensor[n_pred], fpr: FloatTensor[n_pred]"""
    n = pred.shape[0]
    n_truth_pos = (truth == 1).sum()
    n_truth_neg = (truth == 0).sum()
    device = pred.device

    tpr = torch.zeros(n + 1).to(device).float()
    fpr = torch.zeros(n + 1).to(device).float()

    if min(n_truth_pos, n_truth_neg) > 0:
        pred_sort, ind_sort = pred.sort(descending=True)
        truth_sort = truth[ind_sort].view(1, -1).to(device)
        for ind in batch_generator(range(n + 1), batch_size):
            batch_len = batch_size if batch_size <= len(ind) else len(ind)
            pred_batch = torch.ones([batch_len, n]).to(device).tril(min(ind) - 1)  # [batch_len, n]
            truth_batch = truth_sort.expand(batch_len, -1)  # [batch_len, n]
            tp_ = tp(pred_batch, truth_batch)  # [batch_len]
            fn_ = fn(pred_batch, truth_batch)
            tpr[ind] = tp_ / (tp_ + fn_)
            fp_ = fp(pred_batch, truth_batch)
            tn_ = tn(pred_batch, truth_batch)
            fpr[ind] = fp_ / (tn_ + fp_)
    return tpr, fpr


def auc(pred, truth, batch_size=64, **kwargs):
    pred = pred.flatten()
    truth = truth.flatten()
    tpr, fpr = roc(pred, truth, batch_size=batch_size)
    fpr_0 = torch.cat((fpr[0].view(-1), fpr), 0)
    fpr_t = torch.cat((fpr, fpr[-1].view(-1)), 0)
    diff_fpr = (fpr_t - fpr_0)[:-1]
    auc_ = torch.dot(tpr, diff_fpr)
    return auc_.item()
