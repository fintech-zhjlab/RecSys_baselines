import math
from collections import Counter
import numpy as np
import torch


def RMSE(records):
    return math.sqrt(
        sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records]) \
        / float(len(records)))


def MAE(records):
    return sum([abs(rui - pui) for u, i, rui, pui in records]) \
           / float(len(records))


def precision_recall(users_labels, items, topk):
    hit = 0
    n_recall = 0
    n_precision = 0
    for user, labels in users_labels:
        rank = recommend(user, labels, topk)
        hit += len(rank & items)
        n_recall += len(items)
        n_precision += topk
    return [hit / n_recall, hit / n_precision]


def recommend(user, items, topk):
    _, sort_idx = torch.sort((user * items).sum(axis=1), dim=0)
    results = sort_idx[:topk]
    return results


def recommend_mat(users, items, topk):
    sim_mat = users.mm(items.T)
    _, sort_idx = torch.sort(sim_mat, dim=1)
    results = sort_idx[:, :topk]
    return results


def gin_index(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


# def gin_index(item_pops):
#     j = 1
#     n = len(item_pops)
#     G = 0
#     for weight in item_pops:
#         G += (2 * j - n - 1) * weight
#     return G / float(n - 1)


def compute_rec_ginis_ratio(users, items, labels, topk=1):
    rec_items = torch.reshape(recommend_mat(users, items, topk), [-1]).tolist()
    gini_pred = gin_index(rec_items)
    gini_orig = gin_index(labels.tolist())
    return gini_orig, gini_pred, gini_pred - gini_orig


if __name__ == '__main__':
    dssm_users = torch.tensor(torch.load('save/dssm_users.pt'), dtype=torch.double)
    dssm_items = torch.tensor(torch.load('save/dssm_items.pt'), dtype=torch.double)

    dccl_users = torch.tensor(torch.load('save/dccl_users.pt'), dtype=torch.double)
    dccl_items = torch.tensor(torch.load('save/dccl_items.pt'), dtype=torch.double)
    pos_items = torch.tensor(torch.load('save/pos_items.pt'), dtype=torch.int64)

    gini_orig, gini_pred, gini_diff = compute_rec_ginis_ratio(dssm_users, dssm_items, pos_items, topk=15)
    print("dssm: ", gini_orig, gini_pred, gini_diff)
    gini_orig, gini_pred, gini_diff = compute_rec_ginis_ratio(dccl_users, dccl_items, pos_items, topk=15)
    print("dccl: ", gini_orig, gini_pred, gini_diff)

