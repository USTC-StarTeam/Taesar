#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/14
# @Author  : Chung Park and Taesan Kim and Hyungjun Yoon Junui Hong
# @Desc    : util functions

import os
import random
import sys

import numpy as np
import torch
import torchmetrics.functional as M


def log_metrics_table(stage: str, res: list[dict]) -> None:
    """Helper function to log a table of metrics."""
    # Assume all dictionaries have the same keys and order
    if not res:
        print(f"    | {stage} |  * | No results to log |  * |")
        return

    # Dynamically build the header based on the keys of the first dictionary
    header_keys = list(res[0].keys())
    header_metrics = " | ".join(f"{key:>7}" for key in header_keys)

    # Build the metric rows for each domain
    rows = []
    for i, domain_res in enumerate(res):
        # metric_values = " | ".join(f"{value:.4f}" for value in domain_res.values())
        # rows.append(f"Dom {i + 1:<3}| {metric_values}")

        metric_strs = [f"{v:7.5f}" for v in domain_res.values()]
        metric_values_formatted = " | ".join(metric_strs)
        rows.append(f" Dom {i:<3}| {metric_values_formatted}")

    # Format the final message
    msg_header = f"    |   {stage}   | {header_metrics} |"
    msg_content = "\n".join([f"    | {row} |" for row in rows])

    print(msg_header)
    print(msg_content)


def recall(pred, target, k_or_thres, mean=True):
    r"""Calculating recall.

    Recall value is defined as below:

    .. math::
        Recall= \frac{TP}{TP+FN}

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        count = (target > 0).sum(-1)
        output = pred[:, :k].sum(dim=-1).float() / count
        if mean:
            return output.mean()
        else:
            return output
    else:
        thres = k_or_thres
        return M.recall(pred, target, task="binary", threshold=thres)


def _dcg(pred, k):
    k = min(k, pred.size(1))
    denom = torch.log2(torch.arange(k).type_as(pred) + 2.0).view(1, -1)
    return (pred[:, :k] / denom).sum(dim=-1)


def ndcg(pred, target, k, mean=True):
    r"""Calculate the Normalized Discounted Cumulative Gain(NDCG).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    pred_dcg = _dcg(pred.float(), k)
    # TODO replace target>0 with target
    ideal_dcg = _dcg(torch.sort((target > 0).float(), descending=True)[0], k)
    all_irrelevant = torch.all(target <= sys.float_info.epsilon, dim=-1)
    pred_dcg[all_irrelevant] = 0
    pred_dcg[~all_irrelevant] /= ideal_dcg[~all_irrelevant]
    if mean:
        return pred_dcg.mean()
    else:
        return pred_dcg


def mrr(pred, target, k, mean=True):
    r"""Calculate the Mean Reciprocal Rank(MRR).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    row, col = torch.nonzero(pred[:, :k], as_tuple=True)
    row_uniq, counts = torch.unique_consecutive(row, return_counts=True)
    idx = torch.zeros_like(counts)
    idx[1:] = counts.cumsum(dim=-1)[:-1]
    first = col.new_zeros(pred.size(0)).scatter_(0, row_uniq, col[idx] + 1)
    output = 1.0 / first
    output[first == 0] = 0
    return output


#???
def hr(pred,target,k,mean=True):
    r"""Calculate Hit Ratio (HR) at k.

    HR@k measures whether the ground truth item is present in the top-k predicted items.

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.
            For HR, this typically means the predicted top-k items that match a target.

        target(torch.FloatTensor): [B, num_target]. The ground truth.
            This should indicate the true relevant items. For HR, we usually check if any of these
            target items are in the top-k predictions.

        k(int): The number of top items to consider.

        mean(bool): If True, return the mean HR over the batch. Otherwise, return HR for each sample.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor if mean is True, or a 1-dimensional tensor [B] if mean is False.
    """
    hits = (pred[:, :k].sum(dim=-1) > 0).float() 
                                                 
    if mean:
        return hits.mean()
    else:
        return hits

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(2, item_size - 1)
    while item in item_set:
        item = random.randint(2, item_size - 1)
    return item


def neg_hard_sample(item_set, ture_item):
    item = random.choice(list(item_set))
    while item == ture_item:
        item = random.choice(list(item_set))
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print("Validation score increased.  Saving model ...")
        torch.save(model.module.state_dict(), self.checkpoint_path)
        self.score_min = score


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)
