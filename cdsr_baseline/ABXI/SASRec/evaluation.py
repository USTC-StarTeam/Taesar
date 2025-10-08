from typing import Dict, List

import torch


def cal_norm_mask(mask: torch.Tensor) -> torch.Tensor:
    """calculate normalized mask"""
    return mask * mask.sum(1).reciprocal().unsqueeze(-1).nan_to_num(posinf=0.0)


def cal_metrics(ranks: List[float], topk: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculates various metrics (HR, NDCG, MRR) for a list of ranks across
    multiple top-k values.

    Args:
        ranks (List[float]): A list of ranks for each item.
        topk (List[int]): A list of integer values for which to calculate top-k metrics.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
                          Keys are formatted as "metric@k", e.g., "HR@5".
    """
    if not ranks:
        # Return a dictionary with 0.0 for all requested metrics if ranks is empty.
        results = {}
        for k in topk:
            results[f"HR@{k}"] = 0.0
            results[f"NDCG@{k}"] = 0.0
            results[f"MRR@{k}"] = 0.0
        return results

    ranks_tensor = torch.tensor(ranks, dtype=torch.float)
    num_users = len(ranks)
    results = {}

    for k in topk:
        # Calculate HR@k
        # HR@k is the proportion of users for whom the rank of the ground truth item is <= k.
        hr_val = (ranks_tensor <= k).sum().item() / num_users
        results[f"HR@{k}"] = hr_val

        # Calculate NDCG@k and MRR@k
        ndcg_k_val = 0.0
        mrr_k_val = 0.0

        # We use a boolean mask to efficiently select only the ranks that are <= k.
        hit_at_k_mask = ranks_tensor <= k

        # Get the ranks of the hit items
        hit_ranks = ranks_tensor[hit_at_k_mask]

        if hit_ranks.numel() > 0:
            # For NDCG, we sum 1 / log2(rank + 1) for all hit items.
            # The .item() is used to extract scalar values from the tensor for the log2 function.
            ndcg_k_val = (1.0 / torch.log2(hit_ranks + 1.0)).sum().item() / num_users
            results[f"NDCG@{k}"] = ndcg_k_val

            # For MRR, we sum 1 / rank for all hit items.
            mrr_k_val = (1.0 / hit_ranks).sum().item() / num_users
            results[f"MRR@{k}"] = mrr_k_val
        else:
            # If no items are hit within top-k, NDCG@k and MRR@k are 0.
            results[f"NDCG@{k}"] = 0.0
            results[f"MRR@{k}"] = 0.0

    return results
