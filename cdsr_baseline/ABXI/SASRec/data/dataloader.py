import json
import pickle
from argparse import Namespace
from os.path import join
from typing import Dict, List

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def trim_seq(seq: NDArray[np.int32], len_trim: int) -> NDArray[np.int32]:
    """pad sequences to required length"""
    return np.concatenate((np.zeros(max(0, len_trim - len(seq)), dtype=np.int32), seq))[-len_trim:]


def get_spe_seq(
    n_doms: int,
    dom_item_ids: List[NDArray[np.int32]],
    seq: NDArray[np.int32],
    gt: NDArray[np.int32],
) -> Dict[int, NDArray[np.int32]]:
    item_to_dom_map = {}
    for dom_idx, dom_items in enumerate(dom_item_ids):
        for item_id in dom_items:
            item_to_dom_map[item_id] = dom_idx

    seq_doms = np.array([item_to_dom_map.get(x, -1) for x in seq])
    gt_doms = np.array([item_to_dom_map.get(x, -1) for x in gt])

    dom_seqs_raw = {i: seq[seq_doms == i] for i in range(n_doms)}
    result_seqs = {}

    for i in range(n_doms):
        mask = gt_doms == i
        result_seq = np.zeros_like(seq, dtype=np.int32)
        result_seq[mask] = 1
        result_seqs[i] = result_seq

    first_item_dom = seq_doms[0]
    for i in range(n_doms):
        if i != first_item_dom:
            if np.any(result_seqs[i] != 0):
                first_nonzero_idx = np.nonzero(result_seqs[i])[0][0]
                result_seqs[i][first_nonzero_idx] = 0

    last_gt_dom = gt_doms[-1]
    for i in range(n_doms):
        if i != last_gt_dom:
            dom_seqs_raw[i] = dom_seqs_raw[i][:-1]

    for i in range(n_doms):
        mask_indices = np.nonzero(result_seqs[i])[0]
        if len(mask_indices) > 0 and len(dom_seqs_raw[i]) > 0:
            result_seqs[i][mask_indices] = dom_seqs_raw[i]

    return result_seqs


def process_train(
    seq_raw: list[np.int32],
    n_doms: int,
    dom_item_ids: List[NDArray[np.int32]],
    len_trim: int,
) -> tuple[NDArray[np.int32], ...]:
    """process training sequences"""
    seq_x, gt = (
        np.asarray(seq_raw[:-1], dtype=np.int32),
        np.asarray(seq_raw[1:], dtype=np.int32),
    )
    domain_seqs = get_spe_seq(n_doms, dom_item_ids, seq_x, gt)
    seq_x = trim_seq(seq_x, len_trim)
    trimmed_domain_seqs = {d: trim_seq(s, len_trim) for d, s in domain_seqs.items()}
    gt = trim_seq(gt, len_trim)
    return (seq_x, trimmed_domain_seqs, gt, seq_raw)


def process_evaluate(
    seq_raw: list[np.int32],
    len_trim: int,
) -> tuple[NDArray[np.int32], ...]:
    """process evaluation sequences"""
    seq, gt = (
        np.asarray(seq_raw[:-1], dtype=np.int32),
        np.asarray(seq_raw[-1:], dtype=np.int32),
    )
    seq = trim_seq(seq, len_trim)
    return seq, gt, seq_raw


def get_dataset(args: Namespace, rng: np.random.Generator) -> tuple[Dataset, ...]:
    """get datasets"""
    if args.raw:
        print("Reading raw data...")
        with open(join(args.path_data, f"map_item_{args.len_max}.txt"), "r") as f:
            map_i = json.load(f)
            list_dm = np.array(list(map_i.values()))[:, 1].astype(int)
            dom_item_ids = [np.arange(1, len(list_dm) + 1)[list_dm == d] for d in range(args.n_doms)]
            dom_item_nums = [len(item_ids) for item_ids in dom_item_ids]
            args.dom_item_ids = dom_item_ids
            args.dom_item_nums = dom_item_nums

        print("Serializing trn data...")
        trn_seq, data_trn = [], []
        with open(args.f_raw_trn, "r", encoding="utf-8") as f:
            for line in f:
                seq = []
                line = line.strip().split(" ")
                for ui in line[1:][-args.len_max :]:
                    seq.append(int(ui.split("|")[0]))
                trn_seq.append(np.asarray(seq))
        for seq in tqdm(trn_seq, desc="processing", leave=False):
            data_trn.append(process_train(seq, args.n_doms, args.dom_item_ids, args.len_trim))

        print("Serializing val data...")
        val_seq, data_val = [], []
        with open(args.f_raw_val, "r", encoding="utf-8") as f:
            for line in f:
                seq = []
                line = line.strip().split(" ")
                for ui in line[1:][-args.len_max :]:
                    seq.append(int(ui.split("|")[0]))
                val_seq.append(np.asarray(seq))
        for seq in tqdm(val_seq, desc="processing", leave=False):
            data_val.append(process_evaluate(seq, args.len_trim))

        print("Serializing tst data...")
        tst_seq, data_tst = [], []
        with open(args.f_raw_tst, "r", encoding="utf-8") as f:
            for line in f:
                seq = []
                line = line.strip().split(" ")
                for ui in line[1:][-args.len_max :]:
                    seq.append(int(ui.split("|")[0]))
                tst_seq.append(np.asarray(seq))
        for seq in tqdm(tst_seq, desc="processing", leave=False):
            data_tst.append(process_evaluate(seq, args.len_trim))

        print("Saving serialized seqs...")
        with open(args.f_data, "wb") as f:
            pickle.dump((data_trn, data_val, data_tst, args.dom_item_ids, args.dom_item_nums), f)
    else:
        print("Loading serialized seqs...")
        with open(args.f_data, "rb") as f:
            (data_trn, data_val, data_tst, args.dom_item_ids, args.dom_item_nums) = pickle.load(f)

    args.n_item = sum(args.dom_item_nums)
    args.n_user = len(data_trn)
    return (
        TrainDataset(args, data_trn, rng),
        EvalDataset(args, data_val, rng),
        EvalDataset(args, data_tst, rng),
    )


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class TrainDataset(Dataset):
    """training dataset"""

    def __init__(
        self,
        args: Namespace,
        data: list[tuple[NDArray[np.int32]]],
        rng: np.random.Generator,
    ) -> None:
        self.len_trim = args.len_trim
        self.n_neg = args.n_neg
        self.n_doms = args.n_doms
        self.n_neg_per_dom = args.n_neg // args.n_doms
        self.dom_item_ids = args.dom_item_ids
        self.data = data
        self.length = len(self.data)
        self.rng = rng

    def get_neg(
        self,
        gt: NDArray[np.int32],
        seq_raw: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        rng = self.rng
        gt_neg = np.zeros((self.len_trim, self.n_neg), dtype=np.int32)
        for i, x in enumerate(gt):
            if x != 0:
                dom_id = (np.digitize([x], [item_ids.min() for item_ids in self.dom_item_ids]) - 1)[0]
                candidates = self.dom_item_ids[dom_id][~np.isin(self.dom_item_ids[dom_id], seq_raw, assume_unique=True)]

                # Check if there are enough candidates for replacement.
                if len(candidates) >= self.n_neg:
                    gt_neg[i] = rng.choice(candidates, size=self.n_neg, replace=False)
                else:
                    # Fallback to choosing from a wider pool or with replacement
                    gt_neg[i] = np.pad(
                        rng.choice(candidates, size=len(candidates), replace=False),
                        (0, self.n_neg - len(candidates)),
                        "constant",
                        constant_values=0,
                    )
        return gt_neg

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.LongTensor, ...]:
        seq_x, domain_seqs, gt, seq_raw = self.data[index]
        gt_neg = self.get_neg(gt, seq_raw)

        domain_seq_tensors = tuple(torch.LongTensor(s) for s in domain_seqs.values())

        return (torch.LongTensor(seq_x),) + domain_seq_tensors + (torch.LongTensor(gt), torch.LongTensor(gt_neg))


class EvalDataset(Dataset):
    """evaluation dataset"""

    def __init__(
        self,
        args: Namespace,
        data: list[tuple[NDArray[np.int32]]],
        rng: np.random.Generator,
    ) -> None:
        self.len_trim = args.len_trim
        self.n_mtc = args.n_mtc
        self.n_doms = args.n_doms
        self.dom_item_ids = args.dom_item_ids
        self.n_rand = args.n_mtc + args.len_trim
        self.data = data
        self.length = len(self.data)
        self.rng = rng

    def get_mtc(
        self,
        gt: NDArray[np.int32],
        seq_raw: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        dom_id = (np.digitize(gt, [item_ids.min() for item_ids in self.dom_item_ids]) - 1)[0]
        gt_mtc = self.rng.choice(
            self.dom_item_ids[dom_id][~np.isin(self.dom_item_ids[dom_id], seq_raw, assume_unique=True)],
            size=self.n_mtc,
            replace=False,
        )
        return gt_mtc

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.LongTensor, ...]:
        seq, gt, seq_raw = self.data[index]
        gt_mtc = self.get_mtc(gt, seq_raw)
        return tuple(map(lambda x: torch.LongTensor(x), (seq, gt, gt_mtc)))


def get_dataloader(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return loaders for training, evaluation and testing.
    """
    rng = np.random.default_rng()
    train_set, valid_set, test_set = get_dataset(args, rng)
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.n_worker,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=args.bse,
        shuffle=False,
        num_workers=args.n_worker,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bse,
        shuffle=False,
        num_workers=args.n_worker,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
