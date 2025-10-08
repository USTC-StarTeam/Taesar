import time
from argparse import Namespace
from collections import defaultdict
from typing import List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from tqdm import tqdm

from data.dataloader import get_dataloader
from model.SASRec import SASRec
from noter import Noter
from utils import mrr, ndcg, recall


class Trainer(object):
    def __init__(self, args: Namespace, noter: Noter) -> None:
        print("[info] Loading data")
        self.n_warmup = args.n_warmup
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.n_user = args.n_user
        # Store domain-specific information in a more flexible structure
        self.dom_item_nums = args.dom_item_nums
        self.n_doms = args.n_doms
        print("Done.\n")

        self.domain_item_counts = {
            0: 55579,  # A domain
            1: 45133,  # B domain
            2: 31649,  # C domain
            3: 44435,  # D domain
        }

        self.domain_ranges = {}
        start_id = 1
        for dom in [0, 1, 2, 3]:
            end_id = start_id + self.domain_item_counts[dom] - 1
            self.domain_ranges[dom] = (start_id, end_id)
            start_id = end_id + 1

        self.noter = noter
        self.device = args.device

        # model
        self.model = SASRec(args).to(args.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1.0, total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=args.lr_g, patience=args.lr_p)

        noter.log_num_param(self.model)

    def run_epoch(self, i_epoch: int) -> List[List[float]]:
        self.model.train()
        losses = [0.0] * self.n_doms
        t_0 = time.time()

        # training
        for batch in tqdm(self.train_loader, desc="training", leave=False, ncols=70):
            self.optimizer.zero_grad()

            # The train_batch now returns a list of losses
            batch_losses = self.train_batch(batch)

            n_seq = batch[0].size(0)
            for d in range(self.n_doms):
                losses[d] += (batch_losses[d] * n_seq) / self.n_user

        self.noter.log_train(i_epoch, losses, time.time() - t_0)

        # warm-up quit
        if i_epoch <= self.n_warmup:
            return [None] * self.n_doms

        # validating
        self.model.eval()
        ranks_by_domain = [[] for _ in range(self.n_doms)]

        outputs_list = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="validating", leave=False, ncols=70):
                outputs_batch = self.evaluate_batch(batch)
                outputs_list.append(outputs_batch)

            metric_list, bs = zip(*outputs_list)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = (metric * bs).sum() / bs.sum()

        return out

    def run_test(self) -> List[List[float]]:
        self.model.eval()
        ranks_by_domain = [[] for _ in range(self.n_doms)]

        outputs_list = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="validating", leave=False, ncols=70):
                outputs_batch = self.evaluate_batch(batch)
                outputs_list.append(outputs_batch)

            metric_list, bs = zip(*outputs_list)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = (metric * bs).sum() / bs.sum()

        return out
        # with torch.no_grad():
        #     for batch in tqdm(self.test_loader, desc="testing", leave=False):
        #         ranks_batch = self.evaluate_batch(batch)

        #         for d in range(self.n_doms):
        #             ranks_by_domain[d] += ranks_batch[d]

        # return [cal_metrics(ranks) for ranks in ranks_by_domain]

    def train_batch(self, batch: List[torch.Tensor]) -> List[float]:
        # Unpack batch: seq_x, domain_seqs, gt, gt_neg
        seq_x, *domain_seqs, gt, gt_neg = [x.to(self.device) for x in batch]

        # Generalize masks for all domains
        mask_x = (seq_x != 0).unsqueeze(-1).half()
        domain_masks = [(s != 0).unsqueeze(-1).half() for s in domain_seqs]

        # Generalize GT masks
        gt_masks = []
        item_end_ids = [sum(self.dom_item_nums[: d + 1]) for d in range(self.n_doms)]
        item_start_ids = [1] + [i + 1 for i in item_end_ids[:-1]]
        for start, end in zip(item_start_ids, item_end_ids):
            gt_masks.append((gt >= start) & (gt <= end))

        # Pass a list of domain sequences and masks to the model
        h = self.model(seq_x, mask_x)

        # Calculate loss for each domain
        losses = self.model.cal_rec_loss(h, gt, gt_neg)
        total_loss = losses
        total_loss.backward()

        self.optimizer.step()
        return [losses.item()]

    def evaluate_batch(self, batch: List[torch.Tensor]) -> List[List[float]]:
        seq_x, gt, gt_mtc = [x.to(self.device) for x in batch]

        # Generalize GT masks
        gt_masks = []
        item_end_ids = [sum(self.dom_item_nums[: d + 1]) for d in range(self.n_doms)]
        item_start_ids = [1] + [i + 1 for i in item_end_ids[:-1]]
        for start, end in zip(item_start_ids, item_end_ids):
            gt_masks.append((gt >= start) & (gt <= end))

        # Generalize specific-domain sequences and masks
        domain_seqs = [torch.zeros_like(seq_x, dtype=torch.long) for _ in range(self.n_doms)]
        domain_masks = [torch.zeros_like(seq_x, dtype=torch.bool) for _ in range(self.n_doms)]

        for d in range(self.n_doms):
            start, end = item_start_ids[d], item_end_ids[d]
            mask_dom = (seq_x > 0) & (seq_x >= start) & (seq_x <= end)
            domain_seqs[d][mask_dom] = seq_x[mask_dom]
            domain_masks[d] = mask_dom

        # Convert masks to the correct format for the model
        mask_x = (seq_x != 0).unsqueeze(-1).half()
        domain_masks_tensor = [m.unsqueeze(-1).half() for m in domain_masks]
        h = self.model(seq_x, mask_x)

        test_item_emb = self.model.ei.weight
        test_logits = torch.matmul(h[:, -1], test_item_emb.transpose(0, 1))

        # _, topk_items = torch.topk(test_logits, 100, dim=1)  # [B, 100]
        _, topk_items = self.topk(test_logits, 100, dom=0)
        pred = gt.view(-1, 1) == topk_items  # [B, 100]
        target = torch.ones_like(gt, dtype=torch.float)  # [B, 1]
        result = {f"{name}@{cutoff}": func(pred, target, cutoff, mean=False) for name, func in [("recall", recall), ("ndcg", ndcg), ("mrr", mrr)] for cutoff in [5, 10, 20]}

        result = {k: v.mean() for k, v in result.items()}

        return (result, test_logits.shape[0])

    def topk(self, real_score, k, dom):
        begin_item_id, end_item_id = self.domain_ranges[dom]
        domain_mask = torch.ones_like(real_score, dtype=torch.bool)
        domain_mask[:, begin_item_id : end_item_id + 1] = False
        masked_score = real_score.masked_fill(domain_mask, -torch.inf)

        score, topk_items = torch.topk(masked_score, k)
        return score, topk_items
