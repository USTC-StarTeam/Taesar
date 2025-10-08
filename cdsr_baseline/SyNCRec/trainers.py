#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/14
# @Author  : Chung Park and Taesan Kim and Hyungjun Yoon and Junui Hong
# @Desc    : trainer


from collections import defaultdict

import torch
import torch.nn as nn
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam

from utils import get_metric, mrr, ndcg, recall, hr


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        self.args = args
        self.mlm_output = nn.Linear(args.hidden_size, args.item_size - 1)
        self.local_rank = args.local_rank
        self.device = torch.device("cuda:" + str(self.local_rank))
        self.model = model.to(self.device)
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        if self.args.loss_type == "negative":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.NLLLoss()

        ################
        ### ddp part ###
        ################
        self.local_rank = self.args.local_rank
        with_cuda = True
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:" + str(self.local_rank))
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for RecGPT" % torch.cuda.device_count())
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        ################
        ################

        self.scaler = GradScaler(enabled=False, growth_interval=100)

        self.domain_item_counts = {
            5: 70672,  # domain 1
            6: 44278,  # domain 2
            7: 38030,  # domain 3
            8: 31880,  # domain 4
        }
        self.domain_ranges = {}
        start_id = 5
        for dom in [5, 6, 7, 8]:
            end_id = start_id + self.domain_item_counts[dom] - 1
            self.domain_ranges[dom] = (start_id, end_id)
            start_id = end_id + 1

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return ([HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix))

    def get_full_sort_score(self, epoch, answers, pred_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        answers = torch.from_numpy(answers).long().to(device)  # [B]
        pred_list = torch.from_numpy(pred_list).float().to(device)  # [B, N]

        recall_final, ndcg_final, mrr_final = [], [], []
        for k in [5, 10, 15, 20]:
            recall_final.append(recall(pred_list, answers, k))
            ndcg_final.append(ndcg(pred_list, answers, k))
            mrr_final.append(mrr(pred_list, answers, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall_final[0]),
            "NDCG@5": "{:.4f}".format(ndcg_final[0]),
            "HIT@10": "{:.4f}".format(recall_final[1]),
            "NDCG@10": "{:.4f}".format(ndcg_final[1]),
            "HIT@20": "{:.4f}".format(recall_final[3]),
            "NDCG@20": "{:.4f}".format(ndcg_final[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall_final[0], ndcg_final[0], recall_final[1], ndcg_final[1], recall_final[3], ndcg_final[3]], str(post_fix)

    def _dcg(self, pred, k):
        denom = torch.log2(torch.arange(1, k + 1, device=pred.device) + 1.0).view(1, -1)
        return (pred / denom).sum(dim=-1)

    def _ndcg(self, hit_at_k, k, mean=True):
        pred_dcg = self._dcg(hit_at_k, k)

        ideal_hit = torch.zeros_like(hit_at_k)
        ideal_hit[:, 0] = 1.0
        ideal_dcg = self._dcg(ideal_hit, k)

        non_zero_ideal = ideal_dcg > 1e-10
        ndcg = torch.zeros_like(pred_dcg)
        ndcg[non_zero_ideal] = pred_dcg[non_zero_ideal] / ideal_dcg[non_zero_ideal]

        if mean:
            return ndcg.mean()
        else:
            return ndcg

    def _mrr(self, hit_at_k):
        first_hit_pos = torch.zeros(hit_at_k.size(0), device=hit_at_k.device)

        for i in range(hit_at_k.size(0)):
            hits = torch.nonzero(hit_at_k[i] > 0.5, as_tuple=False)
            if hits.size(0) > 0:
                first_hit_pos[i] = hits[0].item() + 1

        mrr_val = torch.zeros_like(first_hit_pos, dtype=torch.float)
        non_zero = first_hit_pos > 0
        mrr_val[non_zero] = 1.0 / first_hit_pos[non_zero]

        return mrr_val.mean()

    def _recall(self, hit_at_k):
        return (hit_at_k.sum(dim=1) > 0).float().mean()

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_negatve_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget) / (torch.sum(istarget) + 1e-24)

        return loss

    def cross_entropy(self, seq_out, pos_ids):
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        # only mlm_loss
        sequence_output = self.mlm_output(seq_emb)  # [batch*seq_len class_num]

        labels = pos_ids.view(-1, 1)  # [batch*seq_len class_num]

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float().view(-1, 1)  # [batch*seq_len, 1]
        labels = (labels * istarget).long()
        loss = self.criterion(
            nn.LogSoftmax(dim=-1)(sequence_output),
            labels.view(
                -1,
            ),
        )
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        # test_item_emb = self.model.embedding_layer.item_embeddings(test_neg_sample)
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        # test_item_emb = self.model.embedding_layer.item_embeddings.weight
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        test_logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return test_logits


class PretrainTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(PretrainTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)

    def pretrain(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"

        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (str_code, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
            ncols=70,
        )
        if train:
            self.model.train()

            loss_contrastive_single_avg = 0.0
            loss_contrastive_cross_avg = 0.0
            mip_loss_avg = 0.0
            loss_avg = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                with autocast(enabled=False):
                    batch = tuple(t.to(self.device) for t in batch)

                    item_input, item_pos, item_neg, test_neg, item_answer, type_input, type_pos = batch
                    loss, loss_contrastive_single, loss_contrastive_cross, mip_loss = self.model.pretrain_seq(
                        item_input, item_pos, item_neg, test_neg, item_answer, type_input, type_pos
                    )

                self.optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optim.step()
                loss_contrastive_single_avg += loss_contrastive_single.detach().item()
                loss_contrastive_cross_avg += loss_contrastive_cross.detach().item()
                mip_loss_avg += mip_loss.detach().item()
                loss_avg += loss.detach().item()

            post_fix = {
                "epoch": epoch,
                "loss_avg": "{:.4f}".format(loss_avg / len(rec_data_iter)),
                "loss_contrastive_single_avg": "{:.4f}".format(loss_contrastive_single_avg / len(rec_data_iter)),
                "loss_contrastive_cross_avg": "{:.4f}".format(loss_contrastive_cross_avg / len(rec_data_iter)),
                "mip_loss_avg": "{:.4f}".format(mip_loss_avg / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            self.model.eval()
            total_result = []
            outputs_list = []
            dom_outputs_list = [[] for _ in [5, 6, 7, 8]]
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    item_input, item_pos, item_neg, test_neg, item_answer, type_input, type_pos = batch
                    recommend_output = self.model.to(self.device).get_last_emb(item_input, type_input, item_pos, item_neg, type_pos, cuda_yn="y")
                    test_logits = self.predict_full(recommend_output)  # [B, item_num]

                    _, topk_items = torch.topk(test_logits, 100, dim=1)  # [B, 100]
                    pred = item_answer.view(-1, 1) == topk_items  # [B, 100]
                    target = torch.ones_like(item_answer, dtype=torch.float)  # [B, 1]
                    # result = {
                    #     f"{name}@{cutoff}": func(pred, target, cutoff, mean=False)
                    #     for name, func in [("recall", recall), ("ndcg", ndcg), ("mrr", mrr)]
                    #     for cutoff in [5, 10, 20, 50]
                    # }
                    #???
                    result = {
                        f"{name}@{cutoff}": func(pred, target, cutoff, mean=False)
                        for name, func in [("hr", hr), ("ndcg", ndcg), ("mrr", mrr)]
                        for cutoff in [10, 20, 50, 100]
                    }

                    result = {k: v.mean() for k, v in result.items()}
                    outputs_list.append((result, test_logits.shape[0]))

                    for dom in [5, 6, 7, 8]:
                        mask = type_pos[:, -1] == dom
                        dom_test_logits = test_logits[mask]
                        dom_item_answer = item_answer[mask]

                        _, topk_items = self.topk(dom_test_logits, 100, dom)
                        pred = dom_item_answer.view(-1, 1) == topk_items  # [B, 100]
                        target = torch.ones_like(dom_item_answer, dtype=torch.float)  # [B, 1]
                        # result = {
                        #     f"{name}@{cutoff}": func(pred, target, cutoff, mean=False)
                        #     for name, func in [("recall", recall), ("ndcg", ndcg), ("mrr", mrr)]
                        #     for cutoff in [5, 10, 20, 50]
                        # }
                        #???
                        result = {
                            f"{name}@{cutoff}": func(pred, target, cutoff, mean=False)
                            for name, func in [("hr", hr), ("ndcg", ndcg), ("mrr", mrr)]
                            for cutoff in [10, 20, 50, 100]
                        }

                        result = {k: v.mean() for k, v in result.items()}
                        dom_outputs_list[dom - 5].append((result, dom_test_logits.shape[0]))

                metric_list, bs = zip(*outputs_list)
                bs = torch.tensor(bs)
                out = defaultdict(list)
                for o in metric_list:
                    for k, v in o.items():
                        out[k].append(v)
                for k, v in out.items():
                    metric = torch.tensor(v)
                    out[k] = (metric * bs).sum() / bs.sum()

                total_result.append(out)

                for dom in [5, 6, 7, 8]:
                    metric_list, bs = zip(*dom_outputs_list[dom - 5])
                    bs = torch.tensor(bs)
                    out = defaultdict(list)
                    for o in metric_list:
                        for k, v in o.items():
                            out[k].append(v)
                    for k, v in out.items():
                        metric = torch.tensor(v)
                        out[k] = (metric * bs).sum() / bs.sum()

                    total_result.append(out)

                # for result in total_result:
                #     print(result)

                return total_result  # (full result + 4 domain result)

    def topk(self, real_score, k, dom):
        begin_item_id, end_item_id = self.domain_ranges[dom]
        domain_mask = torch.ones_like(real_score, dtype=torch.bool)
        domain_mask[:, begin_item_id : end_item_id + 1] = False
        masked_score = real_score.masked_fill(domain_mask, -torch.inf)

        score, topk_items = torch.topk(masked_score, k)
        return score, topk_items
