# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 16:01
# @Author  : cpark

from collections import defaultdict

import torch
import torch.nn as nn
import tqdm
from torch.cuda.amp import GradScaler
from torch.optim import Adam

from utils import get_metric, mrr, ndcg, ndcg_k, recall, recall_at_k, hr


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
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

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
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class PretrainProfileTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(PretrainProfileTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)

    def pretrain(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"

        rec_data_iter = tqdm.tqdm(enumerate(dataloader), desc="Recommendation EP_%s:%d" % (str_code, epoch), total=len(dataloader), bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            seq_loss_avg = 0.0
            loss_avg = 0.0
            acc_avg = 0.0

            print("Device", self.device)
            for i, batch in rec_data_iter:
                # gc.collect()
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                # with autocast(): #enabled=False
                item_input, item_pos, item_neg, test_neg, item_answer, cat1_input, cat1_pos, cat1_neg, cat2_input, cat2_pos, cat2_neg, type_input, type_pos = batch
                # loss를 Pretrain_seq 자체에서 생성
                loss, seq_loss, shaply_values_softmax, shaply_values_datanum = self.model.pretrain_seq(
                    item_input, item_pos, item_neg, test_neg, item_answer, cat1_input, cat1_pos, cat1_neg, cat2_input, cat2_pos, cat2_neg, type_input, self.args.hierarhical
                )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                seq_loss_avg += seq_loss.item()
                loss_avg += loss.item()

            post_fix = {
                "epoch": epoch,
                "loss_avg": "{:.4f}".format(loss_avg / len(rec_data_iter)),
                "seq_loss_avg": "{:.4f}".format(seq_loss_avg / len(rec_data_iter)),
                "shaply_value": shaply_values_softmax,
                "shaply_domain_datanum": shaply_values_datanum,
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            self.model.eval()

            total_result = []
            outputs_list = []
            dom_outputs_list = [[] for dom in [5, 6, 7, 8]]
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    #???
                    #batch = tuple(t.to("cuda:0") for t in batch)
                    batch = tuple(t.to(self.device) for t in batch)

                    item_input, item_pos, item_neg, test_neg, item_answer, cat1_input, cat1_pos, cat1_neg, cat2_input, cat2_pos, cat2_neg, type_input, type_pos = batch

                    #???
                    # sequence_output_1, sequence_output_2, recommend_output = self.model.to("cuda:0").get_last_emb(
                    #     item_input, cat1_input, cat2_input, type_input, item_pos, item_neg, self.args.hierarhical, cuda_yn="y"
                    # )  #'cpu'
                    sequence_output_1, sequence_output_2, recommend_output = self.model.to(self.device).get_last_emb(
                        item_input, cat1_input, cat2_input, type_input, item_pos, item_neg, self.args.hierarhical, cuda_yn="y"
                    )

                    test_logits = self.predict_full(recommend_output)  # [B, item_num]

                    # ???
                    # 假设 ID 0, 1, 2, 3, 4 是特殊 token，不应被推荐
                    # 根据你的 WordVocab 定义，如果 0 是 PAD，1 是 UNK 等，那么这些 ID 的分数应该被遮蔽
                    # 假定这些特殊 ID 是 [0, 1, 2, 3, 4]
                    special_token_ids = torch.arange(5, device=test_logits.device) # 创建一个 tensor [0, 1, 2, 3, 4]
                    test_logits[:, special_token_ids] = -torch.inf # 将这些特殊 token 的分数设为负无穷



                    _, topk_items = torch.topk(test_logits, 100, dim=1)  # [B, 100]
                    pred = item_answer.view(-1, 1) == topk_items  # [B, 100]
                    target = torch.ones_like(item_answer, dtype=torch.float)  # [B, 1]
                    # result = {
                    #     f"{name}@{cutoff}": func(pred, target, cutoff, mean=False) for name, func in [("recall", recall), ("ndcg", ndcg), ("mrr", mrr)] for cutoff in [5, 10, 20]
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

                        #???
                        # if dom_item_answer.shape[0] > 0:
                        #     # 打印当前 domain 的 id 范围
                        #     print(f"Domain {dom} range: {self.domain_ranges[dom]}")
                        #     # 打印部分 dom_item_answer，检查是否包含预期值
                        #     print(f"Domain {dom} item answers (sample): {dom_item_answer[:5]}")
                        #     # 打印 topk_items，检查是否包含预期值以及是否都是该领域内的物品
                        #     print(f"Domain {dom} topk items (sample): {topk_items[:5]}")
                        #     # 打印 pred 的 True 计数，看有多少命中
                        #     print(f"Domain {dom} hits: {pred.sum().item()}")

                        # # --- START: 修正后的调试代码块 ---

                        # # 1. 从 self.domain_ranges 中获取 begin_item_id 和 end_item_id
                        # begin_item_id, end_item_id = self.domain_ranges[dom]

                        # # 2. 现在可以安全地使用这些变量进行调试打印
                        # print(f"DEBUG: dom={dom}, range=[{begin_item_id}, {end_item_id}]")
                        # print(f"DEBUG: dom_test_logits shape: {dom_test_logits.shape}")

                        # # 3. 手动创建和应用 mask 来验证逻辑
                        # temp_mask = torch.ones_like(dom_test_logits, dtype=torch.bool)
                        # temp_mask[:, begin_item_id : end_item_id + 1] = False
                        # temp_masked_score = dom_test_logits.masked_fill(temp_mask, -torch.inf)

                        # # 4. 检查一个领域外的ID和一个领域内的ID的分数，确认mask是否生效
                        # #    我们选择一个肯定在领域外的ID和一个在领域内的ID作为例子
                        # #    (注意：要确保这些ID在 dom_test_logits 的列数范围内)
                        # invalid_id_example = (begin_item_id - 1) if begin_item_id > 0 else (end_item_id + 1)
                        # valid_id_example = begin_item_id + 10 # 选一个领域内的ID

                        # if invalid_id_example < dom_test_logits.shape[1]:
                        #    print(f"DEBUG: Original score at invalid_id={invalid_id_example}: {dom_test_logits[0, invalid_id_example].item()}")
                        #    print(f"DEBUG: Masked score at invalid_id={invalid_id_example}: {temp_masked_score[0, invalid_id_example].item()}")

                        # if valid_id_example < dom_test_logits.shape[1]:
                        #    print(f"DEBUG: Original score at valid_id={valid_id_example}: {dom_test_logits[0, valid_id_example].item()}")
                        #    print(f"DEBUG: Masked score at valid_id={valid_id_example}: {temp_masked_score[0, valid_id_example].item()}")

                        # # 5. 检查应用mask后，得分最高的物品ID是否在领域内
                        # print(f"DEBUG: Masked score max index (should be within range): {temp_masked_score.argmax(dim=1)[:5]}")

                        # # --- END: 修正后的调试代码块 ---

                        _, topk_items = self.topk(dom_test_logits, 100, dom)
                        pred = dom_item_answer.view(-1, 1) == topk_items  # [B, 100]
                        target = torch.ones_like(dom_item_answer, dtype=torch.float)  # [B, 1]
                        # result = {
                        #     f"{name}@{cutoff}": func(pred, target, cutoff, mean=False)
                        #     for name, func in [("recall", recall), ("ndcg", ndcg), ("mrr", mrr)]
                        #     for cutoff in [5, 10, 20]
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

                return total_result

    # def topk(self, real_score, k, dom):
    #     begin_item_id, end_item_id = self.domain_ranges[dom]
    #     domain_mask = torch.ones_like(real_score, dtype=torch.bool)
    #     domain_mask[:, begin_item_id : end_item_id + 1] = False
    #     masked_score = real_score.masked_fill(domain_mask, -torch.inf)

    #     score, topk_items = torch.topk(masked_score, k)
    
    #     return score, topk_items

    #???
    def topk(self, real_score, k, dom):
        

        # print("--- EXECUTING THE NEW, CORRECTED topk FUNCTION! ---")
        # 1. 从 self.domain_ranges 中获取当前领域的物品ID范围
        begin_item_id, end_item_id = self.domain_ranges[dom]

        # 2. 创建一个掩码，初始时将所有位置标记为需要被屏蔽 (True)
        domain_mask = torch.ones_like(real_score, dtype=torch.bool)

        # 3. 将当前领域内的物品ID对应的位置设置为不需要被屏蔽 (False)
        #    这意味着只有这部分的分数会被保留
        domain_mask[:, begin_item_id : end_item_id + 1] = False

        # 4. 使用掩码将所有非当前领域的物品分数设置为负无穷
        masked_score = real_score.masked_fill(domain_mask, -torch.inf)

        # 5. 在被屏蔽过的分数上执行 topk 操作，这样就能确保只推荐领域内的物品
        score, topk_items = torch.topk(masked_score, k, dim=1)

        # print(f"DEBUG_FINAL: For dom={dom} (range=[{begin_item_id}, {end_item_id}]), topk_items to be returned (sample):")
        # print(topk_items[0, :10]) # 打印第一个样本的前10个推荐物品
        
        return score, topk_items
        # ----------------- END: 替换结束 -----------------
