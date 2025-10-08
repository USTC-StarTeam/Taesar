from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_norm_mask(mask: torch.Tensor) -> torch.Tensor:
    """calculate normalized mask"""
    return mask * mask.sum(1).reciprocal().unsqueeze(-1).nan_to_num(posinf=0.0)


def get_absolute_pos_idx(mask: torch.Tensor) -> torch.Tensor:
    """
    Generate position index, default ignoring padding and masking index 0.
    Input mask is non-padded mask
    """
    mask = mask.long().squeeze(-1)
    return mask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) * mask


def init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        elif isinstance(m, nn.Parameter):
            pass  # since only lora use nn.Parameter and it has its own initialization

        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02, a=-0.04, b=0.04)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            nn.init.zeros_(m.bias.data)


class LoRA(nn.Module):
    def __init__(self, d_embed: int, rank: int = 16) -> None:
        super().__init__()
        self.mat_A = nn.Parameter(torch.randn(d_embed, rank) / 50)
        self.mat_B = nn.Parameter(torch.zeros(rank, d_embed))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = F.linear(h, self.mat_A @ self.mat_B)
        return h


class FeedForward(nn.Module):
    """
    SwiGLU-FFN
    Require manual Post-norm residual connection.
    """

    def __init__(self, d_embed: int, d_ffn: int = 128) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_ffn = d_ffn

        self.fc_1 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_3 = nn.Linear(self.d_ffn, d_embed, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h


class MultiHeadAttention(nn.Module):
    """
    Post-norm residual connection integrated.
    """

    def __init__(self, d_embed: int, n_head: int, len_trim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.n_head = n_head
        self.len_trim = len_trim

        self.mha = nn.MultiheadAttention(self.d_embed, self.n_head, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_mha = nn.LayerNorm(self.d_embed)

        self.register_buffer(
            "mask_causal",
            torch.triu(torch.full((self.len_trim, self.len_trim), True), diagonal=1),
            persistent=False,
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h_mha = (
            self.norm_mha(
                h
                + self.dropout(
                    self.mha(
                        h,
                        h,
                        h,
                        attn_mask=self.mask_causal,
                        is_causal=True,
                        need_weights=False,
                    )[0]
                )
            )
            * mask
        )
        return h_mha


class ABXIBlock(nn.Module):
    """
    Single SASRec block containing MHA and FFN with residual connections
    """

    def __init__(
        self,
        d_embed: int,
        n_head: int,
        len_trim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed

        self.mha = MultiHeadAttention(d_embed, n_head, len_trim, dropout)
        self.ffn = FeedForward(d_embed)
        self.norm_ffn = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.mha(h, mask)
        h = self.norm_ffn(h + self.dropout(self.ffn(h))) * mask
        return h


class ABXI(nn.Module):
    def __init__(
        self,
        args: Namespace,
    ) -> None:
        super().__init__()
        self.args = args

        self.n_attn: int = args.n_attn
        self.n_head: int = args.n_head
        self.d_embed: int = args.d_embed

        self.bs: int = args.bs
        self.len_trim: int = args.len_trim
        self.n_neg: int = args.n_neg
        self.temp: float = args.temp
        self.rd: int = args.rd
        self.ri: int = args.ri

        self.n_item: int = args.n_item
        self.n_doms: int = args.n_doms
        self.dom_item_nums: List[int] = args.dom_item_nums
        self.domain_item_counts = {
            0: 70672,  # domain 1
            1: 44278,  # domain 2
            2: 38030,  # domain 3
            3: 31880,  # domain 4
        }
        self.domain_ranges = {}
        start_id = 1
        for dom in [0, 1, 2, 3]:
            end_id = start_id + self.domain_item_counts[dom] - 1
            self.domain_ranges[dom] = (start_id, end_id)
            start_id = end_id + 1

        self.LayerNorm = nn.LayerNorm(self.d_embed, eps=1e-12)
        self.dropout = nn.Dropout(p=args.dropout) if args.dropout > 0.0 else nn.Identity()

        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)

        # encoder, dlora
        # self.mha = MultiHeadAttention(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.blocks = nn.ModuleList([ABXIBlock(args.d_embed, args.n_head, args.len_trim, args.dropout) for _ in range(self.n_attn)])

        self.ffn = FeedForward(self.d_embed)

        # Dynamically create dlora and corresponding norms for each domain
        self.dlora_x = LoRA(self.d_embed, self.rd)
        self.dlora_doms = nn.ModuleList([LoRA(self.d_embed, self.rd) for _ in range(self.n_doms)])
        self.norm_sa_x = nn.LayerNorm(self.d_embed)
        self.norm_sa_doms = nn.ModuleList([nn.LayerNorm(self.d_embed) for _ in range(self.n_doms)])

        # ilora
        self.ilora_doms = nn.ModuleList([LoRA(self.d_embed, self.ri) for _ in range(self.n_doms)])

        # proj
        self.proj_i = FeedForward(self.d_embed)
        self.proj_doms = nn.ModuleList([FeedForward(self.d_embed) for _ in range(self.n_doms)])

        self.norm_i2doms = nn.ModuleList([nn.LayerNorm(self.d_embed) for _ in range(self.n_doms)])
        self.norm_doms2doms = nn.ModuleList([nn.LayerNorm(self.d_embed) for _ in range(self.n_doms)])

        self.apply(init_weights)
        self.loss_fn = nn.CrossEntropyLoss()

    def embed_pos(self, mask: torch.Tensor) -> torch.Tensor:
        return self.ep(get_absolute_pos_idx(mask))

    def forward(
        self,
        seq_x: torch.Tensor,
        domain_seqs: List[torch.Tensor],
        mask_x: torch.Tensor,
        domain_masks: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        # embedding
        input_embed_x = self.ei(seq_x) + self.embed_pos(mask_x)
        input_embed_x = self.LayerNorm(input_embed_x)
        input_embed_x = self.dropout(input_embed_x)
        h_x = input_embed_x * mask_x

        for block in self.blocks:
            h_x = block(h_x, mask_x)

        h_doms = []
        for seq_dom, mask_dom in zip(domain_seqs, domain_masks):
            input_embed_dom = self.ei(seq_dom) + self.embed_pos(mask_dom)
            input_embed_dom = self.LayerNorm(input_embed_dom)
            input_embed_x = self.dropout(input_embed_dom)
            h_dom = input_embed_dom * mask_dom
            for block in self.blocks:
                h_dom = block(h_dom, mask_dom)
            h_doms.append(h_dom)

        # switch training / evaluating
        if self.training:
            gt_masks = [m.unsqueeze(-1) for m in gt_masks]
            masks = [mask_x] + domain_masks
        else:
            h_x = h_x[:, -1]
            h_doms = [h[:, -1] for h in h_doms]
            masks = [1] * (self.n_doms + 1)

        # ffn + dlora
        h_x = self.norm_sa_x(h_x + self.dropout(self.ffn(h_x)) + self.dropout(self.dlora_x(h_x))) * masks[0]
        h_doms = [(self.norm_sa_doms[d](h_doms[d] + self.dropout(self.ffn(h_doms[d])) + self.dropout(self.dlora_doms[d](h_doms[d]))) * masks[d + 1]) for d in range(self.n_doms)]

        # projector + ilora
        h_i = self.proj_i(h_x)
        h_final = torch.zeros_like(h_x)

        for d in range(self.n_doms):
            h_d = self.norm_i2doms[d]((h_x + self.dropout(h_i) + self.dropout(self.ilora_doms[d](h_x))) * gt_masks[d]) + self.norm_doms2doms[d](
                (h_doms[d] + self.dropout(self.proj_doms[d](h_doms[d]))) * gt_masks[d]
            )
            h_final += h_d * gt_masks[d]

        return h_final

    def cal_rec_loss(
        self,
        h: torch.Tensor,
        gt: torch.Tensor,
        gt_neg: torch.Tensor,
        gt_masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """InfoNCE"""
        e_gt = self.ei(gt)
        e_neg = self.ei(gt_neg)

        logits = torch.cat(
            ((h * e_gt).unsqueeze(-2).sum(-1), (h.unsqueeze(-2) * e_neg).sum(-1)),
            dim=-1,
        ).div(self.temp)

        loss = -F.log_softmax(logits, dim=2)[:, :, 0]

        losses_by_domain = []
        for mask in gt_masks:
            losses_by_domain.append((loss * cal_norm_mask(mask.squeeze(-1))).sum(-1).mean())

        return losses_by_domain

    # def cal_rec_loss(
    #     self,
    #     h: torch.Tensor,
    #     gt: torch.Tensor,
    #     gt_neg: torch.Tensor,
    #     gt_masks: List[torch.Tensor],
    # ) -> List[torch.Tensor]:
    #     losses_by_domain = []
    #     for dom, dom_mask in enumerate(gt_masks):
    #         real_score = torch.matmul(
    #             h,
    #             self.ei.weight.transpose(0, 1),
    #         )
    #         begin_item_id, end_item_id = self.domain_ranges[dom]

    #         domain_scores = real_score[:, :, begin_item_id : end_item_id + 1]
    #         dom_gt = gt - begin_item_id  # [128, 125]
    #         domain_item_count = end_item_id - begin_item_id + 1

    #         valid_gt_mask = (dom_gt >= 0) & (dom_gt < domain_item_count)
    #         combined_mask = dom_mask & valid_gt_mask

    #         flattened_scores = domain_scores.reshape(-1, domain_item_count)  # [128*125, domain_items]
    #         flattened_gt = dom_gt.reshape(-1)  # [128*125]
    #         flattened_mask = combined_mask.reshape(-1)  # [128*125]

    #         valid_indices = flattened_mask.nonzero(as_tuple=True)[0]

    #         if len(valid_indices) > 0:
    #             valid_scores = flattened_scores[valid_indices]  # [num_valid, domain_items]
    #             valid_gt = flattened_gt[valid_indices]  # [num_valid]

    #             loss = F.cross_entropy(valid_scores, valid_gt)
    #             losses_by_domain.append(loss)
    #         else:
    #             losses_by_domain.append(torch.tensor(0.0, device=h.device))

    #     return losses_by_domain

    #     """CrossEntropy Loss"""
    # e_gt = self.ei(gt)  # [B, L, D]
    # e_neg = self.ei(gt_neg)  # [B, L, K, D]

    # pos_scores = (h * e_gt).sum(-1).unsqueeze(-1)  # [B, L, 1]
    # neg_scores = (h.unsqueeze(-2) * e_neg).sum(-1)  # [B, L, K]

    # logits = torch.cat([pos_scores, neg_scores], dim=-1).div(self.temp)
    # labels = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)
    # loss = F.cross_entropy(
    #     logits.view(-1, logits.size(-1)),
    #     labels.view(-1),  # [B*L]
    #     reduction="none",
    # ).view(logits.size(0), logits.size(1))

    # losses_by_domain = []
    # for mask in gt_masks:
    #     mask = cal_norm_mask(mask.squeeze(-1))
    #     losses_by_domain.append((loss * mask).sum(-1).mean())

    # return losses_by_domain
