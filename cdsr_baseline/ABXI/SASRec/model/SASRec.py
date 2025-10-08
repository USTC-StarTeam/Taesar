from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SASRecBlock(nn.Module):
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


class SASRec(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args

        self.n_attn: int = args.n_attn  # Number of attention layers
        self.n_head: int = args.n_head
        self.d_embed: int = args.d_embed
        self.dropout_prob: float = args.dropout
        self.layer_norm_eps: float = args.layer_norm_eps

        self.n_item = args.n_item
        self.len_trim = args.len_trim

        self.bs: int = args.bs
        self.len_trim: int = args.len_trim
        self.n_doms: int = args.n_doms
        self.dom_item_nums: List[int] = args.dom_item_nums
        self.n_neg: int = args.n_neg
        self.temp: float = args.temp

        self.LayerNorm = nn.LayerNorm(self.d_embed, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(p=args.dropout) if args.dropout > 0.0 else nn.Identity()

        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)

        # Create multiple SASRec blocks based on n_attn
        self.blocks = nn.ModuleList([SASRecBlock(args.d_embed, args.n_head, args.len_trim, args.dropout) for _ in range(self.n_attn)])

        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(init_weights)

    def embed_pos(self, mask: torch.Tensor) -> torch.Tensor:
        return self.ep(get_absolute_pos_idx(mask))

    def forward(self, seq_x: torch.Tensor, mask_x: torch.Tensor) -> torch.Tensor:
        # embedding
        input_embed = self.ei(seq_x) + self.embed_pos(mask_x)
        input_embed = self.LayerNorm(input_embed)
        input_embed = self.dropout(input_embed)

        # Apply multiple SASRec blocks
        h_x = input_embed * mask_x
        for block in self.blocks:
            h_x = block(h_x, mask_x)

        return h_x

    def cal_rec_loss(self, h: torch.Tensor, gt: torch.Tensor, gt_neg: torch.Tensor) -> List[torch.Tensor]:
        # e_gt = self.ei(gt)  # [B, L, D]
        # e_neg = self.ei(gt_neg)  # [B, L, K, D]

        # pos_scores = (h * e_gt).sum(-1).unsqueeze(-1)  # [B, L, 1]
        # neg_scores = (h.unsqueeze(-2) * e_neg).sum(-1)  # [B, L, K]

        # logits = torch.cat([pos_scores, neg_scores], dim=-1).div(self.temp)
        # labels = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)
        # loss = F.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     labels.view(-1),  # [B*L]
        #     reduction="mean",
        # ).view(logits.size(0), logits.size(1))

        logits = torch.matmul(
            h,
            self.ei.weight.transpose(0, 1),
        )
        mask = gt != 0
        loss = self.loss_fn(logits[mask], gt[mask])

        return loss
