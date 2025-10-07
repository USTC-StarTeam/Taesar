import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from torch import nn

from utils import get_domain_ranges


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        self.config = config
        self.stage = config["stage"]
        self.flag = "full"
        self.domain_ranges = get_domain_ranges(config["item_path"], config["domains"])

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        self.loss_type = config["loss_type"]
        self.num_neg_samples = config["num_neg_samples"]
        self.softmax_temperature = config["softmax_temperature"]

        # define layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        # define loss type
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Only support `CE` loss!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        # Prepare next-item prediction task
        # Shift sequence: input is seq[:-1], target is seq[1:]
        padded_seq = torch.nn.functional.pad(item_seq, (0, 1), value=0).scatter_(
            dim=1,
            index=item_seq_len.unsqueeze(1),
            src=pos_items.unsqueeze(1),
        )
        target_seq = padded_seq[:, 1:]

        if self.stage == "run" and self.flag != "full":
            # if False:
            st_idx = self.domain_ranges[self.flag][0]
            ed_idx = self.domain_ranges[self.flag][1]

            domain_mask = (padded_seq >= st_idx) & (padded_seq <= ed_idx) & (padded_seq != 0)

            cumsum_mask = domain_mask.long().cumsum(dim=1)
            group_indices = torch.where(domain_mask, cumsum_mask, 0)

            # input_group_indices = group_indices[:, :-1]
            # target_group_indices = group_indices[:, 1:]

            # row_indices = torch.arange(input_group_indices.size(0), device=input_group_indices.device)
            # _, max_positions = input_group_indices.max(dim=1)
            # max_mask = torch.zeros_like(input_group_indices, dtype=torch.bool)
            # max_mask[row_indices, max_positions] = True
            # input_group_indices = input_group_indices.masked_fill(max_mask, 0)

            # one_mask = target_group_indices == 1
            # target_group_indices = target_group_indices.masked_fill(one_mask, 0)

            # input_mask = input_group_indices != 0
            # target_mask = target_group_indices != 0

            # logits = torch.matmul(
            #     seq_output,
            #     self.item_embedding.weight[st_idx : ed_idx + 1].transpose(0, 1),
            # )
            # loss = self.loss_fn(logits[input_mask], (target_seq[target_mask] - st_idx))

            row_indices = torch.arange(group_indices.size(0), device=group_indices.device)
            _, max_positions = group_indices.max(dim=1)

            left_mask = torch.zeros_like(group_indices, dtype=torch.bool)
            left_mask[row_indices, max_positions] = True
            right_mask = group_indices == 1

            input_mask = group_indices.masked_fill(left_mask, 0)[:, :-1] != 0
            target_mask = group_indices.masked_fill(right_mask, 0)[:, 1:] != 0

            logits = torch.matmul(
                seq_output,
                self.item_embedding.weight[st_idx : ed_idx + 1].transpose(0, 1),
            )
            loss = self.loss_fn(logits[input_mask], (target_seq[target_mask] - st_idx))

        else:
            logits = torch.matmul(
                seq_output,
                self.item_embedding.weight.transpose(0, 1),
            )
            input_mask = item_seq != 0
            target_mask = target_seq != 0
            loss = self.loss_fn(logits[input_mask], target_seq[target_mask])

        return loss

    @torch.no_grad
    def calculate_logits(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        prob = output @ self.item_embedding.weight.T
        return prob

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        if self.stage == "run":
            if self.flag != "full":
                begin_item_id, end_item_id = self.domain_ranges[self.flag]
                domain_mask = torch.ones_like(scores, dtype=torch.bool)
                domain_mask[:, begin_item_id : end_item_id + 1] = False
                scores = scores.masked_fill(domain_mask, -torch.inf)
        elif self.stage == "tun":
            if self.flag == "full":
                begin_item_id, end_item_id = self.domain_ranges[self.config["target_dom"]]
                domain_mask = torch.ones_like(scores, dtype=torch.bool)
                domain_mask[:, begin_item_id : end_item_id + 1] = False
                scores = scores.masked_fill(domain_mask, -torch.inf)
        return scores
