#!/bin/bash

# python -m debugpy --listen 3187 --wait-for-client main.py \
python main.py \
--n_epoch 50 \
--n_worker 28 \
--n_attn 3 \
--bs 128 \
--bse 1024 \
--d_embed 128 \
--data BEST \
--lr 1e-4 \
--l2 0 \
--rd 32 \
--ri 32 \
--cuda 4 \
--seed 2025 \
--n_doms 4 \
--len_max 128 \
--n_warmup 10 \
--raw

# --rd 8,16,32
# --ri 8,16,32
# --lr 1e-3,3e-4,1e-4
# --n_attn 1,2,3
