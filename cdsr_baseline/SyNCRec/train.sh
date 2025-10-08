export CUDA_VISIBLE_DEVICES=2

TIMESTAMP=$(date +'%Y%m%d-%H%M%S')

LOG_FILE="./log/BEST/Pretrain-BEST-${TIMESTAMP}.txt"


torchrun --nproc_per_node=1 --master_port=15555 run_pretrain.py \
--batch_size=128 \
--hidden_size=128 \
--num_hidden_layers=3 \
--num_attention_head=2 \
--epoch=50 \
--max_seq_length=128 \
--data_name='BEST' \
--cross_expert_ratio=0.8 \
--strd_ym='202509' \
--cross_detach='y' \
--single_detach='y' \
--expert_layer='transformer' \
--lrb='y' \
--mip='y' \
--expert_num=8 \
--task_num=5 \
--lr=1e-4 \
--local_rank=2 \
>> "$LOG_FILE" 2>&1


# --num_hidden_layers 1,2,3
# --lr 1e-3,3e-4,1e-4
# --expert_num 4,6,8