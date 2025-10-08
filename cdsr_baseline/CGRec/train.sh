export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +'%Y%m%d-%H%M%S')

#LOG_FILE="./log/BEST/Pretrain-BEST-20251005-50-100.txt"
LOG_FILE="./log/BEST/Pretrain-BEST-${TIMESTAMP}.txt"

torchrun --nproc_per_node=1 --master_port=15556 run_pretrain.py \
--batch_size=128 \
--hidden_size=128 \
--num_hidden_layers=3 \
--num_attention_head=2 \
--epoch=50 \
--max_seq_length=128 \
--data_name='BEST' \
--strd_ym='202509' \
--shaply_value='y' \
--hierarhical='y' \
--lr=1e-4 \
--local_rank=0 \
>> "$LOG_FILE" 2>&1

# --num_hidden_layers 1,2,3
# --lr 1e-3,3e-4,1e-4