for gpu_id in 0; do
    for seed in 2025; do
        python pretrain.py -m stage=run gpu_id=$gpu_id seed=$seed
        for target_dom in dom1 dom2 dom3 dom4; do
            python decoding.py -m stage=dec gpu_id=$gpu_id seed=$seed target_dom=$target_dom train_batch_size=32
        done
        for target_dom in dom1 dom2 dom3 dom4; do
            python finetune.py -m stage=tun gpu_id=$gpu_id seed=$seed train_type=new target_dom=$target_dom
            python finetune.py -m stage=tun gpu_id=$gpu_id seed=$seed train_type=sim target_dom=$target_dom
            python finetune.py -m stage=tun gpu_id=$gpu_id seed=$seed train_type=full target_dom=$target_dom
        done
    done
done
