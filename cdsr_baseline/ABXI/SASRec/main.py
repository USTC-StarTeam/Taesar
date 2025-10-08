import argparse
import os
import random
from os.path import join

import numpy as np
import torch

from noter import Noter
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="ABXI-Experiment")
    parser.add_argument("--name", type=str, default="ABXI", help="name of the model")
    parser.add_argument("--ver", type=str, default="v1.0", help="final")

    # Data
    parser.add_argument("--data", type=str, default="Full")
    parser.add_argument("--n_doms", type=int, default=1, help="# of domains")
    parser.add_argument("--len_max", type=int, default=128, help="max sequence length")
    parser.add_argument("--raw", action="store_true", help="preprocess or not")
    parser.add_argument("--n_neg", type=int, default=128, help="# negative samples")
    parser.add_argument("--n_mtc", type=int, default=30000, help="# eval samples")

    # Model
    parser.add_argument("--d_embed", type=int, default=128, help="dimension of embed")
    parser.add_argument("--n_attn", type=int, default=2, help="# layer model")
    parser.add_argument("--n_head", type=int, default=2, help="# attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--temp", type=float, default=0.75, help="temperature")

    # Training
    parser.add_argument("--cuda", type=str, default="5", help="running device")
    parser.add_argument("--seed", type=int, default=3407, help="random seeding")
    parser.add_argument("--bs", type=int, default=128, help="train batch size")
    parser.add_argument("--bse", type=int, default=2048, help="eval batch size")
    parser.add_argument("--n_worker", type=int, default=4, help="# dataloader worker")
    parser.add_argument("--n_epoch", type=int, default=300, help="# epoch maximum")
    parser.add_argument("--n_warmup", type=int, default=10, help="# warmup epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay")
    parser.add_argument("--lr_g", type=float, default=0.3162, help="scheduler gamma")
    parser.add_argument("--lr_p", type=int, default=30, help="scheduler patience")

    args = parser.parse_args()
    args.n_warmup = min(max(0, args.n_epoch - 1), args.n_warmup)
    if args.cuda == "cpu":
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.cuda}")
    args.len_trim = args.len_max - 3  # leave-one-out
    args.es_p = (args.lr_p + 1) * 2 - 1

    # paths
    args.path_root = os.getcwd()
    args.path_data = join(args.path_root, "data", args.data)
    args.path_log = join(args.path_root, "log")
    for p in [args.path_data, args.path_log]:
        if not os.path.exists(p):
            os.makedirs(p)

    args.f_raw_trn = join(args.path_data, f"{args.data}_{args.len_max}_preprocessed_trn.txt")
    args.f_raw_val = join(args.path_data, f"{args.data}_{args.len_max}_preprocessed_val.txt")
    args.f_raw_tst = join(args.path_data, f"{args.data}_{args.len_max}_preprocessed_tst.txt")

    args.f_data = join(args.path_data, f"{args.data}_{args.len_max}_seq.pkl")

    if args.raw and not os.path.exists(args.f_raw_trn):
        raise FileNotFoundError(f"Selected preprocessed dataset {args.data}-{args.len_max} does not exist.")
    if not args.raw and not os.path.exists(args.f_data):
        if os.path.exists(args.f_raw_trn):
            raise FileNotFoundError(f'Selected dataset {args.data}-{args.len_max} need process, specify "--raw" in the first run.')
        raise FileNotFoundError(f"Selected processed dataset {args.data}-{args.len_max} does not exist.")

    # seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # modeling
    noter = Noter(args)
    trainer = Trainer(args, noter)

    cnt_es, cnt_lr, ndcg_log = 0, 0, 0.0
    res_ranks = []  # This should be a list to hold results for all domains

    for epoch in range(1, args.n_epoch + 1):
        lr_cur = trainer.optimizer.param_groups[0]["lr"]
        # run_epoch should now return a list of results for each domain
        res_val = trainer.run_epoch(epoch)

        if epoch <= args.n_warmup:
            lr_str = f"{lr_cur:.5e}"
            noter.log_lr(f"| {lr_str[:3]}e-{lr_str[-1]} | warmup |")
            trainer.scheduler_warmup.step()

        else:
            # Sum up MRR from all domains for early stopping
            ndcg_val = res_val["ndcg@10"]
            # noter.log_valid(res_val)

            # if ndcg_val >= ndcg_log:
            if True:
                ndcg_log = ndcg_val
                cnt_es = 0
                cnt_lr = 0
                lr_str = f"{lr_cur:.5e}"
                # noter.log_lr(
                #     f"| {lr_str[:3]}e-{lr_str[-1]} |  0 /{args.lr_p:2} |  0 /{args.es_p:2} |"
                # )

                # run_test should now return a list of results for all domains
                res_ranks = trainer.run_test()
                print(res_ranks)
                # noter.log_test(res_ranks)

                trainer.scheduler.step(epoch)

            else:
                cnt_lr += 1
                cnt_es += 1
                if cnt_es > args.es_p:
                    noter.log_msg("\n[info] Exceeds maximum early-stop patience.")
                    break
                else:
                    trainer.scheduler.step(0)

                    lr_str = f"{lr_cur:.5e}"
                    noter.log_lr(f"| {lr_str[:3]}e-{lr_str[-1]} | {cnt_lr:2} /{args.lr_p:2} | {cnt_es:2} /{args.es_p:2} |")
                    if lr_cur != trainer.optimizer.param_groups[0]["lr"]:
                        cnt_lr = 0

    noter.log_final(res_ranks)
    noter.log_num_param(trainer.model)


if __name__ == "__main__":
    main()
