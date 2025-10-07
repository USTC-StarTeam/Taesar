import glob
import os
from functools import wraps

import hydra
import pandas as pd
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb


class WandbLogger(object):
    def __init__(self, config):
        self.config = config
        self.setup()

    def setup(self):
        self._wandb = wandb
        self._wandb.define_metric("train/*", step_metric="train_step")
        self._wandb.define_metric("valid/*", step_metric="valid_step")

    def log_metrics(self, metrics, head="train", commit=True):
        if head:
            metrics = self._add_head_to_metrics(metrics, head)
            self._wandb.log(metrics, commit=commit)
        else:
            self._wandb.log(metrics, commit=commit)

    def log_eval_metrics(self, metrics, head="eval"):
        metrics = self._add_head_to_metrics(metrics, head)
        for k, v in metrics.items():
            self._wandb.run.summary[k] = v

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}"] = v

        return head_metrics


def wandb_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: DictConfig, *args, **kwargs):
        config = OmegaConf.to_container(config, resolve=True)
        wandb.init(
            project=config.get("base", {}).get("experiment_name", "default_project"),
            name=config.get("base", {}).get("run_name", "default_run"),
            config=config,
            # mode="offline",
        )
        wandb.run.log_code(
            root="./",
            include_fn=lambda path: path.endswith((".py", ".ipynb", ".sh", ".yaml", ".yml")),
        )

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        for file in glob.glob(os.path.join(output_dir, ".hydra/*")):
            wandb.save(file)
        with logging_redirect_tqdm():
            out = func(config, *args, **kwargs)
        wandb.save(os.path.join(output_dir, "main.log"))
        wandb.finish()
        return out

    return wrapper


def get_domain_ranges(filename, domains):
    df = pd.read_csv(filename, header=None, names=["item"])
    full_items = df["item"].tolist()
    domains = [f"dom{i + 1}" for i in range(domains)]

    ranges = {}
    current_idx = 1
    for domain in domains:
        suffix = f"_{chr(65 + domains.index(domain))}"
        domain_items = [item for item in full_items if item.endswith(suffix)]
        count = len(domain_items)
        ranges[domain] = (current_idx, current_idx + count - 1)
        current_idx += count

    return ranges


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction="none") + F.kl_div(q.log(), m, reduction="none"))


def generate_tune_benchmark(config, sorted_data, target_dom, min_idx, max_idx):
    origin_train_path = os.path.join(config["data_path"], f"{config['dataset']}.{target_dom}.train.inter")
    train_path = config["new_inter_path"]

    with open(train_path, "w") as dst, open(origin_train_path, "r") as src:
        dst.write(src.read())

    with open(train_path, "a") as file:
        # file.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for user_id, item_list, seq_len in sorted_data:
            item_seq = item_list[:seq_len]
            item_seq = [str(x) for x in item_seq if x >= min_idx and x <= max_idx]
            item_seq = [item for i, item in enumerate(item_seq) if i == 0 or item != item_seq[i - 1]]
            if len(item_seq) >= 2:
                target_item = item_seq[-1]
                item_seq = item_seq[:-1]
                file.write(f"{user_id}\t{' '.join(item_seq)}\t{target_item}\n")

    input_data = pd.read_csv(train_path, sep="\t")
    input_data = input_data.drop_duplicates()
    input_data = input_data.sort_values(by="user_id:token")
    input_data.to_csv(train_path, sep="\t", index=False)
