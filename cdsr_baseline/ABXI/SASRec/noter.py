import os
import time
from argparse import Namespace
from os.path import join

import torch.nn as nn


class Noter(object):
    """console printing and saving into files"""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.t_start = time.time()
        self.f_log = join(
            args.path_log,
            f"{args.data}-{time.strftime('%m-%d-%H-%M', time.localtime())}-{str(args.device)[0] + str(args.device)[-1]}-{args.seed}-abxi.log",
        )

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

        # welcome
        self.log_msg(f"\n{'-' * 30} Experiment {self.args.name} {'-' * 30}")
        self.log_settings()

    def write(self, msg: str) -> None:
        with open(self.f_log, "a") as out:
            print(msg, file=out)

    def log_msg(self, msg: str) -> None:
        print(msg)
        self.write(msg)

    def log_settings(self) -> None:
        msg = (
            f"[Info] {self.args.name} (data:{self.args.data}, cuda:{self.args.cuda})\n"
            f"| Ver.  {self.args.ver} |\n"
            f"| len_max {self.args.len_max} | d_embed {self.args.d_embed} |\n"
            f"| n_attn {self.args.n_attn} | n_head {self.args.n_head} | dropout {self.args.dropout} |\n"
            f"| lr {self.args.lr:.2e} | l2 {self.args.l2:.2e} | lr_g {self.args.lr_g:.1f} | lr_p {self.args.lr_p} |\n\n"
            f"| seed {self.args.seed} |\n"
        )
        self.log_msg(msg)

    def log_num_param(self, model: nn.Module) -> None:
        self.log_msg(f"[info] model contains {sum(p.numel() for p in model.parameters() if p.requires_grad)} learnable parameters.\n")

    def log_lr(self, msg: str) -> None:
        msg = "           | lr  |     " + msg
        self.log_msg(msg)

    def log_train(
        self,
        i_epoch: int,
        losses: list[float],
        t_gap: float,
    ) -> None:
        """Log training results for multiple domains."""
        loss_str = " | ".join([f"{ll:.4f}"[:6] for ll in losses])
        msg = f"-epoch {i_epoch:>3} | tr  | los | {loss_str} | {t_gap:>5.1f}s |"
        self.log_msg(msg)

    def log_valid(self, res: list[dict]) -> None:
        """Log validation results in a structured table."""
        self._log_metrics_table("val", res)

    def log_test(self, res: list[dict]) -> None:
        """Log test results in a structured table."""
        self._log_metrics_table("tst", res)

    def _log_metrics_table(self, stage: str, res: list[dict]) -> None:
        """Helper function to log a table of metrics."""
        # Assume all dictionaries have the same keys and order
        if not res:
            self.log_msg(f"    | {stage} |  * | No results to log |  * |")
            return

        # Dynamically build the header based on the keys of the first dictionary
        header_keys = list(res[0].keys())
        header_metrics = " | ".join(f"{key:>7}" for key in header_keys)

        # Build the metric rows for each domain
        rows = []
        for i, domain_res in enumerate(res):
            # metric_values = " | ".join(f"{value:.4f}" for value in domain_res.values())
            # rows.append(f"Dom {i + 1:<3}| {metric_values}")

            metric_strs = [f"{v:7.5f}" for v in domain_res.values()]
            metric_values_formatted = " | ".join(metric_strs)
            rows.append(f" Dom {i + 1:<3}| {metric_values_formatted}")

        # Format the final message
        msg_header = f"    |   {stage}   | {header_metrics} |"
        msg_content = "\n".join([f"    | {row} |" for row in rows])

        # self.log_msg(f"\n{'-' * 80}")
        self.log_msg(msg_header)
        # self.log_msg(f"           {'-' * 72}")
        self.log_msg(msg_content)
        # self.log_msg(f"\n{'-' * 80}")

    def log_final(self, res: list[dict]) -> None:
        """Log final summary in a clear, formatted table."""
        self.log_msg(f"\n{'-' * 10} Experiment ended {'-' * 10}")

        # Summary of the experiment duration
        elapsed_minutes = (time.time() - self.t_start) / 60
        self.log_msg(f"[Result] {self.args.name} (Total time: {elapsed_minutes:.1f} min)\n")

        # Dynamically build the header based on the keys of the first dictionary
        header_keys = list(res[0].keys())
        header_metrics = " | ".join(f"{key:>7}" for key in header_keys)

        # Build the metric rows for each domain
        rows = []
        for i, domain_res in enumerate(res):
            metric_strs = [f"{v:7.5f}" for v in domain_res.values()]
            metric_values_formatted = " | ".join(metric_strs)
            rows.append(f" Dom {i + 1:<3}| {metric_values_formatted}")

        # Format the final message
        msg_header = f"    |   fin   | {header_metrics} |"
        msg_content = "\n".join([f"    | {row} |" for row in rows])

        self.log_msg(f"{'-' * 109}")
        self.log_msg(msg_header)
        self.log_msg(f"{'-' * 109}")
        self.log_msg(msg_content)
        self.log_msg(f"{'-' * 109}")
