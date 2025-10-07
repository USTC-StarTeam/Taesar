import logging
import warnings
from typing import Tuple

import hydra
import torch
import torch.nn.functional as F
from recbole.config import Config
from recbole.data.utils import create_samplers, get_dataloader
from recbole.utils import init_seed, set_color
from tqdm import tqdm

from data.sequential_dataset import SequentialDataset
from model.seq2seq_sasrec import SASRec
from utils import (
    generate_tune_benchmark,
    get_domain_ranges,
    js_divergence,
    wandb_start_run_with_hydra,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_model(config, dataset) -> Tuple[SASRec, ...]:
    """Load model checkpoints and return model instances."""
    init_seed(config["seed"], config["reproducibility"])

    # Create all models
    model_names = ["full"] + [f"dom{i + 1}" for i in range(int(config["domains"]))]
    models = [SASRec(config, dataset).to(config["device"]) for _ in model_names]

    # Load checkpoints
    for model, ckpt_key in zip(models, [f"{name}_ckpt" for name in model_names]):
        checkpoint = torch.load(config[ckpt_key], map_location=config["device"], weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))

    return dict(zip(model_names, tuple(models)))


def _process_domain_items(
    logits_mix: torch.Tensor,
    logits_source: torch.Tensor,
    logits_target: torch.Tensor,
    target_mask: torch.Tensor,
    target_indices: torch.Tensor,
    is_dom_mask: torch.Tensor,
    next_tokens: torch.Tensor,
) -> torch.Tensor:
    """Process items from a specific domain using contrastive decoding."""
    # Global filtering with all items
    probs_M = F.softmax(logits_mix, dim=-1)
    probs_A = F.softmax(logits_target, dim=-1)
    probs_B = F.softmax(logits_source, dim=-1)

    H_A = -probs_A * torch.log(probs_A + 1e-10)
    alpha_global = (1 - H_A) / (1 - H_A.max(dim=-1, keepdim=True)[0] + 1e-10)
    js_global = js_divergence(probs_M, probs_B)
    beta_global = js_global / (js_global.max(dim=-1, keepdim=True)[0] + 1e-10)

    contrastive_scores = (1.0 + alpha_global) * logits_target + beta_global * (0 - logits_source)

    # Local filtering with dom1 items
    max_indices = contrastive_scores.argmax(dim=-1)
    convertible_mask = target_mask[max_indices]

    if convertible_mask.any():
        rows = torch.where(convertible_mask)[0]

        conv_logits_mix = logits_mix[rows][:, target_indices]
        conv_logits_target = logits_target[rows][:, target_indices]
        conv_logits_source = logits_source[rows][:, target_indices]

        conv_probs_M = F.softmax(conv_logits_mix, dim=-1)
        conv_probs_A = F.softmax(conv_logits_target, dim=-1)
        conv_probs_B = F.softmax(conv_logits_source, dim=-1)

        H_A_local = -conv_probs_A * torch.log(conv_probs_A + 1e-10)
        alpha_local = (1 - H_A_local) / (1 - H_A_local.max(dim=-1, keepdim=True)[0] + 1e-10)
        js_local = js_divergence(conv_probs_M, conv_probs_B)
        beta_local = js_local / (js_local.max(dim=-1, keepdim=True)[0] + 1e-10)

        adjusted_A = (1 + alpha_local) * conv_logits_target + beta_local * (0 - conv_logits_source)
        contrastive_scores[rows[:, None], target_indices] = adjusted_A

    converted_tokens = torch.argmax(contrastive_scores, dim=1) + 1
    next_tokens[is_dom_mask] = converted_tokens[is_dom_mask]

    return next_tokens


def contrastive_decoding(train_data, all_models, target_domain, domain_ranges):
    """Perform contrastive decoding across multiple domains."""
    # Get models and ranges
    full_model = all_models["full"]

    target_model = all_models[target_domain]
    target_range = domain_ranges[target_domain]

    domain_models = dict(list(all_models.items())[1:])
    other_models = {k: v for k, v in domain_models.items() if k != target_domain}
    other_ranges = {k: v for k, v in domain_ranges.items() if k != target_domain}

    # Set models to eval mode
    full_model.eval()
    target_model.eval()
    for model in other_models.values():
        model.eval()

    iter_data = tqdm(
        train_data,
        total=len(train_data),
        ncols=70,
        desc=set_color("Contrastive Decoding", "pink"),
    )

    all_user_id, all_item_list, all_seq_len = [], [], []

    with torch.no_grad():
        for interaction in iter_data:
            interaction = interaction.to(full_model.device)

            # Get logits from all models
            full_logits = full_model.calculate_logits(interaction)
            target_logits = target_model.calculate_logits(interaction)
            source_logits = {name: model.calculate_logits(interaction) for name, model in other_models.items()}

            # Extract original item sequence and metadata
            all_user_id.extend(interaction["user_id"].tolist())
            all_seq_len.extend((interaction["item_length"] + 1).tolist())

            original_items = F.pad(
                interaction["item_id_list"],
                (0, 1),
                value=0,
            ).scatter_(
                dim=1,
                index=interaction["item_length"].unsqueeze(1),
                src=interaction["item_id"].unsqueeze(1),
            )

            batch_item_list = [original_items[:, 0]]

            for t in range(full_logits.size(1)):
                next_tokens = original_items[:, t + 1].clone()

                # Prepare common variables for all domains (per time step)
                indices = torch.arange(target_logits.size(-1), device=target_logits.device)
                target_mask = (indices >= target_range[0]) & (indices <= target_range[1])
                target_indices = torch.where(target_mask)[0]

                # Process each domain
                domain_masks = {name: (next_tokens >= rng[0]) & (next_tokens <= rng[1]) for name, rng in other_ranges.items()}
                for domain_name, is_dom_mask in domain_masks.items():
                    if is_dom_mask.any():
                        next_tokens = _process_domain_items(
                            full_logits[:, t, :],
                            source_logits[domain_name][:, t, :],
                            target_logits[:, t, :],
                            target_mask,
                            target_indices,
                            is_dom_mask,
                            next_tokens,
                        )

                batch_item_list.append(next_tokens)

            batch_item_list = torch.stack(batch_item_list, dim=1)
            all_item_list.append(batch_item_list)

        all_item_list = torch.vstack(all_item_list).cpu().tolist()

    return sorted(zip(all_user_id, all_item_list, all_seq_len))


@hydra.main(config_path="config", config_name="overall", version_base=None)
@wandb_start_run_with_hydra
def main(config):
    # load configs
    config = Config(model=config["model_name"], dataset=config["dataset"], config_dict={**config})
    logger.info(config)

    dataset = SequentialDataset(config)
    logger.info(dataset)

    # build datasets and dataloader
    build_datasets = dataset.build()
    train_dataset = build_datasets[0]
    train_sampler, *_ = create_samplers(config, dataset, build_datasets[:3])
    train_dataloader = get_dataloader(config, "train")(
        config,
        train_dataset,
        train_sampler,
        shuffle=config["shuffle"],
    )

    # model_loading
    all_models = load_model(config, dataset)

    # contrastive decoding
    domain_ranges = get_domain_ranges(config["item_path"], config["domains"])
    target_min, target_max = domain_ranges[config["target_dom"]]
    sorted_data = contrastive_decoding(train_dataloader, all_models, config["target_dom"], domain_ranges)

    # generate tune benchmark files
    generate_tune_benchmark(config, sorted_data, config["target_dom"], target_min, target_max)


if __name__ == "__main__":
    main()
