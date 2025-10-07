import logging
import warnings

import hydra
from data.sequential_dataset import SequentialDataset
from model.seq2seq_sasrec import SASRec
from recbole.config import Config
from recbole.data.utils import create_samplers, get_dataloader
from recbole.utils import init_seed, set_color
from trainer import Trainer
from utils import wandb_start_run_with_hydra

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train_model(config, dataset, train_dataloader, valid_dataloader, test_dataloader, train_type):
    init_seed(config["seed"], config["reproducibility"])
    model = SASRec(config, dataset).to(config["device"])
    model.flag = train_type

    if train_type != "full":
        import torch

        checkpoint = torch.load(
            config["full_ckpt"],
            map_location=config["device"],
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))

        logger.info("Full checkpoint loaded !")

    trainer = Trainer(config, model)
    trainer.saved_model_file = config[f"{train_type}_ckpt"]

    if train_type == "full":
        trainer.fit(train_dataloader, valid_dataloader, saved=True, show_progress=config["show_progress"])
    else:
        trainer.pretrain(train_dataloader, valid_dataloader, saved=True, show_progress=config["show_progress"])

    test_result = trainer.evaluate(test_dataloader, load_best_model=True, show_progress=config["show_progress"])
    logger.info(set_color(f"{train_type} test ", "yellow") + f": {test_result}")


@hydra.main(config_path="config", config_name="overall", version_base=None)
@wandb_start_run_with_hydra
def main(config):
    # load configs
    config = Config(model=config["model_name"], dataset=config["dataset"], config_dict={**config})
    logger.info(config)

    dataset = SequentialDataset(config)
    logger.info(dataset)

    # build datasets and dataloader
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset, *rest = built_datasets
    dom_datasets = {}
    for i in range(0, len(rest), 3):
        dom_idx = i // 3 + 1
        dom_datasets[f"dom{dom_idx}_train"] = rest[i]
        dom_datasets[f"dom{dom_idx}_valid"] = rest[i + 1]
        dom_datasets[f"dom{dom_idx}_test"] = rest[i + 2]

    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    train_dataloader = get_dataloader(config, "train")(config, train_dataset, train_sampler, shuffle=config["shuffle"])
    valid_dataloader = get_dataloader(config, "valid")(config, valid_dataset, valid_sampler, shuffle=False)
    test_dataloader = get_dataloader(config, "test")(config, test_dataset, test_sampler, shuffle=False)

    dom_dataloader = {}
    for dom_name, dom_dataset in dom_datasets.items():
        if dom_name.endswith("_train"):
            split_type = "train"
            sampler = train_sampler
        elif dom_name.endswith("_valid"):
            split_type = "valid"
            sampler = valid_sampler
        elif dom_name.endswith("_test"):
            split_type = "test"
            sampler = test_sampler
        else:
            continue

        dom_dataloader[dom_name] = get_dataloader(config, split_type)(
            config,
            dom_dataset,
            sampler,
            shuffle=True if dom_name.endswith("_train") else False,
        )

    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f"[{config['train_batch_size']}]", "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f"[{config['train_neg_sample_args']}]", "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("evalid_batch_size", "cyan")
        + " = "
        + set_color(f"[{config['evalid_batch_size']}]", "yellow")
        + set_color(" evalid_args", "cyan")
        + ": "
        + set_color(f"[{config['evalid_args']}]", "yellow")
    )

    # model pretraining
    train_model(config, dataset, train_dataloader, valid_dataloader, test_dataloader, "full")
    for idx in range(int(config["domains"])):
        dom_name = f"dom{idx + 1}"
        train_model(
            config,
            dataset,
            train_dataloader,
            # dom_dataloader[f"{dom_name}_train"],
            dom_dataloader[f"{dom_name}_valid"],
            dom_dataloader[f"{dom_name}_test"],
            dom_name,
        )


if __name__ == "__main__":
    main()
