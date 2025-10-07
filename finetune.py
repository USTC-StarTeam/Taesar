import logging
import warnings

import hydra
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, set_color

import wandb
from model.seq2seq_sasrec import SASRec
from trainer import Trainer
from utils import wandb_start_run_with_hydra

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train_model(config):
    init_seed(config["seed"], config["reproducibility"])

    if config["train_type"] == "sim":
        from recbole.data.dataset import SequentialDataset

        config["benchmark_filename"] = [
            f"{config['target_dom']}.train",
            f"{config['target_dom']}.valid",
            f"{config['target_dom']}.test",
        ]
    elif config["train_type"] == "new":
        from recbole.data.dataset import SequentialDataset

        config["benchmark_filename"] = [
            f"{config['seed']}.{config['target_dom']}.train",
            f"{config['target_dom']}.valid",
            f"{config['target_dom']}.test",
        ]
    elif config["train_type"] == "full":
        from data.sequential_dataset import SequentialDataset

        config["benchmark_filename"] = [
            "full.train",
            f"{config['target_dom']}.valid",
            f"{config['target_dom']}.test",
        ]

    dataset = SequentialDataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRec(config, dataset).to(config["device"])
    model.flag = config["train_type"]

    trainer = Trainer(config, model)
    trainer.saved_model_file = f"{config['tune_ckpt']}-{config['target_dom']}-{config['train_type']}"
    trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        test=True,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    logger.info(set_color("test result", "yellow") + f": {test_result}")
    for k, v in test_result.items():
        wandb.run.summary[k] = v


@hydra.main(config_path="config", config_name="overall", version_base=None)
@wandb_start_run_with_hydra
def main(config):
    # load configs
    config = Config(model=config["model_name"], dataset=config["dataset"], config_dict={**config})
    logger.info(config)

    # models training
    train_model(config)


if __name__ == "__main__":
    main()
