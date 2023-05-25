import wandb
import pytorch_lightning as pl
import logging

from pathlib import Path
import argparse

from paths import Path_Handler
from finetune.main import run_finetuning
from finetune.dataloading import finetune_datasets
from config import load_config_finetune
from architectures.models import MLP
from model_timm import MAE


def init_argparse():
    """
    Parse the config from the command line arguments
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [CONFIG_NAME]...",
        description="Train MAE according to config as determined by CONFIG_NAME file.",
    )
    parser.add_argument("config", default="finetune.yml", help="Finetuning config file name.")
    parser.add_argument("data_config", default="fits.yml", help="Data config for the given experiment.")
    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    args = init_argparse()

    # Load paths
    path_dict = Path_Handler()._dict()
    paths = Path_Handler(data=args.data_config["data"]["data_path"])._dict()

    # Load up finetuning config
    config_finetune = load_config_finetune(
        args.config,
        args.config["finetune"]["data_path"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["dataloading"]["num_workers"],
        prefetch_factor=config["dataloading"]["prefetch_factor"],
        persistent_workers=config["dataloading"]["persistent_workers"],
        pin_memory=config["dataloading"]["pin_memory"],
        img_size=config["data"]["img_size"],
        data_type=config["trainer"]["precision"],
        astroaugment=config["data"]["astroaugment"],
        fft=config["data"]["fft"],
        nchan=config["data"]["in_chans"],
        png=config["data"]["png"],
    )

    ## Run finetuning ##
    for seed in range(config_finetune["finetune"]["iterations"]):
        # for seed in range(1, 10):

        if config_finetune["finetune"]["run_id"].lower() != "none":
            experiment_dir = path_dict["files"] / config_finetune["finetune"]["run_id"] / "checkpoints"
            model = MAE.load_from_checkpoint(experiment_dir / "last.ckpt")
        else:
            model = MAE.load_from_checkpoint("model.ckpt")

        ## Load up config from model to save correct hparams for easy logging ##
        config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim
        # project_name = f"{config['project_name']}_finetune"
        project_name = f"FITS_MAE_finetune"

        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=path_dict["files"] / "finetune" / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = finetune_datasets[config["dataset"]](config)
        run_finetuning(config, model.encoder, finetune_datamodule, logger)
        logger.experiment.finish()
        wandb.finish()


if __name__ == "__main__":
    main()
