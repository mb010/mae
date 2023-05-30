import wandb
import pytorch_lightning as pl
import logging
import torch


from pathlib import Path
import argparse

from paths import Path_Handler
from finetune.main import run_finetuning
from finetune.dataloading import finetune_datasets
from config import load_config_finetune, load_config
from architectures.models import MLP
from model_timm import MAE
from vit import ViT_Encoder, Transformer


def init_argparse():
    """
    Parse the config from the command line arguments
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [CONFIG_NAME]...",
        description="Train MAE according to config as determined by CONFIG_NAME file.",
    )
    parser.add_argument("config", default="finetune.yml", help="Finetuning config file name.")
    parser.add_argument("data_config", default="fits.yml", help="Experiment data config file name.")
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
    # paths = Path_Handler(data=args.data["data"]["data_path"])._dict()

    # Load up finetuning config
    config_finetune = load_config_finetune(args.config)
    experiment_config = load_config(str("global.yml"), str(args.data_config))

    ## Run finetuning ##
    for seed in range(config_finetune["finetune"]["iterations"]):
        # for seed in range(1, 10):

        experiment_dir = path_dict["files"] / config_finetune["finetune"]["run_id"] / "checkpoints"
        checkpoint = torch.load(experiment_dir / "last.ckpt")
        state_dict = checkpoint["state_dict"]
        # Load config from checkpoint
        config = checkpoint["hyper_parameters"]

        encoder = ViT_Encoder(
            img_size=experiment_config["data"]["img_size"],
            in_chans=experiment_config["data"]["in_chans"],
            patch_size=experiment_config["architecture"]["encoder"]["patch_size"],
            embed_dim=experiment_config["architecture"]["encoder"]["embed_dim"],
            depth=experiment_config["architecture"]["encoder"]["depth"],
            num_heads=experiment_config["architecture"]["encoder"]["num_heads"],
            mlp_ratio=experiment_config["architecture"]["encoder"]["mlp_ratio"],
        )

        decoder = Transformer(
            embed_dim=experiment_config["architecture"]["decoder"]["embed_dim"],
            depth=experiment_config["architecture"]["decoder"]["depth"],
            num_heads=experiment_config["architecture"]["decoder"]["num_heads"],
            mlp_ratio=experiment_config["architecture"]["decoder"]["mlp_ratio"],
        )

        model = MAE(encoder, decoder, **experiment_config["model"])
        model.load_state_dict(state_dict)

        ## Load up config from model to save correct hparams for easy logging ##
        # config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim
        project_name = f"FITS_MAE_finetune"
        experiment_config["finetune"] = config["finetune"]

        experiment_config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=path_dict["files"] / "finetune" / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = finetune_datasets[experiment_config["finetune"]["dataset"]](
            path=experiment_config["finetune"]["data_path"],
            batch_size=experiment_config["data"]["batch_size"],
            num_workers=experiment_config["dataloading"]["num_workers"],
            prefetch_factor=experiment_config["dataloading"]["prefetch_factor"],
            persistent_workers=experiment_config["dataloading"]["persistent_workers"],
            pin_memory=experiment_config["dataloading"]["pin_memory"],
            img_size=experiment_config["data"]["img_size"],
            data_type=experiment_config["trainer"]["precision"],
            astroaugment=experiment_config["data"]["astroaugment"],
            fft=experiment_config["data"]["fft"],
            png=experiment_config["data"]["png"],
            nchan=experiment_config["data"]["in_chans"],
            test_size=experiment_config["finetune"]["val_size"],
        )
        run_finetuning(experiment_config, encoder, finetune_datamodule, logger)
        logger.experiment.finish()
        wandb.finish()


if __name__ == "__main__":
    main()
