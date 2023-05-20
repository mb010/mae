import wandb
import pytorch_lightning as pl
import logging
import torch
import argparse

from config import load_config
from dataloading.datamodules import datasets
from paths import Path_Handler, create_path
from pytorch_lightning.callbacks import LearningRateMonitor
from utils import interpolate_pos_embed

from finetune.main import run_finetuning
from finetune.dataloading import finetune_datasets
from model_timm import MAE
from vit import ViT_Encoder, Transformer


def run_pretraining(config, paths, datamodule, experiment_dir, wandb_logger):
    pl.seed_everything(config["seed"])

    # Save model for test evaluation
    # TODO might be better to use val/supervised_loss when available
    ## Creates experiment path if it doesn't exist already ##

    ## Initialise checkpoint ##
    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(
        # **checkpoint_mode[config["evaluation"]["checkpoint_mode"]],
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=experiment_dir / "checkpoints",
        save_last=True,
        # filename="{epoch}-{step}-{loss_to_monitor:.4f}",  # filename may not work here TODO
        filename="model",
        save_weights_only=True,
    )

    ## Initialise data and run set up ##
    # datamodule.prepare_data()
    # datamodule.setup()
    # logging.info(f"mean: {pretrain_data.mu}, sigma: {pretrain_data.sig}")

    ## Initialise callbacks ##
    callbacks = [pretrain_checkpoint]

    # add learning rate monitor, only supported with a logger
    if wandb_logger is not None:
        callbacks += [LearningRateMonitor(logging_interval="epoch")]

    logging.info(f"Threads: {torch.get_num_threads()}")

    ## Initialise pytorch lightning trainer ##
    pre_trainer = pl.Trainer(
        **config["trainer"],
        max_epochs=config["model"]["n_epochs"],
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=callbacks,
        # log_every_n_steps=200,
    )

    encoder = ViT_Encoder(
        img_size=config["data"]["img_size"],
        in_chans=config["data"]["in_chans"],
        patch_size=config["architecture"]["encoder"]["patch_size"],
        embed_dim=config["architecture"]["encoder"]["embed_dim"],
        depth=config["architecture"]["encoder"]["depth"],
        num_heads=config["architecture"]["encoder"]["num_heads"],
        mlp_ratio=config["architecture"]["encoder"]["mlp_ratio"],
    )

    if config["architecture"]["encoder"]["pretrained"]:
        checkpoint = torch.load(
            paths["weights"] / f"mae_{config['architecture']['encoder']['preset']}.pth"
        )
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = encoder.state_dict()

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(encoder, checkpoint_model)

        # interpolate patch_embed
        new_size = encoder.patch_embed.proj.weight.shape[-1]
        orig_size = checkpoint_model["patch_embed.proj.weight"].shape[-1]
        if new_size != orig_size:
            print(
                "patch_embed.proj interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            patch_embed_weight = torch.nn.functional.interpolate(
                checkpoint_model["patch_embed.proj.weight"],
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            checkpoint_model["patch_embed.proj.weight"] = patch_embed_weight

        encoder.load_state_dict(checkpoint_model, strict=False)

    decoder = Transformer(
        embed_dim=config["architecture"]["decoder"]["embed_dim"],
        depth=config["architecture"]["decoder"]["depth"],
        num_heads=config["architecture"]["decoder"]["num_heads"],
        mlp_ratio=config["architecture"]["decoder"]["mlp_ratio"],
    )

    model = MAE(encoder, decoder, **config["model"])

    # profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    # profile_art.add_file(glob.glob(str(experiment_dir / "*.pt.trace.json"))[0], "trace.pt.trace.json")
    # wandb.run.log_artifact(profile_art)

    # Train model #
    pre_trainer.fit(model, datamodule)
    pre_trainer.test(model, dataloaders=datamodule)

    return pretrain_checkpoint, model


def init_argparse():
    """
    Parse the config from the command line arguments
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [CONFIG_NAME]...",
        description="Train MAE according to config as determined by CONFIG_NAME file.",
    )
    parser.add_argument("config", default="global.yml", help="Config file name.")
    parser.add_argument("dataconfig", default=None, help="Data config name.")
    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    args = init_argparse()

    torch.set_float32_matmul_precision("high")

    ## Load up config from yml files ##
    config = load_config(str(args.config), str(args.dataconfig))
    # Merges both configs together. Indexes second config based on the 'project_name' parameter.

    wandb.init(project=config["project_name"], config=config)

    config["run_id"] = str(wandb.run.id)

    paths = Path_Handler(data=config["data"]["data_path"])._dict()

    wandb_logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        # and will then add e.g. run-20220513_122412-l5ikqywp automatically
        save_dir=paths["files"] / config["run_id"],
        # log_model="True",
        # reinit=True,
        config=config,
    )

    config["files"] = paths["files"]

    experiment_dir = paths["files"] / config["run_id"]
    create_path(experiment_dir)

    # Load datamodule
    datamodule = datasets[config["dataset"]](
        paths[config["dataset"]],
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

    ## Run pretraining ##
    pretrain_checkpoint, model = run_pretraining(config, paths, datamodule, experiment_dir, wandb_logger)

    wandb.save(pretrain_checkpoint.best_model_path)
    # wadnb.save()

    finetune_datamodule = finetune_datasets[["dataset"]]()
    run_finetuning(config, model.encoder, finetune_datamodule, wandb_logger)

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
