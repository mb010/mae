import wandb
import pytorch_lightning as pl
import logging
import torch


from mae.config import load_config, update_config
from dataloading.datamodules import datasets
from paths import Path_Handler, create_path
from pytorch_lightning.callbacks import LearningRateMonitor

from finetune.main import run_finetuning
from finetune.dataloading import finetune_datasets
from model_timm import MAE
from mae.vit import ViT_Encoder, Transformer


def run_pretraining(config, datamodule, experiment_dir, wandb_logger):
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    ## Load up config from yml files ##
    config = load_config()
    update_config(config)

    wandb.init(project=config["project_name"])

    config["run_id"] = str(wandb.run.id)

    paths = Path_Handler()._dict()

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
        num_workers=config["data"]["num_workers"],
        prefetch_factor=config["data"]["prefetch_factor"],
        persistent_workers=config["data"]["persistent_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    ## Run pretraining ##
    pretrain_checkpoint, model = run_pretraining(config, datamodule, experiment_dir, wandb_logger)

    wandb.save(pretrain_checkpoint.best_model_path)
    # wadnb.save()

    finetune_datamodule = finetune_datasets[["dataset"]]()
    run_finetuning(config, model.encoder, finetune_datamodule, wandb_logger)

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
