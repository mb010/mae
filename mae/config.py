import yaml
from typing import Optional
from torch.optim import Adam, SGD

from paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def load_config(config_name: str = "global.yml", data_config: Optional[str] = None):
    """Helper function to load yaml config file, convert to python dictionary and return."""

    # load global config
    global_path = path_dict["config"] / config_name
    with open(global_path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    if data_config is None:
        dataset = config["dataset"]
        path = path_dict["config"] / f"{dataset}.yml"
    else:
        dataset = data_config.split(".")[0]
        path = path_dict["config"] / f"{dataset}.yml"

    # load data-set specific config
    with open(path, "r") as ymlconfig:
        dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # if loading a benchmark, use load the specific config
    preset = dataset_config["preset"]
    if preset != "none":
        path = path_dict["config"] / f"{dataset}-{preset}.yml"
        with open(path, "r") as ymlconfig:
            dataset_config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    # combine global with data-set specific config. dataset config has priority
    config.update(dataset_config)

    return config


def update_config(config):
    """
    Update config using presets
    """
    # Encoder presets
    if config["architecture"]["encoder"]["preset"].lower() == "none":
        pass

    elif config["architecture"]["encoder"]["preset"] == "vit_small":
        config["architecture"]["encoder"]["embed_dim"] = 384
        config["architecture"]["encoder"]["depth"] = 12
        config["architecture"]["encoder"]["heads"] = 6
        config["architecture"]["encoder"]["mlp_ratio"] = 4

    elif config["architecture"]["encoder"]["preset"] == "vit_base":
        config["architecture"]["encoder"]["embed_dim"] = 768
        config["architecture"]["encoder"]["depth"] = 12
        config["architecture"]["encoder"]["heads"] = 12
        config["architecture"]["encoder"]["mlp_ratio"] = 4

    elif config["architecture"]["encoder"]["preset"] == "vit_large":
        config["architecture"]["encoder"]["embed_dim"] = 1024
        config["architecture"]["encoder"]["depth"] = 24
        config["architecture"]["encoder"]["heads"] = 16
        config["architecture"]["encoder"]["mlp_ratio"] = 4

    elif config["architecture"]["encoder"]["preset"] == "vit_huge":
        config["architecture"]["encoder"]["embed_dim"] = 1280
        config["architecture"]["encoder"]["depth"] = 32
        config["architecture"]["encoder"]["heads"] = 16
        config["architecture"]["encoder"]["mlp_ratio"] = 4

    else:
        raise ValueError(f"unknown preset: {config['architecture']['encoder']['preset']}")

    # Decoder presets
    if config["architecture"]["decoder"]["preset"].lower() == "none":
        pass

    elif config["architecture"]["decoder"]["preset"] == "default":
        config["architecture"]["decoder"]["embed_dim"] = 512
        config["architecture"]["decoder"]["depth"] = 8
        config["architecture"]["decoder"]["heads"] = 16
        config["architecture"]["decoder"]["mlp_ratio"] = 4

    elif config["architecture"]["decoder"]["preset"] == "small":
        config["architecture"]["decoder"]["embed_dim"] = 256
        config["architecture"]["decoder"]["depth"] = 4
        config["architecture"]["decoder"]["heads"] = 8
        config["architecture"]["decoder"]["mlp_ratio"] = 4


def load_config_finetune(config_name: str = "finetune.yml"):
    """Helper function to load yaml config file, convert to python dictionary and return."""

    path = path_dict["config"] / config_name

    # load data-set specific config
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    return config
