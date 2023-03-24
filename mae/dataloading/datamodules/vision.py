import logging
import pytorch_lightning as pl
import torchvision.transforms as T
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import STL10
from typing import Dict

from mae.dataloading.utils import compute_mu_sig_images
from mae.paths import Path_Handler


class Base_DataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size: int, dataloading_kwargs: Dict):
        """
        Args:
            path: path to dataset
            batch_size: batch size
            dataloading_kwargs: kwargs for dataloader
        """
        super().__init__()

        self.path = path
        self.batch_size = batch_size

        self.dataloading_kwargs = dataloading_kwargs
        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"], batch_size=self.batch_size, shuffle=True, **self.dataloading_kwargs
        )
        return loader

    def val_dataloader(self):
        loaders = [
            DataLoader(data, shuffle=False, **self.dataloading_kwargs) for _, data in self.data["val"]
        ]
        return loaders

    def test_dataloader(self):
        loaders = [
            DataLoader(data, batch_size=250, shuffle=False, **self.dataloading_kwargs)
            for _, data in self.data["test"]
        ]
        return loaders


class FineTuning_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # override default paths via config if desired
        paths = Path_Handler(**config.get("paths_to_override", {}))
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.mu, self.sig = config["data"]["mu"], config["data"]["sig"]

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.config["finetune"]["batch_size"],
            num_workers=8,
            prefetch_factor=20,
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data["val"],
            batch_size=200,
            num_workers=8,
            prefetch_factor=20,
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.data["test"],
            batch_size=200,
            num_workers=8,
            prefetch_factor=20,
            shuffle=False,
        )
        return loader


class STL10_DataModule(Base_DataModule):
    """
    DataModule for STL10 dataset.

    Args:
        path (str): Path to data folder.
        batch_size (int): Batch size for dataloaders.
        dataloading_kwargs (dict): Keyword arguments to pass to dataloaders.
        **kwargs: Additional keyword arguments to pass to dataloaders.
    """

    def __init__(self, path, batch_size: int, dataloading_kwargs: Dict, **kwargs):
        super().__init__(path, batch_size, dataloading_kwargs)

        # Standard imagenet normalization
        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)

    def prepare_data(self):
        STL10(root=self.path, split="train+unlabeled", download=True)
        STL10(root=self.path, split="test", download=True)

    def setup(self, stage=None):
        train_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
                T.RandomResizedCrop(96, scale=(0.2, 1.0), antialias=True),
            ]
        )
        test_transform = T.Compose([T.ToTensor(), T.Normalize(self.mu, self.sig)])

        self.data["train"] = STL10(root=self.path, split="train+unlabeled", transform=train_transform)

        # List of (name, train_dataset) tuples to evaluate linear layer
        self.data["val"] = [
            ("STL10_train", STL10(root=self.path, split="train", transform=test_transform)),
            ("STL10_test", STL10(root=self.path, split="test", transform=test_transform)),
        ]

        self.data["test"] = [
            ("STL10_test", STL10(root=self.path, split="test", transform=test_transform)),
        ]

        # List of (name, train_dataset) tuples to train linear evaluation layer
        self.data["eval_train"] = [
            {
                "name": "STl10_train",
                "n_classes": 10,
                "data": STL10(root=self.path, split="train", transform=test_transform),
            },
        ]


class Imagenette_DataModule(Base_DataModule):
    def __init__(self, config):
        norms = _get_imagenet_norms()
        super().__init__(config, **norms)

    def setup(self):
        self.data["train"] = ImageFolder(self.path / "train", transform=self.T_train)

        # List of (name, train_dataset) tuples to evaluate linear laye
        self.data["val"] = [
            ("imagenette_train", ImageFolder(self.path / "train", transform=self.T_test)),
            ("imagenette_val", ImageFolder(self.path / "val", transform=self.T_test)),
        ]

        self.data["test"] = [
            ("imagenette_val", ImageFolder(self.path / "val", transform=self.T_test)),
        ]

        # List of (name, train_dataset) tuples to train linear evaluation layer
        self.data["eval_train"] = [
            (
                "imagenette_train",
                ImageFolder(self.path / "train", transform=self.T_test),
                {"val": (0, 1), "test": (0,)},
            ),
        ]
