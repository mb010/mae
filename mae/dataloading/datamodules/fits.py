from typing import Dict, Union
import torch.utils.data as D
import torch
import numpy as np
import os
import sys
import torchvision.transforms as T
import pytorch_lightning as pl

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import DataLoader

from mae.dataloading.datamodules.vision import Base_DataModule
from astroaugmentations.datasets.MiraBest_F import MBFRFull, MBFRConfident, MBFRUncertain, MiraBest_FITS
from astroaugmentations.datasets.fits import FitsDataset
import astroaugmentations as AA
import albumentations as A
import torch.fft as fft


class FITS_DataModule(Base_DataModule):
    """DataModule for FITS files of your choice.

    Args:
        path (str): Path to data folder.
        batch_size (int): Batch size for dataloaders.
        and more (TODO)
    """

    def __init__(
        self,
        path="",
        batch_size: int = 32,
        num_workers: int = 1,
        prefetch_factor: int = 8,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        img_size: bool = 128,
        MiraBest_FITS_root: str = "/share/nas2_5/mbowles/_data/MiraBest_FITS",
        data_type: Union[str, type] = torch.float32,
        astroaugment: bool = True,
        fft: bool = True,  # TODO
        png: bool = False,  # TODO
        nchan: int = 3,  # TODO
        **kwargs,
    ):
        super().__init__(
            path,
            batch_size,
            num_workers,
            prefetch_factor,
            persistent_workers,
            pin_memory,
            **kwargs,
        )
        # Standard imagenet normalization
        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)
        # Params
        self.astroaugment = astroaugment
        self.fft = fft
        self.png = png
        self.nchan = nchan
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_type = {
            "torch.float32": torch.float32,
            "16-mixed": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
            "32-true": torch.float32,
            "64-true": torch.float64,
            64: torch.float64,
            32: torch.float32,
            16: torch.float16,
            "64": torch.float64,
            "32": torch.float32,
            "16": torch.float16,
            "bf16": torch.bfloat16,
        }[data_type]
        self.MiraBest_FITS_root = MiraBest_FITS_root
        self.train_transform, self.test_transform, self.eval_transform = self._build_transforms()

    def setup(self):
        self.data["train"] = FitsDataset(
            self.path,
            crop_size=self.img_size,
            stage="train",
            transform=self.train_transform,
            data_type=self.data_type,
            aug_type="albumentations",
        )
        self.data["val"] = [
            (
                "FIRST_val",
                FitsDataset(
                    self.path,
                    crop_size=self.img_size,
                    stage="val",
                    transform=self.test_transform,
                    data_type=self.data_type,
                    aug_type="albumentations",
                ),
            )
        ]
        self.data["test"] = [
            (
                "FIRST_test",
                FitsDataset(
                    self.path,
                    crop_size=self.img_size,
                    stage="test",
                    transform=self.test_transform,
                    data_type=self.data_type,
                    aug_type="albumentations",
                ),
            )
        ]
        # List of (name, train_dataset) tuples to train linear evaluation layer
        self.data["eval_train"] = [
            {
                "name": "MiraBest_FIRST_train",
                "n_classes": 2,
                "data": MiraBest_FITS(
                    root=self.MiraBest_FITS_root,
                    train=True,
                    transform=self.eval_transform,
                    data_type=self.data_type,
                    aug_type="albumentations",
                ),
            },
        ]
