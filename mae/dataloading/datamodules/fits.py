from typing import Dict
import torch.utils.data as D
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
        path,
        batch_size: int,
        num_workers: int = 1,
        prefetch_factor: int = 8,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        img_size: bool = 128,
        MiraBest_FITS_root: str = "/share/nas2_5/mbowles/_data/MiraBest_FITS",
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
        self.batch_size = batch_size
        self.img_size = img_size
        self.MiraBest_FITS_root = MiraBest_FITS_root
        self.train_transform = A.Compose(
            [
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(
                        dropout_p=0.8,
                        dropout_mag=0.5,  # RFI Overflagging
                        noise_p=0.5,
                        noise_mag=0.5,  # Noise Injection
                        rfi_p=0.5,
                        rfi_mag=1,
                        rfi_prob=0.01,  # RFI injection
                    ),
                    p=1,
                ),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(  # Fourrier transform the same way as before.
                        dropout_p=0.0,  # RFI Overflagging
                        noise_p=0.0,  # Noise Injection
                        rfi_p=0.0,  # RFI injection
                    ),
                    p=1,
                ),
            ]
        )

    def setup(self, stage=None):
        self.data["train"] = [
            (
                "FIRST_train",
                FitsDataset(
                    self.path,
                    crop_size=self.img_size,
                    stage="train",
                    transform=self.train_transform,
                ),
            )
        ]
        self.data["val"] = [
            (
                "FIRST_val",
                FitsDataset(
                    self.path,
                    crop_size=self.img_size,
                    stage="val",
                    transform=self.test_transform,
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
                    transform=self.train_transform,
                ),
                # "data": STL10(root=self.path, split="train", transform=test_transform),
            },
        ]
