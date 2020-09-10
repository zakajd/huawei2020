import os
import cv2
import copy
import collections

import torch
import numpy as np
import albumentations as albu
import albumentations.pytorch as albu_pt

from src.augmentations import get_aug
from src.utils import ToCudaLoader

def get_dataloaders(
    root="data/raw",
    augmentation="light",
    batch_size=8,
    size=512,
    val_size=512,
    workers=6,
):
    """
    Args:
        root (str): Path to folder with data
        aumentation (str): Type of aug defined in `src.data.augmentations.py`
        batch_size (int): Number of images in stack
        size (int): Crop size to take from original image for testing
        val_size (int): Crop size to take from original image for validation
        workers (int): Number of CPU threads used to load images
    Returns:
        train_dataloader, val_dataloader
    """

    # Get augmentations
    train_aug = get_aug(augmentation, size=size)

    # Get datasets
    train_dataset = ClassificationDataset(root=root, transform=train_aug, train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    train_loader = ToCudaLoader(train_loader)

    val_loader, _ = get_val_dataloader(
        root,
        augmentation="val",
        batch_size=batch_size,
        size=val_size,
        workers=workers,
    )

    logger.info(f"Using fold: {fold}. Train size: {len(train_dataset)}")
    return train_loader, val_loader

def get_val_dataloader(
    root="data/raw", augmentation="test", batch_size=8, size=512, workers=6,
):
    """
    Returns only validation dataloader
    """
    aug = get_aug(augmentation, size=size)

    val_dataset = ClassificationDataset(root=root, transform=train_aug, train=False)

    # TODO: Add BalancedBatchSampler from Catalyst?
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        shuffle=False,
    )

    val_loader = ToCudaLoader(val_loader)
    logger.info(f"Val size: {len(val_dataset)}")
    return val_loader, val_dataset.classes


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Args:
        size: What type of resized images to take
    """

    def __init__(self, root="data/interim", transform=None, train=True, size=512):   
        # Read file with labels
        with open("data/raw/train_data/label.txt") as f:
            data = f.readlines()

        self.filenames, self.targets = [], []
        for row in data:
            path, label = row.strip("\n").split(",")
            self.filenames.append(os.path.join(root, f"train_data_{size}", path))
            self.targets.append(int(label))

        # Take 20% of the data for validation
        # Use same data as for training, because tasks are different
        if not train:
            val_size = int(len(self.targets) * 0.2)
            self.filenames = self.filenames[: val_size]
            self.targets = self.targets[: val_size]

        self.root = root
        self.transform = albu.Compose([albu_pt.ToTensorV2()]) if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image = cv2.imread(self.filenames[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.filenames)
