import os
import cv2
import pathlib
import itertools

import torch
import numpy as np
import pandas as pd
from loguru import logger
import albumentations as albu
import albumentations.pytorch as albu_pt

from src.augmentations import get_aug
from src.utils import ToCudaLoader


def get_dataloaders(
    root="data/raw",
    augmentation="light",
    batch_size=8,
    size=512,
    val_size=768,
    workers=6,
):
    """
    Args:
        root (str): Path to folder with data
        aumentation (str): Type of aug defined in `src.data.augmentations.py`
        batch_size (int): Number of images in stack
        size (int): Size of images used for training
        val_size (int): Size of images used for validation
        workers (int): Number of CPU threads used to load images
    Returns:
        train_dataloader
        val_dataloader
        val_indexes (list): Order in which indexes are sampled from validation dataset
    """

    # Get augmentations
    train_aug = get_aug(augmentation, size=size)

    # Get datasets
    train_dataset = ClassificationDataset(
        root=root, transform=train_aug, train=True, size=size)

    gb_sampler = GroupedBatchSampler(
        sampler=torch.utils.data.sampler.SequentialSampler(train_dataset),
        # sampler=torch.utils.data.sampler.RandomSampler(train_dataset),
        group_ids=train_dataset.group_ids,
        batch_size=batch_size,
        drop_uneven=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=workers,
        pin_memory=True,
        batch_sampler=gb_sampler,
    )

    train_loader = ToCudaLoader(train_loader)

    val_loader, val_indexes = get_val_dataloader(
        root,
        augmentation="val",
        batch_size=batch_size,
        size=val_size,
        workers=workers,
    )

    logger.info(f"Train size: {len(train_dataset)}")
    return train_loader, val_loader, val_indexes


def get_val_dataloader(
        root="data/interim", augmentation="val", batch_size=8, size=768, workers=6):
    """
    Returns:
        val_loader (DataLoader)
        indexes (list): Order in which indexes are sampled from dataset
    """
    val_aug = get_aug(augmentation, size=size)

    val_dataset = ClassificationDataset(
        root=root, transform=val_aug, train=False, size=size)

    gb_sampler = GroupedBatchSampler(
        sampler=torch.utils.data.sampler.SequentialSampler(val_dataset),
        group_ids=val_dataset.group_ids,
        batch_size=batch_size,
        drop_uneven=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=workers,
        pin_memory=False,
        batch_sampler=gb_sampler,
    )

    val_loader = ToCudaLoader(val_loader)
    logger.info(f"Val size: {len(val_dataset)}")
    # GB sampler changes the order of images depending on batch size
    indexes = [idx for batch in list(gb_sampler) for idx in batch]
    return val_loader, indexes


def get_test_dataloader(
        root="data/interim", augmentation="test", batch_size=8, size=512, workers=6):
    """
    Returns:
        test_loader (DataLoader)
        indexes (list): Order in which indexes are sampled from dataset
    """
    aug = get_aug(augmentation, size=size)

    test_dataset = TestDataset(root=root, transform=aug, size=size)

    gb_sampler = GroupedBatchSampler(
        sampler=torch.utils.data.sampler.SequentialSampler(test_dataset),
        group_ids=test_dataset.group_ids,
        batch_size=batch_size,
        drop_uneven=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=workers,
        pin_memory=True,
        batch_sampler=gb_sampler,
    )

    test_loader = ToCudaLoader(test_loader)
    logger.info(f"Test size: {len(test_dataset)}")
    # GB sampler changes the order of images depending on batch size
    indexes = [idx for batch in list(gb_sampler) for idx in batch]
    return test_loader, indexes


class ClassificationDataset(torch.utils.data.Dataset):
    """
    This dataset implemets tecnique used for ImageNet training and in some image retrieval papers.
    Images are not resized to fixed aspect ratio, but grouped into fixed number of bins and
    resized with keeping initial ratio as close, as possible.

    Args:
        size: What type of resized images to take

    Reference:
        https://arxiv.org/pdf/2003.11211.pdf - select limited number of aspect ratios
        https://github.com/cybertronai/imagenet18/master/training/dataloader.py#L231
    """

    _aspect_ratios = np.array([2, 16 / 9, 3 / 2, 4 / 3, 5 / 4, 1, 4 / 5, 3 / 4, 2 / 3, 9 / 16, 1 / 2])

    def __init__(self, root="data/interim", transform=None, train=True, size=512):
        df = pd.read_csv(os.path.join(root, "train_val.csv"))
        df = df[df["is_train"].astype(np.bool) == train]
        self.filenames = [
            os.path.join(root, f"train_data_{size}", path) for path in df["file_path"].values.tolist()]

        # Сheck that all images exist
        assert map(lambda x: pathlib.Path(x).exists(), self.filenames), "Found missing images!"

        self.targets = df["label"].values.tolist()
        self.ar = df["aspect_ratio"].values

        # For each aspect ration in `ar` find closest value from the
        # `_aspect_ratios` and return it's index (group)
        self.group_ids = np.argmin(
            np.abs(self._aspect_ratios.reshape(1, -1) - self.ar.reshape(-1, 1)),
            axis=1
        )

        self.sizes = []
        for group in self.group_ids:
            ar = self._aspect_ratios[group]
            # Resize image to have `size` shape on smaller side and be devidable by 16 on another
            if ar <= 1:
                H = size
                W = int(size / ar) // 8 * 8
            else:
                W = size
                H = int(size * ar) // 8 * 8
            self.sizes.append((H, W))

        logger.info(f"Using sizes {set(self.sizes)} for {'train'if train else 'validation'}")
        self.transform = albu.Compose([albu_pt.ToTensorV2()]) if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        try: 
            # logger.info(f"Index {index}")
            image = cv2.imread(self.filenames[index], cv2.IMREAD_COLOR)
            # logger.info(f"{index} Size before reshape {image.shape}")
            image = cv2.resize(image, self.sizes[index])
            # logger.info(f"{index} Size after reshape {image.shape}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)["image"]

            target = self.targets[index]
        except:
            logger.error(f"Not found: {self.filenames[index]}")

        return image, target

    def __len__(self):
        return len(self.filenames)


class TestDataset(torch.utils.data.Dataset):
    """
    Args:
        root: Path to preprocessed dataset
        transform: Albumentations transform to be aplied for each image
        size: What type of resized images to take
        test_type: One of {'A', 'B'} for different test sets
    """

    _aspect_ratios = np.array([2, 16 / 9, 3 / 2, 4 / 3, 5 / 4, 1, 4 / 5, 3 / 4, 2 / 3, 9 / 16, 1 / 2])

    def __init__(self, root="data/interim", transform=None, size=512, test_type="B"):
        df = pd.read_csv(os.path.join(root, "test_B.csv"))

        self.filenames = [
            os.path.join(root, f"test_data_B_{size}", path) for path in df["file_path"].values.tolist()]

        # Сheck that all images exist
        assert map(lambda x: pathlib.Path(x).exists(), self.filenames), "Found missing images!"

        self.ar = df["aspect_ratio"].values
        self.size = size

        # For each aspect ration in `ar` find closest value from the
        # `_aspect_ratios` and return it's index (group)
        self.group_ids = np.argmin(
            np.abs(self._aspect_ratios.reshape(1, -1) - self.ar.reshape(-1, 1)),
            axis=1
        )

        self.sizes = []
        for group in self.group_ids:
            ar = self._aspect_ratios[group]
            # Resize image to have `size` shape on smaller side and be devidable by 8 on another
            if ar <= 1:
                H = size
                W = int(size / ar) // 8 * 8
            else:
                W = size
                H = int(size / ar) // 8 * 8
            self.sizes.append((H, W))

        self.transform = albu.Compose([albu_pt.ToTensorV2()]) if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        try:

            image = cv2.imread(self.filenames[index], cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.sizes[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)["image"]
        except:
            logger.error(f"Not found: {self.filenames[index]}")

        return image

    def __len__(self):
        return len(self.filenames)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class GroupedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, torch.utils.data.Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
