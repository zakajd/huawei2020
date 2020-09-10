from typing import Any, Callable, Dict, List, Optional
import os
import cv2

import torch
import albumentations as albu
import albumentations.pytorch as albu_pt

from catalyst.data.dataset.metric_learning import (
    MetricLearningTrainDataset,
    QueryGalleryDataset,
)


class ClassificationDataset(MetricLearningTrainDataset, torch.utils.data.Dataset):
    def __init__(self, root="data/raw", transform=None):   
        # Read file with labels
        with open(os.path.join(root, "train_data/label.txt")) as f:
            data = f.readlines()

        images, labels = [], []
        for row in data:
            path, label = row.strip("\n").split(",")
            images.append(path)
            labels.append(int(label))

        self.data = images
        self.targets = labels

        self.root = root
        self.transform = albu.Compose([albu_pt.ToTensorV2()]) if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image = cv2.imread(os.path.join(self.root, "train_data", self.data[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.data)

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets


class QueryGalleryDataset(QueryGalleryDataset, torch.utils.data.Dataset):
    def __init__(self, root: str = "data/raw", transform: Optional[Callable] = None, gallery_fraq: Optional[float] = 0.2) -> None:
        """
        Args:
            root: root directory for storing dataset
            transform: transform
            gallery_fraq: gallery size
        """
        self.dataset = ClassificationDataset(root=root, transform=transform)


        self._gallery_size = int(gallery_fraq * len(self.dataset))
        self._query_size = len(self.dataset) - self._gallery_size

        # TODO: Change split query / not query
        self._is_query = torch.zeros(len(self.dataset)).type(torch.bool)
        self._is_query[: self._query_size] = True

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get item method for dataset
        Args:
            index: index of the object
        Returns:
            Dict with features, targets and is_query flag
        """
        image, label = self.dataset[index]
        return {
            "features": image,
            "targets": label,
            "is_query": self._is_query[index],
        }

    def __len__(self) -> int:
        """Length"""
        return len(self.dataset)

    @property
    def gallery_size(self) -> int:
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self) -> int:
        """Query Gallery dataset should have query_size property"""
        return self._query_size