"""
Most functions in this file are taken from:
https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
"""
import sys
import torch
import pytorch_tools as pt
from loguru import logger

# sys.path.append("/home/zakirov/repoz/GPU-Efficient-Networks/")
sys.path.append("/raid/dzakirov/code/GPU-Efficient-Networks/")
import GENet  # noqa


class Model(torch.nn.Module):
    """Model for query searches
    Returns:
        x: L2 NORMALIZED features, ready to be used as image embeddings
    """
    def __init__(self, arch='resnet50', embedding_size=512, pooling="avg", model_params={},):
        # All models return raw logits
        super().__init__()
        if arch == 'genet_normal':
            self.model = GENet.genet_normal(
                pretrained=True,
                num_classes=1000,
                root="/home/zakirov/repoz/GPU-Efficient-Networks/GENet_params"
                # root="/raid/dzakirov/code/GPU-Efficient-Networks/GENet_params",
            )
            self.model.fc_linear = torch.nn.Linear(in_features=2560, out_features=embedding_size)
            torch.nn.init.xavier_uniform_(self.model.fc_linear.weight)
            self.model.adptive_avg_pool.netblock = POOLING_FROM_NAME[pooling]
            self.forward = self.forward_genet
        else:
            self.model = pt.models.__dict__[arch](num_classes=embedding_size, **model_params)
            self.pooling = POOLING_FROM_NAME[pooling]

    def forward(self, x):
        """
        Args:
            x: Input images
        Returns:
            Raw model logits
        """
        x = self.model.features(x)
        x = self.pooling(x)
        # Normalize before FC, so that it works as learned PCA
        x = torch.nn.functional.normalize(x, p=2)
        x = torch.flatten(x, 1)
        x = self.model.dropout(x)
        x = self.model.last_linear(x)

        # Normalize features
        x = torch.nn.functional.normalize(x, p=2)
        return x

    def forward_genet(self, x):
        x = self.model(x)
        # Normalize features
        x = torch.nn.functional.normalize(x, p=2)
        return x

# --------------------------------------
# Pooling layers
# --------------------------------------


class GeM(torch.nn.Module):
    """
    Generalized Mean Pooling
    Args:
        p: Power parameter

    Reference:
        https://arxiv.org/abs/1711.02512
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        # self.p = torch.nn.Parameter(torch.ones(1) * p)
        # Parameter is fixed! Not learned
        self.p = torch.tensor(p)
        self.eps = eps

    def forward(self, x):
        # logger.info(f"input {x.shape}, {x.dtype}. P {self.p.shape}, {self.p.dtype}")
        return torch.nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p.to(x)), 1).pow(1. / self.p.to(x))


POOLING_FROM_NAME = {
    "max": torch.nn.AdaptiveMaxPool2d(1),
    "avg": torch.nn.AdaptiveAvgPool2d(1),
    "gem": GeM(p=3.0, eps=1e-6),
}
