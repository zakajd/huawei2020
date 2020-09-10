import torch
import pytorch_tools as pt

class Model(torch.nn.Module):
    """Model for query searches"""
    def __init__(self, arch="resnet50", model_params={}, embedding_size=512}):
        # All models return raw logits
        self.model = pt.models.__dict__[arch](num_classes=embedding_size, model_params)

        # nn.AdaptiveAvgPool2d(1), FastGlobalAvgPool2d()

    def forward(self, x):
        """
        Args:
            x: Input images
        Returns:
            Raw model logits
        """
        x = self.model(x)
        return x

