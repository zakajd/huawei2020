import torch
from loguru import logger


class AngularPenaltySMLoss(torch.nn.Module):
    """PyTorch implementation of
        1. Additive Angular Margin Loss / ArcFace
        2. Large Margin Cosine Loss / CosFase
        3. SphereFace

    Args:
        in_features: Size of model discriptor
        out_features: Number of classes
        loss_type: One of {'arcface', 'sphereface', 'cosface'}
        s: Input features norm
        m1: Margin value for ArcFace
        m2: Margin value for CosFase

    Reference:
        1. CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
            https://arxiv.org/pdf/1801.07698.pdf
        2. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
            https://arxiv.org/pdf/1801.09414.pdf
        3. SphereFace:
            https://arxiv.org/abs/1704.08063

    Code: github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
    """
    _types = ['arcface', 'sphereface', 'cosface']
    # 'name': (s, m)
    _default_values = {
        'arcface': (64.0, 0.5),
        'sphereface': (64.0, 1.35),
        'cosface': (30.0, 0.4),
    }

    def __init__(self, in_features=512, out_features=3097, loss_type='arcface', s=None, m=None):
        super().__init__()
        assert loss_type in self._types, \
            f"Loss type must be in ['arcface', 'sphereface', 'cosface'], got {loss_type}"

        self.s, self.m = self._default_values[loss_type]
        # Overwright default values
        self.s = self.s if not s else s
        self.m = self.m if not m else m

        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

        self.loss_type = loss_type

        # Constant for numerical stability
        self.eps = 1e-7

    def forward(self, features, y_true):
        """
        Args:
            features: Raw logits from the model
            y_true: Class labels, not one-hot encoded
        """

        # Normalize
        features = torch.nn.functional.normalize(features, p=2)
        for W in self.fc.parameters():
            W = torch.nn.functional.normalize(W, p=2)

        # logger.info(f"Loss input shapes {features.shape}, {y_true.shape}")
        # Black magic of matrix calculus
        wf = self.fc(features)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[y_true]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[y_true]), -1. + self.eps, 1 - self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[y_true]), -1. + self.eps, 1 - self.eps)))
        else:
            raise ValueError('Unknown loss type')

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(y_true)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
