# This code is based on https://medium.com/pytorch/metric-learning-with-catalyst-8c8337dfab1a
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import pytorch_tools as pt
from catalyst import data, dl, utils
from catalyst.contrib import datasets, models, nn
import catalyst.data.cv.transforms.torch as t

from src.dataset_catalyst import ClassificationDataset, QueryGalleryDataset

import albumentations as albu
import albumentations.pytorch as albu_pt

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
SIZE = 512
transform = albu.Compose(
    [
        albu.Resize(height=SIZE, width=SIZE, p=1),
        albu.RandomSizedCrop(min_max_height=(400, 400), height=SIZE, width=SIZE, p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Resize(height=SIZE, width=SIZE, p=1),
        albu.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        albu.Normalize(mean=MEAN, std=STD), 
        albu_pt.ToTensorV2(),
    ],
    p=1.0,
)

# 1. train and valid datasets
dataset_train = ClassificationDataset(root="data/raw", transform=transform)
sampler = data.BalanceBatchSampler(labels=dataset_train.get_labels(), p=3, k=10)
print(f"Using batch size: {sampler.batch_size}")
train_loader = DataLoader(dataset=dataset_train, sampler=sampler, batch_size=sampler.batch_size)

dataset_val = QueryGalleryDataset(root="data/raw", transform=transform, gallery_fraq=0.8)
val_loader = DataLoader(dataset=dataset_val, batch_size=64)

# 2. model and optimizer
model = pt.models.__dict__["resnet50"](num_classes=3097)
optimizer = Adam(model.parameters(), lr=0.001)

# 3. criterion with triplets sampling
sampler_inbatch = data.HardTripletsSampler(norm_required=False)
criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

# 4. training with catalyst Runner
callbacks = [
    dl.ControlFlowCallback(dl.CriterionCallback(), loaders="train"),
    dl.ControlFlowCallback(dl.CMCScoreCallback(topk_args=[1]), loaders="valid"),
    dl.PeriodicLoaderCallback(valid=5),
]

runner = dl.SupervisedRunner(device=utils.get_device())
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders={"train": train_loader, "valid": val_loader},
    minimize_metric=False,
    verbose=True,
    valid_loader="valid",
    num_epochs=15,
    main_metric="cmc01",
)

state_dict = model.state_dict()
save_dict = {"state_dict": state_dict}
torch.save(save_dict, "model1.ckpt")