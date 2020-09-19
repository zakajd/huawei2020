import torch


class ToCudaLoader:
    def __init__(self, loader):
        self.loader = loader
        self.batch_size = loader.batch_size

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, (tuple, list)):
                yield [i.cuda(non_blocking=True) for i in batch]
            else:
                yield batch.cuda(non_blocking=True)

    def __len__(self):
        return len(self.loader)


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

LABEL_MAP = {
    2979: 1475,
    1528: 1475,
    2113: 2672,
    1695: 2860,
    188: 1011,
    1697: 2714,
    499: 1937,
    1009: 1658,
    1534: 2244,
    1077: 1767,
    74: 1635,
    381: 2139,
    942: 2812,
    1082: 770,
    1205: 770,
    822: 2170,
    414: 939,
    30: 1390,
    841: 1663,
    756: 1870,
    968: 1807,
}
