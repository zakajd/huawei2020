import torch
import sklearn
import pytorch_tools as pt

## Add TB callback that shows cluster images, ROC AUC curves and ...
class QueryAccuracy():


class RocAucMeter(torch.nn.Module):
    name = "AUC"
    metric_func = sklearn.metrics.roc_auc_score

    def __init__(self):
        super().__init__()
        self.name = self.__class__.name

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float().sigmoid().numpy()
        y_true = y_true.cpu().numpy()
        score = self.__class__.metric_func(y_true.flatten(), y_pred.flatten())
        return score


class APScoreMeter(RocAucMeter):
    name = "AP"
    metric_func = sklearn.metrics.average_precision_score