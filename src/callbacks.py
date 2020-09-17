import torch
import pytorch_tools as pt
from torch.cuda import amp
from sklearn.metrics import roc_auc_score, average_precision_score
# from loguru import logger
# import numpy as np


def cmc_score_count(distances: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 1) -> float:
    """
    From https://github.com/catalyst-team/catalyst
    Function to count CMC from distance matrix and conformity matrix.

    Args:
        distances: distance matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    perm_matrix = torch.argsort(distances)
    position_matrix = torch.argsort(perm_matrix)
    conformity_matrix = conformity_matrix.type(torch.bool)

    position_matrix[~conformity_matrix] = (
        topk + 1
    )  # value large enough not to be counted

    closest = position_matrix.min(dim=1)[0]
    k_mask = (closest < topk).type(torch.float)
    return k_mask.mean().item()


def rank_map_score(distances: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 1) -> float:
    """
    Function to count mean Average Precision from distance matrix and conformity matrix.

    Args:
        distances: distance matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for AP counting

    Returns:
        mAP@k score
    """
    perm_matrix = torch.argsort(distances)
    conformity_matrix = conformity_matrix.type(torch.double)

    # Sort matrix and take first K columns
    conformity_matrix = torch.gather(conformity_matrix, 1, perm_matrix)[:, :topk]

    precision = torch.cumsum(conformity_matrix, dim=-1) * conformity_matrix / torch.arange(start=1, end=topk + 1)
    average_precision = precision.sum(dim=-1) / conformity_matrix.sum(dim=-1)

    # If no match found at first k elements, AP is 0
    average_precision[average_precision.isnan()] = 0
    return average_precision.mean().item()


def cmc_score(
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        conformity_matrix: torch.Tensor,
        topk: int = 1) -> float:
    """
    Function to count CMC score from query and gallery embeddings.
    From https://github.com/catalyst-team/catalyst/issues

    Args:
        query_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in querry
        gallery_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in gallery
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    return cmc_score_count(distances, conformity_matrix, topk)


class ContestMetricsCallback(pt.fit_wrapper.callbacks.Callback):
    """
    Compute Accuracy@1 and mAP@10 for query search results after running full loader
    """

    def __init__(self):
        super().__init__()

        self.metric_names = ["Acc_@1", "mAP@10", ".5Acc+.5mAP", "CMC@10"]
        self.target = None
        self.output = None

        self.query_mask = None
        self.query_created = False

    def on_begin(self):
        for name in self.metric_names:
            self.state.metric_meters[name] = pt.utils.misc.AverageMeter(name=name)

    def on_loader_begin(self):
        self.target = []
        self.output = []

    def on_batch_end(self):
        if not self.state.is_train:
            _, target = self.state.input
            with amp.autocast(self.state.use_fp16):
                self.target.append(target.cpu().detach())
            self.output.append(self.state.output.cpu().detach())

    @torch.no_grad()
    def on_loader_end(self):
        if not self.state.is_train:
            target = torch.cat(self.target)
            output = torch.cat(self.output)

            if not self.query_created:
                # Create random bitmask. 20% of the data usd as a query
                self.query_mask = torch.FloatTensor(len(target)).uniform_() > 0.8
                self.query_created = True

            # Shape (n_embeddings, embedding_dim).  No TTA
            query_embeddings = output[self.query_mask]
            gallery_embeddings = output[~self.query_mask]

            query_labels = target[self.query_mask]
            gallery_labels = target[~self.query_mask]
            # logger.info(f"Query: {query_embeddings.shape} {query_embeddings.type()}, query labels: {query_labels.shape} {query_labels.type()}")
            # logger.info(f"gallery: {gallery_embeddings.shape} {gallery_embeddings.type()}, query labels: {gallery_labels.shape} {gallery_labels.type()}")
            # logger.info(f"Use fp16: {self.state.use_fp16}")
            # Shape (query_size x gallery_size)
            conformity_matrix = query_labels.reshape(-1, 1) == gallery_labels

            # Matrix of pairwise cosin distances
            distances = torch.cdist(query_embeddings, gallery_embeddings)

            acc1 = cmc_score_count(distances, conformity_matrix, topk=1)
            cmc10 = cmc_score_count(distances, conformity_matrix, topk=10)
            map10 = rank_map_score(distances, conformity_matrix, topk=10)
            target_metric = 0.5 * acc1 + 0.5 * map10

            self.metrics = [acc1, map10, target_metric, cmc10]

            for metric, name in zip(self.metrics, self.metric_names):
                self.state.metric_meters[name].update(metric)

# Add TB callback that shows cluster images, ROC AUC curves and ...


class RocAucMeter(torch.nn.Module):
    name = "AUC"
    metric_func = roc_auc_score

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
    metric_func = average_precision_score
