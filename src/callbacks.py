import torch
import pytorch_tools as pt
from torch.cuda import amp
from sklearn.metrics import roc_auc_score, average_precision_score
# from loguru import logger
# import numpy as np
from src.models import kNN


def accuracy_at_1(perm_matrix: torch.Tensor, conformity_matrix: torch.Tensor) -> float:
    """
    From https://github.com/catalyst-team/catalyst
    Function to count Accuracy from perm_matrix and conformity matrix.

    Args:
        perm_matrix: Permutation matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise. Shape (n_embeddings_x, n_embeddings_y)

    Returns:
        Acc@1
    """
    conformity_matrix = torch.gather(conformity_matrix, 1, perm_matrix)
    conformity_matrix = conformity_matrix.type(torch.float)
    return conformity_matrix[:, 0].mean().item()


def cmc_score_count(perm_matrix: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 1) -> float:
    """
    From https://github.com/catalyst-team/catalyst
    Function to count CMC from perm_matrix and conformity matrix.

    Args:
        perm_matrix: Permutation matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise. Shape (n_embeddings_x, n_embeddings_y)
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    position_matrix = torch.argsort(perm_matrix)
    conformity_matrix = torch.gather(conformity_matrix, 1, perm_matrix)
    conformity_matrix = conformity_matrix.type(torch.bool)

    position_matrix[~conformity_matrix] = (
        topk + 1
    )  # value large enough not to be counted

    closest = position_matrix.min(dim=1)[0]
    k_mask = (closest < topk).type(torch.float)
    return k_mask.mean().item()


def map_at_k(perm_matrix: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 10) -> float:
    """Compute mean Average Precision (mAP@k) from distance matrix and conformity matrix.
    If topk parameter is None, returns mAP@R, see [1] for details.

    Args:
        perm_matrix: Permutation matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for AP counting.

    Returns:
        mAP@k score

    Reference:
        [1] A Metric Learning Reality Check
            https://arxiv.org/pdf/2003.08505.pdf
    """
    conformity_matrix = conformity_matrix.type(torch.double)

    if topk is None:  # Compute mAP@R
        # Total total number of references that are the same class as the query.
        R = conformity_matrix.sum(dim=1)
        R_max = int(R.max())
    else:  # Compute mAP@k
        R = torch.ones(perm_matrix.size(0)) * topk
        R_max = topk
    R_max = min(R_max, perm_matrix.shape[1])

    R_mask = torch.cumsum(torch.ones(
        (perm_matrix.size(0), R_max)), dim=1) <= R.reshape(-1, 1)

    # Sort matrix
    conformity_matrix = torch.gather(conformity_matrix, 1, perm_matrix)[
        :, :R_max] * R_mask
    precision = torch.cumsum(conformity_matrix, dim=-1) * conformity_matrix \
        / torch.arange(start=1, end=R_max + 1)
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
    # Slow for big number of queries! Use FAISS instead
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    perm_matrix = torch.argsort(distances)
    return cmc_score_count(perm_matrix, conformity_matrix, topk)


class ContestMetricsCallback(pt.fit_wrapper.callbacks.Callback):
    """
    Compute Accuracy@1 and mAP@10 for query search results after running full loader
    Args:
        is_query: Binary mask to identify query embeddings
    """

    def __init__(self, is_query=None):
        super().__init__()

        self.metric_names = ["Acc@1", "mAP@10", "target", "mAP@R"]
        self.target = None
        self.output = None
        assert is_query is not None, "Empty query mask"
        self.is_query = is_query

    def on_begin(self):
        for name in self.metric_names:
            self.state.metric_meters[name] = pt.utils.misc.AverageMeter(
                name=name)

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

            # Shape (n_embeddings, embedding_dim).  No TTA
            assert len(self.is_query) == len(output), \
                f"Length of embeddings and query mask mismatch! {len(self.is_query)}, {len(output)}"
            query_embeddings = output[self.is_query]
            gallery_embeddings = output[~self.is_query]

            query_labels = target[self.is_query]
            gallery_labels = target[~self.is_query]
            # logger.info(f"Query: {query_embeddings.shape} {query_embeddings.type()}, query labels: {query_labels.shape} {query_labels.type()}")
            # logger.info(f"gallery: {gallery_embeddings.shape} {gallery_embeddings.type()}, query labels: {gallery_labels.shape} {gallery_labels.type()}")
            # logger.info(f"Use fp16: {self.state.use_fp16}")

            # Shape (query_size x gallery_size)
            conformity_matrix = query_labels.reshape(-1, 1) == gallery_labels

            # Init kNN and find neighbours
            knn = kNN(embeddings=gallery_embeddings, distance='cosine')
            # Get distance matrix of shape (query_length x 1000)
            distances, perm_matrix = knn.search(query_embeddings, topk=1000)

            # Compute validation metrics
            acc1 = accuracy_at_1(perm_matrix, conformity_matrix)
            map10 = map_at_k(perm_matrix, conformity_matrix, topk=10)
            target_metric = 0.5 * acc1 + 0.5 * map10
            mapR = map_at_k(perm_matrix, conformity_matrix, topk=None)

            self.metrics = [acc1, map10, target_metric, mapR]

            for metric, name in zip(self.metrics, self.metric_names):
                self.state.metric_meters[name].update(metric)


class CheckpointSaver(pt.fit_wrapper.callbacks.CheckpointSaver):
    """
    Save best model every epoch based on loss
    Args:
        save_dir (str): path to folder where to save the model
        save_name (str): name of the saved model. can additionally
            add epoch and metric to model save name
        monitor (str): quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.
        mode (str): one of "min" of "max". Whether to decide to save based
            on minimizing or maximizing loss
        include_optimizer (bool): if True would also save `optimizers` state_dict.
            This increases checkpoint size 2x times.
        verbose (bool): If `True` reports each time new best is found
    """

    def _save_checkpoint(self, path):
        save_dict = {
            "epoch": self.state.epoch,
            "state_dict": self.state.model.state_dict(),
            "loss": self.state.criterion.state_dict()
        }
        torch.save(save_dict, path)
