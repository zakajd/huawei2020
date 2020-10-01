" diffusion module "

import os
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from tqdm.notebook import tqdm
from sklearn import preprocessing
from loguru import logger

import multiprocessing
import functools

trunc_ids = None
trunc_init = None
lap_alpha = None


def get_single_score(idx, truncation_size=5000):
    """Computes result for one image
    Args: 
        idx
        data (tuple): (idx,  trunc_ids[idx])

    """
    ids = trunc_ids[idx]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=50)
    return scores

trunc_ids, trunc_init, lap_alpha = None, None, None


class Diffusion:
    """Performse Diffusion
    Args:
        embeddings: Query and Gallery embedings concatenated
        distance: Distance metric, one of {'cosine', 'euclidean'}
        
    Reference:
        Efficient Image Retrieval via Decoupling Diffusion ...
        https://arxiv.org/pdf/1811.10907.pdf
        
    """
    def __init__(self, embeddings, distance='cosine', gamma=3):
        self.embeddings = embeddings
        self.N = len(embeddings)
        self.knn = kNN(self.embeddings, distance=distance)
        
        self.gamma = gamma

    def get_offline_results(self, truncation_size=1000, kd=50):
        """Compute diffusion results for each gallery embedding
        Args:
            
        """
        global trunc_ids, trunc_init, lap_alpha

        logger.info("Searching neighbours")
        distances, ids = self.knn.search(self.embeddings, truncation_size)
    
        # We measure cosine simmularity, but for diffusion metric should increase for not-simmilar images
        # thus we revert the metric. Later it's checked to be non-negative
        distances = 1 - distances
        trunc_ids = ids
        logger.info("Computing laplacian")
        lap_alpha = self.laplacian(distances[:, :kd], ids[:, :kd])
        trunc_init = np.zeros(truncation_size)
        trunc_init[0] = 1
        
        logger.info('[offline] 2) gallery-side diffusion')
        
        # Slow, imrove later with multiprocessing
        new_func = functools.partial(get_single_score, truncation_size=truncation_size)
        results = [new_func(i) for i in tqdm(range(self.N))]
        all_scores = np.concatenate(results)

        logger.info('[offline] 3) merge offline results')
        logger.info(f'{all_scores.shape}')
#         return all_scores
        
        rows = np.repeat(np.arange(self.N), truncation_size)
        offline = sparse.csr_matrix((all_scores, (rows, trunc_ids.reshape(-1))),
                                    shape=(self.N, self.N),
                                    dtype=np.float32)

        logger.info(f'{offline.shape}')
        return offline

    def laplacian(self, distances, ids, alpha=0.99):
        """Computaion of Laplacian alpha matrix
        Args:
            sims:
            ids: ???
            alpha: Parameter for Laplacian construction
    
        Returns:
            lap_alpha: Matrix of ...
        """
        # Shape (num x num) (self.N x self.N?)
        affinity = self.affinity(distances, ids)
        
        num = affinity.shape[0]
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
        stochastic = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    def affinity(self, distances, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            distances: Similarities of kNN
            ids: Indexes of kNN
        Returns:
            affinity: Affinity matrix
        """
#         num = self.N ??
        num = distances.shape[0]
        
        # Distance should be non-negative
        distances[distances < 0] = 0

        distances = distances ** self.gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(distances[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                     shape=(num, num), dtype=np.float32)
        return affinity

# query_embeddings, gallery_embeddings = ...
# kd - how many nearest neighbours for each image to search. Rename -> max_neighbours?
kd = 50
kq = 10
gamma = 3
truncation_size = 1000 # 5000? authors say it improves results for bigger datasets


# Search
len_query = query_embeddings.shape[0]

diffusion = Diffusion(
    embeddings = np.vstack([query_embeddings, gallery_embeddings]), 
    distance='cosine',
    gamma=3)

offline = diffusion.get_offline_results(truncation_size, kd=kd)

features = preprocessing.normalize(offline, norm="l2", axis=1)
scores = features[:len_query] @ features[len_query:].T
ranks = np.argsort(-scores.todense())

# Compute validation metrics
ranks = torch.tensor(ranks)
acc1 = accuracy_at_1(ranks, conformity_matrix)
map10 = map_at_k(ranks, conformity_matrix, topk=10)
mapR = map_at_k(ranks, conformity_matrix, topk=None)

logger.info(
    f"Val + Train: Acc@1 {acc1:0.5f}, mAP@10 {map10:0.5f}, Target {0.5 * acc1 + 0.5 * map10:0.5f}, mAP@R {mapR:0.5f}")