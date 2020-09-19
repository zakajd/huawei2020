"""
1. Split all validation data on query / gallery and compute evaluation metrics on that set
2. Take test data and predict on it
"""
import os
import time
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# import pytorch_tools as pt
from loguru import logger

from src.datasets import get_val_dataloader, get_test_dataloader
from src.callbacks import cmc_score_count, map_at_k
from src.models import Model


def query_expansion(query_embeddings, gallery_embeddings, topk=10, alpha=3):
    """
    Implementation of Data Base Augmentation / Alpha Query Expansion
    Args:
        query_embeddings (Tensor): Shape (N, embedding_dim)
        gallery_embeddings (Tensor): Shape (M, embedding_dim)
        topk (int): How many neighbours to use
        alpha (int): Power for neighbours reweighting
        include_self (bool): Flag to include original embed in new one
    """
    # Matrix of pairwise cosin distances
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    # Nearest neighbours
    topk_vals, topk_ind = distances.neg().topk(topk, dim=1)
    # Get weight
    if alpha is None:
        # weight = torch.div(topk - torch.arange(topk), float(topk))[None, :, None]  # N x TOPK x 1
        weight = torch.div(topk - torch.arange(topk) - 1, float(topk))[None, :, None]  # N x TOPK x 1
    else:
        cosine_dist = (2 - topk_vals.neg()) * 0.5  # cos = ((2 - l2_distances) / 2)
        weight = (cosine_dist ** alpha)[..., None]  # N x TOPK -> N x TOPK x 1

    new_embedding = gallery_embeddings[topk_ind] * weight  # N x TOPK x EMBED_SIZE * N x TOPK x 1
    new_embedding = new_embedding.sum(dim=1) + query_embeddings
    # new_embedding = (new_embedding.mean(dim=1) + query_embeddings) / 2
    new_embedding = torch.nn.functional.normalize(new_embedding, p=2, dim=1)
    # new_embedding = torch.nn.functional.normalize(new_embedding, p=2, dim=1)
    return new_embedding


@torch.no_grad()
def predict_from_loader(model, loader):
    """Compute embeddings for each image in loader.
    Args:
        model: Model to use for prediction
        loader: torch.nn.DataLoader
    Returns:
        embeddings: Tensor of shape `len(dataset) x `embedding_size`
        labels: Class labels for each image
    """
    # TODO: Add TTA here with rescaling?
    model.eval()
    embeddings = []
    for batch in tqdm(loader):
        if isinstance(batch, list):
            images, _ = batch
        else:
            images = batch
        embeddings.extend(model(images).cpu().numpy())
    return np.stack(embeddings)


def test(hparams):
    # Check that folder exists
    assert hparams.config_path.exists()

    # Read config
    with open(hparams.config_path / "config.yaml", "r") as file:
        model_configs = yaml.load(file)
    model_configs.update(vars(hparams))
    hparams = argparse.Namespace(**model_configs)

    # Get model
    model = Model(
        arch=hparams.arch,
        model_params=hparams.model_params,
        embedding_size=hparams.embedding_size,
        pooling=hparams.pooling).cuda()
    # logger.info(model)

    # Init
    checkpoint = torch.load(hparams.config_path / f"model.chpn")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # -------------- Get embeddings for val and test data --------------
    if hparams.extract_embeddings:
        if hparams.validation:
            print(f"Using size {hparams.val_size}")
            loader, indexes = get_val_dataloader(
                root=hparams.root,
                augmentation="val",
                batch_size=hparams.batch_size,
                size=hparams.val_size,
                workers=hparams.workers,
            )

            # Load validation query / gallery split and sort it according to indexes from sampler
            df_val = pd.read_csv(os.path.join(hparams.root, "train_val.csv"))
            df_val = df_val[df_val["is_train"].astype(np.bool) == False].iloc[indexes]

            val_embeddings = predict_from_loader(model, loader)

            # Hack to save torch.Tensor into pd.DataFrame
            df_val["embeddings"] = list(map(lambda r: np.array(r).tolist(), val_embeddings))
            # Save results into folder with logs
            df_val.to_csv(hparams.config_path / "train_val.csv", index=None)
            del val_embeddings
            logger.info("Finished extracting validation embeddings")

        if hparams.test:
            loader, indexes = get_test_dataloader(
                root=hparams.root,
                augmentation="test",
                batch_size=hparams.batch_size,
                size=hparams.val_size,
                workers=hparams.workers,
            )
            # Load test DF and sort it according to indexes from sampler
            df_test = pd.read_csv(os.path.join(hparams.root, "test_A.csv")).iloc[indexes]
            test_embeddings = predict_from_loader(model, loader)

            # Hack to save torch.Tensor into pd.DataFrame
            df_test["embeddings"] = list(map(lambda r: np.array(r).tolist(), test_embeddings))

            # Save results into folder with logs
            df_test.to_csv(hparams.config_path / "test_A.csv", index=None)
            del test_embeddings
            logger.info("Finished extracting test embeddings")

    # -------------- Test model on validation dataset --------------
    if hparams.validation:
        # Read DF
        df_val = pd.read_csv(hparams.config_path / "train_val.csv")
        val_embeddings = torch.tensor(list(map(eval, df_val["embeddings"].values)))
        query_mask = df_val["is_query"].values.astype(np.bool)
        val_labels = df_val["label"].values

        # Shape (n_embeddings, embedding_dim)
        query_embeddings, gallery_embeddings = val_embeddings[query_mask], val_embeddings[~query_mask]
        query_labels, gallery_labels = val_labels[query_mask], val_labels[~query_mask]
        logger.info(f"Validation query size - {len(query_embeddings)}, gallery size - {len(gallery_embeddings)}")
        del val_embeddings

        if hparams.dba:
            gallery_embeddings = query_expansion(gallery_embeddings, gallery_embeddings, topk=10, alpha=None)

        if hparams.aqe:
            query_embeddings = query_expansion(query_embeddings, gallery_embeddings, topk=3, alpha=3)

        # Shape (query_size x gallery_size)
        conformity_matrix = torch.tensor(query_labels.reshape(-1, 1) == gallery_labels)

        # Matrix of pairwise cosin distances
        distances = torch.cdist(query_embeddings, gallery_embeddings)

        acc1 = cmc_score_count(distances, conformity_matrix, topk=1)
        map10 = map_at_k(distances, conformity_matrix, topk=10)
        mapR = map_at_k(distances, conformity_matrix, topk=None)

        logger.info(
            f"Val: Acc@1 {acc1:0.5f}, mAP@10 {map10:0.5f}, Target {0.5 * acc1 + 0.5 * map10:0.5f}, mAP@R {mapR:0.5f}")

    # -------------- Predict on  test dataset  --------------
    if hparams.test:
        df_test = pd.read_csv(hparams.config_path / "test_A.csv")
        test_embeddings = torch.tensor(list(map(eval, df_test["embeddings"].values)))
        query_mask = df_test["is_query"].values.astype(np.bool)
        query_files, gallery_files = df_test["file_path"].values[query_mask], df_test["file_path"].values[~query_mask]

        # Shape (n_embeddings, embedding_dim)
        query_embeddings, gallery_embeddings = test_embeddings[query_mask], test_embeddings[~query_mask]
        query_files, gallery_files = df_test["file_path"].values[query_mask], df_test["file_path"].values[~query_mask]
        logger.info(f"Test query size - {len(query_embeddings)}, gallery size - {len(gallery_embeddings)}")
        del test_embeddings

        if hparams.dba:
            gallery_embeddings = query_expansion(gallery_embeddings, gallery_embeddings, topk=10, alpha=None)

        if hparams.aqe:
            query_embeddings = query_expansion(query_embeddings, gallery_embeddings, topk=3, alpha=3)

        # Matrix of pairwise cosin distances
        distances = torch.cdist(query_embeddings, gallery_embeddings)
        perm_matrix = torch.argsort(distances)

        logger.info(f"Creating submission{'_dba' if hparams.dba else ''}{'_aqe' if hparams.aqe else ''}_{hparams.val_size}.csv")
        data = {
            "image_id": [],
            "gallery_img_list": []
        }

        for idx in tqdm(range(len(query_files))):
            query_file = query_files[idx].split("/")[1]
            predictions = gallery_files[perm_matrix[:, : 10][idx]]
            predictions = [p.split("/")[1] for p in predictions]
            data["image_id"].append(query_file)
            data["gallery_img_list"].append(predictions)

        df = pd.DataFrame(data=data)
        df["gallery_img_list"] = df["gallery_img_list"].apply(lambda x: '{{{}}}'.format(",".join(x))).astype(str)
        lines = [f"{x},{y}" for x, y in zip(data["image_id"], df["gallery_img_list"])]
        with open(hparams.config_path \
            / f"submission{'_dba' if hparams.dba else ''}{'_aqe' if hparams.aqe else ''}_{hparams.val_size}.csv", "w") as f:
            for line in lines:
                f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huawei challenge",
    )
    # General
    parser.add_argument(
        "--config_path", "-c", type=Path, help="Path to folder with model config and checkpoint")
    parser.add_argument(
        "--output_path", type=str, default="data/processed", help="Path to save scores")
    parser.add_argument(
        "--extract_embeddings", action="store_true", help="Extract and save embeddings for each image into DataFrame")
    parser.add_argument(
        "--validation", action="store_true", help="Flag to make prediction for validation and compute final score")
    parser.add_argument(
        "--test", action="store_true", help="Flag to predict test")

    # Options
    parser.add_argument(
        "--val_size", default=None, type=int, help="Size for validatiion")
    parser.add_argument(
        "--dba", default=False, action='store_true', help="Use DBA")
    parser.add_argument(
        "--aqe", default=False, action='store_true', help="Use alpha query expansion")
    parser.add_argument(
        "--tta", default=False, action='store_true', help="Use TTA")

    hparams = parser.parse_args()
    print(f"Parameters used for test: {hparams}")
    start_time = time.time()
    test(hparams)
    print(f"Finished test. Took: {(time.time() - start_time) / 60:.02f}m")
