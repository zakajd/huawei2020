"""
1. Split all validation data on query / gallery and compute evaluation metrics on that set
2. Take test data and predict on it
"""
import os
import cv2
import time
import yaml
import argparse
import pathlib

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_tools as pt
from loguru import logger

from src.datasets import get_val_dataloader, get_test_dataloader
from src.callbacks import cmc_score_count, rank_map_score
from src.models import Model


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

    embeddings = []
    labels = []
    for batch in tqdm(loader):
        if isinstance(batch, list):
            images, _ = batch
        else:
            images = batch
        embeddings.extend(model(images).cpu().numpy())
    return np.stack(embeddings)


def test(hparams):
    hparams.config_path = pathlib.Path(hparams.config_path)
    # Check that folder exists
    assert hparams.config_path.exists()

    # Read config
    with open(hparams.config_path / "config.yaml", "r") as file:
        model_configs = yaml.load(file)
    model_configs.update(vars(hparams))
    hparams = argparse.Namespace(**model_configs)

    # Get model
    model = Model(arch=hparams.arch, model_params=hparams.model_params, embedding_size=hparams.embedding_size).cuda()
    # logger.info(model)

    # Init
    checkpoint = torch.load(hparams.config_path / f"model.chpn")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # -------------- Get embeddings for val and test data --------------
    if hparams.extract_embeddings:
        # Val
        loader, _ = get_val_dataloader(
            root=hparams.root,
            augmentation="val",
            batch_size=hparams.batch_size,
            size=hparams.size,
            workers=hparams.workers,
            val_pct=1.0, # !!! 
        )

        val_embeddings = predict_from_loader(model, loader)
        df_val = pd.read_csv(os.path.join(hparams.root, "train_val.csv"))
        # TODO: Check that embeddings match images
        # Hack to save torch.Tensor into pd.DataFrame
        df_val["embeddings"]  = list(map(lambda r: np.array(r).tolist(), val_embeddings))
        # Save results into folder with logs
        df_val.to_csv(hparams.config_path / "train_val.csv", index=None)
        del val_embeddings
        logger.info("Finished extracting validation embeddings")

        # Test
        loader, _ = get_test_dataloader(
            root=hparams.root,
            augmentation="test",
            batch_size=hparams.batch_size,
            size=hparams.size,
            workers=hparams.workers,
        )
        test_embeddings = predict_from_loader(model, loader)
        df_test = pd.read_csv(os.path.join(hparams.root, "test_A.csv"))
        # TODO: Check that embeddings match images
        # Hack to save torch.Tensor into pd.DataFrame
        df_test["embeddings"]  = list(map(lambda r: np.array(r).tolist(), test_embeddings))
        # Save results into folder with logs
        df_test.to_csv(hparams.config_path / "test_A.csv", index=None)
        del test_embeddings
        logger.info("Finished extracting test embeddings")

    # -------------- Test model on validation dataset --------------
    if hparams.test_validation:
        # Read DF
        df_val = pd.read_csv(hparams.config_path / "train_val.csv")
        val_embeddings = torch.tensor(list(map(eval, df_val["embeddings"].values)))
        query_mask = df["is_query"].values

        # Shape (n_embeddings, embedding_dim)
        query_embeddings, gallery_embeddings = val_embeddings[query_mask], val_embeddings[~query_mask]
        query_labels, gallery_labels = val_labels[query_mask], val_labels[~query_mask]
        logger.info(f"Validation query size - {len(query_labels)}, gallery size - {len(gallery_labels)}")

        # Shape (query_size x gallery_size)
        conformity_matrix = query_labels.reshape(-1, 1) == gallery_labels

        # Matrix of pairwise cosin distances
        distances = torch.cdist(query_embeddings, gallery_embeddings)

        acc1 = cmc_score_count(distances, conformity_matrix, topk=1)
        cmc10 = cmc_score_count(distances, conformity_matrix, topk=10)
        map10 = rank_map_score(distances, conformity_matrix, topk=10)
        target_metric = 0.5 * acc1 + 0.5 * map10
        logger.info(
            f"Val: Acc@1 {acc1:0.5f}, CMC@10 {cmc10:0.5f}, mAP@10 {map10:0.5f}, target {target_metric:0.5f}")

    # -------------- Predict on  test dataset  --------------
    # if hparams.test:
    #     loader, is_query = get_test_dataloader(
    #         root=hparams.root,
    #         batch_size=hparams.batch_size,
    #         size=hparams.size,
    #         workers=hparams.workers,
    #     )
    #     test_embeddings = predict_from_loader(model, loader)

    #     is_query = is_query.type(torch.bool)
    #     # Shape (n_embeddings, embedding_dim)
    #     query_embeddings, gallery_embeddings = test_embeddings[is_query], test_embeddings[~is_query]

    #     # Matrix of pairwise cosin distances
    #     distances = torch.cdist(query_embeddings, gallery_embeddings)

    #     # Select first 10 matches, because other values are not scored

    #     for f, line in zip(query_files, torch.argsort(distances)[:, :10]):
    #         print(gallery_files[line])
    #         print(line)

        # final csv should query_img,{gallery_img_list}.


    # fold_predictions = predict_from_loader(model, loader)
    # test_predictions.append(fold_predictions)

    # # Ensemble predictions
    # if hparams.test_hold_out:
    #     holdout_predictons = torch.stack(holdout_predictons, dim=1)
    #     classes = torch.tensor(classes)
    #     # Apply averaging
    #     # holdout_predictons = average_strategy(holdout_predictons)
    #     holdout_predictons = holdout_predictons[:, 0]

    #     AUC = roc_auc_score(classes.cpu().numpy(), holdout_predictons.cpu().numpy())
    #     AP = average_precision_score(classes.cpu().numpy(), holdout_predictons.cpu().numpy())
    #     print(f"{hparams.name}: AUC {AUC:0.5f}, AP {AP:0.5f}")

        # Save to csv
        # ...

    # test_predictions = torch.stack(test_predictions, dim=1)
    # test_predictions = average_strategy(test_predictions)
    # test_predictions_binary = test_predictions > 0.5

    # Save to csv
    # df_data = {
    # "image_name": image_names,
    # "target": test_predictions_binary.type(torch.int)
    # }
    # df = pd.DataFrame(data=df_data)
    # df.to_csv(os.path.join(hparams.output_path, f'{hparams.name}_test.csv'), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huawei challenge",
    )

    parser.add_argument(
        "--config_path", "-c", type=str, help="Path to folder with model config and checkpoint")
    parser.add_argument(
        "--output_path", type=str, default="data/processed", help="Path to save scores")
    parser.add_argument(
        "--extract_embeddings", action="store_true", help="Extract and save embeddings for each image into DataFrame") 
    parser.add_argument(
        "--test_validation", action="store_true", help="Flag to make prediction for validation and compute final score")
    parser.add_argument(
        "--test", action="store_true", help="Flag to predict test")       
    parser.add_argument(
        "--tta", default=False, action='store_true', help="Use TTA" )

    hparams = parser.parse_args()
    print(f"Parameters used for test: {hparams}")
    start_time = time.time()
    test(hparams)
    print(f"Finished test. Took: {(time.time() - start_time) / 60:.02f}m")