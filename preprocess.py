"""
1. Resize images and save
"""
import os
import cv2
import sys
import json
import time
import shutil
import collections
import pathlib
import functools
import multiprocessing
import configargparse as argparse


import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.model_selection import GroupKFold

NUM_THREADS = 10

def resize_one_image(filepath_outputpath, size):
    """
    Args:
        filepath_outputpath (Tuple[str, str]): file path and output path names
        size (int): size of SMALLER side after resize
    """
    file_path, output_path = filepath_outputpath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_path, output_path = str(file_path), str(output_path)  # make sure it's not Path

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # no cvt color on purpose. saving BGR images gives RGB on disk

    # Resize image to have `size` shape on smaller side
    h, w = image.shape[:2]
    smaller_side = min(h, w)
    ratio = size / smaller_side
    resized = cv2.resize(image, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(output_path, resized)


def resize_images(files, folder, size):
    """
    Args:
        files (List[str]): List of files to be resized
        folder (str): General folder
        size (int): Size of smallest side after resize
    """

    # Change all data formats to jpg
    out_filenames = list(map(
        lambda x: pathlib.Path("data/interim", f"{folder}_{size}", os.path.relpath(x, start=f"data/raw/{folder}")), files)
    )
    resize_function = functools.partial(resize_one_image, size=size)

    # print(files[:1],out_filenames[:1])
    # return None

    with multiprocessing.Pool(NUM_THREADS) as pool:
        list(tqdm(pool.imap_unordered(resize_function, zip(files, out_filenames)), total=len(files)))
    return out_filenames


def main(hparams):
    hparams.root = pathlib.Path(hparams.root)
    hparams.output_path = pathlib.Path(hparams.output_path)

    # -------------- Resave images to smaller size --------------
    # Read filenames
    with open(hparams.root / "train_data/label.txt") as f:
        data = f.readlines()

    train_filenames = []
    for row in data:
        filename, _ = row.strip("\n").split(",")
        train_filenames.append(hparams.root / f"train_data" / filename)

    test_A_filenames = list((hparams.root / 'test_data_A').rglob("*.jpg"))
    # test_B_filenames = (hparams.root / 'test_data_B').rglob("*.jpg")

    # Delete old images
    shutil.rmtree(hparams.output_path / f"train_data_{hparams.size}", ignore_errors=True)
    (hparams.output_path / f"train_data_{hparams.size}").mkdir()

    shutil.rmtree(hparams.output_path / f"test_data_A_{hparams.size}", ignore_errors=True)
    (hparams.output_path / f"test_data_A_{hparams.size}").mkdir()

    # shutil.rmtree(hparams.output_path / f"test_data_B_{hparams.size}", ignore_errors=True)
    # (hparams.output_path / f"test_data_B_{hparams.size}").mkdir()

    logger.info("Resizing train images...")
    _ = resize_images(
        files=train_filenames,
        folder=f"train_data",
        size=hparams.size,)

    logger.info("Resizing test A images...")
    _ = resize_images(
        files=test_A_filenames,
        folder=f"test_data_A",
        size=hparams.size,)

    # logger.info("Resizing test B images...")
    # test_outfilenames = resize_images(
    #     files=test_B_filenames,
    #     folder=f"test_data_B",
    #     size=hparams.size)


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(
        description="Huawei Challenge Preprocess",
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )
    add_arg = parser.add_argument

    # base args
    add_arg("--output_path", type=str, default="data/interim", help="Path to save files")
    add_arg("--root", type=str, default="data/raw", help="Path to all raw data as provided by organizers")

    # resize
    add_arg("--size", type=int, default=512, help="Size of min side after resize")

    # Setup logger
    config = {"handlers": [{"sink": sys.stdout, "format": "{time:[HH:mm]}:{message}"},]}
    logger.configure(**config)

    hparams = parser.parse_args()
    logger.debug(f"Parameters used for preprocessing: {hparams}")

    start_time = time.time()
    main(hparams)
    logger.info(f"Finished preprocessing. Took: {(time.time() - start_time) / 60:.02f}m")