"""
1. Resize images and save
"""
import os
import cv2
import sys
import time
import shutil
import pathlib
import functools
import multiprocessing
import configargparse as argparse

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from loguru import logger

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


def get_single_size(filename):
    # use PIL to avoid reading image. it's much faster
    return Image.open(filename).size, filename


def get_sizes(filenames):
    """Returns list of sizes for files in filenames"""
    with multiprocessing.Pool(NUM_THREADS) as pool:
        result = list(tqdm(pool.imap(get_single_size, filenames), total=len(filenames)))
    return result


def main(hparams):
    hparams.root = pathlib.Path(hparams.root)
    hparams.output_path = pathlib.Path(hparams.output_path)

    # Read filenames
    with open(hparams.root / "train_data/label.txt") as f:
        data = f.readlines()

    filenames, labels = [], []
    for row in data:
        file, label = row.strip("\n").split(",")
        filenames.append(file)
        labels.append(int(label))

    train_filenames = [hparams.root / "train_data" / file for file in filenames]

    test_A_query_files = sorted((hparams.root / "test_data_A" / "query").glob("*.jpg"))
    test_A_gallery_files = sorted((hparams.root / "test_data_A" / "gallery").glob("*.jpg"))
    # test_B_query_files = sorted((hparams.root / "test_data_B" / "query").glob("*.jpg"))
    # test_B_gallery_files = sorted((hparams.root / "test_data_B" / "gallery").glob("*.jpg"))

    test_A_filenames = test_A_query_files + test_A_gallery_files
    # test_B_filenames = test_B_query_files + test_B_gallery_files

    # Delete old images
    # shutil.rmtree(hparams.output_path / f"train_data_{hparams.size}", ignore_errors=True)
    # (hparams.output_path / f"train_data_{hparams.size}").mkdir()

    # shutil.rmtree(hparams.output_path / f"test_data_A_{hparams.size}", ignore_errors=True)
    # (hparams.output_path / f"test_data_A_{hparams.size}").mkdir()

    # shutil.rmtree(hparams.output_path / f"test_data_B_{hparams.size}", ignore_errors=True)
    # (hparams.output_path / f"test_data_B_{hparams.size}").mkdir()

    # logger.info("Resizing train images...")
    # _ = resize_images(
    #     files=train_filenames,
    #     folder=f"train_data",
    #     size=hparams.size,)

    # logger.info("Resizing test A images...")
    # _ = resize_images(
    #     files=test_A_filenames,
    #     folder=f"test_data_A",
    #     size=hparams.size,)

    # logger.info("Resizing test B images...")
    # test_outfilenames = resize_images(
    #     files=test_B_filenames,
    #     folder=f"test_data_B",
    #     size=hparams.size)

    logger.info("Creating DF with additional metadata")

    # # Train
    df_data = {
        "file_path": filenames,
        "label": labels,
    }
    df = pd.DataFrame(data=df_data)

    # Get original image size as an additional feature
    result = get_sizes(train_filenames)
    df.sort_values(by="file_path", inplace=True)
    sizes = [str(x[0]) for x in sorted(result, key=lambda x: x[1])]
    df["original_size"] = sizes
    df["aspect_ratio"] = [round(x[0][0] / x[0][1], 4) for x in sorted(result, key=lambda x: x[1])]

    # Take `val_pct` of labels for validation
    unique_labels = np.unique(labels)
    val_labels = unique_labels[:int(len(unique_labels) * hparams.val_pct)]
    df["is_train"] = [False if l in val_labels else True for l in labels]

    # Take 2 images from each class as a query
    is_query = [False] * len(labels)
    for l in val_labels:
        ind = labels.index(l)
        is_query[ind] = True
        is_query[ind + 1] = True
    df["is_query"] = is_query

    # Save results
    df.to_csv(hparams.output_path / "train_val.csv", index=None)

    # Test A
    df_test_data = {
        "file_path": [p.relative_to(hparams.root / "test_data_A") for p in test_A_filenames],
    }
    df_test = pd.DataFrame(data=df_test_data)

    # Get original image size as an additional feature
    result = get_sizes(test_A_filenames)
    df_test.sort_values(by="file_path", inplace=True)
    sizes = [str(x[0]) for x in sorted(result, key=lambda x: x[1])]
    df_test["original_size"] = sizes
    df_test["aspect_ratio"] = [round(x[0][0] / x[0][1], 4) for x in sorted(result, key=lambda x: x[1])]
    df_test["is_query"] = torch.tensor([0.0] * len(test_A_gallery_files) + [1.0] * len(test_A_query_files))

    # Save results
    df_test.to_csv(hparams.output_path / "test_A.csv", index=None)


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

    # Resize
    add_arg("--size", type=int, default=512, help="Size of min side after resize")

    # Split
    add_arg("--val_pct", type=float, default=0.2, help="Part of train data used for validation")

    # Setup logger
    config = {"handlers": [{"sink": sys.stdout, "format": "{time:[HH:mm]}:{message}"}]}
    logger.configure(**config)

    hparams = parser.parse_args()
    logger.debug(f"Parameters used for preprocessing: {hparams}")

    start_time = time.time()
    main(hparams)
    logger.info(f"Finished preprocessing. Took: {(time.time() - start_time) / 60:.02f}m")
