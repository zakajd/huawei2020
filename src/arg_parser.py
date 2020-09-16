import os
import configargparse as argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Huawei 2020 Challenge",
        default_config_files=["configs/default.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )

    add_arg = parser.add_argument

    # General
    add_arg("--name", type=str, help="Name of this run")
    add_arg("--seed", type=int, help="Random seed for reprodusability")
    add_arg("--root", type=str, help="Path to train data")
    add_arg("--debug", dest="debug", default=False, action="store_true", help="Make short epochs")
    add_arg("--resume", default="", type=str, help="Path to checkpoint to start from")

    # DATALOADER
    add_arg("--batch_size", type=int, help="Batch size")
    add_arg("--workers", type=int, help="№ of data loading workers ")
    add_arg("--augmentation", default="light", type=str, help="How hard augs are")

    # Model
    add_arg("--arch", default="unet", type=str, help="Architecture to use")
    add_arg("--embedding_size", type=int, default=512, help="Size of descriptor's dimmension")
    add_arg("--pooling", type=str, default="max", help="Pooling used after last feature map")
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")
    add_arg("--ema_decay", type=float, default=0, help="Decay for ExponentialMoving Average")


    # Training
    add_arg("--use_fp16", default=False, action="store_true", help="Flag to enable FP16 training")
    add_arg("--optim", type=str, default="adamw", help="Optimizer to use (default: adamw)")
    add_arg("--weight_decay", "--wd", default=1e-4, type=float, help="Weight decay (default: 1e-4)")
    add_arg("--size", default=512, type=int, help="Size of crops to train at")
    add_arg(
        "--phases",
        type=eval,
        action="append",
        help="Specify epoch order of data resize and learning rate schedule",
    )

    # Criterion
    add_arg("--criterion", type=str, help="Criterion to use.")
    add_arg("--criterion_params", type=eval, default={}, help="Params to pass to criterion")

    # Validation and testing
    add_arg(
        "--val_frequency",
        dest="val_frequency",
        type=int,
        default=1,
        help="How often to run validation. Default: After each training epochs",
    )
    add_arg("--val_size", type=int, default=512, help="Predict on resized, then upscale")
    add_arg(
        "--tta",
        dest="tta",
        default=False,
        action="store_true",
        help="Flag to use TTA for validation and test sets",
    )
    return parser


def parse_args():
    parser = get_parser()
    args, not_parsed_args = parser.parse_known_args()
    print("Not parsed args: ", not_parsed_args)

    # If folder already exist append version number
    outdir = os.path.join("logs", f"{args.name}")
    if os.path.exists(outdir):
        version = 1
        while os.path.exists(outdir):
            outdir = os.path.join("logs", f"{args.name}_{version}")
            version += 1

    args.outdir = outdir
    return args
