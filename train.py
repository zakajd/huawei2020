import os
import sys
import yaml
import time

import torch
from loguru import logger
import pandas as pd
import numpy as np
import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from pytorch_tools.optim import optimizer_from_name

from src.arg_parser import parse_args
from src.datasets import get_dataloaders
from src.losses import LOSS_FROM_NAME
from src.models import Model
from src.callbacks import ContestMetricsCallback
from src.utils import freeze_batch_norm


def main():
    # Get config for this run
    hparams = parse_args()

    # Setup logger
    config = {
        "handlers": [
            {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm]} - {message}"},
            {"sink": f"{hparams.outdir}/logs.txt", "format": "{time:[MM-DD HH:mm]} - {message}"},
        ],
    }
    logger.configure(**config)
    logger.info(f"Parameters used for training: {hparams}")

    # Fix seeds for reprodusability
    pt.utils.misc.set_random_seed(hparams.seed)

    # Save config
    os.makedirs(hparams.outdir, exist_ok=True)
    yaml.dump(vars(hparams), open(hparams.outdir + "/config.yaml", "w"))

    # Get model
    model = Model(
        arch=hparams.arch,
        model_params=hparams.model_params,
        embedding_size=hparams.embedding_size,
        pooling=hparams.pooling).cuda()

    if hparams.resume:
        checkpoint = torch.load(hparams.resume, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    if hparams.freeze_bn:
        freeze_batch_norm(model)

    # Get optimizer
    optim_params = pt.utils.misc.filter_bn_from_wd(model)
    optimizer = optimizer_from_name(hparams.optim)(optim_params, lr=0, weight_decay=hparams.weight_decay, amsgrad=True)

    num_params = pt.utils.misc.count_parameters(model)[0]
    logger.info(f"Model size: {num_params / 1e6:.02f}M")
    # logger.info(model)

    # Get loss
    loss = LOSS_FROM_NAME[hparams.criterion](in_features=hparams.embedding_size, **hparams.criterion_params).cuda()
    logger.info(f"Loss for this run is: {loss}")

    # Scheduler is an advanced way of planning experiment
    sheduler = pt.fit_wrapper.callbacks.PhasesScheduler(hparams.phases)

    # Save logs
    TB_callback = pt_clb.TensorBoard(hparams.outdir, log_every=20)

    # Get dataloaders
    train_loader, val_loader, val_indexes = get_dataloaders(
        root=hparams.root,
        augmentation=hparams.augmentation,
        size=hparams.size,
        val_size=hparams.val_size,
        batch_size=hparams.batch_size,
        workers=hparams.workers,
    )

    # Load validation query / gallery split and resort it according to indexes from sampler
    df_val = pd.read_csv(os.path.join(hparams.root, "train_val.csv"))
    df_val = df_val[df_val["is_train"].astype(np.bool) == False]
    val_is_query = df_val.is_query.values[val_indexes].astype(np.bool)

    logger.info(f"Start training")
    # Init runner
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            # pt_clb.BatchMetrics([pt.metrics.Accuracy(topk=1)]),
            ContestMetricsCallback(is_query=val_is_query[: 320] if hparams.debug else val_is_query),
            pt_clb.Timer(),
            pt_clb.ConsoleLogger(),
            pt_clb.FileLogger(),
            TB_callback,
            pt_clb.CheckpointSaver(hparams.outdir, save_name="model.chpn", monitor="target", mode="max"),
            pt_clb.CheckpointSaver(hparams.outdir, save_name="model_mapr.chpn", monitor="mAP@R", mode="max"),
            sheduler,
            # EMA must go after other checkpoints
            pt_clb.ModelEma(model, hparams.ema_decay) if hparams.ema_decay else pt_clb.Callback(),
        ],
        use_fp16=hparams.use_fp16,  # use mixed precision by default.  # hparams.opt_level != "O0",
    )

    # Train
    for i, phase in enumerate(sheduler.phases):
        start_epoch, end_epoch = phase["ep"]
        logger.info(f"Start phase #{i + 1} from epoch {start_epoch} to epoch {end_epoch}: {phase}")

        runner.fit(
            train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            epochs=end_epoch,
            steps_per_epoch=20 if hparams.debug else None,
            val_steps=20 if hparams.debug else None,
        )

        logger.info(f"Loading best model from previous phase")
        checkpoint = torch.load(os.path.join(hparams.outdir, f"model.chpn"))
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Evaluate
    loss, [acc1, map10, target, mapR] = runner.evaluate(
        val_loader,
        steps=20 if hparams.debug else None,
    )

    logger.info(
        f"Val: Acc@1 {acc1:0.5f}, mAP@10 {map10:0.5f}, Target {target}, mAP@R {mapR:0.5f}")

    # Save params used for training and final metrics into separate TensorBoard file
    metric_dict = {
        "hparam/Acc@1": acc1,
        "hparam/mAP@10": map10,
        "hparam/mAP@R": target,
        "hparam/Target": mapR,
    }

    # Convert all lists / dicts to avoid TB error
    hparams.phases = str(hparams.phases)
    hparams.model_params = str(hparams.model_params)
    hparams.criterion_params = str(hparams.criterion_params)

    with pt.utils.tensorboard.CorrectedSummaryWriter(hparams.outdir) as writer:
        writer.add_hparams(hparam_dict=vars(hparams), metric_dict=metric_dict)


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
