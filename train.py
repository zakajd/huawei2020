import os
import sys
import yaml
import time

import torch
from loguru import logger
import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from pytorch_tools.optim import optimizer_from_name

from src.arg_parser import parse_args
from src.datasets import get_dataloaders
from src.losses import LOSS_FROM_NAME
from src.models import Model
from src.callbacks import ContestMetricsCallback
# ContestMetricsCallback


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

    logger.info(f"Start training")

    # Get model
    model = Model(
        arch=hparams.arch,
        model_params=hparams.model_params,
        embedding_size=hparams.embedding_size,
        pooling=hparams.pooling).cuda()

    if hparams.resume:
        checkpoint = torch.load(hparams.resume, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint["state_dict"], strict=True)

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

    # Init runner
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            # pt_clb.BatchMetrics([pt.metrics.Accuracy(topk=1)]),
            ContestMetricsCallback(),
            pt_clb.Timer(),
            pt_clb.ConsoleLogger(),
            pt_clb.FileLogger(),
            TB_callback,
            pt_clb.CheckpointSaver(hparams.outdir, save_name="model.chpn"),
            sheduler,
            # EMA must go after other checkpoints
            pt_clb.ModelEma(model, hparams.ema_decay) if hparams.ema_decay else pt_clb.Callback(),
        ],
        use_fp16=hparams.use_fp16,  # use mixed precision by default.  # hparams.opt_level != "O0",
    )

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        root=hparams.root,
        augmentation=hparams.augmentation,
        size=hparams.size,
        val_size=hparams.val_size,
        batch_size=hparams.batch_size,
        workers=hparams.workers,
    )
    logger.info(f"{hasattr(train_loader, 'batch_size')}, {train_loader.batch_size}, {val_loader.batch_size}")

    # Train
    for i, phase in enumerate(sheduler.phases):
        start_epoch, end_epoch = phase["ep"]
        logger.info(f"Start phase #{i + 1} from epoch {start_epoch} to epoch {end_epoch}: {phase}")

        runner.fit(
            train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            epochs=end_epoch - start_epoch,
            steps_per_epoch=20 if hparams.debug else None,
            val_steps=20 if hparams.debug else None,
        )

        logger.info(f"Loading best model from previous phase")
        checkpoint = torch.load(os.path.join(hparams.outdir, f"model.chpn"))
        model.load_state_dict(checkpoint["state_dict"])

    # Save params used for training and final metrics into separate TensorBoard file
    # metric_dict = {
    #     "hparam/accuracy": np.mean(fold_metrics["accuracy"]),
    #     "hparam/roc_auc": np.mean(fold_metrics["roc_auc"]),
    #     "hparam/average_precision": float(np.mean(fold_metrics["average_precision"])),
    # }

    # hparams.config_file = str(hparams.config_file)
    # hparams.phases = str(hparams.phases)
    # hparams.folds = str(hparams.folds)
    # hparams.datasets = str(hparams.datasets)
    # hparams.model_params = str(hparams.model_params)

    # with SummaryWriter(hparams.outdir) as writer:
    #     writer.add_hparams(hparam_dict=vars(hparams), metric_dict=metric_dict)


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
