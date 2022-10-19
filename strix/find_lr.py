import logging
from typing import Union, Tuple
from functools import partial
from types import SimpleNamespace as sn
from pathlib import Path

import numpy as np
import click

import strix.utilities.arguments as arguments
from strix.models import get_engine
from strix.data_io.dataio import get_dataloader
from strix.utilities.click import OptionEx
from strix.utilities.click_callbacks import NumericChoice as Choice
from strix.utilities.click_callbacks import get_exp_name, lr_schedule_params, select_gpu
from strix.utilities.enum import OPTIMIZERS, Phases
from strix.utilities.generate_cohorts import generate_train_valid_cohorts
from strix.utilities.utils import setup_logger
from monai_ex.handlers.lr_record_handler import LearningHistoryRecordHandler
from monai_ex.utils import optional_import

plt, has_matplotlib = optional_import("matplotlib.pyplot")

option = partial(click.option, cls=OptionEx)


def get_steepest_gradient(lrs, losses) -> Union[Tuple[float, float], Tuple[None, None]]:
    """Get learning rate which has steepest gradient and its corresponding loss
    Args:
        lrs: series of learning rates.
        losses: series of losses.
    Returns:
        Learning rate which has steepest gradient and its corresponding loss
    """
    try:
        min_grad_idx = np.gradient(np.array(losses)).argmin()
        return lrs[min_grad_idx], losses[min_grad_idx]
    except ValueError:
        print("Failed to compute the gradients, there might not be enough points.")
        return None, None


def save_plot(lrs, losses, output_dir: Path, log_lr: bool = True, ax=None, steepest_lr: bool = True):
    # Create the figure and axes object if axes was not already given
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot loss as a function of the learning rate
    ax.plot(lrs, losses)

    # Plot the LR with steepest gradient
    if steepest_lr:
        lr_at_steepest_grad, loss_at_steepest_grad = get_steepest_gradient(lrs, losses)
        if lr_at_steepest_grad is not None:
            ax.scatter(
                lr_at_steepest_grad,
                loss_at_steepest_grad,
                s=75,
                marker="o",
                color="red",
                zorder=3,
                label="steepest gradient",
            )
            ax.legend()

    if log_lr:
        ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title(f"Steepest LR: {lr_at_steepest_grad:0.2f}")

    plt.savefig(output_dir / "best_lr_plot.png")


@click.command("find-lr", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@arguments.common_params
@arguments.network_params
@option("--save-n-best", type=int, default=0, help="Save best N models")
@option("--smooth", type=float, default=0.05, help="Smooth factor for losses")
@option("--optim", type=Choice(OPTIMIZERS), default="sgd", help="Optimizer for network")
@option("--momentum", type=float, default=0.0, help="Momentum for optimizer")
@option("--nesterov", type=bool, default=False, help="Nesterov for SGD")
@option("-WD", "--l2-weight-decay", type=float, default=0, help="weight decay (L2 penalty)")
@option("--lr", prompt="Start LR", type=float, default=1e-3, help="learning rate")
@option("--lr-policy-level", type=Choice(["epoch", "iter"]), default="iter", help="Iterlevel or Epochlevel")
@option(
    "--lr-policy",
    prompt=True,
    type=Choice(["exponential", "linear"]),
    callback=lr_schedule_params,
    default="linear",
    help="learning rate strategy",
)
@option("--gpus", type=str, callback=select_gpu, help="The ID of active GPU")
@option("--experiment-path", type=str, callback=get_exp_name, default="")
def find_best_lr(**args):
    cargs = sn(**args)

    if "," in cargs.gpus:
        cargs.gpu_ids = list(range(len(list(map(int, cargs.gpus.split(","))))))
    else:
        cargs.gpu_ids = [0]

    # ! Set logger
    logger_name = "Find_best_LR"
    cargs.logger_name = logger_name
    logging_level = logging.DEBUG if cargs.debug else logging.INFO
    log_path = None if cargs.disable_logfile else cargs.experiment_path.joinpath("logs")
    log_terminator = "\r" if cargs.compact_log and not cargs.debug else "\n"
    logger = setup_logger(logger_name, logging_level, filepath=log_path, reset=True, terminator=log_terminator)

    if not has_matplotlib:
        logger.error("No matplotlib is found!")
        return

    train_data, valid_data = generate_train_valid_cohorts(**vars(cargs), logger=logger)[0]
    train_loader = get_dataloader(cargs, train_data, phase=Phases.TRAIN)
    valid_loader = get_dataloader(cargs, valid_data, phase=Phases.VALID)

    trainer, _ = get_engine(cargs, train_loader, valid_loader, writer=None)
    history_handler = LearningHistoryRecordHandler(lambda x: x["loss"], None)
    history_handler.attach(trainer)
    trainer.run()

    losses = history_handler.history["loss"]
    learningrates = history_handler.history["lr"]

    if not losses:
        logger.error("Cannot log training losses. Please check!")
        return

    if len(losses) != len(learningrates):
        logger.error("Numbers of losses and learningrate do no match. Please check!")
        return

    if cargs.smooth > 0:
        smoothed_losses = np.array(losses[:-1]) * (1 - cargs.smooth) + np.array(losses[1:]) * cargs.smooth
        smoothed_losses = smoothed_losses.squeeze().tolist()
        smoothed_losses.append(float(losses[-1]))
        losses = smoothed_losses

    save_plot(learningrates, losses, cargs.experiment_path)
