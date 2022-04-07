from abc import ABC, abstractmethod
from typing import Optional, Callable, Sequence

import os
import torch
from torch.utils.data import DataLoader

from monai_ex.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandlerEx,
    CheckpointSaverEx,
    CheckpointLoader,
    SegmentationSaver,
    NNIReporterHandler,
    TensorboardDumper,
    from_engine,
)
from medlp.utilities.utils import output_filename_check


class MedlpTrainEngine(ABC):
    """A base class for medlp inner train engines."""

    @abstractmethod
    def __new__(
        self,
        opts,
        train_loader,
        test_loader,
        net,
        loss,
        optim,
        lr_scheduler,
        writer,
        device,
        model_dir,
        logger_name,
        **kwargs,
    ):
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    @staticmethod
    def get_extra_handlers(
        phase,
        model_dir,
        net,
        optimizer,
        tb_summary_writer,
        logger_name: Optional[str] = None,
        stats_dicts: Optional[dict] = None,
        save_checkpoint: bool = False,
        checkpoint_save_interval: int = 10,
        ckeckpoint_n_saved: int = 1,
        save_bestmodel: bool = False,
        model_file_prefix: str = "",
        bestmodel_n_saved: int = 1,
        tb_show_image: bool = False,
        tb_image_batch_transform: Callable = lambda x: (),
        tb_image_output_transform: Callable = lambda x: (),
        dump_tensorboard: bool = False,
        record_nni: bool = False,
        nni_kwargs: Optional[dict] = None,
    ):
        handlers = []

        if stats_dicts is not None:
            for key, output_transform_fn in stats_dicts.items():
                handlers += [
                    StatsHandler(
                        tag_name=key,
                        output_transform=output_transform_fn,
                        name=logger_name,
                    ),
                    TensorBoardStatsHandler(
                        summary_writer=tb_summary_writer,
                        tag_name=key,
                        output_transform=output_transform_fn,
                    ),
                ]
        if save_checkpoint:
            handlers += [
                CheckpointSaverEx(
                    save_dir=model_dir / "Checkpoint",
                    save_dict={"net": net, "optim": optimizer},
                    save_interval=checkpoint_save_interval,
                    epoch_level=True,
                    n_saved=ckeckpoint_n_saved,
                ),
            ]
        if save_bestmodel:
            handlers += [
                CheckpointSaverEx(
                    save_dir=model_dir / "Best_Models",
                    save_dict={"net": net},
                    file_prefix=model_file_prefix,
                    save_key_metric=True,
                    key_metric_n_saved=bestmodel_n_saved,
                )
            ]

        if tb_show_image:
            handlers += [
                TensorBoardImageHandlerEx(
                    summary_writer=tb_summary_writer,
                    batch_transform=tb_image_batch_transform,
                    output_transform=tb_image_output_transform,
                    max_channels=3,
                    prefix_name=phase,
                ),
            ]

        if dump_tensorboard:
            handlers += [
                TensorboardDumper(
                    log_dir=tb_summary_writer.log_dir,
                    epoch_level=True,
                    logger_name=logger_name,
                ),
            ]

        if record_nni:
            handlers += [NNIReporterHandler(**nni_kwargs)]

        return handlers


class MedlpTestEngine(ABC):
    """A base class for medlp inner test engines."""

    @abstractmethod
    def __new__(
        self,
        opts,
        test_loader,
        net,
        device,
        logger_name,
        **kwargs,
    ):
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    @staticmethod
    def get_extra_handlers(
        phase,
        out_dir,
        model_path,
        net,
        logger_name: Optional[str] = None,
        stats_dicts: Optional[dict] = None,
        save_image: bool = False,
        image_resample: bool = False,
        test_loader: Optional[DataLoader] = None,
        image_batch_transform: Callable = lambda x: (),
        image_output_transform: Callable = lambda x: (),
    ):
        handlers = []

        if os.path.exists(model_path):
            handlers += [CheckpointLoader(load_path=model_path, load_dict={"net": net})]

        if stats_dicts is not None:
            for key, output_transform_fn in stats_dicts.items():
                handlers += [
                    StatsHandler(
                        tag_name=key,
                        output_transform=output_transform_fn,
                        name=logger_name,
                    )
                ]

        if save_image:
            handlers += [
                SegmentationSaver(
                    output_dir=out_dir,
                    output_ext=".nii.gz",
                    output_postfix="image",
                    data_root_dir=output_filename_check(test_loader.dataset),
                    resample=image_resample,
                    mode="bilinear",
                    batch_transform=image_batch_transform,
                    output_transform=image_output_transform,
                )
            ]

        return handlers
