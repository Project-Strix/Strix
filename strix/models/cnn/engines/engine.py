import os
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Union
from pathlib import Path

import torch
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.optimizer import Optimizer

from strix.configures import config as cfg
from strix.utilities.enum import Phases
from strix.utilities.utils import output_filename_check
from monai_ex.handlers import (
    CheckpointLoader,
    CheckpointSaverEx,
    NNIReporterHandler,
    ImageBatchSaver,
    StatsHandlerEx as StatsHandler,
    TensorboardDumper,
    TensorBoardImageHandlerEx,
    TensorBoardStatsHandler,
    TensorboardGraphHandler,
    FreezeNetHandler,
)
from monai_ex.utils import ensure_list
from torch.utils.data import DataLoader


class StrixTrainEngine(ABC):
    """A base class for strix inner train engines."""

    @abstractmethod
    def __init__(
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
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @staticmethod
    def get_basic_handlers(
        phase: Phases,
        model_dir: Path,
        net: Module,
        optimizer: Union[Optimizer, Sequence[Optimizer]],
        tb_summary_writer: SummaryWriter,
        logger_name: Optional[str] = None,
        stats_dicts: Optional[Dict] = None,
        save_checkpoint: bool = False,
        checkpoint_save_interval: int = 10,
        ckeckpoint_n_saved: int = 1,
        save_bestmodel: bool = False,
        model_file_prefix: str = "",
        bestmodel_n_saved: int = 1,
        tensorboard_image_kwargs: Optional[Union[Dict, Sequence[Dict]]] = None,
        tensorboard_image_names: Optional[Union[str, Sequence[str]]] = "",
        dump_tensorboard: bool = False,
        graph_batch_transform: Optional[Callable] = None,
        record_nni: bool = False,
        nni_kwargs: Optional[Dict] = None,
        freeze_mode: Optional[str] = None,
        freeze_params: Optional[tuple] = None,
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
                    save_dir=str(model_dir / "Checkpoint"),
                    save_dict={"net": net, "optim": optimizer},
                    save_interval=checkpoint_save_interval,
                    epoch_level=True,
                    n_saved=ckeckpoint_n_saved,
                ),
            ]
        if save_bestmodel:
            handlers += [
                CheckpointSaverEx(
                    save_dir=str(model_dir / "Best_Models"),
                    save_dict={"net": net},
                    file_prefix=model_file_prefix,
                    save_key_metric=True,
                    key_metric_n_saved=bestmodel_n_saved,
                )
            ]

        if tensorboard_image_kwargs is not None:
            tb_img_kwargs = ensure_list(tensorboard_image_kwargs)
            tb_img_names = ensure_list(tensorboard_image_names)
            if len(tb_img_kwargs) != len(tb_img_names):
                raise ValueError(
                    f"Length of 'tensorboard_image_kwargs' ({len(tb_img_kwargs)}) "
                    f"should equal to 'tensorboard_image_names' ({len(tb_img_names)})."
                )

            handlers += [
                TensorBoardImageHandlerEx(
                    summary_writer=tb_summary_writer,
                    prefix_name=name + "-" + phase.value,
                    logger_name=logger_name,
                    **kwargs,
                )
                for kwargs, name in zip(tb_img_kwargs, tb_img_names)
                if kwargs is not None
            ]

        if dump_tensorboard and tb_summary_writer:
            handlers += [
                TensorboardDumper(
                    log_dir=tb_summary_writer.log_dir,
                    epoch_level=True,
                    logger_name=logger_name,
                ),
            ]

        if graph_batch_transform:
            handlers += [
                TensorboardGraphHandler(
                    net=net,
                    writer=tb_summary_writer,
                    batch_transform=graph_batch_transform,
                    logger_name=logger_name
                ),
            ]

        if record_nni:
            raise NotImplementedError
            handlers += [NNIReporterHandler(**nni_kwargs)]

        if freeze_mode:
            handlers += [
                FreezeNetHandler(
                    network=net,
                    freeze_mode=freeze_mode,
                    freeze_params=freeze_params,
                    logger_name=logger_name
                )
            ]

        return handlers


class StrixTestEngine(ABC):
    """A base class for strix inner test engines."""

    @abstractmethod
    def __init__(
        self,
        opts: SimpleNamespace,
        test_loader: DataLoader,
        net: Dict,
        device: Union[str, torch.device],
        logger_name: str,
        **kwargs,
    ):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @staticmethod
    def get_basic_handlers(
        phase: str,
        out_dir: str,
        model_path: Union[str, Sequence[str]],
        load_dict: Union[Dict, Sequence[Dict]],
        logger_name: Optional[str] = None,
        stats_dicts: Optional[Dict] = None,
        save_image: bool = False,
        image_resample: bool = False,
        test_loader: Optional[DataLoader] = None,
        image_batch_transform: Callable = lambda x: (),
    ):
        handlers = []

        model_paths, load_dicts = ensure_list(model_path), ensure_list(load_dict)
        for load_path, load_dict in zip(model_paths, load_dicts):
            if os.path.exists(load_path):
                handlers += [CheckpointLoader(load_path=load_path, load_dict=load_dict, name=logger_name)]
            else:
                raise FileNotFoundError(f"Model not found! {load_path}")

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
            _image = cfg.get_key("image")
            handlers += [
                ImageBatchSaver(
                    output_dir=out_dir,
                    output_ext=".nii.gz",
                    output_postfix=_image,
                    data_root_dir=output_filename_check(test_loader.dataset, meta_key=_image + "_meta_dict"),
                    resample=image_resample,
                    mode="bilinear",
                    image_batch_transform=image_batch_transform,
                )
            ]

        return handlers
