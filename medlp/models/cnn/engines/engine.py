import os
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Union

import torch

from medlp.configures import config as cfg
from medlp.utilities.utils import output_filename_check
from monai_ex.handlers import (
    CheckpointLoader,
    CheckpointSaverEx,
    NNIReporterHandler,
    ImageBatchSaver,
    StatsHandler,
    TensorboardDumper,
    TensorBoardImageHandlerEx,
    TensorBoardStatsHandler,
)
from monai_ex.utils import ensure_list
from torch.utils.data import DataLoader


class MedlpTrainEngine(ABC):
    """A base class for medlp inner train engines."""

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
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    @staticmethod
    def get_basic_handlers(
        phase,
        model_dir,
        net,
        optimizer,
        tb_summary_writer,
        logger_name: Optional[str] = None,
        stats_dicts: Optional[Dict] = None,
        save_checkpoint: bool = False,
        checkpoint_save_interval: int = 10,
        ckeckpoint_n_saved: int = 1,
        save_bestmodel: bool = False,
        model_file_prefix: str = "",
        bestmodel_n_saved: int = 1,
        tensorboard_image_kwargs: Optional[Union[Dict, Sequence[Dict]]] = None,
        tensorboard_image_names: Optional[Union[Dict, Sequence[Dict]]] = "",
        dump_tensorboard: bool = False,
        record_nni: bool = False,
        nni_kwargs: Optional[Dict] = None,
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
                    prefix_name=name + "-" + phase,
                    **kwargs,
                ) for kwargs, name in zip(tb_img_kwargs, tb_img_names) if kwargs is not None
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
    def __init__(
        self,
        opts: SimpleNamespace,
        test_loader: DataLoader,
        net: Dict,
        device: Union[str, torch.device],
        logger_name: str,
        **kwargs,
    ):
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

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
                handlers += [CheckpointLoader(load_path=load_path, load_dict=load_dict)]

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
                    data_root_dir=output_filename_check(test_loader.dataset, meta_key=_image+"_meta_dict"),
                    resample=image_resample,
                    mode="bilinear",
                    image_batch_transform=image_batch_transform,
                )
            ]

        return handlers
