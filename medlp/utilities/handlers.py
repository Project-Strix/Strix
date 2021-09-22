import os
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Optional, Callable, Union

import torch
import numpy as np
import nibabel as nib

from monai_ex.utils import export, exact_version, optional_import
from monai_ex.handlers import TensorBoardImageHandler
from monai_ex.visualize import plot_2d_or_3d_image, GradCAM, LayerCAM

from PIL import Image
from utils_cw import Normalize2

from medlp.models.cnn.layers.snip import SNIP, apply_prune_mask
from medlp.utilities.utils import (
    apply_colormap_on_image,
    dump_tensorboard,
    plot_summary
)

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
    from ignite.metrics import Metric
    from ignite.handlers import Checkpoint
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")
    Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
    Metric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "Metric")
    Checkpoint, _ = optional_import("ignite.handlers", "0.4.4", exact_version, "Checkpoint")

NNi, _ = optional_import("nni")
Torchviz, _ = optional_import('torchviz')


class NNIReporter:
    """
    NNIReporter 

    Args:

    """
    def __init__(
        self,
        metric_name: str,
        logger_name: Optional[str] = None,
        report_final: bool = False
    ) -> None:
        self.metric_name = metric_name
        self.logger_name = logger_name
        self.report_final = report_final
        self.logger = logging.getLogger(logger_name)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.logger_name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        # assert self.metric_name in engine.state.metrics.keys(), f"{self.metric_name} is not in engine's metrics: {engine.state.metrics.keys()}"
        print('----------keys-----------', engine.state.metrics.keys())
        if self.metric_name in engine.state.metrics.keys():
            print('*'*10, engine.state.metrics[self.metric_name], type(engine.state.metrics[self.metric_name]))
            if not self.report_final:
                NNi.report_intermediate_result(engine.state.metrics[self.metric_name])
            else:
                NNi.report_final_result(engine.state.metrics[self.metric_name])


class NNIReporterHandler:
    """
    NNIReporter 

    Args:

    """
    def __init__(
        self,
        metric_name: str,
        max_epochs: int,
        logger_name: Optional[str] = None,
    ) -> None:
        self.metric_name = metric_name
        self.logger_name = logger_name
        self.max_epochs = max_epochs
        self.logger = logging.getLogger(logger_name)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.logger_name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.report_intermediate_result)
        engine.add_event_handler(Events.COMPLETED, self.report_final_result)
        engine.add_event_handler(Events.TERMINATE, self.report_final_result)

    def report_intermediate_result(self, engine):
        self.logger.info(f'{engine.state.epoch} report intermediate')
        NNi.report_intermediate_result(engine.state.metrics[self.metric_name])

    def report_final_result(self, engine):
        if engine.state.epoch == self.max_epochs:
            self.logger.info(f'{engine.state.epoch} report final')
            NNi.report_final_result(engine.state.metrics[self.metric_name])


class SNIP_prune_handler:
    def __init__(self,
                 net,
                 prepare_batch_fn,
                 loss_fn,
                 prune_percent,
                 data_loader,
                 device='cuda',
                 snip_device='cpu',
                 verbose=False,
                 logger_name: Optional[str] = None
    ) -> None:
        self.net = net
        self.prepare_batch_fn = prepare_batch_fn
        self.loss_fn = loss_fn
        self.prune_percent = prune_percent
        self.data_loader = data_loader
        self.device = device
        self.snip_device = snip_device
        self.verbose = verbose
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)

    def __call__(self, engine: Engine) -> None:
        self.logger.debug("-------------- In SNIP handler ---------------")
        keep_masks = SNIP(
            self.net,
            self.prepare_batch_fn,
            self.loss_fn,
            self.prune_percent,
            self.data_loader,
            self.snip_device,
            None
        )
        net_ = apply_prune_mask(self.net, keep_masks, self.device, self.verbose)
        # self.net.load_state_dict(net_.state_dict())


class TorchVisualizer:
    """
    TorchVisualizer for visualize network architecture using PyTorchViz.
    """
    def __init__(
        self,
        net,
        outfile_path: str,
        output_transform: Callable = lambda x: x,
        logger_name: Optional[str] = None
    ) -> None:
        self.net = net
        assert net is not None, "Network model should be input"
        self.outfile_path = outfile_path
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.output_transform = output_transform

    def attach(self, engine: Engine) -> None:
        if self.logger_name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        output = self.output_transform(engine.state.output)
        if output is not None:
            try:
                dot = Torchviz.make_dot(output, dict(self.net.named_parameters()))
                print(output)
                print()
            except:
                self.logger.error('Generate graph failded')
            else:
                try:
                    dot.render(self.outfile_path)
                except:
                    self.logger.error(f"""Failded to save torchviz graph to {self.outfile_path},
                                    Please make sure you have installed graphviz properly!""")


class GradCamHandler:
    def __init__(
        self,
        net,
        target_layers,
        target_class,
        data_loader,
        prepare_batch_fn,
        method: str = 'gradcam',
        fusion: bool = False,
        save_dir: Optional[str] = None,
        device: torch.device = torch.device('cpu'),
        logger_name: Optional[str] = None
    ) -> None:
        self.net = net
        self.target_layers = target_layers
        self.target_class = target_class
        self.data_loader = data_loader
        self.prepare_batch_fn = prepare_batch_fn
        self.save_dir = save_dir
        self.device = device
        self.logger = logging.getLogger(logger_name)
        self.fusion = fusion
        if method == 'gradcam':
            self.cam = GradCAM(nn_module=self.net, target_layers=self.target_layers)
        elif method == 'layercam':
            self.cam = LayerCAM(nn_module=self.net, target_layers=self.target_layers)

    def __call__(self, engine: Engine) -> None:
        for i, batchdata in enumerate(self.data_loader):
            batch = self.prepare_batch_fn(batchdata, self.device, False)
            if len(batch) == 2:
                inputs, targets = batch
            else:
                raise NotImplementedError

            if isinstance(inputs, (tuple, list)):
                self.logger.warn(
                    f"Got multiple inputs with size of {len(batch)}," 
                    "select the first one as image data."
                )
                origin_img = inputs[0].cpu().detach().numpy().squeeze(1)
            else:
                origin_img = inputs.cpu().detach().numpy().squeeze(1)

            self.logger.debug(
                f'Input len: {len(inputs)}, shape: {origin_img.shape}'
            )

            cam_result = self.cam(
                inputs,
                class_idx=self.target_class,
                img_spatial_size=origin_img.shape[1:]
            )

            self.logger.debug(
                f'Image batchdata shape: {origin_img.shape}, '
                f'CAM batchdata shape: {cam_result.shape}'
            )

            if len(origin_img.shape[1:]) == 3:
                for j, (img_slice, cam_slice) in enumerate(zip(origin_img, cam_result)):
                    nib.save(
                        nib.Nifti1Image(img_slice.squeeze(), np.eye(4)),
                        self.save_dir/f'batch{i}_{j}_images.nii.gz'
                    )

                    if cam_slice.shape[0] > 1 and self.fusion:  # multiple cam maps
                        nib.save(
                            nib.Nifti1Image(cam_slice.mean(axis=0).squeeze(), np.eye(4)),
                            self.save_dir/f'batch{i}_{j}_cam_fusion_{self.target_layers}.nii.gz'
                        )
                    else:
                        nib.save(
                            nib.Nifti1Image(cam_slice.transpose(1,2,3,0).squeeze(), np.eye(4)),
                            self.save_dir/f'batch{i}_{j}_cam_map_{self.target_layers}.nii.gz'
                        )

            elif len(origin_img.shape[1:]) == 2:
                cam_result = np.uint8(cam_result.squeeze(1) * 255)
                for j, (img_slice, cam_slice) in enumerate(zip(origin_img, cam_result)):
                    img_slice = np.uint8(Normalize2(img_slice) * 255)

                    img_slice = Image.fromarray(img_slice)
                    no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(img_slice, cam_slice, 'hsv')

                    heatmap_on_image.save(self.save_dir/f'batch{i}_{j}_heatmap_on_img.png')
            else:
                raise NotImplementedError(f"Cannot support ({origin_img.shape}) data.")

        engine.terminate()


class TensorboardDumper:
    """
    Dumper Tensorboard content to local plot and images.
    """
    def __init__(
        self,
        log_dir: Union[Path, str],
        epoch_level: bool = True,
        interval: int = 1,
        save_image: bool = False,
        logger_name: Optional[str] = None,
    ):
        if save_image:
            raise NotImplementedError("Save image not supported yet.")

        if not os.path.isdir(log_dir):
            raise FileNotFoundError(f'{log_dir} not exists!')

        self.log_dir = log_dir
        self.epoch_level = epoch_level
        self.interval = interval
        self.logger = logging.getLogger(logger_name)
        self.db_file = None

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine) -> None:
        if self.db_file is None:
            files = os.listdir(self.log_dir)
            if len(files) == 0:
                self.logger.warn(f'No tensorboard db is found in the dir({self.log_dir})')
                return
            elif len(files) > 1:
                self.logger.warn(f'Multiple tensorboard db files are found! Skip dumping.')
                return
            else:
                self.db_file = os.path.join(self.log_dir, files[0])

        summary = dump_tensorboard(self.db_file, dump_keys=None, save_image=False)
        summary = dict(filter(lambda x: 'loss' not in x[0] and 'Learning_rate' not in x[0], summary.items()))
        plot_summary(summary, output_fpath=os.path.join(self.log_dir, 'metric_summary.png'))
