import logging
from logging import Logger
from typing import TYPE_CHECKING, Dict, Optional, Any, Callable
import numpy as np

from monai.utils.module import export
from monai.utils import exact_version, optional_import
from monai.handlers import TensorBoardImageHandler
from monai.visualize import plot_2d_or_3d_image

import torch
from medlp.models.cnn.layers.snip import SNIP, apply_prune_mask

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "Checkpoint")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")

NNi, _ = optional_import("nni")
Torchviz, _ = optional_import('torchviz')

class NNIReporter:
    """
    NNIReporter 

    Args:

    """
    def __init__(self, 
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
        #assert self.metric_name in engine.state.metrics.keys(), f"{self.metric_name} is not in engine's metrics: {engine.state.metrics.keys()}"
        print('----------keys-----------',engine.state.metrics.keys())
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
    def __init__(self, 
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
                 loss_fn,
                 prune_percent,
                 data_loader,
                 device='cuda',
                 snip_device = 'cpu',
                 verbose = False,
                 logger_name: Optional[str] = None
    ) -> None:
        self.net = net
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
        keep_masks = SNIP(self.net, self.loss_fn, self.prune_percent, self.data_loader, self.snip_device, None)
        net_ = apply_prune_mask(self.net, keep_masks, self.device, self.verbose)
        # self.net.load_state_dict(net_.state_dict())


class TensorBoardImageHandlerEx(TensorBoardImageHandler):
    def __init__(        
        self,
        summary_writer = None,
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        max_frames: int = 64,
        prefix_name: str = ''):
        super().__init__(
            summary_writer=summary_writer,
            log_dir=log_dir,
            interval=interval,
            epoch_level=epoch_level,
            batch_transform=batch_transform,
            output_transform=output_transform,
            global_iter_transform=global_iter_transform,
            index=index,
            max_channels=max_channels,
            max_frames=max_frames
        )
        self.prefix_name = prefix_name
    
    def __call__(self, engine: Engine):
        step = self.global_iter_transform(engine.state.epoch if self.epoch_level else engine.state.iteration)
        show_images = self.batch_transform(engine.state.batch)[0]
        if torch.is_tensor(show_images):
            show_images = show_images.detach().cpu().numpy()
        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise ValueError("output_transform(engine.state.output)[0] must be an ndarray or tensor.")
            plot_2d_or_3d_image(
                show_images, step, self._writer, self.index, self.max_channels, self.max_frames, self.prefix_name+"/input_0"
            )

        show_labels = self.batch_transform(engine.state.batch)[1]
        if torch.is_tensor(show_labels):
            show_labels = show_labels.detach().cpu().numpy()
        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise ValueError("batch_transform(engine.state.batch)[1] must be an ndarray or tensor.")
            plot_2d_or_3d_image(
                show_labels, step, self._writer, self.index, self.max_channels, self.max_frames, self.prefix_name+"/input_1"
            )

        show_outputs = self.output_transform(engine.state.output)
        if torch.is_tensor(show_outputs):
            show_outputs = show_outputs.detach().cpu().numpy()
        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise ValueError("output_transform(engine.state.output) must be an ndarray or tensor.")
            plot_2d_or_3d_image(
                show_outputs, step, self._writer, self.index, self.max_channels, self.max_frames, self.prefix_name+"/output"
            )

        self._writer.flush()

class TensorboardGraph:
    """
    TensorboardGraph for visualize network architecture using tensorboard
    """
    def __init__(self,
                 net,
                 writer,
                 output_transform: Callable = lambda x: x,
                 logger_name: Optional[str] = None
    ) -> None:
        self.net = net
        self.writer = writer
        self.output_transform = output_transform
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
    
    def attach(self, engine: Engine) -> None:
        if self.logger_name is None:
            self.logger = engine.logger
        
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        inputs = self.output_transform(engine.state.output)
        if inputs is not None:
            try:
                self.writer.add_graph(self.net, inputs[0:1,...], False)
            except Exception as e:
                self.logger.error(f'Error occurred when adding graph to tensorboard: {e}')
        else:
            self.logger.warn('No inputs are found! Skip adding graph!')

class TorchVisualizer:
    """
    TorchVisualizer for visualize network architecture using PyTorchViz.
    """
    def __init__(self,
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


        