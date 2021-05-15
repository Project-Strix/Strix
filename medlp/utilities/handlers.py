import os
from pathlib import Path
import struct
import logging
from typing import TYPE_CHECKING, Optional, Callable, Union

from monai_ex.utils import exact_version, optional_import
from medlp.models.cnn.layers.snip import SNIP, apply_prune_mask
from medlp.utilities.utils import dump_tensorboard, plot_summary
# from medlp.utilities.utils import add_3D_overlay_to_summary


Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
Metric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "Metric")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.4", exact_version, "Checkpoint")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.4", exact_version, "reinit__is_reduced")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")

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
