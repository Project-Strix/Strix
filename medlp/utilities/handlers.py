import logging
import bisect
from logging import Logger
import re
from typing import TYPE_CHECKING, Dict, Optional, Any, Callable, List, Sequence, Union
from click.core import Option
import numpy as np
from sklearn.metrics import roc_curve, auc

from monai_ex.utils import export, exact_version, optional_import
from monai_ex.handlers import TensorBoardImageHandler
from monai_ex.visualize import plot_2d_or_3d_image
from monai_ex.handlers import CSVSaverEx
from monai.engines.evaluator import Evaluator

import torch
from torch._C import device
from medlp.models.cnn.layers.snip import SNIP, apply_prune_mask
from medlp.utilities.utils import add_3D_overlay_to_summary

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "Checkpoint")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
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


class AUCGapHandler:
    """
    Attach validator to the trainer engine in Ignite.
    It can support to execute validation every N epochs or every N iterations.

    """

    def __init__(
        self,
        validator: Evaluator,
        interval: int,
        epoch_level: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
        logger_name: Optional[str] = None,
        save_metric: bool = False,
        save_metric_name: str = 'rect_auc'
    ) -> None:
        """
        Args:
            validator: run the validator when trigger validation, suppose to be Evaluator.
            interval: do validation every N epochs or every N iterations during training.
            epoch_level: execute validation every N epochs or N iterations.
                `True` is epoch level, `False` is iteration level.

        Raises:
            TypeError: When ``validator`` is not a ``monai.engines.evaluator.Evaluator``.

        """
        if not isinstance(validator, Evaluator):
            raise TypeError(f"validator must be a monai.engines.evaluator.Evaluator but is {type(validator).__name__}.")
        self.validator = validator
        self.interval = interval
        self.epoch_level = epoch_level
        self.summary_writer = summary_writer
        self.logger_name = logger_name
        self.save_metric = save_metric
        self.metric_name = save_metric_name
        if self.summary_writer is None and self.logger_name is None:
            raise ValueError("Either tensorboard summary writer or logger should be specified!")
        if logger_name is not None:
            self.logger = logging.getLogger(logger_name)

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
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if hasattr(engine.state, "key_metric_name") and isinstance(engine.state.key_metric_name, str):
            train_metric = engine.state.metrics[engine.state.key_metric_name]
        else:
            raise ValueError(f"Cannot find key_metric_name for AUC. Only find {engine.state.metrics.keys()}")

        if hasattr(self.validator.state, "key_metric_name") and isinstance(self.validator.state.key_metric_name, str):
            valid_metric = self.validator.state.metrics[self.validator.state.key_metric_name]
        else:
            raise ValueError(f"Cannot find key_metric_name for AUC. Only find {self.validator.state.metrics.keys()}")

        auc_gap = float(train_metric) - float(valid_metric)
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('AUC Gap', auc_gap, engine.state.epoch)
            # self.summary_writer.add_scalar('Rectified AUC', valid_metric-abs(auc_gap), engine.state.epoch)
        if self.logger_name is not None:
            self.logger.info(f"AUC Gap: {auc_gap}")

        if self.save_metric:
            engine.state.metrics[self.metric_name] = valid_metric-abs(auc_gap)


class ROCOverlapHandler:
    """ROCOverlapHandler

    Log the overlapping between training and validation ROC.
    """
    def __init__(
        self,
        validator: Evaluator,
        interval: int,
        output_transform: Callable = lambda x: (x["pred"], x['label']),
        roc_metric_name: Optional[str] = "train_roc",
        epoch_level: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
        logger_name: Optional[str] = None,
        save_metric: bool = False,
        save_metric_name: str = 'rect_roc',
    ) -> None:
        if not isinstance(validator, Evaluator):
            raise TypeError(f"validator must be a monai.engines.evaluator.Evaluator but is {type(validator).__name__}.")
        self.validator = validator
        self.interval = interval
        self.output_transform = output_transform
        self.roc_metric_name = roc_metric_name
        self.epoch_level = epoch_level
        self.summary_writer = summary_writer
        self.logger_name = logger_name
        self.save_metric = save_metric
        self.metric_name = save_metric_name
        if self.summary_writer is None and self.logger_name is None:
            raise ValueError("Either tensorboard summary writer or logger should be specified!")
        if logger_name is not None:
            self.logger = logging.getLogger(logger_name)

        self._train_preditions = []
        self._train_groundtruth = []
        self._valid_preditions = []
        self._valid_groundtruth = []

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
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if hasattr(engine.state, "key_metric_name") and isinstance(engine.state.key_metric_name, str):
            train_metric = engine.state.metrics[engine.state.key_metric_name]
        else:
            raise ValueError(f"Cannot find key_metric_name for AUC. Only find {engine.state.metrics.keys()}")

        if hasattr(self.validator.state, "key_metric_name") and isinstance(self.validator.state.key_metric_name, str):
            valid_metric = self.validator.state.metrics[self.validator.state.key_metric_name]
        else:
            raise ValueError(f"Cannot find key_metric_name for AUC. Only find {self.validator.state.metrics.keys()}")

        if engine.state.metrics.get(self.roc_metric_name):
            train_roc = engine.state.metrics[self.roc_metric_name]
        else:
            raise ValueError(f"Cannot find metric ({self.roc_metric_name}) for ROC. Only find {engine.state.metrics.keys()}")

        if self.validator.state.metrics.get(self.roc_metric_name):
            valid_roc = self.validator.state.metrics[self.roc_metric_name]
        else:
            raise ValueError(f"Cannot find metric ({self.roc_metric_name}) for ROC. Only find {self.validator.state.metrics.keys()}")

        train_auc, valid_auc = float(train_metric), float(valid_metric)

        train_fpr, train_tpr, _ = train_roc
        valid_fpr, valid_tpr, _ = valid_roc

        def extract_fp2tp_dict(fpr_, tpr_):
            fp2tp_dict = {}
            for i, (fp, tp) in enumerate(zip(fpr_, tpr_)):
                if fp2tp_dict.get(fp):
                    fp2tp_dict[fp].append(tp)
                else:
                    fp2tp_dict[fp] = [tp]
            return fp2tp_dict

        def fill_none_values(fp2tp, fp_idx, all_fps):
            for fp in all_fps:
                if not fp2tp.get(fp):
                    idx_l = bisect.bisect_left(fp_idx, fp)-1
                    idx_r = bisect.bisect_right(fp_idx, fp)
                    try:
                        left_fp, right_fp = fp_idx[idx_l], fp_idx[idx_r]
                    except:
                        print('fp_idx:', fp_idx, 'fp:', fp, 'idx lr:', idx_l, idx_r)
                        raise

                    fp2tp[fp] = [(fp2tp[left_fp][-1]+fp2tp[right_fp][0])/2]
            return fp2tp

        train_fp2tp = extract_fp2tp_dict(train_fpr, train_tpr)
        valid_fp2tp = extract_fp2tp_dict(valid_fpr, valid_tpr)

        train_fps, valid_fps = list(train_fp2tp.keys()), list(valid_fp2tp.keys())
        fps = np.sort(np.unique(np.concatenate([train_fps, valid_fps])))

        train_fp2tp = fill_none_values(train_fp2tp, train_fps, fps)
        valid_fp2tp = fill_none_values(valid_fp2tp, valid_fps, fps)

        final_curve = {}
        for i, fp in enumerate(fps):
            valid_tps = valid_fp2tp[fp] if len(valid_fp2tp[fp]) <= 2 \
                else [min(valid_fp2tp[fp]), max(valid_fp2tp[fp])]
            train_tps = train_fp2tp[fp] if len(train_fp2tp[fp]) <= 2 \
                else [min(train_fp2tp[fp]), max(train_fp2tp[fp])]

            if max(valid_tps) < min(train_tps) or max(train_tps) < min(valid_tps): #not intersection
                last_value = max(final_curve[fps[i-1]])
                if last_value in valid_tps:
                    final_curve[fp] = valid_tps
                elif last_value in train_tps:
                    final_curve[fp] = train_tps
                else:
                    final_curve[fp] = valid_tps if np.mean(valid_tps)<np.mean(train_tps) else train_tps
            else:
                if len(valid_tps+train_tps) == 4 and len(np.unique(valid_tps+train_tps)) < 4:
                    final_curve[fp] = np.sort(np.unique(valid_tps+train_tps))[:2]
                elif len(valid_tps+train_tps) == 3:
                    final_curve[fp] = np.unique(np.sort(valid_tps+train_tps)[:2])
                else:
                    final_curve[fp] = valid_tps if np.mean(valid_tps)<np.mean(train_tps) else train_tps

        # restore to fpr, tpr points
        fpr, tpr = [], []
        for fp, tps in final_curve.items():
            for tp in tps:
                fpr.append(fp)
                tpr.append(tp)

        roc_dice = 2*auc(fpr, tpr)/(train_auc+valid_auc)

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('ROC Dice', roc_dice, engine.state.epoch)
        if self.logger_name is not None:
            self.logger.info(f"ROC Dice: {roc_dice}")

        if self.save_metric:
            engine.state.metrics[self.metric_name] = valid_metric*roc_dice
