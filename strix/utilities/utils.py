import os
import copy
import json
import socket
import struct
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, TextIO, Union, List
import matplotlib
import pylab
import torch
import yaml
from PIL import Image
from termcolor import colored
import logging

matplotlib.use("Agg")
import matplotlib.cm as mpl_color_map
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import tensorboard.compat.proto.event_pb2 as event_pb2
from matplotlib.ticker import ScalarFormatter
from monai_ex.utils import GenericException, catch_exception, ensure_list
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from strix.utilities.enum import LR_SCHEDULES, Phases, SerialFileFormat

trycatch = partial(catch_exception, handled_exception_type=GenericException, path_keywords="strix")


@trycatch()
def get_items(
    filelist: Union[str, Path], format: Optional[SerialFileFormat] = None, allow_filenotfound: bool = False
) -> Any:
    """Get items from given serialized file. Support both json and yaml.

    Args:
        filelist (Union[str, Path]): Input file path.
        format (Optional[SerialFileFormat]): two formats are supported: SerialFileFormat.JSON or SerialFileFormat.YAML.
        allow_filenotfound (bool, optional): donot throw error even if file is not found. Defaults to False.

    Raises:
        FileNotFoundError: filelist not found.
        ValueError: unknown file format is given when format is not specified
        GenericException: json.JSONDecodeError and yaml.YAMLError
        GenericException: data path not exists

    Returns:
        _type_: _description_
    """
    try:
        filelist = Path(filelist)
        if not filelist.is_file():
            raise FileNotFoundError(f"No such file: {filelist}")

        if not format:
            if filelist.suffix in [".json"]:
                format = SerialFileFormat.JSON
            elif filelist.suffix in [".yaml", ".yml"]:
                format = SerialFileFormat.YAML
            else:
                raise ValueError(f"Not supported file format {filelist.suffix}")

        with filelist.open() as f:
            if format == SerialFileFormat.YAML:
                lines = yaml.full_load(f)
            elif format == SerialFileFormat.JSON:
                lines = json.load(f)

        return lines
    except json.JSONDecodeError as e:
        raise GenericException("Content of your json file cannot be parsed. Please recheck it!")
    except yaml.YAMLError as e:
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark  # type: ignore
        raise GenericException(
            f"Content of your yaml file cannot be parsed. Please recheck it!\n"
            f"Error parsing at line {mark.line}, column {mark.column+1}."
        )
    except TypeError as e:
        raise GenericException("Your data path not exists! Please recheck!") from e
    except FileNotFoundError as e:
        if allow_filenotfound:
            return None
        else:
            raise GenericException(f"File not found! {filelist}")


def split_train_test(
    datalist: list,
    test_ratio: float,
    label_key: str = "label",
    splits_num: int = 1,
    output_dir: Optional[Path] = None,
    output_prefix: Optional[str] = "",
    stratify: bool = True,
    random_seed: int = 0,
):
    Split = StratifiedShuffleSplit if stratify else ShuffleSplit

    if test_ratio <= 0 or 1 <= test_ratio:
        raise ValueError(f"Expect: 0 < Test ratio < 1, but got {test_ratio}")

    if isinstance(datalist[0][label_key], (list, tuple)):
        Ys = np.array([np.mean(data[label_key]) for data in datalist])
    else:
        Ys = np.array([int(data[label_key]) for data in datalist])

    output = []
    spliter = Split(n_splits=splits_num, test_size=test_ratio, random_state=random_seed)
    for i, (train_idx, test_idx) in enumerate(spliter.split(np.zeros(len(Ys)), Ys)):
        train_cohort = [datalist[idx] for idx in train_idx]
        test_cohort = [datalist[idx] for idx in test_idx]
        output.append((train_cohort, test_cohort))

        if output_dir and output_dir.is_dir():
            with open(output_dir / f"{output_prefix}_train_cohort{i}_{test_ratio}.json", "w") as f:
                json.dump(train_cohort, f, indent=2)
            with open(output_dir / f"{output_prefix}_test_cohort{i}_{test_ratio}.json", "w") as f:
                json.dump(test_cohort, f, indent=2)

    return output


def generate_synthetic_datalist(data_num: int = 100, logger=None):
    if logger:
        logger.info(" ==== Using synthetic data ====")
    train_datalist = [
        {"image": f"synthetic_image{i}.nii.gz", "label": f"synthetic_label{i}.nii.gz"} for i in range(data_num)
    ]
    return train_datalist


def get_attr_(obj, name, default=None):
    return getattr(obj, name) if hasattr(obj, name) else default


def get_colors(num: Optional[int] = None):
    if num:
        return list(mcolors.TABLEAU_COLORS.values())[:num]  # type: ignore
    else:
        return list(mcolors.TABLEAU_COLORS.values())  # type: ignore


def get_colormaps(num: Optional[int] = None):
    if num is None:
        num = 10
    if num == 1:
        return "tab10"
    else:
        colormap = [list(plt.cm.tab10(0.1 * i)) for i in range(num)]
        new_colormap = LinearSegmentedColormap.from_list("my_colormap", colormap, N=num + 1 if num == 1 else num)
        return new_colormap


def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def bbox_2D(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def create_rgb_summary(label):
    num_colors = label.shape[1]

    cm = pylab.get_cmap("gist_rainbow")

    new_label = np.zeros((label.shape[0], label.shape[1], label.shape[2], 3), dtype=np.float32)

    for i in range(num_colors):
        color = cm(1.0 * i / num_colors)  # color will now be an RGBA tuple
        new_label[:, :, :, 0] += label[:, :, :, i] * color[0]
        new_label[:, :, :, 1] += label[:, :, :, i] * color[1]
        new_label[:, :, :, 2] += label[:, :, :, i] * color[2]

    return new_label


def add_3D_overlay_to_summary(
    data: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    writer,
    index: int = 0,
    tag: str = "output",
    centers=None,
):
    data_ = data[index].detach().cpu().numpy() if torch.is_tensor(data) else data[index]
    mask_ = mask[index].detach().cpu().numpy() if torch.is_tensor(mask) else mask[index]
    # binary_volume = np.squeeze(binary_volume)
    # volume_overlay = np.squeeze(volume_overlay)

    if mask_.shape[1] > 1:
        # there are channels
        mask_ = create_rgb_summary(mask_)
        data_ = data_[..., np.newaxis]

    else:
        data_, mask_ = data_[..., np.newaxis], mask_[..., np.newaxis]

    if centers is None:
        center_x = np.argmax(
            np.sum(
                np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=1,
                keepdims=True,
            ),
            axis=0,
        )
        center_y = np.argmax(
            np.sum(
                np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=1,
        )
        center_z = np.argmax(
            np.sum(
                np.sum(np.sum(mask_, axis=3, keepdims=True), axis=1, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=2,
        )
    else:
        center_x, center_y, center_z = centers

    segmentation_overlay_x = np.squeeze(data_[center_x, :, :, :] + mask_[center_x, :, :, :])
    segmentation_overlay_y = np.squeeze(data_[:, center_y, :, :] + mask_[:, center_y, :, :])
    segmentation_overlay_z = np.squeeze(data_[:, :, center_z, :] + mask_[:, :, center_z, :])

    if len(segmentation_overlay_x.shape) != 3:
        segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = (
            segmentation_overlay_x[..., np.newaxis],
            segmentation_overlay_y[..., np.newaxis],
            segmentation_overlay_z[..., np.newaxis],
        )

    writer.add_image(tag + "_x", segmentation_overlay_x)
    writer.add_image(tag + "_y", segmentation_overlay_y)
    writer.add_image(tag + "_z", segmentation_overlay_z)


def add_3D_image_to_summary(manager, image, name, centers=None):
    image = np.squeeze(image)

    if len(image.shape) > 3:
        # there are channels
        print("add_3D_image_to_summary: there are channels")
        image = create_rgb_summary(image)
    else:
        image = image[..., np.newaxis]

    if centers is None:
        center_x = np.argmax(
            np.sum(
                np.sum(np.sum(image, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=1,
                keepdims=True,
            ),
            axis=0,
        )
        center_y = np.argmax(
            np.sum(
                np.sum(np.sum(image, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=1,
        )
        center_z = np.argmax(
            np.sum(
                np.sum(np.sum(image, axis=3, keepdims=True), axis=1, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=2,
        )
    else:
        center_x, center_y, center_z = centers

    segmentation_overlay_x = np.squeeze(image[center_x, :, :, :])
    segmentation_overlay_y = np.squeeze(image[:, center_y, :, :])
    segmentation_overlay_z = np.squeeze(image[:, :, center_z, :])

    if len(segmentation_overlay_x.shape) != 3:
        segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = (
            segmentation_overlay_x[..., np.newaxis],
            segmentation_overlay_y[..., np.newaxis],
            segmentation_overlay_z[..., np.newaxis],
        )

    manager.add_image(name + "_x", segmentation_overlay_x)
    manager.add_image(name + "_y", segmentation_overlay_y)
    manager.add_image(name + "_z", segmentation_overlay_z)


# todo: check through all data
def output_filename_check(torch_dataset, meta_key="image_meta_dict"):
    if len(torch_dataset) == 1:
        data = torch_dataset[0]
        if isinstance(data, list):
            data = data[0]
        return Path(data[meta_key]["filename_or_obj"]).parent.parent

    prev_data = torch_dataset[0]
    next_data = torch_dataset[1]

    if isinstance(prev_data, list):
        prev_data = prev_data[0]
    if isinstance(next_data, list):
        next_data = next_data[0]

    prev_parents = list(Path(prev_data[meta_key]["filename_or_obj"]).parents)[::-1]
    next_parents = list(Path(next_data[meta_key]["filename_or_obj"]).parents)[::-1]
    for (prev_item, next_item) in zip(prev_parents, next_parents):
        if prev_item.stem != next_item.stem:
            return prev_item.parent

    return ""


def detect_port(port):
    """Detect if the port is used"""
    socket_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_test.connect(("127.0.0.1", int(port)))
        socket_test.close()
        return True
    except:
        return False


def parse_nested_data(data):
    params = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value_ = value.copy()
            if key == "lr_policy":
                policy_name = value_.get("_name", None)
                if policy_name in LR_SCHEDULES:
                    params[key] = policy_name
                    value_.pop("_name")
                    params["lr_policy_params"] = value_
            else:
                raise NotImplementedError(f"{key} is not supported for nested params.")
        else:
            params[key] = value
    return params


def is_avaible_size(value):
    if isinstance(value, (list, tuple)):
        if np.all(np.greater(value, 0)):
            return True
    return False


def plot_summary(summary, output_fpath):
    try:
        f = plt.figure(1)
        plt.clf()
        colors = get_colors()

        for i, (key, step_value) in enumerate(summary.items()):
            # print('Key:', key, type(key), "step_value:", step_value['values'])
            plt.plot(
                step_value["steps"],
                step_value["values"],
                label=str(key),
                color=colors[i],
                linewidth=2.0,
            )

        # plt.ylim([0., 1.])
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(30))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.xlabel("Number of iterations per case")
        plt.grid(True)
        plt.legend(list(summary.keys()))
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
        f.show()

        f.savefig(output_fpath)
    except Exception as e:
        print("Failed to do plot: " + str(e))


def dump_tensorboard(db_file, dump_keys=None, save_image=False, verbose=False):
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f"db_file is not found: {db_file}")

    if dump_keys is not None:
        dump_keys = ensure_list(dump_keys)

    def _read(input_data):
        header = struct.unpack("Q", input_data[:8])
        # crc_hdr = struct.unpack('I', input_data[:4])
        eventstr = input_data[12 : 12 + int(header[0])]  # 8+4
        out_data = input_data[12 + int(header[0]) + 4 :]
        return out_data, eventstr

    with open(db_file, "rb") as f:
        data = f.read()

    summaries = {}
    while data:
        data, event_str = _read(data)
        event = event_pb2.Event()
        event.ParseFromString(event_str)
        if event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    if dump_keys is None or value.tag in dump_keys:
                        if not summaries.get(value.tag, None):
                            summaries[value.tag] = {
                                "steps": [event.step],
                                "values": [value.simple_value],
                            }
                        else:
                            summaries[value.tag]["steps"].append(event.step)
                            summaries[value.tag]["values"].append(value.simple_value)
                        if verbose:
                            print(value.simple_value, value.tag, event.step)
                if value.HasField("image") and save_image:
                    img = value.image
                    # save_img(img.encoded_image_string, event.step, save_gif=args.gif)
    return summaries


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img


def norm_tensor(t, range):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))


def get_bound_2d(mask, connectivity=1):
    from joblib import Parallel, delayed

    if connectivity == 1:
        offset = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)]
    elif connectivity == 2:
        offset = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (0, 1, 1), (0, 1, -1), (0, -1, -1), (0, -1, 1)]
    else:
        raise ValueError(f"Connectivity should be 1 or 2, but got {connectivity}")

    def is_bound(data, coord, offsets):
        zeros = []
        for off in offsets:
            pt = tuple(coord + torch.tensor(off))
            try:
                zeros.append(data[pt] == 0)
            except:
                zeros.append(True)
        return (coord, np.any(zeros))

    boundaries = []
    if mask.ndim == 3 and mask.shape[0] > 1:
        for channel in mask[1:, ...]:
            coords = torch.nonzero(channel[None])
            if len(coords) == 0:
                continue
            bounds = Parallel(n_jobs=5)(delayed(is_bound)(channel[None], c, offset) for c in coords)
            boundary = list(filter(lambda x: x[1], bounds))
            boundaries.append(list(map(lambda x: x[0], boundary)))
    else:
        labels = np.unique(mask[mask > 0])
        for label in labels:
            coords = torch.nonzero(mask == label)
            bounds = Parallel(n_jobs=5)(delayed(is_bound)(mask, c, offset) for c in coords)
            boundary = list(filter(lambda x: x[1], bounds))
            boundaries.append(list(map(lambda x: x[0], boundary)))
    return boundaries



class LogColorFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt=None):
        super().__init__()
        self.fmt = fmt if fmt else "%(asctime)s %(name)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(
    name: Optional[str] = "ignite",
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)",
    color: bool = True,
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
    reset: bool = False,
    terminator: str = "\n",
) -> logging.Logger:
    """
    Extented ignite's setup_logger.
    Extended: `color`, `terminator`.

    Setups logger: name, level, format etc.

    Args:
        name: new name for the logger. If None, the standard logger is used.
        level: logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG.
        stream: logging stream. If None, the standard stream is used (sys.stderr).
        format: logging format. By default, `%(asctime)s %(name)s %(levelname)s: %(message)s`.
        color: whether use colored log
        filepath: Optional logging file path. If not None, logs are written to the file.
        distributed_rank: Optional, rank in distributed configuration to avoid logger setup for workers.
            If None, distributed_rank is initialized to the rank of process.
        reset: if True, reset an existing logger rather than keep format, handlers, and level.
        terminator: change the terminator of output stream. eg. `\r` for concise msg

    Returns:
        logging.Logger

    Examples:
        Improve logs readability when training with a trainer and evaluator:

        .. code-block:: python

            from ignite.utils import setup_logger

            trainer = ...
            evaluator = ...

            trainer.logger = setup_logger("trainer")
            evaluator.logger = setup_logger("evaluator")

            trainer.run(data, max_epochs=10)

            # Logs will look like
            # 2020-01-21 12:46:07,356 trainer INFO: Engine run starting with max_epochs=5.
            # 2020-01-21 12:46:07,358 trainer INFO: Epoch[1] Complete. Time taken: 00:5:23
            # 2020-01-21 12:46:07,358 evaluator INFO: Engine run starting with max_epochs=1.
            # 2020-01-21 12:46:07,358 evaluator INFO: Epoch[1] Complete. Time taken: 00:01:02
            # ...

        Every existing logger can be reset if needed

        .. code-block:: python

            logger = setup_logger(name="my-logger", format="=== %(name)s %(message)s")
            logger.info("first message")
            setup_logger(name="my-logger", format="+++ %(name)s %(message)s", reset=True)
            logger.info("second message")

            # Logs will look like
            # === my-logger first message
            # +++ my-logger second message

        Change the level of an existing internal logger

        .. code-block:: python

            setup_logger(
                name="ignite.distributed.launcher.Parallel",
                level=logging.WARNING
            )
    """
    # check if the logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # if existing, get the logger otherwise create a new one
    logger = logging.getLogger(name)

    if distributed_rank is None:
        import ignite.distributed as idist

        distributed_rank = idist.get_rank()

    # Remove previous handlers
    if distributed_rank > 0 or reset:

        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    if distributed_rank > 0:

        # Add null handler to avoid multiple parallel messages
        logger.addHandler(logging.NullHandler())

    # Keep the existing configuration if not reset
    if existing and not reset:
        return logger

    if distributed_rank == 0:
        logger.setLevel(level)

        formatter = LogColorFormatter(format) if color else logging.Formatter(format)

        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        ch.terminator = terminator
        logger.addHandler(ch)

        if filepath is not None:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(format))  # file no color
            logger.addHandler(fh)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    return logger


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s:\n %s: %s\n" % (filename, lineno, category.__name__, message)


def singleton(cls):
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def save_sourcecode(code_rootdir, out_dir, verbose=True):
    if not os.path.isdir(code_rootdir):
        raise FileNotFoundError(f"Code root dir not exists! {code_rootdir}")

    if verbose:
        print(colored(f"Backup source code under root_dir: {code_rootdir}", color="yellow"))

    outpath = out_dir / f"{os.path.basename(code_rootdir)}_{time.strftime('%m%d_%H%M')}.tar"
    tar_opt = "cvf" if verbose else "cf"
    os.system(f"cd {str(code_rootdir)}; tar -{tar_opt} {outpath} .")


def get_torch_datast(
    strix_dataset, phase: Phases, opts: dict, synthetic_data_num=100, split_func: Optional[Callable] = None
):
    """This function return pytorch dataset generated by registered strix dataset.

    Args:
        strix_dataset (dict): strix dastset got from DatasetRegistry
        phase (Phases): specific phase
        opts (dict): hyper-parameters pass to dataset
        synthetic_data_num (int): if `strix_dataset["PATH"]` is not given, generate some dummy file path
        split_func (callable): function provided to split data list
    Returns:
        dict: pytorch dataset
    """
    try:
        dataset_fn, dataset_list = strix_dataset["FN"], strix_dataset["PATH"]
        if dataset_list:
            filelist = get_items(dataset_list)
        else:
            filelist = generate_synthetic_datalist(synthetic_data_num)

        if split_func is not None:
            train_flist, valid_flist = split_func(filelist)
            if phase == Phases.TRAIN:
                filelist = train_flist
            elif phase == Phases.VALID:
                filelist = valid_flist
            elif phase == Phases.TEST_EX or phase == Phases.TEST_IN:
                filelist = strix_dataset.get("TEST_PATH")
                if not filelist:
                    raise FileNotFoundError("No test path is specified during registering")
        torch_dataset = dataset_fn(filelist, phase, opts)
    except KeyError as e:
        warnings.warn(colored(f"Dataset registered error!\nErr msg: {repr(e)}\n{e.__cause__}", "red"))
        return None
    except Exception as e:
        warnings.warn(colored(f"Creating dataset '{strix_dataset}' failed! \nMsg: {repr(e)}", "red"))
        return None
    else:
        return torch_dataset
