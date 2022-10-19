import csv
import os
import warnings
from pathlib import Path
from termcolor import colored

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from strix.utilities.registry import DatasetRegistry
from strix.utilities.click import NumericChoice as Choice
from strix.utilities.arguments import data_select
from strix.utilities.enum import FRAMEWORKS, Phases
from strix.utilities.utils import get_torch_datast, get_items
from strix.configures import config as cfg
from monai_ex.data import DatasetSummaryEx
from monai_ex.utils.misc import Print


@click.command("merge-roc")
@click.option("--exp-dir", type=click.Path(exists=True), help="Dir contains test folders")
@click.option("--dirname-as-legend", type=bool, default=True, help="Use dirname as legend label")
def merge_roc_curves(exp_dir, dirname_as_legend):
    """Utility for mergeing multiple ROC curves.

    Args:
        root_dir (str): Root dir contains test folders.
        dirname-as-legend (bool): Use dirname as legend label
    """
    roc_files = list(Path(exp_dir).rglob("roc_scores.csv"))
    if len(roc_files) <= 1:
        Print(
            f"Found {len(roc_files)} ROC file, at least 2 files are need. Return!",
            color="y",
        )
        return

    legend_names = []
    auc_values = []
    for roc_file in roc_files:
        if dirname_as_legend:
            legend_name = roc_file.parent.name
        else:
            test_cohort_name = list(roc_file.parent.glob("*_files.*"))
            if len(test_cohort_name) > 0:
                try:
                    cohort_name = test_cohort_name[0].stem.split("_")[0]
                except:
                    cohort_name = None
            else:
                cohort_name = None

            legend_name = click.prompt(
                f"Input legend for {roc_file.parent.name}",
                type=str,
                default=cohort_name,
            )
        legend_names.append(legend_name)

        cls_result_file = roc_file.parent.joinpath("classification_results.json")
        if cls_result_file.is_file():
            auc_value = get_items(cls_result_file).get("AUC", -1)
            auc_values.append(auc_value)
        else:
            auc_value = click.prompt(f"Input ROC value for {roc_file.parent.name}", type=float)
            auc_values.append(auc_value)

    # Draw ROC
    fig = plt.figure(figsize=(8, 5), dpi=200)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 9

    for auc, roc, legend in sorted(zip(auc_values, roc_files, legend_names)):
        with open(roc) as f:
            reader = csv.DictReader(f)
            fpr_tpr = np.array([[float(row["FPR"]), float(row["TPR"])] for row in reader])
        plt.plot(fpr_tpr[:, 0], fpr_tpr[:, 1], label=f"{legend} AUC={auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig(os.path.join(exp_dir, "merged_roc.png"))
    Print("ROC is saved to", os.path.join(exp_dir, "merged_roc.png"), color="g")


options = ["Spacing", "Statistics", "Percentiles"]


@click.command("summarize-data")
@click.option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]), default="2D", help="2D or 3D")
@click.option("--framework", prompt=True, type=Choice(FRAMEWORKS), default="segmentation", help="Choose framework")
@click.option("--data-list", type=str, callback=data_select, default=None, help="Data file list")
@click.option("--skip", "-s", prompt=True, prompt_required=False, type=Choice(options), default=None)
def summarize_data(tensor_dim, framework, data_list, skip):
    datasets = DatasetRegistry()
    strix_dataset = datasets.get(tensor_dim, framework, data_list)
    if strix_dataset is None:
        warnings.warn(colored(f"Dataset {data_list} is not found!", "red"))
        return

    torch_dataset = get_torch_datast(strix_dataset, Phases.TRAIN, {"preload": 0})
    if torch_dataset is None:
        warnings.warn(colored("Get torch dataset failed! Skip summarize!", "red"))
        return

    analyzer = DatasetSummaryEx(
        dataset=torch_dataset,
        image_key=cfg.get_key("image"),
        label_key=cfg.get_key("label"),
        num_workers=2,  #! os.cpu_count()
        select_transforms=1,
    )

    if skip is None or "Spacing" not in skip:
        Print("Begin analysis spacing...", color="g")
        spacing_summary = analyzer.get_target_spacing(
            spacing_key="pixdim",
            anisotropic_threshold=3,
            percentile=10,
        )
        print("=> Spacing:", spacing_summary)

    if skip is None or "Statistics" not in skip:
        Print("Begin analysis statistics...", color="g")
        analyzer.calculate_statistics(foreground_threshold=0)
        print(
            f"=> Statistics:\n"
            f"\tdata_max: {analyzer.data_max}\n"
            f"\tdata_min: {analyzer.data_min}\n"
            f"\tdata_mean: {analyzer.data_mean}\n"
            f"\tdata_std: {analyzer.data_std}\n"
        )

    if skip is None or "Percentiles" not in skip:
        Print("Begin analysis percentiles...", color="g")
        analyzer.calculate_percentiles(
            foreground_threshold=0,
            sampling_flag=True,
            interval=10,
            min_percentile=0.5,
            max_percentile=99.5,
        )
        print(
            f"=> Percentiles:\n",
            f"\tdata_min_percentile: {analyzer.data_min_percentile}"
            f"\tdata_max_percentile: {analyzer.data_max_percentile}"
            f"\tdata_median: {analyzer.data_median}"
        )
