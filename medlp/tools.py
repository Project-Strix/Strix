import os
import csv
from pathlib import Path
import click
from utils_cw import Print, get_items_from_file
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@click.command("merge-roc")
@click.option(
    "--exp-dir", type=click.Path(exists=True), help="Dir contains test folders"
)
@click.option(
    "--dirname-as-legend", type=bool, default=True, help="Use dirname as legend label"
)
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
            auc_value = get_items_from_file(cls_result_file).get("AUC", -1)
            auc_values.append(auc_value)
        else:
            auc_value = click.prompt(
                f"Input ROC value for {roc_file.parent.name}", type=float
            )
            auc_values.append(auc_value)

    # Draw ROC
    fig = plt.figure(figsize=(8, 5), dpi=200)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 10

    for auc, roc, legend in sorted(zip(auc_values, roc_files, legend_names)):
        with open(roc) as f:
            reader = csv.DictReader(f)
            fpr_tpr = np.array(
                [[float(row["FPR"]), float(row["TPR"])] for row in reader]
            )
        plt.plot(fpr_tpr[:, 0], fpr_tpr[:, 1], label=f"{legend} AUC={auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig(os.path.join(exp_dir, "merged_roc.png"))
    Print("ROC is saved to", os.path.join(exp_dir, "merged_roc.png"), color="g")
