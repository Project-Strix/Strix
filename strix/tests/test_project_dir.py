import pytest
import click
import yaml
from pathlib import Path
from functools import partial
from strix.main_entry import train


project_file_content = {
    "LOSS_DIR_PATH": "test_loss_dir",
    "NETWORK_DIR_PATH": "",
    "DATASET_DIR_PATH": "./test_ds_dir"
}

dataset_content = """
from strix.data_io import CLASSIFICATION_DATASETS

@CLASSIFICATION_DATASETS.register('2D', 'test_ds', '')
def get_3d_dataset(files_list, phase, opts):
    return None
"""

def test_project_dir(runner, tmp_path):
    pass
    # with open(tmp_path / "project.yml", "w") as f:
    #     yaml.dump(project_file_content, f)

    # loss_dir = check_dir(tmp_path / "test_loss_dir")
    # ds_dir = check_dir(tmp_path / "test_ds_dir")

    # with open(ds_dir / "test_dataset.py", "w") as f:
    #     f.write(dataset_content)

    # result = runner.invoke(train, ["--tensor-dim", "2D", "--framework", "1", "--project", tmp_path, "--model-name", "resnet50"], input="1\n")
    # print("Output:", result.output)
