import json
from pathlib import Path

from typing import Any
from strix.utilities.utils import get_items
from strix.utilities.imports import ModuleManager


class ProjectLoader:
    def __init__(self) -> None:
        super(ProjectLoader, self).__init__()
        self.project_dict = {}
        self.project_name = None

    def load(self, project_file) -> Any:
        self.project_dict = get_items(project_file)
        self.project_name = project_file.parent.name

        for key in ["LOSS_DIR_PATH", "NETWORK_DIR_PATH", "DATASET_DIR_PATH"]:
            folder_path = self.project_dict[key]
            if not Path(folder_path).is_absolute():
                folder_path = project_file.parent / folder_path
            if Path(folder_path).is_dir():
                ModuleManager.import_all(folder_path)

    def save(self, project_dict, save_path):
        with open(save_path, "w") as f:
            json.dump(project_dict, f, indent=2)
