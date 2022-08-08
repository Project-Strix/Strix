import json
from pathlib import Path
from typing import Any, Union, Optional

from strix.utilities.imports import ModuleManager
from strix.utilities.utils import get_items


class ProjectManager:
    def __init__(self) -> None:
        super(ProjectManager, self).__init__()
        self.project_file = None
        self.project_dict = {}

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ProjectManager, cls).__new__(cls)
        return cls.instance

    @property
    def project_name(self):
        if self.project_file:
            return self.project_file.resolve().parent.name
        return None

    def load(self, project_file: Union[str, Path]) -> Any:
        if isinstance(project_file, str):
            project_file = Path(project_file)

        self.project_file = project_file

        if not self.project_file:
            raise ValueError("No project_file is input!")

        self.project_dict = get_items(project_file)

        for key in ["LOSS_DIR_PATH", "NETWORK_DIR_PATH", "DATASET_DIR_PATH"]:
            folder_path = self.project_dict[key]
            if not Path(folder_path).is_absolute():
                folder_path = project_file.parent / folder_path
            if Path(folder_path).is_dir():
                ModuleManager.import_all(folder_path)

    def save(self, project_dict, save_path):
        with open(save_path, "w") as f:
            json.dump(project_dict, f, indent=2)
