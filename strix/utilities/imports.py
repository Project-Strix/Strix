from pathlib import Path
import importlib
import importlib.util
import sys
import warnings

# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


class ModuleManager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def import_all(folder: str):
        if Path(folder).is_dir():
            python_files = list(Path(folder).glob("*.py"))

            if len(python_files) > 0:
                sys.path.append(str(folder))

            for f in python_files:
                try:
                    import_file(f.stem, str(f))
                except Exception as e:
                    warnings.warn(f"Failed to import file {f}!\nError msg: {e}")
