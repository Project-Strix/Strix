import os

PY_REQUIRED_MAJOR = 3
PY_REQUIRED_MINOR = 8

from . import _version
__version__ = _version.get_versions()['version']

__basedir__ = os.path.dirname(__file__)

from strix.utilities.registry import NetworkRegistry, DatasetRegistry, LossRegistry

strix_networks = NetworkRegistry()
strix_datasets = DatasetRegistry()
strix_losses = LossRegistry()
