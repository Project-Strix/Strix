import os
import sys 

PY_REQUIRED_MAJOR = 3
PY_REQUIRED_MINOR = 7

from . import _version
__version__ = _version.get_versions()['version']

__basedir__ = os.path.dirname(__file__)