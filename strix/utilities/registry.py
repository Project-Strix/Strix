from typing import Optional

import os
import inspect
import warnings
from termcolor import colored
from strix.utilities.enum import DIMS, NETWORK_ARGS, FRAMEWORKS
from strix.utilities.utils import singleton


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


def _register_generic_dim(module_dict, dim, module_name, module):
    assert module_name not in module_dict.get(
        dim
    ), f"{module_name} already registed in {module_dict.get(dim)}"

    module_dict[dim].update({module_name: module})


def _register_network(module_dict, dim, framework, module_name, module):
    if module_name in module_dict[dim][framework]:
        warnings.warn(colored(f"{module_name} already registed! Skip!", "yellow"))
        return False

    module_dict[dim][framework].update({module_name: module})
    return True


def _register_generic_data(
    module_dict, dim, module_name, train_fpath, test_fpath, module
):
    assert module_name not in module_dict.get(
        dim
    ), f"{module_name} already registed in {module_dict.get(dim)}"

    attr = {module_name: {"FN": module, "PATH": train_fpath, "TEST_PATH": test_fpath}}
    module_dict[dim].update(attr)


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn


class DimRegistry(dict):
    def __init__(self, *args, **kwargs):
        super(DimRegistry, self).__init__(*args, **kwargs)
        self.dim_mapping = {
            "2": "2D",
            "3": "3D",
            2: "2D",
            3: "3D",
            "2D": "2D",
            "3D": "3D",
        }
        self["2D"] = {}
        self["3D"] = {}

    def register(self, dim, module_name, module=None):
        dim = self.dim_mapping.get(dim)
        assert dim in DIMS, "Only support '2D'&'3D' dataset now"
        # used as function call
        if module is not None:
            _register_generic_dim(self, dim, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic_dim(self, dim, module_name, fn)
            return fn

        return register_fn


@singleton
class NetworkRegistry(DimRegistry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for dim in DIMS:
            for framework in FRAMEWORKS:
                self[dim].update({framework: {}})

    def check_args(self, func, module_name):
        sig = inspect.signature(func)
        func_args = list(sig.parameters.keys())
        for arg in NETWORK_ARGS:
            if arg not in func_args:
                raise ValueError(
                    f"Missing argument {arg} in your {module_name} network API funcion"
                )

    def register(self, dim, framework, module_name, module=None):
        dim = self.dim_mapping.get(dim)
        assert dim in DIMS, "Only support '2D'&'3D' dataset now"
        assert framework in FRAMEWORKS, f"Given '{framework}' is not supported yet!"
        
        # used as function call
        if module is not None:
            self.check_args(module, module_name)
            _register_network(self, dim, framework, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            self.check_args(fn, module_name)
            _register_network(self, dim, framework, module_name, fn)
            return fn

        return register_fn


class DatasetRegistry(DimRegistry):
    def __init__(self, *args, **kwargs):
        super(DatasetRegistry, self).__init__(*args, **kwargs)

    def register(
        self,
        dim: int,
        module_name: str,
        train_filepath: str,
        test_filepath: Optional[str] = None,
        module=None,
    ):
        assert dim in DIMS, "Only support '2D'&'3D' dataset now"
        dim = self.dim_mapping[dim]
        # used as function call
        if module is not None:
            _register_generic_data(
                self, dim, module_name, train_filepath, test_filepath, module
            )
            return

        # used as decorator
        def register_fn(fn):
            _register_generic_data(
                self, dim, module_name, train_filepath, test_filepath, fn
            )
            return fn

        return register_fn

    def _get_keys(self, val):
        dims = ["2D", "3D"]
        results = []
        for d in dims:
            for key, value in self[d].items():
                if val == value["FN"]:
                    results.append((d, key))
        return results

    def multi_in(self, *keys):
        def register_input(fn):
            dim_module_list = self._get_keys(fn)
            for dim, module_name in dim_module_list:
                self[dim][module_name].update({"M_IN": keys})
            return fn

        return register_input

    def multi_out(self, *keys):
        def register_output(fn):
            dim_module_list = self._get_keys(fn)
            for dim, module_name in dim_module_list:
                self[dim][module_name].update({"M_OUT": keys})
            return fn

        return register_output

    def snapshot(self, fn):
        def register_source(func):
            source = os.path.abspath(inspect.getfile(func))
            dim_module_list = self._get_keys(func)
            for dim, module_name in dim_module_list:
                self[dim][module_name].update({"SOURCE": source})
            return func

        return register_source(fn)

    def project(self, proj_name):
        def register_proj(fn):
            dim_module_list = self._get_keys(fn)
            for dim, module_name in dim_module_list:
                self[dim][module_name].update({"PROJECT": proj_name})
            return fn

        return register_proj
