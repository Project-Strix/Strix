from typing import Optional, Union

import os
import inspect
import warnings
from termcolor import colored
from strix.utilities.enum import DIMS, NETWORK_ARGS, FRAMEWORKS, Phases
from strix.utilities.utils import singleton


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


def _register_with_argument(module_dict, key, module_name, module):
    if module_name in module_dict[key]:
        warnings.warn(colored(f"{module_name} was already registed in {key}", "yellow"))
        return False

    module_dict[key].update({module_name: module})
    return True


def _register_network(module_dict, dim, framework, module_name, module):
    if module_name in module_dict[dim][framework]:
        warnings.warn(colored(f"{module_name} already registed! Skip!", "yellow"))
        return False

    module_dict[dim][framework].update({module_name: module})
    return True


def _register_dataset(module_dict, dim, framework, module_name, train_fpath, test_fpath, module):
    if module_name in module_dict[dim][framework]:
        warnings.warn(colored(f"{module_name} already registed! Skip!", "yellow"))
        return False

    attr = {module_name: {"FN": module, "PATH": train_fpath, "TEST_PATH": test_fpath}}
    module_dict[dim][framework].update(attr)


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
            _register_with_argument(self, dim, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_with_argument(self, dim, module_name, fn)
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
        assert dim in self.dim_mapping, f"Only support {self.dim_mapping} dataset now"
        assert framework in FRAMEWORKS, f"Given '{framework}' is not supported yet!"

        dim = self.dim_mapping.get(dim)
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

    def get(self, dim: Union[int, str], framework: str, name: str):
        """alias of direct access by `[]`

        Args:
            dim (int | str): tensor dim
            framework (str): framework type, eg. segmentation
            name (str): network name which has been register

        Returns:
            Net: speificed network.
            None: if no network is found.
        """
        try:
            dim = self.dim_mapping.get(dim)
            net = self[dim][framework][name]
        except KeyError as e:
            warnings.warn(colored(f"Network is not registered!\nErr msg: {e}", "red"))
            return None
        else:
            return net

    def list(self, dim: Union[int, str], framework: str):
        """List all registered networks by given dim and framework

        Args:
            dim (Union[int, str]): tensor dim
            framework (str): framework type, eg. classification

        Returns:
            list: if no network is found, return empty list [].
        """
        try:
            dim = self.dim_mapping.get(dim)
            nets = list(self[dim][framework].keys())
        except KeyError as e:
            warnings.warn(colored(f"Key error!\nErr msg: {e}", "red"))
            return []
        else:
            return nets


@singleton
class DatasetRegistry(DimRegistry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for dim in DIMS:
            for framework in FRAMEWORKS:
                self[dim].update({framework: {}})

    def register(
        self,
        dim: Union[int, str],
        framework: str,
        module_name: str,
        train_filepath: str,
        test_filepath: Optional[str] = None,
        module=None,
    ):
        assert dim in self.dim_mapping, f"Only support {self.dim_mapping} dataset now"
        assert framework in FRAMEWORKS, f"Given '{framework}' is not supported yet!"
        dim = self.dim_mapping[dim]
        # used as function call
        if module is not None:
            _register_dataset(self, dim, framework, module_name, train_filepath, test_filepath, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_dataset(self, dim, framework, module_name, train_filepath, test_filepath, fn)
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
            dim_module_list = self._get_keys(fn)  # todo: refactor!
            for dim, module_name in dim_module_list:
                self[dim][module_name].update({"M_IN": keys})
            return fn

        return register_input

    def multi_out(self, *keys):
        def register_output(fn):
            dim_module_list = self._get_keys(fn)  # todo: refactor!
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

    def get(self, dim: Union[int, str], framework: str, name: str):
        """Get registered Strix dataset.

        Args:
            dim (Union[int, str]): tensor dim
            framework (str): framwork type
            name (str): registered dastset name

        Returns:
            dict: registered dataset
        """
        try:
            dim = self.dim_mapping.get(dim)
            dataset = self[dim][framework][name]
        except KeyError as e:
            warnings.warn(colored(f"Dataset is not registered!\nErr msg: {e}", "red"))
            return None
        else:
            return dataset

    def list(self, dim: Union[int, str], framework: str):
        """List all registered datasets by given dim and framework

        Args:
            dim (Union[int, str]): tensor dim
            framework (str): framework type, eg. classification

        Returns:
            list: if no dataset is found, return empty list [].
        """
        try:
            dim = self.dim_mapping.get(dim)
            datasets = list(self[dim][framework].keys())
        except KeyError as e:
            warnings.warn(colored(f"Key error!\nErr msg: {e}", "red"))
            return []
        else:
            return datasets


@singleton
class LossRegistry(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for framework in FRAMEWORKS:
            self[framework] = {}


    def register(self, framework: str, loss_name: str, module=None):
        assert framework in FRAMEWORKS, f"Given '{framework}' is not supported yet!"
        # used as function call
        if module is not None:
            _register_with_argument(self, framework, loss_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_with_argument(self, framework, loss_name, fn)
            return fn

        return register_fn

    def get(self, framework: str, name: str):
        """alias of direct access by `[]`

        Args:
            dim (int | str): tensor dim
            framework (str): framework type, eg. segmentation
            name (str): network name which has been register

        Returns:
            callable: speificed loss func.
            None: if no network is found.
        """
        try:
            loss = self[framework][name]
        except KeyError as e:
            warnings.warn(colored(f"Loss is not registered!\nErr msg: {e}", "red"))
            return None
        else:
            return loss
