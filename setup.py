from setuptools import setup, find_namespace_packages
import versioneer

# from pathlib import Path
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import CppExtension
# from torch.utils.cpp_extension import CUDAExtension

# def get_extensions():
#     extensions_dir = Path(__file__).parent.joinpath('strix/models/rcnn/csrc')

#     main_file = list(extensions_dir.glob("*.cpp"))
#     source_cpu = list((extensions_dir/"cpu").glob("*.cpp"))
#     source_cuda = list((extensions_dir/"cuda").glob("*.cu"))

#     sources = main_file + source_cpu
#     extension = CppExtension

#     extra_compile_args = {"cxx": []}
#     define_macros = []

#     if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
#         extension = CUDAExtension
#         sources += source_cuda
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]

#     sources = [ str(s.resolve()) for s in sources ]

#     include_dirs = [str(extensions_dir)]

#     ext_modules = [
#         extension(
#             "strix.models.rcnn._C",
#             sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ]

#     return ext_modules

def get_cmds():
    cmds = versioneer.get_cmdclass()
    return cmds

    # cmds.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})

setup(
    name="strix",
    packages=find_namespace_packages(include=["strix", "strix.*"]),
    version=versioneer.get_version(),
    description="Medical Deep Learning Platform",
    author="Chenglong Wang",
    author_email="clwang@phy.ecnu.edu.cn",
    license="Apache License Version 2.0, January 2004",
    entry_points={
        "console_scripts": [
            "strix-train = strix.main_entry:train",
            "strix-train-from-cfg = strix.main_entry:train_cfg",
            "strix-train-and-test = strix.main_entry:train_and_test",
            "strix-test-from-cfg = strix.main_entry:test_cfg",
            "strix-nni-search = strix.nni_search:nni_search",
            "strix-check-data = strix.data_checker:check_data",
            "strix-gradcam-from-cfg = strix.interpreter:gradcam",
        ],
    },
    scripts=[
        'strix/misc/strix-profile'
    ],
    # ext_modules=get_extensions(),
    cmdclass=get_cmds(),
    keywords=[
        "deep learning",
        "medical image classification",
        "medical image analysis",
        "medical image segmentation",
    ],
)
