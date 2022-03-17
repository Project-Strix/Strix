# import os
# import torch
from setuptools import setup, find_namespace_packages

# from pathlib import Path
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import CppExtension
# from torch.utils.cpp_extension import CUDAExtension

# def get_extensions():
#     extensions_dir = Path(__file__).parent.joinpath('medlp/models/rcnn/csrc')

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
#             "medlp.models.rcnn._C",
#             sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ]

#     return ext_modules


setup(
    name="medlp",
    packages=find_namespace_packages(include=["medlp", "medlp.*"]),
    version="0.0.6",
    description="Medical Deep Learning Platform",
    author="Chenglong Wang",
    author_email="clwang@phy.ecnu.edu.cn",
    license="Apache License Version 2.0, January 2004",
    install_requires=[
        "torch>=1.6.0",
        "tb-nightly",
        "click",
        "tqdm",
        "scikit-image>=0.14",
        "scipy",
        "numpy",
        "scikit-learn",
        "nibabel",
        "pytorch-ignite==0.4.7",
        # "nni",
        # "monai_ex @ git+https://gitlab.com/project-medlp/MONAI_EX@master#egg=monai_ex>=0.0.3",
        "utils_cw @ git+https://gitlab.com/ChingRyu/py_utils_cw@master#egg=utils_cw",
    ],
    entry_points={
        "console_scripts": [
            "medlp-train = medlp.main_entry:train",
            "medlp-train-from-cfg = medlp.main_entry:train_cfg",
            "medlp-train-and-test = medlp.main_entry:train_and_test",
            "medlp-test-from-cfg = medlp.main_entry:test_cfg",
            "medlp-nni-search = medlp.nni_search:nni_search",
            "medlp-check-data = medlp.data_checker:check_data",
            "medlp-gradcam-from-cfg = medlp.interpreter:gradcam",
        ],
    },
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    keywords=[
        "deep learning",
        "medical image classification",
        "medical image analysis",
        "medical image segmentation",
    ],
)
