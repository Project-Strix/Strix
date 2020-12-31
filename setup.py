import os, torch
from pathlib import Path
from setuptools import setup, find_namespace_packages
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

def get_extensions():
    extensions_dir = Path(__file__).parent.joinpath('medlp/models/rcnn/csrc')

    main_file = list(extensions_dir.glob("*.cpp"))
    source_cpu = list((extensions_dir/"cpu").glob("*.cpp"))
    source_cuda = list((extensions_dir/"cuda").glob("*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [ str(s.resolve()) for s in sources ]

    include_dirs = [str(extensions_dir)]

    ext_modules = [
        extension(
            "medlp.models.rcnn._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(name='medlp',
      packages=find_namespace_packages(include=["medlp", "medlp.*"]),
      version='0.0.3',
      description='Medical Deep Learning Platform',
      author='Chenglong Wang',
      author_email='clwang@phy.ecnu.edu.cn',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>=1.5.0",
            "tb-nightly",
            "click",
            "tqdm",
            "scikit-image>=0.14",
            "scipy",
            "numpy",
            "scikit-learn",
            "nibabel",
            # "nni",
            "monai_ex @ git+https://gitlab.com/project-medlp/MONAI_EX@master#egg=monai_ex",
            "utils_cw @ git+https://github.com/ChenglongWang/py_utils_cw@master#egg=utils_cw",
      ],
      ext_modules=get_extensions(),
      cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation']
      )