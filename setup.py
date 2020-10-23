from setuptools import setup, find_namespace_packages

setup(name='medlp',
      packages=find_namespace_packages(include=["medlp", "medlp.*"]),
      version='0.0.1',
      description='Medical Deep learning platform',
      author='Chenglong Wang',
      author_email='clwang@phy.ecnu.edu.cn',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>=1.5.0",
            "tqdm",
            "scikit-image>=0.14",
            "scipy",
            "numpy",
            "scikit-learn",
            "nibabel",
            "nni",
            #"monai @ git+https://github.com/ChenglongWang/MONAI.git"
      ],
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation']
      )