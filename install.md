## How to install?
---
### 1. Create a new virtual env. 

```
conda create -n strix python=3.8
```

### 2. Install pytorch (here show example of installing pytorch1.9)
```
conda activate strix
conda install pytorch==1.9.1 torchvision==0.10.0 cudatoolkit=11.3 -c conda-forge
```

### 3. If you want to develop Strix, you can clone Strix project (along with MONAI_EX).
```
git clone --recursive https://github.com/Project-Strix/Strix.git
pip install -e ./Strix/MONAI_EX
pip install -e ./Strix
```

### 4. If you just want to use Strix, you can directly install Strix by:
```
pip install git+https://github.com/Project-Strix/MONAI_EX.git
pip install git+https://github.com/Project-Strix/Strix.git
```


_Notice that Strix is only tested on **Linux** only, not on Windows. If you find any problem with the deployment on Windows, please submit an [issue](https://github.com/Project-Strix/Strix/issues)._

