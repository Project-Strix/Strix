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

### 3. If you want to develop Strix, you can download Strix project.
```
git clone http://mega/gitlab/project-strix/strix.git
pip install -e ./Strix/MONAI_ex
pip install -e ./Strix
```
### 4. If you just want to use Strix, you can directly install Strix by:
```
pip install git+http://mega/gitlab/project-strix/monai_ex.git
pip install git+http://mega/gitlab/project-strix/strix.git
```