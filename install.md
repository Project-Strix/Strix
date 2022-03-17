## How to install?
---
### 1. Create a new virtual env. 

```
conda create -n medlp python=3.8
```

### 2. Install pytorch (here show example of installing pytorch1.9)
```
conda activate medlp
conda install pytorch==1.9.1 torchvision==0.10.0 cudatoolkit=11.3 -c conda-forge
```

### 3. If you want to develop Medlp, you can download Medlp project.
```
git clone http://mega/gitlab/project-medlp/medlp.git
pip install -e ./Medlp/MONAI_ex
pip install -e ./Medlp
```
### 4. If you just want to use Medlp, you can directly install Medlp by:
```
pip install git+http://mega/gitlab/project-medlp/monai_ex.git
pip install git+http://mega/gitlab/project-medlp/medlp.git
```