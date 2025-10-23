# QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization
[[Paper]](https://arxiv.org/abs/2505.05591)

# Release Schedule
- [x] Basic trainer and model code
- [ ] Dataset and preprocessing code
- [ ] Checkpoints
- [ ] Evaluation scripts

# Setup Environment

Install pytorch
```
conda create -n quicksplat python=3.10
conda activate quicksplat
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```


## Install 2D/3D Gaussian splatting rasterizers

### Need cuda build tools
conda install cuda -c nvidia/label/cuda-11.8.0
conda install gxx_linux-64

```
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-surfel-rasterization
```


## Install MinkowskiEngine for sparse 3D convolution
```
conda install openblas-devel -c anaconda

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
```
This don't work anymore:
```
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Try this instead:

1. Find your conda environment include directory which is ${CONDA_PREFIX}/include

2. Goto setup.py. add

```
 BLAS, argv = _argparse("--blas", argv, False)
+BLAS = "openblas"
 BLAS_INCLUDE_DIRS, argv = _argparse("--blas_include_dirs", argv, False, is_list=True)
+BLAS_INCLUDE_DIRS = ["YOUR_CONDA_ENV_INCLUDE"]
 BLAS_LIBRARY_DIRS, argv = _argparse("--blas_library_dirs", argv, False, is_list=True)
```
3. run
```
pip install .
```


## Install pytorch3d
```
conda install -c iopath iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# or to prevent memory explode due to ninja allocating all the cores
PYTORCH3D_NO_NINJA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```


## Install other dependencies
```
pip install -r requirements.txt
```


# Citation
If you wish to cite us, please use the following BibTeX entry:
```
@article{liu2025quicksplat,
  title={QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization},
  author={Liu, Yueh-Cheng and H{\"o}llein, Lukas and Nie{\ss}ner, Matthias and Dai, Angela},
  journal={arXiv preprint arXiv:2505.05591},
  year={2025}
}
```
