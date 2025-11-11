# QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization
[[Paper]](https://arxiv.org/abs/2505.05591)

# Release Schedule
- [x] Basic trainer and model code
- [x] Dataset and preprocessing code
- [x] Evaluation scripts
- [ ] Checkpoints

# Setup Environment

```
git clone --recursive https://github.com/liu115/QuickSplat quicksplat
cd quicksplat
```

Install pytorch
```bash
conda create -n quicksplat python=3.10
conda activate quicksplat
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```


## Install 2D/3D Gaussian splatting rasterizers

This will requires nvcc and cuda build tools. If you don't have them installed, you can install them via conda:
```bash
conda install cuda -c nvidia/label/cuda-11.8.0
conda install gxx_linux-64
```


```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-surfel-rasterization
```


## Install MinkowskiEngine for sparse 3D convolution
```bash
conda install openblas-devel -c anaconda

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
```

The original installation command won't work anymore:
```
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

To fix the issue, try this instead:

1. Find your conda environment include directory which is ${CONDA_PREFIX}/include

2. Goto setup.py and add two lines to hardcode the BLAS settings:

```
 BLAS, argv = _argparse("--blas", argv, False)
+BLAS = "openblas"
 BLAS_INCLUDE_DIRS, argv = _argparse("--blas_include_dirs", argv, False, is_list=True)
+BLAS_INCLUDE_DIRS = ["YOUR_CONDA_ENV_INCLUDE"]
 BLAS_LIBRARY_DIRS, argv = _argparse("--blas_library_dirs", argv, False, is_list=True)
```
3. Then run pip install:
```bash
pip install .
```


## Install pytorch3d
```bash
conda install -c iopath iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# or to prevent memory explode due to ninja allocating all the cores
PYTORCH3D_NO_NINJA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```


## Install other dependencies
```
pip install -r requirements.txt
```


# Dataset preparation
Please download the ScanNet++ v2 dataset from [here](https://scannetpp.mlsg.cit.tum.de/scannetpp/), and follow the instructions in `preprocess/README.md` to preprocess the dataset for QuickSplat training and/or evaluation.


# Inference and Evaluation
To evaluate the method using the checkpoints on the testing scenes:

```bash
python inference.py \
    --config configs/phase2_eval.yaml \
    --ckpt checkpoints/phase2.ckpt \
    --out inference_outputs \
    DATASET.test_split_path <PATH_TO_TRAIN_DATASET_DIR>/splits/test_scene_ids.txt \
    DATASET.source_path <PATH_TO_TRAIN_DATASET_DIR>/data \
    DATASET.ply_path  <PATH_TO_TRAIN_DATASET_DIR>/colmap \
    DATASET.gt_ply_path <PATH_TO_TRAIN_DATASET_DIR>/mesh
```


# Training

## Phase 1: Train the Gaussian initialization network
```bash
python train.py --config configs/phase1.yaml

# In multi-GPU training (e.g., 2 GPUs), run the following command:
torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/phase1.yaml MACHINE.num_devices 2
```

## Phase 2: Training the optimization network
```bash
python train.py --config configs/phase2.yaml

# In multi-GPU training (e.g., 2 GPUs), run the following command:
torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/phase2.yaml MACHINE.num_devices 2
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
