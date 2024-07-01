# Pytorch - CUDA - Python Compatibility
Project requirement - Pytorch 1.XX

## Release Compatibility Matrix

Following is the Release Compatibility Matrix for PyTorch releases:

| PyTorch version | Python | Stable CUDA | Experimental CUDA | Stable ROCm |
| --- | --- | --- | --- | --- |
| 2.3 | >=3.8, <=3.11, (3.12 experimental) | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 | ROCm 6.0 |
| 2.2 | >=3.8, <=3.11, (3.12 experimental) | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 | ROCm 5.7 |
| 2.1 | >=3.8, <=3.11 | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 | ROCm 5.6 |
| 2.0 | >=3.8, <=3.11 | CUDA 11.7, CUDNN 8.5.0.96 | CUDA 11.8, CUDNN 8.7.0.84 | ROCm 5.4 |
| 1.13 | >=3.7, <=3.10 | CUDA 11.6, CUDNN 8.3.2.44 | CUDA 11.7, CUDNN 8.5.0.96 | ROCm 5.2 |
| 1.12 | >=3.7, <=3.10 | CUDA 11.3, CUDNN 8.3.2.44 | CUDA 11.6, CUDNN 8.3.2.44 | ROCm 5.0 |

We are using Torch 1.13, Python 3.8, and CUDA 11.6


1. Install Conda Env
```
conda create -n pointnet python==3.8
conda activate pointnet
```
2. Install CUDA ToolKit 11.6
```
conda install nvidia/label/cuda-11.6.0::cuda-toolkit
```
3. Install the project using setup.py - except comment out pytorch and install pytorch separately.
```
# Install PyTorch 1.13.0 with CUDA 11.6 support
conda install pytorch==1.13.0 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c nvidia
pip install -e .
```
