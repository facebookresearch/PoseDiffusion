# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.9, CUDA 11.6

conda deactivate

# Set environment variables
export ENV_NAME=posediffusion
export PYTHON_VERSION=3.9
export PYTORCH_VERSION=1.13.0
export CUDA_VERSION=11.6

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using conda
conda install pytorch=$PYTORCH_VERSION torchvision pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Install pip packages
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom accelerate

# Install HLoc for extracting 2D matches (optional if GGS is not needed)
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git dependency/hloc

cd dependency/hloc
python -m pip install -e .
cd ../../

# Ensure the version of pycolmap is not 0.5.0
pip install --upgrade "pycolmap>=0.3.0,<=0.4.0"

