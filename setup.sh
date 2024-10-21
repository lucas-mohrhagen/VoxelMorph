#!/bin/bash

# TODO change path to where you want to save the environment
# on hlrn save the env in scratch_emmy
ENV_NAME='voxel_env'

conda env remove -p $ENV_NAME
conda create -n $ENV_NAME python=3.9 -y
conda init
conda activate $ENV_NAME

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
pip install jupyter
pip install -e .

cd ~/voxel_neuron_morph/src/implicitmorph/mesh_utils/libmesh
mkdir libmesh
python setup.py build_ext --inplace
mv libmesh/* .
rm -rf libmesh
cd -

conda install -c conda-forge cudatoolkit-dev -y

# PointNet++
# pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
# pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# further packages
conda deactivate