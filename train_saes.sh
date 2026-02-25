#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

cd /users/ahmet.turkel/CMPE492
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
export PIP_USER=false
pip install --upgrade pip
pip install -r src/requirements.txt

python src/SAE/train_sae.py