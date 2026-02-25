#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

cd /users/ahmet.turkel/CMPE492

source .venv/bin/activate

python src/SAE/train_sae.py