#!/bin/bash
#SBATCH --job-name=glm4v_lora
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o %x_%j.out

set -euo pipefail

module load anaconda/3 || true
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate medvqa || true

python $(dirname $0)/../scripts/train_model.py \
  --method lora \
  --dataset combined \
  --mode base \
  --model_version 4.1v \
  --model_name_or_path zai-org/GLM-4.1V-9B-Base \
  --llamafactory_path /home/xc1490/LLaMA-Factory \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3

