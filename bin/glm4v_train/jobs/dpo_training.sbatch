#!/bin/bash
#SBATCH --job-name=glm4v_dpo
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

BASE_SFT_MODEL=/home/xc1490/xc1490/projects/medvqa_2025/models/glm4.1v_combined_base_lora
PREF_FILE=/home/xc1490/xc1490/projects/medvqa_2025/data/dpo/medvqa_pairs.json

python $(dirname $0)/../scripts/train_model.py \
  --method dpo \
  --dataset combined \
  --mode base \
  --model_version 4.1v \
  --base_model ${BASE_SFT_MODEL} \
  --preference_file ${PREF_FILE} \
  --llamafactory_path /home/xc1490/LLaMA-Factory \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 1

