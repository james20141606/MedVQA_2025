#!/bin/bash
#SBATCH --job-name=glm4v_combined
#SBATCH --output=/home/xc1490/xc1490/projects/medvqa_2025/logs/glm4v_combined_%j.out
#SBATCH --error=/home/xc1490/xc1490/projects/medvqa_2025/logs/glm4v_combined_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Job for training combined model on all datasets

# Create logs directory
mkdir -p /home/xc1490/xc1490/projects/medvqa_2025/logs

# Load modules and activate environment
module load cuda/11.8
module load python/3.9
source /home/xc1490/venv/llama-factory/bin/activate

# Set up paths
PROJECT_DIR="/home/xc1490/xc1490/projects/medvqa_2025"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
DATA_DIR="${PROJECT_DIR}/data"
LLAMAFACTORY_PATH="/home/xc1490/LLaMA-Factory"

echo "Starting combined training on all MedVQA datasets"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

# Step 1: Convert and combine data to GLM-4.1V format
echo "Converting and combining data to GLM-4.1V format..."
python ${SCRIPTS_DIR}/convert_medvqa_to_glm4v.py \
    --mode combined \
    --data_dir ${DATA_DIR}/medvqa \
    --output_dir ${DATA_DIR}/glm4v_format \
    --image_base ${DATA_DIR}/medvqa/3vqa/images

# Step 2: Train the combined model
echo "Training GLM-4.1V model on combined MedVQA datasets..."
python ${SCRIPTS_DIR}/train_glm4v_combined.py \
    --data_dir ${DATA_DIR}/glm4v_format \
    --llamafactory_path ${LLAMAFACTORY_PATH} \
    --output_dir ${PROJECT_DIR}/models/glm4v_combined

echo "Combined training completed"
