#!/bin/bash
#SBATCH --job-name=glm4v_individual
#SBATCH --output=/home/xc1490/xc1490/projects/medvqa_2025/logs/glm4v_individual_%A_%a.out
#SBATCH --error=/home/xc1490/xc1490/projects/medvqa_2025/logs/glm4v_individual_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-3

# Array job for training individual models on each dataset
# 1=PVQA, 2=SLAKE, 3=RAD

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

# Set dataset based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        DATASET="PVQA"
        ;;
    2)
        DATASET="SLAKE"
        ;;
    3)
        DATASET="RAD"
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "Starting training for dataset: ${DATASET}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

# Step 1: Convert data to GLM-4.1V format (only for individual datasets)
echo "Converting data to GLM-4.1V format..."
python ${SCRIPTS_DIR}/convert_medvqa_to_glm4v.py \
    --mode individual \
    --data_dir ${DATA_DIR}/medvqa \
    --output_dir ${DATA_DIR}/glm4v_format \
    --image_base ${DATA_DIR}/medvqa/3vqa/images

# Step 2: Train the model
echo "Training GLM-4.1V model on ${DATASET} dataset..."
python ${SCRIPTS_DIR}/train_glm4v_individual.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR}/glm4v_format \
    --llamafactory_path ${LLAMAFACTORY_PATH} \
    --output_dir ${PROJECT_DIR}/models/glm4v_${DATASET,,}

echo "Training completed for dataset: ${DATASET}"
