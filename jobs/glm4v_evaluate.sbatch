#!/bin/bash
#SBATCH --job-name=glm4v_eval
#SBATCH --output=/home/xc1490/xc1490/projects/medvqa_2025/logs/glm4v_eval_%A_%a.out
#SBATCH --error=/home/xc1490/xc1490/projects/medvqa_2025/logs/glm4v_eval_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-12

# Array job for evaluation
# Jobs 1-3: Individual models on their respective test sets
# Jobs 4-6: Individual models cross-evaluated on other test sets
# Jobs 7-9: Combined model on all test sets
# Jobs 10-12: Additional cross-evaluations

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

# Set model and dataset based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_pvqa"
        TEST_DATASET="PVQA"
        EVAL_NAME="pvqa_on_pvqa"
        ;;
    2)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_slake"
        TEST_DATASET="SLAKE"
        EVAL_NAME="slake_on_slake"
        ;;
    3)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_rad"
        TEST_DATASET="RAD"
        EVAL_NAME="rad_on_rad"
        ;;
    4)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_pvqa"
        TEST_DATASET="SLAKE"
        EVAL_NAME="pvqa_on_slake"
        ;;
    5)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_pvqa"
        TEST_DATASET="RAD"
        EVAL_NAME="pvqa_on_rad"
        ;;
    6)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_slake"
        TEST_DATASET="RAD"
        EVAL_NAME="slake_on_rad"
        ;;
    7)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_combined"
        TEST_DATASET="PVQA"
        EVAL_NAME="combined_on_pvqa"
        ;;
    8)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_combined"
        TEST_DATASET="SLAKE"
        EVAL_NAME="combined_on_slake"
        ;;
    9)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_combined"
        TEST_DATASET="RAD"
        EVAL_NAME="combined_on_rad"
        ;;
    10)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_slake"
        TEST_DATASET="PVQA"
        EVAL_NAME="slake_on_pvqa"
        ;;
    11)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_rad"
        TEST_DATASET="PVQA"
        EVAL_NAME="rad_on_pvqa"
        ;;
    12)
        MODEL_PATH="${PROJECT_DIR}/models/glm4v_rad"
        TEST_DATASET="SLAKE"
        EVAL_NAME="rad_on_slake"
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "Starting evaluation: ${EVAL_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Test Dataset: ${TEST_DATASET}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path ${MODEL_PATH} does not exist. Skipping evaluation."
    exit 0
fi

# Run evaluation
python ${SCRIPTS_DIR}/evaluate_glm4v.py \
    --model_path ${MODEL_PATH} \
    --test_dataset ${TEST_DATASET} \
    --data_dir ${DATA_DIR}/medvqa \
    --image_base ${DATA_DIR}/medvqa/3vqa/images \
    --llamafactory_path ${LLAMAFACTORY_PATH} \
    --output_dir ${PROJECT_DIR}/results/${EVAL_NAME}

echo "Evaluation completed: ${EVAL_NAME}"
