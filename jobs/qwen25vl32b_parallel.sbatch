#!/bin/bash

#SBATCH --job-name=qwen32b_parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'  
#SBATCH --time=48:00:00
#SBATCH --account=pr_60_tandon_advanced

# Get the instance number from SLURM_ARRAY_TASK_ID (if using job array)
# or from command line argument
INSTANCE_ID=${SLURM_ARRAY_TASK_ID:-$1}
if [ -z "$INSTANCE_ID" ]; then
    echo "Error: Please provide instance ID as argument or use job array"
    echo "Usage: sbatch --array=0-5 qwen25vl32b_parallel.sbatch"
    echo "   or: sbatch qwen25vl32b_parallel.sbatch 1"
    exit 1
fi

echo "Starting Qwen 32B parallel instance $INSTANCE_ID"

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

cd /scratch/xc1490/projects/medvqa_2025/bin

# Use different random seeds for different instances to ensure different dataset selection
# and sample shuffling
RANDOM_SEED=$((42 + $INSTANCE_ID))

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M_blip.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate diff_vlm

echo \"Instance $INSTANCE_ID: Using random seed $RANDOM_SEED\"

# Enforce offline HF/Transformers and shared cache inside container
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/xc1490/hf_cache

# Isolated per-instance cache to avoid lock/contention
export TRANSFORMERS_CACHE=/scratch/xc1490/hf_cache/instances/$INSTANCE_ID
mkdir -p \"$TRANSFORMERS_CACHE\"

# Per-instance output directory to avoid concurrent writes
OUTPUT_DIR=/scratch/xc1490/projects/medvqa_2025/output/qwen25vl32b/$INSTANCE_ID
mkdir -p \"$OUTPUT_DIR\"

python3 test_qwen25vl32b_batch.py \
  --random_dataset \
  --random_seed $RANDOM_SEED \
  --model_path /scratch/xc1490/models/Qwen2.5-VL-32B-Instruct \
  --data_root /scratch/xc1490/projects/medvqa_2025/data/medvqa \
  --images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --output_dir \"$OUTPUT_DIR\"

echo \"Instance $INSTANCE_ID: Completed\"
"
