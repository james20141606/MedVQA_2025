#!/bin/bash

#SBATCH --job-name=glm45v_parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --constraint='a100|h100'  
#SBATCH --time=48:00:00
#SBATCH --account=pr_60_tandon_advanced



cd /home/xc1490/xc1490/projects/medvqa_2025/bin

# Use different random seeds for different instances to ensure different dataset selection
# and sample shuffling
RANDOM_SEED=$((42 + $INSTANCE_ID))

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M_blip.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate Thyme

echo \"Instance $INSTANCE_ID: Using random seed $RANDOM_SEED\"

# Isolated per-instance cache to avoid lock/contention
export TRANSFORMERS_CACHE=/scratch/xc1490/hf_cache/instances/$INSTANCE_ID
mkdir -p \"$TRANSFORMERS_CACHE\"

# Per-instance output directory to avoid concurrent writes
OUTPUT_DIR=/scratch/xc1490/projects/medvqa_2025/output/glm45v/$INSTANCE_ID
mkdir -p /scratch/xc1490/projects/medvqa_2025/output/glm45v

python3 test_glm45v_batch.py \
  --random_dataset \
  --model_path zai-org/GLM-4.5V \
  --data_root /scratch/xc1490/projects/medvqa_2025/data/medvqa \
  --images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --output_dir /scratch/xc1490/projects/medvqa_2025/output/glm45v

echo \"Instance $INSTANCE_ID: Completed\"
"
