#!/bin/bash

#SBATCH --job-name=glm4v_parallel
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
    echo "Usage: sbatch --array=0-5 glm4v_parallel.sbatch"
    echo "   or: sbatch glm4v_parallel.sbatch 1"
    exit 1
fi

echo "Starting GLM4V parallel instance $INSTANCE_ID"

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

python3 test_glm4v_batch.py \
  --random_dataset \
  --random_seed $RANDOM_SEED \
  --model_path zai-org/GLM-4.1V-9B-Thinking \
  --data_root /scratch/xc1490/projects/medvqa_2025/data/medvqa \
  --images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --output_dir /scratch/xc1490/projects/medvqa_2025/output/glm4v

echo \"Instance $INSTANCE_ID: Completed\"
"


