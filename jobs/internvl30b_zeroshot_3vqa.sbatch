#!/bin/bash

#SBATCH --job-name=internvl30b
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'  
#SBATCH --time=12:00:00
#SBATCH --account=pr_60_tandon_advanced

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

cd /scratch/xc1490/projects/medvqa_2025

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M_blip.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate internvl

python3 bin/test_internvl30b_batch.py \
  --datasets pvqa slake rad \
  --model_path OpenGVLab/InternVL3_5-30B-A3B \
  --data_root /scratch/xc1490/projects/medvqa_2025/data/medvqa \
  --pvqa_images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --slake_images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --rad_images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --output_dir /scratch/xc1490/projects/medvqa_2025/output/internvl30b
"
