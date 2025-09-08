#!/bin/bash

#SBATCH --job-name=mmada
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'  
#SBATCH --time=48:00:00
#SBATCH --account=pr_60_tandon_advanced

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

cd /scratch/xc1490/projects/medvqa_2025/MMaDA

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M_blip.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate diff_vlm

python3 test_mmada_batch.py \
  --all \
  --model_path Gen-Verse/MMaDA-8B-MixCoT \
  --data_root /scratch/xc1490/projects/medvqa_2025/data/medvqa \
  --images_root /scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images \
  --output_dir /scratch/xc1490/projects/medvqa_2025/output/mmada
"
