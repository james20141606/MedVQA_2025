#!/bin/bash

#SBATCH --job-name=janus_ground
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --time=2:00:00
#SBATCH --account=pr_60_tandon_advanced

set -euo pipefail

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M_blip.ext3

singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate janus



INDEX=/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/slake/grounding_index.json
OUTPUT=/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/janus_slake.jsonl

mkdir -p \$(dirname \"\$OUTPUT\")
mkdir -p /home/xc1490/xc1490/projects/medvqa_2025/output/slurm

python /home/xc1490/xc1490/projects/medvqa_2025/bin/test_janus_grounding.py \\
  --index \"\$INDEX\" \\
  --output \"\$OUTPUT\" \\
  --max_new_tokens 512 \\
  --per_case_progress
"
