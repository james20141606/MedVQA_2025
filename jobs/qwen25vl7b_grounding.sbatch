#!/bin/bash
#SBATCH --job-name=qwen7b_grounding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --time=24:00:00
#SBATCH --account=pr_60_tandon_advanced

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

cd /scratch/xc1490/projects/medvqa_2025/bin

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M_blip.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate Thyme



# Paths
INDEX=/scratch/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/slake/grounding_index.json
OUTPUT=/scratch/xc1490/projects/medvqa_2025/output/grounding/qwen25vl7b_slake.jsonl
mkdir -p \"$(dirname \$OUTPUT)\"

python3 test_qwen25vl7b_grounding.py \
  --index \$INDEX \
  --output \$OUTPUT \
  --device_map cuda:0 \
  --per_case_progress \
  --offline
"
