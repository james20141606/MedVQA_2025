#!/bin/bash

# 批量提交所有模型的 SLURM 作业脚本
# 使用方法: ./submit_all_models.sh

echo "=== 开始提交所有模型的 SLURM 作业 ==="
echo ""

# 切换到作业脚本目录
cd /scratch/xc1490/projects/medvqa_2025/jobs

# 提交 InternVL 系列 (使用 internvl conda 环境)
echo "--- 提交 InternVL 系列作业 ---"
echo "提交 InternVL 8B..."
sbatch internvl8b_zeroshot_3vqa.sbatch
sleep 2

echo "提交 InternVL 30B..."
sbatch internvl30b_zeroshot_3vqa.sbatch
sleep 2

echo "提交 InternVL 38B (2 GPU)..."
sbatch internvl38b_zeroshot_3vqa.sbatch
sleep 2

echo ""

# 提交 QwenVL 系列 (使用 Thyme conda 环境)
echo "--- 提交 QwenVL 系列作业 ---"
echo "提交 QwenVL 7B..."
sbatch qwenvl7b_zeroshot_3vqa.sbatch
sleep 2

echo "提交 QwenVL 32B..."
sbatch qwenvl32b_zeroshot_3vqa.sbatch
sleep 2

echo ""

# 提交 GLM4V (使用 Thyme conda 环境)
echo "--- 提交 GLM4V 作业 ---"
echo "提交 GLM4V 9B..."
sbatch glm4v_zeroshot_3vqa.sbatch
sleep 2

echo ""

# 提交 Janus (使用 janus conda 环境)
echo "--- 提交 Janus 作业 ---"
echo "提交 Janus..."
sbatch janus_zeroshot_3vqa.sbatch
sleep 2

echo ""

# 提交 Thyme (使用 Thyme conda 环境)
echo "--- 提交 Thyme 作业 ---"
echo "提交 Thyme..."
sbatch thyme_zeroshot_3vqa.sbatch
sleep 2

echo ""

echo "=== 所有作业提交完成 ==="
echo ""
echo "使用以下命令查看作业状态:"
echo "squeue -u $USER"
echo ""
echo "使用以下命令查看特定作业的日志:"
echo "tail -f slurm-<job_id>.out"
echo ""
echo "注意:"
echo "- InternVL 38B 使用 2 GPU"
echo "- 其他模型使用 1 GPU"
echo "- 不同模型使用不同的 conda 环境"
echo "- 建议先检查资源可用性再提交"
