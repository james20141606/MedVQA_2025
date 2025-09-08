#!/bin/bash

#SBATCH --job-name=janus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'  
#SBATCH --time=2:00:00
#SBATCH --account=pr_60_tandon_advanced

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

# 启动server到后台
cd /scratch/xc1490/projects/medvqa_2025/
echo "Starting Janus server..."
python /scratch/xc1490/projects/medvqa_2025/bin/janus_server.py > server.log 2>&1 &
SERVER_PID=$!

# 等待5000端口就绪
wait_for_port() {
  local count=0
  while ! nc -z 127.0.0.1 5000; do
    sleep 5
    count=$((count + 1))
    echo "Waiting for port 5000... (attempt $count)"
    if [ $count -gt 60 ]; then
      echo "Timeout waiting for port 5000. Server may have failed to start."
      echo "Server log:"
      tail -20 server.log
      exit 1
    fi
  done
}

wait_for_port
echo "Port 5000 is ready!"

# 等待模型加载（减少等待时间）
echo "Waiting 120s for model to fully load..."
sleep 120

# 检查服务器是否还在运行
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server process died. Server log:"
    tail -20 server.log
    exit 1
fi

echo "Server ready, starting client..."
echo "Client output will be shown below:"

# 直接运行客户端，不重定向输出
python /scratch/xc1490/projects/medvqa_2025/bin/test_janus_batch_client.py
"
