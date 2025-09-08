# 模型并行运行指南 (Qwen 32B & GLM4V)

## 概述

为了加速大型模型的处理速度，我们实现了并行运行机制，支持：
- 随机选择数据集
- 样本顺序打乱
- 实时进度监控
- 多实例并行处理

## 支持的模型

### 1. Qwen 32B
- 模型路径: `Qwen/Qwen2.5-VL-32B-Instruct`
- 输出目录: `/scratch/xc1490/projects/medvqa_2025/output/qwen25vl32b`
- 批处理脚本: `jobs/qwen25vl32b_parallel.sbatch`
- 监控脚本: `bin/monitor_qwen32b_progress.py`

### 2. GLM4V
- 模型路径: `zai-org/GLM-4.1V-9B-Thinking`
- 输出目录: `/scratch/xc1490/projects/medvqa_2025/output/glm4v`
- 批处理脚本: `jobs/glm4v_parallel.sbatch`
- 监控脚本: `bin/monitor_glm4v_progress.py`

## 新功能

### 1. 随机数据集选择
- 每次运行随机选择一个数据集（PVQA、SLAKE、RAD）
- 使用不同的随机种子确保不同实例处理不同的数据集

### 2. 样本顺序打乱
- 每个数据集内的样本顺序会被随机打乱
- 确保并行实例处理不同的样本，避免重复工作

### 3. 实时进度监控
- 每10个样本显示详细进度信息
- 包括剩余样本数、平均处理时间、预计完成时间

### 4. 并行运行支持
- 支持多个实例同时运行
- 每个实例使用不同的随机种子
- 自动避免重复处理

## 使用方法

### Qwen 32B

#### 方法1：使用作业数组（推荐）
```bash
# 启动6个并行实例（0-5）
sbatch --array=0-5 jobs/qwen25vl32b_parallel.sbatch
```

#### 方法2：手动启动多个实例
```bash
# 启动实例1
sbatch jobs/qwen25vl32b_parallel.sbatch 1

# 启动实例2
sbatch jobs/qwen25vl32b_parallel.sbatch 2

# 启动实例3
sbatch jobs/qwen25vl32b_parallel.sbatch 3
```

#### 方法3：直接运行（测试用）
```bash
cd bin
python3 test_qwen25vl32b_batch.py --random_dataset --random_seed 42
```

### GLM4V

#### 方法1：使用作业数组（推荐）
```bash
# 启动6个并行实例（0-5）
sbatch --array=0-5 jobs/glm4v_parallel.sbatch
```

#### 方法2：手动启动多个实例
```bash
# 启动实例1
sbatch jobs/glm4v_parallel.sbatch 1

# 启动实例2
sbatch jobs/glm4v_parallel.sbatch 2

# 启动实例3
sbatch jobs/glm4v_parallel.sbatch 3
```

#### 方法3：直接运行（测试用）
```bash
cd bin
python3 test_glm4v_batch.py --random_dataset --random_seed 42
```

## 进度监控

### Qwen 32B 监控
```bash
cd bin
python3 monitor_qwen32b_progress.py
```

### GLM4V 监控
```bash
cd bin
python3 monitor_glm4v_progress.py
```

监控脚本会显示：
- 每个数据集的当前进度
- 剩余样本数量
- 完成百分比
- 总体进度

### 监控输出示例
```
================================================================================
QWEN 32B PROGRESS MONITOR
================================================================================
Dataset    Current    Expected   Remaining  Completion Status    
--------------------------------------------------------------------------------
pvqa       1234       6176       4942       20.0%      🔄 Running
slake      500        1061       561        47.1%      🔄 Running
rad        200        451        251        44.3%      🔄 Running
--------------------------------------------------------------------------------
TOTAL      1934       7688       5754       25.2%

Detailed Breakdown:
  pvqa: 4942 samples remaining
  slake: 561 samples remaining
  rad: 251 samples remaining

⏳ 5754 total samples remaining across all datasets
```

## 随机种子说明

每个实例使用不同的随机种子：
- 实例0: 随机种子42
- 实例1: 随机种子43
- 实例2: 随机种子44
- ...

这确保了：
1. 不同实例选择不同的数据集
2. 同一数据集内的样本顺序不同
3. 避免重复处理相同的样本

## 输出文件

### Qwen 32B
所有实例的输出都保存到同一个目录：
```
/scratch/xc1490/projects/medvqa_2025/output/qwen25vl32b/
├── test_pvqa.json
├── test_slake.json
└── test_rad.json
```

### GLM4V
所有实例的输出都保存到同一个目录：
```
/scratch/xc1490/projects/medvqa_2025/output/glm4v/
├── test_pvqa.json
├── test_slake.json
└── test_rad.json
```

由于使用了重复检测机制，多个实例可以安全地写入同一个文件。

## 性能优化建议

### 1. 实例数量
- 建议使用3-6个并行实例
- 每个实例处理一个数据集
- 避免过多实例导致资源竞争

### 2. 监控频率
- 监控脚本每60秒更新一次
- 可以根据需要调整更新频率

### 3. 资源分配
- 每个实例使用1个GPU
- 确保有足够的GPU资源

## 故障排除

### 1. 实例冲突
如果多个实例同时处理同一个样本：
- 系统会自动跳过已处理的样本
- 不会影响结果正确性

### 2. 进度监控
如果监控脚本显示异常：
- 检查输出目录权限
- 确认文件路径正确

### 3. 随机种子
如果需要重现特定结果：
- 使用相同的随机种子
- 确保环境一致

## 预期加速效果

使用6个并行实例的预期加速：
- **理论加速比**: 6倍
- **实际加速比**: 3-4倍（考虑资源竞争和重复检测开销）
- **完成时间**: 从48小时减少到12-16小时

## 注意事项

1. **资源使用**: 确保有足够的GPU资源
2. **存储空间**: 输出文件会持续增长
3. **网络带宽**: 多个实例同时下载模型可能影响速度
4. **监控**: 定期检查进度，确保所有实例正常运行
