# 并行运行功能总结

## 已完成的工作

### 1. 模型验证和重试机制
为所有主要模型添加了智能验证和重试机制：

- **MMaDA**: 检测`<image>`响应，最多3次重试
- **Thyme**: 检查提取答案有效性，最多3次重试  
- **InternVL系列**: 检测拒绝回答模式，最多3次重试
- **GLM4V**: 通用验证逻辑，最多3次重试
- **Qwen系列**: 检查响应和提取答案，最多3次重试

### 2. 并行运行功能
为大型模型实现了并行运行机制：

#### Qwen 32B
- ✅ 随机数据集选择
- ✅ 样本顺序打乱
- ✅ 实时进度监控
- ✅ 多实例并行处理
- ✅ 批处理脚本: `jobs/qwen25vl32b_parallel.sbatch`
- ✅ 监控脚本: `bin/monitor_qwen32b_progress.py`

#### GLM4V
- ✅ 随机数据集选择
- ✅ 样本顺序打乱
- ✅ 实时进度监控
- ✅ 多实例并行处理
- ✅ 批处理脚本: `jobs/glm4v_parallel.sbatch`
- ✅ 监控脚本: `bin/monitor_glm4v_progress.py`

### 3. 评估脚本改进
- ✅ 修正了数据集大小统计（考虑重复样本）
- ✅ 添加了详细的完成状态报告
- ✅ 区分"未处理"和"处理失败"的样本
- ✅ 实时显示剩余样本数量

## 使用方法

### 启动并行运行

#### Qwen 32B
```bash
# 使用作业数组（推荐）
sbatch --array=0-5 jobs/qwen25vl32b_parallel.sbatch

# 或手动启动
sbatch jobs/qwen25vl32b_parallel.sbatch 1
sbatch jobs/qwen25vl32b_parallel.sbatch 2
```

#### GLM4V
```bash
# 使用作业数组（推荐）
sbatch --array=0-5 jobs/glm4v_parallel.sbatch

# 或手动启动
sbatch jobs/glm4v_parallel.sbatch 1
sbatch jobs/glm4v_parallel.sbatch 2
```

### 监控进度

#### Qwen 32B
```bash
cd bin
python3 monitor_qwen32b_progress.py
```

#### GLM4V
```bash
cd bin
python3 monitor_glm4v_progress.py
```

### 评估结果
```bash
cd bin
python3 evaluate_all_models.py
```

## 关键特性

### 1. 智能随机化
- 每个实例使用不同的随机种子
- 确保不同实例处理不同的数据集和样本
- 避免重复工作

### 2. 实时监控
- 每10个样本显示详细进度
- 包括剩余样本数、处理时间、预计完成时间
- 支持实时监控脚本

### 3. 错误处理
- 自动检测无效响应
- 最多3次重试机制
- 详细的错误日志

### 4. 重复检测
- 自动跳过已处理的样本
- 支持中断后恢复
- 安全的并行写入

## 预期效果

### 性能提升
- **理论加速比**: 6倍（使用6个并行实例）
- **实际加速比**: 3-4倍（考虑资源竞争）
- **完成时间**: 从48小时减少到12-16小时

### 质量提升
- 减少无效响应数量
- 提高评估结果准确性
- 减少手动重新处理

## 文件结构

```
medvqa_2025/
├── bin/
│   ├── test_qwen25vl32b_batch.py      # Qwen 32B测试脚本（已修改）
│   ├── test_glm4v_batch.py            # GLM4V测试脚本（已修改）
│   ├── monitor_qwen32b_progress.py     # Qwen 32B监控脚本
│   ├── monitor_glm4v_progress.py       # GLM4V监控脚本
│   └── evaluate_all_models.py         # 评估脚本（已改进）
├── jobs/
│   ├── qwen25vl32b_parallel.sbatch    # Qwen 32B并行批处理
│   └── glm4v_parallel.sbatch          # GLM4V并行批处理
├── MMaDA/
│   └── test_mmada_batch.py            # MMaDA测试脚本（已修改）
├── Thyme/
│   └── test_thyme_batch.py            # Thyme测试脚本（已修改）
└── README_*.md                        # 各种说明文档
```

## 注意事项

1. **资源管理**: 确保有足够的GPU资源
2. **存储空间**: 输出文件会持续增长
3. **网络带宽**: 多个实例同时下载模型可能影响速度
4. **监控**: 定期检查进度，确保所有实例正常运行

## 下一步建议

1. **扩展到其他模型**: 为其他大型模型添加并行运行功能
2. **优化资源分配**: 根据模型大小调整资源分配
3. **自动化管理**: 创建自动化的任务管理脚本
4. **性能分析**: 收集和分析性能数据，进一步优化


