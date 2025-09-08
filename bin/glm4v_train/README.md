# GLM-4.1V Training Framework for MedVQA

完整的GLM-4.1V医学视觉问答训练框架，支持多种模式和训练方法。

## 功能特性

### 模型支持
- **GLM-4.1V-9B-Thinking**: 支持 `<think>...</think>` + `<answer>...</answer>` 格式
- **GLM-4.1V-9B-Base**: 基础版本，不使用thinking标签
- **GLM-4.5V**: 最新版本，简化格式

### 训练方式
- **LoRA SFT**: 高效参数微调
- **Full SFT**: 全参数微调
- **DPO/RL**: 基于人类偏好的强化学习

### 数据集支持
- **SLAKE**: 双语医学VQA数据集
- **VQA-RAD**: 放射学VQA数据集  
- **PathVQA**: 病理学VQA数据集
- **Combined**: 三个数据集的组合训练

## 目录结构

```
glm4v_train/
├── README.md                           # 本文档
├── data_conversion/                    # 数据转换模块
│   ├── __init__.py
│   ├── base_converter.py              # 基础转换类
│   ├── medvqa_converter.py            # MedVQA专用转换器
│   └── format_validator.py            # 格式验证工具
├── training/                          # 训练模块
│   ├── __init__.py
│   ├── lora_trainer.py                # LoRA训练器
│   ├── full_trainer.py                # 全参数训练器
│   ├── dpo_trainer.py                 # DPO训练器
│   └── config_templates.py            # 配置模板
├── evaluation/                        # 评估模块
│   ├── __init__.py
│   ├── evaluator.py                   # 评估器
│   └── metrics.py                     # 评估指标
├── utils/                             # 工具模块
│   ├── __init__.py
│   ├── data_utils.py                  # 数据处理工具
│   ├── model_utils.py                 # 模型工具
│   └── logging_utils.py               # 日志工具
├── configs/                           # 配置文件
│   ├── model_configs.yaml             # 模型配置
│   ├── training_configs.yaml          # 训练配置
│   └── dataset_configs.yaml           # 数据集配置
├── scripts/                           # 执行脚本
│   ├── convert_data.py                # 数据转换脚本
│   ├── train_model.py                 # 训练脚本
│   ├── evaluate_model.py              # 评估脚本
│   └── run_experiments.py             # 实验运行脚本
└── jobs/                              # SLURM作业脚本
    ├── lora_sft.sbatch                # LoRA SFT作业
    ├── full_sft.sbatch                # 全参数SFT作业
    ├── dpo_training.sbatch            # DPO训练作业
    └── evaluation.sbatch              # 评估作业
```

## 快速开始

### 1. 数据转换
```bash
# 转换单个数据集（thinking模式）
python scripts/convert_data.py --dataset slake --mode thinking --model_version 4.1v

# 转换组合数据集（base模式）  
python scripts/convert_data.py --dataset combined --mode base --model_version 4.1v

# 转换为4.5V格式
python scripts/convert_data.py --dataset combined --model_version 4.5v
```

### 2. LoRA训练
```bash
# 单数据集LoRA训练
python scripts/train_model.py --method lora --dataset slake --model_version 4.1v --mode thinking

# 组合数据集LoRA训练
python scripts/train_model.py --method lora --dataset combined --model_version 4.1v --mode base
```

### 3. DPO训练
```bash
# DPO偏好学习
python scripts/train_model.py --method dpo --dataset combined --model_version 4.1v --base_model path/to/sft/model
```

### 4. 评估
```bash
# 评估单个模型
python scripts/evaluate_model.py --model_path path/to/model --test_datasets slake,rad,pathvqa

# 批量评估
python scripts/run_experiments.py --mode evaluate --models_dir path/to/models
```

### 5. SLURM作业提交
```bash
# 提交LoRA训练作业
sbatch jobs/lora_sft.sbatch

# 提交DPO训练作业  
sbatch jobs/dpo_training.sbatch

# 提交评估作业
sbatch jobs/evaluation.sbatch
```

## 配置说明

### 模型配置 (configs/model_configs.yaml)
- 支持的模型版本和路径
- 模型特定的参数设置
- Tokenizer配置

### 训练配置 (configs/training_configs.yaml)  
- LoRA参数设置
- 训练超参数
- 优化器配置

### 数据集配置 (configs/dataset_configs.yaml)
- 数据集路径映射
- 预处理参数
- 评估设置

## 高级功能

### 多轮对话支持
框架支持医学问诊中的多轮对话场景：
```json
{
  "messages": [
    {"role": "user", "content": "<image>What abnormality do you see?"},
    {"role": "assistant", "content": "<answer>Cardiomegaly</answer>"},
    {"role": "user", "content": "What could be the cause?"},
    {"role": "assistant", "content": "<answer>Possible causes include hypertension, heart failure, or valvular disease</answer>"}
  ],
  "images": ["chest_xray.jpg"]
}
```

### 偏好学习数据
支持DPO训练的偏好对数据格式：
```json
{
  "messages": [
    {"role": "user", "content": "<image>Identify the main finding"}
  ],
  "images": ["image.jpg"],
  "chosen": "Pneumothorax in the right lung",
  "rejected": "Normal chest radiograph"
}
```

### 术语标准化
内置医学术语标准化功能：
- 统一同义词表达
- 规范化诊断术语
- 支持多语言术语映射

## 性能优化

- **梯度检查点**: 减少显存占用
- **混合精度**: 加速训练过程
- **数据并行**: 支持多GPU训练
- **模型并行**: 支持大模型分布式训练

## 监控和日志

- **TensorBoard**: 训练过程可视化
- **Weights & Biases**: 实验跟踪
- **详细日志**: 完整的训练和评估日志
- **错误恢复**: 自动检查点恢复

## 注意事项

1. **显存要求**: LoRA训练至少需要24GB显存，全参数训练需要80GB+
2. **数据路径**: 确保图像路径正确，支持相对路径和绝对路径
3. **模型权限**: 某些模型需要申请使用权限
4. **版本兼容**: 注意LLaMA-Factory版本与模型的兼容性

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减小batch_size或启用梯度检查点
2. **数据加载错误**: 检查图像路径和格式
3. **模型加载失败**: 确认模型路径和权限
4. **训练不收敛**: 调整学习率和warmup策略

### 调试模式
```bash
# 启用调试模式
python scripts/train_model.py --debug --log_level DEBUG
```

## 贡献指南

欢迎提交Issue和Pull Request来改进这个框架。

## 许可证

Apache 2.0 License
