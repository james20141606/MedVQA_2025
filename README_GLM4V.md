# GLM-4.1V Training on MedVQA Datasets

This directory contains scripts and configurations for fine-tuning GLM-4.1V on MedVQA datasets using LLaMA-Factory.

## Overview

The project supports two training approaches:
1. **Individual Training**: Train separate models on each dataset (PVQA, SLAKE, RAD)
2. **Combined Training**: Train a single unified model on all three datasets

## Data Format

Your original MedVQA data format:
```json
{
    "id": "train_0422",
    "image": "pvqa/train/train_0422.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhere are liver stem cells located?"},
        {"from": "gpt", "value": "in the canals of hering"}
    ],
    "answer_type": "OPEN",
    "modality": "pathology"
}
```

Converted GLM-4.1V format:
```json
{
    "messages": [
        {"content": "<image>\nWhere are liver stem cells located?", "role": "user"},
        {"content": "<answer>in the canals of hering</answer>", "role": "assistant"}
    ],
    "images": ["/path/to/image.jpg"]
}
```

**Note about `<think>` tags**: Since your original data doesn't include reasoning steps, we omit the `<think>` tags and use only `<answer>` tags for GLM-4.1V fine-tuning.

## Scripts

### Data Conversion
- `scripts/convert_medvqa_to_glm4v.py`: Converts MedVQA format to GLM-4.1V format

### Training
- `scripts/train_glm4v_individual.py`: Train individual models
- `scripts/train_glm4v_combined.py`: Train combined model

### Evaluation
- `scripts/evaluate_glm4v.py`: Evaluate trained models

### Job Management
- `scripts/run_glm4v_experiments.py`: Master script for job submission

## SLURM Jobs

### Training Jobs
- `jobs/glm4v_individual_train.sbatch`: Array job for individual training (3 models)
- `jobs/glm4v_combined_train.sbatch`: Single job for combined training

### Evaluation Job
- `jobs/glm4v_evaluate.sbatch`: Array job for comprehensive evaluation (12 combinations)

## Usage

### Quick Start
```bash
# Run both individual and combined training with evaluation
python scripts/run_glm4v_experiments.py --mode both

# Run only individual training
python scripts/run_glm4v_experiments.py --mode individual

# Run only combined training
python scripts/run_glm4v_experiments.py --mode combined

# Run only evaluation (if models already exist)
python scripts/run_glm4v_experiments.py --eval_only
```

### Manual Execution

1. **Convert Data**:
```bash
# Individual datasets
python scripts/convert_medvqa_to_glm4v.py --mode individual

# Combined dataset
python scripts/convert_medvqa_to_glm4v.py --mode combined
```

2. **Train Models**:
```bash
# Individual training
sbatch jobs/glm4v_individual_train.sbatch

# Combined training
sbatch jobs/glm4v_combined_train.sbatch
```

3. **Evaluate Models**:
```bash
sbatch jobs/glm4v_evaluate.sbatch
```

## Directory Structure

```
medvqa_2025/
├── data/
│   ├── medvqa/                    # Original MedVQA data
│   │   ├── train_all.json         # PVQA training data
│   │   ├── test_slake.json        # SLAKE test data
│   │   ├── test_rad.json          # RAD test data
│   │   └── 3vqa/images/           # Image files
│   └── glm4v_format/              # Converted GLM-4.1V format data
├── scripts/                       # Python scripts
├── jobs/                          # SLURM job scripts
├── models/                        # Trained models (created during training)
│   ├── glm4v_pvqa/
│   ├── glm4v_slake/
│   ├── glm4v_rad/
│   └── glm4v_combined/
├── results/                       # Evaluation results (created during evaluation)
└── logs/                          # Job logs (created during execution)
```

## Configuration

### Training Parameters
- **Model**: GLM-4.1V-9B
- **Method**: LoRA fine-tuning
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Learning Rate**: 5e-5
- **Epochs**: 3
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 8 steps

### Hardware Requirements
- **Individual Training**: 1 GPU, 64GB RAM, 24 hours
- **Combined Training**: 1 GPU, 128GB RAM, 48 hours
- **Evaluation**: 1 GPU, 32GB RAM, 12 hours

## Evaluation Matrix

The evaluation script tests all combinations:

| Model | PVQA Test | SLAKE Test | RAD Test |
|-------|-----------|------------|----------|
| PVQA Individual | ✓ | ✓ | ✓ |
| SLAKE Individual | ✓ | ✓ | ✓ |
| RAD Individual | ✓ | ✓ | ✓ |
| Combined | ✓ | ✓ | ✓ |

## Monitoring

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/glm4v_individual_*.out
tail -f logs/glm4v_combined_*.out
tail -f logs/glm4v_eval_*.out

# Cancel jobs if needed
scancel <job_id>
```

## Troubleshooting

### Common Issues

1. **Missing LLaMA-Factory**: Update `--llamafactory_path` in scripts
2. **Image Path Issues**: Verify `--image_base` points to correct directory
3. **Memory Issues**: Reduce batch size or increase memory allocation
4. **CUDA Issues**: Check GPU availability and CUDA modules

### Data Issues

1. **Missing Test Data**: You may need to create test splits from training data
2. **Image Paths**: Ensure image paths in JSON match actual file locations
3. **Format Errors**: Check JSON syntax and required fields

## Results

Results will be saved in:
- `results/*/predictions_*.json`: Model predictions
- `models/*/`: Trained model checkpoints
- `logs/`: Training and evaluation logs

## Dependencies

- Python 3.9+
- LLaMA-Factory
- PyTorch with CUDA support
- Transformers
- Required Python packages (see LLaMA-Factory requirements)
