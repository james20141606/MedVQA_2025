#!/usr/bin/env python3
"""
Training script for GLM-4.1V on individual MedVQA datasets
"""
import os
import json
import argparse
import subprocess
import sys


def create_dataset_config(dataset_name, train_file):
    """Create dataset configuration for LLaMA-Factory"""
    config = {
        dataset_name: {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }
    return config


def create_training_config(dataset_name, model_name="GLM-4.1V-9B", output_dir=None):
    """Create training configuration for LLaMA-Factory"""
    if output_dir is None:
        output_dir = f"/home/xc1490/xc1490/projects/medvqa_2025/models/glm4v_{dataset_name.lower()}"
    
    config = {
        ### model
        "model_name_or_path": model_name,
        "use_fast_tokenizer": True,
        "flash_attn": "auto",
        "visual_inputs": True,
        
        ### method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        
        ### dataset
        "dataset": dataset_name,
        "template": "glm4",
        "cutoff_len": 2048,
        "max_samples": 100000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        ### output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        ### train
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5.0e-5,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        
        ### eval
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss"
    }
    return config


def train_individual_model(dataset_name, train_file, llamafactory_path, output_dir=None):
    """Train GLM-4.1V model on individual dataset"""
    print(f"Training GLM-4.1V on {dataset_name} dataset...")
    
    # Create dataset config
    dataset_config = create_dataset_config(dataset_name, train_file)
    dataset_config_file = f"/tmp/dataset_info_{dataset_name.lower()}.json"
    
    with open(dataset_config_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    # Create training config
    training_config = create_training_config(dataset_name, output_dir=output_dir)
    training_config_file = f"/tmp/training_config_{dataset_name.lower()}.json"
    
    with open(training_config_file, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    
    # Prepare training command
    cmd = [
        "python", f"{llamafactory_path}/src/train.py",
        "--config", training_config_file,
        "--dataset_info", dataset_config_file
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Train GLM-4.1V on individual MedVQA datasets')
    parser.add_argument('--dataset', choices=['PVQA', 'SLAKE', 'RAD'], required=True,
                        help='Dataset to train on')
    parser.add_argument('--data_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/glm4v_format',
                        help='Directory containing converted GLM-4.1V format data')
    parser.add_argument('--llamafactory_path', default='/home/xc1490/LLaMA-Factory',
                        help='Path to LLaMA-Factory installation')
    parser.add_argument('--output_dir', 
                        help='Output directory for trained model (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Map dataset names to files
    dataset_files = {
        'PVQA': 'train_pvqa.json',
        'SLAKE': 'train_slake.json',
        'RAD': 'train_rad.json'
    }
    
    train_file = os.path.join(args.data_dir, dataset_files[args.dataset])
    
    if not os.path.exists(train_file):
        print(f"Error: Training file {train_file} not found!")
        print("Please run the data conversion script first.")
        sys.exit(1)
    
    # Train the model
    success = train_individual_model(
        dataset_name=args.dataset,
        train_file=train_file,
        llamafactory_path=args.llamafactory_path,
        output_dir=args.output_dir
    )
    
    if success:
        print(f"Successfully trained GLM-4.1V model on {args.dataset} dataset!")
    else:
        print(f"Failed to train GLM-4.1V model on {args.dataset} dataset!")
        sys.exit(1)


if __name__ == '__main__':
    main()
