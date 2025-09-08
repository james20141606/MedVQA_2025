#!/usr/bin/env python3
"""
Evaluation script for GLM-4.1V on MedVQA datasets
"""
import os
import json
import argparse
import subprocess
import sys
from pathlib import Path


def create_test_dataset_config(dataset_name, test_file):
    """Create test dataset configuration for LLaMA-Factory"""
    config = {
        dataset_name: {
            "file_name": test_file,
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


def create_evaluation_config(dataset_name, model_path, output_file):
    """Create evaluation configuration for LLaMA-Factory"""
    config = {
        ### model
        "model_name_or_path": "GLM-4.1V-9B",
        "adapter_name_or_path": model_path,
        "use_fast_tokenizer": True,
        "flash_attn": "auto",
        "visual_inputs": True,
        
        ### method
        "stage": "sft",
        "do_predict": True,
        "finetuning_type": "lora",
        
        ### dataset
        "dataset": dataset_name,
        "template": "glm4",
        "cutoff_len": 2048,
        "max_samples": 10000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        ### output
        "output_dir": os.path.dirname(output_file),
        "overwrite_output_dir": True,
        
        ### eval
        "per_device_eval_batch_size": 1,
        "predict_with_generate": True,
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.9
    }
    return config


def convert_test_data_to_glm4v_format(test_file, output_file, image_base_path):
    """Convert test data to GLM-4.1V format"""
    print(f"Converting test data {test_file} to GLM-4.1V format...")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    glm4v_data = []
    for item in data:
        image_path = os.path.join(image_base_path, item['image'])
        
        # Create test format (only question, no answer)
        messages = [
            {
                "content": item['conversations'][0]['value'],  # human question
                "role": "user"
            }
        ]
        
        glm4v_data.append({
            "messages": messages,
            "images": [image_path],
            "id": item.get('id', len(glm4v_data)),
            "ground_truth": item['conversations'][1]['value']  # Store ground truth for evaluation
        })
    
    # Save converted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(glm4v_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} test samples")
    return len(data)


def evaluate_model(model_path, test_dataset, test_file, llamafactory_path, output_file):
    """Evaluate GLM-4.1V model on test dataset"""
    print(f"Evaluating model {model_path} on {test_dataset} dataset...")
    
    # Create dataset config
    dataset_config = create_test_dataset_config(test_dataset, test_file)
    dataset_config_file = f"/tmp/test_dataset_info_{test_dataset.lower()}.json"
    
    with open(dataset_config_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    # Create evaluation config
    eval_config = create_evaluation_config(test_dataset, model_path, output_file)
    eval_config_file = f"/tmp/eval_config_{test_dataset.lower()}.json"
    
    with open(eval_config_file, 'w', encoding='utf-8') as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)
    
    # Prepare evaluation command
    cmd = [
        "python", f"{llamafactory_path}/src/train.py",
        "--config", eval_config_file,
        "--dataset_info", dataset_config_file
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully!")
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Evaluate GLM-4.1V on MedVQA datasets')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (LoRA adapter)')
    parser.add_argument('--test_dataset', choices=['PVQA', 'SLAKE', 'RAD'], required=True,
                        help='Test dataset to evaluate on')
    parser.add_argument('--data_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa',
                        help='Directory containing original MedVQA data')
    parser.add_argument('--image_base', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images',
                        help='Base path for images')
    parser.add_argument('--llamafactory_path', default='/home/xc1490/LLaMA-Factory',
                        help='Path to LLaMA-Factory installation')
    parser.add_argument('--output_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/results',
                        help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Map dataset names to test files
    test_files = {
        'PVQA': 'test_pvqa.json',  # Note: you might need to create this from train_all.json
        'SLAKE': 'test_slake.json',
        'RAD': 'test_rad.json'
    }
    
    test_file = os.path.join(args.data_dir, test_files[args.test_dataset])
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} not found!")
        sys.exit(1)
    
    # Convert test data to GLM-4.1V format
    converted_test_file = os.path.join(args.output_dir, f"test_{args.test_dataset.lower()}_glm4v.json")
    convert_test_data_to_glm4v_format(test_file, converted_test_file, args.image_base)
    
    # Evaluate the model
    output_file = os.path.join(args.output_dir, f"predictions_{args.test_dataset.lower()}.json")
    success = evaluate_model(
        model_path=args.model_path,
        test_dataset=args.test_dataset,
        test_file=converted_test_file,
        llamafactory_path=args.llamafactory_path,
        output_file=output_file
    )
    
    if success:
        print(f"Successfully evaluated model on {args.test_dataset} dataset!")
        print(f"Results saved to: {output_file}")
    else:
        print(f"Failed to evaluate model on {args.test_dataset} dataset!")
        sys.exit(1)


if __name__ == '__main__':
    main()
