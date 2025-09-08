#!/usr/bin/env python3
"""
Convert MedVQA data format to GLM-4.1V fine-tuning format
"""
import json
import os
import argparse
from collections import defaultdict


def convert_single_dataset(input_file, output_file, image_base_path):
    """Convert a single MedVQA dataset to GLM-4.1V format"""
    print(f"Converting {input_file} to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group conversations by image
    image_conversations = defaultdict(list)
    
    for item in data:
        image_path = os.path.join(image_base_path, item['image'])
        conversation = {
            "content": item['conversations'][0]['value'],  # human question
            "role": "user"
        }
        response = {
            "content": f"<answer>{item['conversations'][1]['value']}</answer>",  # assistant answer without <think>
            "role": "assistant"
        }
        
        image_conversations[image_path].extend([conversation, response])
    
    # Convert to GLM-4.1V format
    glm4v_data = []
    for image_path, messages in image_conversations.items():
        glm4v_data.append({
            "messages": messages,
            "images": [image_path]
        })
    
    # Save converted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(glm4v_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} samples to {len(glm4v_data)} conversations")
    return len(glm4v_data)


def convert_combined_dataset(input_files, output_file, image_base_path):
    """Convert multiple MedVQA datasets to a single GLM-4.1V format file"""
    print(f"Converting combined datasets to {output_file}")
    
    all_data = []
    total_samples = 0
    
    for input_file in input_files:
        print(f"Processing {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group conversations by image
        image_conversations = defaultdict(list)
        
        for item in data:
            image_path = os.path.join(image_base_path, item['image'])
            conversation = {
                "content": item['conversations'][0]['value'],  # human question
                "role": "user"
            }
            response = {
                "content": f"<answer>{item['conversations'][1]['value']}</answer>",  # assistant answer without <think>
                "role": "assistant"
            }
            
            image_conversations[image_path].extend([conversation, response])
        
        # Convert to GLM-4.1V format
        for image_path, messages in image_conversations.items():
            all_data.append({
                "messages": messages,
                "images": [image_path]
            })
        
        total_samples += len(data)
        print(f"Added {len(data)} samples from {input_file}")
    
    # Save combined data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Combined {total_samples} samples to {len(all_data)} conversations")
    return len(all_data)


def main():
    parser = argparse.ArgumentParser(description='Convert MedVQA data to GLM-4.1V format')
    parser.add_argument('--mode', choices=['individual', 'combined'], required=True,
                        help='Convert individual datasets or combine all')
    parser.add_argument('--data_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa',
                        help='MedVQA data directory')
    parser.add_argument('--output_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/glm4v_format',
                        help='Output directory for converted data')
    parser.add_argument('--image_base', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images',
                        help='Base path for images')
    
    args = parser.parse_args()
    
    if args.mode == 'individual':
        # Convert each dataset separately
        datasets = [
            ('train_all.json', 'train_pvqa.json'),
            ('test_slake.json', 'train_slake.json'),
            ('test_rad.json', 'train_rad.json')
        ]
        
        for input_name, output_name in datasets:
            input_file = os.path.join(args.data_dir, input_name)
            output_file = os.path.join(args.output_dir, output_name)
            if os.path.exists(input_file):
                convert_single_dataset(input_file, output_file, args.image_base)
            else:
                print(f"Warning: {input_file} not found, skipping...")
    
    elif args.mode == 'combined':
        # Combine all datasets
        input_files = [
            os.path.join(args.data_dir, 'train_all.json'),
            os.path.join(args.data_dir, 'test_slake.json'),
            os.path.join(args.data_dir, 'test_rad.json')
        ]
        
        # Filter existing files
        existing_files = [f for f in input_files if os.path.exists(f)]
        if not existing_files:
            print("Error: No input files found!")
            return
        
        output_file = os.path.join(args.output_dir, 'train_combined.json')
        convert_combined_dataset(existing_files, output_file, args.image_base)


if __name__ == '__main__':
    main()
