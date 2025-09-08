#!/usr/bin/env python3
"""
Convert MedVQA data format to GLM-4.1V format with generated thinking process
"""
import json
import os
import argparse
import re
from collections import defaultdict


def generate_simple_thinking(question, answer, modality=None, answer_type=None):
    """Generate simple thinking process based on question and answer patterns"""
    
    # Clean question text
    question_clean = question.replace("<image>", "").strip()
    question_lower = question_clean.lower()
    
    # Pattern-based thinking generation
    thinking_templates = []
    
    # Question type detection
    if any(word in question_lower for word in ["what is", "what are", "identify", "name"]):
        thinking_templates.append(f"I need to identify what is shown in this {modality or 'medical'} image.")
    
    elif any(word in question_lower for word in ["where", "location", "located"]):
        thinking_templates.append(f"I need to determine the location or position in this image.")
    
    elif any(word in question_lower for word in ["how many", "count", "number"]):
        thinking_templates.append(f"I need to count the items or structures in this image.")
    
    elif any(word in question_lower for word in ["is there", "are there", "present", "evidence"]):
        thinking_templates.append(f"I need to look for evidence of specific findings in this image.")
    
    elif "yes" in answer.lower() or "no" in answer.lower():
        thinking_templates.append(f"This is a yes/no question about the medical image.")
    
    else:
        thinking_templates.append(f"I need to analyze this {modality or 'medical'} image to answer the question.")
    
    # Add modality-specific thinking
    if modality:
        modality_lower = modality.lower()
        if "ct" in modality_lower:
            thinking_templates.append("Looking at this CT scan...")
        elif "mri" in modality_lower:
            thinking_templates.append("Examining this MRI image...")
        elif "x-ray" in modality_lower or "xray" in modality_lower:
            thinking_templates.append("Analyzing this X-ray...")
        elif "pathology" in modality_lower:
            thinking_templates.append("Examining this pathology slide...")
        elif "ultrasound" in modality_lower:
            thinking_templates.append("Looking at this ultrasound image...")
    
    # Add answer-based thinking
    if answer_type == "CLOSED":
        if "yes" in answer.lower():
            thinking_templates.append("I can see evidence supporting a positive answer.")
        elif "no" in answer.lower():
            thinking_templates.append("I don't see evidence supporting this finding.")
    else:  # OPEN
        if len(answer.split()) <= 3:  # Short answer
            thinking_templates.append(f"The answer appears to be {answer}.")
        else:  # Longer answer
            thinking_templates.append("Based on the image features, I can provide a detailed answer.")
    
    # Combine templates
    if len(thinking_templates) == 1:
        thinking = thinking_templates[0]
    else:
        thinking = " ".join(thinking_templates[:2])  # Use first 2 templates
    
    return thinking


def convert_single_dataset_with_think(input_file, output_file, image_base_path):
    """Convert a single MedVQA dataset to GLM-4.1V format with thinking"""
    print(f"Converting {input_file} to {output_file} (with thinking)")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group conversations by image
    image_conversations = defaultdict(list)
    
    for item in data:
        image_path = os.path.join(image_base_path, item['image'])
        
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        modality = item.get('modality', '')
        answer_type = item.get('answer_type', '')
        
        # Generate thinking process
        thinking = generate_simple_thinking(question, answer, modality, answer_type)
        
        conversation = {
            "content": question,  # human question
            "role": "user"
        }
        response = {
            "content": f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>",
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
    
    print(f"Converted {len(data)} samples to {len(glm4v_data)} conversations (with thinking)")
    return len(glm4v_data)


def convert_combined_dataset_with_think(input_files, output_file, image_base_path):
    """Convert multiple MedVQA datasets to a single GLM-4.1V format file with thinking"""
    print(f"Converting combined datasets to {output_file} (with thinking)")
    
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
            
            question = item['conversations'][0]['value']
            answer = item['conversations'][1]['value']
            modality = item.get('modality', '')
            answer_type = item.get('answer_type', '')
            
            # Generate thinking process
            thinking = generate_simple_thinking(question, answer, modality, answer_type)
            
            conversation = {
                "content": question,  # human question
                "role": "user"
            }
            response = {
                "content": f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>",
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
    
    print(f"Combined {total_samples} samples to {len(all_data)} conversations (with thinking)")
    return len(all_data)


def main():
    parser = argparse.ArgumentParser(description='Convert MedVQA data to GLM-4.1V format with generated thinking')
    parser.add_argument('--mode', choices=['individual', 'combined'], required=True,
                        help='Convert individual datasets or combine all')
    parser.add_argument('--data_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa',
                        help='MedVQA data directory')
    parser.add_argument('--output_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/glm4v_format_with_think',
                        help='Output directory for converted data')
    parser.add_argument('--image_base', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images',
                        help='Base path for images')
    
    args = parser.parse_args()
    
    if args.mode == 'individual':
        # Convert each dataset separately
        datasets = [
            ('train_all.json', 'train_pvqa_with_think.json'),
            ('test_slake.json', 'train_slake_with_think.json'),
            ('test_rad.json', 'train_rad_with_think.json')
        ]
        
        for input_name, output_name in datasets:
            input_file = os.path.join(args.data_dir, input_name)
            output_file = os.path.join(args.output_dir, output_name)
            if os.path.exists(input_file):
                convert_single_dataset_with_think(input_file, output_file, args.image_base)
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
        
        output_file = os.path.join(args.output_dir, 'train_combined_with_think.json')
        convert_combined_dataset_with_think(existing_files, output_file, args.image_base)


if __name__ == '__main__':
    main()
