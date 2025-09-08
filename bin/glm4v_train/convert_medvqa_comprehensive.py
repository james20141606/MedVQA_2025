#!/usr/bin/env python3
"""
Comprehensive MedVQA to GLM-4.1V/4.5V conversion script
Supports SLAKE, VQA-RAD, PathVQA datasets with proper formatting
"""
import json
import os
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any


class MedVQAConverter:
    def __init__(self, use_thinking: bool = True, model_version: str = "4.1v"):
        self.use_thinking = use_thinking
        self.model_version = model_version.lower()
        
        # Terminology standardization mapping
        self.term_mapping = {
            "enlarged heart": "Cardiomegaly",
            "pleural effusions": "Pleural effusion", 
            "pleural fluid": "Pleural effusion",
            "heart enlargement": "Cardiomegaly",
            "cardiac enlargement": "Cardiomegaly",
            "pneumothorax": "Pneumothorax",
            "lung collapse": "Atelectasis",
            "consolidation": "Consolidation",
            "infiltrate": "Infiltration",
            "nodule": "Pulmonary nodule",
            "mass": "Mass lesion",
            "fracture": "Fracture",
            "normal": "Normal",
            "no abnormality": "Normal",
            "unremarkable": "Normal"
        }
        
        # Thinking templates for different question types
        self.thinking_templates = {
            "identification": "I need to identify and analyze the key findings in this medical image.",
            "location": "I need to determine the anatomical location of the finding.",
            "count": "I need to count the specific structures or findings in the image.", 
            "presence": "I need to examine the image for evidence of the specified condition.",
            "comparison": "I need to compare different regions or findings in the image.",
            "diagnosis": "I need to analyze the imaging findings to determine the most likely diagnosis.",
            "modality": "I need to identify the imaging modality based on the image characteristics.",
            "organ": "I need to identify the organ system shown in this image.",
            "pathology": "I need to examine the pathological changes in this tissue sample.",
            "choice": "I need to analyze the options and select the most appropriate answer."
        }
    
    def standardize_answer(self, answer: str) -> str:
        """Standardize medical terminology in answers"""
        answer = answer.strip()
        answer_lower = answer.lower()
        
        for old_term, new_term in self.term_mapping.items():
            if old_term in answer_lower:
                # Case-preserving replacement
                answer = answer.replace(old_term, new_term)
                answer = answer.replace(old_term.title(), new_term)
                answer = answer.replace(old_term.upper(), new_term.upper())
        
        return answer
    
    def detect_question_type(self, question: str) -> str:
        """Detect question type for appropriate thinking template"""
        q_lower = question.lower()
        
        if any(word in q_lower for word in ["what is", "what are", "identify", "name the"]):
            return "identification"
        elif any(word in q_lower for word in ["where", "location", "located", "position"]):
            return "location"  
        elif any(word in q_lower for word in ["how many", "count", "number of"]):
            return "count"
        elif any(word in q_lower for word in ["is there", "are there", "present", "evidence", "sign of"]):
            return "presence"
        elif any(word in q_lower for word in ["compare", "difference", "between"]):
            return "comparison"
        elif any(word in q_lower for word in ["diagnosis", "likely", "consistent with", "suggest"]):
            return "diagnosis"
        elif any(word in q_lower for word in ["modality", "imaging", "scan", "taken"]):
            return "modality"
        elif any(word in q_lower for word in ["organ", "structure", "anatomy"]):
            return "organ"
        elif any(word in q_lower for word in ["pathology", "tissue", "cell", "histology"]):
            return "pathology"
        elif any(word in q_lower for word in ["a.", "b.", "c.", "d.", "choice", "option"]):
            return "choice"
        else:
            return "identification"
    
    def generate_thinking(self, question: str, answer: str, modality: str = None) -> str:
        """Generate contextual thinking process"""
        question_type = self.detect_question_type(question)
        base_thinking = self.thinking_templates[question_type]
        
        # Add modality-specific context
        modality_context = ""
        if modality:
            mod_lower = modality.lower()
            if "ct" in mod_lower:
                modality_context = " This CT scan shows cross-sectional anatomy."
            elif "mri" in mod_lower:
                modality_context = " This MRI provides detailed soft tissue contrast."
            elif "x-ray" in mod_lower or "xray" in mod_lower:
                modality_context = " This X-ray shows the skeletal and soft tissue structures."
            elif "pathology" in mod_lower or "histology" in mod_lower:
                modality_context = " This histological image shows cellular details."
            elif "ultrasound" in mod_lower:
                modality_context = " This ultrasound shows real-time tissue imaging."
        
        # Add answer-specific reasoning
        answer_context = ""
        if question_type == "presence":
            if "yes" in answer.lower():
                answer_context = " I can identify positive findings."
            elif "no" in answer.lower():
                answer_context = " I don't observe the specified findings."
            else:
                answer_context = " I need to carefully evaluate the findings."
        elif question_type == "choice":
            answer_context = f" Based on the image analysis, option {answer} is most appropriate."
        
        return base_thinking + modality_context + answer_context
    
    def create_message_item(self, images: List[str], question: str, answer: str, 
                          modality: str = None, multi_turn: List = None) -> Dict[str, Any]:
        """Create a single training item in LLaMA-Factory format"""
        # Clean and standardize
        question = question.replace("<image>", "").strip()
        answer = self.standardize_answer(answer)
        
        if multi_turn:
            # Multi-turn conversation
            messages = []
            for i, turn in enumerate(multi_turn):
                messages.append({
                    "role": "user",
                    "content": f"<image>{turn['question']}" if i == 0 else turn['question']
                })
                
                if self.use_thinking and self.model_version == "4.1v":
                    thinking = self.generate_thinking(turn['question'], turn['answer'], modality)
                    content = f"<think>\n{thinking}\n</think>\n<answer>{turn['answer']}</answer>"
                else:
                    content = turn['answer']
                
                messages.append({
                    "role": "assistant", 
                    "content": content
                })
        else:
            # Single turn
            messages = [
                {
                    "role": "user",
                    "content": f"<image>{question}"
                }
            ]
            
            if self.use_thinking and self.model_version == "4.1v":
                thinking = self.generate_thinking(question, answer, modality)
                assistant_content = f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>"
            elif self.model_version == "4.1v" and not self.use_thinking:
                assistant_content = f"<answer>{answer}</answer>"
            else:  # 4.5v
                assistant_content = answer
                
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        return {
            "messages": messages,
            "images": images
        }
    
    def convert_slake(self, input_file: str, image_base: str) -> List[Dict]:
        """Convert SLAKE dataset"""
        print(f"Converting SLAKE dataset from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_items = []
        for item in data:
            image_path = os.path.join(image_base, item['image'])
            question = item['conversations'][0]['value']
            answer = item['conversations'][1]['value']
            modality = item.get('modality', 'CT')
            
            converted_item = self.create_message_item(
                images=[image_path],
                question=question,
                answer=answer,
                modality=modality
            )
            converted_items.append(converted_item)
        
        print(f"Converted {len(converted_items)} SLAKE samples")
        return converted_items
    
    def convert_vqa_rad(self, input_file: str, image_base: str) -> List[Dict]:
        """Convert VQA-RAD dataset"""
        print(f"Converting VQA-RAD dataset from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_items = []
        for item in data:
            image_path = os.path.join(image_base, item['image'])
            question = item['conversations'][0]['value']
            answer = item['conversations'][1]['value']
            modality = item.get('modality', 'X-Ray')
            
            converted_item = self.create_message_item(
                images=[image_path],
                question=question,
                answer=answer,
                modality=modality
            )
            converted_items.append(converted_item)
        
        print(f"Converted {len(converted_items)} VQA-RAD samples")
        return converted_items
    
    def convert_pathvqa(self, input_file: str, image_base: str) -> List[Dict]:
        """Convert PathVQA dataset"""
        print(f"Converting PathVQA dataset from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_items = []
        for item in data:
            image_path = os.path.join(image_base, item['image'])
            question = item['conversations'][0]['value']
            answer = item['conversations'][1]['value']
            modality = item.get('modality', 'pathology')
            
            converted_item = self.create_message_item(
                images=[image_path],
                question=question,
                answer=answer,
                modality=modality
            )
            converted_items.append(converted_item)
        
        print(f"Converted {len(converted_items)} PathVQA samples")
        return converted_items


def main():
    parser = argparse.ArgumentParser(description='Comprehensive MedVQA to GLM-4.1V/4.5V converter')
    parser.add_argument('--mode', choices=['individual', 'combined'], required=True,
                        help='Convert individual datasets or combine all')
    parser.add_argument('--model_version', choices=['4.1v', '4.5v'], default='4.1v',
                        help='GLM model version (affects output format)')
    parser.add_argument('--use_thinking', action='store_true', default=True,
                        help='Include <think> tags for GLM-4.1V-Thinking')
    parser.add_argument('--no_thinking', action='store_true',
                        help='Disable <think> tags (for Base models)')
    parser.add_argument('--data_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa',
                        help='MedVQA data directory')
    parser.add_argument('--output_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/data/glm_format',
                        help='Output directory for converted data')
    parser.add_argument('--image_base', default='/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images',
                        help='Base path for images')
    
    args = parser.parse_args()
    
    # Handle thinking flag
    use_thinking = args.use_thinking and not args.no_thinking
    if args.model_version == '4.5v':
        use_thinking = False  # 4.5V doesn't use thinking tags
    
    converter = MedVQAConverter(use_thinking=use_thinking, model_version=args.model_version)
    
    # Dataset file mappings
    dataset_files = {
        'slake': 'test_slake.json',
        'rad': 'test_rad.json', 
        'pathvqa': 'train_all.json'  # This contains PathVQA data
    }
    
    if args.mode == 'individual':
        # Convert each dataset separately
        for dataset_name, filename in dataset_files.items():
            input_file = os.path.join(args.data_dir, filename)
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping {dataset_name}")
                continue
            
            # Convert based on dataset type
            if dataset_name == 'slake':
                converted_data = converter.convert_slake(input_file, args.image_base)
            elif dataset_name == 'rad':
                converted_data = converter.convert_vqa_rad(input_file, args.image_base)
            elif dataset_name == 'pathvqa':
                converted_data = converter.convert_pathvqa(input_file, args.image_base)
            
            # Save individual dataset
            thinking_suffix = "_thinking" if use_thinking else "_base"
            output_file = os.path.join(args.output_dir, f"finetune_{dataset_name}_{args.model_version}{thinking_suffix}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(converted_data)} samples to {output_file}")
    
    elif args.mode == 'combined':
        # Combine all datasets
        all_converted_data = []
        
        for dataset_name, filename in dataset_files.items():
            input_file = os.path.join(args.data_dir, filename)
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping {dataset_name}")
                continue
            
            # Convert based on dataset type
            if dataset_name == 'slake':
                converted_data = converter.convert_slake(input_file, args.image_base)
            elif dataset_name == 'rad':
                converted_data = converter.convert_vqa_rad(input_file, args.image_base)
            elif dataset_name == 'pathvqa':
                converted_data = converter.convert_pathvqa(input_file, args.image_base)
            
            all_converted_data.extend(converted_data)
        
        # Shuffle for better training
        random.shuffle(all_converted_data)
        
        # Save combined dataset
        thinking_suffix = "_thinking" if use_thinking else "_base"
        output_file = os.path.join(args.output_dir, f"finetune_combined_{args.model_version}{thinking_suffix}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(all_converted_data)} combined samples to {output_file}")
    
    print("Conversion completed!")


if __name__ == '__main__':
    main()
