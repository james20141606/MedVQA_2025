#!/usr/bin/env python3
"""
Base converter class for GLM-4.1V data conversion
"""
import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class BaseConverter(ABC):
    """Base class for data format conversion"""
    
    def __init__(self, model_version: str = "4.1v", use_thinking: bool = True):
        self.model_version = model_version.lower()
        self.use_thinking = use_thinking
        
        # 4.5V doesn't use thinking tags
        if self.model_version == "4.5v":
            self.use_thinking = False
    
    @abstractmethod
    def convert_dataset(self, input_file: str, output_file: str, **kwargs) -> int:
        """Convert a dataset file to GLM format"""
        pass
    
    def create_message_item(self, images: List[str], question: str, answer: str, 
                          thinking: str = None, multi_turn: List = None) -> Dict[str, Any]:
        """Create a single training item in LLaMA-Factory format"""
        
        if multi_turn:
            # Multi-turn conversation
            messages = []
            for i, turn in enumerate(multi_turn):
                messages.append({
                    "role": "user",
                    "content": f"<image>{turn['question']}" if i == 0 else turn['question']
                })
                
                if self.use_thinking and thinking:
                    content = f"<think>\n{thinking}\n</think>\n<answer>{turn['answer']}</answer>"
                else:
                    content = self._format_answer(turn['answer'])
                
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
            
            if self.use_thinking and thinking:
                assistant_content = f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>"
            else:
                assistant_content = self._format_answer(answer)
                
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        return {
            "messages": messages,
            "images": images
        }
    
    def _format_answer(self, answer: str) -> str:
        """Format answer based on model version"""
        if self.model_version == "4.1v" and not self.use_thinking:
            return f"<answer>{answer}</answer>"
        else:  # 4.5v or thinking mode handled elsewhere
            return answer
    
    def save_converted_data(self, data: List[Dict], output_file: str, shuffle: bool = True) -> None:
        """Save converted data to JSON file"""
        if shuffle:
            random.shuffle(data)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} samples to {output_file}")
    
    def validate_images(self, image_paths: List[str], base_path: str = "") -> List[str]:
        """Validate and fix image paths"""
        valid_paths = []
        
        for img_path in image_paths:
            if base_path:
                full_path = os.path.join(base_path, img_path)
            else:
                full_path = img_path
            
            if os.path.exists(full_path):
                valid_paths.append(full_path)
            else:
                print(f"Warning: Image not found: {full_path}")
                # Keep the path anyway for now, might be available during training
                valid_paths.append(full_path)
        
        return valid_paths
