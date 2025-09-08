import os
import json
from typing import Dict

def load_original_data(data_root: str) -> Dict[str, str]:
    """Load original data and create a mapping from image path to ground truth answer"""
    print("Loading original data...")
    
    # Load all three datasets
    datasets = ['pvqa', 'slake', 'rad']
    image_to_answer = {}
    
    for dataset in datasets:
        json_path = os.path.join(data_root, f"test_{dataset}.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping...")
            continue
            
        print(f"Processing {dataset}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                # Extract question and answer
                question = conversations[0].get("value", "")
                if question.startswith("<image>\n"):
                    question = question[len("<image>\n"):]
                
                answer = conversations[1].get("value", "")
                
                # Create key: image_path + question
                image_path = item.get("image", "")
                key = f"{image_path}||{question}"
                image_to_answer[key] = answer
    
    print(f"Loaded {len(image_to_answer)} question-answer pairs from original data")
    return image_to_answer

def fix_output_file(output_path: str, image_to_answer: Dict[str, str], model_name: str):
    """Fix ground truth answers in a single output file"""
    print(f"Fixing {output_path}...")
    
    if not os.path.exists(output_path):
        print(f"Warning: {output_path} not found, skipping...")
        return
    
    # Load output file
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_count = 0
    total_count = len(data)
    mismatch_examples = []
    
    for item in data:
        image_path = item.get("image", "")
        question = item.get("question", "")
        
        # Create the same key format as in original data
        key = f"{image_path}||{question}"
        
        if key in image_to_answer:
            original_answer = image_to_answer[key]
            current_answer = item.get("answer", "")
            
            # Always update the answer to the original ground truth
            if current_answer != original_answer:
                # Store example for debugging (especially useful for Thyme)
                if len(mismatch_examples) < 3:  # Keep first 3 examples
                    mismatch_examples.append({
                        'question': question[:50] + "..." if len(question) > 50 else question,
                        'current_answer': current_answer,
                        'ground_truth': original_answer
                    })
                
                item["answer"] = original_answer
                fixed_count += 1
                
                # For Thyme and GLM4V, preserve the model's extracted answer in a separate field
                if model_name in ['thyme', 'glm4v']:
                    item["model_answer"] = current_answer
        else:
            print(f"Warning: No ground truth found for {key}")
    
    # Save fixed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Fixed {fixed_count}/{total_count} answers in {output_path}")
    
    # Show examples of what was fixed (especially useful for Thyme and GLM4V)
    if mismatch_examples and model_name in ['thyme', 'glm4v']:
        print(f"Examples of fixes for {model_name}:")
        for i, example in enumerate(mismatch_examples, 1):
            print(f"  {i}. Q: {example['question']}")
            print(f"     Model extracted: '{example['current_answer']}'")
            print(f"     Ground truth: '{example['ground_truth']}'")
            print()

def main():
    # Configuration
    data_root = "/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa"
    output_root = "/home/xc1490/xc1490/projects/medvqa_2025/output"
    
    # Models to fix - including thyme and glm4v which have answer extraction issues
    models = ['internvl8b', 'internvl30b', 'internvl38b', 'qwen25vl7b', 'thyme', 'glm4v']
    datasets = ['pvqa', 'slake', 'rad']
    
    # Load original data
    image_to_answer = load_original_data(data_root)
    
    # Fix each model's output files
    for model in models:
        print(f"\n=== Fixing {model} ===")
        model_output_dir = os.path.join(output_root, model)
        
        if not os.path.exists(model_output_dir):
            print(f"Warning: {model_output_dir} not found, skipping...")
            continue
        
        for dataset in datasets:
            output_path = os.path.join(model_output_dir, f"test_{dataset}.json")
            fix_output_file(output_path, image_to_answer, model)
    
    print("\n=== Fix completed ===")

if __name__ == "__main__":
    main()

