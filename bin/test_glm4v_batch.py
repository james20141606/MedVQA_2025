import os
import json
import argparse
import random
import time
from typing import List, Dict, Any
from PIL import Image

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration


def load_model(model_path: str):
    print("Loading GLM-4.1V-9B-Thinking model and processor...")
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        use_fast=True,
        trust_remote_code=True
    )
    print("Model and processor loaded.")
    return model, processor


def read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_question_and_answer(sample: Dict[str, Any]) -> tuple[str, str]:
    """Extract question and ground truth answer from sample"""
    conversations = sample.get("conversations") or []
    if len(conversations) < 2:
        return "", ""
    
    # First conversation contains the question
    question = conversations[0].get("value", "")
    if question.startswith("<image>\n"):
        question = question[len("<image>\n"):]
    
    # Second conversation contains the ground truth answer
    answer = conversations[1].get("value", "")
    
    return question, answer


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_evaluation(question: str, image_path: str, model, processor):
    """
    Run evaluation using GLM-4.1V-9B-Thinking model.
    """
    if not image_path or not os.path.exists(image_path):
        return f"[ERROR] Image not found: {image_path}"
    
    try:
        # Prepare messages in GLM-4.1V format following official documentation
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image_path  # Use file path as URL
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
        
        # Process inputs using apply_chat_template (official method)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=8192,
            do_sample=False,
            temperature=0.1
        )
        
        # Decode response (following official example)
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=False
        )
        
        # Clean up the response
        if output_text and isinstance(output_text, str):
            output_text = output_text.strip()
            # Remove common special tokens
            for token in ['<|endoftext|>', '<|im_end|>', '<|assistant|>', '<|user|>']:
                output_text = output_text.replace(token, '').strip()
            return output_text if output_text else "[ERROR] Empty response"
        else:
            return "[ERROR] Invalid response format"
        
    except Exception as e:
        return f"[ERROR] {e}"


def run_dataset(
    name: str,
    json_path: str,
    images_root: str,
    model,
    processor,
    output_dir: str,
    limit: int | None = None,
    random_seed: int = None,
):
    print(f"\n=== Running dataset: {name} ===")
    samples = read_json_array(json_path)
    
    # Set random seed for reproducible shuffling
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    
    # Shuffle samples for better distribution across parallel runs
    random.shuffle(samples)
    print(f"Shuffled {len(samples)} samples")
    
    # Prepare output path and load existing results if present (resume support)
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"test_{name}.json")
    results = []
    processed_keys = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                key = f"{r.get('id')}||{r.get('question')}"
                processed_keys.add(key)
            print(f"Loaded {len(results)} existing results from {out_path}; will skip duplicates.")
        except Exception as e:
            print(f"Warning: failed to load existing results ({e}); starting fresh.")

    newly_processed = 0
    total_samples = len(samples)
    start_time = time.time()
    
    for idx, sample in enumerate(samples):
        rel_image = sample.get("image")
        abs_image = os.path.join(images_root, rel_image) if rel_image else None
        question, answer = extract_question_and_answer(sample)
        key = f"{sample.get('id')}||{question}"
        if key in processed_keys:
            continue
        def is_valid_glm4v_response(response: str) -> bool:
            """Check if GLM4V response is valid"""
            if not response or not response.strip():
                return False
            
            response_clean = response.strip()
            
            # Check for common invalid responses
            invalid_patterns = [
                "[ERROR]",
                "error",
                "failed",
                "exception",
                "sorry",
                "i cannot",
                "i don't",
                "i'm unable",
                "i am unable"
            ]
            
            for pattern in invalid_patterns:
                if pattern.lower() in response_clean.lower():
                    return False
            
            # Check if response is too short (likely incomplete)
            if len(response_clean) < 5:
                return False
            
            return True
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                assistant_response = run_evaluation(question, abs_image, model, processor)
                
                # Check if response is valid
                if is_valid_glm4v_response(assistant_response):
                    # Print full response for debugging
                    print(f"\n=== Full Model Response ===")
                    print(assistant_response)
                    print(f"=== Ground Truth Answer ===")
                    print(answer)
                    print(f"========================\n")
                    break
                else:
                    print(f"Attempt {attempt + 1}: Invalid response '{assistant_response[:50]}...', retrying...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        print(f"Failed to get valid response after {max_retries} attempts")
                        # Print the invalid response for debugging
                        print(f"\n=== Invalid Response (Attempt {attempt + 1}) ===")
                        print(f"Assistant: {assistant_response}")
                        print(f"========================\n")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error processing sample: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    assistant_response = f"[ERROR] {e}"
                    break
        record = {
            "id": sample.get("id"),
            "image": rel_image,
            "abs_image": abs_image,
            "question": question,
            "assistant_response": assistant_response,
            "answer": answer,
        }
        results.append(record)
        processed_keys.add(key)
        newly_processed += 1
        
        # Calculate progress and remaining samples
        processed_count = len(results)
        remaining_count = total_samples - processed_count
        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / (idx + 1) if idx > 0 else 0
        estimated_remaining_time = remaining_count * avg_time_per_sample
        
        # Print progress every 10 samples with detailed information
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{total_samples} ({((idx + 1)/total_samples)*100:.1f}%)")
            print(f"  Newly processed: {newly_processed}")
            print(f"  Total processed: {processed_count}")
            print(f"  Remaining: {remaining_count}")
            print(f"  Avg time per sample: {avg_time_per_sample:.1f}s")
            print(f"  Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            print(f"  Elapsed time: {elapsed_time/60:.1f} minutes")
        # Incremental save to support interruption/resume
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to save intermediate results: {e}")
        if limit is not None and newly_processed >= limit:
            break

    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for GLM-4.1V-9B-Thinking on MEDVQA datasets")
    parser.add_argument("--datasets", nargs="*", default=["pvqa", "slake", "rad"], choices=["pvqa", "slake", "rad"], help="Datasets to run")
    parser.add_argument("--random_dataset", action="store_true", help="Randomly select one dataset from the specified datasets")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducible shuffling")
    parser.add_argument("--model_path", type=str, default="zai-org/GLM-4.1V-9B-Thinking", help="GLM-4.1V model path (HF or local)")
    parser.add_argument(
        "--images_root",
        type=str,
        default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images",
        help="Default root directory for images of all datasets",
    )
    parser.add_argument(
        "--pvqa_images_root",
        type=str,
        default=None,
        help="Optional override for PVQA images root",
    )
    parser.add_argument(
        "--slake_images_root",
        type=str,
        default=None,
        help="Optional override for SLAKE images root",
    )
    parser.add_argument(
        "--rad_images_root",
        type=str,
        default=None,
        help="Optional override for RAD images root",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa",
        help="Root directory containing test_*.json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/xc1490/xc1490/projects/medvqa_2025/output/glm4v",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples per dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, processor = load_model(args.model_path)

    name_to_path = {
        "pvqa": os.path.join(args.data_root, "test_pvqa.json"),
        "slake": os.path.join(args.data_root, "test_slake.json"),
        "rad": os.path.join(args.data_root, "test_rad.json"),
    }

    dataset_to_root = {
        "pvqa": args.pvqa_images_root or args.images_root,
        "slake": args.slake_images_root or args.images_root,
        "rad": args.rad_images_root or args.images_root,
    }

    # Determine which datasets to run
    datasets_to_run = args.datasets
    if args.random_dataset:
        # Randomly select one dataset
        selected_dataset = random.choice(args.datasets)
        datasets_to_run = [selected_dataset]
        print(f"Randomly selected dataset: {selected_dataset}")
    
    # Set random seed for dataset selection if specified
    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"Using random seed: {args.random_seed}")

    for name in datasets_to_run:
        run_dataset(
            name=name,
            json_path=name_to_path[name],
            images_root=dataset_to_root[name],
            model=model,
            processor=processor,
            output_dir=args.output_dir,
            limit=args.limit,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
