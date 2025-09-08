import os
import json
import argparse
from typing import List, Dict, Any
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Build image transformation pipeline"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for image tiling"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image for InternVL"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """Load and preprocess image for InternVL"""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model(model_path: str):
    """Load InternVL3.5-38B model and tokenizer"""
    print("Loading InternVL3.5-38B model and tokenizer...")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        
        print("Model and tokenizer loaded successfully.")
        
        # Print GPU distribution info
        print(f"Model device map: {model.hf_device_map}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def read_json_array(path: str) -> List[Dict[str, Any]]:
    """Read JSON array from file"""
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
    """Ensure directory exists"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_evaluation(question: str, image_path: str, model, tokenizer):
    """
    Run evaluation using InternVL3.5-38B model.
    """
    if not image_path or not os.path.exists(image_path):
        return f"[ERROR] Image not found: {image_path}", "[ERROR]"
    
    try:
        # Load and preprocess image
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        
        # Prepare question with image token
        question_with_image = f"<image>\n{question}"
        
        # Set generation config
        generation_config = dict(
            max_new_tokens=1024, 
            do_sample=True,
            temperature=0.6
        )
        
        # Generate response using model.chat method
        response = model.chat(tokenizer, pixel_values, question_with_image, generation_config)
        
        return response
        
    except Exception as e:
        return f"[ERROR] {e}"

def run_dataset(
    name: str,
    json_path: str,
    images_root: str,
    model,
    tokenizer,
    output_dir: str,
    limit: int | None = None,
):
    """Run evaluation on a dataset"""
    print(f"\n=== Running dataset: {name} ===")
    samples = read_json_array(json_path)
    
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
    for idx, sample in enumerate(samples):
        rel_image = sample.get("image")
        abs_image = os.path.join(images_root, rel_image) if rel_image else None
        question, answer = extract_question_and_answer(sample)
        key = f"{sample.get('id')}||{question}"
        
        if key in processed_keys:
            continue
            
        def is_valid_internvl_response(response: str) -> bool:
            """Check if InternVL response is valid"""
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
                "i'm unable"
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
                assistant_response, _ = run_evaluation(question, abs_image, model, tokenizer)
                
                # Check if response is valid
                if is_valid_internvl_response(assistant_response):
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
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} samples...")
            
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch inference for InternVL3.5-38B on MEDVQA datasets")
    parser.add_argument("--datasets", nargs="*", default=["pvqa", "slake", "rad"], 
                       choices=["pvqa", "slake", "rad"], help="Datasets to run")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-38B", 
                       help="InternVL3.5-38B model path (HF or local)")
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
        default="/home/xc1490/xc1490/projects/medvqa_2025/output/internvl38b",
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
    """Main function"""
    args = parse_args()
    model, tokenizer = load_model(args.model_path)

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

    for name in args.datasets:
        run_dataset(
            name=name,
            json_path=name_to_path[name],
            images_root=dataset_to_root[name],
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            limit=args.limit,
        )

if __name__ == "__main__":
    main()
