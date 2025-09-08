import os
import json
import argparse
from typing import List, Dict, Any
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_model(model_path: str):
    print("Loading Qwen2.5-VL-7B-Instruct model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model and processor loaded.")
    return model, processor


def read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_question(sample: Dict[str, Any]) -> str:
    conversations = sample.get("conversations") or []
    if not conversations:
        return ""
    q = conversations[0].get("value", "")
    if q.startswith("<image>\n"):
        q = q[len("<image>\n"):]
    return q


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_evaluation(question: str, image_path: str, model, processor):
    """
    Run evaluation using Qwen2.5-VL-7B-Instruct model.
    """
    if not image_path or not os.path.exists(image_path):
        return f"[ERROR] Image not found: {image_path}", "[ERROR]"
    
    try:
        # Prepare messages in Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text", 
                        "text": question
                    }
                ]
            }
        ]
        
        # Preparation for inference - following official example
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process images and text separately
        image_inputs = []
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "image":
                    # Load and process the image directly
                    image = Image.open(content["image"]).convert('RGB')
                    image_inputs.append(image)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate response
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=8192,
            do_sample=False,
            temperature=0.1
        )
        
        # Decode response - trim input tokens and decode only new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text, output_text
        
    except Exception as e:
        return f"[ERROR] {e}", "[ERROR]"


def run_dataset(
    name: str,
    json_path: str,
    images_root: str,
    model,
    processor,
    output_dir: str,
    limit: int | None = None,
):
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
        question = extract_question(sample)
        key = f"{sample.get('id')}||{question}"
        if key in processed_keys:
            continue
        def is_valid_qwen_response(response: str, answer: str) -> bool:
            """Check if Qwen response is valid"""
            if not response or not response.strip():
                return False
            
            response_clean = response.strip()
            answer_clean = answer.strip() if answer else ""
            
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
            
            # Check if extracted answer is empty
            if not answer_clean:
                return False
            
            # Check if response is too short (likely incomplete)
            if len(response_clean) < 5:
                return False
            
            return True
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                assistant_response, answer = run_evaluation(question, abs_image, model, processor)
                
                # Check if response is valid
                if is_valid_qwen_response(assistant_response, answer):
                    # Print full response for debugging
                    print(f"\n=== Full Model Response ===")
                    print(assistant_response)
                    print(f"=== Extracted Answer ===")
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
                        print(f"Answer: {answer}")
                        print(f"========================\n")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error processing sample: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    assistant_response, answer = f"[ERROR] {e}", "[ERROR]"
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
    parser = argparse.ArgumentParser(description="Batch inference for Qwen2.5-VL-7B-Instruct on MEDVQA datasets")
    parser.add_argument("--datasets", nargs="*", default=["pvqa", "slake", "rad"], choices=["pvqa", "slake", "rad"], help="Datasets to run")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen2.5-VL-7B-Instruct model path (HF or local)")
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
        default="/home/xc1490/xc1490/projects/medvqa_2025/output/qwen25vl7b",
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

    for name in args.datasets:
        run_dataset(
            name=name,
            json_path=name_to_path[name],
            images_root=dataset_to_root[name],
            model=model,
            processor=processor,
            output_dir=args.output_dir,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
