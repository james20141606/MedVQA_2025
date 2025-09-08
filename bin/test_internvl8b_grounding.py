import os
import json
import argparse
import time
from typing import Dict, Any, Iterator
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model(model_path: str, device_map: str, offline: bool):
    print(f"[Info] Loading model: {model_path}", flush=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=offline,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=offline,
    )
    print("[Info] Model and tokenizer loaded", flush=True)
    return model, tokenizer


def build_prompt(organ_name: str, img_w: int, img_h: int) -> str:
    return (
        f"The image dimensions are {img_w}x{img_h}. "
        "Your task is to locate the object mentioned in the question. "
        "Provide only the bounding box coordinates in the format `[x1, y1, x2, y2]`, "
        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
        "Do not add any other text or explanation.\n\n"
        f"Question: {organ_name}"
    )


def run_single(model, tokenizer, image_path: str, organ_name: str, max_new_tokens: int = 1024) -> str:
    if not image_path or not os.path.exists(image_path):
        return "[ERROR] image not found"
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 0, 0

    prompt = build_prompt(organ_name, img_w, img_h)
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    question_with_image = f"<image>\n{prompt}"
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.1)
    response = model.chat(tokenizer, pixel_values, question_with_image, generation_config)
    return response.strip() if isinstance(response, str) else str(response)


def _parse_and_sanitize_bbox(text: str, img_w: int, img_h: int) -> list[int]:
    import re
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if not m:
        cx, cy = img_w // 2, img_h // 2
        w, h = max(1, int(img_w * 0.6)), max(1, int(img_h * 0.6))
        x1 = clamp(cx - w // 2, 0, max(0, img_w - 1))
        y1 = clamp(cy - h // 2, 0, max(0, img_h - 1))
        x2 = clamp(x1 + w, 1, img_w)
        y2 = clamp(y1 + h, 1, img_h)
        return [x1, y1, x2, y2]
    nums = re.findall(r"-?\d+\.?\d*", m.group(1))
    vals = [float(n) for n in nums[:4]] if len(nums) >= 4 else []
    if len(vals) < 4:
        cx, cy = img_w // 2, img_h // 2
        w, h = max(1, int(img_w * 0.6)), max(1, int(img_h * 0.6))
        x1 = clamp(cx - w // 2, 0, max(0, img_w - 1))
        y1 = clamp(cy - h // 2, 0, max(0, img_h - 1))
        x2 = clamp(x1 + w, 1, img_w)
        y2 = clamp(y1 + h, 1, img_h)
        return [x1, y1, x2, y2]
    a, b, c, d = vals
    if c > a and d > b:
        x1, y1, x2, y2 = int(round(a)), int(round(b)), int(round(c)), int(round(d))
    else:
        x1, y1 = int(round(a)), int(round(b))
        x2, y2 = x1 + int(round(c)), y1 + int(round(d))
    x1 = clamp(int(x1), 0, max(0, img_w - 1))
    y1 = clamp(int(y1), 0, max(0, img_h - 1))
    x2 = clamp(int(x2), x1 + 1, img_w)
    y2 = clamp(int(y2), y1 + 1, img_h)
    return [x1, y1, x2, y2]


def _gt_to_xyxy(gt: list, img_w: int, img_h: int) -> list[int]:
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    if not isinstance(gt, (list, tuple)) or len(gt) < 4:
        return [0, 0, max(1, img_w//2), max(1, int(img_h//2))]
    a, b, c, d = [float(x) for x in gt[:4]]
    if c > a and d > b:
        x1, y1, x2, y2 = int(round(a)), int(round(b)), int(round(c)), int(round(d))
    else:
        x1, y1 = int(round(a)), int(round(b))
        x2, y2 = x1 + int(round(c)), y1 + int(round(d))
    x1 = clamp(x1, 0, max(0, img_w - 1))
    y1 = clamp(y1, 0, max(0, img_h - 1))
    x2 = clamp(x2, x1 + 1, img_w)
    y2 = clamp(y2, y1 + 1, img_h)
    return [x1, y1, x2, y2]


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding with InternVL3.5-8B on SLAKE index")
    parser.add_argument("--index", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/slake/grounding_index.json")
    parser.add_argument("--output", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/internvl8b_slake.jsonl")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda:0"])
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit_pairs", type=int, default=None)
    parser.add_argument("--per_case_progress", action="store_true")
    parser.add_argument("--offline", action="store_true", default=True,
                       help="Load model from local cache only (default: True)")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.output))
    model, tokenizer = load_model(args.model_path, args.device_map, args.offline)

    existing = set()
    if os.path.exists(args.output):
        try:
            t_resume_start = time.time()
            for line in iter_jsonl(args.output):
                key = f"{line.get('id')}||{line.get('organ')}"
                existing.add(key)
            if existing:
                dt = time.time() - t_resume_start
                print(f"[Info] Resuming. Found {len(existing)} existing (id,organ) pairs in output. Scan took {dt:.2f}s", flush=True)
        except Exception:
            pass

    t_index_start = time.time()
    total_cases = 0
    for _ in iter_jsonl(args.index):
        total_cases += 1
    dt_index = time.time() - t_index_start
    print(f"[Info] Index: {args.index}", flush=True)
    print(f"[Info] Total lines in index: {total_cases} (scan {dt_index:.2f}s)", flush=True)

    processed = 0
    written = 0
    with open(args.output, 'a', encoding='utf-8') as fout:
        for item in iter_jsonl(args.index):
            case_id = item.get('id')
            image_path = item.get('image_path')

            detections = item.get('detections')
            if isinstance(detections, dict):
                if args.skip_empty and not detections:
                    continue
                total_organs = len(detections)
                organs_done = 0
                if total_organs == 0:
                    print(f"[Case {case_id}] no detections", flush=True)
                    continue
                else:
                    print(f"[Case {case_id}] organs to process: {total_organs}", flush=True)
                items_iter = detections.items()
            else:
                organ_single = item.get('organ')
                bbox_single = item.get('bbox')
                if organ_single is None or bbox_single is None:
                    print(f"[Case {case_id}] no detections", flush=True)
                    continue
                total_organs = 1
                organs_done = 0
                if args.per_case_progress:
                    print(f"[Case {case_id}] organs to process: 1", flush=True)
                items_iter = [(organ_single, bbox_single)]

            for organ_name, gt_bbox in items_iter:
                key = f"{case_id}||{organ_name}"
                if key in existing:
                    continue
                try:
                    pred_text = run_single(model, tokenizer, image_path, organ_name, args.max_new_tokens)
                except Exception as e:
                    pred_text = f"[ERROR] {e}"

                _w, _h = Image.open(image_path).size
                pred_bbox = _parse_and_sanitize_bbox(pred_text, _w, _h)
                gt_xyxy = _gt_to_xyxy(gt_bbox, _w, _h)

                record = {
                    "id": case_id,
                    "image_path": image_path,
                    "organ": organ_name,
                    "gt_bbox": gt_bbox,
                    "gt_bbox_xyxy": gt_xyxy,
                    "pred_raw": pred_text,
                    "pred_bbox": pred_bbox,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                processed += 1
                organs_done += 1
                if args.per_case_progress:
                    print(f"\r[Case {case_id}] organs {organs_done}/{total_organs} | total written {written}", end='', flush=True)
                if processed % 50 == 0:
                    print(f"[Progress] processed={processed}, newly_written={written}", flush=True)
                if args.limit_pairs is not None and written >= args.limit_pairs:
                    print(f"[Info] Reached limit_pairs={args.limit_pairs}", flush=True)
                    print(f"[Done] Newly written: {written}. Output: {args.output}", flush=True)
                    return
            if args.per_case_progress and total_organs > 0:
                print()

    print(f"[Done] Newly written: {written}. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()

import os
import json
import argparse
import time
from typing import Dict, Any, Iterator
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_pixel_values(image_file: str, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model(model_path: str, device_map: str, offline: bool):
    print(f"[Info] Loading model: {model_path}", flush=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=offline,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=offline,
    )
    print("[Info] Model loaded", flush=True)
    return model, tokenizer


def build_prompt(organ_name: str, img_w: int, img_h: int) -> str:
    return (
        f"The image dimensions are {img_w}x{img_h}. "
        "Your task is to locate the object mentioned in the question. "
        "Provide only the bounding box coordinates in the format `[x1, y1, x2, y2]`, "
        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
        "Do not add any other text or explanation.\n\n"
        f"Question: {organ_name}"
    )


def run_single(model, tokenizer, image_path: str, organ_name: str, max_new_tokens: int = 1024) -> str:
    if not image_path or not os.path.exists(image_path):
        return "[ERROR] image not found"
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 0, 0

    prompt = build_prompt(organ_name, img_w, img_h)
    try:
        pixel_values = load_image_pixel_values(image_path, max_num=12).to(torch.bfloat16).cuda()
    except Exception:
        pixel_values = load_image_pixel_values(image_path, max_num=12).to(torch.bfloat16)

    question_with_image = f"<image>\n{prompt}"
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.1)
    response = model.chat(tokenizer, pixel_values, question_with_image, generation_config)
    return response.strip() if isinstance(response, str) else str(response)


def _parse_and_sanitize_bbox(text: str, img_w: int, img_h: int) -> list[int]:
    import re
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if not m:
        cx, cy = img_w // 2, img_h // 2
        w, h = max(1, int(img_w * 0.6)), max(1, int(img_h * 0.6))
        x1 = clamp(cx - w // 2, 0, max(0, img_w - 1))
        y1 = clamp(cy - h // 2, 0, max(0, img_h - 1))
        x2 = clamp(x1 + w, 1, img_w)
        y2 = clamp(y1 + h, 1, img_h)
        return [x1, y1, x2, y2]
    nums = re.findall(r"-?\d+\.?\d*", m.group(1))
    vals = [float(n) for n in nums[:4]] if len(nums) >= 4 else []
    if len(vals) < 4:
        cx, cy = img_w // 2, img_h // 2
        w, h = max(1, int(img_w * 0.6)), max(1, int(img_h * 0.6))
        x1 = clamp(cx - w // 2, 0, max(0, img_w - 1))
        y1 = clamp(cy - h // 2, 0, max(0, img_h - 1))
        x2 = clamp(x1 + w, 1, img_w)
        y2 = clamp(y1 + h, 1, img_h)
        return [x1, y1, x2, y2]
    a, b, c, d = vals
    if c > a and d > b:
        x1, y1, x2, y2 = int(round(a)), int(round(b)), int(round(c)), int(round(d))
    else:
        x1, y1 = int(round(a)), int(round(b))
        x2, y2 = x1 + int(round(c)), y1 + int(round(d))
    x1 = clamp(int(x1), 0, max(0, img_w - 1))
    y1 = clamp(int(y1), 0, max(0, img_h - 1))
    x2 = clamp(int(x2), x1 + 1, img_w)
    y2 = clamp(int(y2), y1 + 1, img_h)
    return [x1, y1, x2, y2]


def _gt_to_xyxy(gt: list, img_w: int, img_h: int) -> list[int]:
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    if not isinstance(gt, (list, tuple)) or len(gt) < 4:
        return [0, 0, max(1, img_w//2), max(1, img_h//2)]
    a, b, c, d = [float(x) for x in gt[:4]]
    if c > a and d > b:
        x1, y1, x2, y2 = int(round(a)), int(round(b)), int(round(c)), int(round(d))
    else:
        x1, y1 = int(round(a)), int(round(b))
        x2, y2 = x1 + int(round(c)), y1 + int(round(d))
    x1 = clamp(x1, 0, max(0, img_w - 1))
    y1 = clamp(y1, 0, max(0, img_h - 1))
    x2 = clamp(x2, x1 + 1, img_w)
    y2 = clamp(y2, y1 + 1, img_h)
    return [x1, y1, x2, y2]


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding with InternVL3.5-8B on SLAKE index")
    parser.add_argument("--index", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/slake/grounding_index.json")
    parser.add_argument("--output", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/internvl8b_slake.jsonl")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda:0"])
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit_pairs", type=int, default=None)
    parser.add_argument("--per_case_progress", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.output))
    model, tokenizer = load_model(args.model_path, args.device_map, args.offline)

    existing = set()
    if os.path.exists(args.output):
        try:
            t_resume_start = time.time()
            for line in iter_jsonl(args.output):
                key = f"{line.get('id')}||{line.get('organ')}"
                existing.add(key)
            if existing:
                dt = time.time() - t_resume_start
                print(f"[Info] Resuming. Found {len(existing)} existing (id,organ) pairs in output. Scan took {dt:.2f}s", flush=True)
        except Exception:
            pass

    t_index_start = time.time()
    total_cases = 0
    for _ in iter_jsonl(args.index):
        total_cases += 1
    dt_index = time.time() - t_index_start
    print(f"[Info] Index: {args.index}", flush=True)
    print(f"[Info] Total lines in index: {total_cases} (scan {dt_index:.2f}s)", flush=True)

    processed = 0
    written = 0
    with open(args.output, 'a', encoding='utf-8') as fout:
        for item in iter_jsonl(args.index):
            case_id = item.get('id')
            image_path = item.get('image_path')

            detections = item.get('detections')
            if isinstance(detections, dict):
                if args.skip_empty and not detections:
                    continue
                total_organs = len(detections)
                organs_done = 0
                if total_organs == 0:
                    print(f"[Case {case_id}] no detections", flush=True)
                    continue
                else:
                    print(f"[Case {case_id}] organs to process: {total_organs}", flush=True)
                items_iter = detections.items()
            else:
                organ_single = item.get('organ')
                bbox_single = item.get('bbox')
                if organ_single is None or bbox_single is None:
                    print(f"[Case {case_id}] no detections", flush=True)
                    continue
                total_organs = 1
                organs_done = 0
                if args.per_case_progress:
                    print(f"[Case {case_id}] organs to process: 1", flush=True)
                items_iter = [(organ_single, bbox_single)]

            for organ_name, gt_bbox in items_iter:
                key = f"{case_id}||{organ_name}"
                if key in existing:
                    continue
                try:
                    pred_text = run_single(model, tokenizer, image_path, organ_name, args.max_new_tokens)
                except Exception as e:
                    pred_text = f"[ERROR] {e}"

                with Image.open(image_path) as im:
                    _w, _h = im.size
                pred_bbox = _parse_and_sanitize_bbox(pred_text, _w, _h)
                gt_xyxy = _gt_to_xyxy(gt_bbox, _w, _h)

                record = {
                    "id": case_id,
                    "image_path": image_path,
                    "organ": organ_name,
                    "gt_bbox": gt_bbox,
                    "gt_bbox_xyxy": gt_xyxy,
                    "pred_raw": pred_text,
                    "pred_bbox": pred_bbox,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                processed += 1
                organs_done += 1
                if args.per_case_progress:
                    print(f"\r[Case {case_id}] organs {organs_done}/{total_organs} | total written {written}", end='', flush=True)
                if processed % 50 == 0:
                    print(f"[Progress] processed={processed}, newly_written={written}", flush=True)
                if args.limit_pairs is not None and written >= args.limit_pairs:
                    print(f"[Info] Reached limit_pairs={args.limit_pairs}", flush=True)
                    print(f"[Done] Newly written: {written}. Output: {args.output}", flush=True)
                    return
            if args.per_case_progress and total_organs > 0:
                print()

    print(f"[Done] Newly written: {written}. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()


