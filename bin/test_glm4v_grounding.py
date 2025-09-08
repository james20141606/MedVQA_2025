import os
import json
import argparse
import time
from typing import Dict, Any, Iterator
from PIL import Image

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration


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


def load_model(model_path: str, device_map: str, offline: bool):
    print(f"[Info] Loading model: {model_path}", flush=True)
    if offline:
        try:
            # Try local files first
            model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            print(f"[Warning] Failed to load from local cache: {e}")
            print("[Info] Trying to load from network...")
            # Fallback to network if local fails
            model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=False,
            )
            processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=False,
            )
    else:
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=False,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=False,
        )
    print("[Info] Model loaded", flush=True)
    return model, processor


def build_prompt(organ_name: str, img_w: int, img_h: int) -> str:
    # New template: require [x1, y1, x2, y2] (top-left, bottom-right)
    return (
        f"The image dimensions are {img_w}x{img_h}. "
        "Your task is to locate the object mentioned in the question. "
        "Provide only the bounding box coordinates in the format `[x1, y1, x2, y2]`, "
        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
        "Do not add any other text or explanation.\n\n"
        f"Question: {organ_name}"
    )


def run_single(model, processor, image_path: str, organ_name: str, max_new_tokens: int = 128) -> str:
    if not image_path or not os.path.exists(image_path):
        return "[ERROR] image not found"

    # 获取图像尺寸作为提示的一部分
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 0, 0

    prompt = build_prompt(organ_name, img_w, img_h)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.1,
    )

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return output_text


def _parse_and_sanitize_bbox(text: str, img_w: int, img_h: int) -> list[int]:
    """Parse bbox from model text and return [x1,y1,x2,y2] ints within image bounds.
    Supports:
      - <|begin_of_box|> ... [x1,y1,x2,y2] ... <|end_of_box|> (0..1000 normalized)
      - First bracketed list [a,b,c,d] (interprets as [x1,y1,x2,y2] or [x,y,w,h])
    Fallback: centered 60% box.
    """
    import re, math

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # 1) Try GLM boxed normalized pattern
    m = re.search(r"<\|begin_of_box\|>.*?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?<\|end_of_box\|>", text, re.DOTALL)
    if m and img_w > 0 and img_h > 0:
        x1n, y1n, x2n, y2n = [int(g) for g in m.groups()]
        x1 = int(round((x1n / 1000.0) * img_w))
        y1 = int(round((y1n / 1000.0) * img_h))
        x2 = int(round((x2n / 1000.0) * img_w))
        y2 = int(round((y2n / 1000.0) * img_h))
    else:
        # 2) Extract first [...] list of 4 numbers
        m2 = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if not m2:
            # Fallback: centered 60% box
            cx, cy = img_w // 2, img_h // 2
            w, h = max(1, int(img_w * 0.6)), max(1, int(img_h * 0.6))
            x1 = clamp(cx - w // 2, 0, max(0, img_w - 1))
            y1 = clamp(cy - h // 2, 0, max(0, img_h - 1))
            x2 = clamp(x1 + w, 1, img_w)
            y2 = clamp(y1 + h, 1, img_h)
            return [x1, y1, x2, y2]
        content = m2.group(1)
        nums = re.findall(r"-?\d+\.?\d*", content)
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
        # Heuristic: if c>d of x1<x2 semantics satisfied, treat as [x1,y1,x2,y2]; else [x,y,w,h]
        if c > a and d > b:
            x1, y1, x2, y2 = int(round(a)), int(round(b)), int(round(c)), int(round(d))
        else:
            x1, y1 = int(round(a)), int(round(b))
            x2, y2 = x1 + int(round(c)), y1 + int(round(d))

    # Sanitize and clamp
    x1 = clamp(int(x1), 0, max(0, img_w - 1))
    y1 = clamp(int(y1), 0, max(0, img_h - 1))
    x2 = clamp(int(x2), x1 + 1, img_w)
    y2 = clamp(int(y2), y1 + 1, img_h)
    return [x1, y1, x2, y2]


def _gt_to_xyxy(gt: list, img_w: int, img_h: int) -> list[int]:
    """Convert GT list of 4 numbers into [x1,y1,x2,y2] within bounds.
    If appears to be [x,y,w,h], convert accordingly; otherwise assumes [x1,y1,x2,y2].
    """
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
    parser = argparse.ArgumentParser(description="Grounding evaluation with GLM-4.1V-9B-Thinking on SLAKE index")
    parser.add_argument(
        "--index",
        type=str,
        default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images/slake/grounding_index.json",
        help="Path to grounding_index.json (JSONL)"
    )
    parser.add_argument("--output", type=str, default="/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/glm4v_slake.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--model_path", type=str, default="zai-org/GLM-4.1V-9B-Thinking",
                        help="Model path (HF or local)")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda:0"],
                        help="Device placement; use 'cuda:0' to avoid CPU offloading")
    parser.add_argument("--skip_empty", action="store_true",
                        help="Skip cases with empty detections")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit_pairs", type=int, default=None,
                        help="Limit number of (case, organ) pairs for quick test")
    parser.add_argument("--per_case_progress", action="store_true",
                        help="Show inline progress for each case (organs done/total)")
    parser.add_argument("--offline", action="store_true", default=True,
                        help="Force local-only loading (no network), requires all files locally")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.output))
    model, processor = load_model(args.model_path, args.device_map, args.offline)

    # 断点续跑：已存在则不重复推理同一个 (id, organ)
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

    # 统计索引大小
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
            # Legacy format: one line per case with detections dict
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
                # New format: one line per organ with keys 'organ' and 'bbox'
                organ_single = item.get('organ')
                bbox_single = item.get('bbox')
                if organ_single is None or bbox_single is None:
                    print(f"[Case {case_id}] no detections", flush=True)
                    continue
                total_organs = 1
                organs_done = 0
                print(f"[Case {case_id}] organs to process: 1", flush=True) if args.per_case_progress else None
                items_iter = [(organ_single, bbox_single)]

            for organ_name, gt_bbox in items_iter:
                key = f"{case_id}||{organ_name}"
                if key in existing:
                    continue
                try:
                    pred_text = run_single(model, processor, image_path, organ_name, args.max_new_tokens)
                except Exception as e:
                    pred_text = f"[ERROR] {e}"

                # Parse and sanitize bbox from model output
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


