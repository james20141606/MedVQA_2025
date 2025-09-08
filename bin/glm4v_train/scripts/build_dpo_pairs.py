#!/usr/bin/env python3
"""
Build simple DPO preference pairs from GLM-4.xV finetune.json.

Input: finetune.json with items {messages, images}
Output: preference JSONL or JSON with items {messages (user-only), images, chosen, rejected}

Strategy:
- chosen: assistant content from the item (strip <think>...</think> and <answer> wrappers)
- rejected: sample another assistant answer from a different item as a negative

Note: This is a minimal baseline; for higher-quality pairs, consider hard negatives
or rule-based perturbations (flip yes/no, distractors from same MCQ, etc.).
"""
import argparse
import json
import os
import random
import re
from typing import List, Dict


def clean_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # remove <think>...</think>
    text = re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL)
    # unwrap <answer>...</answer>
    text = re.sub(r"</?answer>", "", text)
    return text.strip()


def to_preference_item(src: Dict, neg_answer: str) -> Dict:
    # keep only the first user message; ensure it contains <image> prefix
    user_msg = None
    for m in src.get("messages", []):
        if m.get("role") == "user":
            user_msg = {"role": "user", "content": m.get("content", "")}
            break
    if not user_msg:
        return None
    # extract assistant answer
    ans = None
    for m in src.get("messages", []):
        if m.get("role") == "assistant":
            ans = clean_answer(m.get("content", ""))
            if ans:
                break
    if not ans:
        return None
    return {
        "messages": [user_msg],
        "images": src.get("images", []),
        "chosen": ans,
        "rejected": neg_answer,
    }


def main():
    p = argparse.ArgumentParser(description="Build DPO preference pairs from finetune.json")
    p.add_argument("--input", required=True, help="finetune.json path")
    p.add_argument("--output", required=True, help="Output JSON (pairs) or .jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    # collect candidate negative answers from all items
    negatives: List[str] = []
    for it in data:
        for m in it.get("messages", []):
            if m.get("role") == "assistant":
                s = clean_answer(m.get("content", ""))
                if s:
                    negatives.append(s)
    if not negatives:
        raise RuntimeError("No assistant answers found in input for negatives.")

    pairs: List[Dict] = []
    for it in data:
        # sample a negative different from the chosen when possible
        neg = None
        for _ in range(3):
            cand = random.choice(negatives)
            # get the chosen text
            chosen = None
            for m in it.get("messages", []):
                if m.get("role") == "assistant":
                    chosen = clean_answer(m.get("content", ""))
                    break
            if not chosen:
                break
            if cand.strip() != chosen.strip():
                neg = cand
                break
        if not neg:
            neg = random.choice(negatives)
        pref = to_preference_item(it, neg)
        if pref:
            pairs.append(pref)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.endswith(".jsonl"):
        with open(args.output, "w", encoding="utf-8") as f:
            for obj in pairs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pairs)} DPO pairs -> {args.output}")


if __name__ == "__main__":
    main()

