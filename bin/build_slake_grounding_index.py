#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Any


def read_detection_json(path: str) -> Dict[str, List[float]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # detection.json expected format:
    # [ {"Organ1": [x, y, w, h]}, {"Organ2": [x, y, w, h]}, ... ]
    # Normalize to a single dict: organ -> [x, y, w, h]
    detections: Dict[str, List[float]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for organ, bbox in item.items():
                    detections[str(organ)] = list(bbox)
    elif isinstance(data, dict):
        # Fallback: already a dict
        for organ, bbox in data.items():
            detections[str(organ)] = list(bbox)
    else:
        raise ValueError(f"Unexpected detection.json format at {path}")

    return detections


def build_index(root_dir: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    # Only iterate immediate subdirectories of root_dir
    for name in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, name)
        if not os.path.isdir(subdir):
            continue

        image_path = os.path.join(subdir, 'source.jpg')
        det_path = os.path.join(subdir, 'detection.json')

        if not os.path.isfile(image_path):
            continue
        if not os.path.isfile(det_path):
            continue

        detections = read_detection_json(det_path)
        # Skip folders with empty detections
        if not detections:
            continue

        # Emit one entry per organ
        abs_image = os.path.abspath(image_path)
        for organ_name, bbox in detections.items():
            entries.append({
                'id': name,
                'image_path': abs_image,
                'organ': organ_name,
                'bbox': list(bbox),
            })

    return entries


def main():
    parser = argparse.ArgumentParser(description='Build consolidated grounding index for SLAKE detections.')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing SLAKE subfolders (e.g., .../data/medvqa/3vqa/images/slake)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path. Defaults to <root>/grounding_index.json')
    parser.add_argument('--jsonl', action='store_true',
                        help='Write output as JSON Lines (one JSON object per line) instead of a JSON array')
    args = parser.parse_args()

    output_path = args.output or os.path.join(os.path.abspath(args.root), 'grounding_index.json')

    entries = build_index(args.root)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if args.jsonl:
            for item in entries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    mode = 'JSONL' if args.jsonl else 'JSON'
    print(f"Wrote {len(entries)} entries to {output_path} ({mode})")


if __name__ == '__main__':
    main()


