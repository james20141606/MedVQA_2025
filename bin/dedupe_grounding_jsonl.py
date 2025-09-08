import os
import argparse
import json
from typing import Tuple


def make_key(obj: dict) -> Tuple[str, str]:
    """Build a dedupe key using (image_path, organ).
    If fields are missing, use empty strings to avoid KeyError.
    """
    return (str(obj.get("image_path", "")), str(obj.get("organ", "")))


def dedupe_file(path: str, inplace: bool = False, out_dir: str | None = None) -> Tuple[int, int, str]:
    """Dedupe a JSONL file by (image_path, organ).

    Returns: (total_lines, kept_lines, output_path)
    """
    total = 0
    kept = 0
    seen = set()
    if inplace:
        out_path = path
    else:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(path))
        else:
            out_path = f"{os.path.splitext(path)[0]}.dedup.jsonl"

    with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            total += 1
            try:
                obj = json.loads(line_stripped)
            except Exception:
                # Keep malformed lines as-is to avoid data loss
                fout.write(line)
                kept += 1
                continue
            key = make_key(obj)
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return total, kept, out_path


def main():
    parser = argparse.ArgumentParser(description="Dedupe grounding JSONL by (image_path, organ)")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single JSONL file to dedupe",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/xc1490/xc1490/projects/medvqa_2025/output/grounding",
        help="Directory containing JSONL files (processed if --file not provided)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='/home/xc1490/xc1490/projects/medvqa_2025/output/grounding/dedup/',
        help="Optional output directory for deduped files (ignored if --inplace)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file(s) instead of writing *.dedup.jsonl",
    )
    args = parser.parse_args()

    if args.file:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(args.file)
        total, kept, out_path = dedupe_file(args.file, inplace=args.inplace, out_dir=args.out_dir)
        print(f"[Done] {os.path.basename(args.file)}: total={total}, kept={kept}, out={out_path}")
        return

    # Process all *.jsonl files in directory
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(args.dir)
    entries = [
        os.path.join(args.dir, f)
        for f in sorted(os.listdir(args.dir))
        if f.endswith(".jsonl")
    ]
    if not entries:
        print("[Info] No .jsonl files found.")
        return

    for p in entries:
        total, kept, out_path = dedupe_file(p, inplace=args.inplace, out_dir=args.out_dir)
        print(f"[Done] {os.path.basename(p)}: total={total}, kept={kept}, out={out_path}")


if __name__ == "__main__":
    main()


