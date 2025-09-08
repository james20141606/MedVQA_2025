#!/usr/bin/env python3
"""
Wrapper to convert MedVQA data into GLM-4.xV training format.

Delegates to convert_medvqa_to_glm4v(.py) or _with_think(.py) based on --mode.
"""
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Convert MedVQA to GLM-4.xV format")
    p.add_argument("--mode", choices=["thinking", "base"], default="base")
    p.add_argument("--scope", choices=["individual", "combined"], default="combined")
    p.add_argument("--data_dir", default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa")
    p.add_argument("--output_root", default="/home/xc1490/xc1490/projects/medvqa_2025/data")
    p.add_argument("--image_base", default="/home/xc1490/xc1490/projects/medvqa_2025/data/medvqa/3vqa/images")
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(__file__))
    if args.mode == "thinking":
        script = os.path.join(here, "convert_medvqa_to_glm4v_with_think.py")
        out_dir = os.path.join(args.output_root, "glm4v_format_with_think")
    else:
        script = os.path.join(here, "convert_medvqa_to_glm4v.py")
        out_dir = os.path.join(args.output_root, "glm4v_format")

    cmd = [
        "python",
        script,
        "--mode",
        args.scope,
        "--data_dir",
        args.data_dir,
        "--output_dir",
        out_dir,
        "--image_base",
        args.image_base,
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Conversion failed:", e, file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

