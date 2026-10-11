"""Command-line entry point: python cli.py --help."""
import argparse
import json
import sys
from pathlib import Path

from predictor import MaskPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict one masked token in an English sentence.")
    parser.add_argument("text", help="A sentence containing one [MASK], enclosed in quotes")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU; 0 for the first CUDA GPU")
    parser.add_argument("--revision", help="Optional Hugging Face model commit SHA")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    try:
        result = MaskPredictor(device=args.device, revision=args.revision).predict(args.text, args.top_k)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
