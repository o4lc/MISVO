"""One-time script to download MBPP+ dataset from HuggingFace and save as JSONL."""

import json
import os

from datasets import load_dataset


def main():
    out_path = os.path.join(os.path.dirname(__file__), "mbppplus.jsonl")
    print(f"Downloading evalplus/mbppplus from HuggingFace...")
    ds = load_dataset("evalplus/mbppplus", split="test")
    print(f"Got {len(ds)} problems")

    with open(out_path, "w") as f:
        for entry in ds:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
