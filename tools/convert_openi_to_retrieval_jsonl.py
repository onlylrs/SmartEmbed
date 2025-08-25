#!/usr/bin/env python3
"""
Convert a JSONL file to retrieval format.

Input:  /project/medimgfmod/Generalist/shebd/openi_data_generation_parsed_copy.jsonl (or --input)
Output: /home/shebd/4_Collaboration/FYP2526/data/openi_retrieval.jsonl (or --output)

Each output line:
  {"text": "...", "image": "path", "task": "retrieval"}

Notes:
- Records with missing/empty text are skipped.
- At the end, prints how many were skipped due to empty text.
"""

import argparse
import json
import os
from typing import Tuple


DEFAULT_INPUT = \
    "/project/medimgfmod/Generalist/shebd/openi_data_generation_parsed_copy.jsonl"
DEFAULT_OUTPUT = \
    "/home/shebd/4_Collaboration/FYP2526/data/openi_retrieval.jsonl"


def is_empty_text(value) -> bool:
    """Return True if value is considered empty text."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return len(value.strip()) == 0


def convert_line(record: dict) -> Tuple[bool, dict]:
    """
    Convert a single input record to the desired retrieval format.

    Returns:
        (include, new_record)
        include = False if the record should be skipped (e.g., empty text)
    """
    # Prefer 'text' if present; otherwise allow 'caption' or 'title' as fallback
    text_value = record.get("text")
    if is_empty_text(text_value):
        # Try common alternatives if primary text is empty/missing
        for alt_key in ("caption", "title"):
            alt_val = record.get(alt_key)
            if not is_empty_text(alt_val):
                text_value = alt_val
                break

    if is_empty_text(text_value):
        return False, {}

    image_value = record.get("image")
    # Keep image path as-is; do not transform

    out = {
        "text": text_value,
        "image": image_value,
        "task": "retrieval",
    }
    return True, out


def convert_file(input_path: str, output_path: str) -> Tuple[int, int, int]:
    """
    Convert input JSONL to retrieval-format JSONL.

    Returns:
        (total_read, total_written, skipped_empty_text)
    """
    total_read = 0
    total_written = 0
    skipped_empty_text = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_read += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines silently
                continue

            include, new_record = convert_line(record)
            if not include:
                skipped_empty_text += 1
                continue

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            total_written += 1

    return total_read, total_written, skipped_empty_text


def main():
    parser = argparse.ArgumentParser(description="Convert OpenI JSONL to retrieval format")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Path to output JSONL file")
    args = parser.parse_args()

    total_read, total_written, skipped_empty = convert_file(args.input, args.output)

    print(f"Read: {total_read}")
    print(f"Written: {total_written}")
    print(f"Skipped due to empty text: {skipped_empty}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()


