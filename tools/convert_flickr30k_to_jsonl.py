import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_captions_file(captions_path: str) -> List[Tuple[str, str]]:
    """Parse captions file into (image_name, caption) pairs.

    Supported formats (auto-detected per line):
      1) Comma-separated: "image.jpg,Caption text possibly with commas, too"
         - We split on the FIRST comma only.
      2) Flickr30K original: "image.jpg#0\tA caption..." (tab) or whitespace.
    """
    pairs: List[Tuple[str, str]] = []
    with open(captions_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            img_name: str = ""
            caption: str = ""

            # Prefer comma-separated parsing if a comma exists and appears after a plausible filename
            if "," in line:
                left, right = line.split(",", 1)
                left_s = left.strip()
                right_s = right.strip()
                # Heuristic: left side should look like a filename
                if left_s.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_name = left_s
                    caption = right_s
                else:
                    # Fall back to tab/space split
                    parts = line.split("\t", 1)
                    if len(parts) == 1:
                        parts = line.split(None, 1)
                    if len(parts) == 2:
                        img_name, caption = parts[0].strip(), parts[1].strip()
            else:
                # Tab/space separated
                parts = line.split("\t", 1)
                if len(parts) == 1:
                    parts = line.split(None, 1)
                if len(parts) == 2:
                    img_name, caption = parts[0].strip(), parts[1].strip()

            if not img_name or not caption:
                continue

            # Strip optional trailing index (e.g., "image.jpg#3")
            if "#" in img_name:
                img_name = img_name.split("#", 1)[0]

            caption = caption.strip().strip('"')
            if not caption:
                continue

            pairs.append((img_name, caption))
    return pairs


def group_by_image(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Group captions by image name."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for img_name, caption in pairs:
        grouped[img_name].append(caption)
    return grouped


def split_images(image_names: List[str], eval_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Deterministically split image names into train/eval sets (no leakage)."""
    rng = random.Random(seed)
    shuffled = image_names[:]
    rng.shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_ratio)
    eval_imgs = set(shuffled[:n_eval])
    train_imgs = [n for n in shuffled if n not in eval_imgs]
    eval_imgs_list = [n for n in shuffled if n in eval_imgs]
    return train_imgs, eval_imgs_list


def write_jsonl(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_records_for_split(
    grouped: Dict[str, List[str]],
    images_dir: str,
    image_names: List[str],
    task_label: str = "retrieval",
) -> List[Dict]:
    """Build JSONL records for the given subset of images.

    Each caption of an image becomes one line:
      {"query_image": "/abs/path/.../image.jpg", "positive": "...", "task": "retrieval"}
    Missing image files are skipped.
    """
    records: List[Dict] = []
    for img_name in image_names:
        abs_img_path = os.path.join(images_dir, img_name)
        if not os.path.isabs(abs_img_path):
            abs_img_path = os.path.abspath(abs_img_path)
        if not os.path.exists(abs_img_path):
            # Skip if the image file does not exist
            continue
        for caption in grouped.get(img_name, []):
            records.append({
                "query_image": abs_img_path,
                "positive": caption.strip(),
                "task": task_label,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert Flickr30K (captions.txt + images dir) to JSONL for retrieval training.")
    parser.add_argument("--captions", type=str, required=True, help="Path to Flickr30K captions.txt")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--train-output", type=str, required=True, help="Output path for train JSONL")
    parser.add_argument("--eval-output", type=str, required=True, help="Output path for eval JSONL")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval split ratio (by images), default 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    if not os.path.isfile(args.captions):
        raise RuntimeError(f"Captions file not found: {args.captions}")
    if not os.path.isdir(args.images_dir):
        raise RuntimeError(f"Images directory not found: {args.images_dir}")

    pairs = parse_captions_file(args.captions)
    if not pairs:
        raise RuntimeError(f"No (image, caption) pairs parsed from {args.captions}")

    grouped = group_by_image(pairs)
    image_names = sorted(grouped.keys())

    # Quick sanity: how many of these image files exist in images_dir?
    exist_count = 0
    missing = []
    for name in image_names:
        p = os.path.join(args.images_dir, name)
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if os.path.exists(p):
            exist_count += 1
        else:
            if len(missing) < 5:
                missing.append(name)
    total_imgs = len(image_names)
    print(f"Found {total_imgs} unique images in captions; {exist_count} exist under images-dir, {total_imgs - exist_count} missing.")
    if exist_count == 0:
        sample_hint = ("\n".join(missing)) if missing else "(no sample)"
        raise RuntimeError(
            "No captioned images were found under the provided images directory.\n"
            f"images-dir: {args.images_dir}\n"
            "Please verify the directory and file naming (case-sensitive) matches captions.txt.\n"
            "For example, check whether the directory is 'Images' vs 'images', and that files use '.jpg'.\n"
            f"First few missing names from captions:\n{sample_hint}"
        )
    train_imgs, eval_imgs = split_images(image_names, args.eval_ratio, args.seed)

    train_records = build_records_for_split(grouped, args.images_dir, train_imgs)
    eval_records = build_records_for_split(grouped, args.images_dir, eval_imgs)

    if not train_records:
        raise RuntimeError("Empty training records. Likely no matching image files were found for the train split under images-dir.")
    if not eval_records:
        raise RuntimeError("Empty eval records. Consider lowering eval-ratio or verify image paths.")

    write_jsonl(train_records, args.train_output)
    write_jsonl(eval_records, args.eval_output)

    print(f"Wrote {len(train_records)} train records -> {args.train_output}")
    print(f"Wrote {len(eval_records)} eval records -> {args.eval_output}")


if __name__ == "__main__":
    main()


