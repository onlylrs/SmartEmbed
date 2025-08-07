import json
import os
import csv
import random
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from jina.utils.qwen_utils.data_utils import load_image

def convert_to_contrastive_format(
    input_path: str,
    output_path: str,
    image_dir: str = None,
    text_field: str = "text",
    image_field: str = "image"
):
    """Convert a generic JSONL corpus into the simple **contrastive** format where
    each line contains either a `{"text": ...}` or `{"image": ...}` record.

    The resulting file is suitable for quick experimentation with the
    `MultimodalDataset` implementation in `src.datasets.multimodal_dataset`.
    """
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in tqdm(f_in):
            item = json.loads(line)
            
            # 处理文本
            if text_field in item:
                text = item[text_field]
                f_out.write(json.dumps({"text": text}) + '\\n')
            
            # 处理图像（如果有）
            if image_field in item and image_dir:
                image_path = os.path.join(image_dir, item[image_field])
                try:
                    # 验证图像是否存在
                    load_image(image_path)
                    f_out.write(json.dumps({"image": image_path}) + '\\n')
                except:
                    print(f"Skipping invalid image: {image_path}")


def load_captions(captions_file: str) -> Dict[str, List[str]]:
    """Load the Flickr30K `captions.txt` file and group captions by image name."""

    image_captions: Dict[str, List[str]] = {}

    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 2:
                continue

            image_name, caption = row[0].strip(), row[1].strip()
            image_captions.setdefault(image_name, []).append(caption)

    return image_captions


def create_training_pairs(
    image_captions: Dict[str, List[str]],
    images_dir: str,
    num_negatives: int = 3,
) -> List[Dict]:
    """Generate contrastive training triples for text↔image retrieval tasks."""

    examples: List[Dict] = []

    all_images = list(image_captions.keys())
    for img_name, captions in image_captions.items():
        img_path = os.path.join(images_dir, img_name)

        for i, query_caption in enumerate(captions):
            # positive caption = another caption of the same image (or self)
            positive_choices = [cap for j, cap in enumerate(captions) if j != i] or [query_caption]
            positive_caption = random.choice(positive_choices)

            # sample negative captions from other images
            negative_captions: List[str] = []
            while len(negative_captions) < num_negatives:
                neg_img = random.choice(all_images)
                if neg_img == img_name:
                    continue
                neg_caption = random.choice(image_captions[neg_img])
                if neg_caption not in negative_captions:
                    negative_captions.append(neg_caption)

            # text→text retrieval pairs
            for neg_caption in negative_captions:
                examples.append(
                    {
                        "task": "retrieval",
                        "query": query_caption,
                        "positive": positive_caption,
                        "negative": neg_caption,
                    }
                )

            # (optional) image→text retrieval pairs if the image file exists
            if os.path.exists(img_path):
                for neg_caption in negative_captions:
                    examples.append(
                        {
                            "task": "retrieval",
                            "query": "Describe this image",
                            "positive": query_caption,
                            "negative": neg_caption,
                            "query_image": img_path,
                        }
                    )

    return examples


def save_jsonl(examples: List[Dict], output_file: str):
    """Utility to dump a list of Python dicts to a JSONL file."""

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")


# ------------------------------------------------------------------------- #
# Minimal CLI – run `python -m jina.data.preprocess <subcommand> ...`        #
# ------------------------------------------------------------------------- #

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Data preprocessing helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sub-command: convert
    convert_parser = subparsers.add_parser("convert", help="Generic JSONL → contrastive format")
    convert_parser.add_argument("input_path")
    convert_parser.add_argument("output_path")
    convert_parser.add_argument("--image_dir", default=None)
    convert_parser.add_argument("--text_field", default="text")
    convert_parser.add_argument("--image_field", default="image")

    # sub-command: flickr30k
    flickr_parser = subparsers.add_parser("prepare_flickr30k", help="Prepare Flickr30K captions")
    flickr_parser.add_argument("captions_file")
    flickr_parser.add_argument("images_dir")
    flickr_parser.add_argument("output_dir")
    flickr_parser.add_argument("--num_negatives", type=int, default=3)

    args = parser.parse_args()

    if args.command == "convert":
        convert_to_contrastive_format(
            input_path=args.input_path,
            output_path=args.output_path,
            image_dir=args.image_dir,
            text_field=args.text_field,
            image_field=args.image_field,
        )

    elif args.command == "prepare_flickr30k":
        captions = load_captions(args.captions_file)
        examples = create_training_pairs(captions, args.images_dir, args.num_negatives)

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_file = os.path.join(args.output_dir, "train.jsonl")
        eval_file = os.path.join(args.output_dir, "eval.jsonl")

        random.shuffle(examples)
        split_idx = int(0.9 * len(examples))
        save_jsonl(examples[:split_idx], train_file)
        save_jsonl(examples[split_idx:], eval_file)

        print(
            f"\n✅ Flickr30K processed: {len(examples)} examples (train={split_idx}, eval={len(examples)-split_idx})"
        )


if __name__ == "__main__":
    _cli()
