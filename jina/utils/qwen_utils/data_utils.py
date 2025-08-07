"""Utility helpers previously bundled with a legacy multimodal dataset implementation.

Only the `load_image` helper is retained because it is still used by
`jina/data/preprocess.py` to validate image paths when converting raw
datasets.  The full Dataset / collate_fn definitions were removed to
avoid confusion—our project now uses `src.datasets.multimodal_dataset` as
the single source of truth for data loading.
"""

from io import BytesIO
from typing import Union

import requests
from PIL import Image


def load_image(image_file: Union[str, bytes]) -> Image.Image:
    """Load an image from a local path or an HTTP(S) URL and convert it to RGB.

    This small helper is shared by preprocessing scripts to make sure an
    image actually exists and is readable before its path is written to
    the resulting JSONL file.
    """

    if isinstance(image_file, bytes):
        image_file = image_file.decode()

    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    return image
