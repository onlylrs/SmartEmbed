"""Deprecated helper package.

The project now uses `src.datasets.multimodal_dataset` as the sole data
loading pipeline.  This namespace is kept so older utility scripts that do
`import jina.data` do not crash, but it no longer re-exports any Dataset
implementations.

Only the *unified* preprocessing helper `jina.data.preprocess` is
maintained here.  Run `python -m jina.data.preprocess --help` to see the
available sub-commands (generic *convert* and *prepare_flickr30k*).
"""

from typing import List

__all__: List[str] = []