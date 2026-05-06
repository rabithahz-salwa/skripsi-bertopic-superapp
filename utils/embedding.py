"""
Embedding utilities for BERTopic pipeline.

Generates document embeddings using IndoBERT-p2 via sentence-transformers
wrapper (mean pooling auto-active). Implements load-or-generate caching
pattern to avoid expensive recomputation on notebook re-runs.

Compute budget on CPU (no CUDA):
    wondr  (8,982 reviews)  : ~20-30 min first run, ~2 sec from cache
    BYOND  (19,393 reviews) : ~40-60 min first run, ~3 sec from cache

Cache invalidation: if upstream preprocessing changes (new cleaning step,
different filtering), DELETE the .npy cache file manually before re-run
— the function does NOT detect input changes automatically.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer


# IndoBERT model identifier on Hugging Face Hub.
# Already cached locally at C:\Users\<user>\.cache\huggingface\hub\
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p2"


def load_indobert_model():
    """
    Load IndoBERT-p2 as a sentence-transformers model.

    Mean pooling is auto-active when loading a HuggingFace BERT model
    via SentenceTransformer wrapper (Reimers & Gurevych, 2019).
    Output embeddings have dimensionality 768.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        Loaded IndoBERT-p2 model ready for .encode() calls.

    Examples
    --------
    >>> model = load_indobert_model()
    >>> emb = model.encode(["contoh kalimat"])
    >>> emb.shape
    (1, 768)
    """
    return SentenceTransformer(INDOBERT_MODEL_NAME)


def generate_embeddings(texts, model, cache_path, batch_size=32, show_progress_bar=True):
    """
    Generate or load cached embeddings for a list of texts.

    Implements load-or-generate pattern:
    - If `cache_path` exists -> load from disk (fast, ~seconds).
    - If not -> generate with `model.encode()`, save to disk, return.

    Parameters
    ----------
    texts : list of str
        Documents to embed. Order matters — embedding at index i
        corresponds to texts[i].
    model : sentence_transformers.SentenceTransformer
        Loaded embedding model (use load_indobert_model()).
    cache_path : str
        Path to .npy file for caching. Parent directory must exist.
        Convention: data/embeddings/<app_name>_embeddings.npy
    batch_size : int, default 32
        Batch size for model.encode(). Higher = faster but more RAM.
        32 is safe for CPU; bump to 64 if RAM allows.
    show_progress_bar : bool, default True
        Show tqdm progress bar during generation. Useful for long runs.

    Returns
    -------
    numpy.ndarray
        Embedding matrix of shape (len(texts), 768), dtype float32.

    Raises
    ------
    FileNotFoundError
        If cache_path's parent directory doesn't exist.

    Examples
    --------
    >>> model = load_indobert_model()
    >>> texts = ["aplikasi error terus", "transfer gagal"]
    >>> emb = generate_embeddings(texts, model, "data/embeddings/test.npy")
    >>> emb.shape
    (2, 768)
    """
    # Cache hit: load and return immediately
    if os.path.exists(cache_path):
        print(f"[cache hit] Loading embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        print(f"[cache hit] Shape: {embeddings.shape}")
        return embeddings

    # Cache miss: validate parent dir, generate, save
    parent_dir = os.path.dirname(cache_path)
    if parent_dir and not os.path.exists(parent_dir):
        raise FileNotFoundError(
            f"Parent directory does not exist: {parent_dir}. "
            f"Create it first with: os.makedirs('{parent_dir}', exist_ok=True)"
        )

    print(f"[cache miss] Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    # Cast to float32 to halve disk usage (model returns float32 already
    # but explicit cast is defensive against future model changes)
    embeddings = embeddings.astype(np.float32)

    np.save(cache_path, embeddings)
    print(f"[cache miss] Saved to {cache_path}, shape: {embeddings.shape}")
    return embeddings
