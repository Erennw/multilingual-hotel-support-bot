"""Store the embedded document on disk and search it.

The whole index is two files in index/<embedding model>/: vectors.npy
(one row per article) and chunks.json (the text and metadata for each row, in the same order).
With about 70 articles a vector database would add a dependency and a
moving part for no gain; a NumPy dot product searches this in microseconds.
"""

import json

import numpy as np

from rag import config
from rag.embedder import Embedder


class IndexMismatchError(Exception):
    """The saved index does not fit the current settings or itself."""


def save_index(chunks: list[dict], vectors: np.ndarray, model_name: str) -> None:
    """Write vectors and chunk metadata side by side."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    folder = config.index_dir(model_name)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "vectors.npy", vectors.astype(np.float32))

    payload = {
        # Stored so a model change in .env is caught at load time instead
        # of silently comparing vectors from two different models.
        "embedding_model": model_name,
        "dimension": int(vectors.shape[1]),
        "chunks": chunks,
    }
    (folder / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(expected_model: str = config.EMBEDDING_MODEL) -> tuple[list[dict], np.ndarray]:
    """Read the saved index. Makes no API calls.

    Raises:
        FileNotFoundError: build_index.py has not been run yet.
        IndexMismatchError: the index was built with another model, or
            the two files are out of sync.
    """
    folder = config.index_dir(expected_model)
    vectors_path, chunks_path = folder / "vectors.npy", folder / "chunks.json"
    if not vectors_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(
            f"No index for {expected_model}. Run: python build_index.py --model {expected_model}"
        )

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    chunks = payload["chunks"]

    if payload["embedding_model"] != expected_model:
        raise IndexMismatchError(
            f"Index was built with {payload['embedding_model']}, but the "
            f"current model is {expected_model}. Run: python build_index.py --model {expected_model}"
        )
    if len(chunks) != len(vectors):
        raise IndexMismatchError(
            f"{len(chunks)} chunks but {len(vectors)} vectors. Run: python build_index.py"
        )

    return chunks, vectors


def rank(
    chunks: list[dict], vectors: np.ndarray, query_vector: np.ndarray, top_k: int = config.TOP_K
) -> list[dict]:
    """Return the top_k chunks closest to an already-embedded query.

    Kept separate from search() so the eval can embed all questions in
    one API call and rank them here without further calls.

    Returns:
        Copies of the chunks with a "distance" key added, closest first.
    """
    similarities = vectors @ query_vector
    order = np.argsort(-similarities)[:top_k]
    return [
        {**chunks[i], "distance": float(1.0 - similarities[i])}
        for i in order
    ]


def search(
    chunks: list[dict],
    vectors: np.ndarray,
    embedder: Embedder,
    question: str,
    top_k: int = config.TOP_K,
) -> list[dict]:
    """Embed one question and return the closest chunks."""
    return rank(chunks, vectors, embedder.embed_query(question), top_k)
