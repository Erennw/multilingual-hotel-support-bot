"""Turn text into vectors with the OpenAI embeddings API.

Every API call is counted. The counter is how the Phase 2 acceptance
test is checked: starting the app must not embed the document again,
so the count after loading the index has to be zero.
"""

import sys

import numpy as np
from openai import OpenAI

from rag import config

# OpenAI accepts up to 2048 inputs per request; staying well below keeps
# each request small if the document grows.
BATCH_SIZE = 100


class Embedder:
    """Encode text into unit-length vectors."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and "
                "fill in your key."
            )
        self.model_name = model_name
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.api_calls = 0
        # Set to False to silence the per-call log (the count still runs).
        self.log = True

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return a (len(texts), dim) array.

        Vectors are normalised to length 1 here rather than trusting the
        API to do it. With unit vectors, cosine similarity is a plain dot
        product, which is what index.search() relies on.
        """
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            self.api_calls += 1
            if self.log:
                print(
                    f"[embeddings] call {self.api_calls}: {len(batch)} text(s), "
                    f"{response.usage.total_tokens} tokens, {self.model_name}",
                    file=sys.stderr,
                )
            # Sort by index: the API documents an index per item, and
            # relying on it is safer than relying on response order.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append(item.embedding)

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode document chunks. Used only by build_index.py."""
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Encode several questions in one request (used by the eval)."""
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single user question."""
        return self._encode([text])[0]
