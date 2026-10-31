"""Embed the hotel document once and save the index to disk.

Run this only when data/hotel_info.md changes, or to try another model:

    python build_index.py
    python build_index.py --model text-embedding-3-large

Each model gets its own folder under index/. The app and the eval load
the saved index and never re-embed the document.
"""

import argparse

from rag import config
from rag.chunker import load_articles
from rag.embedder import Embedder
from rag.index import save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the article index.")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model.")
    args = parser.parse_args()

    articles = load_articles(config.DOCUMENT_PATH)
    print(f"Parsed {len(articles)} articles from {config.DOCUMENT_PATH.name}")

    embedder = Embedder(args.model)
    vectors = embedder.embed_documents([a["embed_text"] for a in articles])
    save_index(articles, vectors, embedder.model_name)

    print(
        f"Saved {len(articles)} vectors ({vectors.shape[1]} dimensions) "
        f"to {config.index_dir(args.model).relative_to(config.BASE_DIR)}/"
    )


if __name__ == "__main__":
    main()
