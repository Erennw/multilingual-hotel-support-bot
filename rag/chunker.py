"""Split the hotel document into one chunk per article.

The document is written as numbered articles ("4.2 High season ..."),
so the natural chunk boundary is already there. Splitting by character
count would cut an article in half or glue two unrelated rules together,
and every citation would point at a blurred piece of text.

Each chunk carries its section title and article number as metadata.
Citations are built from that metadata in code, never by the LLM.
"""

import re
from pathlib import Path

# "## 4. Cancellation Policy"
SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")

# "4.2 High season arrivals ..."
ARTICLE_LINE = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")


class DocumentFormatError(Exception):
    """The document breaks the numbering rules the chunker relies on."""


def parse_articles(text: str) -> list[dict]:
    """Turn the markdown document into a list of article chunks.

    Returns:
        A list of {"article", "section_number", "section_title", "text",
        "embed_text"} dicts in document order.

        "text" is the article as written, for showing to the user.
        "embed_text" prefixes the section title, so an article that only
        says "Maximum one per room" still carries the topic it belongs to.

    Raises:
        DocumentFormatError: an article appears outside a section, its
            number does not match its section, or a number repeats.
            Failing loudly here is cheaper than debugging a wrong
            citation in Phase 3.
    """
    chunks: list[dict] = []
    seen: set[str] = set()
    section_number = None
    section_title = None

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()

        heading = SECTION_HEADING.match(line)
        if heading:
            section_number = heading.group(1)
            section_title = heading.group(2)
            continue

        article = ARTICLE_LINE.match(line)
        if not article:
            continue

        major, minor, body = article.groups()
        article_id = f"{major}.{minor}"

        if section_number is None:
            raise DocumentFormatError(
                f"Line {line_number}: article {article_id} has no section heading above it."
            )
        if major != section_number:
            raise DocumentFormatError(
                f"Line {line_number}: article {article_id} sits under section "
                f"{section_number} ({section_title})."
            )
        if article_id in seen:
            raise DocumentFormatError(
                f"Line {line_number}: article {article_id} appears twice."
            )
        seen.add(article_id)

        chunks.append(
            {
                "article": article_id,
                "section_number": section_number,
                "section_title": section_title,
                "text": body,
                "embed_text": f"{section_title}: {body}",
            }
        )

    if not chunks:
        raise DocumentFormatError("No numbered articles found in the document.")

    return chunks


def load_articles(path: str | Path) -> list[dict]:
    """Read the document from disk and parse it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such document: {path}")
    return parse_articles(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    # Quick manual check: python -m rag.chunker
    from rag import config

    articles = load_articles(config.DOCUMENT_PATH)
    sections = {a["section_number"] for a in articles}
    print(f"Sections: {len(sections)}  Articles: {len(articles)}")
    print("\n--- First chunk as embedded ---")
    print(articles[0]["embed_text"])
