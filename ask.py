"""Ask the bot one question from the terminal.

    python ask.py "What time is check-in?"
    python ask.py "Kann ich meinen Hund mitbringen?" --debug
    python ask.py "Kann ich meinen Hund mitbringen?" --debug --translate

--debug prints every retrieved article with its distance, which is what
you look at first when an answer is wrong.
"""

import argparse

from rag import config
from rag.answer import answer_question
from rag.embedder import Embedder
from rag.index import load_index
from rag.limits import check_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the hotel bot one question.")
    parser.add_argument("question", help="The guest question, in quotes.")
    parser.add_argument("--debug", action="store_true", help="Show retrieved articles.")
    parser.add_argument("--translate", action="store_true", default=config.QUERY_TRANSLATION,
                        help="Translate the question to English before searching.")
    parser.add_argument("--max-distance", type=float, default=config.MAX_DISTANCE,
                        help="Override the refusal threshold for this run.")
    args = parser.parse_args()

    blocked = check_question(args.question, questions_asked=0)
    if blocked:
        print(blocked)
        return

    chunks, vectors = load_index()
    embedder = Embedder()
    print(f"Index loaded: {len(chunks)} articles, embedding calls so far: {embedder.api_calls}")

    result = answer_question(
        chunks, vectors, embedder, args.question,
        max_distance=args.max_distance, translate=args.translate,
    )

    print(f"\n{result['answer']}")
    if result["sources"]:
        print()
        for source in result["sources"]:
            print(f"  - {source}")

    if args.debug:
        print(f"\nLanguage: {result['language']}   Refusal: {result['refusal_reason'] or '-'}"
              f"   Output tokens: {result['output_tokens']}")
        if result["search_query"] != args.question:
            print(f"Searched as: {result['search_query']}")
        print(f"Retrieved (threshold {args.max_distance:.3f}):")
        for hit in result["hits"]:
            mark = " " if hit["distance"] <= args.max_distance else "x"
            print(f"  {mark} {hit['distance']:.3f}  {hit['article']:>5}  {hit['text'][:65]}")

    print(f"\nEmbedding API calls this run: {embedder.api_calls}")


if __name__ == "__main__":
    main()
