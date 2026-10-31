"""Measure the bot against eval/questions.json.

Two modes
---------
Retrieval (default): no LLM call. Did search put the expected article in
the top k? Use this to compare embedding models and query translation.

    python -m eval.run_eval                         # English (Phase 2 number)
    python -m eval.run_eval --lang all              # every language, side by side
    python -m eval.run_eval --lang all --translate  # same, with query translation
    python -m eval.run_eval --lang all --model text-embedding-3-large

Answers (--answers): the full pipeline, one LLM call per question that
passes the threshold. Reports the three Phase 3 acceptance numbers.

    python -m eval.run_eval --answers --lang all
    python -m eval.run_eval --answers --lang all --max-distance 0.55 --verbose

Metrics
-------
recall@k  Share of answerable questions whose expected article is in the top k.
MRR       Mean of 1/rank of the first correct hit. Rewards rank 1 over rank 3.
Unanswerable questions are left out of both and reported separately: the
distance of their best hit, next to the distances of correct hits, is
what sets MAX_DISTANCE.
"""

import argparse
import json

from openai import OpenAI

from rag import config
from rag.answer import answer_question
from rag.embedder import Embedder
from rag.index import load_index, rank
from rag.language import LANGUAGES, detect_language
from rag.translate import to_english

# Phase 3 acceptance targets.
TARGET_REFUSED = 7          # of the 8 unanswerable questions
TARGET_SOURCE_SHARE = 0.85  # of answerable questions citing the right article


def load_questions(path=config.QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --- Retrieval mode ------------------------------------------------------

def score_retrieval(questions: list[dict], rankings: list[list[dict]], k: int) -> dict:
    details, reciprocal_ranks, correct, control = [], [], [], []

    for item, hits in zip(questions, rankings):
        top = hits[0]["distance"]
        returned = [h["article"] for h in hits[:k]]

        if not item["answerable"]:
            control.append(top)
            details.append({"item": item, "status": "control", "top": top, "returned": returned})
            continue

        expected = set(item["expected_articles"])
        found_at = next(
            (pos for pos, h in enumerate(hits[:k], start=1) if h["article"] in expected), None
        )
        if found_at:
            reciprocal_ranks.append(1 / found_at)
            correct.append(hits[found_at - 1]["distance"])
            status = f"rank {found_at}"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"
        details.append({"item": item, "status": status, "top": top, "returned": returned})

    answerable = len(reciprocal_ranks)
    found = sum(1 for r in reciprocal_ranks if r > 0)
    return {
        "answerable": answerable,
        "found": found,
        "recall": found / answerable if answerable else 0.0,
        "mrr": sum(reciprocal_ranks) / answerable if answerable else 0.0,
        "correct_distances": correct,
        "control_distances": control,
        "details": details,
    }


def print_calibration(correct: list[float], control: list[float], label: str) -> None:
    if not correct or not control:
        return
    print(f"  {label}")
    print(f"    correct hits  {min(correct):.3f} .. {max(correct):.3f}")
    print(f"    unanswerable  {min(control):.3f} .. {max(control):.3f}")
    if max(correct) < min(control):
        print(f"    bands separate -> MAX_DISTANCE ~ {(max(correct) + min(control)) / 2:.3f}")
    else:
        # Show what each candidate threshold would cost, so the choice
        # is made with numbers instead of by eye.
        print("    bands OVERLAP. Candidate thresholds:")
        print("      value  unanswerable refused  correct hits lost")
        for value in sorted(set(control)):
            refused = sum(1 for d in control if d > value - 1e-9)
            lost = sum(1 for d in correct if d >= value - 1e-9)
            print(f"      {value - 0.001:.3f}  {refused:>10}/{len(control):<11} {lost:>6}/{len(correct)}")


def run_retrieval(questions, chunks, vectors, embedder, args) -> None:
    queries = [q["question"] for q in questions]
    if args.translate:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        queries = [to_english(q["question"], q["lang"], client) for q in questions]

    query_vectors = embedder.embed_queries(queries)
    rankings = [rank(chunks, vectors, v, top_k=max(args.k, 1)) for v in query_vectors]

    mode = "with translation" if args.translate else "no translation"
    print(f"Retrieval   model {embedder.model_name}   k={args.k}   {mode}")
    print(f"\n{'lang':<6}{'found':>8}{'recall@' + str(args.k):>11}{'MRR':>8}")
    print("-" * 33)

    results, all_correct, all_control = {}, [], []
    for lang in [l for l in LANGUAGES if any(q["lang"] == l for q in questions)]:
        idx = [i for i, q in enumerate(questions) if q["lang"] == lang]
        scores = score_retrieval([questions[i] for i in idx], [rankings[i] for i in idx], args.k)
        results[lang] = scores
        all_correct += scores["correct_distances"]
        all_control += scores["control_distances"]
        print(f"{lang:<6}{scores['found']:>4}/{scores['answerable']:<3}"
              f"{scores['recall']:>11.3f}{scores['mrr']:>8.3f}")

    print("\nThreshold calibration:")
    for lang, scores in results.items():
        print_calibration(scores["correct_distances"], scores["control_distances"], lang)
    if len(results) > 1:
        print_calibration(all_correct, all_control, "all languages (one threshold for the app)")

    if args.verbose:
        for lang, scores in results.items():
            print(f"\n--- {lang} ---")
            for d in scores["details"]:
                item = d["item"]
                expected = ",".join(item["expected_articles"]) or "-"
                print(f"  [{d['status']:>7}] {item['id']}  {item['question']}")
                print(f"            expected {expected:<6} returned {d['returned']}  "
                      f"top distance {d['top']:.3f}")
                if "trap" in item:
                    print(f"            trap: {item['trap']}")


# --- Answers mode --------------------------------------------------------

def run_answers(questions, chunks, vectors, embedder, args) -> None:
    rows = []
    for item in questions:
        result = answer_question(
            chunks, vectors, embedder, item["question"],
            top_k=args.k, max_distance=args.max_distance, translate=args.translate,
        )
        refused = not result["grounded"]
        if item["answerable"]:
            source_ok = bool(set(item["expected_articles"]) & set(result["cited_articles"]))
        else:
            source_ok = None
        # Refusals are fixed texts in the detected language, so check them
        # through the detected language; answers are checked on their text.
        answer_lang = result["language"] if refused else detect_language(result["answer"])
        rows.append({
            "item": item,
            "result": result,
            "refused": refused,
            "source_ok": source_ok,
            "lang_ok": answer_lang == item["lang"],
            "answer_lang": answer_lang,
        })

    unanswerable = [r for r in rows if not r["item"]["answerable"]]
    answerable = [r for r in rows if r["item"]["answerable"]]
    refused_ok = sum(r["refused"] for r in unanswerable)
    wrongly_refused = sum(r["refused"] for r in answerable)
    sources_ok = sum(bool(r["source_ok"]) for r in answerable)
    lang_ok = sum(r["lang_ok"] for r in rows)
    reasons = [r["result"]["refusal_reason"] for r in rows]
    by_threshold = reasons.count("threshold")
    truncated = reasons.count("truncated")
    api_errors = reasons.count("api_error")
    tokens = [r["result"]["output_tokens"] for r in rows if r["result"]["output_tokens"]]

    share = sources_ok / len(answerable) if answerable else 0.0
    mode = "with translation" if args.translate else "no translation"
    print(f"Answers   {config.CHAT_MODEL} + {embedder.model_name}   "
          f"threshold {args.max_distance:.3f}   {mode}\n")
    print(f"  (1) unanswerable refused     {refused_ok}/{len(unanswerable)}   target >= {TARGET_REFUSED} of 8")
    print(f"  (2) right article cited      {sources_ok}/{len(answerable)} = {share:.0%}   target >= {TARGET_SOURCE_SHARE:.0%}")
    print(f"      answerable but refused   {wrongly_refused}/{len(answerable)}")
    print(f"  (3) reply in question lang   {lang_ok}/{len(rows)}   target all")
    print(f"      refusals by threshold    {by_threshold} (LLM not called)")
    if tokens:
        print(f"      output tokens            avg {sum(tokens) / len(tokens):.0f}, "
              f"max {max(tokens)} of {config.MAX_OUTPUT_TOKENS} allowed")
    print(f"      cut off (truncated)      {truncated}   must stay 0")
    if api_errors:
        print(f"      API ERRORS               {api_errors}   results are not valid, check key and credit")

    full_set = len(unanswerable) == 8 and len(rows) == 30
    if full_set:
        passed = (refused_ok >= TARGET_REFUSED and share >= TARGET_SOURCE_SHARE
                  and lang_ok == len(rows) and truncated == 0 and api_errors == 0)
        print(f"\n  Phase 3 acceptance: {'PASS' if passed else 'NOT YET'}")
    else:
        print("\n  Phase 3 acceptance is judged on the full set: run with --lang all")

    for r in rows:
        item, result = r["item"], r["result"]
        problem = (
            (not item["answerable"] and not r["refused"])
            or (item["answerable"] and not r["source_ok"])
            or not r["lang_ok"]
        )
        if not (args.verbose or problem):
            continue
        flag = "FAIL" if problem else " ok "
        expected = ",".join(item["expected_articles"]) or "refuse"
        got = "refused (" + result["refusal_reason"] + ")" if r["refused"] else ",".join(result["cited_articles"])
        print(f"\n  [{flag}] {item['id']}  {item['question']}")
        top = f"{result['hits'][0]['distance']:.3f}" if result["hits"] else "-"
        print(f"         expected {expected}   got {got}   reply lang {r['answer_lang']}   "
              f"top distance {top}")
        if not r["refused"]:
            print(f"         {result['answer'][:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hotel bot.")
    parser.add_argument("--lang", choices=list(LANGUAGES) + ["all"], default="en")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="Articles retrieved per question.")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model / index to use.")
    parser.add_argument("--translate", action="store_true", default=config.QUERY_TRANSLATION,
                        help="Translate TR/DE questions to English before searching.")
    parser.add_argument("--answers", action="store_true", help="Run the full pipeline with the LLM.")
    parser.add_argument("--max-distance", type=float, default=config.MAX_DISTANCE,
                        help="Refusal threshold for --answers.")
    parser.add_argument("--verbose", action="store_true", help="Print every question.")
    args = parser.parse_args()

    languages = LANGUAGES if args.lang == "all" else (args.lang,)
    questions = [q for q in load_questions() if q["lang"] in languages]

    chunks, vectors = load_index(args.model)
    embedder = Embedder(args.model)

    if args.answers:
        embedder.log = False  # one call per question; the total is printed at the end
        run_answers(questions, chunks, vectors, embedder, args)
    else:
        run_retrieval(questions, chunks, vectors, embedder, args)

    print(f"\nEmbedding API calls this run: {embedder.api_calls}")


if __name__ == "__main__":
    main()
