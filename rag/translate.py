"""Optional: translate a question into English before searching.

The document is English. If a multilingual embedding still matches a
German question to the wrong article, translating the question first
puts both sides in the same language. The cost is one extra LLM call per
non-English question, which is why it is a switch (config.QUERY_TRANSLATION)
and not always on. eval/run_eval.py --translate measures whether it is
worth that cost.
"""

from openai import OpenAI

from rag import config

INSTRUCTIONS = (
    "Translate the hotel guest's question into English. "
    "Keep names, numbers and dates unchanged. Reply with the translation only."
)


def to_english(question: str, lang: str, client: OpenAI | None = None) -> str:
    """Return the question in English. English input is returned as is."""
    if lang == "en":
        return question
    client = client or OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.responses.create(
        model=config.CHAT_MODEL,
        instructions=INSTRUCTIONS,
        input=question,
        # Translation needs no reasoning; "none" keeps it fast and cheap.
        reasoning={"effort": "none"},
        max_output_tokens=200,
        store=False,
    )
    return response.output_text.strip() or question
