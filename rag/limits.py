"""Guards that run before any API call.

A question that fails here costs nothing: no embedding, no LLM call.
The functions are independent of Streamlit so the CLI, the eval and the
Phase 5 app share exactly the same rules.
"""

from rag import config
from rag.i18n import EMPTY_QUESTION, LIMIT_REACHED, TOO_LONG
from rag.language import detect_language


def check_question(question: str, questions_asked: int) -> str | None:
    """Return a message for the guest if the question must be blocked.

    Args:
        question: The raw text the guest typed.
        questions_asked: How many questions this session has already sent.

    Returns:
        None when the question may go ahead, otherwise a polite message
        in the guest's language.
    """
    lang = detect_language(question)
    text = question.strip()

    if not text:
        return EMPTY_QUESTION[lang]
    if questions_asked >= config.MAX_QUESTIONS_PER_SESSION:
        return LIMIT_REACHED[lang].format(limit=config.MAX_QUESTIONS_PER_SESSION)
    if len(text) > config.MAX_QUESTION_CHARS:
        return TOO_LONG[lang].format(limit=config.MAX_QUESTION_CHARS)
    return None
