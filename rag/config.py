"""Central configuration.

Every tunable value lives here so experiments only touch one file.
Model names come from .env, so swapping a model is a one-line change
that needs no code edit.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = BASE_DIR / "data" / "hotel_info.md"
INDEX_ROOT = BASE_DIR / "index"
QUESTIONS_PATH = BASE_DIR / "eval" / "questions.json"

# --- Secrets -------------------------------------------------------------
def _read_secret(name: str) -> str | None:
    """Read a secret from the environment, then from Streamlit secrets.

    Locally the key comes from .env (loaded above). On Streamlit Community
    Cloud it is entered in the Secrets panel. Streamlit's docs do not say
    whether those always appear in os.environ, so st.secrets is checked
    as a fallback. Streamlit is imported lazily: the CLI and the eval
    work without it installed.

    The value is never printed or logged anywhere in this project.
    """
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        # Streamlit not installed, or no secrets file: not an error here.
        # Embedder raises a clear message if the key is still missing.
        return None


# --- OpenAI --------------------------------------------------------------
OPENAI_API_KEY = _read_secret("OPENAI_API_KEY")
CHAT_MODEL = _read_secret("OPENAI_CHAT_MODEL") or "gpt-6-luna"
EMBEDDING_MODEL = _read_secret("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"

# Link shown in the app's sidebar. Public, so it lives in code rather
# than in secrets, where a leftover placeholder would break the button.
REPO_URL = "https://github.com/Erennw/multilingual-hotel-support-bot"


def index_dir(model_name: str = EMBEDDING_MODEL) -> Path:
    """One index folder per embedding model, e.g. index/text-embedding-3-small/.

    Keeping them side by side lets Phase 3 compare two models without
    rebuilding the other one each time.
    """
    return INDEX_ROOT / model_name

# gpt-6-luna is a reasoning model. Reasoning tokens count toward
# max_output_tokens, so a high effort with a low token cap can cut the
# visible answer short. "low" is a starting point; Phase 3 measures it.
REASONING_EFFORT = "low"
# Upper bound on cost per answer. eval/run_eval.py --answers prints the
# largest output seen and how many replies were cut off; lower this only
# while "truncated" stays at 0.
MAX_OUTPUT_TOKENS = 600

# --- Retrieval -----------------------------------------------------------
TOP_K = 3

# Translate non-English questions into English before searching.
# Costs one extra LLM call per TR/DE question. Turn on only if
# eval/run_eval.py --translate shows a clear gain (Phase 3a).
QUERY_TRANSLATION = False

# Cosine distance = 1 - cosine similarity, so LOWER is more similar.
# Calibrated with eval/run_eval.py --lang all (2026-10-31): correct hits
# span 0.208..0.760, unanswerable questions 0.534..0.734. The bands
# overlap, so no threshold separates them on its own; the model's
# "answerable" flag handles the overlap. 0.80 loses no answerable
# question and still keeps clearly off-topic text away from the LLM.
MAX_DISTANCE = 0.80

# --- Abuse limits (Phase 4) ----------------------------------------------
# These protect the prepaid credit, which is the real spending ceiling.
# The per-session count lives in the browser session, so a page refresh
# resets it. That is accepted for a demo and listed as a known limitation.
MAX_QUESTION_CHARS = 300
MAX_QUESTIONS_PER_SESSION = 15

