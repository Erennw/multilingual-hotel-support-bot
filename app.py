"""Streamlit chat screen for the Any Hotel guest assistant.

    streamlit run app.py

The index is loaded once per server process (st.cache_resource), so
starting the app or opening a new session never calls the embeddings
API for the document. Every question goes through the same guards and
pipeline as the CLI and the eval: rag.limits, then rag.answer.
"""

import streamlit as st

from rag import config
from rag.answer import answer_question
from rag.embedder import Embedder
from rag.i18n import LIMIT_REACHED, PRIVACY_NOTE
from rag.index import load_index
from rag.limits import check_question

st.set_page_config(page_title="Any Hotel Assistant", page_icon="🏨", layout="centered")

# Two per language. The casino question is not in the document on
# purpose, so a visitor can see a refusal without typing anything.
EXAMPLES = [
    ("EN", "What time is check-in?"),
    ("EN", "Is there a casino in the hotel?"),
    ("DE", "Kann ich meinen Hund mitbringen?"),
    ("DE", "Was kostet ein Transfer vom Flughafen Gazipaşa?"),
    ("TR", "Kahvaltı saat kaçta bitiyor?"),
    ("TR", "Geç çıkış yapabilir miyim, ücreti ne kadar?"),
]

FORWARDED_BADGE = {"en": "Forwarded to reception", "tr": "Resepsiyona iletildi", "de": "An die Rezeption weitergeleitet"}
SOURCES_LABEL = {"en": "Source", "tr": "Kaynak", "de": "Quelle"}


@st.cache_resource(show_spinner="Loading hotel information...")
def load_bot():
    """Load the saved index once per server process. No API call."""
    chunks, vectors = load_index()
    embedder = Embedder()
    embedder.log = False
    return chunks, vectors, embedder


def render_message(message: dict) -> None:
    """Draw one chat turn, with sources or the reception badge under answers."""
    with st.chat_message(message["role"], avatar="🧳" if message["role"] == "user" else "🏨"):
        st.markdown(message["content"])
        kind = message.get("kind")
        lang = message.get("lang", "en")
        if kind == "answer" and message.get("sources"):
            with st.expander(f"{SOURCES_LABEL[lang]} ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")
        elif kind == "forwarded":
            st.badge(FORWARDED_BADGE[lang], icon=":material/support_agent:", color="orange")
        elif kind == "notice":
            st.badge("Demo", icon=":material/info:", color="gray")


def ask(question: str) -> None:
    """Run one question through the guards and the pipeline, store both turns."""
    messages = st.session_state.messages
    messages.append({"role": "user", "content": question})

    blocked = check_question(question, st.session_state.questions_asked)
    if blocked:
        messages.append({"role": "assistant", "content": blocked, "kind": "notice"})
        return

    # Counted before the call, so an API error still uses up a question.
    st.session_state.questions_asked += 1
    chunks, vectors, embedder = load_bot()
    with st.spinner("..."):
        result = answer_question(chunks, vectors, embedder, question)

    if result["grounded"]:
        kind = "answer"
    elif result["refusal_reason"] == "api_error":
        kind = "notice"
    else:
        kind = "forwarded"
    messages.append({
        "role": "assistant",
        "content": result["answer"],
        "kind": kind,
        "lang": result["language"],
        "sources": result["sources"],
    })


# --- State -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.questions_asked = 0

remaining = max(config.MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked, 0)

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.metric("Questions left", f"{remaining} / {config.MAX_QUESTIONS_PER_SESSION}")
    if config.REPO_URL:
        st.link_button("Source code on GitHub", config.REPO_URL, icon=":material/code:")
    st.markdown("#### How it works")
    st.markdown(
        "1. Your question is matched against the hotel's information document.\n"
        "2. If nothing relevant is found, the bot does **not** guess: it forwards you to reception.\n"
        "3. Otherwise an AI model answers **only** from the matching articles, in your language.\n"
        "4. The cited articles are added by code, so sources cannot be made up."
    )
    st.caption(
        "Reception forwarding is simulated: nothing is sent anywhere. "
        "Any Hotel is fictional."
    )
    if st.session_state.messages and st.button("Clear chat", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()

# --- Header ------------------------------------------------------------------
st.title("🏨 Any Hotel · Guest Assistant")
st.markdown("Answers guest questions 24/7 in **English, Deutsch and Türkçe**, with the source for every answer.")
st.badge("Demo for a fictional hotel", icon=":material/science:", color="violet")

try:
    load_bot()
except Exception as error:  # missing index or key: show a clear message, not a trace
    st.error(f"The demo is not configured correctly: {type(error).__name__}.")
    st.stop()

# --- Example questions ---------------------------------------------------------
# Open before the first question; folded afterwards so the chat stays in
# view, but still there, so the whole demo works without typing.
with st.expander("**Try a question**", expanded=not st.session_state.messages):
    columns = st.columns(2)
    for position, (flag, text) in enumerate(EXAMPLES):
        if columns[position % 2].button(f"{flag} · {text}", key=f"example-{position}", width="stretch", wrap=True):
            ask(text)
            st.rerun()

# --- Chat ----------------------------------------------------------------------
for message in st.session_state.messages:
    render_message(message)

if remaining == 0:
    st.info(LIMIT_REACHED["en"].format(limit=config.MAX_QUESTIONS_PER_SESSION), icon=":material/block:")

if prompt := st.chat_input(
    "Ask about rooms, check-in, transfers, pets...",
    max_chars=config.MAX_QUESTION_CHARS,
    disabled=remaining == 0,
):
    ask(prompt)
    st.rerun()

st.caption(" · ".join(PRIVACY_NOTE[lang] for lang in ("en", "de", "tr")))
