"""Detect whether a text is Turkish, German or English, without an API call.

The bot only serves three languages, so a full language-detection
library is not needed. A refusal skips the LLM entirely, which means the
language has to be known before any model is called; this module is
what makes that possible at zero cost.

The method is simple scoring: characters and short function words that
are frequent in one language and rare in the other two. English is the
fallback because it is the document's language.
"""

import re

LANGUAGES = ("en", "tr", "de")

# Letters that occur in one of the three languages only. "ö" and "ü"
# are shared by Turkish and German, so they are deliberately missing.
_CHARS = {
    "tr": set("ğışçİĞŞÇ"),
    "de": set("äßÄ"),
}

_WORDS = {
    "tr": {
        "bir", "ve", "mi", "mı", "mu", "mü", "var", "yok", "ne", "nasıl", "kaç",
        "için", "ile", "bu", "da", "de", "ben", "biz", "otelde", "saat", "ücret",
        "miyim", "mıyım", "misiniz", "musunuz", "yapabilir", "edebilir", "olarak",
        "kadar", "neden", "hangi", "merhaba", "teşekkürler", "lütfen", "çocuk",
    },
    "de": {
        "ich", "ist", "der", "die", "das", "und", "gibt", "es", "kann", "können",
        "wie", "was", "bis", "wann", "man", "sie", "ein", "eine", "einen", "meinen",
        "mit", "im", "zum", "zur", "nicht", "auf", "für", "wird", "werden", "bitte",
        "hallo", "danke", "uhr", "kinder", "hotel", "sind", "auch", "von", "dem",
    },
    "en": {
        "the", "is", "are", "what", "how", "can", "do", "does", "you", "i", "my",
        "we", "our", "is", "there", "and", "of", "to", "for", "much", "when",
        "where", "which", "have", "your", "please", "hello", "thanks",
    },
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """Return "tr", "de" or "en" for the given text."""
    words = [w.lower() for w in _WORD.findall(text)]
    scores = {lang: 0 for lang in LANGUAGES}

    for word in words:
        for lang, vocabulary in _WORDS.items():
            if word in vocabulary:
                scores[lang] += 1
        # A word containing a language-specific letter counts once, not
        # once per letter. Otherwise a single place name such as
        # "Gazipaşa" in a German sentence outweighs the German words.
        for lang, chars in _CHARS.items():
            if any(ch in chars for ch in word):
                scores[lang] += 1

    best = max(LANGUAGES, key=lambda lang: scores[lang])
    return best if scores[best] > 0 else "en"
