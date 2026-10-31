"""Fixed texts in the guest's language.

These are written by hand, not generated, because they are shown
without calling the LLM: the refusal skips the model entirely, and the
source labels are built from metadata so the model cannot invent them.

Section titles are keyed by section number. If a section is added to
the document without a translation here, the English title from the
document is shown instead of failing.
"""

REFUSAL = {
    "en": "I don't have this information. I'm forwarding your question to reception, they will be happy to help.",
    "tr": "Bu konuda bilgim yok. Sorunuzu resepsiyona iletiyorum, size yardımcı olacaklar.",
    "de": "Dazu habe ich leider keine Informationen. Ich leite Ihre Frage an die Rezeption weiter, dort hilft man Ihnen gerne.",
}

# Shown when a guard in rag/limits.py stops a question before any API call.
TOO_LONG = {
    "en": "Please keep your question under {limit} characters.",
    "tr": "Lütfen sorunuzu {limit} karakterin altında tutun.",
    "de": "Bitte halten Sie Ihre Frage unter {limit} Zeichen.",
}

LIMIT_REACHED = {
    "en": "You have reached the question limit for this demo ({limit} questions). Thank you for trying it!",
    "tr": "Bu demodaki soru hakkınız doldu ({limit} soru). Denediğiniz için teşekkürler!",
    "de": "Sie haben das Fragelimit dieser Demo erreicht ({limit} Fragen). Danke fürs Ausprobieren!",
}

EMPTY_QUESTION = {
    "en": "Please type a question.",
    "tr": "Lütfen bir soru yazın.",
    "de": "Bitte geben Sie eine Frage ein.",
}

# Shown on any OpenAI error, including an exhausted demo credit. The
# error itself is never shown: it can contain request details.
SERVICE_UNAVAILABLE = {
    "en": "The demo is temporarily unavailable (its usage quota may be used up). Please try again later.",
    "tr": "Demo şu anda kullanılamıyor (kullanım kotası dolmuş olabilir). Lütfen daha sonra tekrar deneyin.",
    "de": "Die Demo ist vorübergehend nicht verfügbar (das Nutzungskontingent ist eventuell aufgebraucht). Bitte später erneut versuchen.",
}

# For the chat screen in Phase 5.
PRIVACY_NOTE = {
    "en": "Demo for a fictional hotel. Questions are not saved. Please do not enter personal data.",
    "tr": "Hayali bir otel için demo. Sorular kaydedilmez. Lütfen kişisel bilgi girmeyin.",
    "de": "Demo für ein fiktives Hotel. Fragen werden nicht gespeichert. Bitte keine persönlichen Daten eingeben.",
}

ARTICLE_LABEL = {"en": "art.", "tr": "md.", "de": "Art."}

SECTION_TITLES = {
    "1": {"tr": "Genel Bilgiler ve Konum", "de": "Allgemeine Informationen und Lage"},
    "2": {"tr": "Oda Tipleri", "de": "Zimmertypen"},
    "3": {"tr": "Giriş ve Çıkış", "de": "Check-in und Check-out"},
    "4": {"tr": "İptal Politikası", "de": "Stornierungsbedingungen"},
    "5": {"tr": "Ödeme Yöntemleri", "de": "Zahlungsmethoden"},
    "6": {"tr": "Kahvaltı ve Restoranlar", "de": "Frühstück und Restaurants"},
    "7": {"tr": "Havuzlar, Plaj ve Spa", "de": "Pools, Strand und Spa"},
    "8": {"tr": "Havalimanı Transferi", "de": "Flughafentransfer"},
    "9": {"tr": "Evcil Hayvanlar", "de": "Haustiere"},
    "10": {"tr": "Çocuk Politikası", "de": "Kinderregelung"},
    "11": {"tr": "Çocuk Kulübü ve Aile Aktiviteleri", "de": "Kinderclub und Familienaktivitäten"},
    "12": {"tr": "Erişilebilirlik", "de": "Barrierefreiheit"},
    "13": {"tr": "İletişim", "de": "Kontakt"},
}


def source_label(hit: dict, lang: str) -> str:
    """E.g. 'İptal Politikası, md. 4.2' for a Turkish question."""
    title = SECTION_TITLES.get(hit["section_number"], {}).get(lang, hit["section_title"])
    return f"{title}, {ARTICLE_LABEL.get(lang, 'art.')} {hit['article']}"
