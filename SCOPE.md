# Scope - Any Hotel Support Bot

A multilingual (TR / EN / DE) guest support chatbot for a **fictional** hotel,
built as a portfolio demo. Read this file whenever a new feature idea shows up.

## The hotel (fictional)

| Field | Value |
|---|---|
| Name | Any Hotel |
| Location | Alanya, Antalya, Türkiye |
| Size | 250 rooms |
| Target guests | Families on holiday |

Any Hotel does not exist. No real hotel's name, logo or photos are used.
Name checked on 2026-10-31: no hotel called "Any Hotel" found in Alanya.

## Provider and models

| Role | Choice | Price (per 1M tokens) |
|---|---|---|
| Provider | OpenAI | - |
| Chat model | `gpt-6-luna` | $0.10 input / $0.50 output |
| Embedding model | `text-embedding-3-small` | ~$0.02 |

- Verified on 2026-10-31 from OpenAI's docs:
  [pricing](https://developers.openai.com/api/docs/pricing),
  [models](https://developers.openai.com/api/docs/models),
  [embeddings](https://developers.openai.com/api/docs/guides/embeddings).
- Model names live only in `.env` (`OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`).
- Cross-language retrieval quality is measured in Phase 3. If it is too low,
  the fallbacks are `text-embedding-3-large` or translating the question first.

## In scope

- Answer guest questions using one hotel information document (written in English).
- Reply in the language of the question: Turkish, English or German.
- Show the source section and article under every answer, generated from
  chunk metadata in code (not written by the LLM).
- When the answer is not in the document, do not guess: return a fixed
  "I'm forwarding this to reception" message (similarity threshold + prompt rule).
- Prebuilt index file; the app never re-creates embeddings at startup.
- A 30-question evaluation set (`eval/questions.json`) and `eval/run_eval.py`.
- Cost guards: prepaid credit, per-session question limit, max question length,
  low `max_tokens`.
- Streamlit chat UI that works on a phone, with example question buttons.
- Public demo on Streamlit Community Cloud.
- README with real evaluation numbers and known limitations.

## Deliberately out of scope

- Reservations, availability or date lookups.
- Payments of any kind.
- User accounts or login.
- Real WhatsApp, Telegram or website widget integration.
- Real handoff to reception (the bot only shows a message; nothing is sent).
- File upload by users.
- Multiple hotels or an admin panel for editing content.
- Storing conversation history or any personal data.
- Languages other than Turkish, English and German.
- Voice input or output.
- LangChain, a vector database or Docker.

## Before uploading to GitHub

This project does not use git. Files are uploaded through the GitHub website,
and browser uploads ignore `.gitignore`. Upload happens in Phase 6 at the latest,
because Streamlit Community Cloud deploys from a GitHub repo.

1. Run `python make_upload_folder.py`. It copies only publishable files into a
   new `multilingual-hotel-support-bot-upload-<time>` folder next to the project
   and scans it for anything shaped like an OpenAI key.
2. If it prints `STOP`, fix the listed files and run it again.
3. Check the printed file list once by eye: no `.env`, `secrets.toml`, `.venv/`
   or `__pycache__/`.
4. Upload from that folder only, never from the working folder.

## Deploy (Phase 6)

1. Record a 20-30 second GIF (answer + source, German question, refusal) and save it as `docs/demo.gif`.
2. Run `python make_upload_folder.py` and upload that folder's contents to a new public GitHub repo.
3. On share.streamlit.io: New app -> that repo -> `app.py`. In Advanced settings choose Python 3.12
   and paste `.streamlit/secrets.example.toml` into Secrets with the real key.
4. Pick a custom subdomain (e.g. `any-hotel-assistant`).
5. Open the link in a private window and on a phone; repeat the Phase 5 button test.
6. Replace `DEMO_URL` in README.md with the live link and upload README.md again.

## Known limitations (copy to README in Phase 7)

- The per-session question limit resets when the page is refreshed. The real
  spending ceiling is the prepaid credit with auto-recharge turned off.
- The prepaid balance is not an instant cut-off: OpenAI may process a little
  usage after it reaches zero, which shows up as a negative balance.
- "Forwarding to reception" is only a message; nothing is sent anywhere.
- Questions are not stored by the app, and requests are sent with `store=False`.

## Parking lot

New ideas go here instead of into the code. Revisit only after the demo is live.

- (empty)
