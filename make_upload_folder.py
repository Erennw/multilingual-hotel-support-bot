"""Copy only the files that belong on GitHub into a clean folder, then scan it.

    python make_upload_folder.py

Browser uploads to GitHub ignore .gitignore, so this script decides what
is published instead. It works from an allow-list: a file is copied only
if its type is listed below and it is not in a blocked folder. Anything
new and unexpected is left out, which is the safe direction to fail in.

The output folder is created next to the project, with a timestamp, so
an old upload folder is never mixed with a new one. Upload from there.

The scan looks for anything shaped like an OpenAI key, and for the
actual key from .env. The key itself is never printed.
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

PROJECT = Path(__file__).resolve().parent

ALLOWED_SUFFIXES = {".py", ".md", ".json", ".npy", ".txt", ".toml", ".png", ".gif"}
ALLOWED_NAMES = {".gitignore", ".env.example", "LICENSE"}

BLOCKED_DIRS = {".venv", "venv", "__pycache__", ".git", ".vscode", ".idea", ".claude"}
BLOCKED_FILES = {".env", "secrets.toml"}

# "sk-" followed by 20+ key characters. The placeholder in .env.example
# ("sk-your-demo-key-here") is shorter, so it does not match.
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


def is_publishable(path: Path) -> bool:
    relative = path.relative_to(PROJECT)
    if any(part in BLOCKED_DIRS for part in relative.parts):
        return False
    if path.name in BLOCKED_FILES or path.name.startswith(".env") and path.name != ".env.example":
        return False
    return path.name in ALLOWED_NAMES or path.suffix in ALLOWED_SUFFIXES


def scan(folder: Path, real_key: str | None) -> list[str]:
    """Return the relative paths of files that look like they hold a key."""
    hits = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix in {".npy", ".png", ".gif"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if KEY_PATTERN.search(text) or (real_key and real_key in text):
            hits.append(str(path.relative_to(folder)))
    return hits


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = PROJECT.parent / f"{PROJECT.name}-upload-{stamp}"

    files = sorted(p for p in PROJECT.rglob("*") if p.is_file() and is_publishable(p))
    for source in files:
        destination = target / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    real_key = dotenv_values(PROJECT / ".env").get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    leaks = scan(target, real_key)

    print(f"Copied {len(files)} files to {target}")
    for source in files:
        print(f"  {source.relative_to(PROJECT)}")

    if leaks:
        print("\nSTOP: these files look like they contain an API key. Do not upload.")
        for name in leaks:
            print(f"  {name}")
        sys.exit(1)

    print("\nNo key found. Upload the contents of that folder, not the project folder.")


if __name__ == "__main__":
    main()
