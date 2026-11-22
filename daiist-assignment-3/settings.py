"""
Settings shared by index.py, agent.py and evaluate.py.

Each setting can be overridden by an environment variable or a line in .env
(e.g. `GEN_MODEL=gemini-2.5-flash`), so a model rename by Google never needs a
code change.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"


def env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, val = line.strip().partition("=")
            if sep and key.strip() == name and val.strip():
                return val.strip().strip('"').strip("'")
    return default


@lru_cache(maxsize=None)
def chat_model(preferred: str) -> str:
    """`preferred` if the API key can use it, else the latest Flash alias."""
    if os.environ.get("FAKE_EMBEDDINGS"):
        return preferred
    try:
        from google import genai
        from api_key import load_api_key
        client = genai.Client(api_key=load_api_key())   # keep a reference: the pager needs an open client
        names = {m.name.removeprefix("models/") for m in client.models.list()}
    except Exception as error:  # noqa: BLE001 - offline or no permission to list: just try the preferred one
        print(f"Could not list Gemini models ({type(error).__name__}); using {preferred}.", file=sys.stderr)
        return preferred
    if preferred in names:
        return preferred
    fallback = next((n for n in ("gemini-flash-latest", "gemini-2.5-flash") if n in names), preferred)
    print(f"Model {preferred} is not available to this key; using {fallback}.", file=sys.stderr)
    return fallback


GEN_MODEL = env("GEN_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = env("JUDGE_MODEL", GEN_MODEL)
EMBED_MODEL = env("EMBED_MODEL", "gemini-embedding-2")
