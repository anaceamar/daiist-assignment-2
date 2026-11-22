"""
Assignment 3: find and verify your Gemini API key. `uv run python api_key.py`

Reads GEMINI_API_KEY from the environment, else from .env. In your code:

    from api_key import load_api_key
    client = genai.Client(api_key=load_api_key())
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
VAR = "GEMINI_API_KEY"
VERIFY_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"


def read_env_file(path: Path = ENV_FILE) -> str | None:
    """Pull GEMINI_API_KEY out of a .env file, if there is one.

    Deliberately a tiny parser rather than a dependency: KEY=value per line,
    blank lines and # comments ignored, surrounding quotes stripped.
    """
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() == VAR:
            value = value.strip().strip('"').strip("'")
            return value if value and value != "paste-your-key-here" else None
    return None


def resolve_api_key() -> tuple[str | None, str]:
    """Return (key, where it came from), without changing anything."""
    from_env = (os.environ.get(VAR) or "").strip()
    if from_env:
        return from_env, f"the {VAR} environment variable"
    from_file = read_env_file()
    if from_file:
        return from_file, ENV_FILE.name
    return None, "nowhere"


def load_api_key(required: bool = True) -> str | None:
    """Return the API key from the environment, else from .env.

    Also puts it into os.environ so libraries that look it up themselves
    (google-genai and langchain-google-genai do) find it too.
    """
    key, _ = resolve_api_key()
    if key:
        os.environ[VAR] = key
        return key
    if required:
        raise RuntimeError(
            f"No {VAR} found.\n"
            f"  Copy .env.example to .env and paste your key in, or set the\n"
            f"  {VAR} environment variable. Get a key from\n"
            f"  https://aistudio.google.com/apikey"
        )
    return None


def masked(key: str) -> str:
    return f"{key[:2]}{'*' * max(len(key) - 4, 4)}{key[-2:]}" if len(key) > 4 else "****"


def verify(key: str, timeout: int = 20) -> tuple[bool, str]:
    """Ask Google whether this key actually works, by listing one model.

    The key goes in a header, not in the URL, so it never ends up in a log.
    """
    request = urllib.request.Request(VERIFY_URL, headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))["error"]["message"]
        except Exception:
            detail = f"HTTP {error.code}"
        return False, detail
    except urllib.error.URLError as error:
        return False, f"could not reach generativelanguage.googleapis.com ({error.reason}). Are you online?"
    except Exception as error:  # noqa: BLE001 - surface anything else plainly
        return False, f"{type(error).__name__}: {error}"
    return True, "key accepted"


def main() -> int:
    key, source = resolve_api_key()
    if not key:
        print(
            f"No API key found.\n\n"
            f"  Copy .env.example to .env and paste your key in:\n\n"
            f"      cp .env.example .env\n\n"
            f"  Get a key from https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return 1

    print(f"Found a key in {source}: {masked(key)}")
    print("Checking it against the Gemini API ...")
    ok, detail = verify(key)
    if not ok:
        print(f"\nThat key did not work: {detail}", file=sys.stderr)
        return 1
    print(f"Key works: {detail}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
