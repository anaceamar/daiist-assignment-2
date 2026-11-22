"""Quick check: which Gemini chat models answer with this key (30 s limit each)."""
import time
from google import genai
from google.genai import types
from api_key import load_api_key

client = genai.Client(api_key=load_api_key(), http_options=types.HttpOptions(timeout=30000))
for model in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview",
              "gemini-omni-1.2-flash"]:
    start = time.time()
    try:
        text = client.models.generate_content(model=model, contents="Say hi in two words.").text
        print(f"OK    {model:28s} {time.time() - start:5.1f}s  {text.strip()[:40]!r}", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"FAIL  {model:28s} {time.time() - start:5.1f}s  {type(error).__name__}: {str(error)[:110]}", flush=True)
