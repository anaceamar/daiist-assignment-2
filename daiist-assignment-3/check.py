"""
Assignment 3: submission check. `uv run python check.py [--evaluate]`

Checks the API key, the committed index and test set, that server.py lists a
search tool (`query`) and an add-document tool (`path`) and that search answers
your first test question, and that `main.py ask` answers it. It never rebuilds
the index or calls add-document. Don't edit this file to make it pass.
"""

import argparse
import asyncio
import csv
import subprocess
import sys
import time
from pathlib import Path

from api_key import masked, resolve_api_key, verify

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
QUESTIONS = ROOT / "eval" / "questions.csv"
HUMAN_LABELS = ROOT / "eval" / "human_labels.csv"
MAX_INDEX_MB = 90
QUESTION_COLUMNS = ["id", "question", "kind", "expected_source", "expected_location", "reference_answer"]
MIN_QUESTIONS, MIN_UNANSWERABLE, MIN_MULTI, MIN_HUMAN_LABELS = 30, 5, 5, 20
SERVER_TIMEOUT = 120
ASK_TIMEOUT = 300

TOTAL_STEPS = 4


def fail(message: str) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr)
    sys.exit(1)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def step(n: int, title: str) -> None:
    print(f"\n=== {n}/{TOTAL_STEPS}  {title} ".ljust(72, "=") + "\n", flush=True)


def check_key() -> None:
    key, source = resolve_api_key()
    if not key:
        fail(
            "No Gemini API key found.\n"
            "  Locally  : copy .env.example to .env and paste your key in.\n"
            "  On GitHub: add it under Settings > Secrets and variables > Actions\n"
            "             on YOUR fork, named GEMINI_API_KEY.\n"
            "  Then run `uv run python api_key.py` on its own to check it."
        )
    print(f"Found a key in {source}: {masked(key)}")
    works, detail = verify(key)
    if not works:
        fail(f"That key did not work: {detail}")
    print(f"Key works: {detail}.")


def check_index() -> None:
    files = [p for p in INDEX_DIR.rglob("*") if p.is_file() and p.name != ".gitkeep"] if INDEX_DIR.is_dir() else []
    if not files:
        fail(
            "index/ is empty.\n"
            "  Run `uv run python main.py index` and commit what it writes into index/.\n"
            "  Your server loads it; nobody should have to re-embed your corpus."
        )
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    if size_mb > MAX_INDEX_MB:
        fail(f"index/ is {size_mb:.0f} MB; the limit is {MAX_INDEX_MB} MB. Use fewer or smaller chunks.")
    print(f"index/ holds {len(files)} file(s), {size_mb:.1f} MB.")


def read_questions() -> list[dict]:
    if not QUESTIONS.is_file():
        fail("eval/questions.csv is missing.")
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in QUESTION_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            fail(f"eval/questions.csv is missing the column(s): {', '.join(missing)}. See eval/README.md.")
        rows = [r for r in reader if (r.get("question") or "").strip()]

    kinds = [r["kind"].strip().lower() for r in rows]
    unanswerable = [r for r in rows if r["kind"].strip().lower() == "unanswerable"]
    problems = []
    if len(rows) < MIN_QUESTIONS:
        problems.append(f"{len(rows)} questions; at least {MIN_QUESTIONS} are required")
    if len(unanswerable) < MIN_UNANSWERABLE:
        problems.append(f"{len(unanswerable)} `unanswerable` questions; at least {MIN_UNANSWERABLE} are required")
    if kinds.count("multi") < MIN_MULTI:
        problems.append(f"{kinds.count('multi')} `multi` questions; at least {MIN_MULTI} are required")
    if any(r["expected_source"].strip() for r in unanswerable):
        problems.append("an `unanswerable` question has an expected_source; leave it empty")
    if any(not r["expected_source"].strip() for r in rows if r not in unanswerable):
        problems.append("an answerable question has no expected_source")
    if len({r["id"] for r in rows}) != len(rows):
        problems.append("question ids are not unique")
    if problems:
        fail("eval/questions.csv: " + "; ".join(problems) + ". See eval/README.md.")
    counts = ", ".join(f"{kinds.count(k)} {k}" for k in sorted(set(kinds)))
    print(f"eval/questions.csv has {len(rows)} questions: {counts}.")

    graded = 0
    if HUMAN_LABELS.is_file():
        with HUMAN_LABELS.open(newline="", encoding="utf-8") as f:
            graded = sum(1 for r in csv.DictReader(f) if (r.get("id") or "").strip())
    if graded < MIN_HUMAN_LABELS:
        warn(f"eval/human_labels.csv has {graded} graded answers; the brief asks for at least {MIN_HUMAN_LABELS}.")
    return rows


def tool_with_argument(tools: list, argument: str):
    """The first tool with a required string argument of this name, or None."""
    for tool in tools:
        schema = tool.input_schema or {}
        if argument in schema.get("required", []) and schema.get("properties", {}).get(argument, {}).get("type") == "string":
            return tool
    return None


async def call_search_tool(question: str) -> tuple[str | None, str]:
    """Start server.py over stdio, as an editor would, find the search tool and call it once.

    Returns (tool name, its result), or (None, what went wrong).
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["server.py"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = ", ".join(t.name for t in tools) or "none"
            search, adder = tool_with_argument(tools, "query"), tool_with_argument(tools, "path")
            if search is None:
                return None, (
                    f"server.py lists these tools: {names}. None has a required string argument "
                    f"named `query`. Your search tool needs one."
                )
            if adder is None:
                return None, (
                    f"server.py lists these tools: {names}. None has a required string argument "
                    f"named `path`. Your add-document tool needs one."
                )
            for tool in (search, adder):
                description = (tool.description or "").strip()
                if not description or description.startswith("TODO"):
                    return None, f"Tool `{tool.name}` has no real description. Its docstring is what the model reads."
            tool = search
            result = await session.call_tool(tool.name, {"query": question})
            text = "\n".join(getattr(c, "text", "") for c in result.content).strip()
            if getattr(result, "is_error", False) or getattr(result, "isError", False):
                return None, f"Tool `{tool.name}` returned an error for {question!r}:\n  {text[:800]}"
            if not text:
                return None, f"Tool `{tool.name}` returned nothing for {question!r}."
            return tool.name, text


def check_server(question: str) -> None:
    try:
        name, text = asyncio.run(asyncio.wait_for(call_search_tool(question), SERVER_TIMEOUT))
    except TimeoutError:
        fail(
            f"server.py did not answer within {SERVER_TIMEOUT}s.\n"
            "  Is it re-embedding the corpus at startup? It must load the committed index.\n"
            "  Is something printing to stdout? Only MCP messages may go there; log to stderr."
        )
    except Exception as error:  # noqa: BLE001 - any failure to start or talk is the finding
        fail(
            f"Could not start server.py or talk to it over MCP: {type(error).__name__}: {error}\n"
            "  Run `uv run server.py` on its own to see its error."
        )
    if name is None:
        fail(text)
    preview = text[:400] + ("..." if len(text) > 400 else "")
    print(f"Tool `{name}` answered {question!r}:\n\n{preview}")


def check_ask(question: str) -> None:
    started = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "ask", question],
            cwd=ROOT, capture_output=True, text=True, timeout=ASK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        fail(f"`main.py ask` did not finish within {ASK_TIMEOUT}s.")
    if result.returncode != 0:
        fail(f"`main.py ask` exited with code {result.returncode}.\n  --- output ---\n{result.stderr[-2000:]}")
    if not result.stdout.strip():
        fail("`main.py ask` printed nothing. It must print the answer and its citations.")
    print(f"{result.stdout.strip()[-1500:]}\n\n(answered in {time.time() - started:.1f}s)")


def main() -> None:
    global TOTAL_STEPS
    parser = argparse.ArgumentParser(description="Check this submission.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="also run main.py evaluate at the end (spends API quota and takes a while)",
    )
    args = parser.parse_args()
    TOTAL_STEPS = 5 if args.evaluate else 4

    step(1, "Checking your Gemini API key")
    check_key()

    step(2, "Checking your committed index and test set")
    check_index()
    rows = read_questions()
    question = next(r["question"] for r in rows if r["kind"].strip().lower() != "unanswerable")

    step(3, "Starting server.py, checking its tools and calling its search tool")
    check_server(question)

    step(4, "Asking your agent (main.py ask)")
    check_ask(question)

    if args.evaluate:
        step(5, "Running your evaluation stage (main.py evaluate)")
        started = time.time()
        result = subprocess.run([sys.executable, "main.py", "evaluate"], cwd=ROOT)
        if result.returncode != 0:
            fail(f"`main.py evaluate` exited with code {result.returncode}.")
        print(f"\nEvaluation stage completed in {time.time() - started:.1f}s.")

    print("\n" + "=" * 72)
    print("All checks passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
