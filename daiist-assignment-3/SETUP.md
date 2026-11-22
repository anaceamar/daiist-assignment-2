# Setup

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), fork this repo, clone your fork and run `uv sync`. Open the repo folder itself in VS Code, so the agent finds `.vscode/mcp.json`, `.mcp.json`, `AGENTS.md` and `.claude/`.

## Gemini API key

Get a free key at <https://aistudio.google.com/apikey>, then:

```bash
cp .env.example .env        # paste your key into .env
uv run python api_key.py    # checks that it works
```

In your code, use `from api_key import load_api_key`. Never commit `.env`. A `GEMINI_API_KEY` environment variable also works.

## Free-tier limits

`gemini-embedding-2` allows about **1,000 texts a day** (each text in a batch counts), per Google Cloud project, reset at midnight Pacific time (09:00 in Spain). Check yours at <https://ai.dev/rate-limit>.

A 50,000-word corpus in 150-word chunks is about 420 texts. Each new chunking costs another full index; changing k, filters or search method does not. Every question your server or evaluation embeds also counts. Save each vector as soon as you get it, so a crash or the daily limit loses nothing.

## MCP server in your editor

`.vscode/mcp.json` and `.mcp.json` register `server.py` as `corpus`. Check its tools in Copilot (Agent mode, tools icon) or Claude Code (`/mcp`).

## Dependencies

`uv add <package>`, then commit `pyproject.toml` and `uv.lock`.

## Check your submission

```bash
uv run python check.py              # key, index, test set, both tools, one answer
uv run python check.py --evaluate   # also runs main.py evaluate
```

It does not rebuild the index. A failure here is a failure for the grader too. The same check runs on GitHub: on your fork, enable Actions, add the `GEMINI_API_KEY` secret, and run **Verify Assignment 3**.
