# Assignment 3

Build a RAG system over a document collection, expose it as MCP tools, use it from a subagent and from your own agent, and measure how often it answers correctly, makes things up, or rightly says it does not know.

Start with [SETUP.md](SETUP.md).

## The task

1. **Use case.** Who asks, what they do with the answer, and what a wrong answer costs compared with "I don't know". Pick one where a wrong answer matters.
2. **Corpus.** At least **3 documents** and **50,000 words** in total. They should address your use case. Public, with a licence that allows this use, and no personal data.
3. **Ingest.** Extract, clean and chunk the text. Every chunk carries its source and location (page, section, article). Embed the chunks with Gemini and store them in `index/` (ChromaDB or similar).
4. **Test set**, written by you from the documents, in `eval/questions.csv`: 30+ questions, including 5+ the corpus cannot answer and 5+ that need two passages. Format in [eval/README.md](eval/README.md).
5. **Retrieval.** Hit rate at k = 1, 3, 5. Compare at least **two designs** (chunk size, structure-based chunking, metadata filter, hybrid search…) and choose by the numbers.
6. **MCP server** (`server.py`) with two tools:
   - **search**: takes a `query`, returns chunks with citations;
   - **add document**: takes a file `path`, runs it through your index pipeline (import it, don't rewrite it), embeds the chunks and stores them. Decide what happens if the document is already indexed.
7. **Agents.**
   - A **subagent** in `.claude/agents/` that uses only the search tool. Use it from Copilot or Claude Code and paste one transcript into REPORT.md.
   - **Your harness** (`agent.py`, LangChain): `AGENTS.md` as system prompt, approval before the add-document tool runs, and one more guardrail of your choice.
8. **Evaluate.** Run the agent on every question. Check in code what code can check (valid citations, refusals); grade the rest with an LLM judge and a rubric. Grade 20+ answers yourself in `eval/human_labels.csv` and report how often you agree with the judge. Then add a document that answers one of your unanswerable questions and show the answer before and after.
9. **Failures and decision.** Classify each failure: retrieval miss, wrong answer from the right passage, unsupported claim, wrong refusal, or bad test question. Would you let users rely on it? Cost per question in tokens and seconds.
10. **Write REPORT.md and submit.**

## Code layout

```bash
uv run python main.py index                 # build the index
uv run python main.py ask "your question"   # answer with citations
uv run python main.py evaluate              # retrieval and end-to-end evaluation
uv run server.py                            # MCP server (your editor starts it)
```

`index` and `evaluate` may be `.py` or `.ipynb` (keep one of each). `server.py` and `agent.py` must be scripts. Everything runs with no manual steps.

**Commit `index/`** (under 90 MB), saved vectors included. `main.py index` embeds only chunks without a saved vector, and the server never re-embeds the corpus. The free tier allows about **1,000 embeddings a day** (see SETUP.md), so save vectors as you go.

## Submission

Push to your fork, open a PR to https://github.com/ami232/daiist-assignment-3, submit the PR link on Blackboard, and upload a zip of the repo as backup.

## Grading

**Hard requirement:** the three `main.py` commands run end to end and `check.py` passes.

| Component | Weight |
|---|---|
| Use case and corpus | 15% |
| Ingestion, test set and retrieval | 25% |
| Tools, subagent and harness | 20% |
| End-to-end evaluation | 25% |
| Failure analysis and decision | 15% |
| **× Written comprehension check** | **0–100%** |

The comprehension check, written in class without your code, multiplies the subtotal.

## AI use

Allowed for code; disclose it in REPORT.md. The decisions (use case, chunking, design choice, refusals, judge rubric, ship decision), the test set, your own grades and the report's analysis must be yours.
