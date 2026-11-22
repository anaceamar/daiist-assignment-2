"""
Assignment 3: MCP server. Your editor and agent.py start it; by hand: `uv run server.py`.

- search(query): returns chunks with source and location.
- add_document(path): runs the file through index.py's pipeline, embeds it
  and stores it.

Loads index/ at startup and never re-embeds the corpus: only the query is
embedded (and cached). Logs go to stderr; stdout carries the protocol.

Environment:
    SEARCH_EXCLUDE_ADDED=1   hide documents added with add_document (used by
                             evaluate.py for the "before" answer)
"""

import os

from mcp.server.mcpserver import MCPServer

import index as pipeline
import retrieval

mcp = MCPServer("corpus", log_level="WARNING")
INCLUDE_ADDED = os.environ.get("SEARCH_EXCLUDE_ADDED") != "1"
TOP_K = 5


@mcp.tool()
def search(query: str) -> str:
    """Search EU passenger rights law: Regulation 261/2004 on air passenger
    rights (denied boarding, cancellation, long delay) with the Commission's
    2024 interpretative guidelines, Regulation 2021/782 on rail passengers,
    Regulation 181/2011 on bus and coach passengers and Regulation 1177/2010 on
    sea and inland waterway passengers, plus any document added later.

    Use it for every question about compensation, refunds, re-routing, care
    (meals, hotels), delays, cancellations, disabled passengers or complaints
    in EU transport. Write the query in English legal terms and name the
    transport mode (e.g. "train delay compensation 60 minutes"); search again
    with other words if the results miss the point.

    Returns the 5 most relevant passages, each headed by
    "cite as [source, location]" (e.g. [air-261-2004, Article 7]). Cite answers
    exactly that way. If no passage covers the question, the corpus does not
    cover it.
    """
    results = retrieval.search(query, k=TOP_K, include_added=INCLUDE_ADDED)
    return retrieval.format_results(results)


@mcp.tool()
def add_document(path: str) -> str:
    """Add a document to the passenger-rights index so search can find it.

    `path` is a local file (EU legal act as HTML/XHTML from EUR-Lex, PDF, TXT
    or Markdown). The file goes through the same pipeline as the corpus:
    extraction, legal-unit chunking with source and location, Gemini
    embeddings, storage in index/.

    If the identical file is already indexed nothing changes; if another
    version of a file with the same name is indexed, its chunks are replaced.
    Returns what happened and how many chunks were stored. Only call it when
    the user asks to add a document; it needs their approval.
    """
    try:
        summary = pipeline.add_file(path)
    except pipeline.QuotaExhausted:
        return "Not added: the daily Gemini embedding quota is spent. Try again after it resets."
    except (FileNotFoundError, ValueError) as error:
        return f"Not added: {error}"
    retrieval.reset_cache()
    if summary["status"] == "already_indexed":
        return f"'{summary['source']}' is already indexed with identical content; nothing changed."
    verb = "Replaced the previous version of" if summary["status"] == "replaced" else "Added"
    counts = ", ".join(f"{n} {design} chunks" for design, n in summary["chunks"].items())
    return f"{verb} '{summary['source']}': {counts}. It is now searchable under the source id '{summary['source']}'."


if __name__ == "__main__":
    try:  # load the saved index now, so the first search is fast
        retrieval.load_index("fixed" if retrieval.chosen_design().startswith("fixed") else "structure")
    except Exception as error:  # noqa: BLE001 - the server should still start and say why searches fail
        pipeline.log(f"Index not loaded: {error}")
    mcp.run()  # stdio
