"""
Assignment 3: the harness (`main.py ask "question"` runs `python agent.py "question"`).

LangChain `create_agent` on Gemini, with the tools of server.py loaded over
MCP and AGENTS.md as the system prompt. Guardrails:

1. Approval: add_document is gated by HumanInTheLoopMiddleware; nothing is
   added unless a person approves (interactive y/N; non-interactive runs reject).
2. Call limits: at most 8 model calls and 4 searches per run.
3. Citation guardrail (our extra one): after the agent answers, code checks
   that the answer cites at least one passage and that every citation names a
   passage the search tool actually returned in this run. If not, the agent
   gets one chance to fix it; if it still fails, the answer is replaced by a
   refusal. An answer with an invented citation never reaches the user.

evaluate.py uses `Harness` to keep one MCP server running across questions.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from api_key import load_api_key
from retrieval import citation_matches, parse_citations, retrieved_from_tool_output
from settings import GEN_MODEL, chat_model

ROOT = Path(__file__).resolve().parent
REFUSAL = "The documents I have do not cover this"
REFUSAL_STEM = "the documents i have do not cover"
MAX_MODEL_CALLS, MAX_SEARCHES = 8, 4


def system_prompt() -> str:
    return (ROOT / "AGENTS.md").read_text(encoding="utf-8")


def is_refusal(answer: str) -> bool:
    """Does the answer open with the refusal phrase? Models often continue it
    ('...do not cover the specific fees...'), so match its stem, not the exact words."""
    return REFUSAL_STEM in answer.lower()[:300]


def check_citations(answer: str, retrieved: list[tuple[str, str]]) -> list[str]:
    """Problems with the answer's citations; an empty list means it passes."""
    if is_refusal(answer) and not parse_citations(answer):
        return []
    citations = parse_citations(answer)
    if not citations:
        return ["the answer has no citations"]
    invalid = [c for c in citations if not any(citation_matches(c, r) for r in retrieved)]
    return [f"[{s}, {l}] was not returned by the search tool in this conversation" for s, l in invalid]


def message_text(message) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):          # a str subclass in recent langchain-core
        return str(text)
    content = message.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def default_approve(request: dict) -> bool:
    """Ask the person at the keyboard. Without a terminal (evaluation, CI) the answer is no."""
    print(f"\nThe agent wants to run {request['name']} with {request['args']}.", file=sys.stderr)
    if not sys.stdin.isatty():
        print("No terminal to ask for approval: rejected.", file=sys.stderr)
        return False
    return input("Approve? [y/N] ").strip().lower() in ("y", "yes", "s", "si", "sí")


class Harness:
    """An agent connected to server.py over MCP. Use as `async with Harness() as h: await h.ask(q)`."""

    def __init__(self, include_added: bool = True, approve=default_approve, model: str = GEN_MODEL):
        self.include_added, self.approve, self.model_name = include_added, approve, model

    async def __aenter__(self) -> "Harness":
        load_api_key()
        env = dict(os.environ)
        if not self.include_added:
            env["SEARCH_EXCLUDE_ADDED"] = "1"
        params = StdioServerParameters(command=sys.executable, args=["server.py"], cwd=str(ROOT), env=env)
        self._stdio = stdio_client(params)
        read, write = await self._stdio.__aenter__()
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()
        self.tools = [self._wrap(t) for t in (await self.session.list_tools()).tools]

        from langchain_google_genai import ChatGoogleGenerativeAI
        self.model_name = chat_model(self.model_name)
        model = ChatGoogleGenerativeAI(model=self.model_name, temperature=0, max_retries=3, timeout=120)
        self.agent = create_agent(
            model,
            tools=self.tools,
            system_prompt=system_prompt(),
            middleware=[
                HumanInTheLoopMiddleware(interrupt_on={"add_document": {"allowed_decisions": ["approve", "reject"]}}),
                ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
                ToolCallLimitMiddleware(tool_name="search", run_limit=MAX_SEARCHES),
            ],
            checkpointer=InMemorySaver(),
        )
        return self

    async def __aexit__(self, *exc) -> None:
        # Close cleanly even when the agent raised: handing the exception to the
        # MCP context managers makes their task groups hang on shutdown.
        await self._session_cm.__aexit__(None, None, None)
        await self._stdio.__aexit__(None, None, None)

    def _wrap(self, tool) -> StructuredTool:
        session = self.session

        async def call(**kwargs) -> str:
            result = await session.call_tool(tool.name, kwargs)
            text = "\n".join(getattr(c, "text", "") for c in result.content).strip()
            if getattr(result, "is_error", False) or getattr(result, "isError", False):
                return f"Tool error: {text}"
            return text

        return StructuredTool.from_function(
            coroutine=call, name=tool.name, description=tool.description or "", args_schema=tool.input_schema,
        )

    async def _run(self, payload, config) -> dict:
        state = await self.agent.ainvoke(payload, config)
        while state.get("__interrupt__"):
            request = state["__interrupt__"][0].value
            decisions = []
            for action in request["action_requests"]:
                if self.approve(action):
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({"type": "reject", "message": "The user did not approve adding this document."})
            state = await self.agent.ainvoke(Command(resume={"decisions": decisions}), config)
        return state

    async def ask(self, question: str) -> dict:
        """Answer one question. Returns the answer and a trace for evaluation."""
        started = time.time()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = await self._run({"messages": [HumanMessage(question)]}, config)
        answer, trace = self._read(state["messages"])

        # The citation check is for answers about the law; a turn that only ran
        # add_document (and never searched) reports on the tool instead.
        searched = any(c["name"] == "search" for c in trace["tool_calls"])
        problems = check_citations(answer, trace["retrieved"]) if searched or parse_citations(answer) else []
        guardrail = None
        if problems:
            guardrail = {"first_answer": answer, "problems": problems}
            fix = (
                "Your answer failed the citation check: " + "; ".join(problems) + ". "
                "Rewrite it using only passages the search tool returned, citing each claim exactly as "
                "'cite as [source, location]' shows. If the passages do not answer the question, start with "
                f"'{REFUSAL}.'"
            )
            state = await self._run({"messages": [HumanMessage(fix)]}, config)
            answer, trace = self._read(state["messages"])
            problems = check_citations(answer, trace["retrieved"])
            guardrail["fixed"] = not problems
            if problems:
                answer = (f"{REFUSAL} with citations I can verify, so I will not answer from it. "
                          "Please check the regulation directly or escalate the case.")
        trace.update(answer=answer, guardrail=guardrail, model=self.model_name, seconds=round(time.time() - started, 2),
                     citations=parse_citations(answer), refusal=is_refusal(answer))
        return trace

    @staticmethod
    def _read(messages) -> tuple[str, dict]:
        tool_calls, retrieved, contexts = [], [], []
        tokens_in = tokens_out = model_calls = 0
        for m in messages:
            if isinstance(m, AIMessage):
                model_calls += 1
                usage = m.usage_metadata or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
                tool_calls += [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
            elif isinstance(m, ToolMessage) and m.name == "search":
                text = message_text(m)
                contexts.append(text)
                retrieved += retrieved_from_tool_output(text)
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
        answer = message_text(final).strip() if final else ""
        if not answer:
            answer = f"{REFUSAL}: the agent stopped before answering (call limit reached)."
        return answer, {
            "tool_calls": tool_calls, "retrieved": list(dict.fromkeys(retrieved)), "contexts": contexts,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "model_calls": model_calls,
        }


async def ask_async(question: str, **kwargs) -> dict:
    async with Harness(**kwargs) as harness:
        return await harness.ask(question)


def ask(question: str, **kwargs) -> dict:
    """Synchronous helper: answer one question and return the trace."""
    return asyncio.run(ask_async(question, **kwargs))


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question"', file=sys.stderr)
        return 2
    trace = ask(" ".join(sys.argv[1:]))
    print(trace["answer"])
    if trace["citations"]:
        print("\nSources:")
        for source, location in dict.fromkeys(trace["citations"]):
            print(f"  - {source}, {location}")
    if trace["guardrail"]:
        print(f"\n[citation guardrail fired: {'; '.join(trace['guardrail']['problems'])}]", file=sys.stderr)
    print(f"\n[{trace['model_calls']} model calls, {len(trace['tool_calls'])} tool calls, "
          f"{trace['tokens_in']} in / {trace['tokens_out']} out tokens, {trace['seconds']}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
