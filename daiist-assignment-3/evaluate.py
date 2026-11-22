"""
Assignment 3: evaluation (`main.py evaluate`).

1. Retrieval: hit rate at k = 1, 3, 5 for four search designs (two chunkings,
   plus hybrid BM25 and a transport-mode filter on the structure chunks).
2. End to end: the agent (agent.py) answers every question in eval/questions.csv
   with added documents hidden, so the corpus is the one the questions were
   written for. Answers are cached in eval/answers.jsonl and only re-run when
   the question, AGENTS.md, the model or the search design change.
3. Code checks: refusal, citations present, citations valid (returned by the
   tool), expected passage retrieved, expected passage cited.
4. LLM judge with a rubric and the retrieved text as evidence (eval/judge.jsonl).
5. Agreement with eval/human_labels.csv (written before reading the judge).
6. Failure causes (suggested by code; override in eval/failures.csv, column cause_final).
7. Before/after: an unanswerable question, then add_document, then the same question.
8. Cost: tokens and seconds per question.

Outputs in eval/: retrieval.csv, results.csv, failures.csv, to_grade.csv,
before_after.md, hit_rate.png, summary.json, summary.md.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import index as pipeline
import retrieval
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
QUESTIONS = EVAL / "questions.csv"
ANSWERS = EVAL / "answers.jsonl"
JUDGE = EVAL / "judge.jsonl"
HUMAN = EVAL / "human_labels.csv"
TO_GRADE = EVAL / "to_grade.csv"
FAILURES = EVAL / "failures.csv"

KS = (1, 3, 5)
DEMO_ID = "q36"                                    # unanswerable until Regulation 1107/2006 is added
DEMO_FILE = ROOT / "extra" / "air-prm-1107-2006.html"
from settings import JUDGE_MODEL, chat_model  # noqa: E402
PAUSE = float(os.environ.get("EVAL_PAUSE", "4"))   # seconds between questions (free-tier rate limits)
# List prices used only to put a number on cost (USD per million tokens); check current pricing.
PRICE_IN, PRICE_OUT = float(os.environ.get("PRICE_IN", "0.30")), float(os.environ.get("PRICE_OUT", "2.50"))
RUBRIC_VERSION = "v1"

RUBRIC = """You grade answers from an assistant that answers questions about EU passenger rights
for travel-agency staff, using only passages retrieved from EU regulations.

You get: the QUESTION, its KIND (single, multi or unanswerable), a REFERENCE answer written
by a person from the documents, the PASSAGES the assistant retrieved, and the ANSWER.

Give two binary grades.

correct = 1 when the answer is right for the question:
- For single and multi questions: it states the key facts of the reference (amounts,
  percentages, thresholds, deadlines, the yes/no) for the right transport mode and does not
  contradict the reference. Extra details are fine if they are true according to the passages.
  It is 0 if a key fact is missing or wrong, if a condition that changes the outcome is missing,
  if it uses a rule from the wrong transport mode, or if it refuses although the passages answer it.
  For multi questions, both parts must be right.
- For unanswerable questions: 1 only if the answer clearly says the documents do not cover the
  question and does not present an answer as fact. Mentioning related rules (for example the
  rule for another transport mode) is fine only if it says plainly that they do not answer this question.

supported = 1 when every factual claim in the answer is stated in, or follows directly from,
the PASSAGES. A claim that is true but appears only in the reference or in general knowledge
makes it 0. A refusal that makes no factual claims is supported (1).

Be strict: a confident wrong number is worse than "I don't know".
Reply in JSON: {"correct": 0 or 1, "supported": 0 or 1, "reason": "one sentence"}"""


# --------------------------------------------------------------------------- helpers

def read_questions() -> list[dict]:
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["question"].strip()]
    for r in rows:
        sources = [s.strip() for s in r["expected_source"].split(";") if s.strip()]
        locations = [l.strip() for l in r["expected_location"].split(";") if l.strip()]
        if len(sources) == 1 and len(locations) > 1:
            sources = sources * len(locations)
        r["expected"] = list(zip(sources, locations))
        r["kind"] = r["kind"].strip().lower()
    return rows


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def is_hit(pair: tuple[str, str], expected: list[tuple[str, str]]) -> bool:
    """Is a retrieved/cited (source, location) one of the expected passages?"""
    return any(retrieval.citation_matches(pair, e) or retrieval.citation_matches(e, pair) for e in expected)


# --------------------------------------------------------------------------- 1. retrieval

def evaluate_retrieval(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, summary = [], []
    answerable = [q for q in questions if q["kind"] != "unanswerable"]
    for design in retrieval.SEARCH_DESIGNS:
        per_k = {k: 0 for k in KS}
        all_found, mrr, words = 0, 0.0, {k: [] for k in KS}
        multi = [q for q in answerable if q["kind"] == "multi"]
        for q in answerable:
            results = retrieval.search(q["question"], k=max(KS), design=design, include_added=False)
            ranks = [r["rank"] for r in results if any(retrieval.location_hit(r, s, l) for s, l in q["expected"])]
            first = min(ranks) if ranks else None
            for k in KS:
                per_k[k] += bool(first and first <= k)
                words[k].append(sum(len(r["text"].split()) for r in results[:k]))
            mrr += 1 / first if first else 0
            found = all(any(retrieval.location_hit(r, s, l) for r in results) for s, l in q["expected"])
            if q["kind"] == "multi":
                all_found += found
            rows.append({
                "design": design, "id": q["id"], "kind": q["kind"], "first_hit_rank": first or "",
                "all_expected_in_top5": int(found),
                "top5": " | ".join(f"{r['source']}:{r['location']}" for r in results),
            })
        n = len(answerable)
        summary.append({
            "design": design, **{f"hit@{k}": round(per_k[k] / n, 3) for k in KS},
            "mrr@5": round(mrr / n, 3),
            "multi_both_in_top5": round(all_found / len(multi), 3) if multi else "",
            **{f"words@{k}": round(sum(words[k]) / n) for k in KS},
        })
        print(f"  {design:24s} " + "  ".join(f"hit@{k} {pct(per_k[k] / n)}" for k in KS) + f"  MRR {mrr / n:.2f}")
    write_csv(EVAL / "retrieval.csv", rows, list(rows[0]))
    write_csv(EVAL / "retrieval_summary.csv", summary, list(summary[0]))
    return rows, summary


def plot_hit_rates(summary: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(summary)
    for i, row in enumerate(summary):
        xs = [j + i * width for j in range(len(KS))]
        bars = ax.bar(xs, [row[f"hit@{k}"] for k in KS], width, label=row["design"])
        ax.bar_label(bars, labels=[pct(row[f"hit@{k}"]) for k in KS], fontsize=7, padding=2)
    ax.set_xticks([j + width * (len(summary) - 1) / 2 for j in range(len(KS))], [f"k = {k}" for k in KS])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("hit rate")
    ax.set_title("Retrieval hit rate by design (answerable questions)")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(EVAL / "hit_rate.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 2. agent answers

def run_config() -> str:
    from agent import GEN_MODEL
    return sha((ROOT / "AGENTS.md").read_text(encoding="utf-8") + GEN_MODEL + retrieval.chosen_design())


def run_agent(questions: list[dict]) -> dict[str, dict]:
    """Answer every question, reusing cached answers made with the same setup."""
    from agent import Harness
    config = run_config()
    cached = {r["id"]: r for r in read_jsonl(ANSWERS)
              if r.get("config") == config and r.get("question_hash")}
    todo = [q for q in questions
            if q["id"] not in cached or cached[q["id"]]["question_hash"] != sha(q["question"])
            or cached[q["id"]].get("error")]
    print(f"  {len(questions) - len(todo)} cached answers, {len(todo)} to run")

    async def go():
        async with Harness(include_added=False, approve=lambda request: False) as harness:
            for i, q in enumerate(todo, 1):
                try:
                    trace = await harness.ask(q["question"])
                    trace.update(id=q["id"], config=config, question_hash=sha(q["question"]))
                except Exception as error:  # noqa: BLE001 - record and continue; re-run retries it
                    trace = {"id": q["id"], "config": config, "question_hash": sha(q["question"]),
                             "error": f"{type(error).__name__}: {str(error)[:300]}"}
                    if "RESOURCE_EXHAUSTED" in str(error) and "day" in str(error).lower():
                        append_jsonl(ANSWERS, trace)
                        print("  Daily quota spent; run `main.py evaluate` again later to finish.")
                        return
                append_jsonl(ANSWERS, trace)
                cached[q["id"]] = trace
                status = "error" if trace.get("error") else ("refused" if trace["refusal"] else "answered")
                print(f"  [{i}/{len(todo)}] {q['id']} {status}"
                      + (" (citation guardrail fired)" if trace.get("guardrail") else ""))
                if i < len(todo):
                    await asyncio.sleep(PAUSE)

    if todo:
        asyncio.run(go())
    return {k: v for k, v in cached.items() if not v.get("error")}


# --------------------------------------------------------------------------- 3. code checks

def code_checks(q: dict, a: dict) -> dict:
    citations = [tuple(c) for c in a["citations"]]
    retrieved = [tuple(r) for r in a["retrieved"]]
    valid = all(any(retrieval.citation_matches(c, r) for r in retrieved) for c in citations)
    checks = {
        "refused": int(a["refusal"]),
        "has_citation": int(bool(citations)),
        "citations_valid": int(valid),
        "guardrail_fired": int(bool(a.get("guardrail"))),
        "searches": sum(1 for c in a["tool_calls"] if c["name"] == "search"),
    }
    if q["kind"] == "unanswerable":
        checks["refusal_ok"] = int(a["refusal"])
        checks["expected_retrieved"] = checks["expected_cited"] = ""
    else:
        checks["refusal_ok"] = int(not a["refusal"])
        checks["expected_retrieved"] = int(all(any(is_hit(r, [e]) for r in retrieved) for e in q["expected"]))
        checks["expected_cited"] = int(any(is_hit(c, q["expected"]) for c in citations))
    return checks


# --------------------------------------------------------------------------- 4. judge

def _gemini_json(prompt: str) -> tuple[dict, int, int]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=load_api_key())
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json",
        response_schema={"type": "object", "properties": {
            "correct": {"type": "integer"}, "supported": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["correct", "supported", "reason"]},
    )
    for attempt in range(8):
        try:
            response = client.models.generate_content(model=chat_model(JUDGE_MODEL), contents=prompt, config=config)
            usage = response.usage_metadata
            return (json.loads(response.text), getattr(usage, "prompt_token_count", 0) or 0,
                    getattr(usage, "candidates_token_count", 0) or 0)
        except Exception as error:  # noqa: BLE001
            message = str(error)
            if "429" in message or "RESOURCE_EXHAUSTED" in message or "503" in message:
                delay = re.search(r"retry in ([\d.]+)s", message)
                time.sleep(float(delay.group(1)) + 2 if delay else 20 * (attempt + 1))
                continue
            raise
    raise RuntimeError("The judge kept failing.")


def judge(questions: list[dict], answers: dict[str, dict]) -> dict[str, dict]:
    cached = {(r["id"], r["key"]): r for r in read_jsonl(JUDGE)}
    verdicts = {}
    for q in questions:
        a = answers.get(q["id"])
        if not a:
            continue
        key = sha(RUBRIC_VERSION + JUDGE_MODEL + q["reference_answer"] + a["answer"])
        if (q["id"], key) in cached:
            verdicts[q["id"]] = cached[(q["id"], key)]
            continue
        passages = "\n\n".join(a.get("contexts", [])) or "(no passages were retrieved)"
        prompt = (f"{RUBRIC}\n\nQUESTION: {q['question']}\nKIND: {q['kind']}\nREFERENCE: {q['reference_answer']}"
                  f"\n\nPASSAGES:\n{passages[:30000]}\n\nANSWER:\n{a['answer']}")
        result, tin, tout = _gemini_json(prompt)
        row = {"id": q["id"], "key": key, "correct": int(result["correct"]), "supported": int(result["supported"]),
               "reason": result["reason"], "tokens_in": tin, "tokens_out": tout, "model": JUDGE_MODEL}
        append_jsonl(JUDGE, row)
        verdicts[q["id"]] = row
        print(f"  judged {q['id']}: correct={row['correct']} supported={row['supported']}")
        time.sleep(PAUSE / 2)
    return verdicts


# --------------------------------------------------------------------------- 5. agreement

def read_human() -> dict[str, dict]:
    if not HUMAN.exists():
        return {}
    with HUMAN.open(newline="", encoding="utf-8") as f:
        return {r["id"].strip(): r for r in csv.DictReader(f)
                if (r.get("id") or "").strip() and str(r.get("correct", "")).strip() in ("0", "1")}


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def agreement(results: list[dict], human: dict[str, dict]) -> dict:
    out = {"graded": 0}
    for field in ("correct", "supported"):
        pairs = [(int(human[r["id"]][field]), int(r[f"judge_{field}"])) for r in results
                 if r["id"] in human and str(human[r["id"]].get(field, "")).strip() in ("0", "1")
                 and r.get(f"judge_{field}") != ""]
        out["graded"] = max(out["graded"], len(pairs))
        out[field] = {"n": len(pairs), "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
                      "kappa": kappa(pairs),
                      "disagreements": [r["id"] for r in results if r["id"] in human
                                        and str(human[r["id"]].get(field, "")).strip() in ("0", "1")
                                        and r.get(f"judge_{field}") != ""
                                        and int(human[r["id"]][field]) != int(r[f"judge_{field}"])]}
    return out


# --------------------------------------------------------------------------- 6. failures

def suggest_cause(r: dict) -> str:
    if r["kind"] == "unanswerable":
        return "unsupported claim" if not r["refused"] else ""
    if r["refused"]:
        return "wrong refusal" if r["expected_retrieved"] else "retrieval miss"
    if not r["correct"]:
        return "wrong answer from right passage" if r["expected_retrieved"] else "retrieval miss"
    if not r["supported"]:
        return "unsupported claim"
    return ""


def write_failures(results: list[dict]) -> list[dict]:
    previous = {}
    if FAILURES.exists():
        with FAILURES.open(newline="", encoding="utf-8") as f:
            previous = {r["id"]: r for r in csv.DictReader(f)}
    rows = []
    for r in results:
        if r["correct"] and r["supported"]:
            continue
        old = previous.get(r["id"], {})
        rows.append({
            "id": r["id"], "kind": r["kind"], "question": r["question"], "answer": r["answer"][:600],
            "correct": r["correct"], "supported": r["supported"], "expected_retrieved": r["expected_retrieved"],
            "judge_reason": r["judge_reason"], "cause_suggested": suggest_cause(r),
            "cause_final": old.get("cause_final", ""), "notes": old.get("notes", ""),
        })
    write_csv(FAILURES, rows, ["id", "kind", "question", "answer", "correct", "supported", "expected_retrieved",
                               "judge_reason", "cause_suggested", "cause_final", "notes"])
    return rows


# --------------------------------------------------------------------------- 7. before / after

def before_after(questions: list[dict], answers: dict[str, dict]) -> dict | None:
    q = next((q for q in questions if q["id"] == DEMO_ID), None)
    before = answers.get(DEMO_ID)
    if not q or not before or not DEMO_FILE.exists():
        return None
    summary = pipeline.add_file(DEMO_FILE)      # same pipeline as the add_document tool; no-op if already indexed
    retrieval.reset_cache()
    from agent import ask_async
    cache = EVAL / "before_after.json"
    key = sha(run_config() + q["question"])
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if saved.get("key") == key and saved.get("after"):
        after = saved["after"]
    else:
        after = asyncio.run(ask_async(q["question"], include_added=True, approve=lambda request: False))
    result = {"key": key, "id": DEMO_ID, "question": q["question"], "document": DEMO_FILE.name,
              "add_status": summary["status"], "before": before, "after": after}
    cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# Before and after adding a document\n",
          f"**Question ({DEMO_ID}):** {q['question']}\n",
          f"**Document added:** `extra/{DEMO_FILE.name}` (Regulation (EC) No 1107/2006, rights of disabled "
          f"persons and persons with reduced mobility when travelling by air), via `index.add_file` - the "
          f"function behind the `add_document` tool. Status: {summary['status']}.\n",
          "## Before\n", before["answer"], f"\n\nRetrieved: {', '.join(f'{s} {l}' for s, l in before['retrieved'])}\n",
          "## After\n", after["answer"], f"\n\nRetrieved: {', '.join(f'{s} {l}' for s, l in after['retrieved'])}\n"]
    (EVAL / "before_after.md").write_text("\n".join(md), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- main

def main() -> int:
    retrieval_only = "--retrieval-only" in sys.argv
    questions = read_questions()
    print(f"{len(questions)} questions\n\n== Retrieval (added documents hidden)")
    _, retrieval_summary = evaluate_retrieval(questions)
    plot_hit_rates(retrieval_summary)
    best = max(retrieval_summary, key=lambda r: (r["hit@3"], r["hit@5"], r["hit@1"], -r["words@3"]))
    print(f"  best by hit@3: {best['design']} (the server uses: {retrieval.chosen_design()})")
    if retrieval_only:
        return 0

    print(f"\n== Agent answers ({len(questions)} questions)")
    answers = run_agent(questions)
    missing = [q["id"] for q in questions if q["id"] not in answers]
    if missing:
        print(f"  no answer yet for {len(missing)} question(s): {', '.join(missing)}")

    print("\n== Judge")
    verdicts = judge(questions, answers)

    results = []
    for q in questions:
        a = answers.get(q["id"])
        if not a:
            continue
        checks = code_checks(q, a)
        v = verdicts.get(q["id"], {})
        # Code decides what code can decide; the judge decides the rest.
        if not checks["refusal_ok"]:
            correct = 0                              # wrong refusal, or an answer to an unanswerable question
        elif q["kind"] == "unanswerable":
            correct = 1                              # a refusal where a refusal was expected
        else:
            correct = v.get("correct", "")
        supported = v.get("supported", "")
        if supported != "" and not checks["citations_valid"]:
            supported = 0
        results.append({
            "id": q["id"], "kind": q["kind"], "question": q["question"], "answer": a["answer"],
            **checks, "judge_correct": v.get("correct", ""), "judge_supported": v.get("supported", ""),
            "judge_reason": v.get("reason", ""), "correct": correct, "supported": supported,
            "tokens_in": a["tokens_in"], "tokens_out": a["tokens_out"], "model_calls": a["model_calls"],
            "seconds": a["seconds"], "citations": "; ".join(f"{s}, {l}" for s, l in a["citations"]),
        })
    write_csv(EVAL / "results.csv", results, list(results[0]) if results else ["id"])

    # The grading sheet: no judge columns, so grades are made blind to the judge.
    human = read_human()
    write_csv(TO_GRADE, [{**r, "reference_answer": next(q["reference_answer"] for q in questions if q["id"] == r["id"]),
                          "retrieved_passages": "\n\n".join(answers[r["id"]].get("contexts", []))[:6000],
                          "correct": human.get(r["id"], {}).get("correct", ""),
                          "supported": human.get(r["id"], {}).get("supported", ""),
                          "notes": human.get(r["id"], {}).get("notes", "")} for r in results],
              ["id", "kind", "question", "reference_answer", "answer", "retrieved_passages", "correct", "supported", "notes"])

    agree = agreement(results, human)
    failures = write_failures([r for r in results if r["correct"] != ""])

    print("\n== Before / after adding a document")
    demo = before_after(questions, answers)
    if demo:
        print(f"  before: {demo['before']['answer'][:160]}...\n  after:  {demo['after']['answer'][:160]}...")

    # Summary
    graded = [r for r in results if r["correct"] != "" and r["supported"] != ""]
    answerable = [r for r in graded if r["kind"] != "unanswerable"]
    unanswerable = [r for r in graded if r["kind"] == "unanswerable"]
    n = len(results) or 1
    tin = sum(r["tokens_in"] for r in results) / n
    tout = sum(r["tokens_out"] for r in results) / n
    counts = {
        "questions": len(questions), "answered_by_agent": len(results), "graded": len(graded),
        "correct": sum(r["correct"] for r in graded),
        "correct_answerable": sum(r["correct"] for r in answerable), "answerable": len(answerable),
        "unsupported": sum(1 for r in graded if not r["supported"]),
        "correct_refusals": sum(r["refused"] for r in unanswerable), "unanswerable": len(unanswerable),
        "answered_unanswerable": sum(1 for r in unanswerable if not r["refused"]),
        "wrong_refusals": sum(r["refused"] for r in answerable),
        "citations_valid": sum(r["citations_valid"] for r in results),
        "guardrail_fired": [r["id"] for r in results if r["guardrail_fired"]],
    }
    causes = {}
    for f in failures:
        cause = f["cause_final"] or f["cause_suggested"] or "unclassified"
        causes[cause] = causes.get(cause, 0) + 1
    cost = {
        "tokens_in_per_question": round(tin), "tokens_out_per_question": round(tout),
        "seconds_per_question": round(sum(r["seconds"] for r in results) / n, 1),
        "model_calls_per_question": round(sum(r["model_calls"] for r in results) / n, 2),
        "searches_per_question": round(sum(r["searches"] for r in results) / n, 2),
        "usd_per_question_at_list_price": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 5),
        "price_assumption_usd_per_million": {"input": PRICE_IN, "output": PRICE_OUT},
    }
    summary = {"retrieval": retrieval_summary, "search_design": retrieval.chosen_design(),
               "best_design_by_hit@3": best["design"], "end_to_end": counts, "agreement": agree,
               "failure_causes": causes, "cost": cost, "judge_model": JUDGE_MODEL,
               "before_after": {"id": DEMO_ID, "before": demo["before"]["answer"], "after": demo["after"]["answer"]} if demo else None}
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(summary)
    print("\n" + (EVAL / "summary.md").read_text(encoding="utf-8"))
    if agree["graded"] < 20:
        print(f"NOTE: only {agree['graded']} human grades in eval/human_labels.csv. Grade at least 20 rows of "
              f"eval/to_grade.csv (before looking at results.csv) and copy id, correct, supported, notes there.")
    return 0


def write_summary_md(s: dict) -> None:
    lines = ["# Evaluation summary\n", "## Retrieval (answerable questions)\n",
             "| design | hit@1 | hit@3 | hit@5 | MRR@5 | multi: both in top 5 | words@3 | words@5 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s["retrieval"]:
        lines.append(f"| {r['design']} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {pct(r['hit@5'])} | {r['mrr@5']} | "
                     f"{pct(r['multi_both_in_top5']) if r['multi_both_in_top5'] != '' else ''} | {r['words@3']} | {r['words@5']} |")
    lines.append(f"\nBest by hit@3: **{s['best_design_by_hit@3']}**. The server uses **{s['search_design']}**.\n")
    e = s["end_to_end"]
    lines += ["## End to end\n", "| metric | value |", "|---|---|",
              f"| graded answers | {e['graded']} / {e['questions']} |",
              f"| correct (all) | {e['correct']} / {e['graded']} |",
              f"| correct (answerable) | {e['correct_answerable']} / {e['answerable']} |",
              f"| unsupported answers | {e['unsupported']} |",
              f"| correct refusals | {e['correct_refusals']} / {e['unanswerable']} |",
              f"| answered an unanswerable question | {e['answered_unanswerable']} |",
              f"| wrong refusals | {e['wrong_refusals']} / {e['answerable']} |",
              f"| answers with valid citations (code) | {e['citations_valid']} / {e['answered_by_agent']} |",
              f"| citation guardrail fired | {', '.join(e['guardrail_fired']) or 'never'} |", ""]
    a = s["agreement"]
    if a["graded"]:
        lines += ["## Human vs judge\n", "| grade | n | agreement | Cohen's kappa | disagreements |", "|---|---|---|---|---|"]
        for field in ("correct", "supported"):
            g = a[field]
            lines.append(f"| {field} | {g['n']} | {pct(g['agreement']) if g['agreement'] is not None else ''} | "
                         f"{g['kappa']} | {', '.join(g['disagreements'])} |")
        lines.append("")
    else:
        lines.append("## Human vs judge\n\nNo human grades yet (eval/human_labels.csv).\n")
    lines += ["## Failure causes\n"] + [f"- {k}: {v}" for k, v in sorted(s["failure_causes"].items())] + [""]
    c = s["cost"]
    lines += ["## Cost per question\n",
              f"- {c['tokens_in_per_question']} input + {c['tokens_out_per_question']} output tokens, "
              f"{c['model_calls_per_question']} model calls, {c['searches_per_question']} searches",
              f"- {c['seconds_per_question']} s per question",
              f"- about ${c['usd_per_question_at_list_price']} at the assumed list price "
              f"(${c['price_assumption_usd_per_million']['input']}/M in, ${c['price_assumption_usd_per_million']['output']}/M out); "
              "free tier: $0", ""]
    (EVAL / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
