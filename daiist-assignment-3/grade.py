"""
Grade the agent's answers yourself, one at a time, WITHOUT seeing the judge.

    uv run python grade.py

Shows each question, the reference answer and the agent's answer. Type:
    correct:   1 (right) / 0 (wrong)
    supported: 1 (every claim is in the passages) / 0 (something is not)
               type p first to see the retrieved passages
    note:      one short line (required when you give a 0)
Grades are saved to eval/human_labels.csv after every answer; Ctrl+C stops,
running it again continues where you left off. Grade at least 20.
"""

import csv
import json
import random
import textwrap
from pathlib import Path

EVAL = Path(__file__).resolve().parent / "eval"
LABELS = EVAL / "human_labels.csv"
GOAL = 20


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(p, 100, initial_indent=indent, subsequent_indent=indent)
                     for p in text.splitlines() if p.strip())


def ask(prompt: str, allowed: tuple[str, ...]) -> str:
    while True:
        value = input(prompt).strip().lower()
        if value in allowed:
            return value
        print(f"    type one of: {', '.join(allowed)}")


def main() -> None:
    rows = list(csv.DictReader((EVAL / "to_grade.csv").open(encoding="utf-8")))
    done = {}
    if LABELS.exists():
        done = {r["id"]: r for r in csv.DictReader(LABELS.open(encoding="utf-8")) if r.get("correct") in ("0", "1")}
    random.Random(7).shuffle(rows)            # a fixed mix of single, multi and unanswerable
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} graded so far, goal {GOAL}. Ctrl+C to stop at any time.\n")
    for r in todo:
        print("=" * 100)
        print(f"{r['id']}  ({r['kind']})   graded: {len(done)}/{GOAL}\n")
        print("QUESTION\n" + wrap(r["question"]) + "\n")
        print("REFERENCE (what the documents say)\n" + wrap(r["reference_answer"]) + "\n")
        print("AGENT'S ANSWER\n" + wrap(r["answer"]) + "\n")
        correct = ask("correct? 1/0 > ", ("1", "0"))
        supported = ask("supported? 1/0 (p = show passages) > ", ("1", "0", "p"))
        if supported == "p":
            print("\nRETRIEVED PASSAGES\n" + wrap(r["retrieved_passages"]) + "\n")
            supported = ask("supported? 1/0 > ", ("1", "0"))
        note = input("note (why, if you gave a 0) > ").strip()
        while (correct == "0" or supported == "0") and not note:
            note = input("    a 0 needs a short note > ").strip()
        done[r["id"]] = {"id": r["id"], "correct": correct, "supported": supported, "notes": note}
        with LABELS.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "correct", "supported", "notes"])
            writer.writeheader()
            writer.writerows(sorted(done.values(), key=lambda x: x["id"]))
        if len(done) == GOAL:
            print(f"\n{GOAL} graded: enough for the assignment. Keep going or press Ctrl+C.")
    print(f"\nDone: {len(done)} grades in eval/human_labels.csv. Now run: uv run python main.py evaluate")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Your grades so far are saved; run it again to continue.")
