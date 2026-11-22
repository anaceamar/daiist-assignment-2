"""
Assignment 3 entry point. Don't rename it or its commands.

    uv run python main.py index                 # index.py or index.ipynb
    uv run python main.py ask "your question"   # agent.py
    uv run python main.py evaluate              # evaluate.py or evaluate.ipynb
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_entrypoint(name: str) -> None:
    py_path = ROOT / f"{name}.py"
    nb_path = ROOT / f"{name}.ipynb"

    if py_path.exists() and nb_path.exists():
        raise SystemExit(
            f"Found both {py_path.name} and {nb_path.name}. main.py won't guess which "
            f"one you want. Delete whichever one you're not using, then try again."
        )

    if py_path.exists():
        subprocess.run([sys.executable, str(py_path)], cwd=ROOT, check=True)
        return

    if nb_path.exists():
        with tempfile.NamedTemporaryFile(
            suffix=".py", dir=ROOT, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("w") as tmp_file:
                subprocess.run(
                    ["jupyter", "nbconvert", "--to", "script", "--stdout", str(nb_path)],
                    cwd=ROOT, check=True, stdout=tmp_file,
                )
            subprocess.run([sys.executable, str(tmp_path)], cwd=ROOT, check=True)
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    raise FileNotFoundError(
        f"Expected {py_path.name} or {nb_path.name} in {ROOT}. Create one of them."
    )


def run_agent(question: str) -> None:
    agent = ROOT / "agent.py"
    if not agent.exists():
        raise FileNotFoundError(f"Expected agent.py in {ROOT}. It must be a script, not a notebook.")
    subprocess.run([sys.executable, str(agent), question], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["index", "ask", "evaluate"])
    parser.add_argument("question", nargs="*", help="the question, for `ask`")
    args = parser.parse_args()
    if args.stage == "ask":
        if not args.question:
            parser.error('ask needs a question: main.py ask "your question"')
        run_agent(" ".join(args.question))
    else:
        run_entrypoint(args.stage)
