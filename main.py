"""
main.py — run the three-agent loop on a fixed task (sklearn digits), no chat.

    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-v1-...     # or type it in when asked
    python main.py

You are asked which OpenRouter model each agent should use before anything runs.
"""
import sys

import model_selection
from loop import run

TASK = (
    "Classify 8x8 handwritten-digit images (sklearn 'digits', 10 classes). "
    "Maximise validation macro-F1; the test split is reserved for the final model."
)

def _ask(prompt: str) -> str:
    return input(f"\n{prompt}\n> ")


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    if model_selection.select_models(_ask, print) is None:
        raise SystemExit("No models selected; nothing was run.")
    run(TASK)
