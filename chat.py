"""
chat.py — the interactive front-end.

Run it:
    python chat.py      # needs an OpenRouter API key (env var, or typed in when asked);
                        # you pick the LLM for each of the three agents

It talks to you in the terminal: asks what you want to build and where your data
is, works out what kind of dataset you pointed it at, then runs the training loop
and explains each step as it goes.
"""
import sys
import time
from pathlib import Path

from loop import Reporter, run
from workspace_files import DATA_PY   # the built-in 'demo' loader

SLOW = "--fast" not in sys.argv       # small pauses make it feel like a chat


# --------------------------------------------------------------------------- #
# little conversation helpers
# --------------------------------------------------------------------------- #
def say(text: str = "", pause: float = 0.35) -> None:
    print(text)
    if SLOW and text:
        time.sleep(pause)


def ask(prompt: str) -> str:
    try:
        return input(f"\n{prompt}\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        say("\n\nOkay, stopping here. See you next time.")
        sys.exit(0)


# --------------------------------------------------------------------------- #
# dataset loaders generated for the user's data (plug into loop.run(data_py=...))
# --------------------------------------------------------------------------- #
def _csv_loader(path: str, label: str, sep: str = ",") -> str:
    from workspace_files import tabular_data_py
    return tabular_data_py(f'''\
import pandas as pd

PATH = {path!r}
LABEL = {label!r}

def _load():
    df = pd.read_csv(PATH, sep={sep!r})
    y, names = pd.factorize(df[LABEL], sort=True)
    X = df.drop(columns=[LABEL]).select_dtypes(include="number").fillna(0).values
    return X, y, [str(n) for n in names]
''')


def _image_loader(path: str) -> str:
    from workspace_files import image_data_py
    return image_data_py(path)


def _make_demo_images(root: Path) -> None:
    """Write sklearn digits out as a folder-per-class PNG dataset for a real DL test."""
    from sklearn.datasets import load_digits
    from PIL import Image
    import numpy as np
    X, y = load_digits(return_X_y=True)
    for cls in range(10):
        (root / str(cls)).mkdir(parents=True, exist_ok=True)
    for i, (img, label) in enumerate(zip(X, y)):
        arr = (img.reshape(8, 8) / 16 * 255).astype("uint8")
        Image.fromarray(arr).convert("RGB").resize((32, 32)).save(root / str(label) / f"{i}.png")


def resolve_dataset() -> tuple[str, str, str]:
    """Ask for a path, inspect it, return (data_py, human_summary, mode). Loops until valid.
    mode is 'tabular' (scikit-learn) or 'image' (PyTorch deep learning)."""
    while True:
        raw = ask("Where is your dataset? Give a file or folder path.\n"
                  "  'demo'        = built-in digits table (scikit-learn, fast)\n"
                  "  'demo-images' = a real image folder for a deep-learning test (PyTorch)")

        if raw.lower() in ("demo", "", "d"):
            return DATA_PY, "the built-in 'digits' table (1,797 rows, 10 classes)", "tabular"

        if raw.lower() in ("demo-images", "demo-image", "di"):
            root = Path("demo_images").resolve()
            if not root.exists():
                say("Building a small image dataset (digits as PNGs)...")
                _make_demo_images(root)
            classes = sorted(d.name for d in root.iterdir() if d.is_dir())
            return _image_loader(str(root)), \
                f"a generated image folder with {len(classes)} classes (deep-learning demo)", "image"

        p = Path(raw).expanduser().resolve()
        if not p.exists():
            say(f"I can't find anything at '{raw}'. Let's try again.")
            continue

        # a CSV/TSV file -> tabular data (scikit-learn)
        if p.is_file() and p.suffix.lower() in (".csv", ".tsv"):
            header = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            cols = [c.strip() for c in header.split(sep)]
            say(f"That's a table with these columns: {', '.join(cols)}")
            label = ask(f"Which column is the label to predict? "
                        f"(press Enter for the last one, '{cols[-1]}')") or cols[-1]
            if label not in cols:
                say(f"'{label}' isn't a column; using the last one, '{cols[-1]}'.")
                label = cols[-1]
            return _csv_loader(str(p), label, sep), \
                f"a table at {p.name}, predicting '{label}' from the numeric columns", "tabular"

        # a folder with class sub-folders -> image classification (PyTorch DL)
        if p.is_dir():
            classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
            if len(classes) >= 2:
                counts = {c: sum(1 for _ in (p / c).iterdir()) for c in classes}
                summary = ", ".join(f"{c} ({counts[c]})" for c in classes)
                say(f"That's an image folder. I see these classes: {summary}")
                say("I'll treat this as a deep-learning task: PyTorch, GPU when available, "
                    "transfer learning from a pretrained backbone.")
                return _image_loader(str(p)), \
                    f"images in {p.name} across {len(classes)} classes (deep learning)", "image"
            say("That folder has no class sub-folders (I expect one sub-folder per "
                "class, each full of images). Let's try another path.")
            continue

        say("I'm not sure how to read that. I understand CSV/TSV files, or a folder "
            "with one sub-folder per class. Try again?")


# --------------------------------------------------------------------------- #
# the conversational reporter: narrates the loop like a helper explaining as it goes
# --------------------------------------------------------------------------- #
class ChatReporter(Reporter):
    def plan(self, task, max_iters, patience, metric):
        say("\nHere's how I'll work, step by step:")
        say("  1. A Model Strategist proposes an experiment; a Coder writes train.py for it.")
        say("  2. I train it, then score it on a VALIDATION set (never the test set).")
        say("  3. An Error Analysis agent reads the confusion matrix, per-class scores, "
            "mistakes and learning curves, and reports back to the Strategist.")
        say(f"  4. I keep whichever version has the best validation {metric}, and stop when it "
            f"stops improving (after {patience} flat rounds, or {max_iters} experiments).")
        say("  5. Only then do I score the chosen model ONCE on the untouched test set.")
        say("\nEach attempt runs in its own isolated environment and installs whatever "
            "libraries the code needs. Let's go.\n")

    def proposal(self, i, p):
        say(f"── Experiment {i}: {p.experiment_name} ({p.architecture})")
        say(f"   Strategist: {p.rationale}")

    def result(self, r):
        if not r.ok:
            last = r.error.strip().splitlines()[-1] if r.error.strip() else r.status
            say(f"   It hit an error: {last[:140]}")
            say("   That's fine — a failed run is information too; the strategist will react to it.")
        else:
            m = r.val_metrics
            say(f"   Trained. Validation: accuracy {m['accuracy']:.1%}, macro-F1 {m['macro_f1']:.3f} "
                f"(train accuracy {r.train_metrics['accuracy']:.1%}).")

    def analysis(self, a):
        if not a:
            return
        say(f"   Error analysis ({a.fit_diagnosis.replace('_', ' ')}): {a.summary}")
        for w in a.systematic_errors[:2]:
            say(f"     · {w}")
        if a.recommendations:
            say(f"   Suggests: {a.recommendations[0]}")

    def verdict(self, is_best, score, stale, patience):
        if is_best:
            say("   That's the best so far — I'll keep this one.")
        else:
            say(f"   Not meaningfully better than my current best, so I'll keep that one "
                f"({stale}/{patience} flat rounds).")

    def progress(self, line):
        # stream the training script's own prints (epochs, etc.) so it's clearly alive
        if line.strip():
            print(f"      · {line}")

    def converged(self, reason):
        say(f"\nStopping: {reason.rstrip('.')}.")

    def final_test(self, metrics):
        say(f"\nFinal check on the untouched test set (done once): accuracy "
            f"{metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f}.")

    def done(self, result):
        say("\n" + "=" * 60)
        best = result.best
        if not best:
            say("I couldn't get a working model this time. Check the dataset path and try again.")
            return
        say(f"Done. Selected experiment {best.iteration}: {best.proposal.experiment_name} "
            f"(validation {result.selection_metric} {best.selection_score:.3f}).")
        say(f"Full report, per-iteration analyses, code and checkpoints: {result.run_dir}")
        say("The winning train.py and model are also in the 'workspace' folder.")


# --------------------------------------------------------------------------- #
def _ensure_openai_installed() -> bool:
    """Make sure the openai package is available; offer to install it if not."""
    import importlib.util
    if importlib.util.find_spec("openai"):
        return True
    ans = ask("The 'openai' package isn't installed (needed to call the model). "
              "Install it now? (Y/n)").lower()
    if ans in ("n", "no"):
        return False
    say("Installing openai...")
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pip", "install", "openai"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "openai",
                            "--break-system-packages"], capture_output=True, text=True)
    ok = importlib.util.find_spec("openai") is not None
    say("Installed." if ok else "Couldn't install it automatically — try "
        "'pip install openai' yourself, then rerun.")
    return ok


def setup_llm() -> bool:
    """Check the prerequisites for calling the LLMs: the openai package and an API key
    (asked for if it isn't in the environment). The models themselves are chosen
    later by the user, per agent. Returns False to quit."""
    if not _ensure_openai_installed():
        return False
    import model_selection
    if not model_selection.ensure_api_key(ask, say):
        say("I need an OpenRouter API key to run the agents. Okay, stopping here.")
        return False
    return True


def main() -> None:
    import preflight
    info = preflight.detect()
    preflight.apply(info)

    say("=" * 60)
    say("Hi — I'm your ML building assistant.")
    say("I'll turn a goal and a dataset into a trained, tested model, improving it "
        "step by step and explaining what I do along the way.")
    say("=" * 60)
    say(preflight.report(info))
    if info["device"] == "cpu" and info["gpu"] is None:
        say("(Deep-learning runs will work on CPU but be slow. Tabular runs are fine on CPU.)")

    if not setup_llm():
        return

    goal = ask("What would you like to build today? Describe the goal in a sentence.")
    while not goal:
        goal = ask("Just a sentence is fine — e.g. 'tell healthy vs diseased leaves apart'.")

    data_py, summary, mode = resolve_dataset()
    loader = "get_loaders" if mode == "image" else "get_splits"

    # optional model preference — a starting point, not a rule
    pref = ask("Do you have a model in mind to start with? "
               "(e.g. 'random forest', 'gradient boosting', 'an SVM' — "
               "or press Enter and I'll choose one myself.)")

    say(f"\nGot it. Goal: {goal}")
    say(f"Data: {summary}")
    if pref:
        say(f"Starting model: {pref} — I'll try it first, and only switch if the "
            f"numbers clearly favour something else (I'll tell you if I do).")
        initial_action = f"Start with the user's suggested model: {pref}."
        task = (f"{goal}. Data is provided via `from data import {loader}`. "
                f"The user suggested starting with: {pref} (a starting point you may "
                f"later move away from only if the evidence clearly favours it).")
    else:
        say("No preference — I'll pick a sensible model to begin with.")
        initial_action = "Establish a simple, sensible baseline model of your own choosing."
        task = f"{goal}. Data is provided via `from data import {loader}`."

    # The user chooses every agent's model; this ends with "Start optimization?".
    import model_selection
    if model_selection.select_models(ask, say) is None:
        say("No problem — rerun me whenever you're ready.")
        return

    run(task, data_py=data_py, reporter=ChatReporter(),
        initial_action=initial_action, preference=pref, mode=mode)


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    main()
