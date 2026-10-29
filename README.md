# Multi-Agent ML Optimisation Pipeline

Give it a **goal** and a **classification dataset** (a CSV/TSV table or a folder of
images). Three LLM agents then build and improve a model step by step:

- **Model Strategist**: decides which experiment to run next, and why.
- **Coder**: writes the training script (`train.py`) for that experiment.
- **Error Analysis Agent**: reads the validation results after every run and reports
  back to the Strategist.

Deterministic Python scripts, not the LLMs, compute every metric. Decisions use the
**validation** set only; the **test** set is used exactly once, on the final model.

```
 Strategist ──proposal──► Coder ──train.py──► train + evaluate on validation
     ▲                                                   │
     └──────── analysis ◄── Error Analysis Agent ◄── validation report
 end: best model by validation score ──► test set (once) ──► final report
```

---

## 1. Requirements

- **Python 3.10 or newer** (Windows, macOS or Linux). Check with `python --version`
  (on macOS/Linux it may be `python3`).
- An **OpenRouter API key**, free at <https://openrouter.ai/keys>. The agents are LLM
  calls through OpenRouter; free models are available.
- Internet access (LLM calls, and installing packages on the first run).
- Optional: an NVIDIA GPU (with the NVIDIA driver installed) or Apple Silicon, for
  faster image training. Everything also works on CPU, only slower.

## 2. Install (one time)

Unzip the folder, open a terminal inside it, and create a virtual environment.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell refuses to run the activate script, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
(In the classic Command Prompt use `.venv\Scripts\activate.bat`.)

Activate the environment again (second line above) each time you open a new terminal.

PyTorch is **not** in `requirements.txt`. It is installed automatically the first time
you train on images, and the right build (NVIDIA GPU, Apple Silicon or CPU) is chosen
for your machine. To see what was detected:
```bash
python preflight.py
```

## 3. Your API key

Either paste the key when the program asks for it (typing is hidden; it is kept in
memory for that session only and never saved), or set it beforehand:

| System | Command (current terminal only) |
|---|---|
| macOS / Linux | `export OPENROUTER_API_KEY=sk-or-v1-...` |
| Windows PowerShell | `$env:OPENROUTER_API_KEY="sk-or-v1-..."` |
| Windows Command Prompt | `set OPENROUTER_API_KEY=sk-or-v1-...` |

## 4. Run

```bash
python chat.py
```

It checks your system, then asks:

1. for the API key (only if it isn't set),
2. what you want to build (one sentence),
3. where your dataset is:
   - `demo`: built-in handwritten-digits table (fast, good first test),
   - `demo-images`: generates a small image dataset to try deep learning,
   - a path to a `.csv` / `.tsv` file: it shows the columns and asks which one is the label,
   - a path to a folder with **one sub-folder per class**, each containing images, e.g.
     `leaves/healthy/*.jpg`, `leaves/rust/*.jpg`,
4. whether you have a model in mind to start with (press Enter to let it choose),
5. which LLM each agent should use (below), then **Start optimization? (y/n)**.

It then explains each experiment as it runs.

`python main.py` runs a fixed task (the digits table) without the chat; it asks the
same API-key and model questions.

### Choosing the LLM for each agent

There is no default model: you choose one for each agent.

- The current list of **free** OpenRouter models is shown with their exact identifiers.
  Answer with a list number or an identifier. `search <text>` finds any model, free or
  paid; `list` shows the free list again.
- Same model for all three agents, or a different one for each: both are fine.
- Paid models (and router aliases such as `openrouter/free`, where OpenRouter picks the
  model) are flagged and need you to type `yes`. No paid call happens without that.
- Each chosen model gets a tiny test call first. If one fails you choose again; models
  are never switched automatically.
- Your choice is saved in `~/.ml_agent_config.json` (model names only, never the key)
  and offered next time; it is reused only if you confirm.

Free models are sometimes slow or rate-limited (HTTP 429). The program waits and
retries automatically.

## 5. What you get

Each run creates `experiments/run_<date>_<time>/`:

| File / folder | Contents |
|---|---|
| `final_report.md` | Start here: every experiment, the selected model, its single test-set result, LLM usage. |
| `final/` | The selected `train.py`, the trained model (`model.pkl` or `model.pt`), test metrics and confusion matrix. |
| `iter_00/`, `iter_01/`, … | Per experiment: the proposal, code, training log, validation report, the analysis, plots. |
| `run_config.json`, `llm_usage.json` | Settings, seed and chosen models; per-call latency, tokens and retries. |

The latest `train.py` and model are also left in the `workspace/` folder.

## 6. Settings (optional, environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SELECTION_METRIC` | `macro_f1` | Validation metric used to pick the best model: `macro_f1`, `accuracy`, `balanced_accuracy`, `weighted_f1`. |
| `PIPELINE_SEED` | `42` | Random seed for the experiments. |
| `ISOLATION` | `venv` | Each experiment runs in its own cached environment under `.envs/`, with the packages its code needs installed automatically. `none` uses your current environment instead (faster, if the packages are already installed). |
| `PIP_ALLOWLIST` | unset | Comma-separated packages the generated code may install. |
| `TRAIN_TIMEOUT` | `3600` | Seconds allowed per training run. |
| `KEEP_CHECKPOINTS` | `1` | `0` keeps model files only for experiments that were the best. |
| `LLM_MAX_RETRIES` | `5` | Retries for rate limits / timeouts. |

Set them like the API key above, e.g. `export ISOLATION=none` (macOS/Linux) or
`$env:ISOLATION="none"` (PowerShell).

## 7. Check that it works (no API key needed)

The automated tests fake the LLM provider but run everything else for real
(training, evaluation, test-set protection, the model-selection dialogue):

```bash
python -m unittest discover -s tests -v
```
Expect `OK` after about a minute. The 5 image-adapter tests are skipped unless PyTorch
is installed in the active environment.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Linux: `venv` fails ("ensurepip is not available") | Install it: `sudo apt install python3-venv` (Ubuntu/Debian), then retry. |
| `python` not found | Install Python 3.10+ from python.org (Windows: tick "Add Python to PATH"), or use `python3`. |
| Invalid API key | Create a new one at openrouter.ai/keys; check for spaces when pasting. |
| Many "429" / slow responses | Free models are busy. Wait, or choose a different free model. |
| First image run takes long | It is installing PyTorch and downloading pretrained weights once; later runs reuse them. |
| No GPU detected | Install the NVIDIA driver (`python preflight.py` prints the link); CPU still works. |

## 9. Safety

The training code is written by an LLM and runs on your computer with your user's
permissions, and missing packages are installed from PyPI automatically. Run it on a
machine or account you are comfortable with (a container or VM for anything
sensitive), and use `PIP_ALLOWLIST` to restrict installs. The test-set lock prevents
accidental leakage in the generated code, but it is not a security sandbox.

## 10. Code overview

| File | Role |
|---|---|
| `chat.py` | Interactive front end (start here). |
| `main.py` | Non-interactive entry point (digits task). |
| `agents.py` | The three agents' prompts and calls. |
| `loop.py` | The optimisation loop: propose → code → train → evaluate → analyse → keep best → final test. |
| `score.py` | Deterministic metrics, learning curves, statistical comparison with the current best. |
| `report.py` | Plots and the final report. |
| `schemas.py` | The structured messages between agents. |
| `workspace_files.py` | Data loaders with the test-set lock, the image input adapter, the Coder's rules. |
| `llm.py` | OpenRouter client: per-agent models, paid-call guard, retries, usage log. |
| `model_selection.py` | Interactive model selection and the API-key prompt. |
| `sandbox.py`, `environments.py` | Run generated code in isolated, cached environments. |
| `preflight.py` | System / GPU check; picks the right PyTorch build. |
| `tests/` | Automated checks. |
