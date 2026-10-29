"""
agents.py — the three LLM roles. Each has ONE responsibility:

  strategist_agent : MODEL STRATEGIST. Proposes the baseline, then — from the
                     experiment history and the latest error analysis — decides the
                     next experiment (architecture, regularisation, augmentation,
                     learning rate, ...). Never writes code.
  coder_agent      : CODER. Turns a StrategyProposal into a complete train.py.
                     Never decides what to try.
  analysis_agent   : ERROR ANALYSIS AGENT. Interprets the deterministic validation
                     report produced by score.py after every iteration. Never
                     computes metrics itself, and never sees test data.

Each role calls exactly the model the user selected for it
(llm.model_for(role); see model_selection.py).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from llm import call_text, call_json
from schemas import ErrorAnalysis, ExperimentRecord, StrategyProposal
from workspace_files import TRAIN_CONTRACT, TRAIN_CONTRACT_IMAGE

# Set by the loop once optimisation is over: agents must never run after the test
# set has been touched, so any call after that point is a bug.
_LOCKED = False


def lock() -> None:
    global _LOCKED
    _LOCKED = True


def unlock() -> None:
    global _LOCKED
    _LOCKED = False


def _guard() -> None:
    if _LOCKED:
        raise RuntimeError("Agents are locked: optimisation is over and the test set "
                           "has been evaluated. No agent may run after that point.")


# --------------------------------------------------------------------------- #
# Shared formatting
# --------------------------------------------------------------------------- #
def history_rows(history: List[ExperimentRecord], metric: str) -> List[Dict[str, Any]]:
    """Compact, validation-only view of past experiments (used in several prompts)."""
    rows = []
    for r in history:
        row = {"iteration": r.iteration, "experiment": r.proposal.experiment_name,
               "architecture": r.proposal.architecture,
               "hyperparameters": r.proposal.hyperparameters,
               "training_config": r.proposal.training_config, "status": r.status}
        if r.ok:
            row.update({f"val_{metric}": r.selection_score,
                        "val_accuracy": r.val_metrics.get("accuracy"),
                        "train_accuracy": r.train_metrics.get("accuracy"),
                        "val_loss": r.val_metrics.get("loss"),
                        "fit_diagnosis": r.analysis.fit_diagnosis if r.analysis else None,
                        "was_best_when_run": r.is_best})
        else:
            row["error"] = r.error[-300:]
        rows.append(row)
    return rows


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str)


# =========================================================================== #
# 1. MODEL STRATEGIST
# =========================================================================== #
STRATEGIST_SYS = (
    "You are the Model Strategist of an automated deep-learning optimisation loop for "
    "classification. You decide WHAT experiment to run next; a separate Coder agent "
    "implements it and a separate Error Analysis agent reports what happened.\n"
    "Rules:\n"
    "- Base every decision on evidence: the experiment history and the latest error "
    "analysis. Say which findings your proposal addresses.\n"
    "- Never repeat a configuration that was already tried (same architecture, "
    "hyperparameters and training config). Do not revisit approaches that failed unless "
    "you change what made them fail.\n"
    "- Typical levers: architecture/backbone changes, fine-tuning depth, regularisation "
    "(dropout, weight decay, label smoothing), data augmentation, learning rate and "
    "schedule, epochs/early stopping, class weighting or resampling for weak classes.\n"
    "- Overfitting (train >> val) -> regularise, augment, reduce capacity or epochs. "
    "Underfitting (both low) -> more capacity, unfreeze layers, train longer, tune LR. "
    "Crash -> remove the cause, keep the rest.\n"
    "- Changes that are within statistical noise are not real gains; prefer genuinely "
    "different hypotheses over tiny knob turns once a family has plateaued.\n"
    "- Model selection uses the VALIDATION set only. There is no test data in this loop; "
    "never ask the coder to use or look for one.\n"
    "- coder_instructions must be specific enough to implement without further decisions.\n"
    "- Set stop=true only when diverse attempts have plateaued and nothing promising remains."
)


def strategist_agent(task: str, history: List[ExperimentRecord],
                     analysis: Optional[ErrorAnalysis], *, mode: str = "tabular",
                     metric: str = "macro_f1", preference: str = "",
                     initial_guidance: str = "", best: Optional[ExperimentRecord] = None,
                     remaining: int = 0) -> StrategyProposal:
    _guard()
    contract = TRAIN_CONTRACT_IMAGE if mode == "image" else TRAIN_CONTRACT
    pref = (f"\nUSER PREFERENCE: {preference}. Treat it as a starting point, not a rule; if "
            f"you move away from it, say why in the rationale.\n" if preference else "")
    if not history:
        ask = (f"Propose the INITIAL BASELINE experiment. Guidance: {initial_guidance or 'a simple, sensible baseline'}.")
    else:
        ask = ("Propose the NEXT experiment. Respond explicitly to the error analysis "
               "below and avoid every configuration in the history.")
    best_line = (f"CURRENT BEST: iteration {best.iteration} '{best.proposal.experiment_name}' "
                 f"with validation {metric} = {best.selection_score}\n" if best else
                 "CURRENT BEST: none yet\n")
    prompt = (
        f"TASK: {task}\nMODE: {mode}\nSELECTION METRIC (validation): {metric}\n"
        f"EXPERIMENTS REMAINING IN BUDGET: {remaining}\n{pref}\n"
        f"CONSTRAINTS THE CODE MUST RESPECT (for your awareness):\n{contract}\n"
        f"EXPERIMENT HISTORY (validation only):\n{_dump(history_rows(history, metric))}\n\n"
        f"{best_line}\n"
        f"LATEST ERROR ANALYSIS:\n{analysis.model_dump_json(indent=1) if analysis else '(none)'}\n\n"
        f"{ask}"
    )
    proposal = call_json("strategist", STRATEGIST_SYS, prompt, StrategyProposal)

    # Deterministic guard against repeating an earlier configuration.
    seen = {r.proposal.fingerprint(): r.iteration for r in history}
    if not proposal.stop and proposal.fingerprint() in seen:
        prompt += (f"\n\nYour proposal is IDENTICAL to iteration {seen[proposal.fingerprint()]}. "
                   f"Propose a genuinely different experiment.")
        proposal = call_json("strategist", STRATEGIST_SYS, prompt, StrategyProposal)
    return proposal


# =========================================================================== #
# 2. CODER
# =========================================================================== #
CODER_SYS = (
    "You are the Coder in an ML optimisation loop. You implement EXACTLY the experiment "
    "the Model Strategist specified: same architecture, hyperparameters and training "
    "configuration. You do not choose a different model or add unrequested changes; if "
    "something is ambiguous, pick the simplest faithful interpretation. You write one "
    "self-contained Python training script, follow the contract exactly, and output "
    "ONLY Python code."
)


def coder_agent(task: str, proposal: StrategyProposal, base_code: str, *,
                mode: str = "tabular", previous_error: str = "", feedback: str = "") -> str:
    _guard()
    contract = TRAIN_CONTRACT_IMAGE if mode == "image" else TRAIN_CONTRACT
    spec = proposal.model_dump(exclude={"stop"})
    prompt = (
        f"TASK: {task}\n\n{contract}\n"
        f"EXPERIMENT TO IMPLEMENT (from the Model Strategist):\n{_dump(spec)}\n\n"
        f"CURRENT BEST train.py (starting point; empty on the first round):\n"
        f"{base_code or '(none yet)'}\n"
    )
    if previous_error:
        prompt += f"\nTHE PREVIOUS ATTEMPT FAILED WITH:\n{previous_error[-800:]}\n"
    if feedback:
        prompt += f"\nFIX THIS PROBLEM WITH YOUR LAST OUTPUT: {feedback}\n"
    prompt += "\nReturn the complete new train.py."
    return call_text("coder", CODER_SYS, prompt)


# =========================================================================== #
# 3. ERROR ANALYSIS AGENT
# =========================================================================== #
ANALYST_SYS = (
    "You are the Error Analysis agent of an ML optimisation loop. After each training "
    "run you receive a VALIDATION report computed by deterministic scripts: train/val "
    "accuracy and loss, confusion matrix, per-class precision/recall/F1, the most "
    "confused class pairs, misclassified examples, learning curves with a summary, a "
    "paired statistical comparison with the current best model, and previous results.\n"
    "Rules:\n"
    "- Interpret; do not recompute. Quote the numbers you rely on.\n"
    "- Keep OBSERVATIONS (facts visible in the report) strictly separate from "
    "HYPOTHESES (possible causes, stated tentatively).\n"
    "- Diagnose fit: overfitting, underfitting, good_fit, or unclear, using the "
    "generalisation gap and the learning curves.\n"
    "- Name class-specific weaknesses and systematic confusions.\n"
    "- Judge significance ONLY from comparison_to_current_best (bootstrap CI, McNemar "
    "p-value, evidence flag). If evidence is insufficient, say so; do not over-claim.\n"
    "- Give concrete, prioritised recommendations to the Model Strategist.\n"
    "- You only ever see training and validation data."
)


def analysis_agent(report: Dict[str, Any], experiment: StrategyProposal,
                   task: str = "") -> ErrorAnalysis:
    _guard()
    prompt = (
        f"TASK: {task}\n"
        f"EXPERIMENT JUST RUN:\n{_dump(experiment.model_dump(exclude={'coder_instructions', 'stop'}))}\n\n"
        f"DETERMINISTIC VALIDATION REPORT:\n{_dump(report)}\n\n"
        f"Analyse this run for the Model Strategist."
    )
    return call_json("analyst", ANALYST_SYS, prompt, ErrorAnalysis)
