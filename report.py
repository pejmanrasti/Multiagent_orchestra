"""
report.py — figures and the final report. Runs on the framework side (not in the
agent's environment) from the metrics that score.py computed, so it is the same
no matter what code the Coder produced.

  plot_iteration()      per iteration, VALIDATION only:
                          confusion_matrix.png, learning_curves.png,
                          misclassified.png (image datasets: ids are file paths)
  write_final_report()  after the single test evaluation:
                          final_report.md + final_report.json in the run folder

Plotting needs matplotlib (and pillow for image grids); without them the text
artefacts are still written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import ExperimentRecord


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(metrics: Dict[str, Any], path: Path, title: str) -> None:
    plt = _plt()
    import numpy as np
    cm = np.asarray(metrics["confusion_matrix"])
    names = metrics["class_names"]
    k = len(names)
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * k + 2), max(3.5, 0.45 * k + 2)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    if k <= 20:
        for r in range(k):
            for c in range(k):
                ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=8,
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_curves(curves: List[Dict[str, Any]], path: Path) -> None:
    if not curves:
        return
    plt = _plt()
    ep = [r.get("epoch", i + 1) for i, r in enumerate(curves)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, kind in ((a1, "loss"), (a2, "acc")):
        for split in ("train", "val"):
            ys = [r.get(f"{split}_{kind}") for r in curves]
            if any(isinstance(y, (int, float)) for y in ys):
                ax.plot(ep, ys, marker="o", label=split)
        ax.set_title("Loss" if kind == "loss" else "Accuracy")
        ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_misclassified(metrics: Dict[str, Any], path: Path, limit: int = 12) -> None:
    """Grid of the most confident errors, when sample ids are image file paths."""
    rows = [r for r in metrics.get("misclassified_examples", [])
            if isinstance(r["id"], str) and Path(r["id"]).is_file()][:limit]
    if not rows:
        return
    plt = _plt()
    from PIL import Image
    cols = 4
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.4, n_rows * 2.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        ax.imshow(Image.open(r["id"]).convert("RGB"))
        conf = f" ({r['confidence']:.0%})" if r["confidence"] is not None else ""
        ax.set_title(f"true:{r['true']}\npred:{r['predicted']}{conf}", fontsize=8)
    fig.suptitle("Most confident misclassifications (validation)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_iteration(out_dir: Path, val_metrics: Dict[str, Any],
                   curves: List[Dict[str, Any]]) -> List[str]:
    """Validation figures for one iteration. Returns the files written."""
    written = []
    jobs = (("confusion_matrix.png", lambda p: plot_confusion(val_metrics, p, "Validation confusion matrix")),
            ("learning_curves.png", lambda p: plot_curves(curves, p)),
            ("misclassified.png", lambda p: plot_misclassified(val_metrics, p)))
    for name, fn in jobs:
        try:
            fn(out_dir / name)
            if (out_dir / name).exists():
                written.append(name)
        except Exception as e:      # plots are a convenience; never fail the run on them
            (out_dir / "plot_errors.txt").open("a", encoding="utf-8").write(f"{name}: {e}\n")
    return written


# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #
def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def _per_class_table(m: Dict[str, Any]) -> List[str]:
    lines = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    lines += [f"| {c['class']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |"
              for c in m.get("per_class", [])]
    return lines


def write_final_report(run_dir: Path, config: Dict[str, Any], history: List[ExperimentRecord],
                       best: Optional[ExperimentRecord], test_metrics: Optional[Dict[str, Any]],
                       stop_reason: str, llm_usage: Optional[Dict[str, Any]] = None) -> Path:
    metric = config["selection_metric"]
    L = [f"# Optimisation report", "",
         f"- Task: {config['task']}",
         f"- Mode: {config['mode']} · selection metric (validation): **{metric}**",
         f"- Seed: {config['seed']} · LLMs: " +
         ", ".join(f"{r} = `{m}`" for r, m in config['llm']['models'].items()),
         f"- Iterations run: {len(history)} · stop reason: {stop_reason}", ""]

    if llm_usage:
        L += ["## LLM usage", "", "| role | model | calls | failed | retries | prompt tok | "
              "completion tok | mean latency (s) |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r} | `{u['model']}` | {u['calls']} | {u['failed']} | {u['retries']} | "
              f"{u['prompt_tokens']} | {u['completion_tokens']} | {_fmt(u['mean_latency_s'])} |"
              for r, u in llm_usage.items()]
        L.append("")

    L += ["## Experiment history (validation set)", "",
          f"| iter | experiment | status | val {metric} | val acc | train acc | val loss | fit | Δ vs best (95% CI, McNemar p) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in history:
        c = r.comparison_to_best
        cmp_ = (f"{c['delta']:+.4f} ({c['bootstrap_ci95'][0]:+.3f}, {c['bootstrap_ci95'][1]:+.3f}), "
                f"p={c['mcnemar']['p_value']}" if c else "—")
        mark = " ★" if best and r.iteration == best.iteration else ""
        L.append(f"| {r.iteration} | {r.proposal.experiment_name}{mark} | {r.status} | "
                 f"{_fmt(r.selection_score)} | {_fmt(r.val_metrics.get('accuracy'))} | "
                 f"{_fmt(r.train_metrics.get('accuracy'))} | {_fmt(r.val_metrics.get('loss'))} | "
                 f"{r.analysis.fit_diagnosis if r.analysis else '—'} | {cmp_} |")
    L.append("")

    L += ["## Iteration details", ""]
    for r in history:
        p = r.proposal
        L += [f"### Iteration {r.iteration}: {p.experiment_name}", "",
              f"- Architecture: `{p.architecture}`",
              f"- Hyperparameters: `{json.dumps(p.hyperparameters, default=str)}`",
              f"- Training config: `{json.dumps(p.training_config, default=str)}`",
              f"- Strategist rationale: {p.rationale}"]
        if p.addresses_findings:
            L.append(f"- Responds to: {'; '.join(p.addresses_findings)}")
        if r.error:
            L.append(f"- Error: `{r.error.strip().splitlines()[-1][:200] if r.error.strip() else ''}`")
        if r.analysis:
            a = r.analysis
            L += [f"- Error analysis ({a.fit_diagnosis}): {a.summary}"]
            L += [f"  - Observation: {o}" for o in a.observations]
            L += [f"  - Hypothesis: {h}" for h in a.hypotheses]
            L += [f"  - Recommendation: {x}" for x in a.recommendations]
            if a.significance:
                L.append(f"  - Significance: {a.significance}")
        L += [f"- Artefacts: `{r.artifacts_dir}`", ""]

    L += ["## Selected model", ""]
    if best:
        L += [f"Iteration **{best.iteration}** (`{best.proposal.experiment_name}`), selected by "
              f"validation {metric} = **{_fmt(best.selection_score)}**.", "",
              "### Validation per-class performance", "", *_per_class_table(best.val_metrics), ""]
    else:
        L += ["No experiment completed successfully, so no model was selected.", ""]

    L += ["## Final test evaluation (run once, after selection)", ""]
    if test_metrics:
        L += [f"- Test accuracy: **{_fmt(test_metrics['accuracy'])}**",
              f"- Test macro F1: **{_fmt(test_metrics['macro_f1'])}**",
              f"- Test balanced accuracy: {_fmt(test_metrics['balanced_accuracy'])}",
              f"- Test loss: {_fmt(test_metrics['loss'])}",
              f"- Test samples: {test_metrics['n_samples']}", "",
              *_per_class_table(test_metrics), "",
              "These numbers were computed after the last agent call and were never shown "
              "to the Model Strategist or the Error Analysis agent.", ""]
    else:
        L += ["Not available (no selected model, or the final evaluation failed).", ""]

    md = run_dir / "final_report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    (run_dir / "final_report.json").write_text(json.dumps({
        "config": config, "stop_reason": stop_reason, "llm_usage": llm_usage,
        "best_iteration": best.iteration if best else None,
        "best_validation": best.val_metrics if best else None,
        "test_metrics": test_metrics,
        "history": [r.model_dump() for r in history],
    }, indent=2, default=str), encoding="utf-8")
    return md
