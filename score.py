"""
score.py — DETERMINISTIC evaluation. No LLM touches these numbers.

Two halves:

  1. EVAL_TABULAR_PY / EVAL_IMAGE_PY: a framework-owned script written into the
     workspace as `eval_model.py` and run in the same environment as training
     (so torch / sklearn are present). It loads the saved model and dumps raw
     predictions per split to predictions_<split>.json:
         python eval_model.py train val      # every iteration
         python eval_model.py test           # once, final model, PIPELINE_ALLOW_TEST=1

  2. Host-side functions (numpy only) that turn predictions into metrics:
     accuracy, balanced accuracy, loss, confusion matrix, per-class P/R/F1,
     top confusions, misclassified examples, learning-curve summary, and a paired
     comparison against the current best model (McNemar exact test + bootstrap CI).
     The Error Analysis Agent only INTERPRETS what these functions produce.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SELECTION_METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1")
MAX_MISCLASSIFIED = 15      # examples listed in the report (most confident errors first)
MAX_CM_CLASSES = 20         # include the full confusion matrix in agent prompts up to this


# --------------------------------------------------------------------------- #
# 1. Workspace-side evaluation scripts (dump predictions only)
# --------------------------------------------------------------------------- #
EVAL_TABULAR_PY = '''\
import json, sys
import joblib, numpy as np
import data

model = joblib.load("model.pkl")
names = data.get_class_names()
Xtr, ytr, Xv, yv = data.get_splits()
for split in sys.argv[1:]:
    if split == "test":
        X, y = data.get_test_split()     # raises unless the framework unlocked it
    else:
        X, y = (Xtr, ytr) if split == "train" else (Xv, yv)
    pred = np.asarray(model.predict(X)).astype(int)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X), dtype=float)
            full = np.zeros((len(y), len(names)))
            full[:, np.asarray(model.classes_, dtype=int)] = p
            proba = np.round(full, 6).tolist()
        except Exception:
            proba = None
    ids = data.get_sample_ids(split) if split != "train" else None
    json.dump({"split": split, "class_names": names, "y_true": np.asarray(y).astype(int).tolist(),
               "y_pred": pred.tolist(), "proba": proba, "ids": ids},
              open(f"predictions_{split}.json", "w"))
    print(f"wrote predictions_{split}.json ({len(y)} samples)")
'''

EVAL_IMAGE_PY = '''\
import json, sys
import torch
import data
from preprocess import AdaptiveInput

device = ("cuda" if torch.cuda.is_available() else
          "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
model = torch.load("model.pt", map_location=device, weights_only=False)   # needs preprocess.py
if not isinstance(model, AdaptiveInput):
    sys.exit("model.pt is not wrapped with preprocess.prepare_model(...); the evaluator "
             "cannot know its input size and normalisation.")
model.to(device).eval()
size = model.input_spec["image_size"]   # load at the model's size; the adapter verifies anyway
names = data.get_class_names()

for split in sys.argv[1:]:
    if split == "test":
        loader = data.get_test_loader(image_size=size)  # raises unless the framework unlocked it
    else:
        tl, vl, _ = data.get_loaders(augment=False, image_size=size)
        loader = tl if split == "train" else vl
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)).float(), 1).cpu()
            ys += y.tolist(); ps += p.argmax(1).tolist(); probs += p.tolist()
    ids = data.get_sample_ids(split) if split != "train" else None   # train loader is shuffled
    json.dump({"split": split, "class_names": names, "y_true": ys, "y_pred": ps,
               "proba": [[round(v, 6) for v in r] for r in probs], "ids": ids},
              open(f"predictions_{split}.json", "w"))
    print(f"wrote predictions_{split}.json ({len(ys)} samples)")
'''


def eval_script(mode: str) -> str:
    return EVAL_IMAGE_PY if mode == "image" else EVAL_TABULAR_PY


def load_predictions(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 2. Metrics
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred, k: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    return np.bincount(k * y_true + y_pred, minlength=k * k).reshape(k, k)


def _scores_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    n = cm.sum()
    present = support > 0
    return {
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
    }


def metric_value(y_true, y_pred, k: int, metric: str) -> float:
    return _scores_from_cm(confusion(y_true, y_pred, k))[metric]


def cross_entropy(y_true, proba) -> Optional[float]:
    if proba is None:
        return None
    p = np.asarray(proba, float)
    y = np.asarray(y_true, int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def compute_metrics(pred: Dict[str, Any], detailed: bool = True) -> Dict[str, Any]:
    """All per-split metrics from a predictions_<split>.json payload."""
    names = pred["class_names"]
    k = len(names)
    y, yp = np.asarray(pred["y_true"], int), np.asarray(pred["y_pred"], int)
    cm = confusion(y, yp, k)
    s = _scores_from_cm(cm)
    out: Dict[str, Any] = {
        "n_samples": int(len(y)),
        "accuracy": round(s["accuracy"], 4),
        "balanced_accuracy": round(s["balanced_accuracy"], 4),
        "macro_f1": round(s["macro_f1"], 4),
        "weighted_f1": round(s["weighted_f1"], 4),
        "loss": None if pred.get("proba") is None else round(cross_entropy(y, pred["proba"]), 4),
    }
    if not detailed:
        return out

    out["class_names"] = names
    out["confusion_matrix"] = cm.tolist()
    out["per_class"] = [
        {"class": names[i], "precision": round(float(s["precision"][i]), 4),
         "recall": round(float(s["recall"][i]), 4), "f1": round(float(s["f1"][i]), 4),
         "support": int(s["support"][i])}
        for i in range(k)
    ]
    pairs = [(int(cm[i, j]), i, j) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    out["top_confusions"] = [
        {"true": names[i], "predicted": names[j], "count": c,
         "share_of_true_class": round(c / max(int(cm[i].sum()), 1), 4)}
        for c, i, j in pairs[:5]
    ]
    proba = None if pred.get("proba") is None else np.asarray(pred["proba"], float)
    ids = pred.get("ids")
    wrong = np.flatnonzero(y != yp)
    rows = []
    for idx in wrong:
        conf = float(proba[idx, yp[idx]]) if proba is not None else None
        rows.append({"id": ids[idx] if ids else int(idx), "true": names[y[idx]],
                     "predicted": names[yp[idx]],
                     "confidence": None if conf is None else round(conf, 4)})
    rows.sort(key=lambda r: -(r["confidence"] or 0.0))
    out["n_misclassified"] = int(len(wrong))
    out["misclassified_examples"] = rows[:MAX_MISCLASSIFIED]
    return out


# --------------------------------------------------------------------------- #
# 3. Learning curves
# --------------------------------------------------------------------------- #
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*\d+.*?loss:\s*([0-9.]+).*?"
                       r"train_acc:\s*([0-9.]+).*?val_acc:\s*([0-9.]+)")


def load_curves(workspace: Path, stdout: str) -> List[Dict[str, Any]]:
    """Per-epoch history from train.py's history.json, else parsed from its epoch lines."""
    f = Path(workspace) / "history.json"
    if f.exists():
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass
    return [{"epoch": int(m.group(1)), "train_loss": float(m.group(2)),
             "train_acc": float(m.group(3)), "val_acc": float(m.group(4))}
            for m in map(_EPOCH_RE.search, stdout.splitlines()) if m]


def summarize_curves(curves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic facts about the curves so the agent does not eyeball numbers."""
    if not curves:
        return {"available": False}
    out: Dict[str, Any] = {"available": True, "epochs": len(curves)}
    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        vals = [r[key] for r in curves if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"{key}_first"], out[f"{key}_last"] = round(vals[0], 4), round(vals[-1], 4)
    va = [(r["epoch"], r["val_acc"]) for r in curves if isinstance(r.get("val_acc"), (int, float))]
    if va:
        best_ep, best = max(va, key=lambda t: t[1])
        out["best_val_acc_epoch"], out["best_val_acc"] = best_ep, round(best, 4)
        out["val_acc_peaked_before_last_epoch"] = best_ep < va[-1][0]
    vl = [r.get("val_loss") for r in curves]
    tl = [r.get("train_loss") for r in curves]
    if len(curves) >= 3 and all(isinstance(v, (int, float)) for v in vl[-3:] + tl[-3:]):
        out["val_loss_rising_while_train_loss_falls"] = (vl[-1] > vl[-3] and tl[-1] < tl[-3])
    if "train_acc_last" in out and "val_acc_last" in out:
        out["final_epoch_train_val_acc_gap"] = round(out["train_acc_last"] - out["val_acc_last"], 4)
        # still improving at the end -> more epochs may help
        if len(va) >= 2:
            out["val_acc_still_improving_at_end"] = va[-1][1] > va[-2][1]
    return out


# --------------------------------------------------------------------------- #
# 4. Paired comparison against the current best (same validation samples)
# --------------------------------------------------------------------------- #
def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def compare_to_best(cand: Dict[str, Any], best: Dict[str, Any], metric: str,
                    n_boot: int = 1000, seed: int = 0) -> Optional[Dict[str, Any]]:
    """Is the candidate's difference from the best model real or noise?"""
    if cand.get("ids") != best.get("ids") or cand["y_true"] != best["y_true"]:
        return None   # not the same validation samples in the same order
    k = len(cand["class_names"])
    y = np.asarray(cand["y_true"], int)
    pa, pb = np.asarray(cand["y_pred"], int), np.asarray(best["y_pred"], int)
    a_ok, b_ok = pa == y, pb == y
    cand_only, best_only = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        diffs[i] = metric_value(y[s], pa[s], k, metric) - metric_value(y[s], pb[s], k, metric)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = _mcnemar_exact_p(cand_only, best_only)
    enough = cand_only + best_only >= 10
    return {
        "metric": metric,
        "delta": round(metric_value(y, pa, k, metric) - metric_value(y, pb, k, metric), 4),
        "bootstrap_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "mcnemar": {"candidate_only_correct": cand_only, "best_only_correct": best_only,
                    "p_value": round(p, 4)},
        "evidence": "sufficient" if enough else
                    "insufficient (fewer than 10 discordant samples)",
        "significant_at_0.05": bool(enough and p < 0.05),
    }


# --------------------------------------------------------------------------- #
# 5. The structured report handed to the Error Analysis Agent
# --------------------------------------------------------------------------- #
def build_iteration_report(train_m: Dict[str, Any], val_m: Dict[str, Any],
                           curves: List[Dict[str, Any]], metric: str,
                           comparison: Optional[Dict[str, Any]],
                           previous: List[Dict[str, Any]]) -> Dict[str, Any]:
    gap_acc = round(train_m["accuracy"] - val_m["accuracy"], 4)
    gap_loss = (round(val_m["loss"] - train_m["loss"], 4)
                if val_m.get("loss") is not None and train_m.get("loss") is not None else None)
    val = dict(val_m)
    if len(val.get("class_names", [])) > MAX_CM_CLASSES:
        val.pop("confusion_matrix", None)   # too large for a prompt; top_confusions remain
    return {
        "selection_metric": metric,
        "validation": val,
        "train": train_m,
        "generalization_gap": {"accuracy_train_minus_val": gap_acc,
                               "loss_val_minus_train": gap_loss},
        "learning_curves": curves,
        "learning_curve_summary": summarize_curves(curves),
        "comparison_to_current_best": comparison,
        "previous_experiments": previous,
    }
