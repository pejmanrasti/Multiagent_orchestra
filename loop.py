"""
loop.py — the harness. The three agents are the brains; this file runs the
experiments, computes nothing by itself that an agent could fake, and decides
deterministically when to stop.

Workflow
   1. Model Strategist proposes a baseline
   2. Coder writes train.py
   3. RUN train.py                      (train split; val only for monitoring)
   4. RUN eval_model.py train val       -> raw predictions           (score.py)
   5. deterministic metrics + report    -> validation report          (score.py)
   6. Error Analysis Agent interprets the report
   7. keep-best on the VALIDATION selection metric; patience/budget stop check
   8. Model Strategist reads the analysis + history, proposes the next experiment
   9. Coder implements it ... repeat
  10. select the best model by validation score
  11. RUN eval_model.py test ONCE on the selected checkpoint (agents locked first)
  12. final report                                                    (report.py)

Everything is written to experiments/run_<timestamp>/ (see README).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import agents
import llm
import report
import score
from agents import analysis_agent, coder_agent, history_rows, strategist_agent
from environments import detect_pip_packages, ensure_env, install_into, missing_module
from sandbox import run_script
from schemas import ErrorAnalysis, ExperimentRecord, PipelineResult, StrategyProposal
from workspace_files import DATA_PY, FORBIDDEN_IN_TRAIN, PREPROCESS_PY, SEED_ENV, TEST_FLAG

# ---- policy knobs (override per call in run(), or via env vars) ---------- #
MAX_ITERS = 6
PATIENCE = 2        # stop after this many completed runs with no real improvement
EPS = 0.005         # "real improvement" = selection-metric gain greater than this
MAX_CRASHES = 3     # stop after this many failed attempts in a row
SEED = int(os.environ.get(SEED_ENV, "42"))
SELECTION_METRIC = os.environ.get("SELECTION_METRIC", "macro_f1")
KEEP_CHECKPOINTS = os.environ.get("KEEP_CHECKPOINTS", "1") == "1"   # 0 = keep only best ones


class Reporter:
    """Default reporter: plain prints. Subclass to change how the loop 'speaks'."""
    def plan(self, task, max_iters, patience, metric):
        print(f"TASK: {task}\nBudget: up to {max_iters} experiments, stop after {patience} "
              f"without improvement. Selection: validation {metric}. Test set: used once at the end.\n")
    def proposal(self, i, p: StrategyProposal):
        print(f"[iter {i}] strategist -> {p.experiment_name}: {p.architecture} "
              f"{p.hyperparameters} {p.training_config}")
        print(f"         rationale: {p.rationale}")
    def progress(self, line):
        print(f"         | {line}")
    def result(self, r: ExperimentRecord):
        if not r.ok:
            last = r.error.strip().splitlines()[-1] if r.error.strip() else r.status
            print(f"         {r.status.upper()}: {last[:160]}")
        else:
            print(f"         train acc={r.train_metrics['accuracy']}  val acc={r.val_metrics['accuracy']}  "
                  f"val {_metric_name(r)}={r.selection_score}")
    def analysis(self, a: Optional[ErrorAnalysis]):
        if a:
            print(f"         analysis ({a.fit_diagnosis}): {a.summary}")
            if a.recommendations:
                print(f"         recommends: {a.recommendations[0]}")
    def verdict(self, is_best, score_, stale, patience):
        if is_best:
            print(f"         -> new best (val {score_})")
        else:
            print(f"         -> no real improvement ({stale}/{patience}); keeping best")
    def converged(self, reason):
        print(f"Stopping: {reason}")
    def final_test(self, metrics):
        print(f"\nFINAL TEST (selected model, evaluated once): accuracy={metrics['accuracy']} "
              f"macro_f1={metrics['macro_f1']}")
    def done(self, result: PipelineResult):
        print("\n=== SELECTED MODEL ===")
        if result.best:
            b = result.best
            print(f"iteration {b.iteration} '{b.proposal.experiment_name}': "
                  f"val {result.selection_metric}={b.selection_score}")
        else:
            print("no successful run")
        print(f"Report: {Path(result.run_dir) / 'final_report.md'}")


def _metric_name(r: ExperimentRecord) -> str:
    return r.val_metrics.get("_selection_metric", "score")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _model_file(mode: str) -> str:
    return "model.pt" if mode == "image" else "model.pkl"


def _setup_workspace(ws: Path, data_py: str, mode: str) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "data.py").write_text(data_py, encoding="utf-8")                # dataset loader
    (ws / "eval_model.py").write_text(score.eval_script(mode), encoding="utf-8")
    if mode == "image":   # input adapter (size check/resize + normalisation), see preprocess.py
        (ws / "preprocess.py").write_text(PREPROCESS_PY, encoding="utf-8")
    # Remove files from older versions of the framework (test-set grader/report) and
    # any stale predictions, so nothing test-related sits in the workspace.
    for stale in ("score.py", "report.py", "predictions_test.json",
                  "predictions_train.json", "predictions_val.json"):
        (ws / stale).unlink(missing_ok=True)


def _new_run_dir(root: Path) -> Path:
    base = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path, n = base, 1
    while path.exists():
        path, n = Path(f"{base}_{n}"), n + 1
    path.mkdir(parents=True)
    return path


_ENV_INFO: dict = {}


def _env_info(py: str) -> dict:
    """Interpreter + package versions of the env that actually ran the experiment."""
    if py not in _ENV_INFO:
        code = ("import json, sys, platform, importlib.metadata as m\n"
                "out = {'python': sys.version.split()[0], 'platform': platform.platform(), 'executable': sys.executable}\n"
                "for p in ('torch', 'torchvision', 'scikit-learn', 'numpy', 'joblib', 'pandas'):\n"
                "    try: out[p] = m.version(p)\n"
                "    except Exception: pass\n"
                "print(json.dumps(out))")
        try:
            out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60)
            _ENV_INFO[py] = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception as e:
            _ENV_INFO[py] = {"error": str(e)}
    return _ENV_INFO[py]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _check_code(code: str, mode: str = "tabular") -> str:
    """Return a problem description, or '' if the generated code is acceptable."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"invalid Python ({e.msg}, line {e.lineno}); return the COMPLETE script"
    bad = [t for t in FORBIDDEN_IN_TRAIN if t in code]
    if bad:
        return f"the script references test-set access ({', '.join(bad)}), which is forbidden"
    if mode == "image":
        if "prepare_model" not in code:
            return ("the model is not wrapped with preprocess.prepare_model(...), so its input "
                    "size and normalisation are not handled")
        if "Normalize(" in code:
            return ("the script adds transforms.Normalize; remove it, prepare_model's adapter "
                    "normalises inputs itself")
    return ""


def _generate_code(task, proposal, base_code, mode, previous_error, rep) -> tuple[str, str]:
    """Ask the Coder for code, retrying on invalid or test-touching output."""
    feedback, code = "", ""
    for _ in range(3):
        try:
            code = coder_agent(task, proposal, base_code, mode=mode,
                               previous_error=previous_error, feedback=feedback)
        except Exception as e:          # provider empty / rate-limit / network error
            rep.progress(f"(couldn't get code from the model: {str(e)[:140]})")
            return "", f"code generation failed: {e}"
        feedback = _check_code(code, mode)
        if not feedback:
            return code, ""
        rep.progress(f"(generated code rejected: {feedback}; regenerating…)")
    return code, f"code rejected: {feedback}"


def _write_usage(run_dir: Path) -> dict:
    """Per-call LLM log (role, model, latency, retries, tokens) + per-role summary."""
    summary = llm.usage_summary()
    _write_json(run_dir / "llm_usage.json", {"summary": summary, "calls": llm.USAGE})
    return summary


def _packages(code: str, data_py: str) -> list[str]:
    return sorted(set(detect_pip_packages(code)) | set(detect_pip_packages(data_py)))


# --------------------------------------------------------------------------- #
# one iteration: train -> evaluate on validation -> analyse
# --------------------------------------------------------------------------- #
def _train_and_evaluate(rec: ExperimentRecord, code: str, ctx: dict, rep: Reporter):
    """Fills `rec` in place. Returns validation predictions (or None on failure)."""
    ws, it_dir, mode = ctx["ws"], Path(rec.artifacts_dir), ctx["mode"]
    model_file = _model_file(mode)
    (ws / "train.py").write_text(code, encoding="utf-8")
    for f in (model_file, "history.json", "predictions_train.json", "predictions_val.json"):
        (ws / f).unlink(missing_ok=True)       # never mistake a stale artefact for this run's

    py = ensure_env(_packages(code, ctx["data_py"]))
    _write_json(it_dir / "environment.json", _env_info(py))
    env = {SEED_ENV: str(rec.seed)}

    # 3. train (retry once if a library is still missing)
    rc, out, err = run_script(ws, "train.py", python=py, on_line=rep.progress, env=env)
    if rc != 0 and (miss := missing_module(err)):
        try:
            install_into(py, [miss])
            rc, out, err = run_script(ws, "train.py", python=py, on_line=rep.progress, env=env)
        except Exception as e:
            err += f"\n[auto-install failed: {e}]"
    (it_dir / "train_log.txt").write_text(out + "\n--- stderr ---\n" + err, encoding="utf-8")
    if rc != 0 or not (ws / model_file).exists():
        rec.status = "train_failed"
        rec.error = ((err or out)[-1500:] if rc != 0 else f"train.py did not save {model_file}")
        return None

    # 4. deterministic prediction dump on train + validation (never test)
    rc2, out2, err2 = run_script(ws, "eval_model.py", python=py, args=("train", "val"), env=env)
    if rc2 != 0:
        rec.status, rec.error = "eval_failed", ("evaluation failed: " + (err2 or out2))[-1500:]
        return None
    train_pred = score.load_predictions(ws / "predictions_train.json")
    val_pred = score.load_predictions(ws / "predictions_val.json")
    shutil.copy2(ws / "predictions_val.json", it_dir / "predictions_val.json")

    # 5. metrics + structured report
    metric = ctx["metric"]
    train_m = score.compute_metrics(train_pred, detailed=False)
    val_m = score.compute_metrics(val_pred)
    curves = score.load_curves(ws, out)
    best_pred = ctx["best_val_pred"]
    cmp_ = score.compare_to_best(val_pred, best_pred, metric, seed=rec.seed) if best_pred else None
    rec.status = "ok"
    rec.train_metrics, rec.val_metrics = train_m, {**val_m, "_selection_metric": metric}
    rec.selection_score, rec.comparison_to_best = val_m[metric], cmp_

    analysis_input = score.build_iteration_report(
        train_m, val_m, curves, metric, cmp_, history_rows(ctx["history"], metric))
    _write_json(it_dir / "validation_report.json", analysis_input)
    _write_json(it_dir / "learning_curves.json", curves)
    report.plot_iteration(it_dir, val_m, curves)

    # 6. Error Analysis Agent interprets the report
    try:
        rec.analysis = analysis_agent(analysis_input, rec.proposal, ctx["task"])
        _write_json(it_dir / "analysis.json", rec.analysis.model_dump())
    except Exception as e:
        rep.progress(f"(error analysis agent failed: {str(e)[:160]})")
        _write_json(it_dir / "analysis.json", {"error": str(e)})
    return val_pred


def _final_test(ws: Path, run_dir: Path, best: ExperimentRecord, best_code: str,
                data_py: str, mode: str, rep: Reporter) -> Optional[dict]:
    """Evaluate the SELECTED checkpoint once on the untouched test split."""
    final_dir = run_dir / "final"
    marker = final_dir / ".test_evaluated"
    if marker.exists():
        raise RuntimeError("The test set has already been evaluated for this run.")
    final_dir.mkdir(exist_ok=True)

    model_file = _model_file(mode)
    shutil.copy2(Path(best.artifacts_dir) / model_file, ws / model_file)   # no retraining
    (ws / "train.py").write_text(best_code, encoding="utf-8")
    shutil.copy2(ws / model_file, final_dir / model_file)
    (final_dir / "train.py").write_text(best_code, encoding="utf-8")
    if mode == "image":
        shutil.copy2(ws / "preprocess.py", final_dir / "preprocess.py")

    rep.progress("Evaluating the selected model once on the held-out test set...")
    py = ensure_env(_packages(best_code, data_py))
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    rc, out, err = run_script(ws, "eval_model.py", python=py, args=("test",),
                              env={TEST_FLAG: "1", SEED_ENV: str(best.seed)})
    if rc != 0:
        rep.progress("(final test evaluation failed: " + (err or out)[-300:] + ")")
        return None
    shutil.move(str(ws / "predictions_test.json"), final_dir / "predictions_test.json")
    metrics = score.compute_metrics(score.load_predictions(final_dir / "predictions_test.json"))
    _write_json(final_dir / "test_metrics.json", metrics)
    try:
        report.plot_confusion(metrics, final_dir / "test_confusion_matrix.png",
                              "Test confusion matrix (selected model)")
    except Exception as e:
        rep.progress(f"(test confusion plot skipped: {e})")
    rep.final_test(metrics)
    return metrics


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run(task: str, data_py: str = DATA_PY, workspace: str = "workspace",
        reporter: Reporter | None = None,
        initial_action: str = "Establish a simple baseline model.",
        preference: str = "", mode: str = "tabular", *,
        selection_metric: str | None = None, max_iters: int = MAX_ITERS,
        patience: int = PATIENCE, seed: int | None = None,
        experiments_dir: str = "experiments") -> PipelineResult:
    if not llm.models_configured():
        raise RuntimeError("Each agent needs a user-selected model; none is chosen "
                           "automatically. Run model_selection.select_models(...) first.")
    metric = selection_metric or SELECTION_METRIC
    if metric not in score.SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {score.SELECTION_METRICS}")
    seed = SEED if seed is None else seed
    rep = reporter or Reporter()
    ws = Path(workspace)
    _setup_workspace(ws, data_py, mode)
    run_dir = _new_run_dir(Path(experiments_dir))

    config = {
        "task": task, "mode": mode, "selection_metric": metric, "seed": seed,
        "max_iters": max_iters, "patience": patience, "eps": EPS, "max_crashes": MAX_CRASHES,
        "preference": preference, "initial_action": initial_action,
        "llm": {"models": llm.selected_models(),
                "base_url": llm.BASE_URL, "max_retries": llm.MAX_RETRIES},
        "data_py_sha256": hashlib.sha256(data_py.encode()).hexdigest(),
        "isolation": os.environ.get("ISOLATION", "venv"),
        "host": {"python": sys.version.split()[0], "platform": platform.platform()},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "run_config.json", config)
    (run_dir / "data.py").write_text(data_py, encoding="utf-8")

    agents.unlock()
    llm.reset_usage()
    rep.plan(task, max_iters, patience, metric)
    rep.progress("Agent models: " + ", ".join(
        f"{llm.ROLE_LABELS[r]} = {m}" for r, m in llm.selected_models().items()))
    history: list[ExperimentRecord] = []
    best: ExperimentRecord | None = None
    best_code, best_val_pred = "", None
    stale = crashes = 0
    stop_reason = f"iteration budget ({max_iters}) used"
    ctx = {"ws": ws, "mode": mode, "data_py": data_py, "metric": metric, "task": task,
           "history": history, "best_val_pred": None}
    common = dict(mode=mode, metric=metric, preference=preference)

    # 1. the Strategist proposes the baseline
    try:
        proposal = strategist_agent(task, [], None, initial_guidance=initial_action,
                                    remaining=max_iters, **common)
    except Exception as e:
        proposal, stop_reason = None, f"strategist failed: {str(e)[:200]}"
        rep.converged(stop_reason)

    for i in range(max_iters if proposal else 0):
        it_dir = run_dir / f"iter_{i:02d}"
        it_dir.mkdir()
        _write_json(it_dir / "proposal.json", proposal.model_dump())
        rep.proposal(i, proposal)
        rec = ExperimentRecord(iteration=i, proposal=proposal, seed=seed, artifacts_dir=str(it_dir))
        t0 = time.time()

        # 2. the Coder implements the proposal (starting from the best code so far)
        prev_error = history[-1].error if history and not history[-1].ok else ""
        code, gen_error = _generate_code(task, proposal, best_code, mode, prev_error, rep)
        if code:
            (it_dir / "train.py").write_text(code, encoding="utf-8")
        val_pred = None
        if gen_error:
            rec.status, rec.error = "codegen_failed", gen_error
        else:
            # 3-6. train, evaluate on validation, analyse
            ctx["best_val_pred"] = best_val_pred
            val_pred = _train_and_evaluate(rec, code, ctx, rep)

        # 7. keep-best on the validation selection metric
        if rec.ok:
            rec.is_best = best is None or rec.selection_score > best.selection_score + EPS
            if rec.is_best:
                best, best_code, best_val_pred, stale = rec, code, val_pred, 0
            else:
                stale += 1
            crashes = 0
            if KEEP_CHECKPOINTS or rec.is_best:
                model_file = _model_file(mode)
                shutil.copy2(ws / model_file, it_dir / model_file)
                if mode == "image":     # the checkpoint needs it to load
                    shutil.copy2(ws / "preprocess.py", it_dir / "preprocess.py")
        else:
            crashes += 1
        rec.duration_s = round(time.time() - t0, 1)
        _write_json(it_dir / "record.json", rec.model_dump())
        _write_usage(run_dir)
        history.append(rec)
        rep.result(rec)
        if rec.ok:
            rep.analysis(rec.analysis)
            rep.verdict(rec.is_best, rec.selection_score, stale, patience)

        # stopping criteria (decided in code)
        if crashes >= MAX_CRASHES:
            stop_reason = f"{MAX_CRASHES} failed attempts in a row"
            rep.converged(stop_reason)
            break
        if stale >= patience:
            stop_reason = f"no improvement > {EPS} in validation {metric} for {patience} experiments"
            rep.converged(stop_reason)
            break
        if i == max_iters - 1:
            rep.converged(stop_reason)
            break

        # 8. the Strategist reads the analysis + history and proposes the next experiment
        try:
            proposal = strategist_agent(task, history, rec.analysis, best=best,
                                        remaining=max_iters - i - 1, **common)
        except Exception as e:
            stop_reason = f"strategist failed: {str(e)[:200]}"
            rep.converged(stop_reason)
            break
        if proposal.stop:
            _write_json(run_dir / "strategist_stop.json", proposal.model_dump())
            stop_reason = f"strategist recommended stopping: {proposal.rationale}"
            rep.converged(stop_reason)
            break

    # 10-12. select, test ONCE, report. No agent may run from here on.
    agents.lock()
    test_metrics = _final_test(ws, run_dir, best, best_code, data_py, mode, rep) if best else None
    usage = _write_usage(run_dir)
    report.write_final_report(run_dir, config, history, best, test_metrics, stop_reason, usage)
    result = PipelineResult(run_dir=str(run_dir), selection_metric=metric, best=best,
                            test_metrics=test_metrics, history=history, stop_reason=stop_reason)
    rep.done(result)
    return result
