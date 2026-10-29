"""
End-to-end and unit checks for the three-agent pipeline (tabular digits demo,
current interpreter, no API key: the LLM provider is faked in tests/fakes.py):

    ISOLATION=none python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("ISOLATION", "none")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402

import agents                                         # noqa: E402
import llm                                            # noqa: E402
import loop                                           # noqa: E402
import sandbox                                        # noqa: E402
import score                                          # noqa: E402
from workspace_files import DATA_PY, TEST_FLAG        # noqa: E402

TASK = "Classify sklearn digits (10 classes)."


class Recorder:
    """Wraps sandbox.run_script and the agents to record what happened, in order."""

    def __init__(self):
        self.events = []
        self._orig_run = loop.run_script
        self._orig = {n: getattr(loop, n) for n in ("strategist_agent", "coder_agent",
                                                     "analysis_agent")}

    def __enter__(self):
        def run_script(ws, script, *a, args=(), env=None, **k):
            self.events.append(("script", script, tuple(args), dict(env or {}),
                                (Path(ws) / "predictions_test.json").exists()))
            return self._orig_run(ws, script, *a, args=args, env=env, **k)
        loop.run_script = run_script
        for name, fn in self._orig.items():
            def wrapped(*a, _fn=fn, _name=name, **k):
                out = _fn(*a, **k)
                self.events.append(("agent", _name, a, k, out))
                return out
            setattr(loop, name, wrapped)
        return self

    def __exit__(self, *exc):
        loop.run_script = self._orig_run
        for name, fn in self._orig.items():
            setattr(loop, name, fn)

    def agents_called(self):
        return [e[1] for e in self.events if e[0] == "agent"]


def _run(tmp, **kw):
    return loop.run(TASK, workspace=str(Path(tmp) / "ws"),
                    experiments_dir=str(Path(tmp) / "exp"), reporter=_Quiet(), **kw)


class _Quiet(loop.Reporter):
    def __getattribute__(self, name):
        if name in ("plan", "proposal", "progress", "result", "analysis", "verdict",
                    "converged", "final_test", "done"):
            return lambda *a, **k: None
        return super().__getattribute__(name)


# --------------------------------------------------------------------------- #
class TestDeterministicMetrics(unittest.TestCase):
    def test_metrics_match_sklearn(self):
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                     confusion_matrix, f1_score, log_loss)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, 300)
        p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 4, 300))
        proba = rng.dirichlet(np.ones(4), 300)
        m = score.compute_metrics({"class_names": list("abcd"), "y_true": y.tolist(),
                                   "y_pred": p.tolist(), "proba": proba.tolist(), "ids": None})
        self.assertAlmostEqual(m["accuracy"], accuracy_score(y, p), places=4)
        self.assertAlmostEqual(m["balanced_accuracy"], balanced_accuracy_score(y, p), places=4)
        self.assertAlmostEqual(m["macro_f1"], f1_score(y, p, average="macro"), places=4)
        self.assertAlmostEqual(m["weighted_f1"], f1_score(y, p, average="weighted"), places=4)
        self.assertAlmostEqual(m["loss"], log_loss(y, proba, labels=range(4)), places=3)
        self.assertEqual(m["confusion_matrix"], confusion_matrix(y, p).tolist())

    def test_mcnemar_and_comparison(self):
        self.assertEqual(score._mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(score._mcnemar_exact_p(0, 10), 2 / 1024, places=6)
        y = [0, 1] * 50
        base = {"class_names": ["a", "b"], "y_true": y, "ids": list(range(100))}
        good = dict(base, y_pred=y)
        bad = dict(base, y_pred=[1 - v if i < 30 else v for i, v in enumerate(y)])
        c = score.compare_to_best(good, bad, "accuracy")
        self.assertEqual(c["mcnemar"]["candidate_only_correct"], 30)
        self.assertTrue(c["significant_at_0.05"])
        self.assertIsNone(score.compare_to_best(good, dict(bad, ids=list(range(1, 101))), "accuracy"))


class TestTestSetIsolation(unittest.TestCase):
    def test_data_py_locks_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data.py").write_text(DATA_PY)
            probe = ("import data\n"
                     "Xtr, ytr, Xv, yv = data.get_splits()\n"
                     "try:\n    data.get_test_split(); print('LEAK')\n"
                     "except PermissionError:\n    print('LOCKED')\n")
            (Path(tmp) / "probe.py").write_text(probe)
            os.environ[TEST_FLAG] = "1"          # even if set in the parent process ...
            try:
                rc, out, err = sandbox.run_script(Path(tmp), "probe.py")
            finally:
                os.environ.pop(TEST_FLAG)
            self.assertEqual(rc, 0, err)
            self.assertIn("LOCKED", out)         # ... the sandbox strips it
            rc, out, _ = sandbox.run_script(Path(tmp), "probe.py", env={TEST_FLAG: "1"})
            self.assertIn("LEAK", out)           # only an explicit unlock opens it

    def test_generated_code_touching_test_is_rejected(self):
        self.assertIn("forbidden", loop._check_code("from data import get_test_split\n"))
        self.assertIn("forbidden", loop._check_code(f"import os; os.environ['{TEST_FLAG}']='1'\n"))
        self.assertIn("invalid", loop._check_code("def f(:\n"))
        self.assertEqual(loop._check_code("from data import get_splits\n"), "")

    def test_image_code_must_use_input_adapter(self):
        self.assertIn("prepare_model", loop._check_code("import torch\n", "image"))
        self.assertIn("Normalize", loop._check_code(
            "from preprocess import prepare_model\nT.Normalize([0.5], [0.5])\n", "image"))
        self.assertEqual(loop._check_code(
            "from preprocess import prepare_model\nm = prepare_model(n, image_size=160)\n", "image"), "")


MODELS = {"strategist": "a/strat:free", "coder": "b/coder:free", "analyst": "c/analyst:free"}


class TestPipeline(unittest.TestCase):
    """The real agents (fake provider), real training and evaluation."""

    @classmethod
    def setUpClass(cls):
        from fakes import FakeProvider
        llm.configure_models(MODELS, free_models=MODELS.values())
        cls.tmp = tempfile.TemporaryDirectory()
        try:
            with FakeProvider(), Recorder() as rec:
                cls.result = _run(cls.tmp.name, max_iters=4, patience=5)
        finally:
            llm.clear_models()
        cls.events = rec.events

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_three_agents_and_order(self):
        called = [e[1] for e in self.events if e[0] == "agent"]
        self.assertEqual(called[:3], ["strategist_agent", "coder_agent", "analysis_agent"])
        self.assertEqual(called[3], "strategist_agent")
        n_ok = sum(r.ok for r in self.result.history)
        self.assertEqual(called.count("analysis_agent"), n_ok)   # analysis after EVERY run
        self.assertEqual(called.count("coder_agent"), len(self.result.history))

    def test_strategist_receives_analysis(self):
        strat = [e for e in self.events if e[0] == "agent" and e[1] == "strategist_agent"]
        analyses = [e[4] for e in self.events if e[0] == "agent" and e[1] == "analysis_agent"]
        for call, prev_analysis in zip(strat[1:], analyses):
            self.assertIs(call[2][2], prev_analysis)   # positional arg `analysis`

    def test_coder_follows_strategist(self):
        for r in self.result.history:
            code = (Path(r.artifacts_dir) / "train.py").read_text()
            self.assertIn(r.proposal.architecture.rsplit(".", 1)[1], code)
            self.assertIn(f"C={r.proposal.hyperparameters['C']}", code)

    def test_test_set_used_exactly_once_after_agents(self):
        test_runs = [i for i, e in enumerate(self.events)
                     if e[0] == "script" and (TEST_FLAG in e[3] or "test" in e[2])]
        self.assertEqual(len(test_runs), 1)
        last_agent = max(i for i, e in enumerate(self.events) if e[0] == "agent")
        self.assertGreater(test_runs[0], last_agent)
        # no test predictions existed in the workspace during optimisation
        self.assertFalse(any(e[4] for e in self.events[:test_runs[0]] if e[0] == "script"))
        with self.assertRaises(RuntimeError):
            agents.strategist_agent(TASK, [], None)        # agents locked afterwards

    def test_selection_by_validation_metric(self):
        ok = [r for r in self.result.history if r.ok]
        # keep-best: a run only replaces the best when it beats it by more than EPS
        best = None
        for r in ok:
            if best is None or r.selection_score > best.selection_score + loop.EPS:
                best = r
        self.assertEqual(self.result.best.iteration, best.iteration)
        self.assertLessEqual(max(r.selection_score for r in ok),
                             self.result.best.selection_score + loop.EPS)
        for r in ok:
            self.assertEqual(r.selection_score, r.val_metrics["macro_f1"])
            self.assertNotIn("test", json.dumps(r.val_metrics).lower().replace("_selection", ""))

    def test_tracking_artifacts(self):
        run_dir = Path(self.result.run_dir)
        self.assertTrue((run_dir / "run_config.json").exists())
        self.assertIn("seed", json.loads((run_dir / "run_config.json").read_text()))
        for r in self.result.history:
            d = Path(r.artifacts_dir)
            for f in ("proposal.json", "train.py", "train_log.txt", "environment.json",
                      "validation_report.json", "analysis.json", "record.json", "model.pkl"):
                self.assertTrue((d / f).exists(), f"{d.name}/{f} missing")
            rep = json.loads((d / "validation_report.json").read_text())
            for key in ("confusion_matrix", "per_class", "top_confusions", "misclassified_examples"):
                self.assertIn(key, rep["validation"])
        self.assertTrue((run_dir / "final" / "test_metrics.json").exists())
        md = (run_dir / "final_report.md").read_text()
        self.assertIn("Final test evaluation", md)
        self.assertIsNotNone(self.result.test_metrics)


class TestAgentWiring(unittest.TestCase):
    """With a fake provider: all three LLM roles are called with the right
    inputs and their own models, the analysis reaches the Strategist, and no prompt
    contains test results."""

    def test_agent_prompts(self):
        from fakes import FakeProvider
        models = MODELS
        llm.configure_models(models, free_models=models.values())
        try:
            with FakeProvider() as fp, tempfile.TemporaryDirectory() as tmp:
                result = _run(tmp, max_iters=3, patience=5)
                usage = json.loads((Path(result.run_dir) / "llm_usage.json").read_text())
                config = json.loads((Path(result.run_dir) / "run_config.json").read_text())
        finally:
            llm.clear_models()

        strat = [u for m, u in fp.by_system(agents.STRATEGIST_SYS)]
        analyst = [u for m, u in fp.by_system(agents.ANALYST_SYS)]
        self.assertTrue(strat and analyst and fp.by_system(agents.CODER_SYS))
        self.assertEqual(len(analyst), 3)
        self.assertIn("ANALYSIS-MARKER-1", strat[1])
        self.assertIn("ANALYSIS-MARKER-2", strat[2])
        self.assertIn("confusion_matrix", analyst[0])
        self.assertIn("per_class", analyst[0])
        self.assertIn("comparison_to_current_best", analyst[1])
        test_json = json.dumps(result.test_metrics)
        for _, _, u in fp.calls:
            self.assertNotIn("predictions_test", u)
            self.assertNotIn(test_json, u)
        self.assertEqual(len(result.history), 3)
        self.assertIsNotNone(result.test_metrics)
        # each agent used exactly its own model; selection + usage are logged
        for role, system in (("strategist", agents.STRATEGIST_SYS), ("coder", agents.CODER_SYS),
                             ("analyst", agents.ANALYST_SYS)):
            self.assertEqual({m for m, _ in fp.by_system(system)}, {models[role]})
            self.assertEqual(usage["summary"][role]["model"], models[role])
            self.assertGreater(usage["summary"][role]["total_tokens"], 0)
        self.assertEqual(config["llm"]["models"], models)
        self.assertTrue(all(c["latency_s"] is not None for c in usage["calls"]))

    def test_refuses_to_start_without_selected_models(self):
        llm.clear_models()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "user-selected"):
            _run(tmp, max_iters=1)


if __name__ == "__main__":
    unittest.main()
