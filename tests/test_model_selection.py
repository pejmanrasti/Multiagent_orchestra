"""
User-controlled LLM selection (model_selection.py), the provider client's
reliability (llm.py), and the interactive flow through chat.py — all offline,
with a fake catalogue and a fake provider.

    ISOLATION=none python -m unittest tests.test_model_selection -v
"""
import builtins
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

os.environ.setdefault("ISOLATION", "none")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import llm                                    # noqa: E402
import model_selection as ms                  # noqa: E402
from fakes import FakeProvider                # noqa: E402
from schemas import ErrorAnalysis             # noqa: E402

CATALOG_PAYLOAD = {"data": [
    {"id": "a/strat:free", "name": "Model A", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "b/coder:free", "name": "Model B", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "c/analyst:free", "name": "Model C", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "p/paid-model", "name": "Paid Model",
     "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    {"id": "openrouter/free", "name": "Free Router", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "m/music:free", "name": "Music", "pricing": {"prompt": "0", "completion": "0"},
     "architecture": {"output_modalities": ["audio"]}},
]}
CATALOG = ms.parse_catalog(CATALOG_PAYLOAD)


class Script:
    """Feeds answers to `ask` and records everything said."""

    def __init__(self, answers):
        self.answers, self.questions, self.said = list(answers), [], []

    def ask(self, q):
        self.questions.append(q)
        if not self.answers:
            raise AssertionError(f"unexpected question: {q}")
        return self.answers.pop(0)

    def say(self, text=""):
        self.said.append(text)

    @property
    def text(self):
        return "\n".join(self.said)


class SelectionTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "cfg.json"
        self.env = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"})
        self.env.start()
        self.fp = FakeProvider().__enter__()
        llm.clear_models()
        llm.reset_usage()

    def tearDown(self):
        self.fp.__exit__()
        self.env.stop()
        llm.clear_models()
        self.tmp.cleanup()

    def select(self, answers, **kw):
        s = Script(answers)
        out = ms.select_models(s.ask, s.say, catalog_fetcher=lambda: CATALOG,
                               config_path=self.cfg, **kw)
        return out, s


class TestCatalog(unittest.TestCase):
    def test_free_detection_and_listing(self):
        self.assertTrue(CATALOG["a/strat:free"].free)
        self.assertFalse(CATALOG["p/paid-model"].free)
        listed = [m.id for m in ms.free_models(CATALOG)]
        self.assertEqual(listed, ["a/strat:free", "b/coder:free", "c/analyst:free"])  # no router/audio
        self.assertTrue(CATALOG["openrouter/free"].is_router)
        bad_price = ms.parse_catalog({"data": [{"id": "x/y", "pricing": {"prompt": "-1"}}]})
        self.assertFalse(bad_price["x/y"].free)          # unknown price is treated as paid

    def test_validation(self):
        self.assertEqual(ms.validate_model("a/strat:free", CATALOG), "")
        self.assertIn("not in the current", ms.validate_model("nope/model", CATALOG))
        self.assertIn("text output", ms.validate_model("m/music:free", CATALOG))
        self.assertIn("no model", ms.validate_model("", CATALOG))


class TestInteractiveSelection(SelectionTestBase):
    def test_different_models_per_agent(self):
        models, s = self.select(["1", "b/coder:free", "c/analyst:free", "y"])
        self.assertEqual(models, {"strategist": "a/strat:free", "coder": "b/coder:free",
                                  "analyst": "c/analyst:free"})
        self.assertEqual(llm.selected_models(), models)
        for r in llm.ROLES:
            self.assertEqual(llm.model_for(r), models[r])
        self.assertIn("Available free OpenRouter models:", s.text)
        self.assertIn("1. Model A (a/strat:free)", s.text)
        self.assertIn("Selected configuration:", s.text)
        self.assertIn("Start optimization?", s.questions[-1])
        self.assertEqual(json.loads(self.cfg.read_text()), {"agent_models": models})
        self.assertEqual({c["model"] for c in llm.USAGE}, set(models.values()))  # each tested once

    def test_same_model_for_all_agents(self):
        models, _ = self.select(["a/strat:free"] * 3 + ["y"])
        self.assertEqual(set(models.values()), {"a/strat:free"})
        self.assertEqual(len(llm.USAGE), 1)               # one connection check per distinct model

    def test_no_default_model(self):
        self.assertFalse(llm.models_configured())
        with self.assertRaisesRegex(RuntimeError, "never chosen automatically"):
            llm.model_for("coder")
        models, s = self.select(["", "", "1", "1", "1", "y"])   # Enter twice -> asked again
        self.assertEqual(s.text.count("Rejected: no model entered"), 2)
        self.assertEqual(models["strategist"], "a/strat:free")
        # nothing at all is selected if the user stops at the end
        llm.clear_models()
        models, _ = self.select(["n", "1", "1", "1", "n", "n"])   # decline reuse, then stop
        self.assertIsNone(models)
        self.assertFalse(llm.models_configured())

    def test_invalid_identifiers_rejected(self):
        models, s = self.select(["made/up-model", "9", "m/music:free", "a/strat:free",
                                 "1", "1", "y"])
        self.assertIn("not in the current OpenRouter model list", s.text)
        self.assertIn("9 is not a number from the list", s.text)
        self.assertIn("does not produce text output", s.text)
        self.assertEqual(models["strategist"], "a/strat:free")

    def test_paid_model_requires_explicit_yes(self):
        # decline ('y' is not 'yes') -> back to selection, then pick free models
        models, s = self.select(["p/paid-model", "1", "1", "y", "1", "1", "1", "y"])
        self.assertIn("WARNING: these models are NOT free", s.text)
        self.assertIn("$3.00 / 1M input tokens", s.text)
        self.assertNotIn("p/paid-model", models.values())
        self.assertFalse(any(c[0] == "p/paid-model" for c in self.fp.calls))  # never called
        # explicit 'yes' allows it
        llm.clear_models()
        models, _ = self.select(["n", "p/paid-model", "1", "1", "yes", "y"])
        self.assertEqual(models["strategist"], "p/paid-model")

    def test_paid_model_cannot_be_configured_or_called_without_approval(self):
        with self.assertRaises(PermissionError):
            llm.configure_models({"strategist": "p/paid-model", "coder": "a", "analyst": "a"},
                                 free_models=["a"])
        with self.assertRaises(PermissionError):
            llm._call("coder", "p/paid-model", "s", "u")
        self.assertEqual(self.fp.calls, [])

    def test_router_alias_needs_confirmation(self):
        models, s = self.select(["openrouter/free", "1", "1", "no", "1", "1", "1", "y"])
        self.assertIn("OpenRouter decides which underlying model answers", s.text)
        self.assertNotIn("openrouter/free", models.values())

    def test_saved_configuration_reused_only_after_confirmation(self):
        ms.save({"strategist": "c/analyst:free", "coder": "b/coder:free",
                 "analyst": "c/analyst:free"}, self.cfg)
        models, s = self.select(["y", "y"])
        self.assertIn("Reuse this configuration?", s.questions[0])
        self.assertEqual(models["strategist"], "c/analyst:free")
        llm.clear_models()
        models, s = self.select(["", "n", "1", "1", "1", "y"])  # Enter is not a confirmation
        self.assertIn("Please answer y or n.", s.text)
        self.assertEqual(models["strategist"], "a/strat:free")

    def test_saved_model_no_longer_available(self):
        ms.save({"strategist": "gone/model", "coder": "b/coder:free",
                 "analyst": "c/analyst:free"}, self.cfg)
        models, s = self.select(["1", "1", "1", "y"])
        self.assertIn("can't be reused", s.text)
        self.assertEqual(models["strategist"], "a/strat:free")

    def test_failed_model_check_never_substitutes(self):
        def broken(model, system, user):
            raise type("NotFoundError", (Exception,), {"status_code": 404})("no endpoints")
        llm._complete = broken
        models, s = self.select(["1", "1", "1", "n"])
        self.assertIsNone(models)
        self.assertIn("Nothing is switched automatically", s.text)
        self.assertFalse(llm.models_configured())

    def test_missing_api_key_is_asked_for(self):
        """No key in the environment: the user is asked (hidden input), the key is checked,
        a bad key is rejected, and the good one is kept in memory only."""
        secrets = iter(["sk-bad", "sk-or-good"])
        checked = []

        def checker(key):
            checked.append(key)
            return (True, "ok") if key == "sk-or-good" else (False, "not recognised")

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_API_KEY": ""}), \
                mock.patch.object(ms, "check_api_key", checker):
            try:
                models, s = self.select(["y", "1", "1", "1", "y"],
                                        ask_secret=lambda q: next(secrets))
                self.assertEqual(checked, ["sk-bad", "sk-or-good"])
                self.assertIn("That key was rejected", s.text)
                self.assertIn("Try another key?", s.questions[0])
                self.assertEqual(llm.api_key(), "sk-or-good")
                self.assertNotIn("sk-or-good", os.environ.get("OPENROUTER_API_KEY", ""))
                self.assertNotIn("sk-or-good", self.cfg.read_text())     # never saved
                self.assertIsNotNone(models)
            finally:
                llm.set_session_key(None)

    def test_cancelling_the_key_prompt(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_API_KEY": ""}):
            models, s = self.select([], ask_secret=lambda q: "")
            self.assertIsNone(llm.api_key())
        self.assertIsNone(models)
        self.assertIn("No OpenRouter API key was found", s.text)

    def test_unverifiable_key_needs_confirmation(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_API_KEY": ""}), \
                mock.patch.object(ms, "check_api_key", lambda k: (None, "timeout")):
            try:
                ok = ms.ensure_api_key(Script(["n", "n"]).ask, Script([]).say,
                                       ask_secret=lambda q: "sk-x")
                self.assertFalse(ok)
                self.assertIsNone(llm.api_key())
                ok = ms.ensure_api_key(Script(["y"]).ask, Script([]).say,
                                       ask_secret=lambda q: "sk-x")
                self.assertTrue(ok)
            finally:
                llm.set_session_key(None)

    def test_api_key_is_not_passed_to_generated_code(self):
        import sandbox
        llm.set_session_key("sk-session")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "probe.py").write_text(
                    "import os\nprint([k for k in ('OPENROUTER_API_KEY','LLM_API_KEY') if os.environ.get(k)])\n")
                rc, out, _ = sandbox.run_script(Path(tmp), "probe.py")
            self.assertEqual(out.strip(), "[]")          # env key (set in setUp) stripped too
        finally:
            llm.set_session_key(None)


class _Err(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.response = mock.Mock(headers={"retry-after": retry_after} if retry_after else {})


class TestClientReliability(unittest.TestCase):
    def setUp(self):
        llm.configure_models({r: "a/strat:free" for r in llm.ROLES}, free_models=["a/strat:free"])
        llm.reset_usage()
        self.sleeps = []
        self._sleep, llm._sleep = llm._sleep, self.sleeps.append
        self._complete = llm._complete

    def tearDown(self):
        llm._sleep, llm._complete = self._sleep, self._complete
        llm.clear_models()

    def _seq(self, *outcomes):
        it = iter(outcomes)

        def fake(model, system, user):
            o = next(it)
            if isinstance(o, Exception):
                raise o
            return o, {"served_model": model, "prompt_tokens": 10, "completion_tokens": 5,
                       "total_tokens": 15}
        llm._complete = fake

    def test_exponential_backoff_on_rate_limit(self):
        self._seq(_Err(429), _Err(503), llm.EmptyResponse("empty"), "print('hi')")
        self.assertEqual(llm.call_text("coder", "s", "u"), "print('hi')")
        self.assertEqual(len(self.sleeps), 3)
        self.assertLess(self.sleeps[0], self.sleeps[2])            # grows exponentially
        row = llm.USAGE[-1]
        self.assertEqual((row["attempts"], row["retries"], row["ok"]), (4, 3, True))
        self.assertEqual(row["total_tokens"], 15)

    def test_retry_after_header_is_honoured(self):
        self._seq(_Err(429, retry_after="7"), "ok")
        llm.call_text("coder", "s", "u")
        self.assertEqual(self.sleeps, [7.0])

    def test_non_retryable_errors_fail_fast(self):
        self._seq(_Err(401))
        with self.assertRaisesRegex(RuntimeError, "Coder Agent call to 'a/strat:free' failed"):
            llm.call_text("coder", "s", "u")
        self.assertEqual(self.sleeps, [])
        self.assertFalse(llm.USAGE[-1]["ok"])

    def test_gives_up_after_max_retries(self):
        self._seq(*[_Err(429)] * (llm.MAX_RETRIES + 1))
        with self.assertRaises(RuntimeError):
            llm.call_text("coder", "s", "u")
        self.assertEqual(len(self.sleeps), llm.MAX_RETRIES)

    def test_malformed_json_is_retried_with_feedback(self):
        good = json.dumps({"fit_diagnosis": "overfitting", "summary": "s"})
        self._seq("not json at all", '{"observations": "wrong type"}', good)
        a = llm.call_json("analyst", "s", "u", ErrorAnalysis)
        self.assertEqual(a.fit_diagnosis, "overfitting")
        self.assertEqual(len(llm.USAGE), 3)
        self.assertIn("not valid JSON", llm.USAGE[0]["json_error"])
        self.assertIn("schema violations", llm.USAGE[1]["json_error"])

    def test_malformed_json_gives_up(self):
        self._seq("x", "y", "z")
        with self.assertRaisesRegex(RuntimeError, "did not return valid ErrorAnalysis"):
            llm.call_json("analyst", "s", "u", ErrorAnalysis)

    def test_usage_summary(self):
        self._seq("a", "b")
        llm.call_text("coder", "s", "u"); llm.call_text("strategist", "s", "u")
        summ = llm.usage_summary()
        self.assertEqual(summ["coder"]["calls"], 1)
        self.assertEqual(summ["strategist"]["total_tokens"], 15)
        self.assertIsNotNone(summ["coder"]["mean_latency_s"])


class TestChatInteractive(unittest.TestCase):
    """The whole chat.py conversation, with fake catalogue + provider."""

    def test_chat_selection_and_run(self):
        import chat
        answers = iter([
            "classify digits",   # goal
            "demo",              # dataset
            "",                  # no model preference
            "1",                 # Strategist -> list item 1
            "b/coder:free",      # Coder -> typed identifier
            "a/strat:free",      # Error Analyst -> same model as the Strategist
            "y",                 # Start optimization?
        ])
        prompts = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            return next(answers)

        with tempfile.TemporaryDirectory() as tmp, FakeProvider() as fp, \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}), \
                mock.patch.object(builtins, "input", fake_input), \
                mock.patch.object(ms, "fetch_catalog", lambda: CATALOG), \
                mock.patch.object(ms, "CONFIG_PATH", Path(tmp) / "cfg.json"), \
                mock.patch.object(chat, "SLOW", False):
            cwd = os.getcwd()
            os.chdir(tmp)
            out = io.StringIO()
            try:
                with redirect_stdout(out):
                    chat.main()
            finally:
                os.chdir(cwd)
                llm.clear_models()
            runs = list((Path(tmp) / "experiments").glob("run_*"))
            self.assertEqual(len(runs), 1)
            config = json.loads((runs[0] / "run_config.json").read_text())
            self.assertTrue((runs[0] / "final_report.md").exists())
            report = (runs[0] / "final_report.md").read_text()

        text = out.getvalue()
        self.assertIn("Model Strategist: Which LLM model would you like to use?", "\n".join(prompts))
        self.assertIn("Coder Agent: Which LLM model would you like to use?", "\n".join(prompts))
        self.assertIn("Error Analysis Agent: Which LLM model would you like to use?", "\n".join(prompts))
        self.assertIn("Start optimization?", prompts[-1])
        self.assertIn("Available free OpenRouter models:", text)
        self.assertIn("Selected configuration:", text)
        expected = {"strategist": "a/strat:free", "coder": "b/coder:free", "analyst": "a/strat:free"}
        self.assertEqual(config["llm"]["models"], expected)
        self.assertIn("LLM usage", report)
        from fakes import ROLE_SYSTEMS
        for role, system in ROLE_SYSTEMS.items():
            self.assertEqual({m for m, _ in fp.by_system(system)}, {expected[role]})


class TestChatAsksForKey(unittest.TestCase):
    def test_chat_prompts_for_missing_key(self):
        import chat
        answers = iter(["classify digits", "demo", "", "1", "1", "1", "y"])
        prompts, secret_prompts = [], []

        def fake_input(prompt=""):
            prompts.append(prompt)
            return next(answers)

        def fake_getpass(prompt=""):
            secret_prompts.append(prompt)
            return "sk-or-typed"

        with tempfile.TemporaryDirectory() as tmp, FakeProvider(), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_API_KEY": ""}), \
                mock.patch.object(builtins, "input", fake_input), \
                mock.patch("getpass.getpass", fake_getpass), \
                mock.patch.object(ms, "check_api_key", lambda k: (True, "ok")), \
                mock.patch.object(ms, "fetch_catalog", lambda: CATALOG), \
                mock.patch.object(ms, "CONFIG_PATH", Path(tmp) / "cfg.json"), \
                mock.patch.object(chat, "SLOW", False):
            cwd = os.getcwd()
            os.chdir(tmp)
            out = io.StringIO()
            try:
                with redirect_stdout(out):
                    chat.main()
                self.assertEqual(llm.api_key(), "sk-or-typed")
            finally:
                os.chdir(cwd)
                llm.clear_models()
                llm.set_session_key(None)
            self.assertEqual(len(list((Path(tmp) / "experiments").glob("run_*"))), 1)
            self.assertNotIn("sk-or-typed", (Path(tmp) / "cfg.json").read_text())

        self.assertEqual(len(secret_prompts), 1)
        self.assertIn("Paste your OpenRouter API key", secret_prompts[0])
        self.assertNotIn("sk-or-typed", out.getvalue())       # never echoed
        self.assertIn("Key accepted", out.getvalue())
        self.assertIn("Start optimization?", prompts[-1])


if __name__ == "__main__":
    unittest.main()
