"""A fake OpenRouter provider for tests: answers the three agent roles plausibly and
records (model, system, user) for every request."""
import json
import re

import agents
import llm


class FakeProvider:
    def __init__(self):
        self.calls = []            # (model, system, user)
        self._orig = None

    def __enter__(self):
        self._orig = llm._complete
        llm._complete = self.complete
        return self

    def __exit__(self, *exc):
        llm._complete = self._orig

    def by_system(self, system):
        return [(m, u) for m, s, u in self.calls if s == system]

    def complete(self, model, system, user):
        self.calls.append((model, system, user))
        info = {"served_model": model, "prompt_tokens": len(user) // 4,
                "completion_tokens": 50, "total_tokens": len(user) // 4 + 50}
        n = len(self.by_system(system))
        if system == agents.STRATEGIST_SYS:
            c = [0.05, 1.0, 3.0, 10.0][min(n - 1, 3)]
            return json.dumps({
                "experiment_name": f"logreg_C{c}",
                "architecture": "sklearn.linear_model.LogisticRegression",
                "hyperparameters": {"C": c, "max_iter": 3000},
                "training_config": {"standardize": True},
                "rationale": "fake", "coder_instructions": "implement it"}), info
        if system == agents.CODER_SYS:
            c = float(re.search(r'"C": ([0-9.]+)', user).group(1))
            return "```python\n" + logreg_script(c) + "\n```", info
        if system == agents.ANALYST_SYS:
            return json.dumps({"fit_diagnosis": "underfitting", "observations": ["obs"],
                               "hypotheses": ["hyp"], "recommendations": ["increase C"],
                               "summary": f"ANALYSIS-MARKER-{n}"}), info
        if "connection test" in system:
            return "OK", info
        raise AssertionError("unknown role")


def logreg_script(c: float) -> str:
    """A real, contract-following train.py (what a well-behaved Coder would return)."""
    return f'''\
import json, os
import joblib
import numpy as np
from data import get_splits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = int(os.environ.get("PIPELINE_SEED", "42"))
np.random.seed(SEED)
X_train, y_train, X_val, y_val = get_splits()

model = make_pipeline(StandardScaler(),
                      LogisticRegression(C={c}, max_iter=3000, random_state=SEED))
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print(json.dumps({{"train_acc": round(float(model.score(X_train, y_train)), 4),
                  "val_acc": round(float(model.score(X_val, y_val)), 4)}}))
'''


ROLE_SYSTEMS = {"strategist": agents.STRATEGIST_SYS, "coder": agents.CODER_SYS,
                "analyst": agents.ANALYST_SYS}
