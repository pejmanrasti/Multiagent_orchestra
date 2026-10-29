"""
environments.py — isolated environments + automatic dependency install.

Two jobs:
  1. detect_pip_packages(code): read the generated train.py and work out which
     pip packages it needs (import names -> pip names, stdlib filtered out).
  2. ensure_env(packages): build (or reuse) an isolated environment that has those
     packages installed, and return the path to *that env's* python executable.
     loop.py then runs train.py / eval_model.py with that python.

ISOLATION (env var):
  venv  (default) — fast, built-in, one env per distinct dependency set.
  conda           — uses `conda create -p ...`; slower, needs conda installed.
  none            — run in the current interpreter (handy for quick testing).

Environments are CACHED by their package set (a hash), so identical dependency
sets are reused instead of reinstalled. Set ENV_FRESH=1 to force a brand-new env
every call (wasteful; only if you truly want per-run isolation).

SAFETY: set PIP_ALLOWLIST="scikit-learn,joblib,numpy,torch,..." to restrict what
may be installed. Unset = allow anything (fine for a throwaway TP box; risky
otherwise, since the agent is both writing code AND pulling packages).
"""
from __future__ import annotations
import ast
import hashlib
import os
import re
import subprocess
import sys
import venv
from pathlib import Path

ISOLATION = os.environ.get("ISOLATION", "venv")           # venv | conda | none
ENV_CACHE = Path(os.environ.get("ENV_CACHE", ".envs")).resolve()
CONDA_PY = os.environ.get("ENV_PYTHON_VERSION", "3.11")   # for conda backend
FRESH = os.environ.get("ENV_FRESH") == "1"

_allow = os.environ.get("PIP_ALLOWLIST", "").strip()
TORCH_INDEX_URL = os.environ.get("TORCH_INDEX_URL", "").strip()  # GPU wheels index
_TORCH_PKGS = {"torch", "torchvision", "torchaudio"}
ALLOWLIST = {p.strip() for p in _allow.split(",")} if _allow else None  # None = allow all

# import-name -> pip-package-name, for the cases where they differ.
IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
}

# Base packages the harness/evaluator always needs (models are unpickled by eval_model.py).
BASE_PACKAGES = ("scikit-learn", "joblib")


# --------------------------------------------------------------------------- #
# 1. Dependency detection
# --------------------------------------------------------------------------- #
def detect_pip_packages(code: str, base=BASE_PACKAGES) -> list[str]:
    """Parse top-level imports from `code` and return the pip packages to install."""
    pkgs = set(base)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return sorted(pkgs)

    stdlib = getattr(sys, "stdlib_module_names", set())
    local = {"data", "score", "train", "model", "eval_model", "preprocess", "workspace_files"}

    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])

    for m in mods:
        if m in stdlib or m in local or m.startswith("_"):
            continue
        pkgs.add(IMPORT_TO_PIP.get(m, m))
    return sorted(pkgs)


def missing_module(stderr: str) -> str | None:
    """Extract the module name from a ModuleNotFoundError, for install-and-retry."""
    m = re.search(r"No module named ['\"]([\w\.]+)['\"]", stderr)
    return m.group(1).split(".")[0] if m else None


# --------------------------------------------------------------------------- #
# 2. Environment creation (cached)
# --------------------------------------------------------------------------- #
def _bin(env_dir: Path) -> str:
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return str(env_dir / sub / exe)


def _check_allow(packages) -> None:
    if ALLOWLIST is None:
        return
    def name(p):  # strip version pins
        return re.split(r"[=<>!~]", p)[0].strip()
    bad = [p for p in packages if name(p) not in ALLOWLIST]
    if bad:
        raise PermissionError(f"packages not in PIP_ALLOWLIST: {bad}")


def _key(packages, salt: str) -> str:
    raw = f"{ISOLATION}|{CONDA_PY}|{salt}|{','.join(sorted(packages))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ensure_env(packages, salt: str = "") -> str:
    """Return the python executable of an isolated env with `packages` installed."""
    packages = sorted(set(packages))
    if ISOLATION == "none":
        return sys.executable
    _check_allow(packages)
    if FRESH and not salt:
        salt = os.urandom(4).hex()

    ENV_CACHE.mkdir(parents=True, exist_ok=True)
    env_dir = ENV_CACHE / _key(packages, salt)
    py = _bin(env_dir)
    ready = env_dir / ".ready"
    if ready.exists():
        return py  # cache hit — reuse

    print(f"    [env] building {ISOLATION} env for: {', '.join(packages) or '(base only)'}")
    if ISOLATION == "conda":
        subprocess.run(["conda", "create", "-p", str(env_dir), f"python={CONDA_PY}", "-y"],
                       check=True)
    else:
        venv.EnvBuilder(with_pip=True).create(env_dir)

    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    if packages:
        _pip_install(py, packages)
    ready.write_text("ok", encoding="utf-8")
    return py


def _pip_install(py: str, packages: list[str]) -> None:
    """Install packages, routing torch wheels through TORCH_INDEX_URL when set (for GPU)."""
    torch_index = os.environ.get("TORCH_INDEX_URL", "").strip()   # read live
    def base(p):
        return re.split(r"[=<>!~]", p)[0].strip()
    torch_pkgs = [p for p in packages if base(p) in _TORCH_PKGS]
    other = [p for p in packages if base(p) not in _TORCH_PKGS]
    if torch_pkgs and torch_index:
        subprocess.run([py, "-m", "pip", "install", "-q", "--index-url",
                        torch_index, *torch_pkgs], check=True)
    else:
        other = packages  # default wheels for everything
    if other:
        subprocess.run([py, "-m", "pip", "install", "-q", *other], check=True)


def install_into(py: str, packages) -> None:
    """Install extra packages into an existing env (used by the missing-module retry)."""
    packages = list(packages)
    _check_allow(packages)
    print(f"    [env] installing missing: {', '.join(packages)}")
    _pip_install(py, packages)
