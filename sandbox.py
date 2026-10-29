"""
sandbox.py — the EXECUTION TOOL. This is what turns "an LLM that writes code"
into "an agent that acts": it actually runs the code and returns what happened.

Output is STREAMED live (line by line) as the script runs, so long training jobs
show progress instead of going silent. Lines prefixed with the training script are
echoed through an optional `on_line` callback (the chat uses this to show epochs).

SAFETY: this runs LLM-generated code as a normal subprocess. Fine for a controlled
TP on a throwaway dataset; for anything real, run it inside Docker or a dedicated
sandbox (e2b, firejail, a VM) so generated code can't touch your machine.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

DEFAULT_TIMEOUT = int(os.environ.get("TRAIN_TIMEOUT", "3600"))  # seconds; raise if needed


TEST_FLAG = "PIPELINE_ALLOW_TEST"   # see workspace_files.py
# Never hand LLM credentials to generated code.
SECRET_VARS = ("OPENROUTER_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def run_script(workspace: Path, script: str, python: str = sys.executable,
               timeout: int = DEFAULT_TIMEOUT,
               on_line: Optional[Callable[[str], None]] = None,
               args: Sequence[str] = (), env: Optional[dict] = None) -> tuple[int, str, str]:
    """Run `<python> <script> <args>` inside the workspace, streaming output live.
    Returns (returncode, stdout, stderr). If on_line is given, each stdout/stderr
    line is passed to it as it appears (so the UI can show live progress).

    `env` adds variables for this run only. The test-unlock flag is always stripped
    from the inherited environment, so only an explicit `env={TEST_FLAG: "1"}` (the
    framework's final evaluation) can reach the test split. API keys are stripped too,
    so generated code never sees the user's LLM credentials."""
    child_env = {k: v for k, v in os.environ.items() if k != TEST_FLAG and k not in SECRET_VARS}
    child_env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})})
    proc = subprocess.Popen(
        [python, "-u", script, *args],    # -u = unbuffered, so prints appear immediately
        cwd=str(workspace),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        encoding="utf-8", errors="replace",
        # Force UTF-8 in the child: on Windows, piped output otherwise uses the
        # legacy code page and crashes on any non-ASCII print.
        env=child_env,
    )
    out_lines, err_lines = [], []

    def _pump(stream, sink):
        for line in iter(stream.readline, ""):
            sink.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
        stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, out_lines), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, err_lines), daemon=True)
    t_out.start(); t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t_out.join(timeout=2); t_err.join(timeout=2)
        return 124, "".join(out_lines), "".join(err_lines) + "\n[timed out]"

    t_out.join(timeout=2); t_err.join(timeout=2)
    return proc.returncode, "".join(out_lines), "".join(err_lines)


def parse_json_line(stdout: str) -> dict:
    """Grab the last valid JSON object printed by a script (the metrics line)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}
