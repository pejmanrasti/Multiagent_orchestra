"""
preflight.py — detect the machine and pick the right PyTorch build automatically.

WHAT CAN AND CANNOT BE AUTO-INSTALLED
  - Python packages, INCLUDING the CUDA runtime PyTorch needs (cuBLAS, cuDNN, ...),
    ship as pip wheels. This tool installs those for you automatically, by choosing
    the correct torch index for your GPU.
  - The NVIDIA GPU DRIVER is a system kernel driver: it needs admin rights, an
    OS-specific installer, and usually a reboot. A script must NOT silently install
    that. So we DETECT it and, if it's missing, tell you exactly what to get.

This sets the TORCH_INDEX_URL environment variable so environments.py installs the
matching torch wheel (GPU wheel if a usable driver is found, CPU wheel otherwise —
which also avoids a huge needless CUDA download on machines with no GPU).

Run standalone to just see the report:
    python preflight.py
"""
from __future__ import annotations
import os
import platform
import re
import shutil
import subprocess
import sys

CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def _driver_cuda(smi: str):
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", smi)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _gpu_name(smi: str) -> str:
    for line in smi.splitlines():
        m = re.search(r"\|\s*\d+\s+(NVIDIA .+?)\s{2,}", line)
        if m:
            return m.group(1).strip()
    return "NVIDIA GPU"


def _torch_index_for(cuda) -> str | None:
    """Pick the highest torch CUDA wheel not exceeding the driver's max CUDA version."""
    if cuda is None:
        return None
    v = cuda[0] * 10 + cuda[1]
    if v >= 124:
        return "https://download.pytorch.org/whl/cu124"
    if v >= 121:
        return "https://download.pytorch.org/whl/cu121"
    if v >= 118:
        return "https://download.pytorch.org/whl/cu118"
    return None  # driver too old for current torch


def detect() -> dict:
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "gpu": None,
        "driver_cuda": None,
        "torch_index": None,
        "device": "cpu",
        "notes": [],
    }

    if sys.version_info < (3, 10):
        info["notes"].append(f"Python 3.10+ is recommended (you have {info['python']}).")

    smi = _nvidia_smi()
    if smi:
        info["gpu"] = _gpu_name(smi)
        info["driver_cuda"] = _driver_cuda(smi)
        idx = _torch_index_for(info["driver_cuda"])
        if idx:
            info["torch_index"] = idx
            info["device"] = "cuda"
        else:
            info["torch_index"] = CPU_INDEX
            info["notes"].append(
                "An NVIDIA GPU was found but the driver's CUDA version is too old for "
                "current PyTorch. Update your NVIDIA driver from nvidia.com/Download/index.aspx, "
                "then re-run. Using CPU for now.")
    elif info["os"] == "Darwin" and info["arch"] == "arm64":
        info["device"] = "mps"           # Apple Silicon GPU; default wheel supports it
        info["notes"].append("Apple Silicon detected — PyTorch will use the MPS GPU backend.")
    else:
        info["torch_index"] = CPU_INDEX
        info["notes"].append(
            "No NVIDIA GPU/driver detected. If you DO have an NVIDIA GPU, install its "
            "driver (nvidia.com/Download/index.aspx) — that's the one piece this tool "
            "cannot install for you. Everything else (incl. the CUDA runtime torch needs) "
            "is installed automatically. Using CPU for now.")
    return info


def apply(info: dict) -> None:
    """Export TORCH_INDEX_URL so the env manager installs the matching wheel."""
    if info["torch_index"] and not os.environ.get("TORCH_INDEX_URL"):
        os.environ["TORCH_INDEX_URL"] = info["torch_index"]


def report(info: dict) -> str:
    lines = [f"System check: {info['os']}/{info['arch']}, Python {info['python']}"]
    if info["gpu"]:
        cu = ".".join(map(str, info["driver_cuda"])) if info["driver_cuda"] else "?"
        lines.append(f"  GPU: {info['gpu']} (driver supports CUDA {cu})")
    lines.append(f"  Training device: {info['device'].upper()}")
    if info["torch_index"]:
        lines.append(f"  PyTorch wheels: {info['torch_index']}")
    for n in info["notes"]:
        lines.append(f"  ! {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    i = detect()
    print(report(i))
