# -*- coding: utf-8 -*-
"""
Runlock environment snapshot (env_details.py)

Captures Python, OS, GPU/driver details, and core ML stack into a single JSON file.
Intended for configuration parity and execution forensics across Windows and Linux
(including Amazon Linux).

Includes:
- Python + OS details
- Torch / CUDA / cuDNN configuration
- GPU device properties (via torch)
- NVIDIA driver/GPU summary (process-free, via nvidia-smi)
- pip freeze (exact dependency set)

Output:
- ./logs/env_details.json
"""

import sys
import json
import platform
import subprocess
import os
from datetime import datetime


def try_cmd(cmd, timeout_s=10):
    """
    Run a subprocess command and return stdout, or a readable ERROR string.
    """
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout_s
        ).strip()
    except subprocess.CalledProcessError as e:
        out = e.output.strip() if getattr(e, "output", None) else ""
        return f"ERROR: exit={e.returncode} cmd={cmd} output={out}"
    except Exception as e:
        return f"ERROR: {e}"


def get_package_versions():
    versions = {}

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except Exception:
        versions["numpy"] = None

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = None

    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = None

    return versions


def get_torch_info():
    info = {
        "installed": False,
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": None,
        "device_name": None,
        "device_capability": None,
        "device_properties": None,
        "determinism": {},
    }

    try:
        import torch
    except Exception:
        return info

    info["installed"] = True
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = getattr(torch.version, "cuda", None)

    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn_version"] = None

    try:
        info["determinism"]["deterministic_algorithms"] = (
            torch.are_deterministic_algorithms_enabled()
        )
    except Exception:
        info["determinism"]["deterministic_algorithms"] = None

    try:
        info["determinism"]["cudnn_deterministic"] = bool(
            torch.backends.cudnn.deterministic
        )
        info["determinism"]["cudnn_benchmark"] = bool(
            torch.backends.cudnn.benchmark
        )
    except Exception:
        info["determinism"]["cudnn_deterministic"] = None
        info["determinism"]["cudnn_benchmark"] = None

    if info["cuda_available"]:
        try:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = list(torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            info["device_properties"] = {
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "multi_processor_count": int(
                    getattr(props, "multi_processor_count", -1)
                ),
                "major": int(getattr(props, "major", -1)),
                "minor": int(getattr(props, "minor", -1)),
            }
        except Exception:
            pass

    return info


def get_nvidia_info():
    """
    Process-free NVIDIA details, portable across Windows + Linux.

    Strategy:
    1) Try a conservative query:
         name, driver_version, cuda_version
    2) Fallback to `nvidia-smi -L`
    """
    info = {
        "available": False,
        "gpu_query_csv": None,
        "fallback_list": None,
        "query_error": None,
        "fallback_error": None,
    }

    query = try_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,cuda_version",
        "--format=csv,noheader",
    ])

    if isinstance(query, str) and query.startswith("ERROR:"):
        info["query_error"] = query

        fallback = try_cmd(["nvidia-smi", "-L"])
        if isinstance(fallback, str) and fallback.startswith("ERROR:"):
            info["fallback_error"] = fallback
            return info

        info["fallback_list"] = fallback
        info["available"] = True
        return info

    info["gpu_query_csv"] = query
    info["available"] = True
    return info


def main():
    env = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform_string": platform.platform(),
        },
        "packages": get_package_versions(),
        "torch": get_torch_info(),
        "nvidia": get_nvidia_info(),
        "pip_freeze": None,
    }

    env["pip_freeze"] = try_cmd(
        [sys.executable, "-m", "pip", "freeze"], timeout_s=60
    )

    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "env_details.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    print(f"Environment snapshot written to {out_path}")


if __name__ == "__main__":
    main()
