# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# runlock_logger.py
# Deterministic, JSON-based logging utility for the Runlock framework.
# Captures reproducible metadata, hashes, and conversation state.
# ------------------------------------------------------------

import os
import sys
import json
import platform
import hashlib
import torch
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# System + environment fingerprint
# ------------------------------------------------------------
def system_fingerprint():
    """
    Return stable, human-readable system + environment metadata.
    This captures enough context to reproduce the environment
    where the deterministic run occurred.
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": os.cpu_count(),
        "default_dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

# ------------------------------------------------------------
# Hashing helpers
# ------------------------------------------------------------
def hash_tokens(ids: np.ndarray) -> str:
    """
    Stable SHA256 hash for a sequence of token IDs.
    This is the atomic proof of determinism for a generated sequence.
    """
    arr = ids.astype(np.int32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def dialog_hash(turn_hashes):
    """
    Compute a stable conversation fingerprint from per-turn hashes.
    Each turn’s token hash is folded into a single final digest.
    """
    h = hashlib.sha256()
    for th in turn_hashes:
        h.update(bytes.fromhex(th))
    return h.hexdigest()

# ------------------------------------------------------------
# JSON log writer
# ------------------------------------------------------------
def log_run(metadata: dict, turns: list, conversation_hash: str, out_path: str):
    """
    Write a deterministic JSON log for a model run.

    Args:
        metadata: dict of system/model info
        turns: list of dicts containing per-turn data
        conversation_hash: final conversation fingerprint
        out_path: full path for JSON log
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "metadata": metadata,
        "conversation_hash": conversation_hash,
        "turns": turns,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[✔] Runlock log written to: {out_path}")
    return out_path

# ------------------------------------------------------------
# Placeholder hooks for advanced instrumentation
# ------------------------------------------------------------
def entropy_metrics(logits_tensor):
    """
    Placeholder for entropy/variance measurement.
    Returns None for now (to avoid overhead in baseline runs).

    Future:
        Compute Shannon entropy or top-k logit variance to
        detect any deviation in model uncertainty profiles
        across deterministic runs.
    """
    return None


def rng_snapshot():
    """
    Capture RNG states for forensic determinism validation.

    Future:
        Return digests of Python, NumPy, and Torch RNG states.
        Used to confirm that no stochastic process occurred
        during deterministic inference.
    """
    return None


def activation_checksum(tensor):
    """
    Compute lightweight numeric checksum for hidden states.
    Used to detect floating-point drift or kernel nondeterminism.

    Returns:
        float checksum value, or None if tensor not provided.
    """
    return float(torch.sum(tensor).item()) if torch.is_tensor(tensor) else None
