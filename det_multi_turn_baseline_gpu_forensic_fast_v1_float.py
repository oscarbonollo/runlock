# -*- coding: utf-8 -*-
"""
det_multi_turn_baseline_gpu_forensic_fast_v1_float.py

Runlock: deterministic multi-turn harness (GPU variant, fast baseline) with
activation probes.

Goal:
- Stay as close as possible to the original CPU architecture.
- Run the same deterministic-style harness on CUDA GPU (best-effort).

Notes on determinism:
- GPU determinism is best-effort. Bit-identical results across different machines
  are NOT guaranteed unless GPU model + driver + CUDA + PyTorch stack match.
- This script keeps the same prompt assembly, hashing, logging, and probe structure.
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import time
import random
import hashlib
import platform
import json
import sys

from functools import lru_cache

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Environment setup (GPU, stable-ish determinism)
# ------------------------------------------------------------

# IMPORTANT: Do NOT force CPU here.
# (In the CPU baseline, CUDA_VISIBLE_DEVICES="" was set; we remove that for GPU.)

# Avoid tokenizer parallelism noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fixed threads (still relevant for tokenizer and any CPU-side work)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# cuBLAS determinism (must be set before first CUDA context init to be effective)
# If you hit errors in some environments, try ":16:8" instead.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
SEED = 42
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW = 128
MAX_TURNS = 32

# Strictly GPU (you asked for GPU). If you prefer fallback-to-CPU, change this.
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Install CUDA-enabled PyTorch and ensure an NVIDIA GPU is present.")

DEVICE = torch.device("cuda")

# The default dialog we’ve been using as the “genesis” test (2-turn grounding)
DEFAULT_USER_TURNS = [
    "What is 2+2?",
    "Explain the reasoning in one sentence.",
]

# ------------------------------------------------------------
# Determinism setup
# ------------------------------------------------------------

def reset_seeds(seed: int = SEED) -> None:
    """Reset all relevant RNGs to a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Global deterministic-ish config (done once at import time)
reset_seeds(SEED)

# Enforce deterministic algorithms where supported
torch.use_deterministic_algorithms(True, warn_only=False)

# Disable TF32 (helps reproducibility; TF32 can change numeric paths)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# cuDNN deterministic settings (mainly affects ops that use cuDNN kernels)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Keep denormal behavior consistent (more relevant on CPU, harmless here)
torch.set_flush_denormal(True)

# ------------------------------------------------------------
# Cached model + tokenizer loader
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """
    Load tokenizer + model once, on GPU, in full precision (fp32).

    Cached so subsequent runs in the same process don't reload weights.
    """
    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Ensure PAD is defined and deterministic
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    model.to(DEVICE)
    model.eval()

    # Warm-up to initialize kernels (helps stabilize first-run timing)
    _ = model.generate(**tok("hi", return_tensors="pt").to(DEVICE), max_new_tokens=1)

    return tok, model

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def hash_tokens(ids: np.ndarray) -> str:
    """
    Hash a sequence of token IDs using SHA-256.
    Ensures consistent dtype and byte layout before hashing.
    """
    arr = ids.astype(np.int32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def dialog_hash(turn_hashes):
    """
    Compute a stable conversation fingerprint from the sequence of per-turn hashes.
    """
    h = hashlib.sha256()
    for th in turn_hashes:
        h.update(bytes.fromhex(th))
    return h.hexdigest()


def build_prompt(history):
    """
    Simple deterministic prompt assembly.
    No roles or templates—just plain concatenation.
    """
    prompt = ""
    for turn in history:
        prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
    prompt += "Assistant:"  # model continues from here
    return prompt.strip()


def compute_float_probe(model, token_ids: np.ndarray):
    """
    Compute drift-sensitive float activation summary at start / mid / end
    token positions, using the last hidden layer.

    Returns JSON-serializable dict:
        {
          "layer": "last",
          "positions": {
            "start": {"idx": 0, "mean": ..., "std": ..., "l2": ...},
            "mid":   {"idx": m, "mean": ..., "std": ..., "l2": ...},
            "end":   {"idx": n, "mean": ..., "std": ..., "l2": ...}
          }
        }
    """
    seq_len = int(token_ids.shape[0])
    if seq_len == 0:
        return None

    idx_start = 0
    idx_end = seq_len - 1
    idx_mid = seq_len // 2

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    last_hidden = out.hidden_states[-1][0]  # [L, H]

    def stats_for_idx(idx: int):
        v = last_hidden[idx]  # [H]
        v_mean = float(v.mean().item())
        v_std = float(v.std(unbiased=False).item())
        v_l2 = float(torch.linalg.norm(v).item())
        return {
            "idx": int(idx),
            "mean": round(v_mean, 8),
            "std": round(v_std, 8),
            "l2": round(v_l2, 8),
        }

    return {
        "layer": "last",
        "positions": {
            "start": stats_for_idx(idx_start),
            "mid": stats_for_idx(idx_mid),
            "end": stats_for_idx(idx_end),
        },
    }


def generate_turn(tok, model, history):
    """
    Run one deterministic(-ish) generation step from current history on GPU.

    Returns:
        text: full decoded sequence (prompt + new tokens)
        token_ids: np.ndarray of token IDs for the full sequence
        elapsed: runtime in seconds
        float_probe: activation summary dict (or None)
    """
    # Reset seeds per turn to keep any RNG use stable (even though we use greedy decoding)
    reset_seeds(SEED)

    prompt = build_prompt(history)
    enc = tok(prompt, return_tensors="pt").to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    elapsed = time.perf_counter() - start

    # Move to CPU for hashing + JSON logging
    seq_t = out.sequences[0].detach().to("cpu")
    seq = seq_t.numpy()

    # Decode on CPU tensor (keeps behavior consistent)
    text = tok.decode(seq_t, skip_special_tokens=True)

    # Activation probe (runs model forward on GPU)
    float_probe = compute_float_probe(model, seq)

    return text, seq, elapsed, float_probe

# ------------------------------------------------------------
# Run deterministic dialog
# ------------------------------------------------------------

def run_dialog(user_turns):
    """
    Run a deterministic multi-turn dialog (GPU).

    Returns:
        {
          "turns": [ { per-turn data } ],
          "conversation_hash": str,
          "total_runtime": float,
          "meta": { environment metadata }
        }
    """
    tok, model = load_model_and_tokenizer()

    history = []
    per_turn = []

    for i, u in enumerate(user_turns[:MAX_TURNS], 1):
        history.append({"role": "user", "content": u})

        a_text, a_ids, elapsed, float_probe = generate_turn(tok, model, history)
        a_hash = hash_tokens(a_ids)

        history.append({"role": "assistant", "content": a_text})

        per_turn.append({
            "turn": i,
            "user": u,
            "assistant": a_text,
            "token_hash": a_hash,
            "len_tokens": int(a_ids.size),
            "runtime_s": elapsed,
            "float_probe": float_probe,
        })

    conv_hash = dialog_hash([t["token_hash"] for t in per_turn])
    total_time = sum(t["runtime_s"] for t in per_turn)

    # Metadata: keep original + add GPU details
    meta = {
        "python_version": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "model_id": MODEL_ID,
        "device": str(DEVICE),
        "cuda": {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32": {
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "deterministic": {
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        },
    }

    return {
        "turns": per_turn,
        "conversation_hash": conv_hash,
        "total_runtime": total_time,
        "meta": meta,
    }

# ------------------------------------------------------------
# JSON logging
# ------------------------------------------------------------

def write_runlock_log(script_name: str, run_result: dict) -> str:
    """
    Write a structured JSON log for the run.
    """
    os.makedirs("logs", exist_ok=True)

    conv_hash = run_result["conversation_hash"]
    total_time = run_result["total_runtime"]
    model_id = run_result["meta"]["model_id"]

    safe_model = model_id.replace("/", "-")
    log_name = (
        f"{script_name}__{safe_model}__{conv_hash[:12]}__{total_time:.2f}s.json"
    )
    log_path = os.path.join("logs", log_name)

    payload = {
        "script": script_name,
        "model": model_id,
        "conversation_hash": conv_hash,
        "total_runtime_s": total_time,
        "meta": run_result["meta"],
        "turns": run_result["turns"],
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return log_path

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

if __name__ == "__main__":
    user_turns = DEFAULT_USER_TURNS

    result = run_dialog(user_turns)

    print("=== FORENSIC DETERMINISTIC MULTI-TURN RUN (GPU v1 + PROBES) ===\n")

    for t in result["turns"]:
        print(f"[Turn {t['turn']}]")
        print(f"User: {t['user']}")
        print(f"Assistant: {t['assistant']}")
        print(
            f"HASH: {t['token_hash']} | LEN: {t['len_tokens']} | "
            f"TIME: {t['runtime_s']:.2f}s"
        )
        print(f"Probe: {t['float_probe']}")
        print()

    print(f"CONVERSATION_HASH: {result['conversation_hash']}")
    print(f"TOTAL_RUNTIME: {result['total_runtime']:.2f} seconds")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_path = write_runlock_log(script_name, result)

    print(f"[✔] Runlock log written to: {log_path}")
