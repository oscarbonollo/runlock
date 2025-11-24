# -*- coding: utf-8 -*-
"""
det_multi_turn_baseline_cpu_forensic_fast_v3.py

Runlock: deterministic multi-turn harness (CPU-only, fast variant) with
activation probes.

- No system prompt, no chat template.
- Pure token → token deterministic generation on CPU.
- Multi-turn dialog with:
    * Per-turn token hashes (SHA-256)
    * Conversation hash (hash of hashes)
    * Timing per turn + total runtime
    * Environment metadata (Python, Torch, platform, threads)
    * Float-activation probes (start/mid/end positions, last layer)
    * JSON log written to ./logs

This is the "forensic fast v3" baseline:
- Same deterministic core as earlier multi-turn scripts.
- Structured JSON logging for replay / audit.
- Real activation probes to capture drift-sensitive signatures.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Environment setup (CPU-only, stable threading)
# ------------------------------------------------------------

# Force CPU for maximum determinism across machines
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Avoid tokenizer parallelism noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reasonable fixed threads (you can tune, but keep it constant)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
SEED = 42
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW = 128
MAX_TURNS = 32

# The default dialog we’ve been using as the “genesis” test
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


# Global deterministic config (done once at import time)
reset_seeds(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_flush_denormal(True)

# ------------------------------------------------------------
# Cached model + tokenizer loader
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """
    Load tokenizer + model once, on CPU, in full precision.

    This function is cached so subsequent runs in the same process
    don't reload weights from disk.
    """
    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Ensure PAD is defined and deterministic
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()

    # Warm-up to initialize kernels (helps stabilize first-run timing)
    _ = model.generate(**tok("hi", return_tensors="pt"), max_new_tokens=1)

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

    Example:
        User: Hello
        Assistant: Hi there!
        User: How are you?
        Assistant:
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

    Returns a small JSON-serializable dict:

        {
            "layer": "last",
            "positions": {
                "start": { "idx": 0, "mean": ..., "std": ..., "l2": ... },
                "mid":   { "idx": m, "mean": ..., "std": ..., "l2": ... },
                "end":   { "idx": n, "mean": ..., "std": ..., "l2": ... }
            }
        }
    """
    # token_ids is a 1D numpy array of shape (seq_len,)
    seq_len = int(token_ids.shape[0])
    if seq_len == 0:
        return None

    # Choose positions
    idx_start = 0
    idx_end = seq_len - 1
    idx_mid = seq_len // 2

    # Build tensors
    input_ids = torch.tensor(token_ids, dtype=torch.long, device="cpu").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cpu")

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # Decoder-only: final hidden state is hidden_states[-1]
    last_hidden = out.hidden_states[-1]  # shape: [1, L, H]
    last_hidden = last_hidden[0]  # -> [L, H]

    def stats_for_idx(idx: int):
        v = last_hidden[idx]  # [H]
        # Use float64 internally for stability, then round for JSON
        v_mean = float(v.mean().item())
        v_std = float(v.std(unbiased=False).item())
        v_l2 = float(torch.linalg.norm(v).item())
        return {
            "idx": int(idx),
            "mean": round(v_mean, 8),
            "std": round(v_std, 8),
            "l2": round(v_l2, 8),
        }

    positions = {
        "start": stats_for_idx(idx_start),
        "mid": stats_for_idx(idx_mid),
        "end": stats_for_idx(idx_end),
    }

    return {
        "layer": "last",
        "positions": positions,
    }


def generate_turn(tok, model, history):
    """
    Run one deterministic generation step from current history.

    Returns:
        text: full decoded sequence (prompt + new tokens)
        token_ids: np.ndarray of token IDs for the full sequence
        elapsed: runtime in seconds
        float_probe: activation summary dict (or None)
    """
    # Reset seeds per turn to keep any RNG use stable (even though we use greedy decoding)
    reset_seeds(SEED)

    prompt = build_prompt(history)
    enc = tok(prompt, return_tensors="pt")

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

    seq = out.sequences[0].detach().cpu().numpy()
    text = tok.decode(out.sequences[0], skip_special_tokens=True)

    # Activation probe: last layer at start/mid/end positions
    float_probe = compute_float_probe(model, seq)

    return text, seq, elapsed, float_probe

# ------------------------------------------------------------
# Run deterministic dialog
# ------------------------------------------------------------

def run_dialog(user_turns):
    """
    Run a deterministic multi-turn dialog.

    Args:
        user_turns: list[str] of user messages.

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
        # Append user turn
        history.append({"role": "user", "content": u})

        # Generate assistant turn deterministically
        a_text, a_ids, elapsed, float_probe = generate_turn(tok, model, history)
        a_hash = hash_tokens(a_ids)

        # The assistant's output becomes part of next turn's context
        history.append({"role": "assistant", "content": a_text})

        per_turn.append({
            "turn": i,
            "user": u,
            "assistant": a_text,
            "token_hash": a_hash,
            "len_tokens": int(a_ids.size),
            "runtime_s": elapsed,
            "float_probe": float_probe,  # now a real drift-sensitive signature
        })

    conv_hash = dialog_hash([t["token_hash"] for t in per_turn])
    total_time = sum(t["runtime_s"] for t in per_turn)

    meta = {
        "python_version": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "transformers_version": "transformers",  # placeholder; can be replaced with transformers.__version__
        "platform": platform.platform(),
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "model_id": MODEL_ID,
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

    Shape:
        {
            "script": str,
            "model": str,
            "conversation_hash": str,
            "total_runtime_s": float,
            "meta": { ... env info ... },
            "turns": [
                {
                    "turn": int,
                    "user": str,
                    "assistant": str,
                    "token_hash": str,
                    "len_tokens": int,
                    "runtime_s": float,
                    "float_probe": { ... activation stats ... }
                },
                ...
            ]
        }
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

    print("=== FORENSIC DETERMINISTIC MULTI-TURN RUN (FAST v3 + PROBES) ===\n")

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
