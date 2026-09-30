# -*- coding: utf-8 -*-
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

# Hide all GPUs so everything runs on CPU (set before any CUDA initialisation)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Avoid tokenizer parallelism noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default thread counts, only if not already set in the environment.
# Note: torch is already imported at this point, so these may not take effect;
# the log records these env vars, not torch.get_num_threads().
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
SEED = 42
MODEL_ID = "microsoft/phi-2"
MAX_NEW = 128
MAX_TURNS = 32

# Default two-turn grounding dialog
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


# Global deterministic configuration
reset_seeds(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_flush_denormal(True)

# ------------------------------------------------------------
# Cached model + tokenizer loader
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """
    Load tokenizer and model on CPU in full precision (fp32).
    Memoised with lru_cache: repeat calls within the same Python process reuse
    the loaded weights. Nothing is cached between separate script runs (beyond
    the Hugging Face download cache on disk).
    """
    print("Loading model and tokenizer...")

    # trust_remote_code: older transformers versions need Phi-2's custom model code
    # use_fast=False: pin the slow (Python) tokenizer so the fast/slow choice can't vary
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    # Phi-2 has no pad token; use EOS as pad so padding is always defined
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Warm-up call so one-off initialisation cost isn't counted in turn 1's timing
    # (does not affect generated tokens)
    _ = model.generate(**tok("hi", return_tensors="pt"), max_new_tokens=1)

    return tok, model

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def hash_tokens(ids: np.ndarray) -> str:
    """Hash a sequence of token IDs using SHA-256."""
    arr = ids.astype(np.int32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def dialog_hash(turn_hashes):
    """Compute a stable conversation fingerprint from per-turn hashes."""
    h = hashlib.sha256()
    for th in turn_hashes:
        h.update(bytes.fromhex(th))
    return h.hexdigest()


def build_prompt(history):
    """
    Deterministic prompt assembly with no templating or roles beyond plain text.
    """
    prompt = ""
    for turn in history:
        prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
    prompt += "Assistant:"
    return prompt.strip()


def compute_float_probe(model, token_ids: np.ndarray):
    """
    Compute a lightweight activation fingerprint from the final hidden layer
    at the start, midpoint, and end token positions.

    This is a separate forward pass over the finished sequence (KV cache off),
    run after generation, so it fingerprints a recomputation rather than the
    exact kernels used during generate().
    """
    seq_len = int(token_ids.shape[0])
    if seq_len == 0:
        return None

    idx_start = 0
    idx_mid = seq_len // 2
    idx_end = seq_len - 1

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    last_hidden = out.hidden_states[-1][0]

    def stats(idx: int):
        v = last_hidden[idx]
        return {
            "idx": int(idx),
            "mean": round(float(v.mean().item()), 8),
            "std": round(float(v.std(unbiased=False).item()), 8),
            "l2": round(float(torch.linalg.norm(v).item()), 8),
        }

    return {
        "layer": "last",
        "positions": {
            "start": stats(idx_start),
            "mid": stats(idx_mid),
            "end": stats(idx_end),
        },
    }


def generate_turn(tok, model, history):
    """
    Run one assistant turn: greedy-decode up to MAX_NEW tokens from the history.

    Returns the full sequence (prompt tokens + generated tokens), both as
    token IDs and decoded text, so the hash and text include the prompt.
    """
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
            # use_cache not set: KV cache is ON (transformers default), unlike the GPU runner
            return_dict_in_generate=True,
            output_scores=False,
        )
    elapsed = time.perf_counter() - start

    # Full sequence (prompt + generated) as token IDs and decoded text
    seq_t = out.sequences[0].detach().to("cpu")
    seq = seq_t.numpy()
    text = tok.decode(seq_t, skip_special_tokens=True)

    float_probe = compute_float_probe(model, seq)

    return text, seq, elapsed, float_probe

# ------------------------------------------------------------
# Run deterministic dialog
# ------------------------------------------------------------
def run_dialog(user_turns):
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

    meta = {
        "python_version": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "model_id": MODEL_ID,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_equals_eos": (tok.pad_token_id == tok.eos_token_id),
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
    os.makedirs("logs", exist_ok=True)

    conv_hash = run_result["conversation_hash"]
    total_time = run_result["total_runtime"]
    model_id = run_result["meta"]["model_id"]

    safe_model = model_id.replace("/", "-")
    log_name = f"{script_name}__{safe_model}__{conv_hash[:12]}__{total_time:.2f}s.json"
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
    result = run_dialog(DEFAULT_USER_TURNS)

    for t in result["turns"]:
        print(f"[Turn {t['turn']}]")
        print(f"User: {t['user']}")
        print(f"Assistant: {t['assistant']}")
        print(
            f"HASH: {t['token_hash']} | LEN: {t['len_tokens']} | "
            f"TIME: {t['runtime_s']:.2f}s"
        )
        print(f"Probe: {t['float_probe']}\n")

    print(f"CONVERSATION_HASH: {result['conversation_hash']}")
    print(f"TOTAL_RUNTIME: {result['total_runtime']:.2f} seconds")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_path = write_runlock_log(script_name, result)

    print(f"Runlock log written to: {log_path}")
