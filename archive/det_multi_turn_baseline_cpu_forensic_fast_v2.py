# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# det_multi_turn_baseline_cpu_forensic_fast_v2.py
# Deterministic multi-turn inference with Mistral-7B-Instruct-v0.2 (CPU-only).
# Uses runlock_logger for clean, reproducible JSON output.
# ------------------------------------------------------------

import os, time, random, numpy as np, torch
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Runlock logger import
# ------------------------------------------------------------
from runlock_logger import (
    system_fingerprint,
    hash_tokens,
    dialog_hash,
    log_run
)

# ------------------------------------------------------------
# Environment setup (CPU-only, stable threading)
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # Force CPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
SEED      = 42
MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW   = 128
MAX_TURNS = 32

# ------------------------------------------------------------
# Determinism setup
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_flush_denormal(True)

# ------------------------------------------------------------
# Cached model + tokenizer loader
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32
    )
    model.eval()

    # Warm-up to initialize kernels
    _ = model.generate(**tok("hi", return_tensors="pt"), max_new_tokens=1)
    return tok, model

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def reset_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    prompt += "Assistant:"
    return prompt.strip()

def generate_turn(tok, model, history):
    """Run one deterministic generation step from current history."""
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
            output_scores=False
        )

    elapsed = time.perf_counter() - start
    seq = out.sequences[0].detach().cpu().numpy()
    text = tok.decode(out.sequences[0], skip_special_tokens=True)
    return text, seq, elapsed

# ------------------------------------------------------------
# Run deterministic dialog
# ------------------------------------------------------------
def run_dialog(user_turns):
    tok, model = load_model_and_tokenizer()
    history = []
    per_turn = []

    for i, u in enumerate(user_turns[:MAX_TURNS], 1):
        history.append({"role": "user", "content": u})
        a_text, a_ids, elapsed = generate_turn(tok, model, history)
        a_hash = hash_tokens(a_ids)

        history.append({"role": "assistant", "content": a_text})
        per_turn.append({
            "turn": i,
            "user": u,
            "assistant": a_text,
            "token_hash": a_hash,
            "len_tokens": int(a_ids.size),
            "runtime_s": elapsed
        })

    conv_hash = dialog_hash([t["token_hash"] for t in per_turn])
    total_time = sum(t["runtime_s"] for t in per_turn)
    return {"turns": per_turn, "conversation_hash": conv_hash, "total_time": total_time}

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
if __name__ == "__main__":
    USER_TURNS = [
        "What is 2+2?",
        "Explain the reasoning in one sentence."
    ]

    print("Loading model and tokenizer...")
    result = run_dialog(USER_TURNS)

    # ---- Structured output ----
    print("=== FORENSIC DETERMINISTIC MULTI-TURN RUN (FAST v2) ===")
    meta = system_fingerprint()
    meta["model"] = MODEL_ID

    for t in result["turns"]:
        print(f"\n[Turn {t['turn']}]")
        print(f"User: {t['user']}")
        print(f"Assistant: {t['assistant']}")
        print(f"HASH: {t['token_hash']} | LEN: {t['len_tokens']} | TIME: {t['runtime_s']:.2f}s")

    print(f"\nCONVERSATION_HASH: {result['conversation_hash']}")
    print(f"TOTAL_RUNTIME: {result['total_time']:.2f} seconds")

    # ---- JSON log output ----
    os.makedirs("logs", exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    safe_model = MODEL_ID.replace("/", "-")
    log_name = f"{script_name}__{safe_model}__{result['conversation_hash'][:12]}__{result['total_time']:.2f}s.json"
    log_path = os.path.join("logs", log_name)

    log_run(meta, result["turns"], result["conversation_hash"], log_path)
