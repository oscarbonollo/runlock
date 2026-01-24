# -*- coding: utf-8 -*-
# det_core_deterministic_fast.py
# ------------------------------------------------------------
# Deterministic inference test for Mistral-7B-Instruct-v0.2
# CPU-only, reproducible outputs, and per-run timing.
# Each run writes a uniquely named log file using:
#   <script_name>__<model_id>__<hash>__<seconds>.log
# ------------------------------------------------------------

import os, time, random, hashlib, numpy as np, torch
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # disable GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ------------------------------------------------------------
# TEST PARAMETERS
# ------------------------------------------------------------
SEED     = 42
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT   = "What is 2+2?"
MAX_NEW  = 64

# ------------------------------------------------------------
# DETERMINISM SETUP
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_flush_denormal(True)

# ------------------------------------------------------------
# MODEL/TOKENIZER LOADING (CACHED)
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model():
    """Load tokenizer and model once, cache them to speed up reruns."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32
    )
    model.eval()

    # Warm-up to initialize kernels and caches deterministically
    _ = model.generate(**tok("hi", return_tensors="pt"), max_new_tokens=1)
    return tok, model

# ------------------------------------------------------------
# HELPER: RESET SEEDS EACH RUN
# ------------------------------------------------------------
def reset_seeds(seed=SEED):
    """Reset all random generators to a known state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------------------------------------------------
# MAIN GENERATION FUNCTION
# ------------------------------------------------------------
def run_deterministic(prompt):
    """Run deterministic generation and return text, hash, length, runtime."""
    reset_seeds()
    tok, model = load_model()
    enc = tok(prompt.strip(), return_tensors="pt")

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
    end = time.perf_counter()

    seq = out.sequences[0].detach().cpu().numpy().astype(np.int32)
    text = tok.decode(out.sequences[0], skip_special_tokens=True)
    token_hash = hashlib.sha256(seq.tobytes()).hexdigest()
    elapsed = end - start

    return text, token_hash, int(seq.size), elapsed

# ------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":
    text, h, n, elapsed = run_deterministic(PROMPT)

    # ---- Console output ----
    print(f"TOKEN_HASH: {h}")
    print(f"LEN_TOKENS: {n}")
    print(f"RUNTIME: {elapsed:.2f} seconds")
    print("\nOUTPUT:\n", text)

    # ---- Auto-named log file ----
    os.makedirs("logs", exist_ok=True)

    # Create filename based on this script, model, hash, and timing
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    safe_model = MODEL_ID.replace("/", "-")  # make filename safe
    log_name = f"{script_name}__{safe_model}__{h[:12]}__{elapsed:.2f}s.log"
    log_path = os.path.join("logs", log_name)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Script: {script_name}\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Prompt: {PROMPT}\n")
        f.write(f"Hash: {h}\n")
        f.write(f"Length: {n}\n")
        f.write(f"Runtime: {elapsed:.2f} seconds\n\n")
        f.write("Output:\n")
        f.write(text)

    print(f"\n[✔] Log written to: {log_path}")
