# -*- coding: utf-8 -*-
# det_multi_turn_baseline_cpu_forensic_fast.py
# ------------------------------------------------------------
# Deterministic multi-turn chat (CPU-only) — forensic + tuned.
# Keeps all reproducibility controls; adds safe performance tweaks:
#   - Uses all CPU cores deterministically (fixed thread counts)
#   - Persists model/tokenizer (no reload between runs in-process)
#   - Uses inference_mode (slightly lower Python overhead)
# Hashes should remain IDENTICAL to the forensic baseline script.
# ------------------------------------------------------------

import os, time, random, hashlib, numpy as np, torch, sys, platform, unicodedata
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Environment setup (CPU-only, stable threading)
# ------------------------------------------------------------
# You can override threads with:  RUNLOCK_THREADS=8  (optional)
NUM_THREADS = int(os.environ.get("RUNLOCK_THREADS", str(os.cpu_count() or 4)))

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # force CPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

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

# Deterministic CPU threading (safe speed-up)
try:
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(1)
except Exception:
    # Older torch may not have set_num_interop_threads; ignore silently
    pass

# ------------------------------------------------------------
# Cached model + tokenizer loader (persists within this process)
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

    # ✅ Forensic: assert all weights are float32
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"Non-float32 param: {name}"

    # Warm-up to initialize kernels predictably (does not affect output)
    with torch.inference_mode():
        _ = model.generate(**tok("hi", return_tensors="pt"), max_new_tokens=1)
    return tok, model

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def reset_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hash_tokens(ids: np.ndarray) -> str:
    arr = ids.astype(np.int32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()

def build_prompt(history):
    """
    Deterministic prompt builder.
    No templates, no system instructions.
    NFC-normalize before tokenization (cross-platform consistency).
    """
    prompt = ""
    for turn in history:
        content = unicodedata.normalize("NFC", turn['content'])
        prompt += f"{turn['role'].capitalize()}: {content}\n"
    prompt += "Assistant:"
    return prompt.strip()

def generate_turn(tok, model, history):
    """Run one deterministic generation step from current history."""
    reset_seeds(SEED)
    prompt = build_prompt(history)
    enc = tok(prompt, return_tensors="pt")
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,          # greedy
            num_beams=1,              # no beam search
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

def dialog_hash(turn_hashes):
    h = hashlib.sha256()
    for th in turn_hashes:
        h.update(bytes.fromhex(th))
    return h.hexdigest()

# ------------------------------------------------------------
# Run deterministic dialog
# ------------------------------------------------------------
def run_dialog(user_turns):
    tok, model = load_model_and_tokenizer()  # persisted within process
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

    result = run_dialog(USER_TURNS)

    print("=== FORENSIC DETERMINISTIC MULTI-TURN RUN (FAST) ===")
    print(f"Python: {sys.version.split()[0]} | Torch: {torch.__version__} | Transformers: {AutoTokenizer.__module__.split('.')[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Model: {MODEL_ID}\n")

    for t in result["turns"]:
        print(f"\n[Turn {t['turn']}]")
        print(f"User: {t['user']}")
        print(f"Assistant: {t['assistant']}")
        print(f"HASH: {t['token_hash']} | LEN: {t['len_tokens']} | TIME: {t['runtime_s']:.2f}s")

    print(f"\nCONVERSATION_HASH: {result['conversation_hash']}")
    print(f"TOTAL_RUNTIME: {result['total_time']:.2f} seconds")

    # ---- Auto-log ----
    os.makedirs("logs", exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    safe_model = MODEL_ID.replace("/", "-")
    log_name = f"{script_name}__{safe_model}__{result['conversation_hash'][:12]}__{result['total_time']:.2f}s.log"
    log_path = os.path.join("logs", log_name)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Script: {script_name}\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Torch: {torch.__version__}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Threads: {NUM_THREADS}\n")
        f.write(f"CONVERSATION_HASH: {result['conversation_hash']}\n")
        f.write(f"Total Runtime: {result['total_time']:.2f} seconds\n\n")
        for t in result["turns"]:
            f.write(f"[Turn {t['turn']}]\n")
            f.write(f"User: {t['user']}\n")
            f.write(f"Assistant: {t['assistant']}\n")
            f.write(f"HASH: {t['token_hash']} | LEN: {t['len_tokens']} | TIME: {t['runtime_s']:.2f}s\n\n")

    print(f"\n[✔] Forensic fast log written to: {log_path}")
