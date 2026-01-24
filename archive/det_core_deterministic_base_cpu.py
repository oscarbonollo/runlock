# -*- coding: utf-8 -*-
# det_core_cpu.py — deterministic CPU-only run

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Disable all GPUs
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random, hashlib, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED     = 42
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT   = "What is 2+2?"
MAX_NEW  = 64

# ---- Seeds + deterministic flags ----
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---- Load model/tokenizer (CPU only) ----
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    dtype=torch.float32,     # use dtype=… to silence deprecation warning
)
model.eval()

# ---- Generate ----
enc = tok(PROMPT.strip(), return_tensors="pt")
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

tokens = out.sequences[0].detach().cpu().numpy().astype(np.int32)
text   = tok.decode(out.sequences[0], skip_special_tokens=True)

# ---- Stable hash ----
token_hash = hashlib.sha256(tokens.tobytes()).hexdigest()

print("TOKEN_HASH:", token_hash)
print("LEN_TOKENS:", len(tokens))
print("OUTPUT:\n", text)
