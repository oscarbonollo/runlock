# Runlock

Runlock is a minimalist framework for producing deterministic, traceable execution fingerprints of neural language models.

Given a fixed interaction, it provides a lightweight set of tools to answer a simple question:

*"Did this model execute identically in this context?"*

Runlock approaches this by combining deterministic decoding, cryptographic hashing (SHA-256) of generated tokens, lightweight internal activation summaries (float probes), and structured JSON logging.

It is intentionally narrow in scope: a harness designed to make multi-turn model execution repeatable, comparable, and auditable across environments.

Runlock is an **execution fingerprinting framework**. The framework is inspired by ideas from systems determinism, numerical reproducibility, and emerging work on internal representation analysis in neural networks.

It combines output hashes and internal activation probes to capture evidence of *how* a model executed, rather than attempting to explain *why* a particular output was produced.

It is **not** intended as a benchmarking tool, an evaluation suite, or a substitute for interpretability research. Instead, it provides low-level, reproducible artifacts that can potentially support engineering, debugging, forensic analysis, and future research into model behavior.

---

## Philosophy
Imagine that you walk through a shallow water stream repeatedly. You have walked this path before, you can see the same rocks and pebbles, you can hear the same sound your footsteps make, and you can pick up the rocks and pebbles, and put them down where you found them.

Runlock attempts to take this analogy to the extreme when looking at the internal activation state, or the 'hidden' state of a neural network. It reaches down, picks up some 'pebbles' (float point values) to remind us of a path we have taken previously.

The output Runlock provides, much like the rocks, is something physical, something that your hardware is actually doing that is not fake, and is based on a real machine calculation.

If by some chance the stream is altered, and the pebbles are displaced, we should be able to observe this at even the slightest change.

By having such a tight view of 'what' a model did, it also is my opinion (and hope) by knowing how and when we ran a token which crossed the same waters, and picked up the same pebbles in the stream, we will start to learn and discover more about 'why' and 'how' perhaps models selected an answer in engineering terms.

---

## Principles

**Determinism first**  
Fixed seeds, fixed decoding strategy, fixed execution paths.

**Minimal surface area**  
No chat templates, no system prompts, and no implicit or untracked execution state.

**Auditable by default**  
Every run produces a structured log with hashes, timing, and environment metadata.

**Execution, not interpretation**  
Runlock fingerprints how a model executed, not necessarily what its outputs mean.

**Open Source and Modifiable**  
Please see included LICENSE file for details about how to use projects with an MIT License.

---

## Prerequisites

- **Python 3.10** (recommended; built and tested on **3.10.11**)
- Tested on **Windows 11** and **Ubuntu 22.04 (EC2)**
- 64-bit CPU (32-bit and alternative architectures are untested)
- **Strongly Recommended:** For Python use a virtual environment (do not install into system Python)
### Notes on determinism
- **CPU runs** are intended to be maximally reproducible.
- **GPU runs** are best-effort deterministic: exact matches are strongest when GPU model + driver + CUDA/PyTorch stack are held constant.

## Minimum Python Packages Required

Pinned core dependencies:

- numpy==2.2.6
- transformers==4.56.1
- torch==2.9.0+cu128
- accelerate (required when using `device_map` / loading patterns that rely on it)

See the Makefile for the canonical install path.

---

## Core Scripts

This repository intentionally contains only two canonical execution paths:

| File                       | Description |
|----------------------------|-------------|
| `runlock_cpu_baseline_.py` | CPU reference runner (maximal determinism; slower). |
| `runlock_gpu_baseline_.py` | GPU reference runner (best-effort determinism; faster). |

**GPU note:** A modern NVIDIA GPU is recommended. Minimum VRAM will depend on model choice.



---
## Optional Scripts
These are not required to run Runlock, just additional utilities and information what these are for.

| File             | Description                                                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `env_details.py` | A non-PII script that records important details about your machine and python environment                                                                               |
| `Makefile`       | This is for users of the Make unix program and will install all Runlock dependencies. I will only provide best-effort support but tested on Windows and Ubuntu working. |
---

## What a Run Produces

Each run generates the following artifacts:

- Per-turn SHA-256 token hashes  
- A conversation hash (hash of per-turn hashes)  
- Per-turn runtime measurements  
- Environment metadata (Python, Torch, platform, device)  
- A lightweight internal activation fingerprint (float probe)  
- A structured JSON log written to `./logs`

Combined, these artifacts form an execution fingerprint that is logged, traceable and auditable.

---

## Usage
Run the CPU script:

```bash
python runlock_cpu_baseline_.py
```

Run the GPU script:

```bash
python runlock_gpu_baseline_.py
```

Once the run completes, a logfile will be placed in ./logs with the filename in the format <script.py><modelname><hash><runtime_in_seconds>.json

---

## What an example logfile looks like 
~~~
{
  "script": "runlock_cpu_baseline_",
  "model": "microsoft/phi-2",
  "conversation_hash": "695446880f3aefec5159c6e121b855a18fc728b111675649d77583f2ff7ef8d8",
  "total_runtime_s": 5.214338800000405,
  "meta": {
    "python_version": "3.10.11",
    "torch_version": "2.9.0+cu128",
    "platform": "Windows-10-10.0.26200-SP0",
    "threads": {
      "OMP_NUM_THREADS": "4",
      "MKL_NUM_THREADS": "4"
    },
    "model_id": "microsoft/phi-2",
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "pad_equals_eos": true
  },
  "turns": [
    {
      "turn": 1,
      "user": "What is 2+2?",
      "assistant": "User: What is 2+2?\nAssistant: 4\n",
      "token_hash": "68940631acc17dd51dc94c30b1c6bf8dab1670b51c08f7b8b4001028668379be",
      "len_tokens": 14,
      "runtime_s": 0.5136550999995961,
      "float_probe": {
        "layer": "last",
        "positions": {
          "start": {
            "idx": 0,
            "mean": -0.01291805,
            "std": 1.2212162,
            "l2": 61.79265213
          },
          "mid": {
            "idx": 7,
            "mean": -0.00372757,
            "std": 1.29427123,
            "l2": 65.48579407
          },
          "end": {
            "idx": 13,
            "mean": 0.04274917,
            "std": 0.91088468,
            "l2": 46.13825226
          }
        }
      }
    },
    {
      "turn": 2,
      "user": "Explain the reasoning in one sentence.",
      "assistant": "User: What is 2+2?\nAssistant: User: What is 2+2?\nAssistant: 4\n\nUser: Explain the reasoning in one sentence.\nAssistant: The sum of 2 and 2 is 4 because when you combine two objects with another two objects, you end up with a total of four objects.\n",
      "token_hash": "07af633e7f318e4f53692545417558395ab43024f6f3c1d62fe00e050b595004",
      "len_tokens": 68,
      "runtime_s": 4.700683700000809,
      "float_probe": {
        "layer": "last",
        "positions": {
          "start": {
            "idx": 0,
            "mean": -0.01291805,
            "std": 1.2212162,
            "l2": 61.79265213
          },
          "mid": {
            "idx": 34,
            "mean": 0.00210294,
            "std": 1.41968358,
            "l2": 71.83101654
          },
          "end": {
            "idx": 67,
            "mean": 0.04172997,
            "std": 0.96857154,
            "l2": 49.05173874
          }
        }
      }
    }
  ]
}
~~~