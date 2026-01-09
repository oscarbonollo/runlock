# Runlock

Deterministic Execution Fingerprinting for Neural Models

Runlock is a minimal framework for producing deterministic, auditable execution fingerprints of neural language models.

Given a fixed interaction, it answers a single question: *did this model execute identically in this context?*

It does so by combining deterministic decoding, cryptographic hashing of outputs, lightweight internal activation fingerprints, and structured JSON logging.

It is a harness built to be lightweight and minimal in scope.

Runlock **is** an execution fingerprinting framework that combines output hashes and internal activation probes to support reproducibility and forensic analysis of LLM executions, with an emphasis on determinism and traceability.  
Runlock **is not** intended as a benchmarking tool, an evaluation suite, or a substitute for an interpretability framework.

---

## Philosophy

Runlock is a framework built on a rigid set of principles:

**Determinism first**  
Fixed seeds, fixed decoding strategy, fixed execution paths.

**Minimal surface area**  
No chat templates, no system prompts, and no implicit or untracked execution state.

**Auditable by default**  
Every run produces a structured log with hashes, timing, and environment metadata.

**Execution, not interpretation**  
Runlock fingerprints how a model executed, not necessarily what its outputs mean.

---

## Core Scripts

This repository intentionally contains only two canonical execution paths:

| File | Description |
|------|-------------|
| `det_multi_turn_baseline_cpu_forensic_fast_v4_float.py` | CPU reference implementation. Maximally deterministic and cross-platform. |
| `det_multi_turn_baseline_gpu_forensic_fast_v1_float.py` | GPU reference implementation. High-performance, best-effort deterministic. |

All other files and earlier experiments have been archived.

---

## What a Run Produces

Each run generates the following artifacts:

- Per-turn SHA-256 token hashes  
- A conversation hash (hash of per-turn hashes)  
- Per-turn runtime measurements  
- Environment metadata (Python, Torch, platform, device)  
- A lightweight internal activation fingerprint (float probe)  
- A structured JSON log written to `./logs`

Combined, these artifacts form an execution fingerprint.

---

## Usage
Run the CPU reference baseline:

```bash
python runlock_cpu_baseline_.py
```

Run the GPU reference baseline:

```bash
python runlock_gpu_baseline_.py
```