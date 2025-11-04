# Runlock-AI 🔒  
**Deterministic AI Test Framework**

Runlock-AI is an open, verifiable framework for testing large language models deterministically — token-for-token, bit-for-bit.

## 🌱 Philosophy
Every model execution should be:
- **Reproducible** – same prompt → same output, every time.  
- **Transparent** – seeds, environment, and system states are logged automatically.  
- **Auditable** – outputs hashed for proof, with no hidden stochastic behavior.  

Runlock-AI’s mission is to make determinism a **first-class engineering primitive** for AI systems.

---

## ⚙️ Structure

| File | Description |
|------|--------------|
| `runlock_logger.py` | Modular JSON-based logging utility (shared across all versions). |
| `det_core_deterministic_base_cpu.py` | First deterministic single-turn proof-of-concept. |
| `det_core_deterministic_fast_cpu.py` | Optimized single-turn CPU version. |
| `det_multi_turn_baseline_cpu_forensic.py` | Forensic multi-turn baseline (first verified reproducible chat). |
| `det_multi_turn_baseline_cpu_forensic_fast.py` | Thread-tuned fast version, same hashes. |
| `det_multi_turn_baseline_cpu_forensic_fast_v2.py` | Modular JSON-logged version, verified deterministic. |

---

## 🧪 Reproducibility Test

Run a two-turn deterministic session:

```bash
python src/det_multi_turn_baseline_cpu_forensic_fast_v2.py
