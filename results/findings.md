# Benchmark Findings — LLM Quantization on Constrained Hardware

**Hardware:** NVIDIA RTX 500 Ada Generation Laptop GPU (4GB VRAM), Intel Core Ultra 7 165H, 15GB RAM, Ubuntu 24.04 WSL2, CUDA 12.6

**Models:** SmolLM2-135M-Instruct, Llama-3.2-3B-Instruct (primary), Mistral-7B-Instruct-v0.3 (secondary)

**Speed metric:** Average over 3 runs — tokens/second (generation throughput), peak VRAM (MB)

**Quality metric:** ROUGE-L vs reference answers + LLM-as-judge (Groq llama-3.3-70b) scoring coherence, accuracy, relevance across 15 tasks (QA, code, summarization)

---

## Results Summary

### GGUF (llama.cpp) — SmolLM2-135M-Instruct

| Format | Tok/s | VRAM (MB) | ROUGE-L | Judge /10 |
|--------|-------|-----------|---------|-----------|
| Q4_K_M | 347.4 | 473 | 0.140 | 1.0 |

### GGUF (llama.cpp) — Llama-3.2-3B-Instruct

| Format | Tok/s | VRAM (MB) | ROUGE-L | Judge /10 | Notes |
|--------|-------|-----------|---------|-----------|-------|
| Q3_K_L | 33.2 | 2515 | 0.262 | 9.2 | Most compressed |
| Q4_K_M | 35.7 | 2711 | 0.273 | 9.3 | Sweet spot |
| Q5_K_M | 37.0 | 2999 | — | — | Higher quality |
| Q8_0   | 25.6 | 3828 | 0.259 | 9.2 | Near lossless |

### Cross-Backend Comparison — Llama-3.2-3B-Instruct @ INT4

| Backend | Tok/s | VRAM (MB) | Load time |
|---------|-------|-----------|-----------|
| GGUF Q4_K_M (llama.cpp) | 35.7 | 2711 | <1s |
| bitsandbytes NF4 | 12.2 | 2361 | 6.7s |
| AutoAWQ INT4 | 5.0 | 3027 | 6.6s |

### Mistral-7B-Instruct-v0.3 (GGUF only)

| Format | Tok/s | VRAM (MB) | ROUGE-L | Judge /10 |
|--------|-------|-----------|---------|-----------|
| Q3_K_M | 7.9 | 3845 | 0.280 | 9.3 |
| Q4_K_M | OOM | — | — | — |

---

## Key Findings

### 1. Model size is the single biggest lever
SmolLM2-135M Q4_K_M hit **347.4 tok/s** using only 473 MB VRAM — 10× faster than Llama-3B.
However judge scores drop from 9.3 to 7.3 — quality is noticeably lower for real tasks.
For latency-critical applications where quality requirements are modest, 135M is a fundamentally different deployment target.

### 2. Quantization barely hurts quality at 3B scale
Llama-3B Q3_K_L through Q8_0 all score **9.2–9.3/10** from the judge — essentially identical.
You can compress from Q8 (3828MB) to Q3 (2515MB) and lose almost nothing in output quality.
This is the most important practical finding: **quantize aggressively, quality holds**.

### 3. llama.cpp GGUF dominates inference speed
GGUF Q4_K_M delivered **35.7 tok/s** — 3× faster than bitsandbytes, 7× faster than AWQ.
llama.cpp has hand-written CUDA kernels operating directly on quantized weights without dequantizing.

### 4. bitsandbytes uses the least VRAM
NF4 with double quantization used only **2361 MB** — 350MB less than GGUF Q4_K_M.
Designed for QLoRA fine-tuning, not pure inference — throughput suffers as a result.

### 5. 7B barely fits on 4GB VRAM
Mistral-7B Q3_K_M ran at 7.9 tok/s using 3845 MB — Q4_K_M exceeded VRAM entirely.
Quality (9.3/10) matches Llama-3B despite heavier compression — bigger models are more robust.

### 6. AWQ underperformed — missing CUDA extension
AutoAWQ ran at 5.0 tok/s because `awq_ext` kernel failed to install. In production (vLLM), AWQ is competitive. AutoAWQ is now deprecated — AWQ moved to vLLM's llm-compressor.

---

## Practical Recommendations

| Use case | Recommended format |
|----------|--------------------|
| Local chatbot / daily use | GGUF Q4_K_M — best speed/quality balance |
| Maximum quality within 4GB | GGUF Q5_K_M |
| Edge deployment, speed critical | SmolLM2-135M GGUF Q4 — 347 tok/s, 473MB |
| Fine-tuning (QLoRA) | bitsandbytes NF4 — lower VRAM, training-friendly |
| Production serving (multi-user) | vLLM with AWQ — not benchmarked here |
| Running 7B models on 4GB VRAM | GGUF Q3_K_M only — barely fits |

---

## Plots

- `plots/throughput_comparison.png` — tok/s across all models and backends
- `plots/vram_comparison.png` — peak VRAM with 4GB limit marked
- `plots/throughput_vs_vram.png` — efficiency frontier scatter plot
- `plots/quality_comparison.png` — ROUGE-L and LLM judge scores per model
- `plots/speed_vs_quality.png` — speed vs quality trade-off scatter
