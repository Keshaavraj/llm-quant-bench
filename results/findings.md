# Benchmark Findings — LLM Quantization on Constrained Hardware

**Hardware:** NVIDIA RTX 500 Ada Generation Laptop GPU (4GB VRAM), Intel Core Ultra 7 165H, 15GB RAM, Ubuntu 24.04 WSL2, CUDA 12.6

**Model:** Llama-3.2-3B-Instruct (primary), Mistral-7B-Instruct-v0.3 (secondary)

**Prompt:** Fixed prompt asking about quantization vs pruning in LLMs (same across all runs)

**Metric:** Average over 3 runs — tokens/second (generation throughput), peak VRAM (MB)

---

## Results Summary

### GGUF (llama.cpp) — Llama-3.2-3B-Instruct

| Format   | Tok/s | VRAM (MB) | Notes |
|----------|-------|-----------|-------|
| Q3_K_L   | 33.2   | 2515     | Most compressed |
| Q4_K_M   | 35.7   | 2711     | Sweet spot |
| Q5_K_M   | 37.0  | 2999      | Higher quality |
| Q8_0     | 25.6   | 3828     | Near lossless |

### Cross-Backend Comparison — Llama-3.2-3B-Instruct @ INT4

| Backend             | Tok/s | VRAM (MB) | Load time |
|---------------------|-------|-----------|-----------|
| GGUF Q4_K_M (llama.cpp) | 35.7 | 2711 | <1s |
| bitsandbytes NF4    | 12.2 | 2361 | 6.7s |
| AutoAWQ INT4        | 5.0 | 3027 | 6.6s |

### Mistral-7B-Instruct-v0.3 (GGUF only)

| Format   | Tok/s | VRAM (MB) |
|----------|-------|-----------|
| Q3_K_M   | 7.9   | 3845 |
| Q4_K_M   | OOM   | — |

---

## Key Findings

### 1. llama.cpp GGUF dominates inference speed on constrained hardware
GGUF Q4_K_M delivered **35.7 tok/s** — roughly **3x faster than bitsandbytes** and **7x faster than AWQ**.
llama.cpp has hand-written CUDA kernels that operate directly on quantized weights without
dequantizing first, which is why it outperforms Python-based alternatives so significantly.

### 2. More bits ≠ more speed (in GGUF)
Q5_K_M (37.0 tok/s) was slightly faster than Q4_K_M (35.7 tok/s) despite using more bits.
Q8_0 dropped to 25.6 tok/s — at 8 bits per weight, the model no longer fits as efficiently
in GPU cache, causing more memory bandwidth pressure. The sweet spot on this GPU is Q4–Q5.

### 3. bitsandbytes uses the least VRAM
NF4 with double quantization used only **2361 MB** — 350MB less than GGUF Q4_K_M.
The tradeoff: bitsandbytes dequantizes weights to float16 before matrix multiplication,
adding overhead that kills throughput. It's designed for QLoRA fine-tuning, not inference.

### 4. 7B model barely fits (and Q4 doesn't)
Mistral-7B Q3_K_M ran at 7.9 tok/s using 3845 MB — close to the 4GB limit.
Q4_K_M (4.1GB model + KV cache overhead) exceeded VRAM and was skipped.
This demonstrates the hard constraint of 4GB VRAM for 7B+ models.

### 5. AWQ underperformed due to missing CUDA extension
AutoAWQ ran at only 5.0 tok/s because the `awq_ext` optimized CUDA kernel failed to install,
and layer fusion was skipped. In production deployments (vLLM), AWQ is competitive with GGUF.
The AutoAWQ library itself is now deprecated — AWQ support has moved to vLLM's llm-compressor.

---

## Practical Recommendations

| Use case | Recommended format |
|----------|--------------------|
| Local chatbot / daily use | GGUF Q4_K_M — best speed/quality balance |
| Maximum quality within 4GB | GGUF Q5_K_M |
| Fine-tuning (QLoRA) | bitsandbytes NF4 — lower VRAM, training-friendly |
| Production serving (multi-user) | vLLM with AWQ — not benchmarked here |
| Running 7B models on 4GB VRAM | GGUF Q3_K_M only — barely fits |

---

## Plots

- `plots/throughput_comparison.png` — tok/s across all models and backends
- `plots/vram_comparison.png` — peak VRAM with 4GB limit marked
- `plots/throughput_vs_vram.png` — efficiency frontier scatter plot
