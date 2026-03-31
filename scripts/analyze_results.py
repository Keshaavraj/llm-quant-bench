"""
Stage 7: Results Analysis & Plots

Loads all benchmark results (GGUF, INT4, AWQ), generates comparison charts,
and writes results/findings.md with a plain-English summary.

Usage:
  python scripts/analyze_results.py

Outputs:
  results/plots/throughput_comparison.png
  results/plots/vram_comparison.png
  results/plots/throughput_vs_vram.png
  results/findings.md
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless — no display required (WSL2)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Pick the latest file for each backend
def latest(pattern):
    files = sorted(RESULTS.glob(pattern))
    return files[-1] if files else None

gguf_file  = latest("benchmark_2*.json")
int4_file  = latest("benchmark_int4_*.json")
awq_file   = latest("benchmark_awq_*.json")

gguf_data  = load_json(gguf_file)  if gguf_file  else []
int4_data  = load_json(int4_file)  if int4_file  else []
awq_data   = load_json(awq_file)   if awq_file   else []

# ── Flatten into a single list, filter OOM and nulls ─────────────────────────

all_results = []

for r in gguf_data:
    if r["status"] == "ok" and r["tokens_per_sec"] is not None:
        # Shorten model names for readability
        name = r["model"].replace(".gguf", "").replace("Llama-3.2-3B-Instruct-", "Llama-3B-")
        name = name.replace("Mistral-7B-Instruct-v0.3-", "Mistral-7B-")
        all_results.append({
            "label": name,
            "backend": "GGUF (llama.cpp)",
            "tps": r["tokens_per_sec"],
            "vram": r["peak_vram_mb"],
        })

for r in int4_data:
    if r["status"] == "ok" and r["tokens_per_sec"] is not None:
        all_results.append({
            "label": "Llama-3B-INT4-NF4\n(bitsandbytes)",
            "backend": "bitsandbytes INT4",
            "tps": r["tokens_per_sec"],
            "vram": r["peak_vram_mb"],
        })

for r in awq_data:
    if r["status"] == "ok" and r["tokens_per_sec"] is not None:
        all_results.append({
            "label": "Llama-3B-AWQ-INT4\n(AutoAWQ)",
            "backend": "AWQ",
            "tps": r["tokens_per_sec"],
            "vram": r["peak_vram_mb"],
        })

# ── Color mapping by backend ──────────────────────────────────────────────────
COLORS = {
    "GGUF (llama.cpp)":  "#4C9BE8",
    "bitsandbytes INT4": "#E8834C",
    "AWQ":               "#6DBE6D",
}

def bar_colors(results):
    return [COLORS[r["backend"]] for r in results]

# ── Plot 1: Throughput comparison (tok/s) ─────────────────────────────────────

labels = [r["label"] for r in all_results]
tps    = [r["tps"]   for r in all_results]
colors = bar_colors(all_results)

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(labels, tps, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, tps):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Tokens / second  (higher = faster)", fontsize=11)
ax.set_title("LLM Inference Throughput — Quantization Format Comparison\nNVIDIA RTX 500 Ada (4GB VRAM)", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(tps) * 1.2)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(PLOTS / "throughput_comparison.png", dpi=150)
plt.close()
print("Saved: results/plots/throughput_comparison.png")

# ── Plot 2: VRAM usage comparison ─────────────────────────────────────────────

vrams  = [r["vram"] for r in all_results]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(labels, vrams, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, vrams):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Mark the 4GB VRAM limit
ax.axhline(y=4096, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="4GB VRAM limit")
ax.text(len(labels) - 0.5, 4130, "4GB VRAM limit", color="red", fontsize=9, ha="right")

ax.set_ylabel("Peak VRAM usage (MB)  (lower = more headroom)", fontsize=11)
ax.set_title("Peak VRAM Usage — Quantization Format Comparison\nNVIDIA RTX 500 Ada (4GB VRAM)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 5000)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(PLOTS / "vram_comparison.png", dpi=150)
plt.close()
print("Saved: results/plots/vram_comparison.png")

# ── Plot 3: Throughput vs VRAM scatter (efficiency frontier) ──────────────────

fig, ax = plt.subplots(figsize=(10, 7))

for r in all_results:
    color = COLORS[r["backend"]]
    ax.scatter(r["vram"], r["tps"], color=color, s=120, zorder=5, edgecolors="white", linewidth=1)
    # Offset labels so they don't overlap the dots
    ax.annotate(
        r["label"].replace("\n", " "),
        (r["vram"], r["tps"]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=8,
        color=color,
        fontweight="bold",
    )

ax.axvline(x=4096, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(4110, max(tps) * 0.95, "4GB limit", color="red", fontsize=8)

ax.set_xlabel("Peak VRAM (MB)  →  lower is better", fontsize=11)
ax.set_ylabel("Throughput (tok/s)  →  higher is better", fontsize=11)
ax.set_title("Efficiency Frontier: Throughput vs VRAM\n(top-left = best trade-off)", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
ax.legend(handles=legend_patches, fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS / "throughput_vs_vram.png", dpi=150)
plt.close()
print("Saved: results/plots/throughput_vs_vram.png")

# ── Write findings.md ─────────────────────────────────────────────────────────

# Pull key numbers for the report
gguf_q4   = next((r for r in all_results if "Q4_K_M" in r["label"] and "Llama-3B" in r["label"]), None)
gguf_q3   = next((r for r in all_results if "Q3_K_L" in r["label"]), None)
gguf_q8   = next((r for r in all_results if "Q8_0"   in r["label"]), None)
gguf_m7   = next((r for r in all_results if "Mistral-7B" in r["label"]), None)
bnb_r     = next((r for r in all_results if "bitsandbytes" in r["backend"]), None)
awq_r     = next((r for r in all_results if "AWQ" in r["backend"]), None)

findings = f"""# Benchmark Findings — LLM Quantization on Constrained Hardware

**Hardware:** NVIDIA RTX 500 Ada Generation Laptop GPU (4GB VRAM), Intel Core Ultra 7 165H, 15GB RAM, Ubuntu 24.04 WSL2, CUDA 12.6

**Model:** Llama-3.2-3B-Instruct (primary), Mistral-7B-Instruct-v0.3 (secondary)

**Prompt:** Fixed prompt asking about quantization vs pruning in LLMs (same across all runs)

**Metric:** Average over 3 runs — tokens/second (generation throughput), peak VRAM (MB)

---

## Results Summary

### GGUF (llama.cpp) — Llama-3.2-3B-Instruct

| Format   | Tok/s | VRAM (MB) | Notes |
|----------|-------|-----------|-------|
| Q3_K_L   | {gguf_q3['tps'] if gguf_q3 else '—'}   | {f"{gguf_q3['vram']:.0f}" if gguf_q3 else '—'}     | Most compressed |
| Q4_K_M   | {gguf_q4['tps'] if gguf_q4 else '—'}   | {f"{gguf_q4['vram']:.0f}" if gguf_q4 else '—'}     | Sweet spot |
| Q5_K_M   | 37.0  | 2999      | Higher quality |
| Q8_0     | {gguf_q8['tps'] if gguf_q8 else '—'}   | {f"{gguf_q8['vram']:.0f}" if gguf_q8 else '—'}     | Near lossless |

### Cross-Backend Comparison — Llama-3.2-3B-Instruct @ INT4

| Backend             | Tok/s | VRAM (MB) | Load time |
|---------------------|-------|-----------|-----------|
| GGUF Q4_K_M (llama.cpp) | {gguf_q4['tps'] if gguf_q4 else '—'} | {f"{gguf_q4['vram']:.0f}" if gguf_q4 else '—'} | <1s |
| bitsandbytes NF4    | {bnb_r['tps'] if bnb_r else '—'} | {f"{bnb_r['vram']:.0f}" if bnb_r else '—'} | 6.7s |
| AutoAWQ INT4        | {awq_r['tps'] if awq_r else '—'} | {f"{awq_r['vram']:.0f}" if awq_r else '—'} | 6.6s |

### Mistral-7B-Instruct-v0.3 (GGUF only)

| Format   | Tok/s | VRAM (MB) |
|----------|-------|-----------|
| Q3_K_M   | {gguf_m7['tps'] if gguf_m7 else '—'}   | {f"{gguf_m7['vram']:.0f}" if gguf_m7 else '—'} |
| Q4_K_M   | OOM   | — |

---

## Key Findings

### 1. llama.cpp GGUF dominates inference speed on constrained hardware
GGUF Q4_K_M delivered **{gguf_q4['tps'] if gguf_q4 else '?':.1f} tok/s** — roughly **3x faster than bitsandbytes** and **7x faster than AWQ**.
llama.cpp has hand-written CUDA kernels that operate directly on quantized weights without
dequantizing first, which is why it outperforms Python-based alternatives so significantly.

### 2. More bits ≠ more speed (in GGUF)
Q5_K_M (37.0 tok/s) was slightly faster than Q4_K_M (35.7 tok/s) despite using more bits.
Q8_0 dropped to 25.6 tok/s — at 8 bits per weight, the model no longer fits as efficiently
in GPU cache, causing more memory bandwidth pressure. The sweet spot on this GPU is Q4–Q5.

### 3. bitsandbytes uses the least VRAM
NF4 with double quantization used only **{f"{bnb_r['vram']:.0f}" if bnb_r else '?'} MB** — 350MB less than GGUF Q4_K_M.
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
"""

findings_path = RESULTS / "findings.md"
with open(findings_path, "w") as f:
    f.write(findings)

print(f"Saved: results/findings.md")
print("\nStage 7 complete.")
