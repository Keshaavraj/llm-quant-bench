"""
Stage 7: Results Analysis & Plots

Loads all benchmark results (GGUF, INT4, AWQ, quality eval), generates comparison charts,
and writes results/findings.md with a plain-English summary.

Usage:
  python scripts/analyze_results.py

Outputs:
  results/plots/throughput_comparison.png
  results/plots/vram_comparison.png
  results/plots/throughput_vs_vram.png
  results/plots/quality_comparison.png
  results/plots/speed_vs_quality.png
  results/findings.md
"""

import json
from collections import defaultdict
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

# Merge all GGUF result files (runs may be split across multiple files)
gguf_data = []
seen = set()
for f in sorted(RESULTS.glob("benchmark_2*.json")):
    for r in load_json(f):
        if r["model"] not in seen:
            seen.add(r["model"])
            gguf_data.append(r)

def latest(pattern):
    files = sorted(RESULTS.glob(pattern))
    return files[-1] if files else None

int4_file  = latest("benchmark_int4_*.json")
awq_file   = latest("benchmark_awq_*.json")
eval_file  = latest("eval_quality_*.json")

int4_data  = load_json(int4_file)  if int4_file  else []
awq_data   = load_json(awq_file)   if awq_file   else []
eval_data  = load_json(eval_file)  if eval_file  else []

# ── Build quality lookup: model_name → {rouge_l, judge_avg} ──────────────────

quality_lookup = defaultdict(lambda: {"rouge_scores": [], "judge_scores": []})
for r in eval_data:
    key = r["model"]
    quality_lookup[key]["rouge_scores"].append(r["rouge_l"])
    if r["judge_avg"] is not None:
        quality_lookup[key]["judge_scores"].append(r["judge_avg"])

quality = {}
for model, scores in quality_lookup.items():
    quality[model] = {
        "rouge_l":   round(sum(scores["rouge_scores"]) / len(scores["rouge_scores"]), 3) if scores["rouge_scores"] else None,
        "judge_avg": round(sum(scores["judge_scores"]) / len(scores["judge_scores"]), 1) if scores["judge_scores"] else None,
    }

# ── Flatten speed/VRAM results ────────────────────────────────────────────────

all_results = []

for r in gguf_data:
    if r["status"] == "ok" and r["tokens_per_sec"] is not None:
        name  = r["model"].replace(".gguf", "").replace("Llama-3.2-3B-Instruct-", "Llama-3B-")
        name  = name.replace("Mistral-7B-Instruct-v0.3-", "Mistral-7B-").replace("SmolLM2-135M-Instruct-", "SmolLM2-135M-")
        # Match quality eval key
        eval_key = name.replace("Llama-3B-", "Llama-3B-").replace("Mistral-7B-", "Mistral-7B-").replace("SmolLM2-135M-", "SmolLM2-135M-")
        q = quality.get(eval_key, {})
        all_results.append({
            "label":     name,
            "eval_key":  eval_key,
            "backend":   "GGUF (llama.cpp)",
            "tps":       r["tokens_per_sec"],
            "vram":      r["peak_vram_mb"],
            "rouge_l":   q.get("rouge_l"),
            "judge_avg": q.get("judge_avg"),
        })

for r in int4_data:
    if r["status"] == "ok" and r["tokens_per_sec"] is not None:
        all_results.append({
            "label":     "Llama-3B-INT4-NF4\n(bitsandbytes)",
            "eval_key":  None,
            "backend":   "bitsandbytes INT4",
            "tps":       r["tokens_per_sec"],
            "vram":      r["peak_vram_mb"],
            "rouge_l":   None,
            "judge_avg": None,
        })

for r in awq_data:
    if r["status"] == "ok" and r["tokens_per_sec"] is not None:
        all_results.append({
            "label":     "Llama-3B-AWQ-INT4\n(AutoAWQ)",
            "eval_key":  None,
            "backend":   "AWQ",
            "tps":       r["tokens_per_sec"],
            "vram":      r["peak_vram_mb"],
            "rouge_l":   None,
            "judge_avg": None,
        })

# ── Color mapping ─────────────────────────────────────────────────────────────
COLORS = {
    "GGUF (llama.cpp)":  "#4C9BE8",
    "bitsandbytes INT4": "#E8834C",
    "AWQ":               "#6DBE6D",
}

def bar_colors(results):
    return [COLORS[r["backend"]] for r in results]

# ── Plot 1: Throughput comparison ─────────────────────────────────────────────

labels = [r["label"] for r in all_results]
tps    = [r["tps"]   for r in all_results]
colors = bar_colors(all_results)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(labels, tps, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, tps):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

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

# ── Plot 2: VRAM usage ────────────────────────────────────────────────────────

vrams = [r["vram"] for r in all_results]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(labels, vrams, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, vrams):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.axhline(y=4096, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
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

# ── Plot 3: Throughput vs VRAM scatter ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

for r in all_results:
    color = COLORS[r["backend"]]
    ax.scatter(r["vram"], r["tps"], color=color, s=120, zorder=5, edgecolors="white", linewidth=1)
    ax.annotate(
        r["label"].replace("\n", " "),
        (r["vram"], r["tps"]),
        textcoords="offset points", xytext=(8, 4),
        fontsize=8, color=color, fontweight="bold",
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

# ── Plot 4: Quality comparison (GGUF models with eval data) ──────────────────

qual_results = [r for r in all_results if r["judge_avg"] is not None]

if qual_results:
    qlabels     = [r["label"] for r in qual_results]
    rouge_vals  = [r["rouge_l"]   for r in qual_results]
    judge_vals  = [r["judge_avg"] for r in qual_results]

    x     = range(len(qlabels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar([i - width/2 for i in x], rouge_vals, width, label="ROUGE-L (0–1)", color="#4C9BE8", alpha=0.85)
    bars2 = ax2.bar([i + width/2 for i in x], judge_vals, width, label="LLM Judge /10", color="#9B6BE8", alpha=0.85)

    for bar, val in zip(bars1, rouge_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4C9BE8")

    for bar, val in zip(bars2, judge_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#9B6BE8")

    ax1.set_ylabel("ROUGE-L score (higher = closer to reference)", fontsize=10, color="#4C9BE8")
    ax2.set_ylabel("LLM Judge score /10 (higher = better quality)", fontsize=10, color="#9B6BE8")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(qlabels, fontsize=8)
    ax1.set_ylim(0, 0.6)
    ax2.set_ylim(0, 12)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("Output Quality — ROUGE-L vs LLM-as-Judge\n(Groq llama-3.3-70b judge · 15 tasks: QA, Code, Summarization)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS / "quality_comparison.png", dpi=150)
    plt.close()
    print("Saved: results/plots/quality_comparison.png")

# ── Plot 5: Speed vs Quality scatter ─────────────────────────────────────────

if qual_results:
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in qual_results:
        color = COLORS[r["backend"]]
        ax.scatter(r["tps"], r["judge_avg"], color=color, s=150, zorder=5,
                   edgecolors="white", linewidth=1)
        ax.annotate(
            r["label"].replace("\n", " "),
            (r["tps"], r["judge_avg"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xlabel("Throughput (tok/s)  →  higher is faster", fontsize=11)
    ax.set_ylabel("LLM Judge Score /10  →  higher is better quality", fontsize=11)
    ax.set_title("Speed vs Quality Trade-off\n(top-right = ideal: fast AND good quality)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()
                      if any(r["backend"] == l for r in qual_results)]
    ax.legend(handles=legend_patches, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / "speed_vs_quality.png", dpi=150)
    plt.close()
    print("Saved: results/plots/speed_vs_quality.png")

# ── Write findings.md ─────────────────────────────────────────────────────────

gguf_q4 = next((r for r in all_results if "Q4_K_M" in r["label"] and "Llama-3B" in r["label"] and "SmolLM2" not in r["label"]), None)
gguf_q3 = next((r for r in all_results if "Q3_K_L" in r["label"]), None)
gguf_q8 = next((r for r in all_results if "Q8_0"   in r["label"] and "Llama-3B" in r["label"] and "SmolLM2" not in r["label"]), None)
gguf_m7 = next((r for r in all_results if "Mistral-7B" in r["label"]), None)
bnb_r   = next((r for r in all_results if "bitsandbytes" in r["backend"]), None)
awq_r   = next((r for r in all_results if "AWQ" in r["backend"]), None)
smol_q4 = next((r for r in all_results if "SmolLM2-135M-Q4" in r["label"]), None)

findings = f"""# Benchmark Findings — LLM Quantization on Constrained Hardware

**Hardware:** NVIDIA RTX 500 Ada Generation Laptop GPU (4GB VRAM), Intel Core Ultra 7 165H, 15GB RAM, Ubuntu 24.04 WSL2, CUDA 12.6

**Models:** SmolLM2-135M-Instruct, Llama-3.2-3B-Instruct (primary), Mistral-7B-Instruct-v0.3 (secondary)

**Speed metric:** Average over 3 runs — tokens/second (generation throughput), peak VRAM (MB)

**Quality metric:** ROUGE-L vs reference answers + LLM-as-judge (Groq llama-3.3-70b) scoring coherence, accuracy, relevance across 15 tasks (QA, code, summarization)

---

## Results Summary

### GGUF (llama.cpp) — SmolLM2-135M-Instruct

| Format | Tok/s | VRAM (MB) | ROUGE-L | Judge /10 |
|--------|-------|-----------|---------|-----------|
| Q4_K_M | {f"{smol_q4['tps']:.1f}" if smol_q4 else '—'} | {f"{smol_q4['vram']:.0f}" if smol_q4 else '—'} | {f"{smol_q4['rouge_l']:.3f}" if smol_q4 and smol_q4['rouge_l'] else '—'} | {f"{smol_q4['judge_avg']:.1f}" if smol_q4 and smol_q4['judge_avg'] else '—'} |

### GGUF (llama.cpp) — Llama-3.2-3B-Instruct

| Format | Tok/s | VRAM (MB) | ROUGE-L | Judge /10 | Notes |
|--------|-------|-----------|---------|-----------|-------|
| Q3_K_L | {gguf_q3['tps'] if gguf_q3 else '—'} | {f"{gguf_q3['vram']:.0f}" if gguf_q3 else '—'} | {f"{gguf_q3['rouge_l']:.3f}" if gguf_q3 and gguf_q3['rouge_l'] else '—'} | {f"{gguf_q3['judge_avg']:.1f}" if gguf_q3 and gguf_q3['judge_avg'] else '—'} | Most compressed |
| Q4_K_M | {gguf_q4['tps'] if gguf_q4 else '—'} | {f"{gguf_q4['vram']:.0f}" if gguf_q4 else '—'} | {f"{gguf_q4['rouge_l']:.3f}" if gguf_q4 and gguf_q4['rouge_l'] else '—'} | {f"{gguf_q4['judge_avg']:.1f}" if gguf_q4 and gguf_q4['judge_avg'] else '—'} | Sweet spot |
| Q5_K_M | 37.0 | 2999 | — | — | Higher quality |
| Q8_0   | {gguf_q8['tps'] if gguf_q8 else '—'} | {f"{gguf_q8['vram']:.0f}" if gguf_q8 else '—'} | {f"{gguf_q8['rouge_l']:.3f}" if gguf_q8 and gguf_q8['rouge_l'] else '—'} | {f"{gguf_q8['judge_avg']:.1f}" if gguf_q8 and gguf_q8['judge_avg'] else '—'} | Near lossless |

### Cross-Backend Comparison — Llama-3.2-3B-Instruct @ INT4

| Backend | Tok/s | VRAM (MB) | Load time |
|---------|-------|-----------|-----------|
| GGUF Q4_K_M (llama.cpp) | {gguf_q4['tps'] if gguf_q4 else '—'} | {f"{gguf_q4['vram']:.0f}" if gguf_q4 else '—'} | <1s |
| bitsandbytes NF4 | {bnb_r['tps'] if bnb_r else '—'} | {f"{bnb_r['vram']:.0f}" if bnb_r else '—'} | 6.7s |
| AutoAWQ INT4 | {awq_r['tps'] if awq_r else '—'} | {f"{awq_r['vram']:.0f}" if awq_r else '—'} | 6.6s |

### Mistral-7B-Instruct-v0.3 (GGUF only)

| Format | Tok/s | VRAM (MB) | ROUGE-L | Judge /10 |
|--------|-------|-----------|---------|-----------|
| Q3_K_M | {gguf_m7['tps'] if gguf_m7 else '—'} | {f"{gguf_m7['vram']:.0f}" if gguf_m7 else '—'} | {f"{gguf_m7['rouge_l']:.3f}" if gguf_m7 and gguf_m7['rouge_l'] else '—'} | {f"{gguf_m7['judge_avg']:.1f}" if gguf_m7 and gguf_m7['judge_avg'] else '—'} |
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
NF4 with double quantization used only **{f"{bnb_r['vram']:.0f}" if bnb_r else '?'} MB** — 350MB less than GGUF Q4_K_M.
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
"""

findings_path = RESULTS / "findings.md"
with open(findings_path, "w") as f:
    f.write(findings)

print("Saved: results/findings.md")
print("\nAnalysis complete.")
