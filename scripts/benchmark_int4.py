"""
LLM Quantization Benchmark — bitsandbytes INT4 backend (Stage 5)

Loads models with 4-bit quantization via bitsandbytes (NF4) using HuggingFace
transformers and measures:
  - load_time_ms    : time to load and quantize model weights into VRAM
  - ttft_ms         : time to first token (prefill latency)
  - tokens_per_sec  : generation throughput
  - peak_vram_mb    : peak GPU memory during inference

Uses the same prompt, output format, and results schema as benchmark.py (GGUF)
so results can be compared directly.

Usage:
  python scripts/benchmark_int4.py                        # benchmark all INT4 models
  python scripts/benchmark_int4.py --runs 3               # 3 runs per model (averaged)
  python scripts/benchmark_int4.py --model Llama          # filter by name substring
  python scripts/benchmark_int4.py --prompt "Your prompt" # custom prompt

Results saved to:
  results/benchmark_int4_<timestamp>.csv
  results/benchmark_int4_<timestamp>.json
"""

import argparse
import csv
import json
import sys
import threading
import time
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rich.console import Console
from rich.table import Table

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ── Benchmark config ─────────────────────────────────────────────────────────
DEFAULT_PROMPT = (
    "Explain the difference between quantization and pruning in large language models. "
    "What are the trade-offs of each approach?"
)
DEFAULT_N_TOKENS = 200   # tokens to generate per run
DEFAULT_RUNS = 3         # runs per model — results are averaged

# ── INT4 models to benchmark ─────────────────────────────────────────────────
INT4_MODELS = [
    {
        "name": "Llama-3.2-3B-Instruct-INT4-NF4",
        "local_dir": "Llama-3.2-3B-Instruct-HF",   # subfolder under models/
    },
]

console = Console()


# ── VRAM poller ───────────────────────────────────────────────────────────────

class VramPoller:
    """Polls nvidia-smi in a background thread and tracks peak VRAM usage."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._stop.clear()
        self.peak_mb = 0
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                mb = int(out.split("\n")[0].strip())
                if mb > self.peak_mb:
                    self.peak_mb = mb
            except Exception:
                pass
            time.sleep(self.interval)


# ── Single model benchmark ────────────────────────────────────────────────────

def run_one(model, tokenizer, prompt, n_tokens, device):
    """Run one inference pass. Returns (ttft_ms, tokens_per_sec, peak_vram_mb)."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warm up CUDA before timing
    torch.cuda.synchronize()

    poller = VramPoller()
    poller.start()

    # ── Time to first token ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        # Generate just 1 token to measure TTFT (prefill latency)
        first_out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # ── Generation throughput ─────────────────────────────────────────────────
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    gen_secs = time.perf_counter() - t1

    poller.stop()

    # Count only newly generated tokens (exclude prompt)
    new_tokens = out.shape[1] - input_len
    tokens_per_sec = new_tokens / gen_secs if gen_secs > 0 else 0.0

    return ttft_ms, tokens_per_sec, poller.peak_mb


def benchmark_model(model_cfg, prompt, n_tokens, runs):
    """Load model with INT4 quantization and run N benchmark passes."""
    model_dir = MODELS_DIR / model_cfg["local_dir"]
    if not model_dir.exists():
        console.print(f"  [red]Model directory not found: {model_dir}[/red]")
        return None

    name = model_cfg["name"]

    # bitsandbytes NF4 config — 4-bit, double quantization for memory savings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 is best for LLMs
        bnb_4bit_use_double_quant=True,      # saves ~0.4 bits per param extra
        bnb_4bit_compute_dtype=torch.float16,
    )

    console.print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    console.print(f"  Loading model with INT4 (NF4) quantization...")
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        device_map="auto",           # auto-places layers on GPU/CPU
        torch_dtype=torch.float16,
    )
    torch.cuda.synchronize()
    load_time_ms = (time.perf_counter() - load_start) * 1000
    console.print(f"  Loaded in {load_time_ms/1000:.1f}s")

    model.eval()
    device = next(model.parameters()).device

    ttfts, tps_list, vrams = [], [], []

    for i in range(runs):
        console.print(f"    run {i+1}/{runs} ...", end=" ")
        sys.stdout.flush()

        ttft_ms, tps, peak_vram = run_one(model, tokenizer, prompt, n_tokens, device)

        ttfts.append(ttft_ms)
        tps_list.append(tps)
        vrams.append(peak_vram)

        console.print(f"[green]{tps:.1f} tok/s[/green]  TTFT {ttft_ms:.0f}ms")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    def avg(lst):
        vals = [v for v in lst if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        "model": name,
        "load_time_ms": round(load_time_ms, 1),
        "ttft_ms": avg(ttfts),
        "tokens_per_sec": avg(tps_list),
        "peak_vram_mb": avg(vrams),
        "runs": runs,
        "backend": "bitsandbytes-int4",
        "status": "ok",
        "note": "NF4, double_quant=True, compute_dtype=float16",
    }


# ── Results output ────────────────────────────────────────────────────────────

def save_results(results, timestamp):
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path  = RESULTS_DIR / f"benchmark_int4_{timestamp}.csv"
    json_path = RESULTS_DIR / f"benchmark_int4_{timestamp}.json"

    keys = ["model", "backend", "load_time_ms", "ttft_ms", "tokens_per_sec", "peak_vram_mb", "runs", "status", "note"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return csv_path, json_path


def print_table(results):
    table = Table(title="INT4 Benchmark Results", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Load (ms)", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tok/s", justify="right", style="green")
    table.add_column("VRAM (MB)", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    for r in results:
        status = r.get("status", "ok")
        if status == "OOM":
            table.add_row(r["model"], "—", "—", "—", "—", "[bold red]OOM[/bold red]")
        else:
            table.add_row(
                r["model"],
                f"{r['load_time_ms']:.0f}" if r["load_time_ms"] else "—",
                f"{r['ttft_ms']:.0f}"      if r["ttft_ms"]      else "—",
                f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] else "—",
                f"{r['peak_vram_mb']:.0f}" if r["peak_vram_mb"] else "—",
                "[green]ok[/green]",
            )

    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark INT4 quantized LLMs via bitsandbytes")
    parser.add_argument("--runs",   type=int, default=DEFAULT_RUNS,   help="Runs per model (averaged)")
    parser.add_argument("--tokens", type=int, default=DEFAULT_N_TOKENS, help="Tokens to generate")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,           help="Prompt to use")
    parser.add_argument("--model",  default=None,                     help="Filter models by name substring")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        console.print("[red]CUDA not available — INT4 benchmarking requires a GPU.[/red]")
        sys.exit(1)

    models = INT4_MODELS
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]
        if not models:
            console.print(f"[red]No models matched filter: {args.model}[/red]")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"\n[bold]LLM Quantization Benchmark — INT4 (bitsandbytes)[/bold]")
    console.print(f"Backend  : bitsandbytes NF4")
    console.print(f"GPU      : {torch.cuda.get_device_name(0)}")
    console.print(f"Models   : {len(models)}")
    console.print(f"Runs     : {args.runs} per model")
    console.print(f"Tokens   : {args.tokens}")
    console.print(f"Prompt   : {args.prompt[:80]}...")
    console.print()

    results = []
    for model_cfg in models:
        console.print(f"[bold cyan]{model_cfg['name']}[/bold cyan]")
        r = benchmark_model(model_cfg, args.prompt, args.tokens, args.runs)
        if r is None:
            console.print(f"  [red]SKIPPED[/red]\n")
        elif r.get("status") == "OOM":
            results.append(r)
            console.print(f"  [bold red]OOM[/bold red]\n")
        else:
            results.append(r)
            console.print(
                f"  → Load {r['load_time_ms']:.0f}ms  |  TTFT {r['ttft_ms']:.0f}ms  |  "
                f"{r['tokens_per_sec']:.1f} tok/s  |  {r['peak_vram_mb']:.0f} MB VRAM\n"
            )

    if not results:
        console.print("[red]No results collected.[/red]")
        sys.exit(1)

    print_table(results)

    csv_path, json_path = save_results(results, timestamp)
    console.print(f"\nSaved: [green]{csv_path}[/green]")
    console.print(f"Saved: [green]{json_path}[/green]")


if __name__ == "__main__":
    main()
