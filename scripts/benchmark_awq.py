"""
LLM Quantization Benchmark — AWQ backend (Stage 6)

Loads pre-quantized AWQ (Activation-aware Weight Quantization) models using
the AutoAWQ library and measures:
  - load_time_ms    : time to load the AWQ model into VRAM
  - ttft_ms         : time to first token (prefill latency)
  - tokens_per_sec  : generation throughput
  - peak_vram_mb    : peak GPU memory during inference

Uses the same prompt, output format, and results schema as benchmark.py (GGUF)
and benchmark_int4.py (bitsandbytes) so all three can be compared directly.

Usage:
  python scripts/benchmark_awq.py                        # benchmark all AWQ models
  python scripts/benchmark_awq.py --runs 3               # 3 runs per model (averaged)
  python scripts/benchmark_awq.py --model Llama          # filter by name substring
  python scripts/benchmark_awq.py --prompt "Your prompt" # custom prompt

Results saved to:
  results/benchmark_awq_<timestamp>.csv
  results/benchmark_awq_<timestamp>.json
"""

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
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

# ── AWQ models to benchmark ──────────────────────────────────────────────────
AWQ_MODELS = [
    {
        "name": "Llama-3.2-3B-Instruct-AWQ-INT4",
        "local_dir": "Llama-3.2-3B-Instruct-AWQ",  # subfolder under models/
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


# ── Single inference run ──────────────────────────────────────────────────────

def run_one(model, tokenizer, prompt, n_tokens):
    """Run one inference pass. Returns (ttft_ms, tokens_per_sec, peak_vram_mb)."""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()

    poller = VramPoller()
    poller.start()

    # ── Time to first token ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
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

    new_tokens = out.shape[1] - input_len
    tokens_per_sec = new_tokens / gen_secs if gen_secs > 0 else 0.0

    return ttft_ms, tokens_per_sec, poller.peak_mb


def benchmark_model(model_cfg, prompt, n_tokens, runs):
    """Load AWQ model and run N benchmark passes."""
    model_dir = MODELS_DIR / model_cfg["local_dir"]
    if not model_dir.exists():
        console.print(f"  [red]Model directory not found: {model_dir}[/red]")
        return None

    name = model_cfg["name"]

    console.print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    console.print(f"  Loading AWQ model...")
    load_start = time.perf_counter()
    model = AutoAWQForCausalLM.from_quantized(
        str(model_dir),
        fuse_layers=True,       # fuse attention layers for faster inference
        trust_remote_code=False,
        safetensors=True,
    )
    torch.cuda.synchronize()
    load_time_ms = (time.perf_counter() - load_start) * 1000
    console.print(f"  Loaded in {load_time_ms/1000:.1f}s")

    model.eval()

    ttfts, tps_list, vrams = [], [], []

    for i in range(runs):
        console.print(f"    run {i+1}/{runs} ...", end=" ")
        sys.stdout.flush()

        ttft_ms, tps, peak_vram = run_one(model, tokenizer, prompt, n_tokens)

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
        "backend": "awq",
        "status": "ok",
        "note": "INT4 AWQ, fuse_layers=True",
    }


# ── Results output ────────────────────────────────────────────────────────────

def save_results(results, timestamp):
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path  = RESULTS_DIR / f"benchmark_awq_{timestamp}.csv"
    json_path = RESULTS_DIR / f"benchmark_awq_{timestamp}.json"

    keys = ["model", "backend", "load_time_ms", "ttft_ms", "tokens_per_sec", "peak_vram_mb", "runs", "status", "note"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return csv_path, json_path


def print_table(results):
    table = Table(title="AWQ Benchmark Results", show_lines=True)
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
    parser = argparse.ArgumentParser(description="Benchmark AWQ quantized LLMs")
    parser.add_argument("--runs",   type=int, default=DEFAULT_RUNS,     help="Runs per model (averaged)")
    parser.add_argument("--tokens", type=int, default=DEFAULT_N_TOKENS, help="Tokens to generate")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,             help="Prompt to use")
    parser.add_argument("--model",  default=None,                       help="Filter models by name substring")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        console.print("[red]CUDA not available — AWQ benchmarking requires a GPU.[/red]")
        sys.exit(1)

    models = AWQ_MODELS
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]
        if not models:
            console.print(f"[red]No models matched filter: {args.model}[/red]")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"\n[bold]LLM Quantization Benchmark — AWQ[/bold]")
    console.print(f"Backend  : AutoAWQ INT4")
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
