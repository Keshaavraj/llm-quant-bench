"""
LLM Quantization Benchmark — GGUF backend (Phase 1)

Runs each model with a fixed prompt via llama-cli and measures:
  - load_time_ms    : time to load model weights into VRAM
  - ttft_ms         : time to first token (load + prompt eval)
  - tokens_per_sec  : generation throughput
  - peak_vram_mb    : peak GPU memory during inference

Usage:
  python scripts/benchmark.py                        # benchmark all models
  python scripts/benchmark.py --runs 3               # 3 runs per model (averaged)
  python scripts/benchmark.py --model Q4_K_M         # filter by name substring
  python scripts/benchmark.py --backend gguf         # explicit backend (default)
  python scripts/benchmark.py --prompt "Your prompt" # custom prompt

Results saved to:
  results/benchmark_<timestamp>.csv
  results/benchmark_<timestamp>.json
"""

import argparse
import csv
import fcntl
import json
import os
import pty
import re
import select
import subprocess
import sys
import termios
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LLAMA_CLI = ROOT / "llama.cpp" / "Build" / "bin" / "llama-cli"

# ── Benchmark config ─────────────────────────────────────────────────────────
DEFAULT_PROMPT = (
    "Explain the difference between quantization and pruning in large language models. "
    "What are the trade-offs of each approach?"
)
DEFAULT_N_TOKENS = 200   # tokens to generate per run
DEFAULT_N_GPU_LAYERS = 99  # offload all layers to GPU
DEFAULT_RUNS = 3          # runs per model — results are averaged

# ── GGUF models — ordered from most to least compressed ──────────────────────
GGUF_MODELS = [
    "SmolLM2-135M-Instruct-Q4_K_M.gguf",
    "SmolLM2-135M-Instruct-Q8_0.gguf",
    "Llama-3.2-3B-Instruct-Q3_K_L.gguf",
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
    "Llama-3.2-3B-Instruct-Q8_0.gguf",
    "Mistral-7B-Instruct-v0.3-Q3_K_M.gguf",
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
]

# Models known to exceed GPU VRAM — skipped automatically with OOM label.
# Mistral-7B-Q4_K_M is 4.1 GB; RTX 500 Ada has 4.0 GB VRAM.
# KV cache overhead during generation pushes it over the limit.
OOM_MODELS = {
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": "model is ~4.1 GB, exceeds 4 GB VRAM (KV cache adds overhead)",
}

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


# ── llama-cli timing parser ───────────────────────────────────────────────────

def parse_timings(text):
    """
    Parse timing info from llama-cli output. Supports two formats:

    Old format (stderr, llama.cpp < b8000):
      llama_print_timings:        load time =    1234.56 ms
      llama_print_timings: prompt eval time =     456.78 ms /  10 tokens
      llama_print_timings:        eval time =    5678.90 ms / 199 runs   (... 35.04 tokens per second)

    New format (stdout, llama.cpp >= b8000):
      [ Prompt: 373.2 t/s | Generation: 10.9 t/s ]
    """
    load_ms = prompt_ms = eval_tps = None

    # ── New format ────────────────────────────────────────────────────────────
    new_match = re.search(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
        text,
    )
    if new_match:
        prompt_tps = float(new_match.group(1))
        eval_tps   = float(new_match.group(2))
        # load_time not available in new format; approximate TTFT from prompt speed
        # prompt_ms left as None — TTFT column will show — in results table
        return load_ms, prompt_ms, eval_tps

    # ── Old format ────────────────────────────────────────────────────────────
    load_match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", text)
    if load_match:
        load_ms = float(load_match.group(1))

    prompt_match = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", text)
    if prompt_match:
        prompt_ms = float(prompt_match.group(1))

    eval_match = re.search(
        r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs.*?([\d.]+)\s*tokens per second",
        text,
    )
    if eval_match:
        eval_tps = float(eval_match.group(1))

    return load_ms, prompt_ms, eval_tps


# ── Single model benchmark ────────────────────────────────────────────────────

def run_one(model_path, prompt, n_tokens, n_gpu_layers):
    """Run llama-cli once, return (load_ms, ttft_ms, tokens_per_sec, peak_vram_mb).

    Uses a pseudo-terminal (pty) so llama-cli thinks it has a real terminal and
    writes all output (including timing stats) to the pty instead of /dev/tty.
    We kill the process as soon as the timing line appears.
    """
    poller = VramPoller()
    poller.start()

    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-ngl", str(n_gpu_layers),
        "-p", prompt,
        "-n", str(n_tokens),
        "-no-cnv",
    ]

    master_fd, slave_fd = pty.openpty()

    def _child_setup():
        # Create new session so the child has no controlling terminal,
        # then assign our pty slave as the controlling terminal.
        # This means llama-cli's /dev/tty writes go to our pty, not the real terminal.
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        preexec_fn=_child_setup,
        pass_fds=(slave_fd,),
    )
    os.close(slave_fd)

    chunks = []
    deadline = time.time() + 120  # 2-minute hard cutoff
    _ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    while time.time() < deadline:
        try:
            ready, _, _ = select.select([master_fd], [], [], 1.0)
        except (ValueError, OSError):
            break
        if ready:
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
            clean = _ANSI.sub("", "".join(chunks))
            # New format: [ Prompt: X t/s | Generation: Y t/s ]
            # Old format: llama_print_timings … eval time … tokens per second
            if ("Generation:" in clean and "t/s" in clean) or \
               ("llama_print_timings" in clean and "eval time" in clean):
                break

    proc.kill()
    proc.wait()
    try:
        os.close(master_fd)
    except OSError:
        pass

    poller.stop()
    peak_vram = poller.peak_mb

    combined = _ANSI.sub("", "".join(chunks))

    if os.environ.get("BENCH_DEBUG"):
        print(f"\n=== CAPTURED (first 1000 chars) ===\n{combined[:1000]}\n===\n")

    load_ms, prompt_ms, eval_tps = parse_timings(combined)

    ttft_ms = None
    if load_ms is not None and prompt_ms is not None:
        ttft_ms = load_ms + prompt_ms

    return load_ms, ttft_ms, eval_tps, peak_vram


def benchmark_model(model_file, prompt, n_tokens, n_gpu_layers, runs):
    """Run a model N times and return averaged metrics, or an OOM/error entry."""
    # Pre-skip models known to exceed VRAM before wasting time loading them.
    if model_file in OOM_MODELS:
        return {
            "model": model_file,
            "load_time_ms": None,
            "ttft_ms": None,
            "tokens_per_sec": None,
            "peak_vram_mb": None,
            "runs": 0,
            "backend": "gguf",
            "status": "OOM",
            "note": OOM_MODELS[model_file],
        }

    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        return None

    load_times, ttfts, tps_list, vrams = [], [], [], []

    for i in range(runs):
        console.print(f"    run {i+1}/{runs} ...", end=" ")
        sys.stdout.flush()

        load_ms, ttft_ms, eval_tps, peak_vram = run_one(model_path, prompt, n_tokens, n_gpu_layers)

        if eval_tps is None:
            console.print("[red]FAILED — could not parse timings[/red]")
            return None

        load_times.append(load_ms)
        ttfts.append(ttft_ms)
        tps_list.append(eval_tps)
        vrams.append(peak_vram)

        console.print(f"[green]{eval_tps:.1f} tok/s[/green]")

    def avg(lst):
        vals = [v for v in lst if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        "model": model_file,
        "load_time_ms": avg(load_times),
        "ttft_ms": avg(ttfts),
        "tokens_per_sec": avg(tps_list),
        "peak_vram_mb": avg(vrams),
        "runs": runs,
        "backend": "gguf",
        "status": "ok",
        "note": "",
    }


# ── Results output ────────────────────────────────────────────────────────────

def save_results(results, timestamp):
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path = RESULTS_DIR / f"benchmark_{timestamp}.csv"
    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    keys = ["model", "backend", "load_time_ms", "ttft_ms", "tokens_per_sec", "peak_vram_mb", "runs", "status", "note"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return csv_path, json_path


def print_table(results):
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Load (ms)", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tok/s", justify="right", style="green")
    table.add_column("VRAM (MB)", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    for r in results:
        status = r.get("status", "ok")
        if status == "OOM":
            table.add_row(
                r["model"].replace(".gguf", ""),
                "—", "—", "—", "—",
                "[bold red]OOM[/bold red]",
            )
        else:
            table.add_row(
                r["model"].replace(".gguf", ""),
                f"{r['load_time_ms']:.0f}" if r["load_time_ms"] else "—",
                f"{r['ttft_ms']:.0f}" if r["ttft_ms"] else "—",
                f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] else "—",
                f"{r['peak_vram_mb']:.0f}" if r["peak_vram_mb"] else "—",
                "[green]ok[/green]",
            )

    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized LLMs")
    parser.add_argument("--backend", default="gguf", choices=["gguf"], help="Inference backend")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Runs per model (averaged)")
    parser.add_argument("--tokens", type=int, default=DEFAULT_N_TOKENS, help="Tokens to generate")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to use")
    parser.add_argument("--model", default=None, help="Filter models by name substring")
    args = parser.parse_args()

    if not LLAMA_CLI.exists():
        console.print(f"[red]llama-cli not found at {LLAMA_CLI}[/red]")
        sys.exit(1)

    models = GGUF_MODELS
    if args.model:
        models = [m for m in models if args.model.lower() in m.lower()]
        if not models:
            console.print(f"[red]No models matched filter: {args.model}[/red]")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"\n[bold]LLM Quantization Benchmark[/bold]")
    console.print(f"Backend  : {args.backend}")
    console.print(f"Models   : {len(models)}")
    console.print(f"Runs     : {args.runs} per model")
    console.print(f"Tokens   : {args.tokens}")
    console.print(f"Prompt   : {args.prompt[:80]}...")
    console.print()

    results = []
    for model_file in models:
        console.print(f"[bold cyan]{model_file}[/bold cyan]")
        r = benchmark_model(model_file, args.prompt, args.tokens, DEFAULT_N_GPU_LAYERS, args.runs)
        if r is None:
            console.print(f"  [red]SKIPPED (model file not found)[/red]\n")
        elif r.get("status") == "OOM":
            results.append(r)
            console.print(f"  [bold red]OOM[/bold red] — {r['note']}\n")
        else:
            results.append(r)
            ttft_str = f"{r['ttft_ms']:.0f}ms" if r['ttft_ms'] else "—"
            console.print(
                f"  → TTFT {ttft_str}  |  {r['tokens_per_sec']:.1f} tok/s  |  {r['peak_vram_mb']:.0f} MB VRAM\n"
            )

    ok_results = [r for r in results if r.get("status") == "ok"]
    if not results:
        console.print("[red]No results collected.[/red]")
        sys.exit(1)
    if not ok_results:
        console.print("[yellow]Warning: all models were skipped or OOM — no benchmark data.[/yellow]")

    print_table(results)

    csv_path, json_path = save_results(results, timestamp)
    console.print(f"\nSaved: [green]{csv_path}[/green]")
    console.print(f"Saved: [green]{json_path}[/green]")


if __name__ == "__main__":
    main()
