"""
LLM Quality Evaluation — ROUGE-L + LLM-as-judge (Stage 9)

Runs each model through 15 fixed tasks (QA, code, summarization),
measures output quality using:

  1. ROUGE-L (always runs — fully offline)
     Compares model output to reference answer word-by-word.
     Score 0.0–1.0 (higher = closer to reference).

  2. LLM-as-judge via Groq (runs only if GROQ_API_KEY is set in .env)
     Sends output to llama-3.3-70b on Groq, which scores it 1-10 for:
       - Coherence   : is the response well-structured and readable?
       - Accuracy    : is the answer factually correct?
       - Relevance   : does it actually answer the question?

Results saved to:
  results/eval_quality_<timestamp>.csv
  results/eval_quality_<timestamp>.json

Usage:
  python scripts/eval_quality.py                     # all models
  python scripts/eval_quality.py --model SmolLM2     # filter by name
  python scripts/eval_quality.py --task qa           # one task type only
  python scripts/eval_quality.py --no-judge          # skip Groq even if key exists
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
import time
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from rouge_score import rouge_scorer
from rich.console import Console
from rich.table import Table

load_dotenv()

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LLAMA_CLI  = ROOT / "llama.cpp" / "Build" / "bin" / "llama-cli"

console = Console()

# ── Task definitions ──────────────────────────────────────────────────────────

QA_TASKS = [
    {
        "id": "qa_01",
        "prompt": "What is the capital of Australia and what is its population?",
        "reference": "The capital of Australia is Canberra. Its population is approximately 460,000 people.",
    },
    {
        "id": "qa_02",
        "prompt": "Explain the difference between RAM and ROM in simple terms.",
        "reference": "RAM is temporary memory used while a computer runs programs. ROM is permanent memory that stores firmware and cannot be easily changed.",
    },
    {
        "id": "qa_03",
        "prompt": "What is gradient descent in machine learning?",
        "reference": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that minimizes a loss function by following the negative gradient.",
    },
    {
        "id": "qa_04",
        "prompt": "What are the main differences between supervised and unsupervised learning?",
        "reference": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Unsupervised learning finds patterns in unlabeled data without predefined output labels.",
    },
    {
        "id": "qa_05",
        "prompt": "What is quantization in the context of large language models?",
        "reference": "Quantization reduces the precision of model weights from float32 or float16 to lower bit formats like INT8 or INT4, reducing memory footprint and increasing inference speed with minimal quality loss.",
    },
]

CODE_TASKS = [
    {
        "id": "code_01",
        "prompt": "Write a Python function that checks if a string is a palindrome.",
        "reference": "def is_palindrome(s: str) -> bool:\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
    },
    {
        "id": "code_02",
        "prompt": "Write a Python function to flatten a nested list.",
        "reference": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    },
    {
        "id": "code_03",
        "prompt": "Write a SQL query to find the top 3 highest-paid employees from an employees table with columns id, name, salary.",
        "reference": "SELECT id, name, salary FROM employees ORDER BY salary DESC LIMIT 3;",
    },
    {
        "id": "code_04",
        "prompt": "Write a Python decorator that measures and prints the execution time of a function.",
        "reference": "import time\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} took {time.time()-start:.4f}s')\n        return result\n    return wrapper",
    },
    {
        "id": "code_05",
        "prompt": "Write a Python class for a simple stack with push, pop, and peek methods.",
        "reference": "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()\n    def peek(self):\n        return self._items[-1]",
    },
]

SUMMARIZATION_TASKS = [
    {
        "id": "sum_01",
        "prompt": "Summarize the following in 2 sentences: Large language models (LLMs) are neural networks trained on vast amounts of text data. They learn statistical patterns in language to generate coherent text, answer questions, write code, and perform reasoning tasks. Models like GPT-4, Claude, and Llama have billions of parameters and require significant compute to train but can run inference on consumer hardware when quantized.",
        "reference": "Large language models are neural networks trained on massive text datasets to perform tasks like text generation, question answering, and reasoning. Modern LLMs have billions of parameters but can be made accessible through quantization techniques.",
    },
    {
        "id": "sum_02",
        "prompt": "Summarize in 2 sentences: Quantization is a model compression technique that reduces the numerical precision of neural network weights. By converting 32-bit or 16-bit floating point weights to 8-bit or 4-bit integers, models consume significantly less memory. This enables deployment on hardware with limited VRAM while maintaining most of the original model quality.",
        "reference": "Quantization compresses neural network weights to lower bit precision, significantly reducing memory usage. It enables deployment on limited hardware while preserving most of the original model quality.",
    },
    {
        "id": "sum_03",
        "prompt": "Summarize in 2 sentences: Australia's technology sector has seen rapid growth in artificial intelligence adoption across finance, healthcare, and government. Major banks and insurance companies are deploying on-premise LLMs to comply with data sovereignty laws that prevent sending sensitive customer data to overseas cloud providers.",
        "reference": "Australia's technology sector is rapidly adopting AI across key industries including finance and healthcare. Data sovereignty requirements are driving demand for on-premise LLM deployments.",
    },
    {
        "id": "sum_04",
        "prompt": "Summarize in 2 sentences: The GGUF format, used by llama.cpp, supports efficient CPU and GPU inference through mixed-precision quantization. It offers quantization levels from Q2 to Q8, where higher numbers preserve more quality at the cost of larger file size and memory use.",
        "reference": "GGUF is a quantization format used by llama.cpp for efficient CPU and GPU inference. It offers multiple precision levels (Q2-Q8) that trade between memory efficiency and output quality.",
    },
    {
        "id": "sum_05",
        "prompt": "Summarize in 2 sentences: Time to first token (TTFT) measures the latency between sending a prompt and receiving the first token of a response. It is the most important latency metric for interactive applications because users perceive it as response time, even if total generation takes longer.",
        "reference": "Time to first token measures the delay before a model begins responding. It is the critical latency metric for interactive applications as it directly determines perceived responsiveness.",
    },
]

ALL_TASKS = {
    "qa":            QA_TASKS,
    "code":          CODE_TASKS,
    "summarization": SUMMARIZATION_TASKS,
}

# ── Models to evaluate ───────────────────────────────────────────────────────

EVAL_MODELS = [
    {"name": "SmolLM2-135M-Q4_K_M", "backend": "gguf", "file": "SmolLM2-135M-Instruct-Q4_K_M.gguf"},
    {"name": "SmolLM2-135M-Q8_0",   "backend": "gguf", "file": "SmolLM2-135M-Instruct-Q8_0.gguf"},
    {"name": "Llama-3B-Q3_K_L",     "backend": "gguf", "file": "Llama-3.2-3B-Instruct-Q3_K_L.gguf"},
    {"name": "Llama-3B-Q4_K_M",     "backend": "gguf", "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
    {"name": "Llama-3B-Q5_K_M",     "backend": "gguf", "file": "Llama-3.2-3B-Instruct-Q5_K_M.gguf"},
    {"name": "Llama-3B-Q8_0",       "backend": "gguf", "file": "Llama-3.2-3B-Instruct-Q8_0.gguf"},
    {"name": "Mistral-7B-Q3_K_M",   "backend": "gguf", "file": "Mistral-7B-Instruct-v0.3-Q3_K_M.gguf"},
    {"name": "Llama-3B-INT4-NF4",   "backend": "int4", "dir": "Llama-3.2-3B-Instruct-HF"},
    {"name": "Llama-3B-AWQ-INT4",   "backend": "awq",  "dir": "Llama-3.2-3B-Instruct-AWQ"},
]


# ── llama-cli inference ───────────────────────────────────────────────────────

def run_inference(model_path: Path, prompt: str, n_tokens: int = 256) -> str:
    """Run llama-cli and return the generated text."""
    cmd = [
        str(LLAMA_CLI), "-m", str(model_path),
        "-ngl", "99", "-p", prompt, "-n", str(n_tokens),
        "-no-cnv", "--log-disable",
    ]

    master_fd, slave_fd = pty.openpty()

    def _child():
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        preexec_fn=_child, pass_fds=(slave_fd,),
    )
    os.close(slave_fd)

    _ANSI   = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    _JUNK   = re.compile(r'^(llama_|ggml_|load_tensors|.cache|build:|warning:|<\|)', re.IGNORECASE)
    _TIMING = re.compile(r'\[\s*Prompt:.*?Generation:.*?t/s\s*\]')

    chunks  = []
    output_lines = []
    prompt_done  = False
    deadline = time.time() + 120

    while time.time() < deadline:
        try:
            ready, _, _ = select.select([master_fd], [], [], 0.05)
        except (ValueError, OSError):
            break
        if ready:
            try:
                data = os.read(master_fd, 512)
            except OSError:
                break
            if not data:
                break
            chunk = _ANSI.sub("", data.decode("utf-8", errors="replace"))
            chunks.append(chunk)
            full = "".join(chunks)

            if _TIMING.search(full):
                break

    proc.kill()
    proc.wait()
    try:
        os.close(master_fd)
    except OSError:
        pass

    full = _ANSI.sub("", "".join(chunks))

    # Extract only generated text — find prompt echo and take everything after it
    # llama-cli echoes the prompt as "> {prompt}" before generating
    prompt_marker = f"> {prompt}"
    idx = full.find(prompt_marker)
    if idx != -1:
        after_prompt = full[idx + len(prompt_marker):]
    else:
        # Fallback: take everything after the last ">" if prompt echo not found
        idx = full.rfind(">")
        after_prompt = full[idx + 1:] if idx != -1 else full

    # Strip backspace chars, leading spinner characters and whitespace
    after_prompt = after_prompt.replace('\x08', '')
    after_prompt = re.sub(r'^[\s|\-\\/]+', '', after_prompt).strip()

    # Remove timing line if present
    after_prompt = _TIMING.sub("", after_prompt).strip()

    return after_prompt


# ── INT4 NF4 inference (bitsandbytes) ────────────────────────────────────────

# Cache to avoid reloading the same HF model between tasks
_hf_cache = {"id": None, "model": None, "tokenizer": None}
_hf_lock  = threading.Lock()

def run_inference_int4(model_dir: Path, prompt: str, n_tokens: int = 256) -> str:
    """Run bitsandbytes NF4 inference and return generated text."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    cache_key = f"int4:{model_dir}"
    with _hf_lock:
        if _hf_cache["id"] != cache_key:
            # Unload previous model
            if _hf_cache["model"] is not None:
                del _hf_cache["model"]
                del _hf_cache["tokenizer"]
                _hf_cache["model"]     = None
                _hf_cache["tokenizer"] = None
                _hf_cache["id"]        = None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                quantization_config=bnb_config,
                device_map="cuda",
            )
            _hf_cache["id"]        = cache_key
            _hf_cache["model"]     = model
            _hf_cache["tokenizer"] = tokenizer

        model     = _hf_cache["model"]
        tokenizer = _hf_cache["tokenizer"]

    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── AWQ inference ─────────────────────────────────────────────────────────────

def run_inference_awq(model_dir: Path, prompt: str, n_tokens: int = 256) -> str:
    """Run AutoAWQ inference and return generated text."""
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    cache_key = f"awq:{model_dir}"
    with _hf_lock:
        if _hf_cache["id"] != cache_key:
            if _hf_cache["model"] is not None:
                del _hf_cache["model"]
                del _hf_cache["tokenizer"]
                _hf_cache["model"]     = None
                _hf_cache["tokenizer"] = None
                _hf_cache["id"]        = None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoAWQForCausalLM.from_quantized(
                str(model_dir), fuse_layers=False, device_map="auto"
            )
            _hf_cache["id"]        = cache_key
            _hf_cache["model"]     = model
            _hf_cache["tokenizer"] = tokenizer

        model     = _hf_cache["model"]
        tokenizer = _hf_cache["tokenizer"]

    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── ROUGE-L scoring ───────────────────────────────────────────────────────────

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_rouge_l(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    scores = _scorer.score(reference, prediction)
    return round(scores["rougeL"].fmeasure, 4)


# ── LLM-as-judge via Groq ─────────────────────────────────────────────────────

_groq_client = None

def _get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
    return _groq_client

JUDGE_PROMPT = """You are an impartial evaluator. Score the following model response on three dimensions.
Return ONLY a JSON object with three integer scores (1-10), nothing else.

Question/Task: {prompt}

Model Response: {response}

Score each dimension 1-10:
- coherence: Is the response well-structured, readable, and logically consistent?
- accuracy: Is the information factually correct and precise?
- relevance: Does it directly and completely answer the question or task?

Return exactly: {{"coherence": X, "accuracy": X, "relevance": X}}"""


def llm_judge(prompt: str, response: str) -> dict | None:
    """Call Groq LLM-as-judge. Returns scores dict or None if unavailable."""
    client = _get_groq()
    if client is None:
        return None

    try:
        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(prompt=prompt, response=response[:800])
            }],
            max_tokens=60,
            temperature=0,
        )
        text = result.choices[0].message.content.strip()
        # Extract JSON from response
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return {
                "judge_coherence":  int(scores.get("coherence", 0)),
                "judge_accuracy":   int(scores.get("accuracy", 0)),
                "judge_relevance":  int(scores.get("relevance", 0)),
                "judge_avg":        round((scores.get("coherence", 0) + scores.get("accuracy", 0) + scores.get("relevance", 0)) / 3, 2),
            }
    except Exception as e:
        console.print(f"    [yellow]Judge error: {e}[/yellow]")
    return None


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_model(model_cfg: dict, task_types: list, use_judge: bool) -> list:
    backend = model_cfg.get("backend", "gguf")

    if backend == "gguf":
        model_path = MODELS_DIR / model_cfg["file"]
        if not model_path.exists():
            console.print(f"  [red]File not found: {model_cfg['file']}[/red]")
            return []
    else:
        model_path = MODELS_DIR / model_cfg["dir"]
        if not model_path.exists():
            console.print(f"  [red]Directory not found: {model_cfg['dir']}[/red]")
            return []

    results = []

    for task_type in task_types:
        tasks = ALL_TASKS[task_type]
        rouge_scores = []
        judge_scores = []

        for task in tasks:
            console.print(f"    [{task_type}] {task['id']} ...", end=" ")
            sys.stdout.flush()

            if backend == "gguf":
                output = run_inference(model_path, task["prompt"])
            elif backend == "int4":
                output = run_inference_int4(model_path, task["prompt"])
            elif backend == "awq":
                output = run_inference_awq(model_path, task["prompt"])
            else:
                output = ""
            rouge  = compute_rouge_l(output, task["reference"])
            rouge_scores.append(rouge)

            judge = None
            if use_judge:
                judge = llm_judge(task["prompt"], output)
                if judge:
                    judge_scores.append(judge["judge_avg"])

            console.print(f"ROUGE-L: [green]{rouge:.3f}[/green]" +
                         (f"  Judge: [cyan]{judge['judge_avg']:.1f}/10[/cyan]" if judge else ""))

            results.append({
                "model":           model_cfg["name"],
                "task_type":       task_type,
                "task_id":         task["id"],
                "rouge_l":         rouge,
                "judge_coherence": judge["judge_coherence"] if judge else None,
                "judge_accuracy":  judge["judge_accuracy"]  if judge else None,
                "judge_relevance": judge["judge_relevance"] if judge else None,
                "judge_avg":       judge["judge_avg"]        if judge else None,
                "output_preview":  output[:200],
            })

        avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0
        avg_judge = round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None

        console.print(f"  [{task_type}] avg ROUGE-L: [bold green]{avg_rouge:.3f}[/bold green]" +
                     (f"  avg judge: [bold cyan]{avg_judge:.1f}/10[/bold cyan]" if avg_judge else ""))

    return results


# ── Results output ────────────────────────────────────────────────────────────

def save_results(results: list, timestamp: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path  = RESULTS_DIR / f"eval_quality_{timestamp}.csv"
    json_path = RESULTS_DIR / f"eval_quality_{timestamp}.json"

    keys = ["model", "task_type", "task_id", "rouge_l",
            "judge_coherence", "judge_accuracy", "judge_relevance", "judge_avg",
            "output_preview"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return csv_path, json_path


def print_summary_table(results: list):
    # Aggregate per model
    from collections import defaultdict
    model_scores = defaultdict(lambda: {"rouge": [], "judge": []})
    for r in results:
        model_scores[r["model"]]["rouge"].append(r["rouge_l"])
        if r["judge_avg"] is not None:
            model_scores[r["model"]]["judge"].append(r["judge_avg"])

    table = Table(title="Quality Evaluation Summary", show_lines=True)
    table.add_column("Model",          style="cyan", no_wrap=True)
    table.add_column("Avg ROUGE-L",    justify="right", style="green")
    table.add_column("Avg Judge /10",  justify="right", style="yellow")
    table.add_column("Tasks evaluated", justify="right")

    for model, scores in sorted(model_scores.items()):
        avg_rouge = round(sum(scores["rouge"]) / len(scores["rouge"]), 3) if scores["rouge"] else 0
        avg_judge = round(sum(scores["judge"]) / len(scores["judge"]), 1) if scores["judge"] else None
        table.add_row(
            model,
            f"{avg_rouge:.3f}",
            f"{avg_judge:.1f}" if avg_judge else "—",
            str(len(scores["rouge"])),
        )

    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM output quality")
    parser.add_argument("--model",     default=None, help="Filter models by name substring")
    parser.add_argument("--task",      default=None, choices=["qa", "code", "summarization"], help="Run one task type only")
    parser.add_argument("--no-judge",  action="store_true", help="Skip Groq LLM-as-judge even if API key exists")
    args = parser.parse_args()

    groq_key   = os.getenv("GROQ_API_KEY")
    use_judge  = bool(groq_key) and not args.no_judge
    task_types = [args.task] if args.task else ["qa", "code", "summarization"]

    models = EVAL_MODELS
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]
        if not models:
            console.print(f"[red]No models matched: {args.model}[/red]")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"\n[bold]LLM Quality Evaluation[/bold]")
    console.print(f"Models     : {len(models)}")
    console.print(f"Task types : {', '.join(task_types)}")
    console.print(f"ROUGE-L    : always on (offline)")
    console.print(f"LLM judge  : {'[green]ON[/green] — Groq llama-3.3-70b' if use_judge else '[yellow]OFF[/yellow] — set GROQ_API_KEY in .env to enable'}")
    console.print()

    if not use_judge and not groq_key:
        console.print("[yellow]Tip: create a .env file with GROQ_API_KEY=your_key to enable LLM-as-judge scoring[/yellow]\n")

    all_results = []
    for model_cfg in models:
        console.print(f"[bold cyan]{model_cfg['name']}[/bold cyan]")
        results = evaluate_model(model_cfg, task_types, use_judge)
        all_results.extend(results)
        console.print()

    if not all_results:
        console.print("[red]No results.[/red]")
        sys.exit(1)

    print_summary_table(all_results)

    csv_path, json_path = save_results(all_results, timestamp)
    console.print(f"\nSaved: [green]{csv_path}[/green]")
    console.print(f"Saved: [green]{json_path}[/green]")


if __name__ == "__main__":
    main()
