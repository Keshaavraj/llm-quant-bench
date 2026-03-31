"""
LLM Quant Bench — Web UI Backend

FastAPI server that:
- Serves the single-page frontend from static/index.html
- Exposes GET /api/models to list available models
- Exposes POST /api/run to run inference with SSE streaming

SSE event format:
  {"type": "token",   "text": "..."}        ← one per generated token
  {"type": "metrics", "tps": 12.3, ...}     ← sent after generation completes
  {"type": "error",   "message": "..."}     ← on failure
  {"type": "done"}                           ← stream end marker
"""

import asyncio
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
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
LLAMA_CLI  = ROOT / "llama.cpp" / "Build" / "bin" / "llama-cli"

app = FastAPI(title="LLM Quant Bench")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = [
    {"id": "smol-q4",  "name": "SmolLM2-135M Q4_K_M", "backend": "gguf",         "file": "SmolLM2-135M-Instruct-Q4_K_M.gguf", "bench_tps": 347.4, "bench_vram": 473,  "rouge_l": 0.204, "judge_avg": 5.3},
    {"id": "smol-q8",  "name": "SmolLM2-135M Q8_0",   "backend": "gguf",         "file": "SmolLM2-135M-Instruct-Q8_0.gguf",   "bench_tps": 324.7, "bench_vram": 511,  "rouge_l": 0.208, "judge_avg": 5.5},
    {"id": "gguf-q3",  "name": "Llama-3B Q3_K_L",     "backend": "gguf",         "file": "Llama-3.2-3B-Instruct-Q3_K_L.gguf", "bench_tps": 33.2,  "bench_vram": 2515, "rouge_l": 0.262, "judge_avg": 9.2},
    {"id": "gguf-q4",  "name": "Llama-3B Q4_K_M",     "backend": "gguf",         "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "bench_tps": 35.7,  "bench_vram": 2711, "rouge_l": 0.273, "judge_avg": 9.3},
    {"id": "gguf-q5",  "name": "Llama-3B Q5_K_M",     "backend": "gguf",         "file": "Llama-3.2-3B-Instruct-Q5_K_M.gguf",        "bench_tps": 37.0,  "bench_vram": 2999, "rouge_l": 0.258, "judge_avg": 9.3},
    {"id": "gguf-q8",  "name": "Llama-3B Q8_0",       "backend": "gguf",         "file": "Llama-3.2-3B-Instruct-Q8_0.gguf",           "bench_tps": 25.6,  "bench_vram": 3828, "rouge_l": 0.259, "judge_avg": 9.2},
    {"id": "gguf-m7",  "name": "Mistral-7B Q3_K_M",   "backend": "gguf",         "file": "Mistral-7B-Instruct-v0.3-Q3_K_M.gguf",      "bench_tps": 7.9,   "bench_vram": 3845, "rouge_l": 0.280, "judge_avg": 9.3},
    {"id": "int4-nf4", "name": "Llama-3B INT4 NF4",   "backend": "bitsandbytes", "dir":  "Llama-3.2-3B-Instruct-HF",                  "bench_tps": 12.2,  "bench_vram": 2361, "rouge_l": 0.190, "judge_avg": 8.6},
    {"id": "awq-int4", "name": "Llama-3B AWQ INT4",   "backend": "awq",          "dir":  "Llama-3.2-3B-Instruct-AWQ",                 "bench_tps": 5.0,   "bench_vram": 3027, "rouge_l": 0.144, "judge_avg": 7.9},
]

# ── Cached HF model (avoid reloading same model) ──────────────────────────────

_cache = {"id": None, "model": None, "tokenizer": None}
_lock  = threading.Lock()


def _unload_cache():
    if _cache["model"] is not None:
        del _cache["model"]
        del _cache["tokenizer"]
        _cache["model"] = None
        _cache["tokenizer"] = None
        _cache["id"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── VRAM helper ───────────────────────────────────────────────────────────────

def _peak_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split("\n")[0].strip())
    except Exception:
        return 0


class VramPoller:
    def __init__(self):
        self.peak_mb = 0
        self._stop   = threading.Event()
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
            mb = _peak_vram()
            if mb > self.peak_mb:
                self.peak_mb = mb
            time.sleep(0.5)


# ── GGUF streaming via llama-cli ──────────────────────────────────────────────

def _stream_gguf(model_file: str, prompt: str, n_tokens: int):
    """Generator that yields (event_type, data) tuples for GGUF inference."""
    model_path = MODELS_DIR / model_file

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

    _ANSI     = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    _TIMING   = re.compile(r'\[\s*Prompt:\s*[\d.]+\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]')
    _JUNK     = re.compile(r'^(llama_|ggml_|load_tensors|.cache|build:|warning:|<\|.*?\|>)', re.IGNORECASE)

    poller    = VramPoller()
    load_start = time.perf_counter()
    poller.start()

    buffer       = ""
    raw_chunks   = []
    prompt_done  = False
    load_time_ms = None
    first_token_t = None
    token_count  = 0
    gen_start    = None
    deadline     = time.time() + 180

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
            raw_chunks.append(chunk)
            buffer += chunk

            # Detect when prompt processing is done — model starts generating
            if not prompt_done and ("\n" in buffer or len(buffer) > 50):
                lines = buffer.split("\n")
                for line in lines[:-1]:
                    clean = line.strip()
                    if clean and not _JUNK.match(clean) and len(clean) > 2:
                        # First real output line = model started generating
                        if load_time_ms is None:
                            load_time_ms = (time.perf_counter() - load_start) * 1000
                            gen_start    = time.perf_counter()
                        prompt_done = True
                        break
                buffer = lines[-1]

            # Stream tokens once prompt is done
            if prompt_done and "\n" in buffer:
                lines  = buffer.split("\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    clean = line.strip()
                    if not clean or _JUNK.match(clean):
                        continue
                    # Stop at timing line
                    if "Generation:" in clean and "t/s" in clean:
                        break
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    token_count += len(clean.split())
                    yield ("token", clean + "\n")

            # Check for timing line to finish
            full = "".join(raw_chunks)
            m    = _TIMING.search(full)
            if m:
                tps = float(m.group(1))
                break

    proc.kill()
    proc.wait()
    try:
        os.close(master_fd)
    except OSError:
        pass
    poller.stop()

    full = _ANSI.sub("", "".join(raw_chunks))
    m    = _TIMING.search(full)
    tps  = float(m.group(1)) if m else None

    ttft_ms  = ((first_token_t - load_start) * 1000) if first_token_t else None
    gen_secs = (time.perf_counter() - gen_start) if gen_start else None

    yield ("metrics", {
        "tps":          round(tps, 1)          if tps          else None,
        "vram_mb":      poller.peak_mb,
        "ttft_ms":      round(ttft_ms, 0)      if ttft_ms      else None,
        "load_time_ms": round(load_time_ms, 0) if load_time_ms else None,
        "backend":      "gguf (llama.cpp)",
    })


# ── bitsandbytes INT4 streaming ───────────────────────────────────────────────

def _stream_int4(model_dir: str, prompt: str, n_tokens: int, model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

    with _lock:
        if _cache["id"] != model_id:
            _unload_cache()
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            tok   = AutoTokenizer.from_pretrained(str(MODELS_DIR / model_dir))
            t0    = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                str(MODELS_DIR / model_dir),
                quantization_config=bnb,
                device_map="auto",
                dtype=torch.float16,
            )
            load_ms = (time.perf_counter() - t0) * 1000
            _cache.update({"id": model_id, "model": model, "tokenizer": tok, "load_ms": load_ms})
        else:
            load_ms = _cache.get("load_ms", 0)

    model     = _cache["model"]
    tokenizer = _cache["tokenizer"]
    model.eval()

    inputs    = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer  = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    poller    = VramPoller()
    poller.start()

    ttft_ref  = [None]
    gen_start = time.perf_counter()

    def _generate():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    token_count = 0
    for text in streamer:
        if ttft_ref[0] is None:
            ttft_ref[0] = time.perf_counter()
        token_count += 1
        yield ("token", text)

    thread.join()
    poller.stop()

    gen_secs = time.perf_counter() - gen_start
    tps      = token_count / gen_secs if gen_secs > 0 else None
    ttft_ms  = (ttft_ref[0] - gen_start) * 1000 if ttft_ref[0] else None

    yield ("metrics", {
        "tps":          round(tps, 1)      if tps      else None,
        "vram_mb":      poller.peak_mb,
        "ttft_ms":      round(ttft_ms, 0)  if ttft_ms  else None,
        "load_time_ms": round(load_ms, 0),
        "backend":      "bitsandbytes NF4",
    })


# ── AWQ streaming ─────────────────────────────────────────────────────────────

def _stream_awq(model_dir: str, prompt: str, n_tokens: int, model_id: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, TextIteratorStreamer

    with _lock:
        if _cache["id"] != model_id:
            _unload_cache()
            tok   = AutoTokenizer.from_pretrained(str(MODELS_DIR / model_dir))
            t0    = time.perf_counter()
            model = AutoAWQForCausalLM.from_quantized(
                str(MODELS_DIR / model_dir),
                fuse_layers=True,
                safetensors=True,
            )
            load_ms = (time.perf_counter() - t0) * 1000
            _cache.update({"id": model_id, "model": model, "tokenizer": tok, "load_ms": load_ms})
        else:
            load_ms = _cache.get("load_ms", 0)

    model     = _cache["model"]
    tokenizer = _cache["tokenizer"]
    model.eval()

    inputs   = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    poller   = VramPoller()
    poller.start()

    ttft_ref  = [None]
    gen_start = time.perf_counter()

    def _generate():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    token_count = 0
    for text in streamer:
        if ttft_ref[0] is None:
            ttft_ref[0] = time.perf_counter()
        token_count += 1
        yield ("token", text)

    thread.join()
    poller.stop()

    gen_secs = time.perf_counter() - gen_start
    tps      = token_count / gen_secs if gen_secs > 0 else None
    ttft_ms  = (ttft_ref[0] - gen_start) * 1000 if ttft_ref[0] else None

    yield ("metrics", {
        "tps":          round(tps, 1)      if tps      else None,
        "vram_mb":      poller.peak_mb,
        "ttft_ms":      round(ttft_ms, 0)  if ttft_ms  else None,
        "load_time_ms": round(load_ms, 0),
        "backend":      "AutoAWQ INT4",
    })


# ── SSE wrapper ───────────────────────────────────────────────────────────────

async def _sse(generator) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()

    def _run():
        return list(generator)

    events = await loop.run_in_executor(None, _run)
    for event_type, data in events:
        payload = json.dumps({"type": event_type, "data": data})
        yield f"data: {payload}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(html)


@app.get("/api/models")
def list_models():
    result = []
    for m in MODELS:
        available = (MODELS_DIR / m["file"]).exists() if "file" in m else (MODELS_DIR / m["dir"]).exists()
        result.append({**m, "available": available})
    return result


class RunRequest(BaseModel):
    model_id: str
    prompt:   str
    n_tokens: int = 200


@app.post("/api/run")
def run_inference(req: RunRequest):
    model = next((m for m in MODELS if m["id"] == req.model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")

    if model["backend"] == "gguf":
        gen = _stream_gguf(model["file"], req.prompt, req.n_tokens)
    elif model["backend"] == "bitsandbytes":
        gen = _stream_int4(model["dir"], req.prompt, req.n_tokens, req.model_id)
    elif model["backend"] == "awq":
        gen = _stream_awq(model["dir"], req.prompt, req.n_tokens, req.model_id)
    else:
        raise HTTPException(400, "Unknown backend")

    return StreamingResponse(_sse(gen), media_type="text/event-stream")
