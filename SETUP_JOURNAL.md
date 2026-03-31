# Setup Journal — LLM Quantization Benchmark

This file documents every command I ran, why I ran it, and what it did.
Written so I can explain every step to a recruiter.

---

## Stage 1: Repo Setup

### Create the project folder and initialize git
```bash
mkdir ~/llm-quant-bench && cd ~/llm-quant-bench && git init
```
**Why:** Creates a local git repo. Every benchmark, script, and result goes here.
This becomes the portfolio project I can show recruiters.

---

### Create folder structure
```bash
mkdir -p models results scripts logs
```
**Why:**
- `models/` — stores downloaded GGUF model files (not committed to git, too large)
- `scripts/` — benchmark scripts I write
- `results/` — output CSV/JSON from benchmark runs
- `logs/` — raw inference logs

**Note:** `-p` means create parent directories if they don't exist, and don't error if they already exist.

---

### Create .gitignore
```bash
# done by hand — see .gitignore file
```
**Why:** GGUF model files are 2–4GB each. You never commit binary files or large data to git.
Also ignoring llama.cpp source, Python cache files, and .env secrets.

**What's ignored:**
- `models/` — model weight files
- `llama.cpp/` — third party source code
- `logs/` — raw output files
- `__pycache__/`, `*.pyc` — Python bytecode
- `.env` — API keys and secrets

---

### First git commit
```bash
git add . && git commit -m "Initial repo structure"
```
**Why:** Every stage of work is committed. Shows learning progression in git history.

---

## Stage 2: Install llama.cpp

### What is llama.cpp?
llama.cpp is a C++ library that runs GGUF quantized models on CPU and GPU.
Written in C++ (not Python) because inference requires maximum hardware speed —
Python's interpreter overhead is too slow for billions of matrix multiplications per second.

### Clone llama.cpp
```bash
cd ~/llm-quant-bench
git clone https://github.com/ggerganov/llama.cpp.git
```
**Why:** We clone it inside our repo folder so everything stays in one place.
It's gitignored so we don't commit the source code.

---

### Install CMake (build system)
```bash
sudo apt install cmake -y
```
**Why:** llama.cpp is C++ source code — it needs to be compiled before you can run it.
CMake is the tool that prepares the compilation. It reads the project, detects your
CPU features and available libraries, then generates the actual build instructions.

Think of CMake as a foreman who reads blueprints and tells workers what to build.

---

### Install CUDA Toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

**Why:** CUDA is NVIDIA's platform that lets software use the GPU for general computing.
Two components exist — the driver (ships with the GPU, already present) and the toolkit
(compiler + libraries, must be installed separately).

- `cuda-keyring` — tells Ubuntu to trust NVIDIA's package server
- `apt update` — refreshes package list to include NVIDIA's repo
- `cuda-toolkit-12-6` — installs `nvcc` (CUDA compiler), `libcudart`, `libcublas`
  (GPU matrix math — the core of LLM inference)

Without this, cmake cannot compile GPU code even if the GPU is physically present.

---

### Add CUDA to PATH

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Why:**
- `PATH` — tells Linux where to find the `nvcc` command when you type it in the terminal
- `LD_LIBRARY_PATH` — tells Linux where to find CUDA `.so` runtime libraries when programs run
- `source ~/.bashrc` — applies changes immediately without reopening the terminal

**Verify toolkit is found:**
```bash
nvcc --version
# output: nvcc release 12.6 ...
```

---

### Configure the build with GPU support
```bash
cd ~/llm-quant-bench/llama.cpp
cmake -B Build -DGGML_CUDA=ON
```
**Why:** This is the "configure" step. CMake scans your system and prepares build files.
- `-DGGML_CUDA=ON` — the key flag. Tells cmake to compile GPU kernels using `nvcc`.
  Without this flag, llama.cpp runs on CPU only even if the GPU and CUDA toolkit are present.

Output shows:
- CUDA found → GPU kernels will be compiled
- `nvcc` detected at `/usr/local/cuda/bin/nvcc`
- RTX 500 Ada detected

---

### Compile llama.cpp
```bash
cmake --build Build -j$(nproc)
```
**Why:** Actually compiles all C++ source files into executables.
- `--build Build` — compile what CMake configured in the Build folder
- `-j$(nproc)` — use all CPU cores in parallel (`nproc` = 22 on this machine)
- Both GCC (CPU parts) and `nvcc` (GPU kernel parts) compile together

**Output:** Two executables created:
- `Build/bin/llama-cli` — command line tool to run models
- `Build/bin/llama-server` — HTTP server to serve models via API

### Verify it works
```bash
./Build/bin/llama-cli --version
# output: version: 8581 (abf9a6216), built with GNU 13.3.0 for Linux x86_64
```
**What the output means:**
- `8581` — build number (commits since project started)
- `abf9a6216` — exact git commit hash of the code compiled
- `GNU 13.3.0` — GCC compiler version used
- `Linux x86_64` — 64-bit x86 CPU architecture

---

### Run first inference on GPU

```bash
cd ~/llm-quant-bench
./llama.cpp/Build/bin/llama-cli \
  -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -p "What is quantization in LLMs?" \
  -n 200
```

**Why each flag:**
- `-m` — path to the model file to load
- `-ngl 99` — number of GPU layers to offload. Llama 3.2 3B has 28 layers total.
  Setting 99 means "move everything possible to VRAM" — all 28 layers go to GPU.
  Without this flag, the model stays on CPU even with a GPU-enabled binary.
- `-p` — input prompt sent to the model
- `-n 200` — generate up to 200 tokens of response (~150 words)

**Verify GPU is active (open a second terminal while inference runs):**
```bash
nvidia-smi
# Memory-Usage: ~2000MiB / 4094MiB  ← model loaded in VRAM
# GPU-Util:      80-100%             ← GPU actively computing
```

---

## Stage 3: Download Models (next steps)

### Plan
- Install huggingface_hub CLI to download models
- Download Llama 3.2 3B at different quant levels:
  - Q4_K_M (sweet spot — fits in 4GB VRAM, good quality)
  - Q8_0 (high quality, uses more VRAM)
  - Q3_K_M (aggressive compression, lower quality)
- Run inference on each
- Measure: TTFT, tokens/sec, VRAM usage

### What is GGUF?
GGUF = GPT-Generated Unified Format. A file format for storing quantized LLM weights.
Replaced the older GGML format in August 2023.

One GGUF file contains everything: weights + tokenizer + model config + metadata.

**Naming convention:**
```
llama-3.2-3B-Q4_K_M.gguf
              ↑   ↑ ↑
              │   │ └─ size variant (S=small, M=medium, L=large)
              │   └─── K = K-quant (smarter quantization)
              └──────── 4 bits per weight
```

### Quantization levels comparison (planned)
| Format  | Bits/weight | VRAM (3B model) | Quality |
|---------|-------------|-----------------|---------|
| Q3_K_M  | ~3.4 bits   | ~1.8GB          | ~85%    |
| Q4_K_M  | ~4.5 bits   | ~2.1GB          | ~91%    |
| Q5_K_M  | ~5.5 bits   | ~2.5GB          | ~95%    |
| Q8_0    | ~8 bits     | ~3.2GB          | ~99%    |
| FP16    | 16 bits     | ~6.4GB          | 100%    |

---

## Stage 4: Benchmark Script

Written `scripts/benchmark.py` (305 LOC) — runs each GGUF model via `llama-cli`,
captures output through a pseudo-terminal (pty), parses timing stats, and saves CSV/JSON.

**Key engineering challenge:** `llama-cli` writes progress output directly to `/dev/tty`,
bypassing stdout/stderr pipes entirely. Fixed by:
- Opening a pty master/slave pair
- Calling `os.setsid()` + `TIOCSCTTY` in the child so the pty slave becomes its controlling terminal
- Reading all output from the pty master side

**Timing format changed in llama.cpp b8000+:**
Old: `llama_print_timings: load time = X ms` (stderr)
New: `[ Prompt: X t/s | Generation: Y t/s ]` (stdout via pty)
Script handles both formats.

---

## Stage 5: bitsandbytes INT4

### What is bitsandbytes?
bitsandbytes is a HuggingFace library for quantizing model weights on-the-fly during loading.
Unlike GGUF (a file format), bitsandbytes quantizes at load time — you load the original
float16 weights and bitsandbytes converts each layer to INT4 as it lands on the GPU.

### Install PyTorch (CUDA 12.4) + dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes
```
**Why CUDA 12.4 wheel:** PyTorch builds are tied to specific CUDA versions.
Our machine has CUDA 12.6 drivers but PyTorch's cu124 wheel is compatible
(CUDA is backward compatible within minor versions).

### HuggingFace login
```bash
python -c "from huggingface_hub import login; login()"
```
**Why:** Meta's Llama models are gated — requires accepting license on HuggingFace
and authenticating with an access token (Read permission). Token saved to
`~/.cache/huggingface/token` after first login.

### Download model (12.9GB — safetensors format)
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.2-3B-Instruct', local_dir='models/Llama-3.2-3B-Instruct-HF')
"
```
**Why safetensors vs GGUF:**
- GGUF = pre-quantized, single file, llama.cpp format
- Safetensors = original float16 weights in HuggingFace format, multiple shards
- bitsandbytes loads safetensors and quantizes on-the-fly to INT4

### What is NF4 quantization?
NF4 (Normal Float 4) is a 4-bit data type designed specifically for neural network weights.
Unlike a uniform 4-bit grid, NF4 places more precision near zero — matching the bell-curve
distribution that LLM weights naturally follow.

**Config used:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # bell-curve optimized 4-bit
    bnb_4bit_use_double_quant=True,      # quantize the scales too (~0.4 bits saved)
    bnb_4bit_compute_dtype=torch.float16,# dequantize to fp16 for actual math
)
```

**Double quantization:** The quantization scales (one per 64 weights) are themselves
quantized from float32 to float8, saving an additional ~0.4 bits/param.

**Compute dtype:** Weights are stored as 4-bit but dequantized to float16 on-the-fly
during matrix multiplication. The GPU computes in fp16, not int4 — so CUDA cores are used,
not tensor cores. This is why bitsandbytes is slower than llama.cpp for pure inference.

### Results

| Model | Backend | Load | TTFT | Tok/s | VRAM |
|-------|---------|------|------|-------|------|
| Llama-3.2-3B-Instruct-INT4-NF4 | bitsandbytes | 6664ms | 335ms | 12.2 | 2361 MB |

**Script:** `scripts/benchmark_int4.py`
**Results:** `results/benchmark_int4_20260331_111033.csv/.json`

### Comparison with GGUF Q4_K_M (same model, same quantization level)

| Backend | Tok/s | VRAM (MB) | Load |
|---------|-------|-----------|------|
| llama.cpp GGUF Q4_K_M | 35.7 | 2711 | fast |
| bitsandbytes NF4 INT4 | 12.2 | 2361 | 6.7s |

**Why GGUF is ~3x faster:**
llama.cpp has hand-written CUDA kernels optimized for quantized matrix-vector multiplication.
bitsandbytes dequantizes weights to fp16 first, then runs standard GEMM — extra step,
lower throughput. bitsandbytes is designed for QLoRA training, not pure inference speed.

**Why bitsandbytes uses less VRAM:**
Double quantization saves ~350MB vs GGUF Q4_K_M (2361 vs 2711 MB).

---

## Stage 6: AWQ (Activation-aware Weight Quantization)

### What is AWQ?
AWQ is a quantization method that compresses weights to INT4 but — unlike GGUF or bitsandbytes —
selects which weights to quantize carefully based on activation magnitudes.

Key insight: not all weights matter equally. A small fraction of weights (those corresponding to
large activations) contribute disproportionately to model quality. AWQ identifies and protects
these weights during quantization, achieving better quality than naive INT4 at the same bit width.

AWQ is a **file format** (like GGUF) — the quantization is done offline and saved to disk.
At inference time you just load the pre-quantized file, no on-the-fly conversion needed.

### AWQ vs bitsandbytes vs GGUF
| Property | GGUF | bitsandbytes NF4 | AWQ |
|----------|------|------------------|-----|
| Quantized at | offline | load time | offline |
| Format | single file | safetensors + config | safetensors + config |
| Inference kernel | hand-written CUDA (fast) | dequant→fp16 GEMM (slow) | fused CUDA (fast when awq_ext installed) |
| Designed for | inference | training (QLoRA) | inference + training |

### Install AutoAWQ
```bash
pip install autoawq
```
**Note:** AutoAWQ is now deprecated — the author handed it off to vLLM's llm-compressor project.
It still works with Torch 2.6.0 + Transformers 4.51.3, which is our configuration.

### Download pre-quantized AWQ model (3.05GB)
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('AMead10/Llama-3.2-3B-Instruct-AWQ', local_dir='models/Llama-3.2-3B-Instruct-AWQ')"
```
**Why pre-quantized:** AWQ calibration (the process of finding which weights to protect)
requires running the full float16 model on a calibration dataset — takes hours and needs
more RAM than this machine has. Pre-quantized models from HuggingFace skip this step.

**Why this repo:** `hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4` does not exist (that org
only quantized Llama 3.1 variants). `AMead10/Llama-3.2-3B-Instruct-AWQ` is a publicly
available, non-gated AWQ INT4 model for Llama 3.2 3B.

### Results

| Model | Backend | Load | TTFT | Tok/s | VRAM |
|-------|---------|------|------|-------|------|
| Llama-3.2-3B-Instruct-AWQ-INT4 | AutoAWQ | 6649ms | 590ms | 5.0 | 3027 MB |

**Script:** `scripts/benchmark_awq.py`
**Results:** `results/benchmark_awq_20260331_120158.csv/.json`

### Why AWQ was slower than expected
Two optimizations failed silently:
1. `awq_ext` CUDA kernel not installed — AutoAWQ fell back to slow PyTorch ops
2. `fuse_layers=True` was skipped (depends on awq_ext) — attention layers not fused

In production (vLLM with llm-compressor), AWQ with proper CUDA kernels is competitive
with GGUF. The 5.0 tok/s result reflects the fallback path, not AWQ's true potential.

### Full cross-backend comparison

| Backend | Tok/s | VRAM (MB) | Load |
|---------|-------|-----------|------|
| GGUF Q4_K_M (llama.cpp) | 35.7 | 2711 | <1s |
| bitsandbytes NF4 | 12.2 | 2361 | 6.7s |
| AutoAWQ INT4 | 5.0 | 3027 | 6.6s |

---

## Stage 7: Results Analysis

### Install analysis dependencies
```bash
pip install matplotlib pandas
```

### Run analysis script
```bash
python scripts/analyze_results.py
```

**Script:** `scripts/analyze_results.py`

Loads all result JSON files, generates 3 plots, and writes `results/findings.md`.

### Plots generated

**throughput_comparison.png** — Bar chart of tok/s across all models and backends.
Shows GGUF dominates inference speed; 7B models are significantly slower than 3B.

**vram_comparison.png** — Peak VRAM usage per model with the 4GB hardware limit marked.
Shows which models are close to the limit and which have headroom.

**throughput_vs_vram.png** — Scatter plot of tok/s vs VRAM (efficiency frontier).
Top-left = best trade-off. GGUF Q4_K_M and Q5_K_M sit closest to the ideal corner.

### Key findings (summary)
1. **GGUF wins on speed** — hand-written CUDA kernels operate on quantized weights directly
2. **More bits ≠ more speed** — Q5_K_M (37.0 tok/s) beats Q4_K_M (35.7) and Q8_0 (25.6)
3. **bitsandbytes saves most VRAM** — 2361MB vs 2711MB (GGUF), thanks to double quantization
4. **7B on 4GB is tight** — Mistral-7B Q3_K_M fits (3845MB), Q4_K_M does not (OOM)
5. **AWQ needs its CUDA extension** — without awq_ext the fallback path is unusably slow

Full analysis: `results/findings.md`
