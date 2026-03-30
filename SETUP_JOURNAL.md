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

### Configure the build
```bash
cd ~/llm-quant-bench/llama.cpp
cmake -B Build
```
**Why:** This is the "configure" step. CMake scans your system and prepares build files.
Output showed:
- OpenMP found → parallel CPU inference enabled (uses multiple cores)
- OpenSSL missing → no HTTPS support, but doesn't affect inference
- x86 detected → will use AVX2 optimized CPU math instructions

---

### Compile llama.cpp
```bash
cmake --build Build -j$(nproc)
```
**Why:** Actually compiles all C++ source files into executables.
- `--build Build` — compile what CMake configured in the Build folder
- `-j$(nproc)` — use all CPU cores in parallel (`nproc` = 22 on this machine)

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

## TODO — Upcoming Stages

### Stage 4: Write benchmark script
- Python script that runs same prompt through multiple quant levels
- Measures TTFT, tokens/sec, VRAM
- Saves results to results/

### Stage 5: bitsandbytes INT4
- Install PyTorch + bitsandbytes
- Load HuggingFace model in 4-bit on the fly
- Compare quality vs GGUF Q4

### Stage 6: AWQ
- Install autoawq
- Download AWQ quantized model from HuggingFace
- Compare with GGUF and bitsandbytes

### Stage 7: Results analysis
- Plot VRAM vs quality tradeoff
- Write findings in results/findings.md
- Push everything to GitHub
