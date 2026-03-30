# LLM Quantization Benchmark

Hands-on benchmarking of LLM quantization formats on local hardware.

## Hardware
- GPU: NVIDIA RTX 500 Ada (4GB VRAM)
- CPU: Intel Core Ultra 7 165H (22 threads)
- RAM: 15GB
- OS: Ubuntu 24.04 (WSL2)

## What I'm measuring
- VRAM usage (MB)
- Tokens per second (throughput)
- Time to first token / TTFT (latency)
- Output quality vs FP16 baseline

## Formats covered
- GGUF via llama.cpp (Q3, Q4, Q5, Q8)
- INT4 via bitsandbytes
- AWQ via autoawq

## Models
- Llama 3.2 3B
- Mistral 7B

## Stages
- Stage 1: llama.cpp + GGUF quantization
- Stage 2: bitsandbytes INT4
- Stage 3: AWQ
