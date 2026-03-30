"""
Download GGUF model files from HuggingFace for benchmarking.

Models targeted:
  - Llama 3.2 3B Instruct: Q3_K_M, Q5_K_M, Q8_0  (already have Q4_K_M)
  - Mistral 7B Instruct v0.3: Q3_K_M (fits in 4GB VRAM), Q4_K_M (borderline)

Why these models:
  - Llama 3.2 3B at 4 quant levels = pure quantization comparison (same model, same task)
  - Mistral 7B adds a cross-model dimension (larger model, same quant format)

Usage:
  python scripts/download_models.py              # download all missing models
  python scripts/download_models.py --dry-run    # just show what would be downloaded
"""

import argparse
import sys
import requests
from pathlib import Path
from huggingface_hub import hf_hub_url, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

MODELS_DIR = Path(__file__).parent.parent / "models"

# Models to download: (repo_id, filename, description)
MODELS = [
    # Llama 3.2 3B — remaining quant levels
    (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q3_K_L.gguf",
        "Llama 3.2 3B Q3_K_L  (~1.5GB) — aggressive compression, lower quality",
    ),
    (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "Llama 3.2 3B Q5_K_M  (~2.3GB) — high quality, still fits 4GB VRAM",
    ),
    (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Llama 3.2 3B Q8_0    (~3.3GB) — near-full quality, uses most of VRAM",
    ),
    # Mistral 7B — cross-model comparison
    # Q3_K_M (~3.3GB) is safe; Q4_K_M (~4.1GB) may OOM on 4GB VRAM
    (
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-Q3_K_M.gguf",
        "Mistral 7B Q3_K_M    (~3.3GB) — fits 4GB VRAM, cross-model baseline",
    ),
    (
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "Mistral 7B Q4_K_M    (~4.1GB) — may be tight on 4GB VRAM, worth trying",
    ),
]


def sizeof_fmt(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def get_remote_size(url, headers):
    """HEAD request to get expected file size from server."""
    try:
        r = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        return int(r.headers.get("content-length", 0))
    except Exception:
        return 0


def clean_partial(repo_id, filename, dry_run=False):
    """Remove file if its size doesn't match the remote size (partial download)."""
    dest = MODELS_DIR / filename
    if not dest.exists():
        return

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    token = HfApi().token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    remote_size = get_remote_size(url, headers)

    local_size = dest.stat().st_size
    if remote_size and local_size != remote_size:
        print(
            f"  [PARTIAL] {filename} — local {sizeof_fmt(local_size)} != remote {sizeof_fmt(remote_size)}"
        )
        if not dry_run:
            dest.unlink()
            print(f"  [REMOVED] {filename}")
        else:
            print(f"  [DRY-RUN] would remove {filename}")


def download_model(repo_id, filename, description, dry_run=False):
    dest = MODELS_DIR / filename

    if dest.exists():
        print(f"  [SKIP]  {filename} — already downloaded ({sizeof_fmt(dest.stat().st_size)})")
        return True

    print(f"  [DOWN]  {filename}")
    print(f"          {description}")

    if dry_run:
        print(f"          → would save to {dest}")
        return True

    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        token = HfApi().token
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            last_reported_pct = -1
            start_time = __import__("time").monotonic()

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        pct = int(downloaded / total * 100)
                        # print a line every 5%
                        if pct >= last_reported_pct + 5:
                            elapsed = __import__("time").monotonic() - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            eta_sec = (total - downloaded) / speed if speed > 0 else 0
                            eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"
                            print(
                                f"  {pct:3d}%  {sizeof_fmt(downloaded):>9} / {sizeof_fmt(total)}"
                                f"  {sizeof_fmt(speed):>10}/s  ETA {eta_str}",
                                flush=True,
                            )
                            last_reported_pct = pct

        print(f"          Saved: {dest} ({sizeof_fmt(dest.stat().st_size)})")
        return True

    except EntryNotFoundError:
        print(f"  [FAIL]  {filename} not found in {repo_id}")
        if dest.exists():
            dest.unlink()
        return False
    except RepositoryNotFoundError:
        print(f"  [FAIL]  Repo not found: {repo_id}")
        return False
    except Exception as e:
        print(f"  [FAIL]  {e}")
        if dest.exists():
            dest.unlink()  # remove partial download
        return False


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for benchmarking")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nModels directory: {MODELS_DIR}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DOWNLOADING'}\n")

    print("Checking for partial downloads...")
    for repo_id, filename, _ in MODELS:
        clean_partial(repo_id, filename, dry_run=args.dry_run)
    print()

    results = []
    for repo_id, filename, description in MODELS:
        ok = download_model(repo_id, filename, description, dry_run=args.dry_run)
        results.append((filename, ok))
        print()

    print("=" * 60)
    print("Summary:")
    for filename, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6}] {filename}")

    failed = [f for f, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} download(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll done. Models in {MODELS_DIR}")


if __name__ == "__main__":
    main()
