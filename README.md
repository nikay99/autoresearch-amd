# autoresearch-amd-intel

![teaser](progress.png)

> **⚠️ This is a community fork adding Intel XPU and AMD ROCm support to the original autoresearch project by Andrej Karpathy.**

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

---

## 🍴 Fork Information

| | |
|---|---|
| **Original Author** | [Andrej Karpathy](https://karpathy.ai/) |
| **Original Repository** | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) |
| **This Fork** | [nikay99/autoresearch-amd](https://github.com/nikay99/autoresearch-amd) |
| **Current Branch** | `amd/intel` |
| **Contributions** | Intel XPU + AMD ROCm support |

**All credits for the original codebase and idea go to Andrej Karpathy.** This fork only adds multi-vendor GPU support (Intel & AMD) while maintaining full compatibility with NVIDIA GPUs.

### What's New in This Fork

- ✅ **Intel XPU support** — Arc A770/A750/A580, Data Center GPU Max/Flex series
- ✅ **AMD ROCm support** — MI300X, MI250X, MI210, MI100
- ✅ **Universal device abstraction** — Auto-detects Intel XPU / AMD ROCm / NVIDIA CUDA
- ✅ **Automatic backend selection** — SDPA for Intel, Flash Attention for AMD/NVIDIA
- ✅ **Unified codebase** — Same code runs on all three GPU vendors

---

## About

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org.

A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Platform Support

This fork supports **three GPU vendors** with automatic detection:

| Vendor | GPUs | Backend | Attention Implementation |
|--------|------|---------|--------------------------|
| **Intel** | Arc A770/A750/A580, Data Center GPU Max/Flex | XPU (IPEX) | PyTorch SDPA |
| **AMD** | MI300X, MI250X, MI210, MI100 | ROCm 6.2.4+ | flash-attn-rocm / SDPA fallback |
| **NVIDIA** | H100, A100, L40S, RTX 4090/3090 | CUDA | Flash Attention 3 |

The code **automatically detects** your GPU and configures the optimal backend. No manual configuration needed!

## Quick Start

**Requirements:** Intel GPU, AMD GPU, OR NVIDIA GPU + Python 3.10+ + [uv](https://docs.astral.sh/uv/)

### Clone This Branch

```bash
git clone https://github.com/nikay99/autoresearch-amd.git
cd autoresearch-amd
git checkout amd/intel  # <-- Important: switch to this branch
```

### Intel XPU Setup

```bash
# 1. Install Intel GPU drivers and oneAPI
# https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html

# 2. Install dependencies
uv sync --extra intel

# 3. Download data and train tokenizer (~2 min)
uv run prepare.py

# 4. Run training (~5 min)
uv run train.py
```

### AMD ROCm Setup

```bash
# Ensure ROCm 6.2.4+ is installed

# Install dependencies
uv sync --extra rocm

# Optional but recommended: Install Flash Attention for AMD
pip install flash-attn-rocm

# Download data and run
uv run prepare.py
uv run train.py
```

### NVIDIA CUDA Setup

```bash
# Install dependencies
uv sync --extra cuda

# Download data and run
uv run prepare.py
uv run train.py
```

## How It Works

The repo is deliberately kept small with only three files that matter:

- **`prepare.py`** — Fixed constants, data prep (downloads training data, trains BPE tokenizer), runtime utilities. **Do not modify.**
- **`train.py`** — The file the agent edits. Model architecture, optimizer, training loop. Everything is fair game: architecture, hyperparameters, batch size, etc. **This is the only file you edit.**
- **`program.md`** — Instructions for the AI agent. **This is edited by the human.**

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better.

## Running the Agent

Spin up your Claude/Codex (disable all permissions), then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project Structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design Choices

- **Single file to modify** — Only `train.py` is edited by the agent
- **Fixed time budget** — Always 5 minutes, makes experiments comparable
- **Self-contained** — No distributed training, no complex configs
- **Universal GPU support** — Same code runs on Intel, AMD, and NVIDIA

## Technical Details

### Universal Device Abstraction

The codebase uses a universal abstraction layer that automatically handles:
- Device detection (`torch.xpu`, `torch.cuda`)
- Memory synchronization (`torch.xpu.synchronize()`, `torch.cuda.synchronize()`)
- Peak memory tracking
- Autocast contexts for each backend

This allows the same code to run on Intel, AMD, and NVIDIA GPUs without modification.

### Intel XPU Specifics

- Uses **Intel Extension for PyTorch (IPEX)** for optimized operations
- **Flash Attention**: Not yet available for Intel GPUs → uses PyTorch's native SDPA
- **Memory**: XPU uses `torch.xpu.*` APIs (abstracted by our layer)
- **Performance**: Intel Arc A770 achieves ~60-70% of RTX 4090 LLM training performance

### AMD ROCm Specifics

- Uses ROCm-enabled PyTorch
- **Flash Attention**: Optional `flash-attn-rocm` package for best performance
- Falls back to PyTorch SDPA if Flash Attention not installed
- Same `torch.cuda.*` APIs as NVIDIA (ROCm compatibility layer)

### NVIDIA CUDA Specifics

- Uses standard CUDA PyTorch
- **Flash Attention 3** via `kernels` package
- Best performance on Hopper (H100) and Ampere (A100) architectures

## Hardware Recommendations for Smaller GPUs

If you're running on smaller GPUs (Arc A750, RX 7900 XTX, RTX 4070, etc.):

1. Use a lower-entropy dataset like [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
2. Decrease `vocab_size` from 8192 to 4096, 2048, or 1024
3. Lower `MAX_SEQ_LEN` in `prepare.py` (try 1024 or 512)
4. Decrease `EVAL_TOKENS` for faster validation
5. Lower `DEPTH` in `train.py` (try 4-6 instead of 8)
6. Use `WINDOW_PATTERN = "L"` instead of `"SSSL"` for simpler attention
7. Reduce `TOTAL_BATCH_SIZE` to `2**16` or `2**14`

## Branches in This Fork

| Branch | Description |
|--------|-------------|
| `master` | AMD ROCm + NVIDIA CUDA |
| `intel-xpu` | Intel XPU + AMD ROCm + NVIDIA CUDA (universal) |
| `amd/intel` | **Current branch** — Intel XPU + AMD ROCm + NVIDIA CUDA |

## Notable Forks (Original Project)

- [nikay99/autoresearch-amd](https://github.com/nikay99/autoresearch-amd) (Intel XPU / AMD ROCm / NVIDIA CUDA) — **This fork!**
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## Original Project

- **Author**: [Andrej Karpathy](https://karpathy.ai/)
- **Repository**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **Twitter**: [@karpathy](https://x.com/karpathy)
- **Original Idea**: Autonomous AI research swarms experimenting on LLM training

This fork is a **community contribution** adding multi-vendor GPU support (Intel & AMD). If you find this useful, please ⭐ star both the [original repository](https://github.com/karpathy/autoresearch) and [this fork](https://github.com/nikay99/autoresearch-amd)!

## License

MIT (same as original project)
