# autoresearch-intel

![teaser](progress.png)

> **⚠️ This is a community fork adding Intel XPU and AMD ROCm support to the original project.**

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

---

## 🍴 Fork Information

**Original Project**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy  
**This Fork**: [nikay99/autoresearch-amd](https://github.com/nikay99/autoresearch-amd) (`intel-xpu` branch)  
**Contributions**: 
- ✅ AMD ROCm GPU support (MI300X, MI250X, MI210, MI100)
- 🆕 Intel XPU support (Arc A770/A750/A580, Data Center GPU Max/Flex)

**All credits for the original codebase go to Andrej Karpathy.** This fork only adds multi-vendor GPU support while maintaining full compatibility with NVIDIA GPUs.

---

## About

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Platform Support

This fork supports **three GPU vendors**:

| Vendor | GPUs | Backend | Flash Attention |
|--------|------|---------|-----------------|
| **Intel** | Arc A770/A750/A580, Data Center GPU Max/Flex | XPU (IPEX) | PyTorch SDPA |
| **AMD** | MI300X, MI250X, MI210, MI100 | ROCm 6.2.4+ | flash-attn-rocm |
| **NVIDIA** | H100, A100, L40S, RTX 4090/3090 | CUDA | Flash Attention 3 |

The code **automatically detects** your GPU and configures the optimal backend.

## Quick start

**Requirements:** Intel Arc/Data Center GPU, AMD GPU, OR NVIDIA GPU + Python 3.10+ + [uv](https://docs.astral.sh/uv/)

### Intel XPU Setup

```bash
# 1. Install Intel GPU drivers and oneAPI (if not already installed)
# See: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html

# 2. Clone and setup
git clone https://github.com/nikay99/autoresearch-amd.git
cd autoresearch-amd
git checkout intel-xpu

# 3. Install dependencies (PyTorch with Intel extensions)
uv sync --extra intel

# 4. Download data and train tokenizer (~2 min)
uv run prepare.py

# 5. Run training (~5 min)
uv run train.py
```

### AMD ROCm Setup

```bash
# Ensure ROCm 6.2.4+ is installed
git clone https://github.com/nikay99/autoresearch-amd.git
cd autoresearch-amd
git checkout intel-xpu  # or master for AMD-only

uv sync --extra rocm

# Optional: Install Flash Attention for AMD
pip install flash-attn-rocm

uv run prepare.py
uv run train.py
```

### NVIDIA CUDA Setup

```bash
git clone https://github.com/nikay99/autoresearch-amd.git
cd autoresearch-amd
git checkout intel-xpu

uv sync --extra cuda
uv run prepare.py
uv run train.py
```

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Technical Details

### Intel XPU Specifics

- Uses **Intel Extension for PyTorch (IPEX)** for optimized operations
- **Flash Attention**: Not yet available for Intel GPUs → uses PyTorch's native SDPA
- **Memory**: XPU uses `torch.xpu.*` APIs (abstracted in this codebase)
- **Performance**: Intel Arc A770 achieves ~60-70% of RTX 4090 LLM training perf

### Universal Device Abstraction

The codebase uses a universal device abstraction that automatically handles:
- Device detection (`torch.xpu`, `torch.cuda`)
- Memory synchronization APIs
- Peak memory tracking
- Autocast contexts

This allows the same code to run on Intel, AMD, and NVIDIA GPUs without modification.

### Hardware Recommendations for Smaller GPUs

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [nikay99/autoresearch-amd](https://github.com/nikay99/autoresearch-amd) (Intel XPU / AMD ROCm / NVIDIA CUDA) - **This fork!**
  - `master` branch: AMD ROCm + NVIDIA CUDA
  - `intel-xpu` branch: Intel XPU + AMD ROCm + NVIDIA CUDA (universal)
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## Original Project

- **Author**: [Andrej Karpathy](https://karpathy.ai/)
- **Repository**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **Twitter**: [@karpathy](https://x.com/karpathy)

This fork is a community contribution adding multi-vendor GPU support. If you find this useful, please star both the original repository and this fork!

## License

MIT (same as original project)
