<p align="center">
  <img src="./assets/minivllm.png" alt="图片描述" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# miniVLLM

A custom implementation of vLLM inference engine with attention mechanism benchmarks, based on Nano-vLLM but with self-contained paged attention and flash attention implementation. 

Benchmarking on flash attention in prefilling time and paged attention in decoding time are provided.

**New to vLLM?** Check out [HowToApproachvLLM.md](HowToApproachvLLM.md) for a step-by-step implementation guide covering layers, models, paged attention, CUDA graphs, and scheduling.

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project env
uv venv
source .venv/bin/activate

# Install project deps: pulls torch packages from the configured XPU index
uv sync --extra xpu --inexact

# Run the main inference engine
uv run python main.py

# Run prefilling benchmark
uv run python benchmark_prefilling.py

# Run decoding benchmark
uv run python benchmark_decoding.py
```

To run multi-GPU setting, simply change world_size to n > 1 in config in main.py

## Intel XPU Quickstart

This project targets `torch.xpu` for the main inference path.

```bash
# Create / activate your env first
uv venv
source .venv/bin/activate

# Install project deps with the XPU extra
uv sync --extra xpu --inexact

# Run on XPU explicitly
MINIVLLM_DEVICE=xpu uv run --extra xpu --no-sync python main.py
```

For multi-XPU distributed runs, install oneCCL bindings separately and set `MINIVLLM_DIST_BACKEND=ccl`. Single-XPU runs do not need oneCCL.

## What Each Script Does

```bash
uv run python main.py
```

This is the main inference engine demo

Demonstrates the complete LLM inference pipeline using a custom engine implementation:
- Create a small version of Qwen3 with random initialization
- Creates 60 chat prompts (2 base prompts repeated 30 times each)
- Processes them through the custom LLM engine with batch processing
- Uses paged attention and KV cache management for efficient inference
- Generates up to 256 tokens per prompt with temperature sampling

This showcases how the custom vLLM implementation handles batched text generation with memory-efficient attention.

```bash
uv run python benchmark_prefilling.py
```

This is the pefilling phase comparison

Compares the prefill implementations used by the project on the PyTorch path.

```bash
uv run python benchmark_decoding.py
```

This is the decoding phase comparison

Compares the decoding implementations used by the project on the PyTorch path.


## Project Structure

```
myvllm/
├── src/
│   └── myvllm/           # Core vLLM implementation
│       ├── models/       # Model implementations
│       ├── engine/       # LLM engine logic, including sequence definition for input prompts, block management for KV cache management for GPU, scheduler for iteration-based scheduling of sequences, runner for actual implementation of running prefilling and decoding, and engine for generation API interface
│       ├── layers/       # Components for model/
│       ├── utils/        # context
│       └── sampling_parameters.py
├── main.py              # Full inference demo
├── benchmark_prefilling.py   # Prefilling attention comparison
└── benchmark_decoding.py     # Decoding attention comparison
```

## Requirements

- Python ≥3.11, < 3.12
- Main engine: `transformers`, `xxhash`
- Backend runtime: install `torch` through the `xpu` extra
- Intel XPU requires an Intel GPU plus the configured PyTorch XPU wheel index


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://www.star-history.com/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)
