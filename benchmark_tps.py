import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine as MiniLLM
from myvllm.sampling_parameters import SamplingParams as MiniSamplingParams

try:
    from vllm import LLM as VLLM
    from vllm import SamplingParams as VLLMSamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


DEVICE = resolve_device()


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "xpu":
        torch.xpu.synchronize()


config = {
    "device": DEVICE.type if DEVICE.type != "cpu" else None,
    "max_num_sequences": 16,
    "max_num_batched_tokens": 1024,
    "max_cached_blocks": 1024,
    "block_size": 256,
    "world_size": 1,
    "model_name_or_path": "Qwen/Qwen3-0.6B",
    "enforce_eager": True,
    "vocab_size": 151936,
    "hidden_size": 1024,
    "num_heads": 16,
    "head_dim": 128,
    "num_kv_heads": 8,
    "intermediate_size": 3072,
    "num_layers": 28,
    "tie_word_embeddings": True,
    "base": 1000000,
    "rms_norm_epsilon": 1e-6,
    "qkv_bias": False,
    "scale": 1,
    "max_position": 32768,
    "ffn_bias": False,
    "max_num_batch_tokens": 4096,
    "max_model_length": 128,
    "gpu_memory_utilization": 0.9,
    "eos": 151645,
}

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "give me your opinion on the impact of artificial intelligence on society",
]
WARMUP_STEPS = 2
OUTPUT_TOKENS = 256


def build_prompts(tokenizer):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]


def run_minivllm(tokenizer):
    llm = MiniLLM(config=config)
    sampling = MiniSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
        max_model_length=128,
    )
    prompts = build_prompts(tokenizer)

    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    sync()
    end = time.perf_counter()

    total_tokens = sum(len(x) for x in outputs["token_ids"])
    latency = end - start
    return {"latency": latency, "tokens": total_tokens, "tps": total_tokens / latency}


def run_vllm(tokenizer):
    if not VLLM_AVAILABLE:
        print("Skipping vLLM benchmark: `vllm` is not installed.")
        return None
    if DEVICE.type != "cuda":
        print(f"Skipping vLLM benchmark: current device is `{DEVICE.type}`, vLLM path here is CUDA-oriented.")
        return None

    llm = VLLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=False,
        gpu_memory_utilization=0.75,
        max_model_len=256,
        speculative_config=None,
    )
    sampling = VLLMSamplingParams(temperature=0.6, max_tokens=OUTPUT_TOKENS)
    prompts = build_prompts(tokenizer)

    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    sync()
    end = time.perf_counter()

    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    latency = end - start
    return {"latency": latency, "tokens": total_tokens, "tps": total_tokens / latency}


def run_transformers_test(tokenizer):
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    attention_mask = inputs["attention_mask"]

    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=OUTPUT_TOKENS,
            )
        sync()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=OUTPUT_TOKENS,
        )
    sync()
    end = time.perf_counter()

    total_tokens = sum(len(output) for output in outputs)
    latency = end - start
    return {"latency": latency, "tokens": total_tokens, "tps": total_tokens / latency}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    print(f"Benchmark device: {DEVICE}")

    print("Running miniVLLM benchmark...")
    mini = run_minivllm(tokenizer)

    print("Running vLLM benchmark...")
    vllm_result = run_vllm(tokenizer)

    print("Running transformers benchmark...")
    transformers_result = run_transformers_test(tokenizer)

    results = {
        "miniVLLM": mini,
        "vLLM": vllm_result,
        "transformers": transformers_result,
    }

    print("\n=== Benchmark Results ===")
    for name, metrics in results.items():
        if metrics is None:
            continue
        print(f"{name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
