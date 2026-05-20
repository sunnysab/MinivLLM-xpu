import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_tp_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def get_tp_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(device: str | None = None, rank: int = 0) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
        if resolved.type != "cpu" and resolved.index is None:
            resolved = torch.device(f"{resolved.type}:{rank}")
        return resolved

    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    if is_xpu_available():
        return torch.device(f"xpu:{rank}")
    return torch.device("cpu")


def get_distributed_backend(device: torch.device, world_size: int, configured: str | None = None) -> str | None:
    if world_size <= 1:
        return None
    if configured:
        return configured
    if device.type == "cuda":
        return "nccl"
    if device.type == "xpu":
        if hasattr(dist, "is_xccl_available") and dist.is_xccl_available():
            return "xccl"
        return "gloo"
    return "gloo"


def set_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "xpu":
        torch.xpu.set_device(device)


def supports_cuda_graphs(device: torch.device) -> bool:
    return device.type == "cuda"


def supports_triton(device: torch.device) -> bool:
    return device.type in {"cuda", "xpu"}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats(device)


def mem_get_info(device: torch.device) -> tuple[int, int]:
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "mem_get_info"):
        return backend.mem_get_info(device)
    if backend is not None and hasattr(backend, "get_device_properties"):
        props = backend.get_device_properties(device)
        total_mem = getattr(props, "total_memory")
        allocated = memory_stats(device)["allocated_bytes.all.current"]
        return max(total_mem - allocated, 0), total_mem
    raise RuntimeError(f"Memory query is not implemented for device type {device.type!r}")


def memory_stats(device: torch.device) -> dict[str, int]:
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "memory_stats"):
        return backend.memory_stats(device)

    current = 0
    peak = 0
    if backend is not None and hasattr(backend, "memory_allocated"):
        current = backend.memory_allocated(device)
    if backend is not None and hasattr(backend, "max_memory_allocated"):
        peak = backend.max_memory_allocated(device)
    return {
        "allocated_bytes.all.current": current,
        "allocated_bytes.all.peak": peak,
    }
