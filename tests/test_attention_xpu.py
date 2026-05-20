import pytest
import torch

from myvllm.layers.attention import (
    TRITON_AVAILABLE,
    _flash_attention_prefill_fallback,
    _paged_attention_decode_fallback,
    _store_kvcache_fallback,
    flash_attention_prefill,
    paged_attention_decode,
    store_kvcache,
)
from myvllm.utils.device import supports_triton


HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
XPU_DEVICE = torch.device("xpu") if HAS_XPU else None


def _sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.skipif(not (TRITON_AVAILABLE and HAS_XPU), reason="requires triton + xpu")
def test_supports_triton_accepts_xpu() -> None:
    assert supports_triton(torch.device("xpu"))


@pytest.mark.skipif(not (TRITON_AVAILABLE and HAS_XPU), reason="requires triton + xpu")
def test_store_kvcache_matches_fallback_on_xpu() -> None:
    torch.manual_seed(0)

    num_tokens = 9
    num_kv_heads = 2
    head_dim = 32
    block_size = 4
    total_blocks = 3

    key = torch.randn(num_tokens, num_kv_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    slot_mapping = torch.tensor([5, -1, 1, 7, 0, 8, 3, -1, 6], device=XPU_DEVICE, dtype=torch.int32)

    k_cache_triton = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    v_cache_triton = torch.zeros_like(k_cache_triton)
    k_cache_ref = torch.zeros_like(k_cache_triton)
    v_cache_ref = torch.zeros_like(v_cache_triton)

    store_kvcache(key, value, k_cache_triton, v_cache_triton, slot_mapping, block_size)
    _store_kvcache_fallback(key, value, k_cache_ref, v_cache_ref, slot_mapping, block_size)
    _sync(XPU_DEVICE)

    assert torch.equal(k_cache_triton.cpu(), k_cache_ref.cpu())
    assert torch.equal(v_cache_triton.cpu(), v_cache_ref.cpu())


@pytest.mark.skipif(not (TRITON_AVAILABLE and HAS_XPU), reason="requires triton + xpu")
@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "num_kv_heads", "head_dim"),
    [
        ([7, 11], 4, 4, 32),
        ([13, 9], 8, 2, 64),
        ([17], 8, 8, 128),
    ],
)
def test_flash_attention_prefill_matches_fallback_on_xpu(
    seq_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(sum(seq_lens) + num_heads + num_kv_heads + head_dim)

    total_tokens = sum(seq_lens)
    q = torch.randn(total_tokens, num_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=XPU_DEVICE, dtype=torch.int32)
    scale = head_dim ** -0.5

    out_triton = flash_attention_prefill(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    out_ref = _flash_attention_prefill_fallback(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    _sync(XPU_DEVICE)

    assert torch.allclose(out_triton.cpu(), out_ref.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not (TRITON_AVAILABLE and HAS_XPU), reason="requires triton + xpu")
@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim", "block_size"),
    [
        (2, 19, 4, 4, 32, 8),
        (3, 33, 8, 2, 64, 16),
        (1, 97, 8, 8, 128, 16),
    ],
)
def test_paged_attention_decode_matches_fallback_on_xpu(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    torch.manual_seed(batch_size + seq_len + num_heads + num_kv_heads + head_dim + block_size)

    scale = head_dim ** -0.5
    query = torch.randn(batch_size, num_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    context_lens = torch.tensor(
        [seq_len - i * max(1, seq_len // max(batch_size, 2)) for i in range(batch_size)],
        device=XPU_DEVICE,
        dtype=torch.int32,
    )
    max_num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks

    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=XPU_DEVICE, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)

    block_tables = torch.full((batch_size, max_num_blocks), -1, device=XPU_DEVICE, dtype=torch.int32)
    next_block = 0
    for batch_idx in range(batch_size):
        blocks_needed = (int(context_lens[batch_idx].item()) + block_size - 1) // block_size
        physical = list(range(next_block, next_block + blocks_needed))
        if batch_idx % 2 == 1:
            physical.reverse()
        block_tables[batch_idx, :blocks_needed] = torch.tensor(physical, device=XPU_DEVICE, dtype=torch.int32)
        next_block += blocks_needed

    out_triton = paged_attention_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    )
    out_ref = _paged_attention_decode_fallback(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    )
    _sync(XPU_DEVICE)

    assert torch.allclose(out_triton.cpu(), out_ref.cpu(), atol=2e-2, rtol=2e-2)
