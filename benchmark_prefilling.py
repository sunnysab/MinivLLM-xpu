import time

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


DEVICE = resolve_device()
USE_TRITON = TRITON_AVAILABLE and DEVICE.type in {"cuda", "xpu"}


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "xpu":
        torch.xpu.synchronize()


def pytorch_standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    del head_dim
    output = torch.zeros_like(q)
    cu_seqlens_cpu = cu_seqlens.cpu().tolist()

    for i in range(len(cu_seqlens_cpu) - 1):
        start = cu_seqlens_cpu[i]
        end = cu_seqlens_cpu[i + 1]
        seq_len = end - start

        q_seq = q[start:end].transpose(0, 1)
        k_seq = k[start:end].transpose(0, 1)
        v_seq = v[start:end].transpose(0, 1)

        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(num_groups, dim=0)
            v_seq = v_seq.repeat_interleave(num_groups, dim=0)

        attn_scores = torch.matmul(q_seq, k_seq.transpose(1, 2)) * scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out_seq = torch.matmul(attn_probs, v_seq).transpose(0, 1)
        output[start:end] = out_seq

    return output


if USE_TRITON:
    @triton.jit
    def naive_triton_attention_kernel(
        Q,
        K,
        V,
        O,
        cu_seqlens_q_ptr,
        scale,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)

        seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
        seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
        seq_len = seq_end - seq_start
        if seq_len > BLOCK_SIZE:
            return

        offs_m = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, head_dim)
        mask = offs_m < seq_len

        q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + head_idx * head_dim + offs_d[None, :]
        k_ptrs = K + (seq_start + offs_m[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v_ptrs = V + (seq_start + offs_m[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]

        q = tl.load(q_ptrs, mask=mask[:, None], other=0.0)
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        causal_mask = offs_m[:, None] >= offs_m[None, :]
        seq_mask = mask[:, None] & mask[None, :]
        qk = tl.where(causal_mask & seq_mask, qk, float("-inf"))

        qk_max = tl.max(qk, axis=1)
        qk_exp = tl.exp(qk - qk_max[:, None])
        qk_sum = tl.sum(tl.where(seq_mask, qk_exp, 0.0), axis=1)
        attn = qk_exp / qk_sum[:, None]
        out = tl.dot(attn.to(v.dtype), v)

        o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + head_idx * head_dim + offs_d[None, :]
        tl.store(o_ptrs, out, mask=mask[:, None])


    def naive_triton_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = torch.empty_like(q)
        num_seqs = cu_seqlens.shape[0] - 1

        BLOCK_SIZE = 128 if head_dim <= 64 else 64
        actual_size = 2 ** ((max_seq_len - 1).bit_length())
        actual_size = min(actual_size, BLOCK_SIZE)
        grid = (num_seqs, num_heads)

        naive_triton_attention_kernel[grid](
            q,
            k,
            v,
            output,
            cu_seqlens,
            scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_SIZE=actual_size,
        )
        return output


    @triton.jit
    def flash_attention_kernel(
        Q,
        K,
        V,
        O,
        cu_seqlens_q_ptr,
        scale,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        seq_idx = tl.program_id(2)
        kv_head_idx = off_h // (num_heads // num_kv_heads)

        seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
        seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
        seq_len = seq_end - seq_start
        if start_m * BLOCK_M >= seq_len:
            return

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, head_dim)
        q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
        mask_m = offs_m < seq_len
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
        acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
        num_blocks = tl.cdiv(seq_len, BLOCK_N)

        for block_n in range(num_blocks):
            start_n = block_n * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len

            k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
            qk = tl.dot(q, k) * scale

            mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
            qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
            acc = acc * alpha[:, None]

            v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            acc = acc + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new

        acc = acc / l_i[:, None]
        o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


    def flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = torch.empty_like(q)

        if head_dim <= 64:
            BLOCK_M, BLOCK_N = 64, 64
        elif head_dim <= 128:
            BLOCK_M, BLOCK_N = 32, 32
        else:
            BLOCK_M, BLOCK_N = 16, 16

        num_seqs = cu_seqlens.shape[0] - 1
        cu_seqlens_cpu = cu_seqlens.cpu()
        max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
        grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

        flash_attention_kernel[grid](
            q,
            k,
            v,
            output,
            cu_seqlens,
            scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return output

else:
    def naive_triton_attention(*args, **kwargs):
        raise RuntimeError("Triton benchmark requires a Triton-supported device (`cuda` or `xpu`) and `triton`.")


    def flash_attention(*args, **kwargs):
        raise RuntimeError("Triton benchmark requires a Triton-supported device (`cuda` or `xpu`) and `triton`.")


def setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim):
    total_tokens = num_seqs * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=torch.float16)
    cu_seqlens = torch.tensor([i * seq_len for i in range(num_seqs + 1)], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / (head_dim ** 0.5)
    return q, k, v, cu_seqlens, scale


def benchmark(num_seqs, seq_len, num_heads=32, num_kv_heads=8, head_dim=128, num_iter=50):
    print(f"\n{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Benchmark: {num_seqs} seqs × {seq_len} tokens (total: {num_seqs * seq_len} tokens)")
    print(f"Heads: {num_heads}/{num_kv_heads}, Dim: {head_dim}")
    print(f"{'='*80}")

    q, k, v, cu_seqlens, scale = setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim)
    results = {}

    print("\n[1/3] PyTorch Standard (O(N²) memory)...")
    for _ in range(5):
        _ = pytorch_standard_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    sync()
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = pytorch_standard_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    sync()
    results["PyTorch (O(N²))"] = (time.perf_counter() - start) / num_iter
    print(f"      {results['PyTorch (O(N²))'] * 1000:.3f} ms")

    if USE_TRITON:
        max_safe_seq = 64 if head_dim > 64 else 128
        if seq_len <= max_safe_seq:
            print("\n[2/3] Naive Triton (O(N²), materializes full attention)...")
            for _ in range(5):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            sync()
            start = time.perf_counter()
            for _ in range(num_iter):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            sync()
            results["Naive Triton (O(N²))"] = (time.perf_counter() - start) / num_iter
            print(f"      {results['Naive Triton (O(N²))'] * 1000:.3f} ms")
        else:
            print(f"\n[2/3] Naive Triton: SKIPPED (seq_len={seq_len} > {max_safe_seq}, would exceed shared memory)")

        print("\n[3/3] Flash Attention (O(N), online softmax)...")
        for _ in range(5):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        sync()
        start = time.perf_counter()
        for _ in range(num_iter):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        sync()
        results["Flash Attention (O(N))"] = (time.perf_counter() - start) / num_iter
        print(f"      {results['Flash Attention (O(N))'] * 1000:.3f} ms")
    else:
        print("\n[2/3] Naive Triton: SKIPPED (requires triton + supported device: cuda/xpu)")
        print("[3/3] Flash Attention: SKIPPED (requires triton + supported device: cuda/xpu)")

    return results


def find_crossover_point():
    if not USE_TRITON:
        print("\nCrossover analysis skipped: Triton requires `triton` plus a supported device (`cuda` or `xpu`).")
        return

    print("\n" + "=" * 80)
    print("FINDING CROSSOVER POINT: When does Flash beat Naive?")
    print("=" * 80)

    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_seqs = 2
    seq_lengths = [16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512, 1024]

    for seq_len in seq_lengths:
        print(f"\nTesting seq_len = {seq_len}...")
        q, k, v, cu_seqlens, scale = setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim)

        max_safe_seq = 64 if head_dim > 64 else 128
        naive_time = None
        if seq_len <= max_safe_seq:
            for _ in range(10):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            sync()
            start = time.perf_counter()
            for _ in range(50):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            sync()
            naive_time = (time.perf_counter() - start) / 50

        for _ in range(10):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        sync()
        start = time.perf_counter()
        for _ in range(50):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        sync()
        flash_time = (time.perf_counter() - start) / 50

        if naive_time is None:
            print(f"  Naive: SKIPPED | Flash: {flash_time * 1000:.3f}ms")
        else:
            winner = "Naive" if naive_time < flash_time else "Flash"
            print(f"  Naive: {naive_time * 1000:.3f}ms | Flash: {flash_time * 1000:.3f}ms | Winner: {winner}")


def analyze_kernel_launches():
    if not USE_TRITON:
        return

    print("\n" + "=" * 80)
    print("KERNEL LAUNCH ANALYSIS")
    print("=" * 80)
    num_seqs = 2
    seq_len = 60
    num_heads = 32
    block_m = 32
    naive_grid = (num_seqs, num_heads)
    naive_kernels = num_seqs * num_heads
    num_blocks_m = (seq_len + block_m - 1) // block_m
    flash_grid = (num_blocks_m, num_heads, num_seqs)
    flash_kernels = num_blocks_m * num_heads * num_seqs

    print(f"\nFor {num_seqs} sequences × {seq_len} tokens:")
    print(f"  Naive Triton grid:    {naive_grid}")
    print(f"  Naive total kernels:  {naive_kernels}")
    print(f"  Flash Attention grid: {flash_grid}")
    print(f"  Flash total kernels:  {flash_kernels}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PREFILL ATTENTION BENCHMARK")
    print("Comparing: PyTorch | optional Triton kernels")
    print("=" * 80)

    benchmark(num_seqs=2, seq_len=60, num_iter=100)
    benchmark(num_seqs=4, seq_len=64, num_iter=100)
    benchmark(num_seqs=2, seq_len=1024, num_iter=30)
    benchmark(num_seqs=1, seq_len=4096, num_iter=10)
    find_crossover_point()
    analyze_kernel_launches()
