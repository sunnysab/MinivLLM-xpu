import torch
import torch.nn as nn
import torch.nn.functional as F

from myvllm.utils import get_context, supports_triton

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _expand_kv_heads(x: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    # GQA/MQA 场景下，多个 Q head 会共享一个 KV head。
    # 下面的 PyTorch fallback 写法默认每个 Q head 都有一个“对应的” KV head，
    # 所以这里先把 KV head 显式重复展开。
    if num_heads == num_kv_heads:
        return x
    repeat_factor = num_heads // num_kv_heads
    return x.repeat_interleave(repeat_factor, dim=1)


def _store_kvcache_fallback(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    valid = slot_mapping >= 0
    if not torch.any(valid):
        return

    """
    kv cache 一共有 num_blocks 个块，一个块有 block_size 个 slot.
    每个 slot 存一份 K/V
    
    例如：
    - block_size = 4
    - slot_mapping = [5, -1, 1, 7]
    那么：
      - token0 写到 slot 5，也就是 block 1, offset 1
      - token1 是 -1，表示这个 token 不写 cache
      - token2 写到 block 0, offset 1
      - token3 写到 block 1, offset 3
    """
    slot_idx = slot_mapping[valid]
    block_idx = torch.div(slot_idx, block_size, rounding_mode="floor")
    block_offset = slot_idx % block_size
    k_cache[block_idx, block_offset] = key[valid]
    v_cache[block_idx, block_offset] = value[valid]


def _flash_attention_prefill_fallback(
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
    output = torch.empty_like(q)

    # 这里使用 packed varlen 布局：
    #   q/k/v shape == (total_tokens, num_heads_or_kv_heads, head_dim)
    #   cu_seqlens = [0, len(seq0), len(seq0)+len(seq1), ...]
    # 这样就能在不做 padding 的前提下，还原出每条序列在大张量中的切片范围。
    for seq_start, seq_end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        q_seq = q[seq_start:seq_end].transpose(0, 1)
        k_seq = _expand_kv_heads(k[seq_start:seq_end], num_heads, num_kv_heads).transpose(0, 1)
        v_seq = _expand_kv_heads(v[seq_start:seq_end], num_heads, num_kv_heads).transpose(0, 1)

        scores = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * scale
        causal_mask = torch.triu(
            torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output[seq_start:seq_end] = torch.matmul(probs, v_seq).transpose(0, 1)

    return output


def _paged_attention_decode_fallback(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    del head_dim
    batch_size = query.shape[0]
    output = torch.empty_like(query)

    for batch_idx in range(batch_size):
        context_len = int(context_lens[batch_idx].item())
        logical_blocks = (context_len + block_size - 1) // block_size
        # block_tables 负责把逻辑 block 下标映射到物理 block 下标。
        physical_blocks = block_tables[batch_idx, :logical_blocks]
        physical_blocks = physical_blocks[physical_blocks >= 0]

        # 先把分页存储的 block 收集回来，重新拼成逻辑上的 token 序列，
        # 再去掉最后一个未填满 block 中多余的 padding 位置。
        keys = k_cache[physical_blocks].reshape(-1, num_kv_heads, query.shape[-1])[:context_len]
        values = v_cache[physical_blocks].reshape(-1, num_kv_heads, query.shape[-1])[:context_len]
        keys = _expand_kv_heads(keys, num_heads, num_kv_heads)
        values = _expand_kv_heads(values, num_heads, num_kv_heads)

        # query[batch_idx] 表示这条序列在当前 decode 步的 query。
        scores = torch.einsum("hd,lhd->hl", query[batch_idx], keys) * scale
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output[batch_idx] = torch.einsum("hl,lhd->hd", probs, values)

    return output


if TRITON_AVAILABLE:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        value_ptr,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        slot_idx = tl.load(slot_mapping_ptr + token_idx)

        if slot_idx == -1:
            return

        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size
        head_idx = tl.program_id(1)
        head_offsets = tl.arange(0, head_dim)

        input_offset = (
            token_idx * num_kv_heads * head_dim
            + head_idx * head_dim
            + head_offsets
        )
        cache_offset = (
            block_idx * block_size * num_kv_heads * head_dim
            + block_offset * num_kv_heads * head_dim
            + head_idx * head_dim
            + head_offsets
        )

        key = tl.load(key_ptr + input_offset)
        value = tl.load(value_ptr + input_offset)
        tl.store(k_cache_ptr + cache_offset, key)
        tl.store(v_cache_ptr + cache_offset, value)


    def store_kvcache(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ):
        num_tokens, num_kv_heads, head_dim = key.shape

        # 下面的 Triton kernel 默认输入张量在内存中是紧凑连续排布的。
        if not key.is_contiguous():
            key = key.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()

        assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
        assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

        grid = (num_tokens, num_kv_heads)
        store_kvcache_kernel[grid](
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
        )


    @triton.jit
    def flash_attention_varlen_kernel(
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


    def flash_attention_prefill(
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
            BLOCK_M = 64
            BLOCK_N = 64
        elif head_dim <= 128:
            BLOCK_M = 32
            BLOCK_N = 32
        else:
            BLOCK_M = 16
            BLOCK_N = 16

        num_seqs = cu_seqlens.shape[0] - 1
        cu_seqlens_cpu = cu_seqlens.cpu()
        max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
        # grid 三个维度分别表示：
        #   轴 0：单条序列内部的 query block
        #   轴 1：attention head 下标
        #   轴 2：packed batch 中的序列下标
        grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

        flash_attention_varlen_kernel[grid](
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


    @triton.jit
    def paged_attention_decode_kernel(
        output_ptr,
        query_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        scale: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        context_len = tl.load(context_lens_ptr + batch_idx)

        offs_d = tl.arange(0, head_dim)
        q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
        q = tl.load(query_ptr + q_offset)

        acc = tl.zeros([head_dim], dtype=tl.float32)
        l_i = 0.0
        m_i = -1e10
        max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

        for chunk_idx in range(max_chunks):
            token_start = chunk_idx * BLOCK_N
            if token_start < context_len:
                offs_n = token_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < context_len
                qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10

                for i in range(BLOCK_N):
                    token_idx = token_start + i
                    if token_idx < context_len:
                        block_num = token_idx // block_size
                        block_offset = token_idx % block_size
                        if block_num < max_num_blocks:
                            block_table_offset = batch_idx * max_num_blocks + block_num
                            physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                            if physical_block_idx != -1:
                                k_offset = (
                                    physical_block_idx * block_size * num_kv_heads * head_dim
                                    + block_offset * num_kv_heads * head_dim
                                    + kv_head_idx * head_dim
                                    + offs_d
                                )
                                k_vec = tl.load(k_cache_ptr + k_offset)
                                score = tl.sum(q * k_vec) * scale
                                mask_i = tl.arange(0, BLOCK_N) == i
                                qk = tl.where(mask_i, score, qk)

                qk = tl.where(mask_n, qk, -1e10)
                m_ij = tl.max(qk)
                m_i_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_i_new)
                p = tl.exp(qk - m_i_new)
                acc = acc * alpha
                l_i = l_i * alpha

                for i in range(BLOCK_N):
                    token_idx = token_start + i
                    if token_idx < context_len:
                        block_num = token_idx // block_size
                        block_offset = token_idx % block_size
                        if block_num < max_num_blocks:
                            block_table_offset = batch_idx * max_num_blocks + block_num
                            physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                            if physical_block_idx != -1:
                                v_offset = (
                                    physical_block_idx * block_size * num_kv_heads * head_dim
                                    + block_offset * num_kv_heads * head_dim
                                    + kv_head_idx * head_dim
                                    + offs_d
                                )
                                v_vec = tl.load(v_cache_ptr + v_offset)
                                mask_i = tl.arange(0, BLOCK_N) == i
                                weight = tl.sum(tl.where(mask_i, p, 0.0))
                                acc = acc + weight * v_vec
                                l_i = l_i + weight

                m_i = m_i_new

        output = acc / l_i
        output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
        tl.store(output_ptr + output_offset, output)


    def paged_attention_decode(
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        max_num_blocks = block_tables.shape[1]
        query = query.contiguous()
        output = torch.empty_like(query)
        BLOCK_N = 64 if head_dim <= 128 else 32
        # grid 两个维度分别表示：
        #   轴 0：batch 中的序列下标
        #   轴 1：attention head 下标
        grid = (batch_size, num_heads)

        paged_attention_decode_kernel[grid](
            output,
            query,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            scale=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            BLOCK_N=BLOCK_N,
        )
        return output

else:
    def store_kvcache(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ):
        _store_kvcache_fallback(key, value, k_cache, v_cache, slot_mapping, block_size)


    def flash_attention_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        return _flash_attention_prefill_fallback(
            q,
            k,
            v,
            cu_seqlens,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
        )


    def paged_attention_decode(
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
    ) -> torch.Tensor:
        return _paged_attention_decode_fallback(
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


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q/k/v 都是这一层当前这次 QKV projection 刚计算出来的结果。
        # prefill 通常一次传入很多 token；decode 通常传入每条运行中序列的最新 token。
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 如果已经配置了 KV cache，就先把本轮的 K/V 写入 paged cache，
        # 然后再计算 attention。slot_mapping 给出了每个扁平化 token 的目标 slot。
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            #   这里 k/v 有两种可能形状：
            #   - prefill/batched: (B, N, num_kv_heads, head_dim)
            #   - varlen/flattened: (total_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                batch_size, seq_len, num_kv_heads, head_dim = k.shape
                # store_kvcache 期望输入是扁平 token 列表，这样第 i 个 token
                # 才能和 slot_mapping[i] 一一对应。
                k_to_store = k.reshape(batch_size * seq_len, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(batch_size * seq_len, num_kv_heads, head_dim).contiguous()
            else:
                # 这里本来就是扁平布局；调用 contiguous() 是为了让底层内存排布更规整，
                # 便于 Triton kernel 和 cache scatter 写入。
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            use_triton = TRITON_AVAILABLE and supports_triton(k_to_store.device)
            if use_triton:
                store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)
            else:
                _store_kvcache_fallback(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)
        use_triton = TRITON_AVAILABLE and supports_triton(q.device)

        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")

            # prefill 会对 packed 后的 prompt token 做 attention。
            # cu_seqlens_q 负责指出每条序列在这个大张量中的起止位置。
            if use_triton:
                o = flash_attention_prefill(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    scale,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )
            else:
                o = _flash_attention_prefill_fallback(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    scale,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)

        # decode 路径下，q 是当前步的 query；历史 K/V 则通过
        # block_tables + context_lens 从 paged cache 中查回。
        if use_triton:
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )
        else:
            o = _paged_attention_decode_fallback(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        return o.reshape(o.shape[0], self.num_heads * self.head_dim)
