from myvllm.layers import *
from myvllm.utils.device import get_tp_world_size
import torch 
import torch.nn as nn

# 这个文件实现的是一个简化版的 Qwen3 Causal LM 结构。
# 可以按下面的层级去理解：
#   Qwen3ForCausalLM
#     -> Qwen3Model
#       -> 多层 Qwen3DecoderLayer
#         -> Qwen3Attention + Qwen3MLP
#
# 其中最重要的主干路径是：
#   input_ids
#     -> embedding
#     -> decoder layers
#     -> final norm
#     -> lm_head（compute_logits 时调用）
class Qwen3Attention(nn.Module):
    # Qwen3Attention 负责这一层的注意力子模块：
    #   1. 线性投影得到 q/k/v
    #   2. 可选地对 q/k 做归一化
    #   3. 对 q/k 应用 RoPE
    #   4. 调用底层 Attention 完成 prefill / decode
    #   5. 通过输出投影回 hidden_size
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = get_tp_world_size()

        # total_num_heads 表示模型全局 head 数；
        # self.num_heads 表示当前 TP rank 上实际持有的 head 数。
        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        # Qwen3 可以使用 GQA/MQA，因此 KV head 数可以小于 Q head 数。
        # 同样地，self.num_kv_heads 表示当前 TP rank 上持有的 KV head 数。
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        # 每个 head 的维度。通常 hidden_size = num_heads * head_dim。
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale

        # 一次线性层同时生成 Q/K/V。
        # 输出总维度 = head_dim * (num_heads + num_kv_heads + num_kv_heads)。
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 当前 TP rank 上 Q 和单份 K/V 的扁平维度。
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # Qwen3 的注意力里会对 Q/K 做 norm 来稳定注意力分数。
        # 这里的 LayerNorm 实际上是对最后一个维度（head_dim）做归一化。
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        # 旋转位置编码只作用在 Q/K 上，不作用在 V 上。
        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        # 这里的 Attention 是底层推理态 attention：
        # 它会根据 context 自动选择 prefill 或 decode，
        # 并在 decode 时读写 paged KV cache。
        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        # 输出投影把 attention 输出映射回 hidden_size。
        # RowParallelLinear 会在张量并行场景下完成必要的聚合。
        self.o_proj = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # x 表示当前层输入的 hidden states。
        #
        # 常见两种形状：
        #   1. batched prefill: (B, N, hidden_size)
        #   2. packed varlen / decode: (total_tokens_or_batch, hidden_size)
        #
        # 这里不直接区分 prefill / decode，而是统一先做 QKV 投影。
        # 最后一维度匹配隐藏层维度即可，不受形状影响

        # QKV 投影后，最后一个维度里顺序排着 [Q | K | V]。
        # 当前 TP rank 上的输出维度是：
        #   head_dim * (self.num_heads + 2 * self.num_kv_heads)
        qkv = self.qkv_projection(x)

        # 沿最后一维切成 q/k/v 三块。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 把最后一维的扁平 head 表示还原成显式 head 维：
        #   q: (..., num_heads, head_dim)
        #   k/v: (..., num_kv_heads, head_dim)
        #
        # 2 维输入通常出现在 packed varlen 或 decode 场景；
        # 3 维输入通常出现在普通 batched prefill 场景。
        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # 如果 qkv projection 没有 bias，则这里对 q/k 做归一化。
        # q 和 k 会参与 attention score 的点积，因此它们的数值稳定性更关键。
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 应用旋转位置编码。positions 的构造由上层 decoder layer 决定。
        q, k = self.rotary_emb(positions, q, k) 

        # 底层 Attention 会：
        #   - prefill: 在当前 prompt token 上做 attention
        #   - decode: 读取历史 KV cache，再与当前 q 做 attention
        o = self.attention(q, k, v)

        # 输出投影把多头输出重新合并回 hidden_size。
        o = self.o_proj(o)

        return o

class Qwen3MLP(nn.Module):
    # Qwen3 的前馈层采用门控 MLP：
    #   hidden
    #     -> gate_up（一次线性层同时得到两份中间表示）
    #     -> SiluAndMul（SiLU(gate) * value）
    #     -> down_proj
    #     -> hidden
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up(x) 会在最后一维产出两份 intermediate_size，
        # SiluAndMul 会把它切成两半后做 silu(x1) * x2，
        # 最后 down_proj 再映射回 hidden_size。
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x


class Qwen3DecoderLayer(nn.Module):
    # 一个 decoder layer 的结构大致是：
    #   residual + input_layernorm
    #   -> self attention
    #   -> residual + post_attention_layernorm
    #   -> mlp
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        block_size: int = 256,
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
        )
        self.post_attention_layernorm = LayerNorm(gamma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 这里使用了带 residual 的归一化写法：
        #   - 第一层进入时 residual=None，此时 residual 直接取原始输入 x
        #   - 后续层中 residual 会沿着网络一路传递，减少重复加法/拷贝
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            # 第一层还没有累计 residual，先把原始输入保存下来。
            residual = x
            x = self.input_layernorm(x)

        # positions 决定 RoPE 的位置索引。
        # 这里不是简单无脑用 arange，而是要根据当前推理上下文区分：
        #   1. batched prefill：每条序列的位置都应从 0 重新开始
        #   2. 单条 prefill：位置就是 [0, 1, 2, ...]
        #   3. decode：当前位置就是 context_len - 1
        from myvllm.utils import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # batched prefill 下，x 往往是 packed 之后的大张量。
            # 例如 cu_seqlens_q = [0, 3, 5] 表示：
            #   序列 0 的位置应为 [0, 1, 2]
            #   序列 1 的位置应为 [0, 1]
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            # 单条序列 prefill，直接顺序编号即可。
            positions = torch.arange(x.size(0), device=x.device)
        else:
            # decode 每条序列本轮只处理一个“最新 token”。
            # 如果当前长度是 L，那么这个 token 的位置下标就是 L - 1。
            positions = context.context_lens - 1

        x = self.self_attn(x, positions=positions)
        # attention 之后继续走残差归一化，再进入 MLP。
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual

class Qwen3Model(nn.Module):
    # Qwen3Model 是不带 lm_head 的主干网络：
    #   token embedding -> decoder layers -> final norm
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        block_size: int = 256,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim = hidden_size
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids 先映射成 hidden states。
        x = self.embed_tokens(input_ids)
        residual = None

        # 每层都会返回新的 x 和更新后的 residual。
        for layer in self.layers:
            x, residual = layer(x, residual)

        # 最后一层 norm 也会把 residual 合并进去。
        x, _ = self.norm(x, residual)
        return x



class Qwen3ForCausalLM(nn.Module):
    # Qwen3ForCausalLM = 主干模型 + 语言模型输出头。
    # forward 这里只返回 hidden states；
    # 真正投到词表 logits 是在 compute_logits 里做。
    packed_module_mapping = {
        "q_proj": ('q_proj', 'q'),
        "k_proj": ('k_proj', 'k'),
        "v_proj": ('v_proj', 'v'),
        "gate_up": ('gate_up_proj', '0'),
        "gate_down": ('gate_down_proj', '1'),
    }
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 这里只跑主干，方便上层根据 prefill/decode 自己控制何时算 logits。
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states -> vocab logits
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    # 简单的本地 smoke test。
    from myvllm.utils import set_context

    device = "xpu"
    model = Qwen3ForCausalLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    ).to(device)

    # Attention 的 prefill 路径要求 packed 输入和 cu_seqlens_q。
    input_ids = torch.randint(0, 50257, (16,), device=device)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_k=8,
    )
    output = model(input_ids)
    print(output.shape, output.device)
