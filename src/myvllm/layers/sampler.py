import torch
import torch.nn as nn


class SamplerLayer(nn.Module):
    """
    自定义采样层，使用 Gumbel-Max Trick 从 logits 中采样下一个 token。
    相比 torch.multinomial，此实现更适配 torch.compile，且完全在 GPU 上完成。
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # 为什么需要 temperature（温度）
        #   温度控制采样随机性：T → 0 退化为贪心（总是选概率最大的），T → ∞ 退化为均匀随机。
        #   这里 logits /= T，值域被缩放：低温让高概率 token 更突出，高温让分布更平坦。

        # 怎么做的？
        #   步骤 1: 温度缩放
        #   unsqueeze(-1) 让 temperature 形状从 (batch,) 变成 (batch, 1) 再广播，
        #   确保 batch 内每个序列使用各自的温度。
        logits /= temperature.unsqueeze(-1)

        # 怎么做的？
        #   步骤 2: softmax 转概率
        #   将缩放后的 logits 转为概率分布 p_i = exp(z_i) / Σ exp(z_j)
        probs = torch.softmax(logits, dim=-1)

        # 怎么做的？怎么算得？
        #   步骤 3: Gumbel-Max Trick 采样
        #
        #   标准 Gumbel-Max: 要按概率 p_i 采样类别 i，等价于：
        #     sample = argmax( log(p_i) + G_i )，其中 G_i ~ Gumbel(0, 1)
        #
        #   等价的指数形式（数值更稳定，省一次 log）：
        #     令 E_i = -log(U_i) ~ Exp(1)，其中 U_i ~ Uniform(0, 1)
        #     则 argmax( p_i / E_i ) 等价于 argmax( log(p_i) + G_i )
        #     （因为 G_i = -log(E_i)，省掉 log 和 -log 的对消）
        #
        #   具体计算：
        #     - torch.empty_like(probs).exponential_(1):  每个位置生成 Exp(λ=1) 随机数
        #     - .clamp_min_(1e-10):                        防止 division by zero（Expo 可能接近 0）
        #     - probs / Expo:                              对每个 token 计算 p_i / E_i
        #     - .argmax(dim=-1):                            取最大值的索引 → 采样结果
        #
        #   为什么不用 torch.multinomial？
        #     - multinomial 内含 CPU-GPU 同步，对 torch.compile 不友好
        #     - Gumbel-Max 全是元素级运算 + argmax，可被 compile 完全融合
        #     - decode 阶段 kernel 计算量小、调度频繁，融合操作收益显著
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
