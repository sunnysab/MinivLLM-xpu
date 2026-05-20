import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 1. x.chunk(2, -1)
              按最后一个维度把输入一分为二。
              如果输入 shape 是 (8, 4000, 8000)，那会变成：
              - x: (8, 4000, 4000)
              - y: (8, 4000, 4000)
            2. F.silu(x)
               对前一半做 SiLU 激活： silu(x) = x * sigmoid(x)
            3. F.silu(x) * y
               再和后一半做逐元素相乘，输出 shape 仍然是 (8, 4000, 4000)。
               这个层本质上是在做一种 gated activation：
               output = silu(gate_part) * value_part
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().xpu()
    # 生成 8 个张量  (8, 4000, 8000)
    # - 8 个样本
    # - 每个样本是一个 4000 x 8000 的矩阵
    input_tensor = torch.randn(8, 400, 8000).xpu()
    
    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.xpu.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.xpu.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
