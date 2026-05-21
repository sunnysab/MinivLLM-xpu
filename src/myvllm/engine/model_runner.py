import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # 分布式推理相关配置
        # block_size: PagedAttention 中每个 KV cache 块容纳的 token 数
        self.block_size = config['block_size']
        # world_size: 张量并行的 GPU 数量
        self.world_size = config['world_size']
        # enforce_eager: 强制使用 eager 模式（不使用 CUDA Graph）
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        self.device = resolve_device(config.get('device'), rank)
        self.device_type = self.device.type
        self.dist_backend = get_distributed_backend(
            self.device,
            self.world_size,
            config.get('dist_backend'),
        )
        # 仅在设备支持且未强制 eager 时才启用 CUDA Graph 加速
        self.use_cuda_graphs = supports_cuda_graphs(self.device) and not self.enforce_eager

        if self.device.type != "cpu":
            set_device(self.device)

        # 多卡场景下初始化进程组，使用 TCP 方式进行 rendezvous
        if self.world_size > 1:
            dist.init_process_group(
                self.dist_backend,
                "tcp://localhost:12345",
                world_size=config['world_size'],
                rank=rank,
            )

        # 根据模型名称实例化对应的模型结构
        path_str = self.config['model_name_or_path']
        model_name = self.config.get('model_type') or Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B' | 'qwen3-0.6b':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct' | 'llama-3.2-1b-instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # 先将模型移到 GPU，再加载权重
        # 这样做的好处是权重可以直接从磁盘加载到 GPU 内存，避免 CPU->GPU 的额外拷贝
        self.model = self.model.to(self.device)

        # 如果指定了预训练权重路径，则加载 checkpoint 文件中的权重
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # 另一种策略（已弃用）：先在 CPU 加载权重再移到 GPU，适用于 GPU 内存紧张的情况
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # 保存默认数据类型，在 allocate_kv_cache 中计算每个 block 字节数时需要用到
        self.default_dtype = torch.get_default_dtype()

        # 调试标志：标记是否为首次 decode 步骤
        self._first_decode = False

        # 启动流程：warmup -> 分配 KV cache -> 捕获 CUDA Graph
        # warmup：用最大 batch 跑一次前向，触发所有 CUDA kernel 的 JIT 编译和内存分配
        self.warmup_model()
        # 根据剩余显存计算并分配全局 KV cache 池
        self.allocate_kv_cache()
        # 捕获 CUDA Graph：把 decode 阶段的 kernel 调用序列录制下来，后续 replay 可减少 CPU 开销
        if self.use_cuda_graphs:
            self.capture_cudagraph()

        torch.set_default_device(str(self.device))
        torch.set_default_dtype(self.default_dtype)

        # 重要：在所有模型初始化完成后才设置共享内存和同步屏障
        # 这确保所有 rank 在进入事件循环前都已完成 warmup 和 KV cache 分配
        if self.world_size > 1:
            # 同步所有 rank，确保都完成了初始化
            dist.barrier()
            if self.rank == 0:
                # rank 0 负责创建共享内存，先尝试清理可能残留的旧共享内存
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # barrier 确保 rank 1 等到共享内存创建完毕后再连接
                dist.barrier()
            else:
                # 非 rank 0 的 worker 等待 rank 0 创建共享内存后再连接
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')

    # 从共享内存中读取方法名和参数（仅非 rank 0 的 worker 调用）
    # 通信协议：前 4 字节为 payload 长度（小端序），之后是 pickle 序列化的 (method_name, *args)
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        # 阻塞等待 rank 0 通过 event 通知有新任务
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little')
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # 将方法名和参数写入共享内存（仅 rank 0 调用）
    # 写入完成后通过 event.set() 通知所有 worker 有新任务
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 清理资源：关闭共享内存、销毁进程组、释放 CUDA Graph
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if self.use_cuda_graphs:
            del self.graphs
            del self.graph_vars
        synchronize(self.device)
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # worker 的主循环（仅非 rank 0 调用）
    # 不断从共享内存读取任务并执行，收到 'exit' 信号时退出
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == 'exit':
                self.exit()
                break

    # 统一的方法调度接口，rank 0 和 worker 都通过此方法执行具体操作
    # rank 0 调用时会先将指令写入共享内存通知 worker，然后自己也执行
    # worker 调用时直接执行（已经从共享内存读到了指令）
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # 模型预热（warmup）
    # 目的：用最大可能的输入跑一次前向，让 PyTorch/CUDA 完成所有延迟初始化
    # 包括：CUDA kernel JIT 编译、cuDNN benchmark、内存分配器预分配等
    # warmup 后可以准确测量峰值内存，从而计算剩余显存能分配多少 KV cache block
    def warmup_model(self):
        empty_cache(self.device)
        reset_peak_memory_stats(self.device)
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        empty_cache(self.device)

    # 分配全局 KV cache 池
    # PagedAttention 的核心思想：不为每个序列单独分配 KV cache，
    # 而是维护一个全局的 KV cache 池，按 block 为单位动态分配给各个序列
    # 这样可以避免显存碎片，提高 GPU 利用率
    # 这个函数调用前已经跑了一次 warmup，即，ffn 的权重已经加载
    def allocate_kv_cache(self):
        # 计算当前可用显存
        free_mem, total_mem = mem_get_info(self.device)
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        stats = memory_stats(self.device)
        peak_mem_usage = stats['allocated_bytes.all.peak']
        current_mem_usage = stats['allocated_bytes.all.current']
        # 预留峰值内存（模型前向传播时的临时张量），剩余的才能分配给 KV cache
        # peak_mem_usage - current_mem_usage = 前向传播中临时激活/中间张量的峰值额外开销
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # 计算单个 KV cache block 的字节数
        # 每个 block 形状: [block_size, num_kv_heads, head_dim]
        # 乘以 2 是因为 K 和 V 各需要一份
        # 乘以 num_layers 是因为每一层都有独立的 KV cache
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # block_bytes = block_size × 2(K+V) × num_layers × num_kv_heads × head_dim × dtype字节数
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # 在所有 rank 之间同步 KV cache block 数量，取最小值
        # 原因：各 rank 的可用显存可能不同（rank 0 通常因为 NCCL buffer 等开销而显存更少）
        # Scheduler 只在 rank 0 运行，如果用 rank 0 本地的值可能导致其他 rank OOM
        # 因此取所有 rank 中最保守（最小）的值，确保所有 rank 都不会 OOM
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=self.device
            )
            # all_reduce + MIN：所有 rank 取最小值，确保最紧张的那块 GPU 也不会 OOM
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # 单 GPU 场景直接使用本地值
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # 一次性分配全局 KV cache 池
        # 形状: [2(K/V), num_layers, max_cached_blocks, block_size, num_kv_heads, head_dim]
        # 用 zeros 而非 empty，避免未初始化的垃圾值影响注意力计算
        allocated_kv_cache = torch.zeros(
            2,
            self.config['num_layers'],
            self.config['max_cached_blocks'],
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
        )
        # 将全局 KV cache 池的各层切片绑定到对应的 Attention 模块上
        # 这样模型前向时直接写入/读取全局池，无需额外拷贝
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # 为 prefill 阶段准备输入数据
    # Prefill 的特点：一次性处理序列的所有 token（或前缀缓存后的剩余 token）
    # 需要准备：
    #   - input_ids: 所有序列拼接后的 token id（1D 张量，用于 flash_attn_varlen）
    #   - slot_mapping: 告诉模型 KV cache 写入全局池的哪些位置
    #   - cu_seqlens_q/k: 累积序列长度，标记每个序列在拼接张量中的边界
    #     例如 cu_seqlens_q = [0, 3, 5, 9] 表示 3 个序列，长度分别为 3, 2, 4
    #   - block_tables: 每个序列的 block 映射表（仅在有前缀缓存时需要）
    #
    # 前缀缓存（prefix cache）：如果序列有部分 token 的 KV 已被缓存，
    # 则只需处理未缓存的部分（query），但 attention 仍需访问全部已缓存的 KV（key）
    # 所以 cu_seqlens_q 可能 < cu_seqlens_k
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        slot_mappings = []
        seqlens_q = []
        seqlens_k = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            # 只取未被前缀缓存覆盖的 token 作为输入
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # 遍历未缓存的 block，计算每个 token 在全局 KV cache 池中的写入位置
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        # 非最后一个 block：填满整个 block
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        # 最后一个 block：只填实际有 token 的部分
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # 存在前缀缓存时，需要 block_tables 来告诉 attention 去哪里读取已缓存的 KV
            # 对 block_table 进行 padding 使所有序列的 block 数一致，方便构造张量
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        # pin_memory + non_blocking: 异步传输到 GPU，与后续计算重叠
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True) if block_tables else None,
        )
        return input_ids


    # 为 decode 阶段准备输入数据
    # Decode 的特点：每个序列每次只生成一个 token（自回归生成）
    # 需要准备：
    #   - input_ids: 每个序列最新生成的一个 token
    #   - slot_mapping: 新 token 的 KV 写入位置
    #   - context_lens: 每个序列目前已有的总 token 数（attention 需要知道看多远）
    #   - block_tables: 每个序列的 block 映射表（读取历史 KV）
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            # 新 token 写入最后一个 block 的对应位置
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        # padding block_tables 使所有序列的 block 数一致
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True) if block_tables else None,
        )
        return input_ids    

    # 准备采样参数（温度）
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)

    # 执行模型前向传播
    # Prefill 阶段：直接运行模型前向（因为序列长度不固定，无法用 CUDA Graph）
    # Decode 阶段：如果启用了 CUDA Graph，则使用 graph.replay() 执行
    #   CUDA Graph 的优势：把多次 kernel launch 的 CPU 开销合并为一次 replay 调用
    #   对 decode 特别有效，因为 decode 每次只处理 1 个 token，kernel 计算量小，CPU 开销占比大
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or not self.use_cuda_graphs:
            # Prefill 使用 flash_attn_varlen_func，输入是 1D 拼接张量 + cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # 找到不小于当前 batch size 的最小已捕获 graph
            # 例如 bs=5 会使用 bs=8 的 graph（多出的位置用 padding 填充）
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # 将实际输入数据拷贝到 graph 捕获时使用的固定缓冲区中
            # CUDA Graph 要求输入/输出地址不变，所以必须用 copy_ 而非赋值
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放已捕获的 CUDA Graph，执行完整的 decode 前向
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # 完整的单步推理流程：准备输入 -> 模型前向 -> 采样 -> 清理上下文
    # 返回每个序列采样得到的下一个 token id
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # 只在 rank 0 上做采样，其他 rank 不需要（结果会由 rank 0 分发）
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    # 捕获 CUDA Graph
    # 原理：预先在 GPU 上"录制"一组固定的 kernel 调用序列，之后通过 replay() 重放
    # 好处：消除 decode 阶段大量小 kernel 的 CPU launch 开销（decode 时每次只处理 1 token，
    #       kernel 执行时间很短，CPU 调度开销反而成为瓶颈）
    #
    # 实现细节：
    #   1. 按最大可能的 batch size 预分配所有输入/输出缓冲区（固定地址）
    #   2. 为多种常见 batch size 各捕获一个 graph：[1, 2, 4, 8, 16, 32, ...]
    #      运行时选择 >= 当前 bs 的最小 graph 来 replay
    #   3. 所有 graph 共享同一个 memory pool（graph_pool），避免重复分配
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config.get('max_num_seqs')
        if max_bs is None:
            max_bs = self.config.get('max_num_sequences', self.config['max_num_batch_tokens'])
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # decode 阶段每个序列只输入 1 个 token，所以 input_ids 形状为 (max_bs,)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=self.device)
        # slot_mapping: 新生成 token 的 KV 写入位置
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=self.device)
        # context_lens: 每个序列的上下文长度（历史 token 总数）
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=self.device)
        # block_tables: 每个序列在全局 KV cache 池中的 block 映射
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=self.device)
        # outputs: 模型输出的 hidden_states
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=self.device)

        # 为多种 batch size 捕获 graph，运行时选择最接近的
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        # 从大到小捕获，共享 memory pool
        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            # 先做一次 eager 运行（warmup），确保所有 lazy 初始化完成
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            # 正式捕获 CUDA Graph
            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            self.graphs[batch_size] = graph

            # 确保当前 graph 捕获完成后再进行下一个
            synchronize(self.device)
            reset_context()

        # 保存所有 graph 共享的缓冲区引用，供 run_model() 中 copy_ 使用
        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
