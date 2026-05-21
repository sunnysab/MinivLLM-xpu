# MinivLLM 面试模拟记录

## 第一组：连续批处理 + PagedAttention 核心

### Q1: 传统静态批处理在大模型推理场景下为什么效率低？你的 Scheduler 是如何解决的？

**我的回答**: 传统静态批处理没有 kv cache 缓存. 这个项目的 scheduler 会缓存用户之前提问的 token 的 kv cache 到一个个 block 中。

**标准答案**: 静态批处理的核心问题是：一个 batch 内所有序列必须等最长的那个生成完毕，整个 batch 才能释放。短序列早已结束却仍占用 GPU 资源（padding/idle），导致 GPU 利用率低。Continuous Batching 的解决方式是：调度器在每一步（iteration）都可以动态移除已完成的序列、插入新的等待序列，从而消除"等最慢"的浪费。具体到 MinivLLM 的 Scheduler：每次 step 调用 schedule()，prefill 阶段从 waiting 队列取新请求加入 running，decode 阶段逐 token 推进 running 中的序列，完成的立即移除并释放资源。

**评分: 2/10** — 混淆了"连续批处理"和"PagedAttention/KV缓存管理"两个概念。KV cache 分块存储是 PagedAttention 解决的问题，不是 continuous batching 解决的问题。Continuous batching 解决的是"batch 内序列生成长度不一致导致 GPU 空闲"的调度问题。

---

### Q2: 为什么选择 "prefill 优先于 decode" 的策略？对延迟和吞吐量的影响？decode 饿死怎么办？

**我的回答**: 不知道

**标准答案**:
- **为什么 prefill 优先**: 新请求越早开始 prefill，用户越早看到第一个 token（降低 TTFT, Time To First Token）。这是面向用户体验的设计。
- **对延迟的影响**: TTFT 低（好），但如果频繁有新请求进来，decode 请求会被反复打断，导致已在生成的请求的 TPOT（Time Per Output Token）升高（坏）。
- **对吞吐量的影响**: prefill 是计算密集（大量矩阵乘），decode 是访存密集（每次只算 1 token）。频繁切换到 prefill 会导致 decode 吞吐下降。
- **饿死问题的解决**: 生产系统（如 vLLM）会设置 `max_num_batched_tokens` 限制 prefill 占用的资源，或引入 chunked prefill（将大 prefill 分块与 decode 交替执行），或设置 decode 优先级提升机制。MinivLLM 简化版没有实现 chunked prefill。

**评分: 0/10**

---

### Q3: decode OOM 时的抢占策略是什么？为什么不 swap 到 CPU？

**我的回答**: 不知道

**标准答案**:
- **MinivLLM 的策略**: deallocate 被抢占序列的所有 block，将其状态从 RUNNING 移回 WAITING 队首。下次被调度时需要重新 prefill（重计算所有 KV）。
- **为什么不 swap 到 CPU**: Swap 需要实现 GPU→CPU 内存拷贝（PCIe 带宽有限）、CPU 端内存管理、以及恢复时 CPU→GPU 拷回。实现复杂度高，且 PCIe 带宽（~32GB/s）远低于 GPU HBM 带宽（~2TB/s），swap 回来的延迟不可忽视。
- **Trade-off**: 重计算策略实现简单、无额外内存开销，但被抢占的序列需要重新 prefill（浪费计算）；Swap 策略保留了已计算的 KV，恢复快，但需要额外 CPU 内存且有 PCIe 传输开销。短序列适合重计算，长序列适合 swap。

**评分: 0/10**

---

### Q4: PagedAttention 的核心思想？传统 KV 缓存为什么低效？BlockManager 如何解决碎片化？

**我的回答**: paged attention 是自己做了一层类似页表的东西，类似操作系统的虚拟内存，避免单次预分配大量内存，使显存中出现大量空缺。

**标准答案**:
- **传统方式的问题**: 为每个序列按 max_seq_len 预分配连续 KV 缓存。实际生成长度远小于 max_seq_len 时，大量预留内存被浪费（内部碎片）；不同序列释放后留下不连续的空洞（外部碎片）；无法在序列间共享公共前缀的 KV。
- **PagedAttention 解决方案**: 将 KV 缓存切分为固定大小的 block（如 16 tokens），通过 block_table（页表）维护逻辑块→物理块的映射。按需分配（序列增长一个 block 才分配一个 block），物理块不需要连续，消除碎片化。通过引用计数实现 block 共享（如 beam search、前缀缓存）。
- **BlockManager 的实现**: 维护 free_block_ids 池 + used_block_ids 集合 + hash_to_block_id 哈希表。allocate() 分配、append() 追加、deallocate() 回收，通过 ref_count 支持安全共享。

**评分: 6/10** — 抓住了核心类比（虚拟内存/页表），但缺少关键细节：内部碎片 vs 外部碎片的区分、按需分配的机制、引用计数共享。表述"大量空缺"不够精确，应明确说"内存碎片化"。

---

### Q5: 前缀缓存为什么用链式哈希？不链式会怎样？

**我的回答**: 如果部分块一致，部分不一致，上下文就对不上了呀。

**标准答案**:
- **为什么链式**: 因为 KV cache 的值依赖于因果注意力（causal attention），即第 N 个 block 的 KV 值不仅取决于该 block 内的 token，还取决于之前所有 token（位置编码 + 注意力计算）。所以两个序列即使第 N 个 block 的 token 相同，如果前面的 token 不同，它们的 KV cache 值也完全不同，不能共享。
- **不链式的后果**: 两个序列 A="你好世界|今天天气" 和 B="再见世界|今天天气"，如果只对每个 block 独立 hash，第二个 block "今天天气" 的 hash 相同，会错误共享 KV cache。但实际上 B 中"今天天气"的 KV 值（受前文"再见世界"影响）与 A 中的完全不同。链式哈希将前一个 block 的 hash 作为 seed，确保只有完全相同前缀的 block 才能匹配。

**评分: 5/10** — 直觉方向正确（"上下文对不上"），但缺少关键解释：WHY 对不上？因为 causal attention 使得 KV 值依赖所有历史 token，而不仅仅是当前 block 的 token。需要能举出具体的错误共享场景。

---

## 第一组总分: 13/50

**总评**: Q1 概念混淆严重，Q2/Q3 完全空白，Q4/Q5 有直觉但缺少工程细节和精确表述。需要重点补强：
1. 区分 Continuous Batching（调度问题）vs PagedAttention（内存管理问题）
2. 理解 prefill vs decode 的计算特征差异（compute-bound vs memory-bound）
3. 抢占策略的 trade-off 分析（重计算 vs swap）
4. 能结合代码讲出具体实现流程
