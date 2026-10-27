---
type: Guide
title: 对齐代价表与 CV 分块候选
description: 保存四类尾块代价和 Decode/GQA 分块启发式的具体形式，区分候选模型与实测结论。
tags:
- task.design
- topic.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P16_cost_driven_core_partition.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P17_cv_parallel_block_sizing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P16.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P17.yaml
  role: supporting
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# 对齐代价表与 CV 分块候选

对齐代价模型可按基本块的实际指令粒度估计开销。来源示例是 `cost(M,S2)=6*ceil(M/16)+10*ceil(S2/64)`；16/64 是两个轴的对齐粒度，6/10 是待校准权重。它是可比较的线性代理，既不是矩阵乘的 FLOP 公式，也不是时间单位。加法、乘法应在足够宽的类型中进行，不能先用 uint32 算完再转 int64 以为避免了溢出。

二维代价表分别包含主×主、尾×主、主×尾、尾×尾。某尾长为零时，对应不存在的块可记零；稀疏或 Mask 块只有在该算子确实不执行其工作时才可记零。如果仍做初始化、同步或屏蔽计算，要保留这些成本。代价感知分配可以按 Batch→Row→Block 分层，允许一定代价偏差以限制碎片化；代价零不允许遗漏实际有效行，端点处理见[前缀切分](cost_prefix_partition_boundaries.md)。FlashDecode 的跨核 partial 合并需另计工作量与调度。

来源的 CV 分块示例对 TND/NTD 取 mBase=512；其他布局在 s1<=16 时，按对齐 sInner 的 <=512/1024/2048/4096/更大分档取 mBase=512/256/128/64/32，较长 s1 仍取 512。该形式旨在让 mBase*sInner 受控，阈值和布局分支均是来源案例，不能据此把所有短 s1 认定为 Decode，或保证所有平台该积都能装入 workspace。

另一个 GQA 候选按 `idx=min(gSize/5,2)` 选 sInner=[8192,4096,2048]，即整数 group 0–4/5–9/>=10 三档；group=0 是否是合法输入要由算子定义决定。分档值只提供搜索起点，不构成普遍的性能规律。实际容量要乘 dtype 字节数、阶段数和物理槽数，再计其他状态，不能只限制元素乘积。

增大 M 或 S2 会同时改变 Cube、Vector、搬运、尾块与状态容量；不能仅凭轴名称断言必然是谁在等待。保留静态特化也是合法方案。候选应以真实 shape/layout/dtype、逐阶段耗时及完整时延校准，同时计入 Host 枚举、元数据获取和计算代价表的时间。验证对齐边界两侧、双尾、全屏蔽块、不同 GQA 组和短序列。
