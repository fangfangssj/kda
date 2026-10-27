---
type: Guide
title: 批处理与缓存复用优化的语义前提
description: 修正逐行 Sub 展平、16KiB 下限和双缓冲检查的过度推论。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-perf.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-api.md
  role: supporting
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# 批处理与缓存复用优化的语义前提

批量搬运/计算可摊薄循环与指令开销，但合并前需保持每行的广播、stride、padding、mask 和独立状态。来源逐行 Sub(x[r*pitch],tmp,cols) 的 tmp 是每行重复使用的向量，直接 Sub(x,tmp,totalSize) 会将 tmp 当成全长连续输入；若未显式广播或构造等价 repeat/stride，该改法可能越界或改变结果。不是所有循环内 API 都应删除。

GM 缓存只有在重用数据不变、UB 生命周期和容量允许、同步代价合理时才有收益；多轮读取可能是正确观察更新或控制资源的必要行为。跨核写者仍需可见性与复用协议。逐元素 GM 访问在大吞吐路径通常成本高，但不能据此禁止所有标量元数据访问或抹掉 SuperKernel 的合法访问接口。

双缓冲需实际 tile 循环能重叠生产/消费、不同槽位不会提前覆盖、搬运与计算资源可并行，且所有缓冲份数计入预算。BUFFER_NUM>=2 或输入>8K 元素都不足以证明优化生效；双缓冲压缩 tile 后也可能变慢。Reduce scratch 应来自真实 API/dtype/shape 需求，不能固定套 32*sizeof(float)。

单次 16KiB 是来源启发建议，不是合法搬运下限。示例 availableUB/3 与元素总量混用了单位，又以 max 提高到16KiB，可能超过可用 UB 或剩余输入。应先将容量换算为元素、计入所有存活缓冲，再选不超过合法容量的候选 tile；尾块仍只搬真实有效长度。大 tile 也可能降低并行度或增加延迟，来源“下降30%”无本批测量支持。

多核均分示例 q=N/C、r=N%C、len_b=q+[b<r]、offset_b=b*q+min(b,r) 在 C>0、0<=b<C 且算术不溢出时覆盖互不重叠。工作量不等于元素数时需按实际成本评估；空核可跳过计算但不能丢弃全核协议角色。各级 UB/L1/L0 分项核算，来源256KiB/1MiB常量是特定上下文，不替代平台资源。

中间 FP32 累加可减少 FP16 连续舍入，仍要匹配最终舍入、特殊值和验收合同，并计入增大的存储/指令成本。不能将提升精度直接等同性能收益或所有算子必须采用的公式。
