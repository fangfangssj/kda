---
type: Guide
title: 算子族选型的结构与测量条件
description: 把族名和固定阈值变为候选线索，保留每类算子的实际性能约束。
tags:
- task.optimization
- topic.performance.analysis
- topic.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/families/attention_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/families/elementwise_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/families/matmul_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/families/reduction_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/strategy-index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/SCHEMA.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/prompts/ops-partial-prompt.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/prompts/supervisor-prompt.md
  role: supporting
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# 算子族选型的结构与测量条件

选择Ascend C优化首先需要实际dtype、逻辑/物理shape、数据流、核映射和测量范围。族名、源码关键词、统计占比及经验阈值可以召回候选，不能独立证明硬件瓶颈、策略适用或已接近理论最优。缺少profiling字段只表示未知，grep命中循环/BUFFER_NUM不证明存在可重叠流水；失败实现也不独立否定其算法方向。

Elementwise/foreach可从有效字节量、转换链、中间GM读写和每tile调度看起，候选包括批量DMA、UB融合、减少GetValue、Counter模式和循环不变量外提。Add/Mul/ReLU不一律Vector占50–80%，memory-bound也不罕见；占比>50%不直接证明关键路径。Counter降低控制开销，不减少数学必需的Exp等运算；FP16/BF16也不要求所有算子无条件内部FP32。

Reduction/Normalization先区分轴、行长、行数、stride与统计量；小D多行合并、长轴分片或树形归约各有空间和并行代价。D<=128、D>2048等仅是来源候选阈值，outer axis不必然需要跨核。sum/mean同样受浮点顺序影响，Welford是递增/可合并统计法而非固定“两遍算法”；RMSNorm不计算中心化方差，不应凭同族强制换Welford。BlockReduce与WholeReduce替换必须保持真正需要的归约范围。

Matmul先计算M/N输出块数、K贡献、搬运/计算比、L1/L0空间及已有共享。输出块数大于2*AIC、K>=2048、带宽70%或小边<=L1/2均不具备跨shape有效性；矩阵边长与字节容量也不能直接比较。FullLoad、IBShare、K错峰与Split-K分别作用于不同通路，不由名字认定等价。有限支持的AIC/AIV比例必须匹配编译和启动合同。

Attention需保留QK→scale/mask→softmax→PV及完整状态。S2>4K不意味着强制七缓冲/预发射，空/全masked和特殊值仍是数值边界。来源七槽例实际为四QP加三KV，不是Q/K/V/dQ/dK/dV/mask各一。Grad融合来源已论证的D=64不能外推到D=128；输出归一化和全部梯度不能因融合遗漏。

MTE2搬运的典型输入方向是GM到片上存储，不能沿用元数据“Scalar→UB”解释硬件通路。Scalar控制与等待MTE的症状仍需分开。L2策略作用于L2，不可把P52写成scale合并或L1/L0缓存；API的mask模式也不等于Cast舍入。

在相同语义与测量条件下，比较总时间、每shape精度、资源峰值、实际DMA和有效核工作量；仅目标shape通过不能代表其他shape。源66→63.3、37.6→32.7微秒及各族百分比/加速倍数是未在本批复现的样例。Skill的编号、等级、提示词和筛选程序继续留在原库，它们不作为技术关系的事实源。
