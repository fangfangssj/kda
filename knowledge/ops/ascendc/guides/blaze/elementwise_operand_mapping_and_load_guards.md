---
type: Guide
title: Blaze 逐元素 Epilogue 的广播映射与加载边界
description: 补全 MemBase/RegBase 的 operand 映射、独立 slot 预算与 mask 前整向量读取。
tags:
- task.design
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/elementwise-broadcast-epilogue-fusion-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/elementwise-broadcast-epilogue-fusion-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/block-l0c2ub-extension.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/fused-kernel-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/epilogue-membase-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/epilogue-regbase-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/precision-diagnosis.md
  role: supporting
created_at: '2026-09-11T03:07:47Z'
updated_at: '2026-09-11T03:07:47Z'
platforms: []
---

# Blaze 逐元素 Epilogue 的广播映射与加载边界

Matmul最终T[m,n]之后的逐元素/broadcast DAG，应使每个Y[m,n]仅依赖同一位置T及明确映射的operand。常见scalar索引为0、row输入取m、column取n、同形输入取(m,n)，实际物理地址再由各自stride与dtype得到。quant/dequant若scale已给定且仅逐元素应用可满足此条件；现场求行scale、Softmax、LayerNorm、scan、gather或邻域运算则增加跨元素依赖，需要不同状态/调度。

L0C有效范围、实际Copy extent、目的UB pitch与最终GM有效列是不同量。每个slot分别预算C、每路operand staging、DAG中间值、输出和必要guard；不能给每slot都分“全部UB”。GetTensor(slot)返回字节偏移、元素偏移还是Tensor视图须按接口处理一次。SplitM已把本地C放slot起点时不再叠sub偏移，GM全量operand/输出则仍要正确全局行列。

MemBase通常以LocalTensor表达中间运算，RegBase可在VF内保留寄存器中间值。操作数量不能独自决定优劣：API可用性、broadcast实现、实际load/store、寄存器活跃量、UB staging、事件与数值顺序都影响选择。较长链可能减少UB往返，也可能增加寄存器压力或不支持的调用。

active mask只约束具体API允许的运算/存储行为。如果某load在mask应用之前读取完整向量，即使只计算尾部几个lane，也必须有合法地址和完整物理容量，必要时增加按真实读取跨度计算的guard。guard保护访问范围，不改变逻辑N，也不能让padding参与结果。各operand的窄dtype/不同pitch单独计算，不能套C的容量。

输入ready、输出store完成及下一轮覆盖是不同依赖；cross-core wait要覆盖Epilogue真正首消费者，然后按需要连接本地MTE/V。全常量输入可能掩盖行列地址错，用行编码、列编码、one-hot及非零known-C检查非零offset、tail、slot复用和空sub。tanh/GELU的指数写法在大值可能先Inf再Inf/Inf，需按原合同选择稳定实现并验证中间值，不能仅凭最终absmax忽略NaN。

来源五模式和单变量对照可支持定位，但固定拒绝所有非有限值、强制某条工程路线、固定验证次数不是所有算子的数学合同。本文不移入PLAN/资产操作规则；设备收益与支持范围仍需实际实例验证。
