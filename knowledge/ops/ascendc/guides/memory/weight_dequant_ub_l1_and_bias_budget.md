---
type: Guide
title: 权重反量化的 UB、L1 与 Bias 联合预算
description: 保留双输入输出槽和 metadata 的参数化容量，并在输出 padding 后重新计算。
tags:
- task.design
- topic.memory
- topic.compute.quantization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/prologue-ub-layout-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/vc-sync-and-l1-layout-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/weight-quant-tiling-contract.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/layout-variant-boundary.md
  role: supporting
created_at: '2026-09-11T03:07:47Z'
updated_at: '2026-09-11T03:07:47Z'
platforms: []
---

# 权重反量化的 UB、L1 与 Bias 联合预算

单AIV单槽权重块为Ku*Nu元素，输入每元素w字节、反量化输出d字节，Scale/可选offset在该样例也各为d字节。两槽未加额外padding时，逻辑UB量为2*(w+d)*Ku*Nu+2*d*Nu*(1+hasOffset)。int8/FP8输入w=1、BF16/FP16输出d=2时数据项系数为6，但此式仍未计对齐间隙、guard、额外Cast暂存等实际资源。

若bOut物理外轴增加1，必须替换输出项：来源transB=false候选每槽(Ku+1)*Nu*d，transB=true候选Ku*(Nu+1)*d。统一预算为2*(w*Ku*Nu+bOutPhysicalBytes+scalePhysicalBytes+offsetPhysicalBytes)+其他资源；不能仍用未padding的6*Ku*Nu。真实分形补齐可能使physicalBytes更大，应按store/copy跨度计算。

bIn从GM搬入到VF读完、bOut从VF写完到UB→L1读完分别占槽。Scale/offset按每个N通道窗口加载，跨K复用要求通道坐标和槽内版本不变；改变N tile、group或metadata时不能继续用首轮缓存。hasOffset=false可移除对应区域，但后续起点也需同步更新。

来源L1按两half各放B_dequant与A，尾部可放64B对齐Bias。预算应含两份实际A/B物理块、对齐和Bias，不把单half量当总量。Bias的GM指针、L1/BT Tensor类型、Block BiasType和Host biasElemBytes须一致；不能用输出CType代替BiasType。改变Bias类型后重新计算Align(baseN*sizeof(BiasType),64)及所有偏移。

L0C量依实际累加类型和buffer数，FP32例为baseM*baseN*4*DB。l0cDB字段与Block分配/事件模式必须一致，不能一边单槽一边按双槽回收。kL1、baseK、Ku/Nu、边块拆分字段都需按消费者检查单位与合法域。来源transB=true要求N整16且不拆N尾轮、false要求K整16，是所述Prologue分形/切分实现的限制，不证明所有反量化Matmul都要求相同对齐或M任意。

验证容量边界、hasBias/hasOffset组合、不同输入/输出宽度、K-L1多轮、N窗口变化和padding尾部。来源模板缺少某路线或固定SWAT不代表硬件永远不支持StreamK/常驻；任何替代都需独立预算和同步。本批未上板。
