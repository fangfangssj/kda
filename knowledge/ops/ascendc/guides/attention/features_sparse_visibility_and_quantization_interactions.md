---
type: Guide
title: Attention 特性的可见集合与量化交互
description: 抽取 causal/window/sink、稀疏索引、RoPE 和后量化的组合边界。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/composition.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/extension_points.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/specialization/feature_flags.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/specialization/sparse_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/patterns.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# Attention 特性的可见集合与量化交互

causal与window可以组合，合法KV集合通常按各自规则取交集；sink扩展的集合还需按接口处理因果边界，不能简单说二者互斥或前N项永远无mask。sink可能是额外KV，也可能是分母logit等其他算子语义，不能从同名字段推断。prefix需绑定物理KV布局、分页/长度和全局位置；chunk query偏移按全序列位置，不能把前一query块的输出当当前query的累积状态。

来源将PSE扩写为Partial Sum Extension并解释为部分输出累加，没有算子合同证明。应按真实pseType、realShift/score偏置及scale应用次序解释该参数，不能由缩写发明新的阶段。RoPE对Q/K的位置、配对维和cos/sin取值有明确数学作用；与量化/反量化的顺序会改变数值，也不能仅声称可在L1→L0间插入Vector计算而不证明执行位置支持。

稀疏设计来源明确尚未验证，但可保留可见集合约束：块索引需合法、处理重复/空列表并保证没有遗漏有效token；动态Host索引生成需要数据确实可在Host取得。部分mask块的实际Cube发射FLOP可能大于有效FLOP。gather/pack有效token后仍可能用Cube，不能断言稀疏无法Cube计算。

在线rescale由最大值变化决定，不由已处理块数决定。跳过块应保持已有状态，所有块都跳过时仍要完成空输出/事件；最后有效块不一定是循环末尾。量化数据与scale需按同一绝对KV坐标取值，不能跳过数据却独立推进scale计数。稀疏选择可能打散原32元素scale分组，需要保留原分组映射或重新量化。

后量化应对最终归一化后的O应用合同规定的scale、舍入和饱和，不要求输入也使用MX。amax=max(abs(O))，不能用max(O)替代；全负输出即为反例。amax=0、NaN/Inf及overflow需明确定义，E8M0表达正的幂次scale时不能直接把任意amax/qmax当可精确表示的编码。RINT等转换对异常值的行为受真实API约束，不作统一猜测。

量化并不取消mask/padding和统计FP32预算；1-bit mask只在实现真正采用该编码时才节省存储。特性表中的勾选与“无相互影响”不是已测试的组合能力，本次保存可组合的数学条件及来源未验证限制，不把路线图提升为支持承诺。
