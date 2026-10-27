---
type: Guide
title: 量化舍入、饱和边界与 Bias 数值域
description: 明确整数/低比特量化的完整数值链，以及整数 Bias 与实数 Bias 的不同加入位置。
tags:
- task.design
- topic.precision
- topic.compute.elementwise.type_conversion
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A3_rounding_mode.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A8_quant_precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D4_fp8_int4_conversion.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A3_rounding_mode.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A8_quant_precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D4_fp8_int4_conversion.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A3.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A8.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D4.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# 量化舍入、饱和边界与 Bias 数值域

仿射整数量化常写为 `q = clamp(round(x/scale + zero_point), qmin, qmax)`，反量化为 `(q-zero_point)*scale`。必须明确 scale/zero_point 的逐 Tensor、通道或分组范围，正有限 scale 的要求，以及不合法 scale、NaN/Inf 和零块的处理。FakeQuant 还要把量化结果反量化回规定输出类型，不能仅把输入取整就视为模拟量化。

舍入中点是合同的一部分。到最近偶数、通常所说四舍五入、向下、向上和向零不是同一种规则；负数中点同样要测。来源不同文档对 CAST_RINT/CAST_ROUND 的解释互相矛盾，不能据其“标准量化/QAT/对称/非对称”标签机械选模式。应按具体 API、类型组合和 oracle 核对实际枚举：固定实现中 RINT 用于对齐 nearest-even 的案例不等于所有转换方向都支持它。float→float 同类型 Cast 也不能未经接口确认当作取整指令。

对普通有限数且 qmin<=qmax，`min(max(v,qmin),qmax)` 与 `max(min(v,qmax),qmin)` 等价；来源“Maxs/Mins 绝对不能交换”不成立。NaN 传播、饱和和具体接口的标量类型需另核对。应在有能力表示中间量的域内处理范围，避免先窄化溢出再试图 clamp 修复。

典型完整整数表示范围为 INT8 [-128,127]、UINT8 [0,255]、INT4 [-8,7]、UINT4 [0,15]，但某些量化合同会选择更窄的有效区间。4-bit 存储还包含打包顺序、两个逻辑元素共享字节和奇数尾元素，不能以一元素一字节的寻址直接外推。

FP8 是浮点编码，不能与 INT4 的整数范围/取整流程混为一谈。来源所列 E4M3FN 最大有限值 448、E5M2 最大有限值 57344 对应特定格式；HIFLOAT8 另有编码规则。缩放约定必须区分 `x/amax` 与为了使用目标有限范围而定义的 `x/scale`，不得混用 amax 和 scale。E4M3/E5M2 常见用途只是选择背景，不能推出禁止用于其他 Tensor；“所有 FP8 标准强制某枚举”也不能由该材料确立。

反量化 Bias 需要先明确单位。若 INT32 Bias 已按整数累计域定义，可在该域先加 Bias 再转换/缩放；实数 Bias 则通常在反量化到实数域后相加。这是两套不同语义，不能仅按类型替换位置。先加的整数运算要检查溢出；INT32 转 FP32 对超过精确表示范围的值会舍入。来源通过 ReinterpretCast 获得同地址 FP32 视图后执行 Cast 的形式，还须确认该重载支持原地重叠，不能当成通用实现。

对参数两端、半整数及负半整数、极大值、零 scale、各通道/组 Bias、INT32 精确表示边界和低比特尾部逐项比较。性能收益、固定指令条数和统一“输入/输出必用哪种 RoundMode”没有在这些材料中获得跨平台证据。
