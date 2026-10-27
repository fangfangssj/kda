---
type: Guide
title: 量化 Bias 单位、范围与舍入声明
description: 保留动态INT4、FakeQuant和Gemma模式，纠正不可比基线。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_08_quant_precision.md
  role: primary
created_at: '2026-09-11T05:41:33Z'
updated_at: '2026-09-11T05:41:33Z'
platforms: []
---

# 量化 Bias 单位、范围与舍入声明

INT32 bias若以量化累加单位给出，反量化为(x+bias)*scale；浮点bias若已在实数单位，则为x*scale+bias。两种顺序不同源于单位，不能仅以一个更精确为由互换。INT32相加须防溢出，转FP32也只对可精确表示整数无损；超过2^24并非所有整数仍精确。ReinterpretCast只改视图，真正Cast才数值转换，原地重叠需API允许；INT32先转half再升FP32不会恢复已丢位。

动态有符号INT4候选s=max((max-min)/15,1e-12)，z=7-max/s，q=round(x/s+z)，反量化(q-z)*s。未触发floor且有限范围时max映7、min映-8；常量、极小range、max-min溢出和NaN/Inf需明确处理，floor触发后端点不再同时恰好满量程。SafeDiv在|b|<eps时返回a不等于除法，只有有限s≥eps且分支严格小于eps才通常不可达。

FakeQuant在有限qmin≤qmax时，先Min后Max与先Max后Min得到相同夹值，不能宣称换顺序必提高精度；若先窄化Cast已溢出，随后clamp不能修复。反向cache mask应由规定的舍入/饱和前域及边界包含规则生成，不能因为仅展示前向clip就遗漏梯度语义。

来源将RINT和ROUND的nearest-even/中点规则写反于现有接口材料，不能据“量化标准”标签直接选枚举。应按实际Cast类型组合核验最近偶数与中点远离零规则，正负2.5/3.5均需验证；某方向不支持的枚举也不能沿用到另一方向。量化误差、rounding和饱和各自独立。

Gemma RMS Norm的gamma+1应按合同在FP32中加1后作用于归一化值，普通RMS Norm未必有此项。gemmaMode模板可消除运行时分支，收益仍取决于代码大小和实际编译/调度。对照片段若公式不同，不能用其时间宣称优化收益。
