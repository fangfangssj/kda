---
type: Guide
title: 舍入链、模式 ABI 与窄化前提
description: 保存多级INT8转换及模式分发，限定NONE和Trait语义。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_03_rounding_mode.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# 舍入链、模式 ABI 与窄化前提

Cast模式必须按源/目标dtype、指令家族和平台选择。CAST_NONE不能统一解释为截断、无舍入或最快：扩宽低精度有限数到FP32可精确，窄化浮点仍受实际接口定义；BF16输出RINT与FP16输出NONE是来源路径实例，非统一最优规则。浮点舍入以目标可表示值中点为界，不是所有接近0.5的数都特殊，也不能因RINT声称恢复BF16丢失信息。

来源FP32→INT32(RINT)→half(deqScale1)→INT8(TRUNC)是受支持指令链候选。在已舍入整数属于INT8目标范围时，half可精确承载这些整数，末级不需再次取整；任意INT32经过half则会舍入/溢出。TRUNC不等于饱和或防溢出，先验证或在宽域按合同clip，特别是[-128,127]或[-127,127]不同区间。SetDeqScale是状态设置，生命周期、后续使用和恢复按实际API约定。

ReinterpretCast取得同buffer的int32/half视图后执行Cast，只有重载允许对应重叠、地址对齐和源数据不被提前覆盖才合法；更多转换并不必然比向量Round更精确。FakeQuant需保存round、zeroPoint、clip、cacheMask、反量化的完整顺序，source所示片段未含完整反量化，不能直接当成最终y。倒数乘法与除法的舍入也未必相同。

Host字符串round/floor/ceil/trunc应明确映射，未知字符串必须拒绝或采用文档默认；源GetRoundMode缺少最终返回路径。MODE_ROUND有时映到CAST_RINT，业务属性名不等于硬件枚举值。sqrt_mode片段实际再次乘scale，即x*scale²，不是开方；保留名称时同时保留公式。Tiling枚举、Kernel模板和注册允许组合须一致。

索引INT64→INT32先验证范围，float除法再FLOOR只有在精度足以区分相邻整数时可作为索引候选；大整数转FP32已失位后向下取整不能恢复。MicroAPI CastTrait中的RegLayout、SatMode、MaskMergeMode、RoundMode分别控制布局/饱和/无效lane/舍入，NO_SAT明确不提供饱和保护，UNKNOWN也不是“精度自动正确”。不同家族Trait顺序及支持需接口确认。

foreach scalar示例castTimes=dataCount/maxCastDataCount只给整块循环，余数必须单独处理；maxCastDataCount需正且FP32各输入/输出slice总容量足够。固定chunk段不能用总dataCount当最后chunk有效count。输入两块、scalar与输出是否可原地重叠须匹配op依赖。

ASDOPS_QUANT_MIN_NEG_127在来源Host读取字符串1时选择-127，否则-128，是该算子的版本化配置，不在本任务修改环境。此值进入tiling/缓存身份且Kernel的Maxs/Mins使用同值，才能保持可复现。来源__CCE_AICORE__<220选NONE、其他RINT是特定dtype路径，不可推成所有新老架构转换支持规则；打包scale的round_mode0/1另是位截断与ties-even协议，不能混用枚举。
