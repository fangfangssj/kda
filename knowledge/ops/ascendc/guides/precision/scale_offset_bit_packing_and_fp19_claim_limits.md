---
type: Guide
title: Scale Offset 位打包与 FP19 声明边界
description: 保存掩码、Int9和Uint64字段，纠正浮点位宽及缺失实现。
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

# Scale Offset 位打包与 FP19 声明边界

来源以FP32位模式AND 0xFFFFE000清低13位，再置于uint64 scale字段。这一操作保留原FP32高19位，即sign1、exponent8、fraction10；并不能证明文中宣称的sign1/exponent9/mantissa9硬件FP19格式。此处仅保存给定打包表达式的可验证位语义，真实硬件解码和FP19命名须按目标接口确认。

示例offset字段是(offset&0x1FF)<<37，占bit37..45；bit46是flag。文字“scale占46bit”不能把位位置当有效精度。移位前先扩大为uint64，否则32位中间值左移可能溢出/非法；被截断的转换函数甚至没有return，缺失FP32_EXP等常量也不能补猜为完整可运行实现。

mode0低位清零是位截断。若mode1要求最近偶数，需根据被舍弃位的guard、sticky及保留最低位奇偶决定增量，且处理尾数进位到指数、指数溢出、subnormal、Inf/NaN和符号；只写一个rounding常量不足以证明所有情况正确。不能据示意位运算承诺精度与吞吐。

有符号Int9范围[-256,255]，合法整数再mask0x1FF得到补码编码。先clip后mask与直接mask不同，例如256直接mask解成-256；NaN或巨大浮点值不能先做未定义/超域整数Cast再夹紧。复用已打包word时先清除旧offset字段再OR，避免旧bit残留。

offset长度1且值不变时可预计算字段并与每个scale组合；逐元素offset须长度/广播合同匹配。缓存跨调用是否失效还取决于输入内容，不能只看指针相同。打包scale/offset并不是普通逐元素除法，源baseline若算vdiv则不可直接作等价速度对照；3–5倍声明本批不作为实测事实。
