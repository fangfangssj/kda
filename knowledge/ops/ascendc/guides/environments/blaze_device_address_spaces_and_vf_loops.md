---
type: Guide
title: Blaze 设备地址空间与 VF 循环约束
description: 保留真实编译诊断中的 GM/UB 指针、宏分区、Reg mask 和向量循环宽度条件。
tags:
- task.debugging
- topic.compilation
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/troubleshooting/compile-troubleshooting.md
  role: primary
created_at: '2026-09-11T02:57:41Z'
updated_at: '2026-09-11T02:57:41Z'
platforms: []
---

# Blaze 设备地址空间与 VF 循环约束

LocalTensor产生的UB地址在后续表达式中仍须保留目标工具链的__ubuf__地址空间，GM地址同理保留__gm__。先存入普通T*再强转不能证明设备语义等价；可采用工具链支持的显式类型或auto推导，但仍需核对最终调用签名。const __gm__ TilingData也不能机械绑定普通C++引用。

GM POD可通过当前工具链证明的typed loader读到本地结构，再传给消费者；若具体MakeMemPtr入口接受raw GM_ADDR，调用合同就应保留这种形式，不能只因Host能编译就替换成另一种指针ABI。来源要求某些资产统一loader是工程流程，不能推广为所有Ascend C程序禁止合法逐字段GM读取。

ASCEND_IS_AIC/AIV的实际宏展开可能是编译期分区语法，不能按普通布尔值拼成if ASCEND_IS_AIC && condition。运行时条件应置于对应合法分区内，另一侧不能访问未实例化对象。MIX入口修饰词与ratio按当前版本的实际语法核验，不能从相似样例叠加关键字。

来源__VEC_SCOPE__诊断要求induction为uint16_t时，应将循环变量及边界置于可证明范围，地址、shape与剩余量仍保留足够宽类型。若循环次数超过该域，可拆分并重新计算每批mask，不能窄化后回绕；此约束限于出现该DSL诊断的接口/版本，不推广到普通Device循环。

Reg mask若接口要求可变引用，传入const值不能用const_cast掩盖；elementwise、reduction、rounding和store各有具体签名、lane及地址空间约束。Tensor API NDExt视图传legacy Muls出现PrimType缺失，表示两个接口的Tensor合同不匹配，不能据此认定float不支持，也不能伪造成员绕过。

用实际目标的最小编译实例确认声明/重载/地址空间，再以完整Kernel验证数据与同步；Host编译、Device链接和真实运行是不同证据。本批只保留来源诊断及可复用约束，未将其历史修复宣称为本批复现。
