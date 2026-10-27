---
type: Guide
title: 手写 FA 缓冲与状态故障模式
description: 保留多任务覆盖和事件问题，修正共享UB与最终Max公式。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-handcraft-pitfalls.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# 手写 FA 缓冲与状态故障模式

FAQK两个实例若共享事件/资源管理且构造析构都操作同一信用，第二实例即使未调用也可能破坏配平；来源建议单实例分趟是该实现修复，不是所有CATLASS同类对象不可共存。每事件生命周期需平衡，不能简化为所有id每tile必恰好1Set1Wait，预置、借用、多消费者和排空会改变局部计数。

mode0x2来源双AIV→AIC使用双方set和一个逻辑wait，反向单set可供双AIV；参与者不可单边return。其真实硬件/封装语义需匹配核型和版本，不能替换为950mode4表。纯Vector merge没有混合subblock语义，需按其真实block数覆盖全部行。

来源“两个subblock共享物理UB”不能从覆写症状推出；物理核资源、LocalTensor地址空间、手工allocator和GM共享工作区应分别核验。需要给实际共享区按sub/core分段，不应无证据把所有AIV私有UB再强制减半。ubMax/ubSum512B对齐只在其广播/写出布局合同下成立，不是所有归约API通用起点下限。

oTmp单槽在多task复用时，上一任务最后块也必须等待消费结束后释放；只保护非末块会跨task覆盖。GM双槽基址应含depth，例如(core*2+slot)*slotSize，不能core*slotSize+slot*slotSize使相邻核重叠。B1通过不能覆盖这些多任务映射，Q/O和mask前缀也需随batch递增。

两launch方案第一阶段保存每块P=exp(S-m_j)、m_j和最终l，第二阶段需exp(m_j-m_final)缩放PV结果再归一化。m_final为最后有效running max或各独立分区max的最大值，来源“末块m+sum(m_j)”错误。P采用固定物理pitch时尾块必须写正确中性值；逻辑valid长度和pitch不同，不能混用。

将计算和搬运分离或物理分开scratch可以消除某些复用冲突，但PIPE_ALL后仍错不证明该API无法同步，需检查跨核和真正最后消费者。设备printf、int→float、lambda均有编译模式/API范围，来源个例不能变成Ascend C全局禁令。同步标记、binary指纹和输入/输出身份比单看mtime更能排除stale观测。
