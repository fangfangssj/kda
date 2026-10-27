---
type: Guide
title: TilingData 紧凑布局、字段位宽与 ABI
description: 以字段值域、物理布局、序列化和消费者一致性约束 TilingData 压缩，验证真实加载成本而非只看字段数量。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/assets/rules/special_rules/R_MEM_TILINGDATA_SIZE/R_MEM_TILINGDATA_SIZE.md
  role: primary
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# TilingData 紧凑布局、字段位宽与 ABI

TilingData 是 Host 向 Device 传递切分与执行参数的合同。字段冗余或结构体 padding 可能增加传输/加载的字节数；小工作量中初始化占比可能更明显。但字段变少不必然降低栈加载、缓存 miss 或 kernel 时延，实际行为还取决于序列化宏、编译器生成代码和 Kernel 使用的字段。

缩窄字段前应分别证明原始值、派生值、乘积、偏移、哨兵和分支取值的范围。例如 uint16_t 只能表示 0 到 65535；单维可表示不证明维度乘积、对齐后的大小或 workspace 偏移可表示。先截断再做范围检查无效，Host 与 Device 的计算提升、符号和溢出规则也需一致。

按字段宽度排序是减少某些普通结构体 padding 的候选，不是消除全部 padding 的通用规则。应对实际布局核对 sizeof、alignof、offsetof 和序列化字节数；宏生成布局或单独编码格式不能直接用 C++ struct 的假设替代。强制 packed 还可能制造未对齐访问，需确认所用消费者支持。

修改字段类型或位置后，Host 写入、Device 解码、所有模板实例、fallback 分支及其他消费者必须使用同一 ABI。移除字段前检查全部消费者，不能只看当前 shape 没有读取。已有 ABI 需版本一致的制品，不能让旧二进制解读新布局。

验证应覆盖字段极值、非对齐 shape、宽地址积、全部执行分支和解码后的语义，再比较实际 TilingData 字节、生成加载指令、初始化区间与端到端时延。总 Task Duration 或 Scalar 活动比例只能提示方向，不能独立证明 GET_TILING_DATA 是瓶颈；符号表本身也不提供完整运行时成本。
