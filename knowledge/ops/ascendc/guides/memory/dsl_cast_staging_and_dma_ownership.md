---
type: Guide
title: DSL 转换的 Cast Staging 与 DMA 所有权
description: 区分参数字节数、有效区、补齐、事件和缓冲复用。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/multi_dtype/multi_dtype_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/multi_dtype/add_custom_reference.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/common_pitfalls.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/non_aligned/non_aligned_example_for_lowering.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# DSL 转换的 Cast Staging 与 DMA 所有权

普通DataCopy的count重载有方向、dtype、长度及地址限制；7个FP32是28B、1个是4B，不能假定任意短尾都可搬，也不一定在编译期报固定alignment错误。DataCopyPad可以按有效byte数搬短尾，但GM→UB带padding与UB→GM不带padding是不同调用。uint16_t(count*sizeof(T))须先验证范围，窄化不是自动分批。

DataCopyPad写LocalTensor[i]仍需满足目标地址对齐，单个float放任意i并非通用合法示例。可以在对齐staging槽搬入再由合法计算提取，或构造符合物理行距的批量搬运。分配量覆盖padding后的物理span和实际消费者，Get<float>(200)不会把仅100float的TBuf扩容。

isPad控制UB填充语义，来源Add注释“isPad=false必然从GM越界读到32B、true就精确”不能作为所有重载的硬件结论。有效blockLen、实际可访问GM范围、dummy来源和计算mask需按API确定；需要0中性值时显式填充有效补齐区，而不是以开关代替地址证明。来源CANN8.5 BF16 ExtParams ABI bug未提供本轮可核验外部实现，保留为排障线索，不能禁止所有DataCopyPadExtParams<T>或声称simple支持任意元素类型。

对tileSize2048、3个单份I/O队列，FP32为24576B、FP16为12288B；BF16加3个2048FP32 staging共36864B，即24/12/36KiB。实际还需队列份数、padding、scratch和其他存活区，192KiB不是全平台恒定UB。来源“Cast每次无条件按16个输入写64B”需重载/mask证据；ceil(N/16)*64可作保守容量候选，不能把它当所有Cast必需物理写入量。N196时32B上对齐800B与64B上对齐832B相差32B，算术成立不等于已证明写越界。

TBuf只是存储视图，不提供自动跨pipe同步。MTE2搬入→V Cast需要生产完成，V Cast→MTE3写出也需依赖；连续复用时还要V读完→下次MTE2覆盖以及MTE3读完→下次V覆写。来源模板只列两个正向事件，复用同一ioConvBuf时并不构成完整无限循环协议；float直搬分支省略事件也必须由调用方/队列建立。

固定event0只有在该HardEvent类别的生命周期无冲突时可用，不能与TPipe或其他helper占用冲突。Queue EnQue/DeQue表达相应方向的数据交接，但不替代所有TBuf或Scalar依赖。模板LoadRowF32示例Alloc后直接DeQue前省略了EnQue，省略号不能当完整队列协议。初始化后重复TBuf.InitBuffer还可能重复消耗资源，不应只视为malloc慢；复用必须等待最后消费者。
