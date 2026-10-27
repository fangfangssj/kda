---
type: Guide
title: UB 驻留生命周期与架构相关bank端口
description: 保留常驻与分时复用，补齐3510两读一写冲突和地址映射限制。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/common/ub_resident_design.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_2201.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md
  role: constraint
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# UB 驻留生命周期与架构相关bank端口

跨tile不变参数可每Process一次搬入/转换后常驻，递推状态也可驻留但需正确初始化与更新；身份至少含参数版本、group/head和dtype，不可跨不同输入调用盲目复用。常驻占用应先从预算扣除，转换前后两份在转换完成前可能同时存活。TBuf视图不会自动建立MTE2→Vector或Vector→MTE覆盖依赖，PIPE_V只覆盖该流水，不能替代跨流水事件。

分时复用把互不重叠生命周期映到同一物理区，预算是各时刻同时存活字节之和的最大值，另加常驻与对齐；不是任意max(单个buffer)。RMSNorm输出若后续RoPE仍需读，不能将对应zone先改cos/sin。流水重叠改变存活区间，原串行复用须重新核算。FP32累加可以扩大精度域，但不保证无限长无溢出/误差，不能无条件替换数值合同。

固定既有架构说明给2201的192KiB/48bank/16group/每组3、3510的256KiB/16bank/8group/每组2。2201读读/写写按group、读写按bank；3510每bank每拍一读或一写、每group两读零写或一读一写，因此源表遗漏同group两读一写冲突。3510离散访问还需8B subbank：同subbank同起点可共享、不同起点会冲突，Gather/Scatter还可与连续/MTE争用。这里只引用已有固定知识卡，未恢复原始图片。

不能按32B行对bank总数简单取模推导全部地址：2201既有说明示例0x10000在bank16，简单(地址/32)%48并不匹配。bank容量4KiB/16KiB也不等同连续地址每隔这个大小回到同bank。完整编码依架构，分析各操作数实际起点、block/repeat stride及同时读写集合，不能只用buffer大小或单源连续证明无冲突。

padding256B和把下一buffer放64KiB边界是特定布局候选，必须真正改变消费者地址且保留全部容量/对齐；≥256B间隔不保证任意架构无冲突。源连续读跳写示例dstBlkStride8仍可能有写冲突，不可仅改善src就断言完全消除。改变stride同时改变逻辑坐标，需等价布局重排。Vector时间高且MTE2不高仅提示候选原因，性能30–60%、每repeat1–2拍需实际指令与冲突证据。
