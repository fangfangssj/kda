---
type: Guide
title: Grouped POD 消费与 MIX 唯一写者
description: 保留完整 Tiling 序列化、GM loader、按组任务重置和单行 sub-block ownership。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/group-matmul-delta.md
  role: primary
created_at: '2026-09-11T03:07:47Z'
updated_at: '2026-09-11T03:07:47Z'
platforms: []
---

# Grouped POD 消费与 MIX 唯一写者

Grouped Host/Device共享POD应固定类型、packing、alignment、sizeof/offsetof与字段语义。来源普通GMM包裹普通MM tiling并增加group地址/数量/编码；Quant/MX的POD不能由此推导。共享定义与一次完整构造有助于避免多处补字段漂移，但是否必须特定grouped overload是该工程设计，不是所有Blaze的硬件规则。

来源SWAT用totalM/N/K求普通MM参数，group metadata不参与搜索，这是该tiler的合同，不证明所有GMM性能模型都可忽略分组。字段存在不等于Block消费，需核对Host产生值、序列化偏移、Device loader和首消费者；sizeof通过仍可能遗漏loader字段并让Kernel按默认值提前返回。

__gm__结构整体复制到本地是否支持取决于工具链地址空间/构造规则；可使用证明过的loader或按共享schema逐字段读取，不另维护第二套Tiling算法。用非空且预期非零的group检查workspace/scale是否仍为初始化哨兵，能暴露control字段未消费；全零结果不独立证明只是loader，也可能是计算或同步问题。

每组重新建立[groupM,N,K]问题并枚举其M/N tile，空组不生成tile。Kernel统一group地址与任务生命周期，不能把普通MM Scheduler改名就认为已支持group。baseM/N/K是合法物理基形，逻辑K不足基形时需要真实tail裁剪；把N拆小不能让每个局部片独自产生最终行scale，必须保留跨片合并或唯一完整行owner。

MIX物理GetBlockIdx、GetSubBlockIdx与逻辑AIC tile/AIV行rank需按真实入口映射。GetBlockIdx()/GetTaskRation仅在该映射已成立时适用。单行baseM=1尤其需要唯一写者，其他sub不应触碰同一C/workspace/y/yScale，但仍可能负责完成通知。物理ratio不能推导所有lane都产生数据或所有行平均存在。

验证完整POD的每个关键字段、group地址/编码、stride、实际dispatch、单行/奇偶M、空组、多tile和重复执行。手写Block诊断可隔离问题，但绕过的Scheduler/同步不能由单次launch返回补证；来源限制正式路线的流程要求不移入知识。本批未上板验证。
