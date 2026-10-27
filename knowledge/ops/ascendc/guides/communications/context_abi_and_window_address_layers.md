---
type: Guide
title: 通信上下文 ABI 与窗口地址的分层
description: 区分上下文对象、按rank的窗口基址和业务slot偏移，避免用精简样例结构解析真实HCCL资源。
tags:
- task.implementation
- topic.memory
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/mte-address-access.md
  role: primary
created_at: '2026-09-10T14:40:34Z'
updated_at: '2026-09-10T14:40:34Z'
platforms: []
---

# 通信上下文 ABI 与窗口地址的分层

在来源描述的 MoE 直调方案里，host 创建通信资源后向 kernel 传入 mc2Context。kernel 需要把上下文解释为与资源生产端一致的结构，再取得本卡或远端窗口地址。

## 结构名不能证明 ABI 兼容

结构体字段顺序、大小、对齐和嵌套结构共同决定访问偏移。来源明确警告：样例的精简 HcclA3OpResParam/HcclA5OpResParam 仅覆盖部分字段，不能据此解析真实 HCCL 返回的 context。中间字段被省略后，后续同名字段也可能读错。

真实集成应采用目标 SDK 版本的完整声明及其支持的创建、访问方式。平台型号相同不等于不同 SDK 的私有布局相同；本文不把参考中的结构体、固定 1 MiB 偏移或 A3/A5 别名发布为跨版本 ABI。

## 三层地址语义

1. 上下文层描述资源、rank 和生命周期，不是某条通信记录的地址。
2. 窗口层取得指定 rank 的数据区或状态区基址。两者可能是独立指针，也可能位于同一分配的不同区域，须按真实资源合同解释。
3. 业务布局层在对应基址上叠加 expert、来源 rank、token、slot 与轮次偏移，并验证访问范围没有超出容量。

compat helper 可以隐藏实现分支，但它不是同名公共 API 的证明。把完整窗口起点误当纯数据起点，会把载荷写入状态区；把数据区展平公式用于状态区，也可能访问正确分配中的错误记录。

## 生命周期与验证

明确 context 指针属于 host 还是 device 地址、由谁创建和释放，确保它及关联窗口存活至异步 kernel 和通信消费完成。来源方案由通信库持有 context，不能当普通输入 buffer 任意覆写或释放。

可先核对目标 SDK 声明及版本，再检查 rank 范围、容量、字段偏移和由此推导的小规模地址表；真实多卡访问仍需执行验证。mock 自建 context 成功只证明 mock 内部布局一致。窗口内记录语义见[MoE 窗口发布和复用](../moe/window_publication_and_reuse.md)，公开算子的入口差异见[MoeDistributeDispatch A2/A3](../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)。
