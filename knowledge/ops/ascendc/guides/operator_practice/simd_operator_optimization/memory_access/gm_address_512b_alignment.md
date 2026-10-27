---
type: Guide
title: GM地址尽量512B对齐
description: 【描述】由于AI处理器内部设计约束，从GM向Local Memory搬运数据时，保证GM地址512B对齐可以最有效的发挥出带宽的效率。如下图示例，展示了在512B对齐以及32B对齐情况下单核的带宽效率：搬运同等数据量，带宽差距最大的情况，32B对齐场景只能达到512B对齐场景的70%。
tags:
- task.lookup
- topic.performance.optimization.optimization_overview
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# GM地址尽量512B对齐<a name="ZH-CN_TOPIC_0000001892919369"></a>

【优先级】高

【描述】由于AI处理器内部设计约束，从GM向Local Memory搬运数据时，保证GM地址512B对齐可以最有效的发挥出带宽的效率。如下图示例，展示了在512B对齐以及32B对齐情况下单核的带宽效率：搬运同等数据量，带宽差距最大的情况，32B对齐场景只能达到512B对齐场景的70%。

>[!NOTE]说明 
>该性能优化手段适用于如下产品型号：
>
>- Ascend 950PR&950DT系列产品
>
>
>- Atlas A2系列产品
>
>- 测试数据与处理器型号相关，且实际测试时可能会存在略微抖动，具体带宽数值并不一定和下文的测试数据严格一致。

**图1**  GM-\>UB方向512B对齐和32B对齐实测带宽的差异对比<a name="fig5376636154310"></a><br>
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/gm_ub_align.png "GM--UB方向512B对齐和32B对齐实测带宽的差异对比")

**图2**  UB-\>GM方向512B对齐和32B对齐实测带宽的差异对比<a name="fig13744224715"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/ub_gm_align.png "UB--GM方向512B对齐和32B对齐实测带宽的差异对比")
