---
type: Guide
title: 尽量一次搬运较大的数据块
description: 【描述】搬运不同大小的数据块时，对带宽的利用率（有效带宽/理论带宽）不一样。根据实测经验，单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能。因此对于单次搬运，应考虑尽可能的搬运较大的数据块。下图展示了某款AI处理器上实测的不同搬运数据量下带宽的变化图。
tags:
- task.lookup
- topic.performance.optimization.optimization_overview
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/transfer_larger_data_blocks.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 尽量一次搬运较大的数据块<a name="ZH-CN_TOPIC_0000001893771813"></a>

【优先级】高

【描述】搬运不同大小的数据块时，对带宽的利用率（有效带宽/理论带宽）不一样。根据实测经验，单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能。因此对于单次搬运，应考虑尽可能的搬运较大的数据块。下图展示了某款AI处理器上实测的不同搬运数据量下带宽的变化图。

>[!NOTE]说明 
>测试数据与处理器型号相关，且实际测试时可能会存在略微抖动，具体带宽数值并不一定和下文的测试数据严格一致。

**图1**  UB-\>GM方向不同单次搬运数据量下实际占用带宽的变化<a name="fig925755124312"></a><br>
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/ub_gm_bw.png "UB--GM方向不同单次搬运数据量下实际占用带宽的变化")

**图2**  GM-\>UB方向不同单次搬运数据量下实际占用带宽的变化<a name="fig859712315453"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/gm_ub_bw.png "GM--UB方向不同单次搬运数据量下实际占用带宽的变化")
