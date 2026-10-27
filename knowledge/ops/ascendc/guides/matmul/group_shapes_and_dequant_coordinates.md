---
type: Guide
title: GroupedMatmul 的组内尺寸、地址与反量化坐标
description: 区分独立 tensor 与拼接存储，按每组有效范围分派矩阵块，并对齐 per-token 和 per-channel scale。
tags:
- task.design
- topic.tiling
- topic.compute.quantization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-gmm.md
  role: primary
created_at: '2026-09-10T12:53:40Z'
updated_at: '2026-09-10T12:53:40Z'
platforms:
- a2
- a3
---

# GroupedMatmul 的组内尺寸、地址与反量化坐标

GroupedMatmul 每组的 M/K/N 可以不同。输入是独立 tensor 列表、共享 tensor 还是拼接存储，会改变 shape 来源及地址取得方式。只有明确采用连续拼接时，下一组 A/B/C 的起点才能按前组实际存储跨度累计；不能把所有多 tensor 输入都解释成一个连续 GM 区间。

使用最大 shape 构造统一任务网格时，每个任务仍需检查对应组的实际范围。设块起点为 `(m0,n0)`，若任一坐标已在本组之外，该任务没有有效输出；否则有效尺寸分别为 `min(baseM,M_group-m0)` 和 `min(baseN,N_group-n0)`。只检查整个网格最后一块的尾部，会漏掉较小组中的空任务或越界。

Shape 数组大小、特殊值编码和 groupList 解释属于实际算子版本契约。来源的 128 项数组与 `mList[0]==-1` 不能当作所有 GMM 的固定 ABI。

## 反量化链

对采用 int32 累加后 per-channel/per-token 放缩的实现，可将第 g 组有效元素的关系表示为 `Y_g[m,n] = cast(C_g[m,n] * channel_scale_g[n] * token_scale_g[m])`，具体中间类型、bias 与量化规则以算子语义为准。行 scale 与列 scale 要分别按组起点和 tile 坐标偏移。

普通 count 形式的逐元素 Mul 不会因一个输入较短就自动完成行列广播。需要显式构造合法的广播地址或采用支持的广播接口。来源中的简化伪代码不能作为可编译实现。

AIC 向 workspace 写累加结果、AIV 读取并反量化时，数据就绪和 workspace 可复用需分别交接。资源预算按真实存活 buffer 和 dtype 字节数计算；核比例、缓冲份数及分组阈值不能直接从示例固定下来。

验证应覆盖不等组 shape、空组、单/多 tensor 组织、尾块、不同 scale 以及多轮槽位复用。已有 [GroupedMatmul 调优案例](../operator_practice/best_practices/grouped_matmul_tuning.md)提供一种 A2 上调整 AIC/AIV 比例和 workspace 的具体证据。
