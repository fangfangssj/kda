---
type: API
title: Mmad接口边界值汇总
description: 本节汇总介绍了Mmad计算接口在边界值输入下的输出结果，数据类型见基础API下的mmad，以下内容针对如下型号生效：
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/appendix/mmad_interface_boundary_value_summary.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Mmad接口边界值汇总<a name="ZH-CN_TOPIC_0000002607849175"></a>

本节汇总介绍了Mmad计算接口在边界值输入下的输出结果，数据类型见基础API下的[mmad](../simd_api/basic_api/cube_compute_isasi/mmad_compute/mmad.md)，以下内容针对如下型号生效：


- Ascend 950PR&950DT系列产品


- Atlas A3系列产品


- Atlas A2系列产品


**表1**  Mmad特殊值/边界值输入的计算结果说明

<a name="table7822641411"></a>

| C矩阵的数据类型 | Mmad子阶段 | A矩阵的元素取值 | B矩阵的元素取值 | dst的元素取值<br>（INF/NAN模式） |
| --- | --- | --- | --- | --- |
| float | 乘法阶段 | +inf | norm(> 0.0)/+inf | +inf |
| float | 乘法阶段 | +inf | norm(< 0.0)/-inf | -inf |
| float | 乘法阶段 | -inf | norm(> 0.0)/+inf | -inf |
| float | 乘法阶段 | -inf | norm(< 0.0)/-inf | +inf |
| float | 乘法阶段 | +inf/-inf | 0.0 | nan |
| float | 乘法阶段 | norm(> 0.0) | +inf | +inf |
| float | 乘法阶段 | norm(> 0.0) | -inf | -inf |
| float | 乘法阶段 | norm(< 0.0) | +inf | -inf |
| float | 乘法阶段 | norm(< 0.0) | -inf | +inf |
| float | 乘法阶段 | 0.0 | +inf/-inf | nan |
| float | 乘法阶段 | nan | 任意输入 | nan |
| float | 乘法阶段 | 任意输入 | nan | nan |
| float | 乘法阶段 | 任意输入 | 任意输入 | 结果溢出情况下，正向溢出为inf，负向溢出为-inf |
| float | 加法阶段 | +inf | -inf | nan |
| float | 加法阶段 | +inf | +inf/norm | +inf |
| float | 加法阶段 | -inf | +inf | nan |
| float | 加法阶段 | -inf | -inf/norm | -inf |
| float | 加法阶段 | nan | 任意输入 | nan |
| float | 加法阶段 | 任意输入 | nan | nan |

注：

INF/NAN模式（非饱和模式）下，乘法阶段：inf\*非零值输出inf，inf\*0输出nan，输入包含nan时输出为nan。加法阶段：正负inf相加时输出为nan。
