---
type: API
title: 直方图计算
description: 本节介绍寄存器直方图统计接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/histogram_compute/histogram_compute.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 直方图计算

本节介绍寄存器直方图统计接口。

## 接口列表

| 接口 | 功能 |
| --- | --- |
| [histograms](histograms.md) | • 频率统计：<br>$dst_i = dst_i + \sum_{j:\,src.mask_j=1}\mathbf{1}(src_j=h+i)$<br>• 累计统计：<br>$dst_i = dst_i + \sum_{j:\,src.mask_j=1}\mathbf{1}(src_j\le h+i)$<br>其中，低位模式下$h=0$，高位模式下$h=128$。 |

输入数据的有效范围由`src.mask`决定，返回对象继承`dst.mask`。
