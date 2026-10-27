---
type: API
title: 连续计算（SIMD计算）
description: 连续计算API，支持Tensor前n个数据计算。针对源操作数的连续n个数据进行计算并连续写入目的操作数，解决一维Tensor的连续计算问题。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/SIMD_compute/continuous_compute.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 连续计算

**连续计算API**，支持Tensor前n个数据计算。针对源操作数的连续n个数据进行计算并连续写入目的操作数，解决一维Tensor的连续计算问题。

以Add接口为例，接口调用形式如下：

```cpp
Add(dst, src1, src2, n);
```

下图以矢量加法为例，展示了连续计算API的特点：

- 源操作数src1、src2和目的操作数dst地址连续。
- 接口简洁，仅需指定计算元素个数n。

**图1** **连续计算API**  
![连续计算API](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/continuous_calc_api.png)
