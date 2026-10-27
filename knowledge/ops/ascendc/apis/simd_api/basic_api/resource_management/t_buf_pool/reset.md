---
type: API
title: Reset
description: 切换TBufPool资源池时使用，结束当前资源池相关事件，资源池及Buffer仍然存在。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TBufPool/Reset.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_tpipe.h
---

# Reset<a name="ZH-CN_TOPIC_0000001912722605"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

在切换TBufPool资源池时使用，结束当前TbufPool资源池正在处理的相关事件。调用后当前资源池及资源池分配的Buffer仍然存在，只是Buffer内容可能会被改写。可以切换回该资源池后，重新开始使用该Buffer，无需再次分配。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void Reset()
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

参考[InitBufPool](init_buf_pool.md)。

- [NormRopeConcatGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/norm_rope_concat_grad.md)
- [NormRopeConcatGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/norm_rope_concat_grad_950.md)
