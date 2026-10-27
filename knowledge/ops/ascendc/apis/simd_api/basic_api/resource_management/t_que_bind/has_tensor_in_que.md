---
type: API
title: HasTensorInQue
description: 查询TQueBind队列中目前是否已有入队的Tensor。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQueBind/HasTensorInQue.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tpipe.h
---

# HasTensorInQue<a name="ZH-CN_TOPIC_0000001857672841"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

查询Que中目前是否已有入队的Tensor。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline bool HasTensorInQue()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](../../../../../guides/programming_guide/appendix/common_operations/tensor_inplace_performance.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

-   true - 表示Queue中存在已入队的Tensor
-   false - 表示Queue为完全空闲

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 根据VacantInQue判断当前que中是否有已入队的Tensor，当前que的深度为4，无内存Enque动作，返回为false
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.HasTensorInQue();
```
