---
type: API
title: FreeAllEvent
description: 释放TQueBind队列中申请的所有同步事件，便于在Buffer数量超过同步事件限制时重新申请队列。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQueBind/FreeAllEvent.md
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

# FreeAllEvent<a name="ZH-CN_TOPIC_0000001857594165"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

释放队列中申请的所有同步事件。队列分配的Buffer关联着同步事件的eventID，因为同步事件的数量有限制，如果同时使用的队列Buffer数量超过限制，将无法继续申请队列，使用本接口释放队列中的事件后，可以再次申请队列。详细介绍请参考[TQue Buffer限制](../t_que/t_que_intro.md#section466543213575)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void FreeAllEvent()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](../../../../../guides/programming_guide/appendix/common_operations/tensor_inplace_performance.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 接口: DeQue Tensor
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);
tensor1 = que.DeQue<half>(); // 将tensor从VECOUT的Queue中搬出
que.FreeTensor<half>(tensor1);
que.FreeAllEvent();
```
