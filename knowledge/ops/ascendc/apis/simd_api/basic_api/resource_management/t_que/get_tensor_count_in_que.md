---
type: API
title: GetTensorCountInQue
description: 查询Que中已入队的Tensor数量，不支持Tensor原地操作场景。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQue/GetTensorCountInQue.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
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

# GetTensorCountInQue<a name="ZH-CN_TOPIC_0000001500027581"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

查询Que中已入队的Tensor数量。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline int32_t GetTensorCountInQue()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](../../../../../guides/programming_guide/appendix/common_operations/tensor_inplace_performance.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

Que中已入队的Tensor数量

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 通过GetTensorCountInQue查询que中已入队的Tensor数量，当前通过AllocTensor接口分配了内存，并加入que中，num为1。
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);// 将tensor加入VECOUT的Queue中
int32_t numb = que.GetTensorCountInQue();
```
