---
type: API
title: HasIdleBuffer
description: 查询Que中是否有空闲的内存块可供AllocTensor分配，不支持Tensor原地操作场景。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQue/HasIdleBuffer.md
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

# HasIdleBuffer<a name="ZH-CN_TOPIC_0000001499708581"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

查询Que中是否有空闲的内存块。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline bool HasIdleBuffer()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](../../../../../guides/programming_guide/appendix/common_operations/tensor_inplace_performance.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

-   true - 表示Queue中存在空闲内存
-   false - 表示Queue中不存在空闲内存

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 当前Que中已经分配了4块内存
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.HasIdleBuffer(); // 没有AllocTensor的操作，返回值为true
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
ret = que.HasIdleBuffer(); // AllocTensor了一块内存，返回值为true
AscendC::LocalTensor<half> tensor2 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor3 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor4 = que.AllocTensor<half>();
ret = que.HasIdleBuffer(); // AllocTensor了四块内存，当前无空闲内存，返回值为false，继续AllocTensor会报错
```
