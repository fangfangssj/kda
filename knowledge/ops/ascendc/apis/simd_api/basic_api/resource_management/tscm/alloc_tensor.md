---
type: API
title: AllocTensor
description: 从TSCM中分配tensor，大小为InitBuffer时设置的每块内存长度，分配内容可能为随机值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TSCM/AllocTensor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_tpipe.h
---

# AllocTensor<a name="ZH-CN_TOPIC_0000002161196177"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

从TSCM中分配tensor，tensor所占大小为InitBuffer时设置的每块内存长度。注意，分配的tensor内容并非全0，可能会是随机值。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline LocalTensor<T> AllocTensor()
```

## 参数说明<a name="section622mcpsimp"></a>

| 参数名称 | 说明 |
| -------- | ---- |
| T | 待获取Tensor的数据类型，支持的类型请见[LocalTensor](../../data_structures/local_tensor/index.md)相关描述。|

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor对象。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
AscendC::TPipe pipe;
AscendC::TSCM<AscendC::TPosition::VECIN, 1> tscm;
int num = 4;
int len = 1024;
pipe.InitBuffer(tscm, num, len); // InitBuffer分配内存块数为4，每块大小为1024Bytes
AscendC::LocalTensor<half> tensor1 = tscm.AllocTensor<half>(); // AllocTensor分配Tensor长度为1024Bytes
tscm.EnQue(tensor1);
tensor1 = tscm.DeQue<half>();
tscm.FreeTensor(tensor1);
```
