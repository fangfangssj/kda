---
type: API
title: GetSize
description: 获取GlobalTensor的元素个数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/GlobalTensor/GetSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
---

# GetSize<a name="ZH-CN_TOPIC_0000002167649065"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取GlobalTensor的元素个数。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline uint64_t GetSize() const
```

## 参数说明<a name="section622mcpsimp"></a>

无。

## 返回值说明<a name="section640mcpsimp"></a>

GlobalTensor的元素个数。

## 约束说明<a name="section633mcpsimp"></a>

使用仅传入全局数据指针的[SetGlobalBuffer](set_global_buffer.md)接口对GlobalTensor进行初始化，通过本接口获取到的元素个数为0。

## 调用示例<a name="section17531157161314"></a>

```cpp
AscendC::LocalTensor<uint8_t> stackBuffer;
AscendC::PopStackBuffer<uint8_t, AscendC::TPosition::LCM>(stackBuffer);
uint32_t stackBufferSize = stackBuffer.GetSize();
```

- [QuantConvolution 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution_950.md)
