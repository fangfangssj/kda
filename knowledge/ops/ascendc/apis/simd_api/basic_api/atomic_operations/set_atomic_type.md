---
type: API
title: SetAtomicType
description: 通过模板参数设定后续原子操作的数据类型。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/SetAtomicType.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_set_atomic_intf.h
---

# SetAtomicType<a name="ZH-CN_TOPIC_0000001835083633"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_set_atomic_intf.h"`。

通过设置模板参数来设定原子操作不同的数据类型。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void SetAtomicType()
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 设定原子操作不同的数据类型。 |

## 数据类型


- Ascend 950PR&950DT系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas A3系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas A2系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas 200I/500 A2推理产品，支持的数据类型为int16_t、half、int32_t、float。



- Atlas推理系列产品AI Core，支持的数据类型为int16_t、half、float。


## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- [SetAtomicAdd](set_atomic_add.md)、[SetAtomicMax](set_atomic_max_isasi.md)、[SetAtomicMin](set_atomic_min_isasi.md)内部已集成与SetAtomicType相同的功能。建议调用上述接口时，通过设置模板参数显式指定原子操作的数据类型，无需额外调用SetAtomicType接口。
- 使用完成后，建议使用DisableDmaAtomic清空原子操作的状态，以免影响后续相关指令功能。

## 调用示例<a name="section177231425115410"></a>

根据约束说明可知，使用[SetAtomicAdd](set_atomic_add.md)、[SetAtomicMax](set_atomic_max_isasi.md)、[SetAtomicMin](set_atomic_min_isasi.md)时，已通过模板参数指定原子操作的数据类型，无需额外调用`SetAtomicType`接口，因此本接口不提供调用示例。

- [BlockSparseAttentionGrad 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/block_sparse_attention_grad_950.md)
- [FlashAttentionScoreGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_score_grad.md)
- [FlashAttentionUnpaddingScoreGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_unpadding_score_grad.md)
