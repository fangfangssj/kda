---
type: API
title: GetSubBlockNum
description: 在分离模式架构下，获取一个逻辑AI Core上AIC或AIV的数量。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSubBlockNum_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetSubBlockNum(ISASI)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

在[分离模式架构](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，获取当前配置中，一个逻辑AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。

## 函数原型

```cpp
__aicore__ inline int64_t GetSubBlockNum()
```

## 参数说明

无

## 返回值说明

在核函数（Kernel）直调工程场景下，不同[算子类型](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#section1074418132518)在AIC和AIV上调用该接口的返回值如下：

**表1** 通过函数执行空间限定符设置核函数（Kernel）类型时的返回值

| 算子类型 | 限定符 | AIC | AIV |
| --- | --- | --- | --- |
| Vector算子 | \_\_vector\_\_ | - | 1 |
| Cube算子 | \_\_cube\_\_ | 1 | - |
| Mix算子 | \_\_mix\_\_(0, 1) | - | 1 |
| Mix算子 | \_\_mix\_\_(1, 0) | 1 | - |
| Mix算子 | \_\_mix\_\_(1, 1) | 1 | 1 |
| Mix算子 | \_\_mix\_\_(1, 2) | 1 | 2 |

在自定义算子工程和核函数（Kernel）直调工程场景下，不同核函数（Kernel）类型（通过[设置核函数（Kernel）类型](../../kernel_tiling/set_kernel_type.md)设置）在AIC和AIV上调用该接口的返回值如下：

**表2** 通过Kernel Type接口设置核函数（Kernel）类型时的返回值

| 核函数（Kernel）类型 | KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_2 | KERNEL_TYPE_MIX_AIC_1_1 | KERNEL_TYPE_MIX_AIC_1_0 | KERNEL_TYPE_MIX_AIV_1_0 |
| --- | --- | --- | --- | --- | --- | --- |
| AIV | 1 | - | 2 | 1 | - | 1 |
| AIC | - | 1 | 1 | 1 | 1 | - |

**注意，两种场景中返回值的含义相同，仅设置类型的方式不同，编写算子时推荐使用核函数（Kernel）直调工程。**

## 约束说明

无

## 调用示例

```cpp
int64_t subBlockNum = AscendC::GetSubBlockNum();
```

- [Rasterizer A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/rasterizer.md)
- [QuantConvolution A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution.md)
- [BlockSparseAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention.md)
- [WeightQuantBatchMatmul 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/matmul/weight_quant_batch_matmul_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
- [FlashAttn 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attn_950.md)
- [FusedInferAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/fused_infer_attention_score_950.md)
- [NsaCompressAttentionInfer A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/nsa_compress_attention_infer.md)
- [AlltoAllQuantMatmul A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul.md)
- [MatmulReduceScatter A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_reduce_scatter.md)
