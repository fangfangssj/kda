---
type: API
title: Utils APImax
description: 比较相同数据类型的两个数，返回最大值。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/algorithms/max.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
---

# max

## 功能说明

比较相同数据类型的两个数，返回最大值。

## 函数原型

```
template <typename T, typename U>
__host__ __aicore__ inline constexpr auto max(const T& src0, const U& src1)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| T | 输入数据src0的数据类型。当前支持的数据类型为bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、float、int64_t、uint64_t。 |
| U | 输入数据src1的数据类型。当前支持的数据类型为bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、float、int64_t、uint64_t。<br><br>预留类型，当前必须与T保持一致。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| src0 | 输入 | 源操作数。参与比较的输入。 |
| src1 | 输入 | 源操作数。参与比较的输入。 |

## 约束说明

两个源操作数的数据类型必须相同。

## 返回值说明

两个输入数据中的最大值。

## 调用示例

```
int64_t src0 = 1;
int64_t src1 = 2;

int64_t result = AscendC::Std::max(src0, src1);
// result: 2
```

- [Rasterizer A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/rasterizer.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
- [AdaLayerNorm A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/ada_layer_norm.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
- [AdaLayerNormQuant A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/ada_layer_norm_quant.md)
- [BatchNorm 950 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/batch_norm_950.md)
