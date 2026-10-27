---
type: API
title: Destroy
description: 释放TPipe资源，用于重复申请释放TPipe，创建TPipe对象后可调用Destroy手动释放资源。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/Destroy.md
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

# Destroy<a name="ZH-CN_TOPIC_0000001764926901"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

释放资源。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void Destroy()
```

## 约束说明<a name="section633mcpsimp"></a>

用于重复申请释放tpipe，创建tpipe对象后，可调用Destroy手动释放资源。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; //输出数据Queue队列管理对象，TPosition为VECOUT
uint8_t num = 2;
uint32_t len = 128;
pipe.InitBuffer(que, num, len);
pipe.Destroy();
```

- [AddLora A2 ACLNN 算子知识](../../../../../operators/aclnn/math/math/add_lora.md)
- [ScatterNd A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/scatter_nd.md)
- [BlockSparseAttentionGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_grad.md)
- [ScatterNdUpdate A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/scatter_nd_update.md)
- [FlashAttentionScoreGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_grad.md)
- [TopKTopPSample A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/top_k_top_p_sample.md)
- [GroupedMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul.md)
- [TopKTopPSample 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/top_k_top_p_sample_950.md)
- [grouped_matmul_swiglu_quant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant.md)
- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
