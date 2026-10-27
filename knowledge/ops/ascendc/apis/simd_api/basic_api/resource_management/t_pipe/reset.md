---
type: API
title: Reset
description: 完成资源的释放与eventId等变量的初始化操作，恢复到TPipe的初始化状态。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/Reset.md
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

# Reset<a name="ZH-CN_TOPIC_0000001785898192"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

完成资源的释放与eventId等变量的初始化操作，恢复到TPipe的初始化状态。


针对Ascend 950PR&950DT系列产品，调用该接口后，CTRL\[48\]保持原值，CTRL\[60\]复位为1。CTRL寄存器比特位说明请见[SetCtrlSpr](../../special_register_access/set_ctrl_spr_isasi.md)。


## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void Reset()
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 1> que; // 输出数据Queue队列管理对象，TPosition为VECOUT
uint8_t num = 1;
uint32_t len = 192 * 1024;
for (int i = 0; i < 2; i++) {
    pipe.InitBuffer(que, num, len);
    ... // process
    pipe.Reset();
}
```

- [MaskedSelect 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/masked_select_950.md)
- [SwigluGroupQuantGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_grad_950.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [Sort 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/sort_950.md)
- [IndexFill 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/index_fill_950.md)
- [GroupedMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul.md)
- [TopK 950 Direct Invoke 算子知识](../../../../../operators/direct/math/math/top_k_950.md)
- [IndexFillTensor 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/index_fill_tensor_950.md)
- [grouped_matmul_swiglu_quant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant.md)
- [InplaceIndexFillTensor 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/inplace_index_fill_tensor_950.md)
