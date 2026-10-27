---
type: API
title: GetBlockNum
description: 获取当前任务配置的逻辑AI Core的数量，用于多核逻辑控制等。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetBlockNum.md
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
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetBlockNum

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取当前任务配置的逻辑AI Core的数量，用于代码内部的多核逻辑控制等。

## 函数原型

```cpp
__aicore__ inline int64_t GetBlockNum()
```

## 参数说明

无

## 返回值说明

返回当前任务配置的逻辑AI Core的数量。

- 返回值与<<<\>\>\>调用时配置的numBlocks一致，为参与计算的逻辑AI Core的数量。
- 当仅启动Cube Core（AIC）或Vector Core（AIV）时，实际启动的AIC或AIV的数量与numBlocks一致。
- 当同时启动Cube Core（AIC）与Vector Core（AIV）时，实际启动的AIC或AIV的数量可以通过[GetTaskRatio\(\)](get_task_ratio.md) \* numBlocks计算得到，在当前支持的配置下：
    - 当AIC与AIV比例为1:2时，AIC启动的数量为numBlocks，AIV启动的数量为2 \* numBlocks；
    - 当AIC与AIV比例为1:1时，AIC启动的数量为numBlocks，AIV启动的数量为numBlocks。

## 约束说明

无

## 调用示例

```cpp
// srcGm、dstGm为外部输入的gm空间
AscendC::GlobalTensor<float> srcGlobal;
AscendC::GlobalTensor<float> dstGlobal;
int64_t blockNum = AscendC::GetBlockNum(); // 获取当前任务配置的逻辑AI Core的数量
int64_t perBlockSize = srcDataSize / blockNum; // 每个核平分处理相同个数
int64_t blockIdx = AscendC::GetBlockIdx(); // 获取当前工作的核ID
srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm + blockIdx * perBlockSize * sizeof(float)), perBlockSize);    // 分配每个核上的srcGlobal的内存地址
dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm + blockIdx * perBlockSize * sizeof(float)), perBlockSize);    // 分配每个核上的dstGlobal的内存地址
```

- [GatherV2 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/gather_v2_950.md)
- [Rasterizer A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/rasterizer.md)
- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [Celu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/celu_950.md)
- [BlockSparseAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention.md)
- [UpsampleBicubic2dAA 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_950.md)
- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [InplaceCelu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/inplace_celu_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
- [UpsampleBicubic2dAAGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad_950.md)
- [MaskedSelect A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/masked_select.md)
