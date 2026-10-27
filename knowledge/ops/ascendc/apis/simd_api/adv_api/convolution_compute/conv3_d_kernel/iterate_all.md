---
type: API
title: IterateAll
description: 通过设置Output在GM上的首地址，一次性计算单核卷积输出数据块并写入结果矩阵。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/IterateAll.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# IterateAll

## 功能说明

通过设置结果矩阵Output在GM上的首地址，本接口一次性计算singleCo \* singleDo \* singleM大小的数据块，并写到结果矩阵Output中。

本接口提供单核内卷积计算能力，singleCo为多核切分后单个核内的输出通道大小；singleDo为多核切分后单个核内的Dout大小；singleM为多核切分后单个核内的M大小。singleCo、singleDo和singleM的大小通过[SetSingleOutputShape](set_single_output_shape.md)接口设置。

## 函数原型

```
__aicore__ inline void IterateAll(const AscendC::GlobalTensor<OutputT>& output, bool enPartialSum = false)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| output | 输入 | Output在GM上的地址。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。结果矩阵Output支持的数据类型为：half、bfloat16_t。 |
| enPartialSum | 输入 | 预留参数。 |

## 返回值说明

无

## 约束说明

-   IterateAll接口仅支持处理单batch数据，在多batch计算场景中，需要通过batch次循环调用IterateAll接口完成计算。

    ```
    for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
        conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);
        conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);
        conv3dApi.End();
    }
    ```

-   IterateAll接口必须在初始化接口及输入输出配置接口之后进行调用，完成Conv3D计算，调用顺序如下。

    ```
    Init(...);
    ... // 输入输出配置
    IterateAll(...);
    End();
    ```

## 调用示例

```
TPipe pipe;
conv3dApi.Init(&tiling);
conv3dApi.SetWeight(weightGm);
if (biasFlag) {
    conv3dApi.SetBias(biasGm);
}
conv3dApi.SetInputStartPosition(diIdxStart, mIdxStart);
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);
    conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);
    conv3dApi.End();
}
```

- [QuantConvolution A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution.md)
