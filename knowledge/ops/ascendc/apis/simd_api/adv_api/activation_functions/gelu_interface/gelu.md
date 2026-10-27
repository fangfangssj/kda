---
type: API
title: Gelu
description: GELU激活函数实现，灵感来源于Relu和Dropout，在激活中引入随机正则思想，支持高精度和高性能模式。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.activation
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/activation_functions/Gelu_interface/Gelu.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/adv_api/activation/gelu.h
---

# Gelu

## 功能说明

在神经网络中，GELU是一个重要的激活函数，其灵感来源于Relu和Dropout，在激活中引入了随机正则的思想。计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002316009816.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000001750683329.png)，化简后可得![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000001702905034.png)

## 函数原型

-   接口框架申请临时空间

    ```
    template <typename T, bool highPrecision = false, bool highPerformance = false>
    __aicore__ inline void Gelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dataSize)
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool highPrecision = false, bool highPerformance = false>
    __aicore__ inline void Gelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
    ```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |
| highPrecision | 是否开启高精度模式，以提升运算准确度。默认值为false，表示不开启高精度模式。<br><br>注意：高精度模式只在half数据类型下开启后生效，该参数的取值不影响float数据类型下的接口精度和性能。 |
| highPerformance | 是否开启高性能模式，以提升运算效率。默认值为false，表示不开启高性能模式。<br><br>注意：开启高性能模式相比于默认不开启高精度和高性能模式会有精度下降，同时开启高精度和高性能模式相比于仅开启高性能模式可能会有性能下降。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcLocal | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>用于接口内部复杂计算时存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[GetGeluMaxMinTmpSize](get_gelu_max_min_tmp_size.md)。 |
| dataSize | 输入 | 实际计算数据元素个数。 |

## 返回值说明

无

## 约束说明

-   源操作数和目的操作数的Tensor空间可以复用。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。
-   仅支持输入shape为ND格式。


-   针对Ascend 950PR&950DT系列产品，接口内部计算对Subnormal的处理方式受编译选项`--cce-ftz`控制（默认值为`true`）：
    -   配置为`false`时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    -   配置为`true`时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例

```
// dstLocal: 存放Gelu计算结果的Tensor
// srcLocal: 存放Gelu的输入Tensor
// sharedTmpBuffer: 存放Gelu计算过程中临时缓存的Tensor

// 接口框架申请临时空间，部分参与计算,需要参与计算的元素个数为32
AscendC::Gelu<srcType, false>(dstLocal, srcLocal, 32);

// 通过sharedTmpBuffer入参传入临时空间，部分参与计算,需要参与计算的元素个数为32
AscendC::Gelu<srcType, false>(dstLocal, srcLocal, sharedTmpBuffer, 32);
```

结果示例如下：

```
输入数据(srcLocal):
[-2.56   -2.395  -2.23   -2.066  -1.9    -1.735  -1.571  -1.406
 -1.241  -1.076  -0.9116 -0.7466 -0.582  -0.417  -0.2522 -0.0874
  0.0774  0.2423  0.407   0.572   0.737   0.902   1.066   1.231
  1.396   1.561   1.726   1.891   2.055   2.22    2.385   2.55  ]
输出数据(dstLocal):
[-0.01295471 -0.01953125 -0.02836609 -0.03991699 -0.05453491 -0.07196045 -0.09130859 -0.11254883
 -0.13342285 -0.15185547 -0.16516113 -0.17004395 -0.16320801 -0.14111328 -0.10101318 -0.04067993
  0.04107666  0.14428711  0.26782227  0.40942383  0.56689453  0.7363281   0.9135742   1.0966797
  1.2822266   1.4677734   1.6533203   1.8349609   2.0136719   2.1914062   2.3632812   2.5390625 ]
```

- [FFN A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/ffn/ffn.md)
- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
