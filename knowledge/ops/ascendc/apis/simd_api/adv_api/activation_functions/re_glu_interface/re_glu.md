---
type: API
title: ReGlu
description: ReGlu是一种GLU变体，使用Relu作为激活函数。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.activation
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/activation_functions/ReGlu_interface/ReGlu.md
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
- include/adv_api/activation/reglu.h
---

# ReGlu

## 功能说明

ReGlu是一种GLU变体，使用Relu作为激活函数，计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002316175130.png)

其中Relu激活函数的计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000001792534088.png)

## 函数原型

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const uint32_t calCount)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetReGluMaxMinTmpSize](get_re_glu_max_min_tmp_size.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。不同型号支持的数据类型请参考[支持的数据类型](#li356910105419)。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor0 | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| srcTensor1 | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>用于ReGlu内部复杂计算时存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[GetReGluMaxMinTmpSize](get_re_glu_max_min_tmp_size.md)。 |
| calCount | 输入 | 实际计算数据元素个数。 |

## 返回值说明

无

## 约束说明

-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。
-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   当前仅支持ND格式的输入，不支持其他格式。
-   支持的数据类型<a id="li356910105419"></a>

    
    Ascend 950PR&950DT系列产品，支持的数据类型为：half、bfloat16\_t、float。
    

    
    Atlas A3系列产品，支持的数据类型为：half、bfloat16\_t、float。
    

    
    Atlas A2系列产品，支持的数据类型为：half、bfloat16\_t、float。
    

    
    Atlas推理系列产品AI Core，支持的数据类型为：half、float。
    

    
    Kirin X90处理器系列产品，支持的数据类型为：half、float。
    

    
    Kirin 9030处理器系列产品，支持的数据类型为：half、float。
    

## 调用示例

```
// dstLocal: 存放ReGlu计算结果的Tensor
// src0Local: 存放ReGlu计算输入的Tensor
// src1Local: 存放ReGlu计算输入的Tensor
// sharedTmpBuffer: 存放ReGlu计算过程中临时缓存的Tensor

// 接口框架申请临时空间，部分参与计算,需要参与计算的元素个数为30
AscendC::ReGlu(dstLocal, src0Local, src1Local, 30);

// 通过sharedTmpBuffer入参传入临时空间，部分参与计算,需要参与计算的元素个数为30
AscendC::ReGlu(dstLocal, src0Local, src1Local, sharedTmpBuffer, 30);
```

结果示例如下：

```
输入数据(src0Local):
[2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
输入数据(src1Local):
[-1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1
  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.   1.1  1.2  1.3  1.4  1.5]
输出数据(dstLocal):
[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6 1.8 2.  2.2 2.4 2.6 0.  0. ]
```
