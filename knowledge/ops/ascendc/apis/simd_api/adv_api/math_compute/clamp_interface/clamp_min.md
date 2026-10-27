---
type: API
title: ClampMin
description: 将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。 ! !
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/Clamp_interface/ClampMin.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/adv_api/math/clamp.h
---

# ClampMin

## 功能说明

将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002285449304.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002319985561.png)

## 函数原型

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ClampMin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ClampMin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const T scalar, const uint32_t calCount)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetClampMaxMinTmpSize](get_clamp_max_min_tmp_size.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>用于ClampMin内部复杂计算时存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[GetClampMaxMinTmpSize](get_clamp_max_min_tmp_size.md)。 |
| scalar | 输入 | scalar数据，数据类型与srcTensor一致，支持数据类型为：half、float。 |
| calCount | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

## 调用示例

```
// dstLocal: 存放计算结果的Tensor
// srcLocal: 参与计算的输入Tensor
// sharedTmpBuffer: 临时缓存,内部复杂计算时存储中间变量
// 输入shape信息为128，scalar为2，参与计算元素为128
AscendC::ClampMin<half>(dstLocal, srcLocal, sharedTmpBuffer, static_cast<half>(2), 128);
```

结果示例如下：

```
输入数据(srcLocal): [0 1 2 ...  126 127]
输出数据(dstLocal): [2 2 2  ... 126 127]
```
