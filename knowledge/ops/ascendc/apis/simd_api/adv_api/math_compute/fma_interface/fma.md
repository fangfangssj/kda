---
type: API
title: Fma
description: 按元素计算两个输入相乘后与第三个输入相加的结果
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/Fma_interface/Fma.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/adv_api/math/fma.h
---

# Fma

## 功能说明

按元素计算两个输入相乘后与第三个输入相加的结果。计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002353015644.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002386692741.png)

## 函数原型

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <const FmaConfig& config = DEFAULT_FMA_CONFIG, typename T>
    __aicore__ inline void Fma(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,const LocalTensor<T>& src2, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
    ```

-   接口框架申请临时空间

    ```
    template <const FmaConfig& config = DEFAULT_FMA_CONFIG, typename T>
    __aicore__ inline void Fma(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const LocalTensor<T>& src2, const uint32_t count)
    ```

由于该接口的内部实现中涉及精度转换。需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetFmaMaxMinTmpSize](get_fma_max_min_tmp_size.md)中提供的接口获取需要预留空间的大小。

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| FmaConfig | Fma算法的相关配置。此参数可选配，FmaConfig类型，具体定义如下方代码所示，其中参数的含义为：<br><br>isReuseSource：是否允许修改源操作数。该参数预留，传入默认值false即可。 |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |

```
struct FmaConfig {
    bool isReuseSource;
};
```

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0、src1、src2 | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>用于Fma内部复杂计算时存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[GetFmaMaxMinTmpSize](get_fma_max_min_tmp_size.md)。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

## 调用示例

-   通过sharedTmpBuffer入参传入

    ```
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
    pipe.InitBuffer(tmpQue, 1, bufferSize); // bufferSize通过Host侧tiling参数获取
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
    // 输入tensor长度为1024,算子输入的数据类型为half,实际计算个数为512
    static constexpr AscendC::FmaConfig fmaConfig = {false}; // 不修改源操作数
    // dst、src0、src1、src2为half类型的LocalTensor
    AscendC::Fma<fmaConfig, half>(dst, src0, src1, src2, sharedTmpBuffer, 512);
    ```

-   接口框架申请临时空间

    ```
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
    pipe.InitBuffer(tmpQue, 1, bufferSize); // bufferSize通过Host侧tiling参数获取
    // 输入tensor长度为1024,算子输入的数据类型为half,实际计算个数为512
    static constexpr AscendC::FmaConfig fmaConfig = {false}; // 不修改源操作数
    // dst、src0、src1、src2为half类型的LocalTensor
    AscendC::Fma<fmaConfig, half>(dst, src0, src1, src2, 512);
    ```

结果示例如下：

```
输入数据(src0):[1 1 1 ...... 1 1 1]
输入数据(src1):[2 2 2 ...... 2 2 2]
输入数据(src2):[1 1 1 ...... 1 1 1]
输出数据(dst):[3 3 3 ...... 3 3 3]
```
