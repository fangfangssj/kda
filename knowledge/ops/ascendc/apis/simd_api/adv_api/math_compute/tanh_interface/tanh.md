---
type: API
title: Tanh
description: 按元素做双曲正切函数计算，支持直接计算和误差补偿两种算法。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/Tanh_interface/Tanh.md
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
- include/adv_api/math/tanh.h
---

# Tanh

## 功能说明

按元素做逻辑回归Tanh，计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002218327685.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002193334172.png)

## 函数原型

-   通过sharedTmpBuffer入参传入临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false, const TanhConfig& config = DEFAULT_TANH_CONFIG>
        __aicore__ inline void Tanh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false, const TanhConfig& config = DEFAULT_TANH_CONFIG>
        __aicore__ inline void Tanh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
        ```

-   接口框架申请临时空间

    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false, const TanhConfig& config = DEFAULT_TANH_CONFIG>
        __aicore__ inline void Tanh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false, const TanhConfig& config = DEFAULT_TANH_CONFIG>
        __aicore__ inline void Tanh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
        ```

    由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

    -   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
    -   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

    通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetTanhMaxMinTmpSize](get_tanh_max_min_tmp_size.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |
| config | Tanh算法的相关配置。此参数可选配，TanhConfig类型，具体定义如下方代码所示，其中参数的含义为：<br>algo：Tanh内部实现使用的算法。TanhAlgo类型，支持的取值如下：INTRINSIC：默认值。该算法直接计算公式y = (e^(2x)-1)/(e^(2x)+1)，性能更好。SUBSECTION_COMPENSATION：该算法通过误差补偿实现Tanh，精度更高。 |

```
enum class TanhAlgo {
    INTRINSIC = 0,
    SUBSECTION_COMPENSATION,
};
struct TanhConfig {
    TanhAlgo algo = TanhAlgo::INTRINSIC;
};
```

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>用于Tanh内部复杂计算时存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[GetTanhMaxMinTmpSize](get_tanh_max_min_tmp_size.md)。 |
| calCount | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。


-   针对Ascend 950PR&950DT系列产品，接口内部计算对Subnormal的处理方式受编译选项`--cce-ftz`控制（默认值为`true`）：
    -   配置为`false`时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    -   配置为`true`时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例

```
// dstLocal: 存放Tanh计算结果的Tensor
// srcLocal: 存放Tanh计算输入的Tensor
// sharedTmpBuffer: 存放Tanh计算过程中临时缓存的Tensor

// 接口框架申请临时空间，全部参与计算
AscendC::Tanh(dstLocal, srcLocal);
// 接口框架申请临时空间，部分参与计算,需要参与计算的元素个数为512
AscendC::Tanh(dstLocal, srcLocal, 512);

// 通过sharedTmpBuffer入参传入临时空间，全部参与计算
AscendC::Tanh(dstLocal, srcLocal, sharedTmpBuffer);
// 通过sharedTmpBuffer入参传入临时空间，部分参与计算,需要参与计算的元素个数为512
AscendC::Tanh(dstLocal, srcLocal, sharedTmpBuffer, 512);
static constexpr AscendC::TanhAlgo algo = AscendC::TanhAlgo::SUBSECTION_COMPENSATION;
static constexpr AscendC::TanhConfig config = {algo};
AscendC::Tanh<half, false, config>(dstLocal, srcLocal, sharedTmpBuffer, 512);
```

结果示例如下：

```
输入数据(srcLocal):
[-2.56 -2.55 -2.54 ... 0. ... 2.53  2.54  2.55]
输出数据(dstLocal):
[-0.98813187 -0.9878992 -0.98761402 ... 0. ... 0.98737127 0.98761402 0.9878992]
```

- [MishBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/mish_backward.md)
- [LSTM A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/rnn/lstm.md)
- [ThnnFusedLstmCell A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/rnn/thnn_fused_lstm_cell.md)
- [ThnnFusedLstmCellBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/rnn/thnn_fused_lstm_cell_backward.md)
- [LSTM 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/rnn/lstm_950.md)
- [ForeachTanh A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tanh.md)
