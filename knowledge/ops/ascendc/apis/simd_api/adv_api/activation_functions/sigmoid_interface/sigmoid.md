---
type: API
title: Sigmoid
description: 按元素做逻辑回归Sigmoid计算。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.activation
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/activation_functions/Sigmoid_interface/Sigmoid.md
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
- include/adv_api/activation/sigmoid.h
---

# Sigmoid

## 功能说明

按元素做逻辑回归Sigmoid，计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002353088881.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000001539852417.png)

## 函数原型

-   通过sharedTmpBuffer入参传入临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
        ```

-   接口框架申请临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
        ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetSigmoidMaxMinTmpSize](get_sigmoid_max_min_tmp_size.md)中提供的接口获取需要预留空间的范围大小。

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
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>用于Sigmoid内部复杂计算时存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[GetSigmoidMaxMinTmpSize](get_sigmoid_max_min_tmp_size.md)。 |
| calCount | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。


-   针对Ascend 950PR&950DT系列产品，接口内部计算对Subnormal的处理方式受编译选项`--cce-ftz`控制（默认值为`true`）：
    -   配置为`false`时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    -   配置为`true`时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例

```
// dstLocal: 存放Sigmoid计算结果的Tensor
// srcLocal: 存放Sigmoid计算输入的Tensor
// sharedTmpBuffer: 存放Sigmoid计算过程中临时缓存的Tensor

// 接口框架申请临时空间，全部参与计算
AscendC::Sigmoid(dstLocal, srcLocal);
// 接口框架申请临时空间，部分参与计算,需要参与计算的元素个数为32
AscendC::Sigmoid(dstLocal, srcLocal, 32);

// 通过sharedTmpBuffer入参传入临时空间，全部参与计算
AscendC::Sigmoid(dstLocal, srcLocal, sharedTmpBuffer);
// 通过sharedTmpBuffer入参传入临时空间，部分参与计算,需要参与计算的元素个数为32
AscendC::Sigmoid(dstLocal, srcLocal, sharedTmpBuffer, 32);
```

结果示例如下：

```
输入数据(srcLocal):
[-100.    -50.    -20.    -10.     -5.     -4.     -3.     -2.
   -1.     -0.5    -0.4    -0.3    -0.2    -0.1    -0.01    0.
    0.      0.01    0.1     0.2     0.3     0.4     0.5     1.
    2.      3.      4.      5.     10.     20.     50.    100.  ]
输出数据(dstLocal):
[0.         0.         0.         0.0000454  0.00669285 0.01798621 0.04742587 0.11920292
 0.26894142 0.37754067 0.40131234 0.42555748 0.450166   0.47502081 0.49750002 0.5
 0.5        0.50249998 0.52497919 0.549834   0.57444252 0.59868766 0.62245933 0.73105858
 0.88079708 0.95257413 0.98201379 0.99330715 0.9999546  1.         1.         1.        ]
```

- [LSTM A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/rnn/lstm.md)
- [ThnnFusedLstmCell A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/rnn/thnn_fused_lstm_cell.md)
- [LSTM 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/rnn/lstm_950.md)
- [ForeachSigmoid A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_sigmoid.md)
