---
type: API
title: Log
description: 对输入张量按元素以e、2或10为底进行对数运算，支持half和float数据类型。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/Log_interface/Log.md
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
- include/adv_api/math/log.h
---

# Log

## 功能说明

按元素以e、2、10为底做对数运算，计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002182888036.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002188813710.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002218328397.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002188654338.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002182888172.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002188814822.png)

## 函数原型

-   以e为底：

    -   源操作数Tensor全部/部分参与计算

    ```
    template<typename T, bool isReuseSource = false>
    __aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
    ```

    -   源操作数Tensor全部参与计算

    ```
    template<typename T, bool isReuseSource = false>
    __aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
    ```

-   以2为底
    -   通过sharedTmpBuffer入参传入临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template<typename T, bool isReuseSource = false>
            __aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false>
            __aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
            ```

    -   接口框架申请临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template<typename T, bool isReuseSource = false>
            __aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false>
            __aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
            ```

-   以10为底：
    -   源操作数Tensor全部/部分参与计算

        ```
        template<typename T, bool isReuseSource = false>
        __aicore__ inline void Log10(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template<typename T, bool isReuseSource = false>
        __aicore__ inline void Log10(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
        ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetLogMaxMinTmpSize](get_log_max_min_tmp_size.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>临时空间大小BufferSize的获取方式请参考[GetLogMaxMinTmpSize](get_log_max_min_tmp_size.md)。 |
| calCount | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。


-   针对Ascend 950PR&950DT系列产品，接口内部计算对Subnormal的处理方式受编译选项`--cce-ftz`控制（默认值为`true`）：
    -   配置为`false`时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    -   配置为`true`时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例

完整的调用样例可参考[Log样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/04_advanced_api/10_math/log)。

```
// dstLocal: 存放计算结果的Tensor
// srcLocal: 参与计算的输入Tensor

// 以e为底，所有元素全部参与计算
AscendC::Log(dstLocal, srcLocal);
// 以10为底
// AscendC::Log10(dstLocal, srcLocal);
// 以2为底
// AscendC::Log2(dstLocal, srcLocal);
```

Log接口结果示例如下：

```
输入数据(srcLocal): [144.22607 9634.764 ... 1835.1245 3145.5125]
输出数据(dstLocal): [4.971382 9.173133 ... 7.514868 8.053732]
```

Log2接口结果示例如下：

```
输入数据(srcLocal): [6299.54 338.45963 ... 2.853525 5752.1323]
输出数据(dstLocal): [12.621031 8.40284 ... 1.5127451 12.4898815]
```

Log10接口结果示例如下：

```
输入数据(srcLocal): [712.7535 78.36265 ... 3099.0571 9313.082]
输出数据(dstLocal): [2.8529394 1.8941091 ... 3.4912295 3.9690933]
```

- [SimThreadExponential A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/random/sim_thread_exponential.md)
- [CrossEntropyLoss A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/loss/cross_entropy_loss.md)
