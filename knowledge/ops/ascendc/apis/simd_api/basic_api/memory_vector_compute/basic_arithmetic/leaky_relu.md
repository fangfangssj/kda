---
type: API
title: LeakyRelu
description: 矢量元素执行Leaky ReLU操作：非负值原样输出，负值乘以标量系数后输出。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/LeakyRelu.md
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
- include/basic_api/kernel_operator_vec_binary_scalar_intf.h
---

# LeakyRelu<a name="ZH-CN_TOPIC_0000001475000405"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_binary_scalar_intf.h"`。

LeakyRelu属于双目标量类计算接口，输入为一个矢量tensor和一个标量scalar，LeakyRelu负责将矢量tensor内的每个元素执行Leaky ReLu（Leaky Rectified Linear Unit）操作。计算公式如下：

$dst_i = \begin{cases} src_i, & src_i \geq 0 \\ scalarValue \times src_i, & src_i < 0 \end{cases}$

Leaky ReLU带泄露线性整流函数是一种人工神经网络中常用的激活函数，其数学表达式为：

$f(x) = \begin{cases} x, & x \geq 0 \\ \alpha x, & x < 0 \end{cases}$

其中 $\alpha$ 为负斜率（对应参数scalarValue）。

和ReLU的区别是：ReLU是将所有的负值都设为零，而Leaky ReLU是给所有负值赋予一个斜率。下图表示了ReLU和Leaky ReLU的区别：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/image_0000001475211729.png) ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/image_0000001425411698.png)

## 函数原型<a name="section620mcpsimp"></a>

- tensor前n个数据计算

    ```cpp
    template <typename T, bool isSetMask = true>
    __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count)
    ```

- tensor高维切分计算
    - mask逐bit模式

        ```cpp
        template <typename T, bool isSetMask = true>
        __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    - mask连续模式

        ```cpp
        template <typename T, bool isSetMask = true>
        __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

dst和src使用[TensorTrait](../../aux_data_structures/tensor_trait/index.md)类型时，其数据类型TensorTrait和scalarValue的数据类型（对应TensorTrait中的LiteType类型）不一致。因此新增模板类型U表示scalarValue的数据类型，并通过std::enable\_if检查T中萃取出的LiteType和U是否完全一致，一致则接口通过编译，否则编译失败。接口原型定义如下：

- tensor前n个数据计算

    ```cpp
    template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
    __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count)
    ```

- tensor高维切分计算
    - mask逐bit模式

        ```cpp
        template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    - mask连续模式

        ```cpp
        template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section166351181919"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
|---|---|
| T | 操作数数据类型。 |
| U | scalarValue的数据类型。仅在dst和src使用[TensorTrait](../../aux_data_structures/tensor_trait/index.md)类型时有效。 |
| isSetMask | 是否在接口内部设置mask。<br>&bull; true，表示在接口内部设置mask。<br>&bull; false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](../mask_operations/set_vector_mask.md)接口设置mask值。这种模式下：<br>&nbsp;&nbsp;&bull; 针对tensor前n个数据计算接口，接口入参中的count不生效，建议设置成1。<br>&nbsp;&nbsp;&bull; 针对tensor高维切分计算接口，接口入参中的mask值设置为占位符`MASK_PLACEHOLDER`，用于占位，无实际含义。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| dst | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>地址对齐约束参考[通用地址对齐约束](../../../general_description_and_constraints.md)。 |
| src | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>地址对齐约束参考[通用地址对齐约束](../../../general_description_and_constraints.md)。 |
| scalarValue | 输入 | 源操作数，数据类型需要与目的操作数中的元素类型保持一致。 |
| count | 输入 | 参与计算的元素个数。关于该参数的具体说明请参考[连续计算](../simd_compute/continuous_compute.md)。 |
| mask[]/mask | 输入 | mask用于控制每次迭代内参与计算的元素。详细设置参考[掩码](../simd_compute/mask.md)。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。<br>关于该参数的具体说明请参考[高维切分](../simd_compute/high_dim_split.md)。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。[UnaryRepeatParams](../../aux_data_structures/unary_repeat_params.md)类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。<br>相邻迭代间的地址步长参数说明请参考[repeatStride](../simd_compute/high_dim_split.md)；同一迭代内DataBlock的地址步长参数说明请参考[dataBlockStride](../simd_compute/high_dim_split.md)。 |

## 数据类型<a name="zh-cn_topic_0000002530007180_section4219135304818"></a>

T和U支持的数据类型为：half、float。

## 返回值说明<a name="section194321251175110"></a>

无

## 约束说明<a name="section1636318590"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md)。


- 当参数count或repeatTime取值为0时，该接口的行为如下：
  
  - 针对如下型号，当参数count或repeatTime取值为0时，不会执行计算操作，不会对目的操作数进行写入，该接口将被视为NOP（空操作）。
    
    - Atlas A3系列产品
    
    
    - Atlas A2系列产品
    
  
  
  - 针对Ascend 950PR&950DT系列产品：该接口通过VF调用[Reg矢量计算API](../../reg_vector_compute/index.md)实现兼容，当参数count或repeatTime取值为0时，软仿行为不保证该接口被视为NOP（空操作）。
  


- 对Unified Buffer（UB）空间的占用说明。针对Ascend 950PR&950DT系列产品：
  - tensor高维切分计算占用8KB UB。
  - tensor前n个数据连续计算不涉及8KB UB的占用。
- 针对Ascend 950PR&950DT系列产品，tensor前n个数据计算API中的isSetMask参数不生效，保持默认值即可。


## 调用示例<a name="section112161084424"></a>

- tensor高维切分计算样例-mask连续模式

    ```cpp
    uint64_t mask = 128;
    half scalar = 0.001;
    AscendC::LeakyRelu(dstLocal, srcLocal, scalar, mask, 4, {1, 1, 8, 8});
    ```

- tensor高维切分计算样例-mask逐bit模式

    ```cpp
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    half scalar = 0.001;
    AscendC::LeakyRelu(dstLocal, srcLocal, scalar, mask, 4, {1, 1, 8, 8});
    ```

- tensor前n个数据计算样例

    ```cpp
    half scalar = 0.001;
    AscendC::LeakyRelu(dstLocal, srcLocal, scalar, 512);
    ```

结果示例如下：

```bash
输入数据src0Local：[-287.  246. -438.  177. 596. -950.  -293.  322. ... -900.]
输入数据scalar = 0.001
输出数据dstLocal：[-0.287 246. -0.438 177. 596. -0.950 -0.293 322. ... -0.900]
```
