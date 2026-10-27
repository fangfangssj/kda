---
type: API
title: Abs
description: 单目矢量计算接口，对输入tensor按元素取绝对值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Abs.md
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
- include/basic_api/kernel_operator_vec_unary_intf.h
---

# Abs<a name="ZH-CN_TOPIC_0000001429123797"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_unary_intf.h"`。

Abs属于单目矢量类计算接口，负责将输入的tensor按元素取绝对值。计算公式如下：

$dst_i = |src_i|$

## 函数原型<a name="section620mcpsimp"></a>

- tensor前n个数据连续计算

    ```cpp
    template <typename T>
    __aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
    ```


- tensor前n个复数计算

    ```cpp
    // 该函数原型，仅支持Ascend 950PR&950DT系列产品。
    template <typename T, typename U>
    __aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t& count)
    ```


- tensor高维切分计算
    - mask逐bit模式

        ```cpp
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    - mask连续模式

        ```cpp
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
|---|---|
| T | 操作数数据类型。 |
|  |
| isSetMask | 是否在接口内部设置mask。<br>&bull; true，表示在接口内部设置mask。<br>&bull; false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](../mask_operations/set_vector_mask.md)接口设置mask值。这种模式下，接口入参中的mask值设置为占位符`MASK_PLACEHOLDER`，用于占位，无实际含义。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| dst | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>地址对齐约束参考[通用地址对齐约束](../../../general_description_and_constraints.md)。 |
| src | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>地址对齐约束参考[通用地址对齐约束](../../../general_description_and_constraints.md)。 |
| count | 输入 | 参与计算的元素个数。关于该参数的具体说明请参考[连续计算](../simd_compute/continuous_compute.md)。 |
| mask[]/mask | 输入 | mask用于控制每次迭代内参与计算的元素。详细设置参考[掩码](../simd_compute/mask.md)。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。<br>关于该参数的具体说明请参考[高维切分](../simd_compute/high_dim_split.md)。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。[UnaryRepeatParams](../../aux_data_structures/unary_repeat_params.md)类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。<br>相邻迭代间的地址步长参数说明请参考[repeatStride](../simd_compute/high_dim_split.md)；同一迭代内DataBlock的地址步长参数说明请参考[dataBlockStride](../simd_compute/high_dim_split.md)。 |

## 数据类型


- 针对Ascend 950PR&950DT系列产品
    - T支持的数据类型为：int8_t、int16_t、half、int32_t、float、int64_t。数据类型int8_t、int64_t仅支持tensor前n个数据计算接口。
    - U支持的数据类型为：complex32、complex64；当U为complex32类型时，T必须是half类型；当U为complex64类型时，T必须是float类型。


- 针对Atlas A3系列产品，T支持的数据类型为：half、float。


- 针对Atlas A2系列产品，T支持的数据类型为：half、float。


- 针对Atlas 200I/500 A2推理产品，T支持的数据类型为：int16_t、half、float。


- 针对Atlas推理系列产品AI Core，T支持的数据类型为：half、float。


- 针对Atlas训练系列产品，T支持的数据类型为：half、float。


- 针对Kirin X90处理器系列产品，T支持的数据类型为：half、float。


- 针对Kirin 9030处理器系列产品，T支持的数据类型为：half、float。


## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

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


- 针对Ascend 950PR&950DT系列产品，仅complex32和complex64类型的接口内部计算涉及Subnormal，处理方式受编译选项--cce-ftz控制（默认值为true）：
    - --cce-ftz=false时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    - --cce-ftz=true时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例<a name="section176061616102911"></a>

- tensor高维切分计算样例-mask连续模式

    ```cpp
    uint64_t mask = 256 / sizeof(half);
    // repeatTime = 4, 128 elements one repeat, 512 elements total.
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat.
    // dstRepStride, srcRepStride = 8, no gap between repeats.
    AscendC::Abs(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
    ```

    结果示例如下：

    ```bash
    输入数据srcLocal:[0.0 -1.0 2.0 -3.0 ...]
    输出数据dstLocal：[0.0 1.0 2.0 3.0 ...]
    ```

- tensor高维切分计算样例-mask逐bit模式

    ```cpp
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // repeatTime = 4, 128 elements one repeat, 512 elements total.
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat.
    // dstRepStride, srcRepStride = 8, no gap between repeats.
    AscendC::Abs(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
    ```

    结果示例如下：

    ```bash
    输入数据srcLocal：[0.0 -1.0 2.0 -3.0 ...]
    输出数据dstLocal：[0.0 1.0 2.0 3.0 ...]
    ```

- tensor前n个数据计算接口样例

    ```cpp
    AscendC::Abs(dstLocal, srcLocal, 512);
    ```

    结果示例如下：

    ```bash
    输入数据srcLocal：[0.0 -1.0 2.0 -3.0 ...]
    输出数据dstLocal：[0.0 1.0 2.0 3.0 ...]
    ```


针对Ascend 950PR&950DT系列产品：

- tensor前n个复数计算接口样例

    ```cpp
    LocalTensor<complex64> srcLocal = inQueue.DeQue<complex64>();
    LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
    AscendC::Abs(dstLocal, srcLocal, 512);
    ```

    结果示例如下：

    ```bash
    输入数据srcLocal：[1+2j  2-3j ...]
    输出数据dstLocal：[2.236 3.605 ...]
    ```

- [AddRmsNormDynamicQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_quant_950.md)
- [ApplyAdamWQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/optim/apply_adam_w_quant.md)
- [AddRmsNormDynamicQuant 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/norm/add_rms_norm_dynamic_quant_950.md)
