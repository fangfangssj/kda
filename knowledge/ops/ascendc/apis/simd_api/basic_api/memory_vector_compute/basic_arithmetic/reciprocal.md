---
type: API
title: Reciprocal
description: 单目矢量计算接口，对输入tensor按元素取倒数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Reciprocal.md
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

# Reciprocal<a name="ZH-CN_TOPIC_0000001429443329"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_unary_intf.h"`。

Reciprocal属于单目矢量类计算接口，负责将输入的tensor按元素取倒数。计算公式如下：

$dst_i = \frac{1}{src_i}$

## 函数原型<a name="section620mcpsimp"></a>

- 传入config的原型

    - tensor前n个数据连续计算

        ```cpp
        template <typename T, const ReciprocalConfig& config = DEFAULT_RECIPROCAL_CONFIG>
        __aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
        ```

    - tensor高维切分计算
        - mask逐bit模式

            ```cpp
            template <typename T, bool isSetMask = true, const ReciprocalConfig& config = DEFAULT_RECIPROCAL_CONFIG>
            __aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
            ```

        - mask连续模式

            ```cpp
            template <typename T, bool isSetMask = true, const ReciprocalConfig& config = DEFAULT_RECIPROCAL_CONFIG>
            __aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
            ```

- 不传入config的原型

    - tensor前n个数据连续计算

        ```cpp
        template <typename T>
        __aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
        ```

    - tensor高维切分计算
        - mask逐bit模式

            ```cpp
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
            ```

        - mask连续模式

            ```cpp
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
            ```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
|---|---|
| T | 操作数数据类型。 |
| isSetMask | 是否在接口内部设置mask。<br>&bull; true，表示在接口内部设置mask。<br>&bull; false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](../mask_operations/set_vector_mask.md)接口设置mask值。这种模式下，接口入参中的mask值设置为占位符`MASK_PLACEHOLDER`，用于占位，无实际含义。 |
|  |

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


- 针对Ascend 950PR&950DT系列产品，T支持的数据类型为：half、float、int64_t、uint64_t。数据类型int64_t、uint64_t仅支持tensor前n个数据计算接口。


- 针对Atlas A3系列产品，T支持的数据类型为：half、float。


- 针对Atlas A2系列产品，T支持的数据类型为：half、float。


- 针对Atlas 200I/500 A2推理产品，T支持的数据类型为：half、float。


- 针对Atlas推理系列产品AI Core，T支持的数据类型为：half、float。


- 针对Atlas训练系列产品，T支持的数据类型为：half、float。


- 针对Kirin X90处理器系列产品，T支持的数据类型为：half、float。


- 针对Kirin 9030处理器系列产品，T支持的数据类型为：half、float。


## 返回值说明<a name="section15227195416231"></a>

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


- 如果src中的数值为0，可能会产生未知结果。
- 使用Reciprocal时，half的算子结果对比误差不满足双千分之一的要求，float的算子结果对比误差不满足双万分之一的要求，如果需要高精度，建议使用[Div](div.md)替代实现。



## 关键特性<a id="reciprocal-key-features"></a>

针对Ascend 950PR&950DT系列产品，有如下关键特性：

### 最大精度误差

ReciprocalAlgo::INTRINSIC、ReciprocalAlgo::PRECISION_1ULP_FTZ_TRUE和ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE的最大精度误差均为1ulp。

### 配置Subnormal模式

FTZ（Flush-To-Zero）：一种浮点运算模式，当结果为[Subnormal](../../data_structures/builtin_data_types.md#p7381131713310)时，将其直接清零（近似为0），而非保留其精确的微小数值。

#### 默认算法

ReciprocalAlgo::INTRINSIC为默认算法。--cce-ftz=false时保留Subnormal；--cce-ftz=true（默认值）时采用FTZ模式。

#### 显式指定的算法

- ReciprocalAlgo::PRECISION_1ULP_FTZ_TRUE使用单指令计算，始终采用FTZ模式。
- ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE通过软件仿真实现，支持Subnormal数据计算。

#### 使用建议

由于保留Subnormal的计算行为通过软件仿真实现，在--cce-ftz=true（默认值）时，一般场景建议使用默认的ReciprocalAlgo::INTRINSIC或显式选择ReciprocalAlgo::PRECISION_1ULP_FTZ_TRUE，以获得更好的性能；需要精确输出Subnormal时，使用ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE。


## 调用示例<a name="section176061616102911"></a>

- tensor高维切分计算样例-mask连续模式

    ```cpp
    uint64_t mask = 256 / sizeof(half);
    // repeatTime = 4, 128 elements one repeat, 512 elements total.
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat.
    // dstRepStride, srcRepStride = 8, no gap between repeats.
    AscendC::Reciprocal(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
    ```

- tensor高维切分计算样例-mask逐bit模式

    ```cpp
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // repeatTime = 4, 128 elements one repeat, 512 elements total.
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat.
    // dstRepStride, srcRepStride = 8, no gap between repeats.
    AscendC::Reciprocal(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
    ```

- tensor前n个数据计算接口样例

    ```cpp
    AscendC::Reciprocal(dstLocal, srcLocal, 512);
    ```

    
    以下示例仅支持Ascend 950PR&950DT系列产品

    ```cpp
    static constexpr ReciprocalConfig config = { ReciprocalAlgo::PRECISION_1ULP_FTZ_FALSE };
    AscendC::Reciprocal<T, config>(dstLocal, srcLocal, 512);
    ```
    

结果示例如下：

```bash
输入数据srcLocal：[-7.152 -7.24 1.771 ... -1.339 4.473]
输出数据dstLocal：[-0.1396 -0.1382 0.5645 ... -0.748 0.2231]
```
