---
type: API
title: ShiftRight
description: 对源操作数中的每个元素进行右移操作，右移位数由标量scalarValue决定，无符号类型执行逻辑右移，有符号类型执行算术右移，支持舍入功能。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/logical_compute/ShiftRight.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_vec_binary_scalar_intf.h
---

# ShiftRight<a name="ZH-CN_TOPIC_0000001475200925"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_binary_scalar_intf.h"`。

对源操作数中的每个元素进行右移操作，右移的位数由标量scalarValue决定。根据源操作数的数据类型，右移操作分为以下两种情况：

- 数据类型为无符号类型：执行逻辑右移。逻辑右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位用0填充。例如，二进制数1010101010101010（uint16\_t类型）逻辑右移1位后，结果为0101010101010101。
- 数据类型为有符号类型：执行算术右移。算术右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位复制符号位。例如，二进制数1010101010101010（int16\_t类型）算术右移1位后，结果为1101010101010101；算术右移3位后，结果为1111010101010101。

计算公式如下：

$dst_i = src_i \gg scalarValue$

## 函数原型<a name="section620mcpsimp"></a>

- tensor前n个数据连续计算

    ```cpp
    template <typename T, bool isSetMask = true>
    __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count)
    ```

- tensor高维切分计算
    - mask逐bit模式

        ```cpp
        template <typename T, bool isSetMask = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false)
        ```

    - mask连续模式

        ```cpp
        template <typename T, bool isSetMask = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false)
        ```

dst和src使用[TensorTrait](../../aux_data_structures/tensor_trait/index.md)类型时，其数据类型TensorTrait和scalarValue的数据类型（对应TensorTrait中的LiteType类型）不一致。因此新增模板类型U表示scalarValue的数据类型，并通过std::enable\_if检查T中萃取出的LiteType和U是否完全一致，一致则接口通过编译，否则编译失败。接口原型定义如下：

- tensor前n个数据计算

    ```cpp
    template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
    __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count)
    ```

- tensor高维切分计算
    - mask逐bit模式

        ```cpp
        template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
        ```

    - mask连续模式

        ```cpp
        template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
        ```

## 参数说明<a name="section1436019411811"></a>

### 模板参数及接口参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
|---|---|
| T | 操作数数据类型。 |
| U | scalarValue的数据类型。 |
| isSetMask | 是否在接口内部设置mask。<br>&bull; true，表示在接口内部设置mask。<br>&bull; false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](../mask_operations/set_vector_mask.md)接口设置mask值。这种模式下：<br>&nbsp;&nbsp;&bull; 针对tensor前n个数据计算接口，接口入参中的count不生效，建议设置成1。<br>&nbsp;&nbsp;&bull; 针对tensor高维切分计算接口，接口入参中的mask值设置为占位符`MASK_PLACEHOLDER`，用于占位，无实际含义。<br>  |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| dst | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>地址对齐约束参考[通用地址对齐约束](../../../general_description_and_constraints.md)。 |
| src | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>地址对齐约束参考[通用地址对齐约束](../../../general_description_and_constraints.md)。<br>源操作数的数据类型需要与目的操作数保持一致。 |
| scalarValue | 输入 | 右移的位数，数据类型需要与目的操作数中的元素数据类型保持一致。具体取值说明请参考下面的[scalarValue取值说明](#scalarvalue取值说明)。 |
| count | 输入 | 参与计算的元素个数。关于该参数的具体说明请参考[连续计算](../simd_compute/continuous_compute.md)。 |
| mask[]/mask | 输入 | mask用于控制每次迭代内参与计算的元素。详细设置参考[掩码](../simd_compute/mask.md)。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。<br>关于该参数的具体说明请参考[高维切分](../simd_compute/high_dim_split.md)。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。[UnaryRepeatParams](../../aux_data_structures/unary_repeat_params.md)类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。<br>相邻迭代间的地址步长参数说明请参考[repeatStride](../simd_compute/high_dim_split.md)；同一迭代内DataBlock的地址步长参数说明请参考[dataBlockStride](../simd_compute/high_dim_split.md)。 |
| roundEn | 输入 | 舍入功能开启开关，支持数据类型：bool，true为开启，false为不开启。仅当src为int16\_t/int32\_t类型时开启有效。<br>例：开启舍入功能，src数据类型为int16\_t，将src算术右移5位，如果src\_ele二进制数中的第5位为1，则dst\_ele值为对src\_ele算术右移5后加1。<br>src\_ele = 17 = 0b00000000000**1**0001第五位为1<br>dst\_ele = arithmetic\_right\_shift(src\_ele, 5) + 1 = 0b00000000000**00000** + 1 = 0b00000000000**00001**<br> |

### scalarValue取值说明


- 针对Ascend 950PR&950DT系列产品，scalarValue的取值应大于等于0。逻辑右移的情况下，如果右移的位数大于src数据类型位宽，dst的全部元素被赋值为0；算术右移的情况下，如果右移的位数大于src数据类型位宽，源操作数元素为正数，对应的目的操作数元素被赋值为0，源操作数元素为负数，对应的目的操作数元素被赋值为-1。


- 针对Atlas A3系列产品，当src为uint16_t或int16_t类型时，scalarValue的取值范围为[0, 16]；当src为uint32_t或int32_t类型时，scalarValue的取值范围为[0, 32]。


- 针对Atlas A2系列产品，当src为uint16_t或int16_t类型时，scalarValue的取值范围为[0, 16]；当src为uint32_t或int32_t类型时，scalarValue的取值范围为[0, 32]。


- 针对Atlas 200I/500 A2推理产品，当src为uint16_t或int16_t类型时，scalarValue的取值范围为[0, 16]；当src为uint32_t或int32_t类型时，scalarValue的取值范围为[0, 32]。


## 数据类型


- 针对Ascend 950PR&950DT系列产品，T和U支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。数据类型int8_t、uint8_t、int64_t、uint64_t仅支持tensor前n个数据计算接口。


- 针对Atlas A3系列产品，T和U支持的数据类型为：int16_t、uint16_t、int32_t、uint32_t。


- 针对Atlas A2系列产品，T和U支持的数据类型为：int16_t、uint16_t、int32_t、uint32_t。


- 针对Atlas 200I/500 A2推理产品，T和U支持的数据类型为：int16_t、uint16_t、int32_t、uint32_t。


## 返回值说明<a name="section194321251175110"></a>

无

## 约束说明<a name="section23625411480"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md)。

- 当参数count或repeatTime取值为0时，该接口的行为如下：
  
  - 针对如下型号，该接口不会执行计算操作，不会对目的操作数进行写入，该接口将被视为NOP（空操作）。
    
    - Atlas A3系列产品
    
    
    - Atlas A2系列产品
    
  
  
  - 针对Ascend 950PR&950DT系列产品，该接口通过VF调用[Reg矢量计算API](../../reg_vector_compute/index.md)实现兼容，当参数count或repeatTime取值为0时，不保证该接口将被视为NOP（空操作）。
  


- 针对Ascend 950PR&950DT系列产品，tensor前n个数据计算API中的isSetMask参数不生效，保持默认值即可。


## 调用示例<a name="section642mcpsimp"></a>

- tensor高维切分计算样例-mask连续模式

    ```cpp
    uint64_t mask = 128;
    int16_t scalar = 2;
    // repeatTime = 4, 128 elements one repeat, 512 elements total.
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat.
    // dstRepStride, srcRepStride = 8, no gap between repeats.
    AscendC::ShiftRight(dstLocal, srcLocal, scalar, mask, 4, { 1, 1, 8, 8 }, false);
    ```

- tensor高维切分计算样例-mask逐bit模式

    ```cpp
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    int16_t scalar = 2;
    // repeatTime = 4, 128 elements one repeat, 512 elements total.
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat.
    // dstRepStride, srcRepStride = 8, no gap between repeats.
    AscendC::ShiftRight(dstLocal, srcLocal, scalar, mask, 4, {1, 1, 8, 8}, false);
    ```

- tensor前n个数据计算样例

    ```cpp
    int16_t scalar = 2;
    AscendC::ShiftRight(dstLocal, srcLocal, scalar, 512);
    ```

结果示例如下：

```plain
输入数据srcLocal: [1 2 3 ... 512]
输入数据scalar = 2
输出数据dstLocal: [0 0 0 1 1 1 1 ... 128]
```
