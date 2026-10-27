---
type: API
title: Axpy
description: 计算向量和标量的乘积，并将乘积结果逐元素加到的输出向量上。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.compute.composite
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/composite_compute/Axpy.md
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
- include/basic_api/kernel_operator_vec_ternary_scalar_intf.h
---

# Axpy<a name="ZH-CN_TOPIC_0000001424952592"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_ternary_scalar_intf.h"`。

向量和标量的乘积，并将乘积结果逐元素加到的输出向量上。计算公式如下：

$$
dst_j = scalar × src_i + dst_i
$$

## 函数原型<a name="section620mcpsimp"></a>

- tensor前n个数据连续计算

  ```cpp
  template <typename T, typename U>
  __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, const int32_t& count)
  ```

- tensor高维切分计算
  - mask逐bit模式

    ```cpp
    template <typename T, typename U, bool isSetMask = true>
    __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    ```

  - mask连续模式

    ```cpp
    template <typename T, typename U, bool isSetMask = true>
    __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表**  模板参数说明

| 参数名 | 描述 |
| :----- | :--- |
| T | 目的操作数数据类型。 |
| U | 源操作数数据类型。 |
| isSetMask | 是否在接口内部设置mask。<br>&bull; true，表示在接口内部设置mask。<br>&bull; false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](../mask_operations/set_vector_mask.md)接口设置mask值。这种模式下，接口入参中的mask值设置为占位符`MASK_PLACEHOLDER`，用于占位，无实际含义。 |

**表**  参数说明

| 参数名称     | 输入/输出 | 说明 |
| :----------- | :-------- | :--- |
| dst          | 输出      | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src          | 输入      | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| scalarValue  | 输入      | 源操作数，scalar标量。scalarValue的数据类型需要和src保持一致。 |
| count        | 输入      | 参与计算的元素个数。关于该参数的具体说明请参考[连续计算](../simd_compute/continuous_compute.md)。 |
| mask/mask[]  | 输入      | mask用于控制每次迭代内参与计算的元素。<br>设置详见[掩码操作](../simd_compute/mask.md) |
| repeatTime   | 输入      | 重复迭代次数。<br>矢量计算单元，每次读取连续的256字节数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。<br>关于该参数的具体描述请参考[高维切分API](../simd_compute/high_dim_split.md)。 |
| repeatParams | 输入      | 控制操作数地址步长的参数。[UnaryRepeatParams](../../aux_data_structures/unary_repeat_params.md)类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。<br>相邻迭代间的地址步长参数说明请参考[repeatStride](../simd_compute/high_dim_split.md#地址间隔配置)；同一迭代内DataBlock的地址步长参数说明请参考[dataBlockStride](../simd_compute/high_dim_split.md#地址间隔配置)。 |

## 数据类型<a name="zh-cn_topic_0000002547893099_section4219135304818"></a>

PAR列表示矢量计算单元一个迭代能够处理的元素个数。



**表**  Ascend 950PR&950DT系列产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |
| int64_t     | int64_t        | int64_t     | 64  |
| uint64_t    | uint64_t       | uint64_t    | 64  |
| bfloat16_t  | bfloat16_t     | bfloat16_t  | 128 |

针对Ascend 950PR&950DT系列产品，int64_t、uint64_t数据类型仅支持tensor前n个数据计算接口。





**表** Atlas A3系列产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |





**表** Atlas A2系列产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |





**表** Atlas 200I/500 A2推理产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |





**表** Atlas推理系列产品AI Core支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |





**表** Atlas训练系列产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |





**表** Kirin X90处理器系列产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |





**表** Kirin 9030处理器系列产品支持的数据类型组合情况

| src数据类型 | scalarValue数据类型 | dst数据类型 | PAR |
| ----------- | -------------- | ----------- | --- |
| half        | half           | half        | 128 |
| float       | float          | float       | 64  |
| half        | half           | float       | 64  |



## 返回值说明<a name="section17124037164714"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md#section668772811100)。
- 源操作数的数据类型为half、目的操作数的数据类型为float的情况下，不支持地址重叠。
- 使用tensor高维切分计算接口时，src和scalarValue的数据类型为half、dst的数据类型为float的情况下，一个迭代处理内最多处理64个输入数据。

- 当参数count或repeatTime取值为0时，该接口的行为如下：
  
  - 针对如下型号，该接口不会执行计算操作，不会对目的操作数进行写入，该接口将被视为NOP（空操作）。
    
    - Atlas A3系列产品
    
    
    - Atlas A2系列产品
    
  
  
  - 针对Ascend 950PR&950DT系列产品，该接口通过VF调用[Reg矢量计算API](../../reg_vector_compute/index.md)实现兼容，当参数count或repeatTime取值为0时，不保证该接口将被视为NOP（空操作）。
  


- 对Unified Buffer（UB）空间的占用说明。针对Ascend 950PR&950DT系列产品：
  - tensor高维切分计算接口占用8KB UB临时空间。
  - tensor前n个数据连续计算接口不涉及8KB UB临时空间的占用。


## 调用示例<a name="section642mcpsimp"></a>

本样中只展示部分关键代码。如果您需要运行样例代码，请参考完整样例[复合计算样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/element_wise_compound_compute)。

- tensor高维切分计算样例-mask连续模式。

  ```cpp
  // repeatTime = 4, mask = 128, 128 elements one repeat, 512 elements total
  // srcLocal数据类型为half，scalarValue数据类型为half，dstLocal数据类型为half
  // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
  // dstRepStride, srcRepStride = 8, no gap between repeats 
  AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 128, 4,{ 1, 1, 8, 8 });
  
  // srcLocal数据类型为half，scalarValue数据类型为half，dstLocal数据类型为float
  // repeatTime = 8, mask = 64, 64 elements one repeat, 512 elements total
  // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
  // dstRepStride = 8, srcRepStride = 4, no gap between repeats 
  AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 64, 8,{ 1, 1, 8, 4 }); // 每次迭代选取源操作数前4个datablock参与计算
  ```

- tensor高维切分计算样例-mask逐bit模式。

  ```cpp
  uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
  // repeatTime = 4, 128 elements one repeat, 512 elements total, half精度组合
  // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
  // dstRepStride, srcRepStride = 8, no gap between repeats
  AscendC::Axpy(dstLocal, srcLocal, (half)2.0, mask, 4,{ 1, 1, 8, 8 });
  ```

- tensor前n个数据计算样例。

  ```cpp
  AscendC::Axpy(dstLocal, src0Local, (half)2.0, 512);// half精度组合
  ```

结果示例如下：

```plain
输入数据(src0Local):
[1. 2. 3. 4. 5. 6. ... 512.]
输入数据(scalarValue):2.0
输出数据(dstLocal)初始值:
[0. 0. 0. 0. 0. 0. ... 0.]
进行Axpy计算后，输出数据(dstLocal):
[2. 4. 6. 8. 10. 12. ... 1024.]
```
