---
type: API
title: DataCopy
description: 支持UB之间的连续数据搬运，数据传输过程中保持原始格式和内容不变。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_UBToUB_continuous.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# DataCopy（UBToUB连续数据搬运）<a name="ZH-CN_TOPIC_0000002382908021"></a>

## 功能说明<a name="section474617392321"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

支持Unified Buffer（UB）内部的连续数据搬运，数据在传输过程中保持原始格式和内容不变。

具体支持的数据通路为（以[逻辑位置TPosition](../../aux_data_structures/t_position.md)表示）：

- UB -> UB
    - VECIN -> VECCALC
    - VECCALC -> VECOUT

## 函数原型<a name="section1954364615315"></a>

```cpp
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :--- | :--- |
| T | 操作数的数据类型。支持的数据类型请参考[数据类型](#section4219135304818)。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| dst | 输出 | 目的操作数。类型为[LocalTensor](../../data_structures/local_tensor/local_tensor_intro.md)，存储位置为UB，目的地址需要32字节对齐。 |
| src | 输入 | 源操作数。类型为LocalTensor，存储位置为UB，源地址需要32字节对齐。 |
| count | 输入 | 参与搬运的元素个数。<br>**注：count * sizeof(T)需要32字节对齐，若未对齐，搬运量会向下取整到32字节对齐。** |

以half数据类型为例，源操作数的shape为1 \* 128。当count = 128时，[图1](#zh-cn_topic_0000002567699425_fig13481537531)将源操作数中128个元素连续搬运至目的操作数。

**图1**  连续搬运示意图<a name="zh-cn_topic_0000002567699425_fig13481537531"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/continuous_data_copy_diagram.png "连续搬运示意图")

## 数据类型<a name="section4219135304818"></a>

源操作数和目的操作数支持的数据类型保持一致，具体如下：


- Ascend 950PR&950DT系列产品，支持的数据类型为：bool、int8_t、uint8_t、hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t、double、complex64。



- Atlas A3系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。



- Atlas A2系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。



- Atlas推理系列产品AI Core，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float、int64_t、uint64_t、double。



- Atlas训练系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float、int64_t、uint64_t、double。




- Kirin 9050处理器系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float。


- KirinDev0001处理器系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float。


- KirinDev0002处理器系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float。


- KirinDev0003处理器系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float。


## 返回值说明<a name="section44801012174220"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 位于UB的地址必须32字节对齐。
- 调用连续搬运接口时，count \* sizeof\(T\)需要32字节对齐，若未对齐，则搬运量会向下取整到32字节对齐。

- 针对如下产品型号：
    - Atlas A2系列产品
    - Atlas A3系列产品

    在跨卡通信算子开发场景，DataCopy类接口支持跨卡数据搬运，仅支持HCCS物理链路，不支持其他通路；开发者开发过程中，需要关注涉及卡间通信的物理通路，可通过`npu-smi info -t topo`命令查询HCCS物理链路。


## 调用示例<a name="section122101199486"></a>

```cpp
// srcLocal、dstLocal为half类型的LocalTensor。
// 使用传入count参数的搬运接口，完成连续搬运。
AscendC::DataCopy(dstLocal, srcLocal, 512);
```

结果示例：

```text
输入数据srcLocal：[1 2 3 ... 512]
输出数据dstLocal：[1 2 3 ... 512]
```
