---
type: API
title: DataCopy
description: 软仿接口，支持从UB到GM搬运过程中进行NZ到ND格式的转换。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_UBToGM_NZ2ND.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# DataCopy（UBToGM随路转换NZ2ND搬运）<a name="ZH-CN_TOPIC_0000002391805265"></a>

## 功能说明<a name="section12840195813362"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

该接口为软仿接口，从易用性角度出发进行设计，支持在从Unified Buffer（UB）到Global Memory的数据搬运过程中进行NZ到ND格式的转换。

具体支持的数据通路为（以[逻辑位置TPosition](../../aux_data_structures/t_position.md)表示）：

- UB -> Global Memory
    - VECOUT -> GM
    
    - CO2 -> GM（仅Atlas推理系列产品AI Core支持）
    

## 函数原型<a name="section1792117555586"></a>

```cpp
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const Nz2NdParamsFull& intriParams)
```

## 参数说明<a name="section14983445508"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :--- | :--- |
| T | 源操作数或者目的操作数的数据类型。支持的数据类型请参考[数据类型](#section4219135304818)。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| dst | 输出 | 目的操作数，类型为[GlobalTensor](../../data_structures/global_tensor/global_tensor_intro.md)。 |
| src | 输入 | 源操作数，类型为[LocalTensor](../../data_structures/local_tensor/local_tensor_intro.md)。 |
| intriParams | 输入 | 搬运参数，类型为Nz2NdParamsFull，具体参数请参考[表3](#table_nz2nd_3)。<br>具体定义请参考`${INSTALL_DIR}/asc/include/basic_api/kernel_struct_data_copy.h`，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。 |

**表3**  Nz2NdParamsFull结构体内参数定义<a name="table_nz2nd_3"></a>

| 参数名 | 描述 |
| :--- | :--- |
| ndNum | 传输NZ矩阵的数目，取值范围：ndNum∈[0, 4095]。 |
| nValue | NZ矩阵的行数，取值范围：nValue∈[1, 8192]。 |
| dValue | NZ矩阵的列数，取值范围：dValue∈[1, 8192]。dValue必须为16的倍数。 |
| srcNdMatrixStride | 源相邻NZ矩阵的偏移（头与头），取值范围：srcNdMatrixStride∈[1, 512]，单位256（16 \* 16）个元素。 |
| srcNStride | 源同一NZ矩阵的相邻Z排布的偏移（头与头），取值范围：srcNStride∈[0, 4096]，单位16个元素。 |
| dstDStride | 目的ND矩阵的相邻行的偏移（头与头），取值范围：dstDStride∈[1, 65535]，单位为元素。 |
| dstNdMatrixStride | 目的ND矩阵中，来自源相邻NZ矩阵的偏移（头与头），取值范围：dstNdMatrixStride∈[1, 65535]，单位为元素。 |

以half数据类型为例，NZ2ND转换示意图如下，样例中参数设置值和解释说明如下：

- ndNum = 2，表示源NZ矩阵的数目为2（NZ矩阵1为A1\~A4 + B1\~B4，NZ矩阵2为C1\~C4 + D1\~D4）。
- nValue = 4，NZ矩阵的行数，也就是矩阵的高度为4。
- dValue = 32，NZ矩阵的列数，也就是矩阵的宽度为32个元素。
- srcNdMatrixStride = 1，表达相邻NZ矩阵起始地址间的偏移，即为A1\~C1的距离，即为256个元素（16个DataBlock \* 16个元素）。
- srcNStride = 4，表示同一个源NZ矩阵的相邻Z排布的偏移，即为A1到B1的距离，即为64个元素（4个DataBlock \* 16个元素）。
- dstDStride = 160，表达一个目的ND矩阵的相邻行之间的偏移，即A1和A2之间的距离，即为10个DataBlock，即10 \* 16 = 160个元素。
- dstNdMatrixStride = 48，表达dst中第x个目的ND矩阵的起点和第x+1个目的ND矩阵的起点的偏移，即A1和C1之间的距离，即为3个DataBlock，3 \* 16 = 48个元素。

**图1**  NZ2ND转换示意图（half数据类型）<a name="fig15851251122815"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/nz2nd_conversion_half.png "NZ2ND转换示意图（half数据类型）")

以float数据类型为例，NZ2ND转换示意图如下，样例中参数设置值和解释说明如下：

- ndNum = 2，表示源NZ矩阵的数目为2（NZ矩阵1为A1\~A8 + B1\~B8，NZ矩阵2为C1\~C8 + D1\~D8）。
- nValue = 4，NZ矩阵的行数，也就是矩阵的高度为4。
- dValue = 32，NZ矩阵的列数，也就是矩阵的宽度为32个元素。
- srcNdMatrixStride = 1，表达相邻NZ矩阵起始地址间的偏移，即A1到C1的距离，为256个元素（32个DataBlock \* 8个元素）。
- srcNStride = 4，表示同一个源NZ矩阵的相邻Z排布的偏移，即A1到B1的距离，为64个元素（8个DataBlock \* 8个元素）。
- dstDStride = 144，表示一个目的ND矩阵的相邻行之间的偏移，即A1和A3之间的距离，为18个DataBlock，即18 \* 8 = 144个元素。
- dstNdMatrixStride = 40，表示dst中第x个目的ND矩阵的起点和第x+1个目的ND矩阵的起点的偏移，即A1和C1之间的距离，为5个DataBlock，5 \* 8 = 40个元素。

**图2**  NZ2ND转换示意图（float数据类型）<a name="fig5586175192811"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/nz2nd_conversion_float.png "NZ2ND转换示意图（float数据类型）")

## 数据类型<a name="section4219135304818"></a>


- Ascend 950PR&950DT系列产品，支持的数据类型为：bool、int8_t、uint8_t、hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t、double、complex64。



- Atlas A3系列产品，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。



- Atlas A2系列产品，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。



- Atlas推理系列产品AI Core，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。


## 返回值说明<a name="section129001927113216"></a>

无

## 约束说明<a name="section830051273220"></a>

- 该接口为软仿接口，从易用性角度出发进行设计。
- Nz2NdParamsFull结构体参数的值需在取值范围内：

    **表4**  Nz2NdParamsFull结构体参数取值范围

    | 参数名 | 取值范围 |
    | --- | --- |
    | ndNum | [0, 4095] |
    | nValue | [1, 8192] |
    | dValue | [1, 8192] |
    | srcNdMatrixStride | [1, 512] |
    | srcNStride | [0, 4096] |
    | dstDStride | [1, 65535] |
    | dstNdMatrixStride | [1, 65535] |

## 调用示例<a name="section2409153316111"></a>

intriParams参数解析请参考[图1](#fig15851251122815)。

```cpp
// srcLocal为half类型的LocalTensor，dstGlobal为half类型的GlobalTensor。
AscendC::Nz2NdParamsFull intriParams{1, 32, 32, 1, 32, 32, 1};
// Local Memory -> Global Memory
AscendC::DataCopy(dstGlobal, srcLocal, intriParams);
```

结果示例：

```text
输入数据(srcLocal): [1 2 3 ... 1024]
输出数据(dstGlobal):[1 2 ... 15 16 513 514 ... 527 528 17 18 ... 31 32 529 530 ... 543 544 ...497 498 ...  511 512  1009 1010... 1023 1024]
```
