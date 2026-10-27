---
type: API
title: 通用说明和约束（SIMD API）
description: 说明 SIMD 基础与高阶 API 的头文件、链接方式、存储单元对齐要求及 TPosition 内存映射。
tags:
- task.lookup
- topic.programming_model
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/general_description_and_constraints.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 通用说明和约束<a name="ZH-CN_TOPIC_0000001714258422"></a>

## 所需头文件和库文件<a name="section1299152225418"></a>

> [!NOTE]说明
> Ascend C API 所在头文件目录为：
> - 基础API：`${INSTALL_DIR}/asc/include/basic_api`
> - 高阶API：（注意，如下目录头文件中包含的接口如果未在资料中声明，属于间接调用接口，开发者无需关注）
>   - `${INSTALL_DIR}/asc/include/adv_api`
>   - `${INSTALL_DIR}/asc/include/tiling`
> 
> `${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径。以 root 用户安装为例，安装后文件默认存储路径为：`/usr/local/Ascend/cann`。
> 
> 使用 Ascend C API 依赖的库文件说明如下：
> - 基础API：不涉及
> - 高阶API：因高阶 API 配套 Host Tiling 接口，需要链接 `libtiling_api.a`。

为方便开发者使用，Ascend C基础API和高阶API均支持通过包含核函数（Kernel）\_operator.h文件来调用相应接口。如无特殊说明，包含该头文件即可满足接口调用需求。若API文档中有特殊说明，则应遵循API的具体说明。

```
#include "kernel_operator.h"
```

## 逻辑位置和物理存储的映射关系<a name="section1359919519819"></a>

Ascend C API的操作数通常为[GlobalTensor](basic_api/data_structures/global_tensor/index.md)和[LocalTensor](basic_api/data_structures/local_tensor/index.md)，Tensor数据的存储位置均使用逻辑位置（[TPosition](basic_api/aux_data_structures/t_position.md)）来表达，从而隐藏了硬件架构的差异。TPosition类型包括：VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2等，其与物理内存的映射关系如下表所示。

**表1**  TPosition与物理内存映射关系

<a name="table07372185712"></a>

| TPosition | 物理内存 |
| --- | --- |
| GM | Global Memory |
| VECIN | UB |
| VECCALC | UB |
| VECOUT | UB |
| A1 | L1 Buffer |
| A2 | L0A Buffer |
| B1 | L1 Buffer |
| B2 | L0B Buffer |
| C1 |  |
| C2 |  |
| CO1 | L0C Buffer |
| CO2 |  |
| TSCM | L1 Buffer |
| SPM |  |
| C2PIPE2GM |  |

## 通用地址对齐约束<a name="section796754519912"></a>

AI Core上的存储单元用于存储矢量计算、矩阵计算的源操作数和目的操作数，各类存储单元的对齐要求如[表2](#table16278354141117)所示，因此Ascend C API操作数的起始地址对齐要求应与这些存储单元的对齐要求保持一致。**需要注意的是，如果接口中已明确说明操作数起始地址对齐要求，则以具体API中的说明为准。**

**表2**  不同存储单元的对齐要求

<a name="table16278354141117"></a>

| 存储单元 | 对齐要求 |
| --- | --- |
| Global Memory | 无对齐要求。 |
| UB | 32Byte对齐。 |
| L1 Buffer | 32Byte对齐。 |
| L0A Buffer/L0B Buffer | 512Byte对齐。 |
| L0C Buffer | 64Byte对齐。 |
| BiasTable Buffer | 64Byte对齐。 |
| Fixpipe Buffer | 64Byte对齐。 |

## 通用地址重叠约束<a name="section668772811100"></a>

使用基础API的Tensor高维切分计算接口时，为了节省地址空间，开发者可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠）。使用时需要注意以下约束：

-   单次迭代内：源操作数与目的操作数必须100%完全重叠，不支持部分重叠。
-   多次迭代间：不支持前序迭代的目的操作数与后序迭代的源操作数重叠。例如，第N次迭代的目的操作数是第N+1次的源操作数（如下图所示）。在这种情况下，第N次迭代可能会改写覆盖源操作数的数值，导致无法得到预期结果。特别地，对于部分双目计算类的API（Add、Sub、Mul、Max、Min、AddRelu、SubRelu），当数据类型为half、int32\_t、float时，支持前序迭代的目的操作数与后序迭代的源操作数重叠：仅针对目的操作数和第二个源操作数重叠的情况，且src1RepStride或者dstRepStride必须为0。
-   多次迭代间：不支持前序迭代的目的操作数与后序迭代的目的操作数重叠。例如，第一轮循环写入的DataBlock编号为0、2、4、6、8、10、12、14，第二轮循环写入的编号为8、10、12、14、16、18、20、22，即上一轮迭代会踩踏前一轮迭代的后4个DataBlock，导致无法得到预期的结果。

**图1**  地址重叠示例（不支持）<a name="fig12654416181216"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/address_overlap_unsupported.png "地址重叠示例（不支持）")

> [!NOTE]说明 
>本节所述地址重叠通用约束适用于一般情况，API参考中如有额外特殊说明的，则以具体API中的说明为准。
>API中没有描述地址重叠约束的，视为不支持Tensor高维切分计算的地址重叠，地址重叠时计算结果可能不满足预期。
