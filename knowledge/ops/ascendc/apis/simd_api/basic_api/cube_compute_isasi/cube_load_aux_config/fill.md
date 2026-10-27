---
type: API
title: Fill
description: 将L1 Buffer/L0A Buffer/L0B Buffer上的LocalTensor初始化为指定数值，通过InitConstValueParams配置迭代次数、数据块个数和填充值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_load_aux_config/Fill.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
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
- include/basic_api/kernel_operator_mm_intf.h
---

# Fill<a name="ZH-CN_TOPIC_0000001834660669"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_mm_intf.h"`。

将特定物理存储位置的LocalTensor初始化为某一具体数值。仅支持L1 Buffer/L0A Buffer/L0B Buffer上的LocalTensor初始化。

如下图所示，将数据空间内最后三行初始化为0，配置InitConstValueParams参数：initValue=0表示初始化值为0，repeatTimes=5表示按照行方向迭代5次，blockNum=3表示每次repeat对列方向的3块32Byte数据块清0，dstGap=5表示相邻repeat之间间隔5个数据块。

**图1** L1 Buffer中Fill示意图<a name="zh-cn_topic_0000002512331632_fig197511640134513"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fill_L1.png "L1 Buffer中Fill示意图")

如下图所示，将内存空间内的值全部初始化为0，需要配置InitConstValueParams参数：initValue=0表示初始化值为0，repeatTimes=5表示按照行方向迭代5次，blockNum=2表示每次repeat对列方向的2块512Byte数据块清0，dstGap=3表示相邻repeat之间间隔3个数据块。

**图2** L0A Buffer中Fill示意图<a name="zh-cn_topic_0000002512331632_fig11499125816478"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fill_L0A.png)

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Fill(const LocalTensor<T>& dst, const InitConstValueParams<U>& initConstValueParams)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 模板参数说明

| 参数名称 | 描述 |
| ------- | ------ |
| T | dst的数据类型。 |
| U | 初始化值的数据类型。<br>&nbsp;&nbsp;&bull;当dst使用基础数据类型时，U和dst的数据类型T需保持一致，否则编译失败。<br>&nbsp;&nbsp;&bull;当dst使用[TensorTrait](../../aux_data_structures/tensor_trait/index.md)类型时，U和dst的数据类型T的LiteType需保持一致，否则编译失败。<br>&nbsp;&nbsp;最后一个模板参数仅用于上述数据类型检查，用户无需关注。 |

**表2** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| --------- | ---------- | ------ |
| dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor。 |
| InitConstValueParams | 输入 | 初始化相关参数，类型为InitConstValueParams。<br>具体定义请参考`${INSTALL_DIR}/asc/include/basic_api/kernel_struct_mm.h`，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。<br>参数说明请参考[表3](#table3)。 |

<a name="table3"></a>
**表3** InitConstValueParams结构体参数说明

| 参数名称 | 含义 |
| --------- | ------ |
| repeatTimes | 迭代次数。默认值为0。取值范围：repeatTimes∈[0, 32767]。**注：repeatTimes = 0表示不执行，该指令将被视为NOP（空操作）。** |
| blockNum | 每次迭代初始化的数据块个数，取值范围：blockNum∈[0, 32767]。默认值为0。**注：blockNum = 0表示不执行，该指令将被视为NOP（空操作）。**<br>&nbsp;&nbsp;&bull; dst的物理存储位置为L1 Buffer（TPosition: A1/B1）时，每一个block（数据块）大小是32B；<br>&nbsp;&nbsp;&bull; dst的物理存储位置为L0A Buffer（TPosition: A2）/L0B Buffer（TPosition: B2）时，每一个block（数据块）大小是512B。 |
| dstGap | 目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离。<br>&nbsp;&nbsp;&bull; dst的物理存储位置为L1 Buffer（TPosition: A1/B1）时，单位是32B；<br>&nbsp;&nbsp;&bull; dst的物理存储位置为L0A Buffer（TPosition: A2）/L0B Buffer（TPosition: B2）时，单位是512B。<br>&nbsp;&nbsp;取值范围：dstGap∈[0, 32767]。默认值为0。 |
| initValue | 初始化的value值，支持的数据类型与dst保持一致。 |

## 数据类型

<br>
<br>
<br>
<br>
<br>
<br>
<br>


## 返回值说明

无

## 约束说明<a name="section633mcpsimp"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。
- 当目的操作数位于L0A Buffer/L0B Buffer时，指令执行占用的流水为PIPE\_MTE1，目的操作数地址需要512Byte对齐。当目的操作数位于L1 Buffer时，指令执行占用的流水为PIPE\_MTE2，目的地址需要32Byte对齐。
- 当repeatTimes为0或者blockNum为0时，指令不会进行任何操作。
- 不同的型号在设置InitConstValueParams参数时，支持配置参数的不同。
    - 仅支持配置迭代次数和初始化值场景下，其他参数配置无效。每次迭代处理固定数据量（512字节），迭代间无间隔。
    - 支持配置所有参数场景下，支持配置迭代次数、初始化值、每个迭代处理的数据块个数和迭代间间隔。
    
    - 特别地，针对如下型号，仅支持配置迭代次数和初始化值。
        
        
        - Atlas训练系列产品。
        
        
        - Atlas推理系列产品AI Core。
        
    

- 支持的物理存储位置为：L1 Buffer（TPosition: A1/B1）、L0A Buffer（TPosition: A2）、L0B Buffer（TPosition: B2）。
    
    - 特别针对Ascend 950PR&950DT系列产品，支持物理存储位置为：L1 Buffer（TPosition: A1/B1）。
    
    - 如果物理存储位置为L1 Buffer（TPosition: A1/B1），起始地址需要满足32B对齐；
    - 如果物理存储位置为L0A Buffer（TPosition: A2）、L0B Buffer（TPosition: B2），起始地址需要满足512B对齐。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
uint32_t mLength = 16;
uint32_t kLength = 16;
Fill(leftMatrix, {1, static_cast<uint16_t>(mLength * kLength * sizeof(float) / 32), 0, 1}); // 给leftMatrix填充mLength * kLength长度的数据为1，按32B的颗粒度进行填充
```

若接口用于对L1 Buffer/L0A Buffer/L0B Buffer上的LocalTensor清0使用，示例请参考：[Fill调用样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/06_compatibility_guide/fill)。
