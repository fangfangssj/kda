---
type: API
title: SetLoadDataBoundary
description: 设置LoadData（卷积数据搬运）接口所需的L1 Buffer（A1/B1）边界值，源操作数地址超出边界时从起始地址开始读取。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_load_aux_config/SetLoadDataBoundary.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_mm_intf.h
---

# SetLoadDataBoundary<a name="ZH-CN_TOPIC_0000001787861430"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_mm_intf.h"`。

设置[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)接口所需的L1 Buffer（TPosition: A1/B1）边界值。

如果[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)指令在处理源操作数时，源操作数在L1 Buffer（A1/B1）上的地址超出设置的边界，则会从L1 Buffer（A1/B1）起始地址开始读取数据。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ------ | ------ | ------ |
| boundaryValue | 输入 | 边界值：单位是Byte。<br>&nbsp;&nbsp;&bull; LoadData（卷积数据搬运）v1指令：单位是32字节。<br>&nbsp;&nbsp;&bull; LoadData（卷积数据搬运）v2指令：单位是字节。 |

## 返回值说明

无

## 约束说明<a name="section633mcpsimp"></a>

- 如果使用SetLoadDataBoundary接口设置了边界值，配合LoadData（卷积数据搬运）指令使用时，LoadData（卷积数据搬运）指令的L1 Buffer（A1/B1）的初始地址要在设置的边界内。
- 如果boundaryValue设置为0，则表示无边界，可使用整个L1 Buffer（A1/B1）。
- 配合[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)指令使用时，边界值大小最小值要求是1024，如果设置1~1023值是未定义行为。
- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

参考[调用示例](set_fmatrix.md#section642mcpsimp)。

本接口用于配合[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)接口使用，展示代码示例片段：

```cpp
// LoadData（卷积数据搬运）指令完成img2col的过程，可知img2col后A矩阵高度为ho * wo，根据ho和wo的计算公式，代入卷积核宽度、卷积核滑动步长、卷积核膨胀系数等参数可知：A矩阵的高度为CeilAlign(k, fractalShape[0])；img2col后A矩阵宽度为ci * kh * kw，代入kh=1，kw=1，可知A矩阵的宽度为CeilAlign(m, fractalShape[1])。最后，配置loadDataParams.enTranspose = true，将整个A矩阵转置并且将其中每一个分形转置
// 使用load3d接口，实现NZ2ZZ
AscendC::LoadData3DParamsV2<T> loadDataParams;
// 设置loadDataParams相关参数
...

// SetLoadDataBoundary设置为0，可使用整个L1 Buffer
AscendC::SetLoadDataBoundary(0);

AscendC::LoadData(a2Local, a1Local, loadDataParams);
```
