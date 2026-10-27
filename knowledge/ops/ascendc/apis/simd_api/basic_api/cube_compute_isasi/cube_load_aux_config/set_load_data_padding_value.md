---
type: API
title: SetLoadDataPaddingValue
description: 调用LoadData（卷积数据搬运）接口时设置Pad填充的数值，当LoadData模板参数isSetPadding为false时通过本接口设置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_load_aux_config/SetLoadDataPaddingValue.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_mm_intf.h
---

# SetLoadDataPaddingValue<a name="ZH-CN_TOPIC_0000001834740625"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_mm_intf.h"`。

用于调用[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)接口时设置Pad填充的数值。LoadData（卷积数据搬运）的模板参数isSetPadding设置为false时，用户需要通过本接口设置Pad填充的数值，设置为true时，本接口设置的填充值不生效。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void SetLoadDataPaddingValue(const T padValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ------ | ------ | ------ |
| padValue | 输入 | Pad填充值的数值。 |

## 数据类型

<br>
<br>
<br>
<br>
<br>
<br>


## 返回值说明

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

参考[调用示例](set_fmatrix.md#section642mcpsimp)

展示代码示例片段：

```cpp
// LoadData（卷积数据搬运）v2指令完成img2col的过程，可知img2col后A矩阵高度为ho * wo，根据ho和wo的计算公式，代入卷积核宽度、卷积核滑动步长、卷积核膨胀系数等参数可知：A矩阵的高度为CeilAlign(k, fractalShape[0])；img2col后A矩阵宽度为ci * kh * kw,代入kh=1，kw=1，可知A矩阵的宽度为CeilAlign(m, fractalShape[1])。最后，配置loadDataParams.enTranspose = true，将整个A矩阵转置并且将其中每一个分形转置
// 使用load3d接口，实现NZ2ZZ
AscendC::LoadData3DParamsV2<T> loadDataParams;
// 设置loadDataParams相关参数
...
// 设置不在LoadData（卷积数据搬运）接口内部设置相关属性，通过外部接口显式调用设置
static constexpr AscendC::IsResetLoad3dConfig LOAD3D_CONFIG = {false, false};

// 使用SetLoadDataPaddingValue接口，设置LoadData（卷积数据搬运）接口的padValue参数
AscendC::SetLoadDataPaddingValue(0);

AscendC::LoadData<T, LOAD3D_CONFIG>(a2Local, a1Local, loadDataParams);
```

- [AllGatherMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/all_gather_matmul.md)
- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
- [AlltoAllQuantMatmul A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul.md)
- [MatmulReduceScatter A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_reduce_scatter.md)
