---
type: API
title: MaskMergeMode
description: 枚举类型，用于指定寄存器数据写入模式，包括ZEROING置零和MERGING保留原值两种模式。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/aux_data_types/MaskMergeMode.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-10T04:14:14Z'
platforms: []
---

# MaskMergeMode<a name="ZH-CN_TOPIC_0000001956862301"></a>

| 参数名 | 描述 |
| :-- | :-- | 
| MaskMergeMode | 用于指定写入寄存器数据模式。<br>enum class MaskMergeMode { <br>    UNKNOWN, <br>    MERGING, <br>    ZEROING <br>};<br>&bull;&nbsp;&nbsp;UNKNOWN：当转换类型不支持MaskMergeMode时，选择该模式；<br>&bull;&nbsp;&nbsp;ZEROING：mask未选择的元素在dst中置零；<br>&bull;&nbsp;&nbsp;MERGING：mask未选择的元素对应dst元素中保留dst原值。 |

```cpp
enum class MaskMergeMode {
    UNKNOWN,
    MERGING,
    ZEROING
};
```

以Duplicate接口为例，ZEROING和MERGING模式图示如下：

**图1** ZEROING/MERGING对比<a id="fig1"></a>

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/zeroing_merging_comparison.png "ZEROING/MERGING对比")

## 配套 API 与常见组合

- [MaskReg](../register_data_types/mask_reg.md)——MaskMergeMode 的语义即定义 mask 未选中元素在 dst 的写入模式，Add 签名同时含 MaskMergeMode 模板参数与 MaskReg 入参
- [RegTensor](../register_data_types/reg_tensor.md)——MaskMergeMode 指定 dst RegTensor 中未被掩码选中元素的置零/保留行为，同一指令的模板参数与目的操作数
- [RegLayout](reg_layout.md)——同一 CastTrait 的共同配置字段，位宽变化时与 MaskReg 联合决定 dst 排布（Cast 约束明文三者联合作用）
- [SatMode](sat_mode.md)——同一 CastTrait 初始化列表中的共同字段，配置同一条类型转换指令
