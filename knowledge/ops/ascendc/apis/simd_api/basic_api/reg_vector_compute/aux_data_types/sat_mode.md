---
type: API
title: SatMode
description: SatMode 用于设置 Cast 类型转换的饱和与不饱和模式，包括 UNKNOWN、NO_SAT 和 SAT。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/aux_data_types/SatMode.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-10T02:36:08Z'
platforms: []
---

# SatMode<a name="ZH-CN_TOPIC_0000001956862301"></a>

| 参数名 | 描述 |
| :-- | :-- | 
| SatMode | 用于设置Cast类型转换饱和与不饱和模式。<br>enum class SatMode { <br>    UNKNOWN = -1, <br>    NO_SAT, <br>    SAT <br>};<br>&bull;&nbsp;&nbsp;UNKNOWN：当转换类型不支持SatMode时，选择该模式；<br>&bull;&nbsp;&nbsp;NO_SAT：不饱和模式；<br>&bull;&nbsp;&nbsp;SAT：饱和模式。 |

```cpp
enum class SatMode {
    UNKNOWN = -1,
    NO_SAT,
    SAT
};
```

## 配套 API 与常见组合

- [MaskMergeMode](mask_merge_mode.md)——同一 CastTrait 初始化列表中的共同字段，配置同一条类型转换指令
- [RegLayout](reg_layout.md)——CastTrait 共同配置字段（证据为同一初始化列表），P=1.00 完全伴随，组合即 950 类型转换的惯用配方
- [MaskReg](../register_data_types/mask_reg.md)——Cast 调用的掩码入参与 trait.satMode 在同一条指令上共同配置
- [RegTensor](../register_data_types/reg_tensor.md)——SatMode 配置 Cast 对 RegTensor 源/目的操作数转换的饱和行为，同一指令的范式构件
