---
type: API
title: RoundMode
description: 枚举类型，用于设置精度转换时的舍入模式，支持向偶数舍入、向下、向上、四舍五入、截断等多种模式。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/aux_data_types/RoundMode.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# RoundMode<a name="ZH-CN_TOPIC_0000001956862301"></a>

| 参数名 | 描述 |
| :-- | :-- | 
| RoundMode | 用于设置控制舍入模式。<br>enum class RoundMode { <br>    CAST_NONE = 0, <br>    CAST_RINT, <br>    CAST_FLOOR,<br>    CAST_CEIL,<br>    CAST_ROUND,<br>    CAST_TRUNC,<br>    CAST_ODD,<br>    CAST_HYBRID <br>};<br>&bull;&nbsp;&nbsp;CAST_NONE：当转换过程存在精度损失时，按 CAST_RINT 模式处理；当不存在精度损失时，不进行舍入。<br>&bull;&nbsp;&nbsp;CAST_RINT：向最近的偶数舍入。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若待舍入部分的首位为 0，则不进位。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若待舍入部分的首位为 1 且后续位不全为 0，则进位。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若待舍入部分的首位为 1 且后续位全为 0：当非舍入部分的末位为 0，则不进位。当非舍入部分的末位为 1，则进位。<br>&bull;&nbsp;&nbsp;CAST_FLOOR：向负无穷大方向舍入。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若符号位（S）为 0（正数），不进位。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若符号位（S）为 1（负数）：当待舍入部分全为 0，则不进位。否则，进位。<br>&bull;&nbsp;&nbsp;CAST_CEIL：向正无穷大方向舍入。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若符号位（S）为 1（负数），不进位。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若符号位（S）为 0（正数）：当待舍入部分全为 0，则不进位。否则，进位。<br>&bull;&nbsp;&nbsp;CAST_ROUND： 四舍五入。若待舍入部分的首位为0，则不进位。否则，进位。<br>&bull;&nbsp;&nbsp;CAST_TRUNC： 直接丢弃待舍入部分。<br>&bull;&nbsp;&nbsp;CAST_ODD：向最近的奇数舍入。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若待舍入部分全为 0，则不进位。<br>&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;若待舍入部分不全为 0：非舍入部分的末位为 1，则不进位。当非舍入部分的末位为 0，则进位。<br>&bull;&nbsp;&nbsp;CAST_HYBRID：随机舍入，目前特指输出结果是hifloat8_t数据类型时的随机舍入。 |

```cpp
enum class RoundMode {
    CAST_NONE = 0,
    CAST_RINT,
    CAST_FLOOR,
    CAST_CEIL,
    CAST_ROUND,
    CAST_TRUNC,
    CAST_ODD,
    CAST_HYBRID
};
```

- [FastGelu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/fast_gelu_950.md)
