---
type: Guide
title: SIMT 数值宏族与高低位常量
description: 保存原参考的完整常量导航，并区分最小次正规数、负零与高低位分解。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-best-practices/references/macro-api.md
  role: primary
created_at: '2026-09-11T02:35:33Z'
updated_at: '2026-09-11T02:35:33Z'
platforms: []
---

# SIMT 数值宏族与高低位常量

来源按头文件组织SIMT特殊值宏：asc_bf16.h使用后缀BF16，asc_fp16.h使用FP16，math_constants.h使用F。每组包括ASCRT_INF_*、ASCRT_MAX_NORMAL_*、ASCRT_MIN_DENORM_*、ASCRT_NAN_*、ASCRT_NEG_ZERO_*、ASCRT_ONE_*、ASCRT_ZERO_*，其中*替换为上述后缀。MAX_NORMAL是最大有限正常数，MIN_DENORM是最小正次正规量级，不是最负值；是否在具体运算中保留次正规值另取决于指令合同。

math_constants.h的来源还列出以下float常量，需核对安装版本中的准确拼写和声明类型后使用：

| 宏 | 数学含义 |
|---|---|
| ASCRT_SQRT_HALF_F | √0.5 |
| ASCRT_SQRT_HALF_HI_F / ASCRT_SQRT_HALF_LO_F | √0.5的高/低位分解 |
| ASCRT_SQRT_TWO_F | √2 |
| ASCRT_THIRD_F | 1/3 |
| ASCRT_PIO4_F / ASCRT_PIO2_F / ASCRT_PI_F / ASCRT_TWOPI_F | π/4、π/2、π、2π |
| ASCRT_ONE_OVER_PI_F / ASCRT_TWO_OVER_PI_F | 1/π、2/π |
| ASCRT_LN2_F / ASCRT_LN10_F | ln2、ln10 |
| ASCRT_LOG2E_F / ASCRT_LOG10E_F | log₂e、log₁₀e |
| ASCRT_LN2_HI_F / ASCRT_LN2_LO_F | ln2的高/低位分解 |

高/低位不是两个可互换的近似值；数值算法若依赖分拆补偿，需按原精度与运算顺序使用。不同dtype的ONE/ZERO和NaN宏不意味着按位编码可互换，+0/-0数值比较可相等但符号传播有区别。数学常量是目标浮点近似值，不能据宏名宣称实数精确表示。

已有宏总览覆盖主要族但未逐项保存这些float常量。其ASCRT_L2E_F拼写与来源LOG2E_F不同，本卡保留来源导航并不确认两者都是当前SDK别名；需对实际头文件作编译核对。本批没有下载或构建SDK宏示例，不给出跨平台可用性承诺。
