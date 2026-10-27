---
type: Guide
title: MX Matmul scale 驻留与预取权衡
description: 保存量化scale块、L1倍率及预取规模的独立预算。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/fallback-algorithm.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-variants.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-flow.md
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# MX Matmul scale 驻留与预取权衡

MX示例每32个数据对应1字节UE8 scale，逻辑FP8 scale/data字节比为1/32，FP4半字节数据则为1/16；不是所有低比特都1/32。实际scale按A/B布局、ceil(K/32)、对齐和打包计算，kScale可表示原数据K跨度而非scale元素数，必须看字段单位。

Mb=Nb=256、驻留K768的FP4 A/B数据合计196608B，逻辑scale合计12288B，总208896B，双份417792B；K1024双份则557056B超过512KiB。此算例说明倍率3在逻辑预算下可能成立，仍需实际scale padding、bias和其他保留区，不能直接认定最优或所有平台合法。名为L1_BUFFER_NUM的K倍率与物理buffer份数2应分别记录。

扩大scale预取可减少过小MTE2请求，例如来源coalesce factor16/8/4/2、kScale=factor*kL1、scale双缓冲，可能需把数据四缓冲改双缓冲腾空间。ScaleFactorMax127、最小32KiB事务等是源搜索参数，需要重新校核平台和格式。预取太多也会压缩数据tile/并发，且必须在覆盖前等最后消费者。

enablePreload有时是编译策略、有时由tiling字段传入，不能据“无需改字段”断言随时可开。所有分支重算scale数量、搬运描述符上限、峰值L1与最后有效K组，并测试尾组和不整除倍率。
