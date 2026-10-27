---
type: Guide
title: Softmax Host容量公式与空维失效条件
description: 保存六种分配成本，纠正强制最小tile和注释中的低估。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_small_r.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_full_load.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_recompute.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ara_full_load.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ara_recompute.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ara_online.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Softmax Host容量公式与空维失效条件

六个Host函数独立返回blockDim并填各自POD，没有统一selector或TilingKey；调用方必须先验证a1/r/a0、core数、UB、dtypeCode。AR要求a0=1但函数本身不校验；DtypeSize把任何非0代码按2B处理也不是合法dtype验证。空维、0核或最小tile装不下时可能出现除0、负值或无效ubFactor，应返回明确不可用并选择合法路径。

记d为输入输出相同的字节宽、L为tileA0、rP为物理对齐R。SmallR预算L*(rP*(4*d+8)+4)。AR FullLoad按每行rP*(4*d+4)，另预留1024B及512B临时；TilingData注释rP*(2*d+4)少算两份队列。AR Recompute每chunk U*(5*d+4)，另32B max、32B sum、2048B cache。ARA FullLoad L*((2*d+12)*R+8)，其中多一行float零填充和一行sum；ARA Recompute L*(2*128*(d+4)+4*(11+cacheLevels))。ARA Online L*(2*chunk*(d+4)+8)，输出队列仍按float宽度分配。

这些是声明的buffer字节，真正InitBuffer的逐份对齐、VF完整读范围、事件和平台保留空间还需计入。较小dtype未必同比放大tile，因为float中间量主导。传入不同输入输出dtype时Host公式不再匹配，模板说明要求Tx==Ty。

CPU直接编译给定Host头：UB196608、32核、A4096、R129、FP32得到ubFactor71、rAligned136；该返回值不保证全部隐含临时区安全。R100000返回ubFactor0。ARA Online在UB仅64B时强制factorMax≥1，仍产L8和16448B分配；强制最小块只避免某些除0，不能构成合法兜底。其他函数factorMax0直接进入除0风险。

AR Recompute的Lcm使用blockSize32数值与元素对齐数混合，结果是额外保守的32元素对齐，并非严谨统一字节推导。U=min(R,alignedU)在单chunk时又可能不对齐，须按实际分配与尾读要求处理。ARA FullLoad实际a0TileBase为FP32的64或16位的128元素，结构说明8/16是另一个惯例，不可混用。
