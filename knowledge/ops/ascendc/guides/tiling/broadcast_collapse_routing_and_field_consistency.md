---
type: Guide
title: Broadcast 合轴、路由与字段一致性
description: 保留广播的输入坐标和分阶段展开，指出fallback返回值缺口。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/broadcast/fallback/tiling-fields.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/broadcast/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/broadcast/fallback/script/broadcast_tiling.py
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# Broadcast 合轴、路由与字段一致性

先将输入左补1到输出rank，逐轴验证每输入维等于输出维或1，再按各输入的广播位图合并连续同模式轴；实际存储还需可合并的stride。广播轴stride0，其他轴由真实输入物理布局向右乘积得出。source仅依据其他输入是否>1设flag，未校验输出shape与输入兼容、输入rank过长/空列表，不能直接接受任意指定output。

OneDim合轴后可用标量专用Adds/Muls等，但减法非交换，scalar-x不能直接用x-scalar形式的Subs。多维可在GM→UB阶段以NDDMA zero-stride展开，或在UB内Broadcast；中间结果已经UB时不应假装重新从GM加载。DAV_2201静态和DAV_3510动态/NDDMA的能力边界需实际API，force标志不能强行启用不支持平台。

NDDMA内层最多5维的示例通过外层循环处理更多轴，schMode1无外循环/2有循环要根据实际交给搬运的剩余轴决定，不能只看原始rank。静态Broadcast、DataCopyPad+Copy+GatherMask、短行Duplicate是不同候选，必须匹配dtype、展开方向、对齐与scratch；GatherMask省空间需真实接口允许复用。

选路_NLast检查尾轴stride非0且前轴存在广播，脚本把尾轴元素数与CACHE_LINE/2=64比较，文档阈值是字节概念，dtype变化要重新换算。dtype主映射接受float32而选路局部map不识别，会默认2字节误选dynamic_ub；未知dtype也不应默认为FP16。硬件AIV/AIC核数要跟实际Kernel类型，字段表写AIC不能自动指导Vector launch。

多维容量max=(UB-extra)*8/(buffer_num*maxBits)的bit/byte单位一致，buffer_num=Ninputs+1只覆盖一份各输入输出，不包含双缓冲、转换/scratch与压缩广播张量实际大小；需真实清单。输入较窄或mask不同类型时单一dtype参数不描述全部内存。

脚本OneDim结果先生成input_strides=[[1]]，最终只回写collapsed input_shapes和scalar_flags，未恢复all_strides，双输入含scalar会缺一个stride0条目。to_dict还省input_shapes/shape_len/fused_product等，消费者不能把序列化结果当完整ABI。验证可用每输入单独编码坐标，覆盖广播交替轴、rank0/零轴、不同dtype别名、非连续输入及强制分支。
