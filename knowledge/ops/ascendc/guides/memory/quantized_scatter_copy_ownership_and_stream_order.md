---
type: Guide
title: 量化 Scatter 的复制归属与 Stream 顺序
description: 保留双 Kernel 设计条件，纠正只读索引竞争及搬运对齐绝对化。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_quantization_patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_shuffle_patterns.md
  role: supporting
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# 量化 Scatter 的复制归属与 Stream 顺序

output=clone(input)后量化updates并按indices更新时，若大部分数据仅复制，可把完整复制与少量更新分别分析。两个Kernel在同一stream按正确次序提交通常建立设备执行顺序，无需仅为这项依赖加Host synchronize；跨stream、Host访问或输出生命周期另有同步需求。来源1.13倍Device、0.83倍端到端是特定案例，不能把任何Host同步一律禁止。

单Kernel也可通过无重叠写所有权设计融合，例如复制者跳过所有更新目标，再由唯一更新者写入；是否收益取决于索引处理和流量，不是必然无收益。重复indices需要明确覆盖先后/归约/未定义合同，不能多个核无保护写同一段。未更新区域必须来自正确input快照，不能为减少复制改变out-of-place语义。

大段DataCopy/Pad与合法双缓冲可摊薄固定开销，8–64KiB chunk、512B加双PIPE_ALL约100GB/s等只作来源场景。DataCopy count的32B要求不能推广到DataCopyPad有效尾长；507035也有其他UB访问/对齐原因，不能仅凭错误码认定count。int8借int32视图搬运需满足地址、总字节、类型别名及尾部，不表示dtype随意reinterpret即可。

把indices/scales/zero-point一次搬到UB可减少Scalar读开销，但多核只读稳定索引本身没有写竞态；来源“禁止GlobalTensor::GetValue因DCache竞争”缺少写入者/可见性条件。真正风险是并发更新、cache一致性、错误索引宽度或未等生产完成。独立TBuf和depth1都不是唯一正确修复，先证明生命周期。

量化应在能表示中间值的域内按合同处理zero-point、clamp、round及窄化。对有限值及整数饱和边界，先clamp后nearest整型可避免转换溢出；NaN的Min/Max传播和±Inf行为需查API，不能声称任意cast链天然处理。输出偏移i*axisStride+indices[i]*lastDim仅对应来源固定切片模型，需加真实batch/axis映射、合法索引与dtype字节。

updates全驻留、按batch流式、按连续段再切分是容量候选；scale/zero-point“始终常驻”仅当其总量可容纳。多任务批处理和单长任务分片需考虑是否共享参数/输出。Host侧ATen预处理会增加launch与可能复制，但应按整体成本选择，不能因脚本位置就判断CPU算术或一律不允许。

公平计时需区分in-place与out-of-place、完整复制是否计入、Device task、Device span及Host端到端。eager组合可作有效应用baseline，只是不能与单融合Kernel时长假装同一计量范围；官方同语义融合算子可另作比较。
