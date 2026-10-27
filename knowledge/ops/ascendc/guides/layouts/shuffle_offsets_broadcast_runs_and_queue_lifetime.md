---
type: Guide
title: 重排的偏移、广播行段与队列生命周期
description: 保存规则重排方案并修正 Gather 源必须独立的泛化。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_shuffle_patterns.md
  role: primary
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# 重排的偏移、广播行段与队列生命周期

规律解交织可用GatherMask固定pattern，也可用分段搬移；RoPE half旋转[-x[D/2:],x[:D/2]]需要D可合法分半、原始布局对应且负号精度正确。来源data_move是其封装/DSL名称，不能直接当任意dtype/任意步长的Ascend C原生API。GatherMask FP32在相关版本可支持，来源stride2实测不符应检查模式、源跨度和mask单位，不能据此宣布FP32无能力。

不规则Gather需要int32/uint32等受支持偏移类型，偏移按字节缩放且每个访问均落在源物理区。count不存在来源臆测的统一64元素上限，但仍受具体重载mask、repeat、整数范围和buffer约束。mask符号扩展、错乘dtype宽度及基址复用可引发越界，错误不证明half不能Gather。

队列DeQue出的tensor只要保留到Gather全部消费者完成、之后正确Free，就不必强制复制到独立VECCALC。提前Free后下一轮CopyIn覆盖slot才是生命周期缺陷，double buffer本身不造成竞态。独立副本可延长安全存活但增加流量/容量；depth1也不是逻辑修复的充分证明。

规则offset表可用种子加Vector偏移倍增扩展，需处理非2幂尾、Scalar→Vector依赖、单位与整数溢出；也可持久预计算，哪种更快由实际复用成本决定。仅按有效行生成须同时满足Vector物理padding读取范围。来源4096次SetValue约17.7µs、建表17.5µs是案例，不能把任意Scalar表初始化一律禁止。

broadcast可用受支持repeat-stride为0的源反复消费，并按源行相同的连续行段组织；输出repeat-stride也可直接形成最终布局。repeat/count/stride范围、有效tail、源行内对齐必须满足。tile开始于扁平行r、每S个目标行共享一源行时，非空v行涉及源行数为floor((r+v-1)/S)-floor(r/S)+1；来源v/S+1忽略起点，可能少算。非连续源行也不能强制“一段连续去重搬入”。

小shape固定成本可能主导，大shape才可能带宽主导；来源60–100ns/核ramp、中档4–16核等需目标测量，不保证所有dtype/broadcast同阈值。删除屏障、合并InitBuffer或Host表缓存的收益需完整路径及噪声分析，不能用一批±0.3µs结论认定普遍无效。
