---
type: Guide
title: 优化器与 Foreach 分发映射
description: 保存具体Key、类型组合与状态能力，限定泛化收益。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_02_template_kernel.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md
  role: supporting
created_at: '2026-09-11T05:51:16Z'
updated_at: '2026-09-11T05:51:16Z'
platforms: []
---

# 优化器与 Foreach 分发映射

| 算子 | 来源分发 | 适用约束 |
|---|---|---|
| ApplyAdamWV2 | 101..106同构BF16/FP16/FP32×step float/INT64，107..110混合状态FP32/grad低精度×step类型 | 源仅给部分GetTilingKey分支，不猜每个具体数字对应的完整顺序；step值与类型另需验证 |
| ApplyAdagradD | ASCENDC_TPL参数schMode、updateSlots、dType | 两个DAG分别更新/不更新accum，注册输出与别名必须一致，DoTiling模板选择不是直接Host调用Device Compute |
| Foreach Abs/AddList/AddScalarList/AddcDivList | 1half、2float、4BF16示例，部分家族另有INT32 | T存储/P计算及adapter参数个数不同，不能只换类名；half/half与BF16/float不是同一中间精度 |
| InplaceAddRMS | key10half、20float、30BF16 | 其mode在其他版本可能另编码，必须同Host版本 |
| EmbeddingDenseGradV2 | 0基础、1频率缩放、10/11确定性及scale组合、100/101小维及scale组合 | 小维≤512是来源策略域；determinist名称不能代替固定贡献顺序/冲突证明 |
| AdaptiveAvgPool3D | key=10*mode+dtype | 与AddRMS的10*dtype+mode相反，不能互换解码 |

DequantBias<XTYPE,WSTYPE,BIASTYPE,YTYPE,IFBIAS>源X为INT32、scale BF16/FP32、bias BF16/FP16/FP32/INT32、y BF16/FP16。模板参数列表不是这些类型的全笛卡尔积，IFBIAS=false路径仍需完整处理activation scale等独立可选项；固定INT8输入baseline属于另一合同，不能作为同算子精度比较。

RMSNormGrad T_DY/T_GAMMA、Gemma T/T_GAMMA和Pooling T1/T2/T3各有不同角色，不能根据模板位置约定统一类型含义。Host必须对实际支持集合、step/scalar格式与alias输出作校验，不能让剩余else进入默认合法类型。所有源声明的8–16倍Cast收益、2–4倍带宽利用、20–50%整体收益未附本批测量，作为候选设计而非实测保证。
