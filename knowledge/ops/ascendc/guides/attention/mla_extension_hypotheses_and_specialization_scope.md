---
type: Guide
title: MLA 扩展假设与 Attention 特化范围
description: 保留扩展点而不把未验证路线写成平台能力。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/composition.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/extension_points.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/_governance.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/specialization/feature_flags.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# MLA 扩展假设与 Attention 特化范围

MLA涉及潜在KV表示、投影及可能的矩阵吸收，不能仅把KV head设为1，再在V1后加入两个逐元素操作就视为完成。共享latent cache也不意味着所有外部接口必须Hkv=1；QK score维度Dqk、value维度Dv及投影shape应由实际方程确定，投影FLOP和流量可能显著。

来源提出用int32中间“修复FP32精度”等说法但无范围、scale和溢出证明，不能采为正确实现。应保存为未建立的假设，并保留真正可复用的扩展位置：Q/K预处理、score偏置/可见性、P格式转换、V/输出投影及最终后量化。每个位置都需要输入输出单位、布局、精度和依赖合同。

特化可按shape、dtype、mask及硬件能力选不同Bq/Bk、G分片、流水深度和执行比例，但模板分支不自动证明所有组合可编译/可执行。容量不等式、映射覆盖、数值与同步证据应分别建立；来源的治理步骤、强制审批/Agent角色和声称最优的流程不作为知识库指令。
