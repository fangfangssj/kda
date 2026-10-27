---
type: Guide
title: Attention 打分维度与 Value 输出维度
description: 在 MLA 等分段输入路径中分别管理 QK 打分维度、Value 宽度及 latent 数据消费范围。
tags:
- task.design
- topic.compute.composite
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/mla_pattern.md
  role: primary
created_at: '2026-09-10T14:32:33Z'
updated_at: '2026-09-10T14:32:33Z'
platforms: []
---

# Attention 打分维度与 Value 输出维度

Attention 的 QK 归约宽度 Dqk 与 PV 输出宽度 Dv 可以不同。Kernel 的 Q/K 搬运、score 计算、V 搬运、输出跨度及资源预算应分别使用对应维度，不能用一个 headDim 替代全部含义。

在合同将 Q/K 分为 nope 与 rope 两段时，score 的两段点积先合成同一 S，再施加所需 scale、mask 和 softmax。不能对两段独立归一化。默认 scale 若按 Dqk 定义，就不能误用 Dv；显式 scale、位置编码和模型特有缩放仍以实际接口为准。

某些已经完成投影或吸收变换的 latent cache 布局允许 K 路读完整 Dqk、V 路仅读同一行前 Dv 列。这是特定表示的消费合同，不能推广到所有 MLA。`Dqk!=Dv` 本身也不能识别全部 MLA，更不能证明 weight absorption 已完成。

若缓存采用分页或离散选择，K/V 必须指向同一逻辑 token 的相应数据段。即使 workspace 为 Dqk 宽，PV 和最终写回也只覆盖约定 Dv；物理 pitch 可包含额外补齐。验证使用两段独立可识别数据、Dqk/Dv 不同取值、非零 scale 及分页尾块，分别核对合成 score 和输出。
