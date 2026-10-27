---
type: Guide
title: Matmul fallback 资源等式与评分单位
description: 保存SWAT分块链路，区分L0/L1预算及文档中的量纲错误。
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

# Matmul fallback 资源等式与评分单位

参考SWAT按base块、边缘块、尾核、L0C、L1深度、缓冲和字段组装组织候选，MX量化、Batch和Group有不同变体；顺序是该算法组织，不是所有Matmul必须遵循的Skill流程。来源把若干Matmul事实归到conv_api_tiling文件且含自我修正聊天片段，未验证的路径与结论不作官方实现证据。

基本资源分别满足Mb*Kb*dA*DBa≤L0A，Kb*Nb*dB*DBb≤L0B，Mb*Nb*accBytes*DBc≤L0C；不同dtype和容量不能简化为max(Mb,Nb)*Kb*d乘单一L0上限。bias BT约束按真实bias格式与缓冲份数，Nb*4*2只是FP32双份示例。M/N16及K16/32/64应区分Cube接口粒度、MX组32和存储打包约束，不能从来源冲突常量选一个全局规则。

评分1/Mb+1/Nb只能在固定K、相同dtype与流量假设下作强度代理。来源loadBalance分子M*K*N/C而分母ceil(tileCount/C)*Mb*Nb少K，量纲不对；相同K下可用M*N/[C*ceil(tileCount/C)*Mb*Nb]估有效工作/全部波次容量，仍不含搬运、尾块实现和实际调度差异。

L1中的A/B/scale/bias分别算字节再求和。若al1/bl1已是字节，判断48KiB时不能再乘sizeof(dtype)；4缓冲项Mb*k*4仍需dtype。scale按实际驻留K跨度而非仅baseK，align/pack后规模可能超过逻辑ceil(K/32)。资源预算必须覆盖全部同时存活的pipeline槽。

stepK/depth/nBuffer字段各变体并不统一。depth=stepK*DB的约定下depth6对应step3，偶数不推出2的幂；来源{1,2,4}或上限4/8是局部搜索集合，与全载ceil(K/baseK)或其他MC2的step3不构成硬件矛盾。nBuffer2/4是该策略取值，不证明物理单缓冲非法。通用kL1=baseK*min(stepA,stepB)不能覆盖独立kAL1/kBL1；A16无step字段、Group无nBuffer等应按真实变体而非勾选总表解析。
