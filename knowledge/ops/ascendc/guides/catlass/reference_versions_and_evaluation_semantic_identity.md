---
type: Guide
title: CATLASS 参考版本与评测语义身份
description: 保留线性 Attention 参考定位信息及独立证据范围。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/flash-linear-attention-npu-reference.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/open-source-linear-attention-map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/attention/linear-attention.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS 参考版本与评测语义身份

来源记录flashserve/flash-linear-attention-npu的参考commit e7a05dc24ede580fd74f3ffc97d0ea71738e91fb及2026-08-10核对日期，建议定位fla/ops/ascendc/kda，按chunk_kda/chunk_bwd/gated_delta、dAv/bwd_dav/chunk_kda_bwd、cu_seqlens/chunk_indices查入口、阶段及变长映射。本次仅抽取该本地reference的版本记录，未访问或验证此上游commit中的实现；不能将它当作已读一手源码。

实现参考、数学定义、评测baseline和历史工程经验可承担不同证据职责。参考版本及真实文件位置应可追溯，baseline必须与所测阶段、dtype/layout、scale、gate、状态和输出合同一致。一个实现可以同时成为参考与baseline，是否允许复用由任务要求决定；来源禁止特定baseline参与实现是Skill来源政策，不是技术规律。

源码不可获取时，本地摘要只能支持它实际陈述的边界，不能根据路径名称补出ABI、事件协议或性能支持。复用inputs/bench可节省昂贵golden生成，但必须绑定输入hash、shape/属性、生成版本和输出身份；旧报告不能自动证明新Kernel。

远程clone位置、USER_LOCAL/OPEN_SOURCE审批、Agent读取顺序、交付模板和不得扫描本地目录等流程不迁入为知识库规则。Triton实现及其block/grid细节不在本次迁移范围；本卡只保存Ascend C/CATLASS相关的来源定位及证据边界。
