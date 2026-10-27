---
type: Guide
title: APACE AllToAll 变体组合边界
description: APACE AllToAll 变体组合边界的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/index.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE AllToAll 变体组合边界

固定A2A范式交换K分片输入后计算本rank输出M段。通信data+scale同channel、data段后scale段、self本地直读及source-rank K贡献累加是可复用合同。更换dtype、bias、scale和切分轴都必须同步Host实参、模板策略、物理layout、窗口和golden；类型存在不证明所有组合可实例化。

去MX scale至少涉及MatmulWithScaleMx策略、Params中的scale地址、ResetGmAddr、Tensor/Slice/BlockMmad调用及Host tiling/分配/通信对象。纯BF16/INT8是待证实扩展，不仅删除两个入口参数。bias通道在kernel存在，UDMA SetupParams尚未传而hcomm有例，仍需核对isBias、bias地址、L1/BT预算和rank累加时机。

来源PUT开发A4混入C-ready/V-done环形slot和__mix__描述，与本范式主要AIV→AIC输入就绪、全量Win不复用的设计不同；不能照搬GET/compute-first反向flag。指令顺序、consumer实际等待pipe、最终完成必须来自同一编排，不能拼多个模板的局部片段。

调tileCnt须重新派生依赖它的通信字段与计算调度；若某计算tiling经证明与T独立，可复用其不变量部分，而非机械每次全重算或全复用。单轮512KiB是未定边界经验，改变粒度前还要检查flag/掩码/tail合法性。源码无调用边界保护时，Host拒绝真实非法组合是必要的；未验证不是数学不支持。
