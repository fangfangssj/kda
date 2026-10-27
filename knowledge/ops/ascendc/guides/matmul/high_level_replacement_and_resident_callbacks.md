---
type: Guide
title: Matmul 高阶替换与常驻回调坐标
description: 保留 Host、单核尺寸、输出模式和 L1 自定义回调的增量约束。
tags:
- task.implementation
- topic.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P46_matmulimpl_highlevel_api.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P46_matmulimpl_highlevel_api.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P53_l1_resident_reuse.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P46.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P53.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate_all.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/set_org_shape.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/set_single_shape.md
  role: constraint
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# Matmul 高阶替换与常驻回调坐标

高阶 Matmul 可以承担输入搬入、L1/L0 管理、矩阵计算和输出等职责，但配置及执行模式仍需与实际 shape、dtype、layout、转置、bias、workspace 和参与核一致。来源 MatmulImpl/CFG_MDL 是特定框架版本的写法；不能把仅有 aT/bT/cT 类型别名的片段当作所有版本完整模板，或从“160行减到10行”推导性能与正确性。

SetOrgShape 描述原矩阵形状及寻址跨度，SetSingleShape 描述当前单核有效范围，SetTensorA/B 绑定本次数据起点。三者不能相互代替；尾M/N和K区间需与Host tiling一致。来源 `IterateAll<true>(dst,false)` 的 true 对应同步模式，第二个false对应不开启原子操作，不能把它解释成异步调用。单次输出容量至少容纳实际 singleCoreM*singleCoreN，还需与真实写出布局匹配；其他模式的结果取得方法见[执行模式合同](execution_mode_and_host_contracts.md)。

MDL 的 depthA1/depthB1、stepKa/stepKb 与数据复用、搬运粒度及容量相连，不只由 K 一个数字决定。自动管理不保证自动选择最优K重排；不支持的布局可采用独立路径，但“非标准layout”本身不是所有高阶接口的统一禁令。框架对象初始化、workspace设置、结束与异步等待按具体合同完整实现。

来源 L1 常驻回调用 reuseLeft 选择 GetBuffer/AllocTensor，再取 `callTimes*baseWidth*64` 的元素偏移。这是特定数据排布中的子块序号；baseWidth与64的单位、实际slot容量及每次加载跨度需一致。若 callTimes 在每次 N 复用时继续无界累加，不重置到相应子块范围，会返回已加载区之外。复用需要稳定的 M/K/batch/head/转换键、正确加载完成信号和生命周期，不能仅依据布尔标志。

Q常驻期间，其L1区不能同时交给BMM2的P或其他写者。来源128*576*2=144KiB可作为该布局的容量算例，却不证明M>128必然不可常驻或常驻必须小于L1的20%；例如在512KiB假设下144KiB已占28.125%，两个建议并非同一硬件定律。多轮减少GM读取需权衡首轮集中加载、可用tile及其他阶段资源。

验证不同tail、重复对象调用、callTimes重置、切换复用键和框架/手写路径数值一致，测完整时延与资源峰值。本文仅做固定来源的静态合同提炼，未编译来源模板。
