---
type: Guide
title: 原子模式切换与首次覆盖顺序
description: 保持覆盖写先于原子累加，区分 FIX 局部排序与跨核初始化完成。
tags:
- task.debugging
- topic.implementation.atomic
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/sync-checklist.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/sync-mechanisms.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/atomic_operations/set_atomic_add.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/pipe_barrier_isasi.md
  role: constraint
created_at: '2026-09-11T02:35:33Z'
updated_at: '2026-09-11T02:35:33Z'
platforms: []
---

# 原子模式切换与首次覆盖顺序

Split-K或多阶段累计若首片用覆盖写、其他片用AtomicAdd，必须保证首次覆盖先完成；否则晚到的覆盖会抹掉早先已累加贡献。另一种候选是先初始化目标再让全部片原子累加，但初始化完成必须对所有写者可见。不同核之间不能只靠其中一核的PIPE_FIX屏障建立全局初始化顺序。

SetAtomicAdd改变后续受支持的GM搬运写语义，需按接口与DisableDmaAtomic配对，不能让模式泄漏影响其他输出。实际搬运可能来自UB、L0C或L1；应按所用通路选择完成与排序机制，不能把每个DataCopy都当FIX流水。源例在Fixpipe写前加PIPE_FIX以串起同一核的覆盖与累加，表达的是前序写完成后才继续的要求。

来源文字称必须在SetAtomicAdd之后插屏障，给出的“正确”代码却在切换之前；具体放置需遵守目标API对模式切换和前序/后续搬运的合同，不能把位置关键词作为证明，也不能以此推定全部同流水原子指令普遍乱序。最后一次原子搬运与关闭模式/复用缓冲仍要按具体完成语义检查。

原子性不保证浮点结果逐位确定，局部屏障也不固定跨核加法顺序。验证覆盖值与多份非零partial、不同到达顺序、零K/尾K、初始化stream依赖和后续普通输出。记录原始时序/中间值才能把“部分贡献丢失”归于覆盖竞争；本批未重放来源issue或设备程序。
