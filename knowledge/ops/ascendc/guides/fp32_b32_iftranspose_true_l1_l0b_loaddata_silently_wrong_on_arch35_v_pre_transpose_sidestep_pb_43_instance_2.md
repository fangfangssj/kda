---
type: Guide
title: "fp32 (b32) ifTranspose=true L1->L0B LoadData silently wrong on arch35 — V-pre-transpose sidestep (PB-43 instance #2)"
description: "fp32 b32 transpose-fractal L0B load miscomputes on arch35 while ifTranspose=false fp32 loads are bit-exact; kStep-units variants falsified. Mitigation: host pre-transposes V; kernel stages [D,BLOCK_N] and uses the proven ifTranspose=false K-form. Verified bit-exact on 11_BigBirdAttention fp32 case."
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.cube]
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/cube-manual-mmad-crossgen-001.md
  role: supporting
created_at: "2026-09-26T12:23:45Z"
updated_at: "2026-10-16T00:00:00Z"
platforms: ["950"]
aliases:
- OL-612
---

> **来源说明（2026-10-16）**：本卡核心主张的实测证据来自车队私有 run（见 Evidence 节 run_id / verified_on），sources 为同缺陷族公开先例（背景，不承载本卡具体主张）；按社区评审口径降为 draft，待补可复核来源后回升。

# fp32 (b32) ifTranspose=true L1->L0B LoadData silently wrong on arch35 — V-pre-transpose sidestep (PB-43 instance #2)

## Evidence

- **run_id**: 5450b083-4f53-4620-a1b2-8656e7bb9889
- **soc**: Ascend950DT_9582, CANN 9.2.0
- **clock**: 2026-09-26T12:23Z
- **trigger**: 11_BigBirdAttention: fp32 case 25 wholesale-wrong; stage isolation (S exact, P correct, out wrong); sidestep verified bit-exact

## 现象 / 触发
Ascend950DT_9582, CANN 9.2.0, manual-Mmad MIX attention kernel: fp16 path 49/49 PASS,
fp32 path S=QK^T bit-exact and P=exp(t-m) correct, but P@V output element-wise wrong
(max_abs_diff 0.31, 99.95% elements bad). Only the MM2 B-operand load used
`LoadData2DParamsV2{ifTranspose=true}` with fp32; every fp32 `ifTranspose=false` load
(MM1 A/B, MM2 A) was bit-exact.

## 根因
The b32 transpose-fractal L1->L0B load form is silently wrong on arch35 (same class as
[PB-43/cube_manual_mmad_crossgen_001](../runbooks/precision/cube_manual_mmad_crossgen_001.md): V220-era fractal transpose forms compile and run
clean but misplace data on A5). Param-unit variants (16-element vs 32B-block kStep)
falsified — not a units bug.

## 修法（已验证，bit-exact）
为何不采用 PB-43 的处方（route-guard 手工 cube 路径、改用 MatmulImpl）：本 kernel 按设计就是手工 Mmad 的 MIX 结构（AIC/AIV 分工 + 手工流水），不存在可回退的 MatmulImpl 路径——故必须在手工路径内绕开（host 预转置 + 已证 ifTranspose=false 载入形）。


Host passes V pre-transposed ([B,H,D,Skv_pad]) for the fp32 instantiation (pure layout
op); the kernel stages a [D, BLOCK_N] L1 tile via Nd2Nz (nValue=D, dValue=BLOCK_N,
srcDValue=Skv_pad) and loads L0B with the fp32-PROVEN `ifTranspose=false` MM1-K form
(mStep=D/16, kStep=BLOCK_N*sizeof(T)/32, srcStride=dstStride=D/16), yielding
B=(V^T)^T=V [k=BLOCK_N, n=D]. fp32 out went bit-exact (diff=0.0) vs torch reference.

## 判别
- fp32 S exact + P correct + out diffuse-wrong => isolate MM2 B-load.
- fp16 same code PASS => dtype-specific transpose path, not the Mmad.
