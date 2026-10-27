---
type: Runbook
title: "Fixpipe L0C→GM `dstStride` unit is PER-FORMAT — elements for NZ2ND (`CFG_ROW_MAJOR`, the default) vs 32-byte blocks for NZ — and Nd2Nz `srcDValue` is a GM row stride in elements [910B2C / 220x / CANN 8.5.1]"
description: "Provenance: contributed by the customer's Kimi-K3 + a5_ops gated_delta_rule bring-up (PR #200, author liuyu15819). Reconciled onto latest main (DS 2026-07-20): this is the CONCRETE arch-220x hand-cube"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-279-fixpipe-l0c-gm-dststride-unit-is-per-format-elemen.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a2, a3]
aliases: ["OL-279", "dstStride", "CFG_ROW_MAJOR", "srcDValue"]
---

# Fixpipe L0C→GM `dstStride` unit is PER-FORMAT — elements for NZ2ND (`CFG_ROW_MAJOR`, the default) vs 32-byte blocks for NZ — and Nd2Nz `srcDValue` is a GM row stride in elements [910B2C / 220x / CANN 8.5.1]

> 原 OKF v1 卡号：**OL-279**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Provenance: contributed by the customer's Kimi-K3 + a5_ops gated_delta_rule bring-up (PR #200, author liuyu15819). Reconciled onto latest main (DS 2026-07-20):

> Provenance: contributed by the customer's Kimi-K3 + a5_ops gated_delta_rule bring-up (PR #200, author liuyu15819). Reconciled onto latest main (DS 2026-07-20): this is the CONCRETE arch-220x hand-cube stride-param knowledge the DS capstone/T1 probe identified as the universal #51 gap; it grounds P-P119's "wrong Fixpipe unit → 507015" note. applies_to preserved as-declared (910B2C / 220x / CANN 8.5.1); the customer PR's deprecated bare arch-alias rewritten to canonical `220x` (terminology gate #171). ⚠ Concrete stride values verified on CANN 8.5.1 — re-verify on CANN 9.1.0 (a3/Ascend910_9382) before relying on them there (pre-merge device gate; same 220x arch → expected to transfer).

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A2 系列、Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910 (910B2C, arch 220x); cann=8.5.1; bisheng=n/a; op_class=user_cube_manual_mmad`
`verified_on: soc=Ascend910 (910B2C, arch 220x); cann=8.5.1 (gated_delta_rule, 2026-07-17~20; unit semantics read from dav_c220/kernel_operator_fixpipe_v2_impl.h GetGMLength)`
`unverified_on: soc=Ascend950PR (351x/A5) — OL-197 class: 220x load/move idioms do not transfer blindly; re-derive from the arch35 impl before relying. Also re-verify on CANN 9.1.0 (a3/Ascend910_9382) before use there.`

**Principle**: for `Fixpipe(GlobalTensor, LocalTensor, FixpipeParamsV220)` the meaning of `dstStride` depends on the `FixpipeConfig` format, and the DEFAULT is `CFG_ROW_MAJOR` (nz2nd): in nz2nd mode `dstStride` is `dst_D` — the row-to-row stride of the destination ND matrix in **elements** (impl: `gmLen = (ndNum-1)*dstNdStride*ele + (mSize-1)*dstStride*ele + ...`). In NZ (non-nd) mode it is in **32-byte units** (`(cburstNum-1)*dstStride*32`). A stride computed in the wrong unit either compresses the output rows (elements-vs-32B under-count) or sprays them far out of the intended region. Similarly `Nd2NzParams.srcDValue` is the source ND matrix's row stride in **elements** — for a `[B, T, H, D]` GM tensor, the time-contiguous per-head row stride is `H*D` (NOT `D`; `D` reads head-interleaved rows).

**Concrete anchor**: writing a compact `[64,64]` fp32 block per head with `FixpipeParamsV220{nSize=64, mSize=64, srcStride=64, dstStride=64 /* elements, compact */}` is the verified-good nz2nd config; scattering the same block into a `[B,T,Hv,128]` output needs `dstStride = Hv*128` **elements**, and a `K^T` workspace read `Nd2Nz{nValue=64, dValue=64, srcDValue=padT}` uses the padded-time row stride.

**Evidence**: gated_delta_rule Cube K@K^T bring-up (2026-07-17~20): `srcDValue = D (=128)` read head-interleaved rows (fixed to `H*D = 512` → exact KKT); compact-block Fixpipe `dstStride=64` verified bit-exact (max_err 8e-6). Unit semantics confirmed in CANN 8.5.1 source `x86_64-linux/asc/impl/basic_api/dav_c220/kernel_operator_fixpipe_v2_impl.h` (`GetGMLength` nz2nd branch) and `kernel_operator_fixpipe_impl.h` comment ("910b soc, dst_stride in unit of 32B" for the NZ path).

**Other instances (predicted)**: any hand-rolled cube epilogue writing rectangular tiles to strided GM layouts (multi-head outputs, chunked workspaces); any Nd2Nz operand staging from `[B,T,H,D]`-family tensors.

**Cross-ref**: P-P119 (the a3 hand-rolled single-block cube primitive whose "wrong `Fixpipe srcStride`/dstStride unit → 507015 ECC read" note this OL grounds), `fa_class/cv_reference_concrete_params.md` (the concrete matmul-primitive param grounding doc P-P119 references), OL-197 (A5-side 2D `LoadData2DParamsV2` — do NOT port these 220x idioms blindly), OL-224 (manual-cube tail-K / transpose-A correctness), P-P117 (the a3 gated_delta_rule kernel this was verified in).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-279（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
