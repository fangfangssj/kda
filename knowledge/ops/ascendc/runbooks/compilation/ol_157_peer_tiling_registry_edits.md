---
type: Runbook
title: "arch22->arch35 port: peer-tiling-registry edits when the op-family shares a tiling-template library"
description: "Porting into a family that shares a tiling-template library fails to compile (unknown type name '<RegbaseTilingDataClass>') until 3 peer-side regbase tiling registrations are added."
tags: [task.debugging, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-157-peer-tiling-registry-edits.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-157"]
---

# arch22->arch35 port: peer-tiling-registry edits when the op-family shares a tiling-template library

> 原 OKF v1 卡号：**OL-157**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - _apt.cpp fails to compile with `unknown type name '<RegbaseTilingDataClass>'` inside `GET_TILING_DATA_WITH_STRUCT(...)` when porting into an op-family that shares a tiling-template library

## 现象 / 触发

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=op_family_with_shared_tiling_registry`.
Source: `workspace/foreach_erf/knowledge_update.md` (kw-1, 2026-05-13).

Porting an op into an op-family whose peer dependency is a **shared tiling-template library** (a
`*_utils` / otherwise non-op library that owns `op_host/<family>_tiling_def.h` and exports a
tiling-key → tiling-template registry) fails at compile with
`unknown type name '<RegbaseTilingDataClass>'` inside `GET_TILING_DATA_WITH_STRUCT(...)` in the
kernel's `_apt.cpp`, because the op has not been wired into the A5 regbase registry.

**Detection signal**: grep `<family>_tiling_def.h` for
`REGISTER_TILING_DATA_CLASS(<CurrentOpName>_<tilingkey>, <RegbaseTilingDataClass>)`. If only the
membase line `REGISTER_TILING_DATA_CLASS(<CurrentOpName>, <MembaseTilingDataClass>)` exists and the
per-dtype regbase lines are MISSING, the op is not yet registered against the A5 regbase data layout.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

This is the sibling of OL-131 (aclnn-router peer-edit pattern), but here the peer dependency is a
shared tiling-template library rather than an aclnn router — the peer exports neither an op kernel
nor an aclnn entry, just a tiling registry that downstream ops register into. Porting an op into this
family on A5 requires **3 peer-side edits** beyond the local `op_host` + `op_kernel` (a mechanical
mirror of any already-A5-ported sibling op in the family):

1. **`<family>_tiling_def.h`** — add per-dtype regbase tiling-key register lines after the existing
   membase REGISTER:
   ```cpp
   REGISTER_TILING_DATA_CLASS(<OpName>_10001, <FamilyRegbaseTilingData>)  // half
   REGISTER_TILING_DATA_CLASS(<OpName>_10002, <FamilyRegbaseTilingData>)  // float
   REGISTER_TILING_DATA_CLASS(<OpName>_10004, <FamilyRegbaseTilingData>)  // bf16
   ```
   The `_10001/2/4` suffix is the family's per-dtype `TILING_KEY` (defined in
   `op_kernel/arch35/<family>_regbase_common.h`). The regbase data class encodes A5-specific layout
   constants (`MAX_TENSOR_CONT_950`, `MAX_CORE_CONT_950`).

2. **`<family>_tiling_func.cpp`** — three sub-edits:
   - (a) Replace the hardcoded `Tiling4<OpName>Tiling` body with a registry dispatch
     `return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);`, switching
     `IMPL_OP_OPTILING` from "membase-only hardcoded" to "registry-dispatched" so priority-based
     selection between membase and regbase templates works.
   - (b) Declare the membase tiling class:
     `MEMBASE_TILING(<OpName>, <FAMILY>_<OP>_OP_CODE, <FamilyInputType>);`.
   - (c) Register the membase template at lower priority via
     `REGISTER_OPS_TILING_TEMPLATE(<OpName>, ...)` (register the regbase template above it so the
     regbase path is selected on A5).
