---
type: Runbook
title: "框架从 op_tiling/lib/（而非 op_api/lib/）加载 host tiling 库——残留旧 opmaster 伪装成 GetTilingData null"
description: "aclnn DoTiling 阶段从 op_impl/ai_core/tbe/op_tiling/lib/<arch>/ 加载 libcust_opmaster_rt2.0.so；构建工具只同步 op_api/lib 会留下过期 opmaster，TilingFunc 改动看似全部无效、verify 持续报 GetTilingData null / do tiling failed（EZ9999）——先查实际加载的 opmaster mtime（strings 探针），再怀疑 tiling 注册。"
tags:
- task.debugging
- topic.compilation
- topic.tiling
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- agnostic
aliases:
- OL-301
---

# 框架从 op_tiling/lib/（而非 op_api/lib/）加载 host tiling 库——残留旧 opmaster 伪装成 GetTilingData null

The framework loads the MC2/AscendC host tiling library from `op_tiling/lib/<arch>/` (NOT `op_api/lib/`) — a build tool that stages only `op_api/lib` leaves a stale opmaster → host `GetTilingData<T>()` null illusion (the true root cause of the matmul_allreduce GetTilingData null).

**Principle**: the framework's aclnn DoTiling stage loads the host tiling library (`libcust_opmaster_rt2.0.so`) from `op_impl/ai_core/tbe/op_tiling/lib/<arch>/`, **NOT** from `op_api/lib/`. If the build tool stages only `op_api/lib` and misses `op_tiling/lib` (+ the `liboptiling.so` symlink), the consequence: after every rebuild `op_api/lib` is new, but `op_tiling/lib` still holds the FIRST (stale) opmaster → the framework loads the stale one → every host TilingFunc change (adding a probe printf, changing the tiling scheme, adding/removing the `GET_TILING_DATA` macro) **appears to take no effect**, and verify keeps reporting `GetTilingData returned null` / `do tiling failed (EZ9999)`.

**Camouflage**: it presents as "built artifact ≠ executed artifact" — the freshly built `op_api/lib` artifacts ARE new, but what the framework actually loads (`op_tiling/lib`) is old. Easily misdiagnosed as a tiling-registration issue (OL-268); but if the null persists after OL-268's registration fix (restoring the full `GET_TILING_DATA` macro set), it points to THIS entry.

**Triage order** (for `GetTilingData null` / `do tiling failed`):
1. First check the opmaster the framework actually loads (the install-prefix `op_tiling/lib/<arch>/libcust_opmaster_rt2.0.so`): is its mtime from the latest build? Run `strings` on it — is your newly added probe printf there?
2. Stale → **build staging gap (this entry)**, not tiling registration.
3. Fresh yet still null → only then investigate the tiling registration style (OL-268).

**Fix**: drop manual copy-based staging steps in the build tool and switch to `cmake --build --target binary` followed by `--target install` with `-DCMAKE_INSTALL_PREFIX` pointing at a dedicated install directory, which stages `op_tiling/lib` + `liboptiling.so` + `op_api/lib` + the kernel binary + `npu_supported_ops` completely. Key cmake detail: `--target install` does NOT build the kernel binary automatically (`binary` is not an install dependency) — run `binary` first, then `install`; `ASCEND_PACK_SHARED_LIBRARY=True` hangs `cmake --install` at `libcustomize.a` — use **False (package mode) + install** for a complete stage.

Cross-ref: [[OL-268]] (the two tiling registration styles — same `GetTilingData null` symptom but a different root cause; triage staging FIRST (opmaster mtime), registration second).
