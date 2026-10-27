---
type: Runbook
title: "工程化 op-build 路径 host tiling 的 ws[0] = usrWorkspace + sysWorkspace（求和），绝不能取 max——否则 MTE DDR 越界"
description: "工程化 op-build 路径（GetWorkspaceSizes/msopgen/GE）下框架在 kernel 入口把 workspace 指针前移越过 system workspace，host tiling 必须设 ws[0] = usrSize + GetLibApiWorkSpaceSize()（求和）；写成 max 会少配总量，首个写入 user workspace 的 MTE 报 'The DDR address of the MTE instruction is out of range'（err95/507015 类）。"
tags:
- task.debugging
- topic.compilation
- topic.tiling
- topic.memory
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- agnostic
aliases:
- EC-89
---

# 工程化 op-build 路径 host tiling 的 ws[0] = usrWorkspace + sysWorkspace（求和），绝不能取 max——否则 MTE DDR 越界

Engineering op-build path (`GetWorkspaceSizes`) host tiling `ws[0]` must be `usrWorkspace + sysWorkspace` (SUM), NOT `max` — else kernel writes user workspace out-of-range → MTE "DDR address of the MTE instruction is out of range".

- **Severity**: HIGH (kernel hangs / crashes mid-run; the AscendC errorStr is easy to misread as a compute / DataCopy alignment bug, sending the worker on a long wrong bisection).
- **Symptom**: op compiles + installs clean, launches, then hangs or crashes mid-run. Ascend runtime log errorStr: **`The DDR address of the MTE instruction is out of range`** (err95 / 507015-class). Fires on the first MTE (DataCopy) that writes into the user-workspace region.
- **Root cause** (Ascend workspace contract, engineering op-build path): device workspace = **system workspace** (`GetLibApiWorkSpaceSize`, reserved for AscendC API temporaries) **+ user workspace** (the op's own scratch). In the **engineering op-build path** (`GetWorkspaceSizes` / msopgen / GE), the framework inserts code at the kernel entry that **advances the incoming workspace pointer past the system workspace** — so the `__global__` kernel's `GM_ADDR workspace` arg is ALREADY offset to the user-workspace start. Therefore the host tiling MUST set `ws[0] = usrSize + sysWorkspaceSize` (**sum**). Writing `ws[0] = max(usrSize, sysWorkspace)` understates the total: the framework still advances by `sysWorkspace`, leaving a user region of `max(usr,sys) - sys < usrSize` (or `0` when `sys >= usr`) → kernel user-workspace writes overrun into the system workspace / unallocated DDR → MTE instruction DDR out of range.
  - Ascend doc (operator development → workspace): `currentWorkspace[0] = usrSize + sysWorkspaceSize;` (SUM). Kernel-direct path (`HAVE_WORKSPACE`) documents the same: the entry workspace pointer is the offset (post-sys) user workspace.
- **Fix** — host `TilingFunc`:
  ```cpp
  size_t* ws = context->GetWorkspaceSizes(1);
  size_t sysWorkspace = static_cast<size_t>(plat.GetLibApiWorkSpaceSize());
  ws[0] = usrWsBytes + sysWorkspace;   // SUM — never max(usrWsBytes, sysWorkspace)
  ```
- **Detection**: grep host `op_host/*.cpp` tiling for `ws[0] = .*libApi` / `max(.*libApi)` / `(wsBytes > libApi) ? wsBytes : libApi` / `GetWorkspaceSizes`. Any `max`-style total-workspace assignment (instead of `usr + sys`) is the bug. Runtime smoking gun: Ascend log errorStr `The DDR address of the MTE instruction is out of range` on an op whose host tiling uses `max`.
- **Authoritative source**: Ascend C 官方文档：**如何使用 workspace** — device workspace = system workspace (`GetLibApiWorkSpaceSize`, reserved for AscendC API temporaries) + user workspace; engineering op-build path contract: `currentWorkspace[0] = usrSize + sysWorkspaceSize`; the framework advances the kernel-entry workspace pointer past the system workspace (kernel-direct path sets `HAVE_WORKSPACE` for the same offset semantics). This entry codifies that contract; the `max` form violates it.
- **Cross-ref**: EC-68 (same workspace-offset contract, but the ACLRT_LAUNCH kernel-direct path — there the kernel owns the workspace via `GetUserWorkspace` and MUST call `SetSysWorkspaceForce(workspaceGM)` first; the engineering op-build path offsets automatically and needs NO `SetSysWorkspaceForce`, only the SUM in host tiling). EC-53 (GM-extent overshoot, same err95 family). PB-22 / EC-23 (int8 DataCopy alignment — the WRONG-hypothesis trap this bug lures workers into).
- **Worker-brief lesson**: a worker debugging MTE "DDR out of range" on a workspace-writing op MUST check the host `TilingFunc` `ws[0]` (max vs sum) BEFORE bisecting kernel DataCopy/alignment. stw_brief / kw_brief should inject this host-first check (see DEBT for stw_brief debug-method gap).
