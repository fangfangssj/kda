---
type: Runbook
title: "A kernel that owns its workspace via `GetUserWorkspace` under an `ACLRT_LAUNCH` host stub must call `SetSysWorkspaceForce(workspaceGM)` FIRST, or temporaries land out-of-range → MTE \"DDR addr out of range\" err95 (`507015`)"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; kernel_type=ACLRT_LAUNCH+GetUserWorkspace"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-68-a-kernel-that-owns-its-workspace-via-getuserworksp.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-68", "GetUserWorkspace", "ACLRT_LAUNCH"]
---

# A kernel that owns its workspace via `GetUserWorkspace` under an `ACLRT_LAUNCH` host stub must call `SetSysWorkspaceForce(workspaceGM)` FIRST, or temporaries land out-of-range → MTE "DDR addr out of range" err95 (`507015`)

> 原 OKF v1 卡号：**EC-68**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - a kernel launched through an ACLRT_LAUNCH host stub (host-stub-generated entry, NOT the full CANN GE op-build path) that obtains its own scratch GM via GetUserW

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; kernel_type=ACLRT_LAUNCH+GetUserWorkspace`
`verified_on: soc=Ascend950PR; cann=9.1.T500 — chunk_gated_delta_rule (GDN) light-port 2026-06-15 (122/122 T1 PASS after fix)`

- **Symptom**: a kernel launched through an `ACLRT_LAUNCH` host stub (host-stub-generated entry, NOT the full CANN GE op-build path) that obtains its own scratch GM via `GetUserWorkspace(workspaceGM)` aborts at runtime with an AIV MTE fault `"DDR addr out of range"` / error 95, host return code `507015`. The abort is FAST (not a hang) and fires the instant the kernel writes its first temporary, before any compute progress.
- **Root cause**: an `ACLRT_LAUNCH` host stub passes `workspaceGM` as a plain kernel argument but does NOT set the sys-workspace base. `GetUserWorkspace(ws)` does NOT use the pointer you hand it — it returns `g_sysWorkspaceReserved + RESERVED_WORKSPACE` (the 16MB sys region the runtime auto-allocates), NOT your large allocation. So the kernel writes temporaries into the runtime's small auto sys-workspace and runs off its end → DDR out-of-range. (Compare: the full CANN GE op-build path sets the sys-workspace base for you; the bare `ACLRT_LAUNCH` stub does not.)
- **Fix**: call `SetSysWorkspaceForce(workspaceGM)` as the FIRST statement of the kernel — before `TPipe` construction, before any `matmul`/`MatmulImpl` `Init`. Then `GetUserWorkspace` returns `workspaceGM + 16MB` and the matmul KFC path uses the same base. The host side must allocate ONE workspace sized `16MB(sys) + interWorkspaceSz + stageWorkspaceSz` and pass its base as `workspaceGM`.
  ```cpp
  extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, /*...*/ GM_ADDR workspaceGM, GM_ADDR tiling) {
      AscendC::SetSysWorkspaceForce(workspaceGM);   // FIRST — before TPipe / matmul Init
      GM_ADDR userWs = AscendC::GetUserWorkspace(workspaceGM);  // now = workspaceGM + 16MB
      // ... TPipe pipe; mm.Init(...); ...
  }
  ```
- **Detection**: grep the kernel for `GetUserWorkspace(` with NO preceding `SetSysWorkspaceForce(` in the same `__global__` body, when the launch path is `ACLRT_LAUNCH` (not GE op-build). Runtime smoking gun: clean build/launch, then `507015` / err95 "DDR addr out of range" on the first temporary write.
- **Note**: this also explains the earlier A5 FA-sync probe's identical err95 — that probe never set the sys-workspace base either; its "transpose org-shape" hypothesis was a red herring, the real cause was the missing sys-workspace base.
- **Cross-ref**: EC-60 (`ACLRT_LAUNCH_KERNEL blockDim=0`), CAND-KFC-standalone-bootstrap-teardown (the `SetSysWorkspaceForce` + `REGIST_MATMUL_OBJ` standalone-KFC bootstrap), PB-34 (the GDN light-port that surfaced this).

<!-- 迁移自 porter kb/target/ascendc/（EC-68，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
