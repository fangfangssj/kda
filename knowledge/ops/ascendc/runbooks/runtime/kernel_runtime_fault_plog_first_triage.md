---
type: Runbook
title: "内核运行期故障初判：先读 plog 提取 errorStr/errorcode 再二分 kernel"
description: "任何 AscendC kernel 运行期故障（507015/507035/EZ9999/aicore exception/errcode）第一步是读设备/主机日志提取 errorStr：从 ASCEND_PROCESS_LOG_PATH / ASCEND_WORK_PATH / $HOME/ascend/log 解析 LOG_ROOT，find -mmin 5 找最新 debug/plog 文件，再按 errorStr→故障类映射表对号入座，不要上来就二分 kernel 计算/DataCopy。"
tags:
- task.debugging
- topic.data_movement
- topic.terminology
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- agnostic
aliases:
- OL-296
---

# 内核运行期故障初判：先读 plog 提取 errorStr/errorcode 再二分 kernel

Kernel runtime fault first-triage — find the plog and extract errorStr/errorcode BEFORE bisecting the kernel.

When an AscendC kernel crashes / hangs / returns a runtime error (507015 / 507035 / EZ9999 / aicore exception / errcode), the FIRST step is to read the device/host log and extract the `errorStr` — do NOT start bisecting the kernel compute/DataCopy. The errorStr usually names the fault class and, via the EC/OL index, points straight at the root cause — avoiding long wrong-bisections (EC-89's host `ws[0]` max-vs-sum bug was mis-bisected as an int8 DataCopy alignment issue for ~40 iterations because the worker never read the log).

### Where the logs land (priority)
Ascend log disk root, by priority:
1. `$ASCEND_PROCESS_LOG_PATH` (if set)
2. `$ASCEND_WORK_PATH` (if set)
3. default `$HOME/ascend/log`

Resolve the root: `LOG_ROOT="${ASCEND_PROCESS_LOG_PATH:-${ASCEND_WORK_PATH:-$HOME/ascend/log}}"`. The runtime-fault detail lives under **`$LOG_ROOT/debug/plog/`** — the primary directory to inspect.

### Find the freshest log
```
find "$LOG_ROOT/debug/plog" -type f -mmin 5
```
(`-mmin 5` = files modified in the last 5 min — the just-crashed run's logs, not stale ones.) Also check the run's per-rank logs and the host-side run stdout.

### Extract the errorStr
```
grep -rEi "errorStr|errcode|507015|507035|EZ9999|aicore exception|out of range|illegal instruction" "$LOG_ROOT/debug/plog" | tail -30
```
Capture the `errorStr:` line + `errcode:(N)` + `subErrType` — these are the triage keys.

### Triage map (errorStr/errorcode → likely class → KB entry)
- `The DDR address of the MTE instruction is out of range` (err95 / 507015) → **host workspace/tiling**: EC-89 (`ws[0]` max-vs-sum) OR GM-extent overshoot (EC-53) OR ACLRT_LAUNCH missing `SetSysWorkspaceForce` (EC-68). Check host `TilingFunc` `ws[0]` FIRST.
- `507035 aicore exception ... illegal instruction` → UB overflow / queue mis-use / undefined symbol (EC-82 InitBuffer, EC-39 MatmulApiStaticTiling).
- `507015 ... fixp_error0/1 subErrType:4` → may be host-level NPU contention, NOT a kernel bug (OL-268) — check `npu-smi` AICore%.
- `507015 subErrType:4` + errorStr `L0 read/write conflict in the MTE (same address)` on a MIX catlass kernel → **catlass Resource lifetime** (EC-92 → OL-297 Rule 1): function-scoped `Arch::Resource` double-ctor corrupts the TPipe/event environment — check ONE-Resource-per-core BEFORE geometry/fill/policy/cadence bisection. (Same errcode+subErrType as the host-contention mimic above, different errorStr.)
- runtime segfault, no AscendC diag → msopgen auto-gen arg-order (EC-88 `EXEC_NPU_CMD`), or MIX `clearWorkspace` (EC-87).
- `507015 subErrType:4` + errorStr `The configuration of FIXP is illegal` → **static Fixpipe-configuration bug, NOT a hang** — an invalid layout/stride parameter reaching the L0C→GM Fixpipe (canonical instance: a catlass `layout::RowMajor` base default-constructed without shape args yields `stride=(0,1)`; `GetTileLayout()` preserves the base stride while replacing the shape, so every tile's row offset collapses and the first Fixpipe is issued with a degenerate C layout). Distinguishing signal at runtime: a fast 507015 return is a hardware TRAP (statically inspectable — audit every layout/stride constructor against the authoritative example BEFORE designing experiments); a 90s watchdog expiry is a hang. Do not spend spawn after spawn on flag-id/barrier/TPipe-lifetime bisection for this signature.
- wrong output, no crash → precision (Cast dtype PB-50/PB-26, sync EC-81 `SToMTE3Sync`, multi-core race OL-214 single-core-first).

### Worker-brief lesson
Debug/verify briefs MUST inject: on ANY runtime fault, BEFORE editing kernel code — (1) resolve `LOG_ROOT`, (2) `find "$LOG_ROOT/debug/plog" -type f -mmin 5`, (3) grep errorStr/errcode, (4) look up the triage map / EC-OL index. This log-first discipline prevents the wrong-bisection trap.
