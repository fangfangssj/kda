---
type: Runbook
title: "Free, healthy and unclaimed is not usable: card admission requires a live, externally-timed execution, and a timeout must reject"
description: "Free, healthy and unclaimed is not usable: card admission requires a live, externally-timed execution, and a timeout must reject"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-512"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-512 — OL-512: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=any Ascend NPU on a shared host; cann=any; bisheng=n/a; op_class=any; build=any lane that claims a device before running。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — shared host 2026-09-05 (degraded card reproduced twice from fresh containers; admission probe with offline positive/negative controls)。

**OL-512** — **FREE, HEALTHY and UNCLAIMED is not USABLE: card admission requires a LIVE, EXTERNALLY-TIMED execution, and a timeout must REJECT.** Observed on a shared A5 host: `npu-smi` reported Health OK / Error NA / "No process in device" and no container mapped the device, while a 512x512 matmul + `torch.npu.synchronize()` TIMED OUT in a fresh container, twice; a container restart did not clear it. Occupancy checks, health bits and claim scans all read "free" — none of them EXECUTES anything. Admission must therefore be: SoC readable AND device node openable AND a real matmul + explicit `synchronize()` completing inside a hard deadline. **TIMEOUT ⇒ REJECT, never "inconclusive, retry"** — retrying is how a bad card gets taken on the second look. backend=all.

- Former A5Ops ID: `OL-512`
- Fusion disposition: `additive_new`

`applies_to: soc=any Ascend NPU on a shared host; cann=any; bisheng=n/a; op_class=any; build=any lane that claims a device before running`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — shared host 2026-09-05 (degraded card reproduced twice from fresh containers; admission probe with offline positive/negative controls)`
`unverified_on: the UNREAPABLE branch — a child surviving SIGKILL because the driver call has not returned cannot be produced without a hung device, so that path is logic-tested only`

- **Category**: shared-resource safety / admission
- **Trigger**: taking any NPU on a shared host.
- **Implementation notes that are load-bearing**:
  - the failure signature is a **HANG, not an error**, so an in-process timeout (SIGALRM) does NOT work: Python signal handlers do not run while a thread is inside a native call, so the probe blocks instead of rejecting. Bounding must come from OUTSIDE the process — spawn the probe as its own PID, poll to a deadline, then SIGKILL.
  - `synchronize()` is mandatory; an async launch alone returns "fine".
  - after a timeout, VERIFY the child is actually gone, and report `REJECT_TIMEOUT_REAPED` vs `REJECT_TIMEOUT_UNRESOLVED` — a client-side rc and an empty host process table both fail to prove the DEVICE side is clear.
  - ⚠️ **two numbering spaces**: a container given one `--device /dev/davinciN` exposed it to torch as index 0 while `npu-smi -i N` on the host still used N. Feeding one number to both checks the WRONG card — and a wrong card reads "no process", i.e. it fails in the direction that looks clean. (Observed for a single-card container; re-read it rather than assuming indices are always 0.)
- **Second lesson, easily missed**: an op's PRECISION leg passed 50/50 with the very setting that later hung the device, because that leg does not run msprof and never reaches the failing path. **Passing one gate is not evidence about a path it does not execute.**
- **Scope**: record such a card as "persistent anomaly across new containers, root cause UNDETERMINED" unless the mechanism is established; a timeline is not proof that a preceding change damaged hardware. "Other cards were not touched" may be stated; "other cards are unaffected" is not a global guarantee — probe each.
- **Cross-ref**: OL-510 (a probe that hangs instead of rejecting is that family), OL-506. backend=all.
