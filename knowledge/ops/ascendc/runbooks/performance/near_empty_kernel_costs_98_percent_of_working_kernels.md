---
type: Runbook
title: "A near-empty kernel of this op costs ~7.7 us of device task time, 98% of what its working kernels cost — and the same profiling window resolves a CANN library task at ~1.4 us"
description: "A near-empty kernel of this op costs ~7.7 us of device task time, 98% of what its working kernels cost — and the same profiling window resolves a CANN library task at ~1.4 us"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-517"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-517 — OL-517: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any kernel declared KERNEL_TYPE_MIX_AIC_1_2 and timed by msprof task_time; build=any。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-46 DAModule, main 2026-09-06, ONE msprof window containing 43 near-empty MIX launches and 13 full forwards。

**OL-517** — **A near-empty kernel of this op costs ~7.7 us of device task time, 98% of what its working kernels cost — and the same profiling window resolves a CANN library task at ~1.4 us, so the instrument can see much shorter tasks than that.** Those two are the measurement. What the ~6.3 us gap consists of, whether it sits before the kernel body, and whether it is recoverable, are **hypotheses this entry sets up and does not settle**. It is deliberately NOT named a "MIX floor": the near-empty kernel happened to be MIX, but nothing here isolates the launch type as the cause.

- Former A5Ops ID: `OL-517`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any kernel declared KERNEL_TYPE_MIX_AIC_1_2 and timed by msprof task_time; build=any`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-46 DAModule, main 2026-09-06, ONE msprof window containing 43 near-empty MIX launches and 13 full forwards`
`unverified_on: AIC_ONLY launches (still not measured). AIV_ONLY was not measured in the first round, but a later arm did measure this AIV construction -- see the engine-declaration result below. Also unverified: other SoCs, other CANN versions`

- **Category**: performance / launch cost
- **Trigger**: an operator decomposed into several kernels, on problem sizes where each kernel's own work is only a few microseconds. The symptom is a per-kernel time that barely moves with the shape.
- **The measurement**: `da_probe_ping` — a MIX kernel whose body is a few `SetValue` plus one 16-byte `DataCopyPad`, and which does **not** read the tiling struct — was launched 43 times in the **same** msprof window as 13 calls of the real 8-kernel forward. Medians: **ping 7.676 us**, `da_final` (real work) **7.832 us**, ratio **0.9801**. Every other stage sat between 7.70 and 8.17 us. Doing nothing costs 98% of doing the work.
- **What the second control does and does not show**: a CANN library kernel, `ZerosLike_4_BYTES_high_performance`, appears in that same window at **1.424 us** median (min 1.140, n=44). That establishes exactly one thing — **this instrument, in this window, resolves tasks far shorter than 7.7 us** — so the ~7.7 us is not profiler granularity and not a floor the part imposes on every task. It does **not** identify what our kernels are spending it on, and a vendor kernel differs from ours in many ways at once.
- **Second control — task counts**: 43 ping tasks for 43 ping calls, 13 tasks for each once-per-forward kernel, and 26 for `da_conv` which is launched twice per forward. The counts pin the attribution: these are the right tasks.
- **⚠️ What does NOT follow yet**: the difference between two *different kernels* does not by itself locate the ~6.3 us before the kernel body, nor attribute it to MIX bring-up specifically. Both are **untested hypotheses**. So are the figures that follow from them — "8 launches spend ~50 us of an ~89 us call" and "work inside a kernel cannot reach it" are arithmetic on the hypothesis, not results.
- **The one adjacent datapoint**: narrowing that op's tiling-struct copy from 88 single-byte GM reads to 22 four-byte reads moved per-launch cost by only ~0.55 us. Consistent with a body-independent cost; also consistent with that particular copy simply being cheap. It discriminates weakly.
- **How to actually settle it**: change the launch type and re-measure the same kernels. Moving the single-engine kernels into their own AIV_ONLY translation unit (per OL-516 and EC-42, a per-kernel annotation will not do it) holds the bodies and the worker count fixed and varies exactly the thing in question.
- **How to run it yourself, cheaply**: put a do-nothing kernel and your real kernels in ONE profiling window and compare medians. Cross-run comparison is not needed and is weaker. Look for a vendor kernel in the same window as a positive control; if none appears, add a trivial `torch` op so one does.
- **A follow-up decomposition (same op, 2026-09-06, one msprof window, identical host wrappers so only the kernel body differs):**
  | kernel | median us |
  |---|---|
  | `da_probe_nop` — body is literally `{}` | **6.844** |
  | `+ TPipe` + one `TBuf` | 6.793 (**-0.051**) |
  | `+ the 88-byte tiling copy` | 6.835 (**-0.009**) |
  | `da_final`, real work | 7.467 (**+0.623**) |
  | cheapest non-ours in that window (n=173) | 1.453 |
  An **empty** kernel costs **91.7%** of a working one and **4.7x** the cheapest vendor task in that window. What this shows is a **large cost shared across this group of probes**: adding `TPipe`/`TBuf`, adding the tiling copy, or doing `da_final`'s actual work each moved the total very little **in these five probe cases**. It does not identify what that shared cost is, and the mechanism is **still to be isolated**.
- **Still a hypothesis, from source only**: the auto_gen wrapper ends every kernel, unconditionally under `__DAV_C310__`, with `pipe_barrier(PIPE_ALL); dsb(mem_dsb_t::DSB_ALL); dci();` and opens it with `SetSysWorkspaceForce` / `GetUserWorkspace` on the workspace pointer. That is toolchain-generated and body-independent, which fits — but it was **not measured**, and nothing here isolates it.
- **Changing the engine declaration did not show the expected magnitude of reduction**: moving four single-engine kernels into an AIV_ONLY translation unit (verified in the generated config) moved them 34.166 -> 33.087 us, **x0.9684 — about 3.2%**, while three kernels left as MIX in the same runs moved x1.0027. The untreated group not moving means the manipulation was specific and the instrument had the resolution to see it; a ~3% change is far short of what a launch-type explanation predicted. That bounds the engine declaration's contribution in **this** configuration; it does not prove the launch type is irrelevant in general.
- **What it suggests for practice, as a direction to test rather than a result**: if a large part of the cost really is shared per launch, then editing kernel bodies cannot reach it and **reducing the number of launches** is the thing to try. Treat `8 x 6.844 us` arithmetic as a **scenario calculation, not a measured fixed tax**: it assumes the whole per-launch cost disappears with the launch and that fusion adds nothing, and neither is established. The benefit across the full 186-case denominator has to come from a new candidate measured end to end.
- **Cross-ref**: OL-516 (per-kernel `KERNEL_TASK_TYPE` is ignored by auto_gen, so acting on the engine declaration means splitting translation units, not annotating kernels). backend=ascendc.
