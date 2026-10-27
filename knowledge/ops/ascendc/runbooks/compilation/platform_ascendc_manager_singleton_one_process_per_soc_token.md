---
type: Runbook
title: "`PlatformAscendCManager::GetInstance(socVersion)` is a process-wide singleton — one process per SoC token"
description: "`PlatformAscendCManager::GetInstance(socVersion)` is a process-wide singleton — one process per SoC token"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-80"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-80 — PB-80: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR；CANN 9.1.0-beta.1；op_class=all（host 侧 A/B 中变化 SoC token 的场景）。
- 验证来源：soc=Ascend950PR_9579 + Ascend950PR_958b；CANN 9.1.0-beta.1（FlashKDA A5，2026-08-01）。

**PB-80** — **`PlatformAscendCManager::GetInstance(socVersion)` is a process-wide singleton**: on the measured Ascend950PR/CANN 9.1 beta path, a second SoC token in the same process can reuse the first platform object. **Rule: one process per token.** `~1.00×` plus bit-identical output is only a suspicion; prove the comparison with fresh-process arms and distinct resolved platform/tiling identity before interpreting it. Cross-ref OL-297（A5Ops 绑定外部引用；本语料未收录目标）, OL-305（A5Ops 绑定外部引用；本语料未收录目标）, OL-330（A5Ops 绑定外部引用；本语料未收录目标）, OL-331（A5Ops 绑定外部引用；本语料未收录目标）.

- Former A5Ops ID: `PB-80`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (host-side A/B that varies the SoC token)`
`verified_on: soc=Ascend950PR_9579 + Ascend950PR_958b; cann=9.1.0-beta.1 (FlashKDA A5, 2026-08-01)`

**Symptom**: a host-side A/B that runs two different SoC tokens back-to-back inside one process reports the two
arms as **bit-identical output at ~1.00× time**. There is no error and no warning. The result reads as a clean,
well-controlled null — "the token makes no difference" — and it is the single most convincing shape a wrong
measurement can take.

**Mechanism**: the platform object is obtained via a **singleton** accessor:

```cpp
// 内部路径（不公开）flash_kda_fwd_launch.cpp — the launch path under test
auto *plat = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
```

The **first** call in the process fixes the platform object. A later call passing a *different* `socVersion`
returns **that same first object** — the argument is ignored once the instance exists. The second arm therefore
runs the first arm's platform configuration (core counts, `platform_config` derived tiling), so the A/B compares
a configuration against itself. Identical output is the correct result of the test that was actually run; it is
simply not the test that was intended.

**Rule: one process per token.** Fork/exec a fresh process for each SoC token and compare across processes.
Nothing at the call site can be varied to make an in-process A/B valid.

**Detection boundary.** `~1.00×` plus bit-identical output is a reason to suspect configuration collapse, not
proof of it: equal deterministic configurations may be bit-identical and a displayed ratio may round to 1.00.
Repeat each token in a **fresh process**, record the resolved platform attributes/tiling identity, and require the
two intended configurations to differ before interpreting their output or timing comparison. The same-process
result may be retained as the known-bad arm, but it cannot establish either a hardware null or singleton reuse by
itself.

**Why this entry is worth its length — the second-order failure.** A recorded "hardware refutation" of a real
defect was produced exactly this way. It listed the (correct) lane-drop hypothesis as dead, and was then **cited
as established fact to push back on the correct report — the defect survived two rounds on that citation.**
The general form is worth internalising: **a recorded refutation inherits every defect of the test that produced
it, and the citation displays none of them.** A result, once written down, loses its method; downstream it is
indistinguishable from a sound one. So: do not accept a null on a defect you have independent reason to suspect
without first checking the *process boundary* of the run that produced it.

**Cross-ref**: OL-297 (write the defect-present and defect-absent outputs before trusting a check), OL-305
(controls and baselines are themselves under test), OL-330 (bound a claim by an independent budget before
quoting it), OL-331 (a metric that looks like it answers the question and does not).


Warning: truncated output (original token count: 32642)
Total output lines: 432
