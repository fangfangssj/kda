---
type: Runbook
title: "pgrep -f is a discovery hint, never a process identity — keep and wait on the launcher-recorded PID and re-bind command line, start time, and process tree before any kill"
description: "pgrep -f is a discovery hint, never a process identity — keep and wait on the launcher-recorded PID and re-bind command line, start time, and process tree before any kill"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-501"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-501 — OL-501: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=n/a; bisheng=n/a; op_class=all; scope=Linux host-process supervision。
- 验证来源：Linux procps-ng pgrep plus /proc PID identity on the controller host, 2026-08-22。

**OL-501** — `pgrep -f` is a discovery hint, never a process identity: a literal can match the waiter or peer waiters while missing the real target when intervening flags break the substring. Keep and wait on the launcher-recorded PID, assert that exact command's rc, and before any kill/resource action re-bind its command line, start time, and expected process tree; mismatch means refuse, never `pkill -f`.

- Former A5Ops ID: `OL-501`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=n/a; bisheng=n/a; op_class=all; scope=Linux host-process supervision`
`verified_on: Linux procps-ng pgrep plus /proc PID identity on the controller host, 2026-08-22`
`unverified_on: non-Linux hosts and supervisors without /proc/<pid> identity fields`

**Principle.** `pgrep -f` is a discovery hint, never a process identity. It applies a regex to
rendered argv; it does not know which process a launcher created or which command a controller
intended. One literal can therefore fail in both directions at once:

- **false positive** — the waiter and peer waiters carry the searched literal in their own argv,
  so the self-check reports them as the target. A diagnostic command containing the target string
  can likewise make its own shell satisfy the probe.
- **false negative** — the real target has intervening flags or wrapper arguments, so the selected
  words are not one contiguous argv substring even though they describe the intended command.

A zero exit from `pgrep` can consequently mean "I found my waiter while missing the workload."
Filtering one spelling such as `grep -v bash` only moves the boundary; another wrapper or peer
waiter recreates the ambiguity.

**Rule.** Keep the process handle returned by the launcher: wait on the recorded PID, then assert
the exact command's return code. This completion path does not need `/proc`: do not try to re-bind a
PID after `wait()` has reaped it. For detached or resumed supervision, persist a controller-bound
identity containing PID plus process start time and command line (or a Linux pidfd when available),
rather than reconstructing identity from a name search. `pgrep` remains useful for enumeration and
forensics, but every candidate must be read back through `/proc/<pid>` and classified explicitly.

**Action-time binding.** A destructive or resource-owning action (`kill`, lane reclaim, reschedule,
or "device is free") requires a fresh action-time identity check: the recorded PID's command line,
start time, and expected process tree must still match the spawn receipt. An absent PID, a reused
PID, an argv mismatch, or unexpected descendants is a refusal, not permission to fall back to
`pkill -f` or the nearest name match. Bounded readiness polling belongs only to fixture setup; an
action authorizer takes one fresh identity snapshot and fails closed when any `/proc` field is
absent, empty, or changes during validation.

```python
proc = subprocess.Popen(argv)             # PID/handle comes from the spawn
spawn_id = bind_running_tree(proc.pid)     # cmdline + start time + expected descendants
if destructive_or_resource_action:
    current = bind_running_tree(proc.pid)  # one fresh read immediately before the action
    if current != spawn_id:
        refuse_action()
    signal_exact_pid_or_pidfd(proc.pid)
rc = proc.wait()                           # independent exact-PID completion observation
assert rc == expected_rc                   # never infer completion from pgrep disappearance
```

- **Known-bad control**: a hermetic target whose argv inserted `-c <code>` between the executable
  and marker was absent from the literal `pgrep -f` result, while the bash waiter carrying that
  literal matched and made `pgrep` return zero. Reproduction:
  `src/scripts/tests/test_kb_process_monitor_pid_binding.py:150`.
- **Positive control**: the one-read action check refused an argv mismatch, a changed start time
  that simulates PID reuse, an absent PID, and an unexpected live descendant. The exact PID,
  start time, argv, and tree then authorised SIGTERM, and `wait()` returned `-SIGTERM`.
  Reproduction: `src/scripts/tests/test_kb_process_monitor_pid_binding.py:192` and
  `src/scripts/tests/test_kb_process_monitor_pid_binding.py:233`.
- **Evidence**: 2026-08-22 background process monitoring reproduced the same joint signature:
  waiter/peer-waiter self-matches plus a missed real target caused by intervening flags. No product
  process was identified or acted on through the ambiguous match.
- **Other instances (predicted)**: build/profiler wait loops, CI job monitors, remote-wrapper
  supervision, lane/resource reclaimers, and any self-check that embeds its own search target.
- **Cross-reference**: OL-121 covers the separate remote-signal propagation boundary; OL-122 covers
  a separate file-marker wait-loop ambiguity. Neither makes a process-name match an identity.
