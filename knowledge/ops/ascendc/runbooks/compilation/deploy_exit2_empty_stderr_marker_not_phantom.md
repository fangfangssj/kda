---
type: Runbook
title: "`deploy` exit 2 + empty `.last_build.stderr` is not a phantom — the compiler never ran; read the log's last line and check container mounts"
description: "`deploy` exit 2 + empty `.last_build.stderr` is not a phantom — the compiler never ran; read the log's last line and check container mounts"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-190"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-190 — EC-190: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all；CANN=all；phase=deploy（deploy 阶段 harness 行为；两类已知成因：stage 路径对容器不可见、并行部署共享临时文件冲突）。
- 验证来源：A5 容器 back_port_dt，soc=Ascend950DT_9582，CANN 9.0.0（celu kw-1，2026-08-02；成因 2 为 2026-05-02 观测）。

**EC-190** — **`deploy` exit 2 + EMPTY `.last_build.stderr` is NOT a phantom.** Log stops after `Extracting … in <container> from <path>/`; no compiler output because the compiler never ran. TWO known causes: (1) **stage path not visible to the container** — host `scp` succeeds but in-container `tar` cannot open it (OL-274; celu/A5 `back_port_dt` 2026-08-02); (2) shared-temp collision under parallel deploys (OL-123). The real `tar (child): … Cannot open` is captured into `run_with_retry`'s private `mktemp` file and the script exits before writing `$STDERR_MARKER` ⇒ **empty marker means "not yet written", not "no error"** (OL-321). Do NOT retry: read the log's LAST line, then `docker inspect <container> --format '{{json .Mounts}}'`. Open harness gap: write the marker BEFORE returning.

- Former A5Ops ID: `EC-190`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; phase=deploy`
`verified_on: A5 container back_port_dt, soc=Ascend950DT_9582, CANN 9.0.0 (celu kw-1, 2026-08-02)`

**Symptom**: `deploy_to_npu.sh` / `deploy_to_npu_lane.sh` returns **exit 2**;
`.last_build.stderr` is **zero bytes**; the deploy log simply **stops** after a line like
`Extracting … in <container> from <path>/`. No compiler output, because the compiler never ran.
Indistinguishable from a phantom/flaky failure — which is the trap: the natural next move is to
retry, and the retry fails identically.

**Root causes** — this signature has (at least) **two** distinct causes; check both:

1. **Stage path not visible to the container** (celu 2026-08-02). The host-side `scp`
   succeeded, but the in-container `tar` cannot open a file staged outside any bind mount, so
   extraction dies before build. ⇒ **OL-274** for the probe + the
   `A5_/A3_DEPLOY_STAGE_HOST/_CONTAINER` overrides.
2. **Shared-temp collision under parallel deploys** (2026-05-02). Concurrent workers reusing one
   `/tmp/current_task_deploy.tar.gz`. ⇒ **OL-123** (per-process tagging).

**Why the message is invisible** — the actual `tar (child): … Cannot open` text is captured into
`run_with_retry`'s own `mktemp` stderr file. The script exits **before** the code that copies
that file to `$STDERR_MARKER`, so the marker stays empty and the text is never written anywhere
an agent is instructed to look. **An empty stderr marker means "not yet written", not "no error"**
(OL-321 — an empty result is `UNDETERMINED`, not `NO`).

**Detection**: on exit 2 + empty marker, do NOT retry. Read the deploy log's **last** line to
learn which stage aborted, then verify that stage's path against the container's actual mounts:
`docker inspect <container> --format '{{json .Mounts}}'`.

**Harness gap (open, worth fixing rather than documenting)**: `run_with_retry` should write
`$STDERR_MARKER` **before** returning on failure, or echo the first N captured lines into the
log. Until then, every failure in this class arrives as a phantom.

**Cross-ref**: OL-274 (cause 1 + the override anchors), OL-123 (cause 2), OL-321 (empty ≠
absent), EC-44 (the other "deploy exits without deploying" signature — TARGET alias rejected).
backend=ascendc.
