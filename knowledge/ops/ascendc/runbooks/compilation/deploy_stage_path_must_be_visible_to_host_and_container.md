---
type: Runbook
title: "Deploy stages must use a path visible to both host and container; host /tmp may not be mounted inside the A3 build container"
description: "Deploy stages must use a path visible to both host and container; host /tmp may not be mounted inside the A3 build container"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-274 — OL-274: Deploy stages must use a path visible to both host and container; host `/tmp` may not be mounted inside the A3 build container

> 关联：同一 OL-274 主题已由既有卡 [ol_274_507035_vendor_op_probe_before_infra.md](ol_274_507035_vendor_op_probe_before_infra.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all; target=a3; container=npu-a3; phase=deploy_stage。
- 验证来源：host=198.51.100.92; container=npu-a3; soc=Ascend910_9382; cann=9.0.0。

`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all; target=a3; container=npu-a3; phase=deploy_stage`
`verified_on: host=198.51.100.92; container=npu-a3; soc=Ascend910_9382; cann=9.0.0`
`unverified_on: other A3 hosts/containers (mount layout is host-specific; probe the bind mount before relying on `/tmp`)`

### Principle

An A3 deploy script that uploads a tarball on the host and extracts it from inside Docker must choose a stage directory that is mounted into the target container. A host-local `/tmp/...` path can exist and accept the upload but be invisible to the container, causing extraction to fail before build with empty or missing remote markers. Prefer an explicitly bind-mounted directory such as `/home/npu_user` for both the host and container stage roots, and keep the normal deploy script in charge of packaging, upload, extraction, file-count verification, build, and archive creation.

### Concrete anchor

```bash
A3_DEPLOY_STAGE_HOST=/home/npu_user
A3_DEPLOY_STAGE_CONTAINER=/home/npu_user
DEPLOY_STAGE_DIR=op_<op>_lane<lane>_<run_id>
bash src/scripts/deploy_to_npu_lane.sh --lane <lane> --build
```

The deploy log should show the same container-visible root during extraction, for example: `Extracting ... in npu-a3 from /home/npu_user/<stage>/`, followed by matching local/remote file counts.

### Evidence

- codex_e2e_add_a3 (2026-07-06/07, A3 host 198.51.100.92, container `npu-a3`): the worker brief supplied `DEPLOY_STAGE_DIR=/tmp/op_codex_e2e_add_a3_lane0_1783394350`; the first deploy failed because host `/tmp` was not mounted in the container. Re-running the same `deploy_to_npu_lane.sh --lane 0 --build` path with `A3_DEPLOY_STAGE_HOST=/home/npu_user` and `A3_DEPLOY_STAGE_CONTAINER=/home/npu_user` uploaded, extracted, verified `local=17 files, remote=17 files`, built successfully, and archived iter_002 at `/data/build_archive/codex_e2e_add_a3/op_codex_e2e_add_a3/iter_002`. Evidence cross-check: `workspace/codex_e2e_add_a3/.cc_stream_log_aog-kernel-worker_1.jsonl` records the mounted-stage env and successful deploy log; `PROGRESS.md` records the P6 decision to rerun the normal deploy script with mounted stage roots instead of manual copying.

### Other instances (predicted)

- Any remote deploy flow where packaging/upload happens in the host namespace and extraction/build happens in a Docker container.
- A3/A2 containers whose home or data directories are bind-mounted but `/tmp` is container-private.
- Build recovery runs where empty remote markers or missing tarballs appear despite a successful host-side upload.

Cross-ref: [EC-30](ec_30_deploy_to_a5_sh_syncs_current_task_not_workspace_o.md) (deploy script reads staged `current_task`, not arbitrary workspace edits), EC-44 (A3 deploy target alias handling), CAND-DEPLOY-STAGE-LOCAL_TASK (a different deploy-stage canonicality failure: stale `LOCAL_TASK` in standalone worker mode), ANTI_PRESSURE_PROTOCOLS.md P6 (fix or use the deploy path, do not paper over with manual copies).
