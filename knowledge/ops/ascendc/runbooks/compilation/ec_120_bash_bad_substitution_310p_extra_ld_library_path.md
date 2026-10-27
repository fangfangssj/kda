---
type: Runbook
title: "bash bad substitution `${310P_EXTRA_LD_LIBRARY_PATH:-}` — bash identifiers cannot start with a digit; never derive env-var names from target tokens"
description: "bash 标识符不能以数字开头：`${310P_EXTRA_LD_LIBRARY_PATH:-}` 直接 bad substitution，deploy 脚本的 TARGET 前缀变量要加下划线或换字母前缀。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-120"]
---

# bash bad substitution `${310P_EXTRA_LD_LIBRARY_PATH:-}` — bash identifiers cannot start with a digit; never derive env-var names from target tokens

<!-- applies_to_backend: all -->
`applies_to: harness=deploy shell (src/scripts/deploy_to_npu.sh); soc=all (any per-target env templating); cann=n/a; bisheng=n/a; op_class=all`
`verified_on: 12_Permute kw-5 iter-0 2026-09-09 (Ascend310P3, local lane) — symptom reproduced, identifier-guard fix landed same spawn`
- **Category**: build/deploy (shell)
- **Loaded by**: Worker/Generator whenever a per-target env hook (`${TARGET}_EXTRA_LD_LIBRARY_PATH` style) is evaluated on the LOCAL deploy arm, and by anyone templating TARGET tokens into bash variable names (310P/310B are numeric-leading by construction).
- **Symptom**: `bad substitution` on the `${<token>_VAR:-...}` expansion; downstream here: the DEBT-482 local load gate could not resolve `libtorch_npu.so`, and the (correct) endpoint build was rejected as INDEPENDENT_BUILD_ERROR.
- **Mechanism**: bash identifiers may not begin with a digit. `310P`/`310B` make `${310P_EXTRA_LD_LIBRARY_PATH:-}` an illegal expansion at parse time of that line; the `:-` default syntax is irrelevant.
- **Fix**: numeric-leading target tokens skip per-target eval via an identifier guard (`deploy_to_npu.sh:844-847`（a5_ops 路径，未迁移）); toolchain library paths resolve by discovery instead of per-target env vars — `_aog_local_torch_ld` find_spec mirroring the build link dirs (`deploy_to_npu.sh:936`（a5_ops 路径，未迁移）).
- **Evidence**: 12_Permute kw-5 2026-09-09 iter-0 (fixed same spawn). Sibling splice hazard from the same audit (fixed): four remote-arm splice sites from merge bdfba1f8e left the script `bash -n`-broken at `done`/EOF — run `bash -n` over shell scripts after any multi-site merge splice before deploying.
