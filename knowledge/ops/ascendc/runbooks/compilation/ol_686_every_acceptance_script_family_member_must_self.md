---
type: Runbook
title: "Every acceptance-script family member must self-source its CANN environment — sibling-script env asymmetry produces a \"precision works, perf crashes\" split-brain"
description: "验收脚本家族的每个成员都必须自己 source CANN 环境——兄弟脚本环境不对称会产出\"精度能跑、性能崩\"的分裂状态。"
tags: [task.debugging, topic.compilation, topic.performance.analysis, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-686"]
---

# Every acceptance-script family member must self-source its CANN environment — sibling-script env asymmetry produces a "precision works, perf crashes" split-brain

`applies_to: soc=all (acceptance-harness / verification-script design); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.1.0 (sort 310p benchmark-acceptance lane, 2026-08-18)`
**Principle**: an acceptance harness is a FAMILY of sibling scripts (precision leg,
perf leg, profiling wrappers) invoked from bare orchestrator/agent shells that carry
NO CANN environment (nothing in /etc/profile.d or the shell rc files — the CANN env
comes from the launcher process only, and a `command -v msprof` check finds msprof on
PATH without implying CANN PYTHONPATH). EVERY member of the family must self-source
CANN `set_env.sh` itself, with a fallback chain (ASCEND_HOME_PATH →
ASCEND_TOOLKIT_HOME → /usr/local/Ascend/ascend-toolkit/latest); never rely on the
caller's env. Asymmetry — one sibling self-heals, another doesn't — produces a
split-brain signature: the precision leg is green while the perf leg crashes EVERY
profiled app in GE init (`ModuleNotFoundError: No module named 'tbe'` →
`AclSetCompileopt(ACL_PRECISION_MODE)` error 500001). The failure reads like a
perf-infra mystery; it is an env-contract bug in one family member.
**Concrete anchor**: cannbot-skills 仓（公开）/ops/ops-profiling/scripts/msprof_profile_run.sh
(the perf leg that did NOT self-source — fixed to source set_env.sh with the fallback
chain, cannbot-skills commit 9bf6922d) vs its precision-leg sibling
cannbot-skills 仓（公开）/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh
(which already self-sourced).
**Evidence**: sort 310p (2026-08-18, first instance, kw-2/kw-3): the perf leg lost
10/10 cases × ref+asc to the missing self-source while the precision leg passed the
same workspace; post-fix 10/10 cases measured both sides (geomean 0.814×).
- **1_BatchMatmul kw-2 2026-09-06 — INVERSE-hazard extension: the env contract is bidirectional; on this CANN host, PRESENCE of any `PYTHONPATH` breaks GE init exactly like its absence** — setting even `PYTHONPATH=/tmp/empty_dir` reproduces `GEInitializeV2 failed / No module named 'tbe' / error code 500001` at `torch_npu._C._npu_init()`. Trigger shape: the PR #103 canonical `benchmark_torch_npu_profiler.py` resolves callables via `resolve("module:function")` → needs the driver importable → the tempting "just export PYTHONPATH to the workspace" fix IS the trap. Compliant driver: importlib `spec_from_file_location` + `exec_module` (with a `__main__` guard) plus runtime `sys.path.insert(0, .../kernel/build)` for the extension — env stays PYTHONPATH-free (`代码片段 1_BatchMatmul/perf_w5a5_runner.py.snippet.py（见文末）`). Same error signature as the sort 310p split-brain above (tbe → 500001), OPPOSITE root cause: env pollution vs env absence — check both directions before blaming the CANN install.
**Other instances (predicted)**: any harness script family whose members are launched
from minimally-configured shells (CI runners, agent spawns, cron) — profiling
wrappers, report generators, finalize-gate re-measure scripts.
**Cross-ref**: OL-661 (the OTHER perf-leg env pitfall from the same op's kw-3 — device
steer via `ASCEND_RT_VISIBLE_DEVICES` vs here the CANN env itself; a perf leg needs
BOTH), OL-180 (the in-process half — .so loading needs CANN env / torch_npu import
order). backend=all.

## 内嵌代码片段

### 1_BatchMatmul/perf_w5a5_runner.py.snippet.py

```python
                      "benchmark_torch_npu_profiler.py")
_TRACE_ROOT = "/tmp/profiler_trace_1bmm_w5a5"

sys.path.insert(0, os.path.join(_HERE, "kernel", "build"))  # NO PYTHONPATH env

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
import batch_matmul_ext as _ext  # noqa: E402

_spec = importlib.util.spec_from_file_location("bm_profiler_canonical", _CANON)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)
```

## 适用场景

- **工程形态**：验收脚本家族的每个成员都必须自己 source CANN 环境。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
