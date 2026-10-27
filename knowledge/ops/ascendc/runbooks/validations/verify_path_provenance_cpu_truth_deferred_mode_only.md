---
type: Runbook
title: "verify_path_provenance sanctions CPU-truth (cpu_canonical) only in deferred mode (marker-gated) and recognizes the P140 pybind runner artifacts (kernel/pybind11.cpp + pass_a_runner.py), not just legacy *_runner.cpp/aclnn_*.cpp"
description: "verify_path_provenance sanctions CPU-truth (cpu_canonical) only in deferred mode (marker-gated) and recognizes the P140 pybind runner artifacts (kernel/pybind11.cpp + pass_a_runner.py), not just legacy *_runner.cpp/aclnn_*.cpp"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-541"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-541 — OL-541: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (harness/port_a3 verify-path-provenance gate — chip-independent); cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5。
- 验证来源：soc=Ascend310P3 (dav-m200, CANN 9.0.0)。

**OL-541** — verify_path_provenance` sanctions CPU-truth (`cpu_canonical`) ONLY in deferred mode (marker-gated) and recognizes the P140 pybind runner artifacts (`kernel/pybind11.cpp` + `pass_a_runner.py`), not just legacy `*_runner.cpp`/`aclnn_*.cpp`.** (1) `cpu_canonical` joins `SANCTIONED_TRUTH_SOURCE_SUBSTRINGS` only when `.a3_ref_unavailable_cpu_truth_deferred`/`.truth_source_override` is present — CPU-truth is a fallback SOURCE, never a default; an op with a reachable aclnn that used CPU-truth (e.g. top_k_top_p_sample) is correctly rejected. (2) the runner-artifact check accepts the P140 pybind path but NOT `model_new_ascendc.py` alone (no pybind import ≠ real-NPU proof) — extends the OL-160 entry-point net. Decision rule (3 branches) in the body. Anchor `port_a3/__init__.py:698` + `:53` + `:745/:752` + `:781`. Evidence: ge_glu_grad_v2 310p kw-3 (A3 ref unreachable → deferred cpu_canonical). backend=ascendc.

- Former A5Ops ID: `OL-541`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=all (harness/port_a3 verify-path-provenance gate — chip-independent); cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5`
`verified_on: soc=Ascend310P3 (dav-m200, CANN 9.0.0)`
`unverified_on: soc=Ascend950PR (A5) / Ascend910_220x (A3) — the same check_verify_path_provenance gate serves port_a3 on those SoCs; principle transfers, observed only on the 310P P140-pybind port so far.`

**Principle**: The port_a3 `verify_path_provenance` gate (the anti-fraud check that a PASS verdict was produced by a REAL NPU verify path, not a wiring-only / CPU claim) has two non-obvious behaviors that were gaps before this op's kw-3 and are now codified in the gate:
1. **CPU-truth (`cpu_canonical`) is conditionally sanctioned — neither blanket-accepted nor blanket-rejected.** `SANCTIONED_TRUTH_SOURCE_SUBSTRINGS` accepts the real-NPU substrings (`aclnn`, `a3_cann`, `a3_capture_via_pybind_aclrtlaunch_kernel`). `cpu_canonical` is added to that sanctioned set ONLY when the workspace carries an orchestrator-set deferred-mode marker (`.a3_ref_unavailable_cpu_truth_deferred` or `.truth_source_override`) — i.e. CPU-truth is legitimate ONLY when the A3 reference is genuinely unreachable / missing-entry and the orchestrator explicitly deferred to CPU. WITHOUT the marker, `cpu_canonical` is rejected: an op whose `aclnn` entry EXISTS but whose worker used CPU-truth anyway (e.g. top_k_top_p_sample) is correctly flagged as not-a-real-NPU-verify. Lesson: CPU-truth is a fallback truth SOURCE, never a default; the gate enforces "deferred-only" via the marker, so a worker cannot silently substitute its own CPU code for a missing vendor reference (the OL-68 / 2026-05-24 scope-narrowed anti-circular-validation rule, mechanically enforced here).
2. **The runner-artifact recognition covers the P140 pybind path, not just the legacy aclnn-direct names.** Pre-P140 the gate looked for `*_runner.cpp` / `*_shim.cpp` / `aclnn_*.cpp` at workspace root. The P140 standalone-pybind path uses `kernel/pybind11.cpp` (the pybind host glue) + `pass_a_runner.py` (the canonical Pass A verifier) instead. The gate now accepts those P140 artifacts — but deliberately does NOT accept `model_new_ascendc.py` ALONE (a dummy file with no pybind import doesn't prove real-NPU execution). This extends the OL-160 / `_check_universal_entrypoints` mode-entry-point safety net to recognize the P140 runner artifacts, so the fraud gate does not silently no-op on the standalone-pybind path (the exact failure mode of the 2026-05-14 name-coupling incident).

**Concrete anchor** (resolving citations):
- gate entry — `src/scripts/orchestrator/plugins/port_a3/__init__.py:698` (`def check_verify_path_provenance`)
- the sanctioned-substring constant — `src/scripts/orchestrator/plugins/port_a3/__init__.py:53` (`SANCTIONED_TRUTH_SOURCE_SUBSTRINGS`)
- conditional `cpu_canonical` acceptance — `src/scripts/orchestrator/plugins/port_a3/__init__.py:745` and `:752` (`effective_sanctioned = SANCTIONED_TRUTH_SOURCE_SUBSTRINGS + ("cpu_canonical",)` inside the marker branch only)
- the deferred-mode marker files — `src/scripts/orchestrator/plugins/port_a3/__init__.py:68` (`.a3_ref_unavailable_cpu_truth_deferred`, `.truth_source_override`)
- P140 pybind runner-artifact recognition — `src/scripts/orchestrator/plugins/port_a3/__init__.py:781` (accepts `kernel/pybind11.cpp` / root `pybind11.cpp`); the error message enumerating the P140 names alongside the legacy ones — `src/scripts/orchestrator/plugins/port_a3/__init__.py:802`
- regression test for the marker-gated accept/reject — `src/scripts/orchestrator/tests/ut/test_a5_verify_path_fraud_gate.py:202`

**Decision rule** (when this gate fires on your op — pick the branch by the rejection message):
- gate rejects `truth_source='cpu_canonical'` AND no deferred marker present → the op HAS a reachable aclnn/A3 reference; switch `truth_source` to the real-NPU path. Do NOT drop a marker to force-accept — that re-opens the circular-validation hole.
- gate rejects `truth_source='cpu_canonical'` AND the A3 reference IS genuinely unreachable → the legitimate deferred case; the ORCHESTRATOR (host-side, not worker-faked) sets the `.a3_ref_unavailable_cpu_truth_deferred` marker, after which `cpu_canonical` is sanctioned.
- gate rejects "no recognized runner artifact" on the P140 pybind path → ensure `kernel/pybind11.cpp` (with a real pybind import of the kernel `.so`) + `pass_a_runner.py` exist; `model_new_ascendc.py` alone is insufficient proof.
