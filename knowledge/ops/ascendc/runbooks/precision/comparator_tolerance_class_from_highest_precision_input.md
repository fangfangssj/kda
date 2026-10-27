---
type: Runbook
title: "The NPUKernelBench comparator picks its tolerance class from the highest-precision input dtype — an int8 output with any fp32 input is graded by the quantisation class at +-1 LSB"
description: "The NPUKernelBench comparator picks its tolerance class from the highest-precision input dtype — an int8 output with any fp32 input is graded by the quantisation class at +-1 LSB"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-451"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-451 — OL-451: A5Ops additive extension

## 适用场景

- 适用平台与范围：scope=official verify.py / precision_eval consumers grading integer or quantised outputs; soc=all; cann=all。
- 验证来源：official verify.py sha256 7519d7a4b37b84d5c03cb0fb0accb9b7b7028cfca359e5e6dbb6657aef6fd978 (11_GroupedMatmulSwigluQuantV2, 2026-09-11)。

**OL-451** — **the NPUKernelBench comparator picks its tolerance class from the highest-PRECISION input dtype, so an int8 output accompanied by any fp32 input is graded by the QUANTISATION class at +-1 LSB, not by integer strict equality.** `y+1` accepted and `y+2` rejected is the gate working, not the gate broken.

- Former A5Ops ID: `OL-451`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: scope=official verify.py / precision_eval consumers grading integer or quantised outputs; soc=all; cann=all`
`verified_on: official verify.py sha256 7519d7a4b37b84d5c03cb0fb0accb9b7b7028cfca359e5e6dbb6657aef6fd978 (11_GroupedMatmulSwigluQuantV2, 2026-09-11)`

- **Category**: process, trust_calibration
- **Trigger**: reading a known-bad control that accepts a 1-LSB perturbation of an integer output, or reasoning about which tolerance branch a case will take.
- **Mechanism**: `_infer_input_type(inputs)` collects the bare `torch.Tensor` inputs (a `list` of tensors is NOT counted), takes `max(dtypes, key=_dtype_rank)`, and `_dtype_rank` orders by **precision**. With an int8 `x`, an fp32 `x_scale` and an int64 `group_list`, the winner is **fp32**, so `input_type` is `"float"` — not `"int"`. An int8 output then lands in the quantisation branch: `验证失败 dtype=torch.int8 (量化类 fp->int，要求|diff|<=1)`.
- **Measured, by loading the comparator and calling it** on case 0 of the frozen 50: `_infer_input_type -> ("float", torch.float32)`; `y` exact ACCEPTED, `y+1` ACCEPTED, `y+2` REJECTED. This reproduces the boundary an independent non-author review recorded for the same operator's earlier arm.
- **Mechanism — how the verdict is signalled** (scope: every output class of the official comparator — integer, quantised, float, bool and `non_compute` — wider than this entry's header; measured on the `fd2cc3d7` bytes below): `compare()` returns `None` on acceptance and signals rejection only by raising. Value rejections raise `AccuracyError`, which subclasses `AssertionError`; the structural checks (the measured ones are listed below) raise a bare `AssertionError` that is NOT an `AccuracyError`. Two misreadings follow. A caller that reads the return value as a bool records every accepted case as a failure (`bool(None)` is `False`; failures raise, they never return). A caller that catches only `AccuracyError` crashes on a structural rejection instead of recording it.
  - **Measured by calling it** (CPU, 2026-09-12) on cannbot-skills 仓（公开）/ops/triton-op-verifier/scripts/verify.py:611 (cannbot-skills commit `068c77bd`, 2026-07-29; sha256 `fd2cc3d72b43c6f01e6e2356517f38149b13481bb87a47e370e76f7bb86027fc`, bytes identical before and after): identical int8 and fp32 pairs, and int8 `y+1` under the quantisation class, returned `None`; int8 `y+2` (quantisation class), int8 `y+1` (integer class) and fp32 `y+1.0` raised `AccuracyError`; an element-count mismatch (16 vs 15), a NaN-position mismatch, an Inf-sign mismatch and a `non_compute` dtype mismatch each raised a bare `AssertionError`. On the default path (`non_compute=False`) `compare()` flattens both outputs before its shape check, so a same-count reshape (`[4,4]` vs `[2,8]`) returned `None`; with `non_compute=True` the unflattened shapes are compared and the same reshape raised a bare `AssertionError`. Called directly on a `list` candidate output, `compare()` raised `AttributeError` on both paths — not an `AssertionError` at all; a numpy array is converted and graded on the default path. Replaying the consumers: a bool read recorded the accepted pair as `FAIL`; `except AccuracyError` let the element-count and NaN rejections escape as `AssertionError`; `except AssertionError` recorded all three rejections as `FAIL`.
  - **Not checked**: the `7519d7a4` revision that adjudicated the 4-4 evidence below was not checked for the non-`AccuracyError` classes.
- **Why this is worth an entry rather than a footnote — the failure it prevents**: a reviewer working from the source alone concluded the opposite (that the int64 `group_list` outranks fp32, so `y` is graded by `|diff| == 0`) and reported it as measured. **A wrong tolerance claim does not fail loudly — it makes a CORRECT known-bad gate look broken**, and the natural next action is to "repair" a gate that was working. The reviewer had the comparator imported in the same session and still derived the branch by reading `_dtype_rank` instead of calling it.
- **Rules**:
  1. Determine a comparator branch by **calling** `_infer_input_type` and `compare` on the real case, never by reading the ranking helper.
  2. A known-bad control that accepts `y+1` on an integer output is evidence about the **tolerance class**, not about the gate's liveness. Liveness is shown by `null` and `y+2` being rejected.
  3. A `list[Tensor]` argument does not participate in the input-type vote; only bare tensors do. An op whose weights arrive as a one-element TensorList therefore has its class decided by the remaining scalars.
  4. **Adjudicate the official comparator by exception, never by its return value**: `except AssertionError`, which also catches `AccuracyError`, records a rejection. Any other exception from `compare()` is an evaluation error, never PASS. In the official driver the per-case verdict is set in `_try_run_case`: a case passes only if `run_single_case` returns, and its `except Exception` records any exception as a failed case (cannbot-skills 仓（公开）/ops/triton-op-verifier/scripts/verify.py:945-975 (at `068c77bd`)). The official driver never reaches that path for a non-tensor output: `run_single_case` calls `compare()` only when both outputs are tensors (cannbot-skills 仓（公开）/ops/triton-op-verifier/scripts/verify.py:904 (at `068c77bd`)) and skips any other pair without a verdict, so a candidate output that is a numpy array, a `list` or a Python scalar was recorded as a passed case even with wrong values (CPU, `fd2cc3d7`). Check candidate output types before trusting an official PASS. `run_single_case` itself only catches both assertion classes and re-raises them with the case index (cannbot-skills 仓（公开）/ops/triton-op-verifier/scripts/verify.py:913-916 (at `068c77bd`)). This applies to the official `verify.py` only. The repo's own `compare`-named functions return verdicts and must be read by field: the typed precision profiles return `PrecisionDecision(accepted: bool, ...)` (`src/scripts/orchestrator/precision_profiles.py:43-44`), and the retained-case grader's `compare` in `pass_a_case_semantics` returns a `bool` (`src/scripts/orchestrator/pass_a_case_semantics.py:113`). Check which `compare` you hold before choosing how to read it.
- **Evidence**: 内部路径（不公开）agent-team-wiki/projects/op11-groupedmatmulswigluquantv2/RESULT_20260911T0407Z_SCAN_ARMC_P50_F0957_ROOT_CAUSE.md; the independently recorded boundary is in 内部路径（不公开）agent-team-wiki/projects/op11-groupedmatmulswigluquantv2/BINDING_20260908T1530Z_DS_OP11_ARM3B_EXACT_HEAD.md ("y+1 50/50 ACCEPTED <- the documented 1-LSB boundary").
- **Evidence (verdict signalling, rule 4)**: 4-4 `4_GroupedMatmulSwigluQuant` P 60/60 on two passes, adjudicated by `verify.compare()` unmodified at sha256 `7519d7a4…`, whose rejection signal is the raise: known-bad `null` and `y+2` were REJECTED and `y+1` was accepted (`output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/README.md:16-37`; `output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/evidence/precision/P_official_7519d7a4_fixed_2pass.json:10-57`). The same ELF also passed 60/60 in one pass under the `fd2cc3d7` bytes measured above (`output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/evidence/precision/P_official_op44_verifier_20260907T2345Z_fixed.json:3-49`). The 4-4 P driver itself is not retained in the 4-4 archive; the retained exception-adjudicating driver is the 4-11 non-author one, which calls `verify.compare(...)  # raises on failure` under `except AssertionError` (`output/npukernelbench/src/kernels/11_GroupedMatmulSwigluQuantV2/retained_versions/arm3b_current_contract_20260908/evidence/nonauthor/ds_p50_driver.py:137-143`). No incident of a consumer reading the return value as a bool is recorded. A recorded 4-4 near-miss motivates rule 4's "check which `compare` you hold": a false P alarm came from reading a custom `PASS_count` field as the official comparator verdict (内部路径（不公开）agent-team-wiki/agents/ds/SESSION_HANDOVER_20260908T0100Z.md:113-116).
