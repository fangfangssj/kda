---
type: Runbook
title: "A signature-level API-parity check is not a semantic-parity check — a parameter can match by name, type and position, be accepted, and never be read"
description: "A signature-level API-parity check is not a semantic-parity check — a parameter can match by name, type and position, be accepted, and never be read"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-308"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-308 — OL-308: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all (API-compatibility harnesses, reimplementations of an upstream reference)。
- 验证来源：a communication-library reimplementation audited against its upstream reference, 2026-07-31 -- AST-level scan, host-only, positive control included。

**OL-308** — **A signature-level API-parity check is not a semantic-parity check: a parameter can match by name, type and position, be accepted, and never be read, while the harness prints `0 failures`.** An AST name+order parity test reported "API parity machine-verified, 0 failures"; adding an **accept-and-ignore gate** (parameter in the signature, absent from the body) surfaced four live cases in one library (2026-07-31, host-only). The worst: an optional round-trip channel documented upstream as *"pass the dispatch-returned weights to also gather them back"* was accepted and dropped, the reimplementation producing the equivalent result from the caller's own input — yet **still returning** the bypassed channel, carrying a value it no longer depended on and had never verified. Measured: that value's live region is **non-deterministic run-to-run** while a payload tensor from the same call is bit-identical across runs (**the asymmetry is the finding, not any max-diff**); six call sites forwarded it and the receiving function declared the parameter and never read it ⇒ blast radius zero, and **structurally** so rather than by luck; the first party to actually read it was burned and mis-attributed the error to their own instrument for several rounds. Also found: an `async_finish`-style flag accepted and ignored while the returned event handle was unconditionally `None` ⇒ the caller believes it is asynchronous. **Silent acceptance is worse than `NotImplementedError`, which at least reaches the caller** — the library's two unconditionally-raising entry points served as the gate's positive control (correctly NOT flagged). **Rule**: pair every signature check with a body-reachability check that exits non-zero; for any parameter accepted but not honoured, raise — or document it inert **and stop returning a value for it**. "Nobody has read this value" and "this value is fine" are indistinguishable in source.

- Former A5Ops ID: `OL-308`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all (API-compatibility harnesses, reimplementations of an upstream reference)`
`verified_on: a communication-library reimplementation audited against its upstream reference, 2026-07-31 -- AST-level scan, host-only, positive control included`
**Principle**: when a reimplementation is validated against an upstream API by comparing
**parameter names and order**, the checker confirms the *shape* of compatibility and says
nothing about whether any accepted argument reaches a computation. Such a harness reports
"API parity machine-verified, 0 failures" over parameters that are dead on arrival. The
resulting defect class is invisible in the most literal sense: **"nobody has read this value"
and "this value is fine" are indistinguishable in the source.**
**Instance.** A signature-comparing parity test (AST: names + order) passed cleanly. Adding
an **accept-and-ignore gate** — parameter appears in the signature but never again in the
function body — turned up four live cases in one library, including one where an optional
round-trip channel documented upstream as *"pass the dispatch-returned weights to also gather
them back"* was accepted and dropped, because the reimplementation produced the equivalent
result from the caller's original input instead. Consequences observed:
```
- the bypassed channel was still RETURNED to callers, carrying a value the implementation
  no longer depended on and had never verified -> measured run-to-run NON-DETERMINISTIC
  content in the live region, while a payload tensor returned from the same call was
  bit-identical across runs (the asymmetry, not any max|diff|, is the finding)
- six call sites forwarded that value onward; the receiving function declared the parameter
  and never read it -> the garbage travelled a full round trip and was silently ignored
  => today's blast radius was ZERO, and STRUCTURALLY so, not by luck
- the first person to actually read the value was burned by it, and mis-attributed the
  resulting error to their own instrument for several rounds
- worst sub-case: an `async_finish`-style flag accepted and ignored while the returned
  event handle was unconditionally None -> the caller believes it is asynchronous
```
⇒ **Silent acceptance is worse than an unimplemented error**, because `NotImplementedError`
at least reaches the caller. The library's two genuinely unimplemented entry points, which
raise unconditionally, were the gate's **positive control** — correctly *not* flagged.
**Rule**: for any compatibility harness, add a body-reachability check alongside the signature
check, and make it fail (non-zero exit). For any parameter you accept but do not honour:
raise, or document it as inert **and stop returning a value for it**. When a returned value is
no longer on your own critical path, either verify it or remove it — an unverified return that
matches upstream's *shape* is a trap primed for the first caller who trusts the docstring.
**Second instance — a kernel wrapper, with a precision consequence** (4_GroupedMatmulSwigluQuant, Ascend950PR_9579 / CANN 9.1.0.B150, 2026-09-07): the pybind wrapper received the contract's per-token input scale `x_scale` and discarded it (`(void)xScale;`); the kernel entry had no such parameter. The operator still built and ran; the measurement below was taken on it. Against a truth that applies the scale, the candidate matched 0/12 cases bit-exact (max|dy| 21-85); after the parameter was threaded pybind -> entry -> kernel, the rebuilt arm (`ds_l4_04_CURCONTRACT_232353Z`, `.so` `a16c818752b3c81e`) was bit-exact on 38/60 with the rest at |dy| = 1 — that bit-exact count belongs to that build (OL-424). The scale does not cancel through the non-linear activation (P-P250), so the omission is a correctness defect, not a no-op. The check that would have fired on day one: for every contract input, assert that it reaches a kernel read — grep the wrapper for `(void)<name>` and for inputs with no kernel parameter. Evidence (agent-team-wiki @5433fae0): `projects/op4_4/RESULT_20260907T2340Z_DS_OP4_4_CURRENT_CONTRACT_REBIND.md`.
**Cross-ref**: OL-297 (a check whose output is identical with the defect present and absent —
this is that, at API granularity), OL-305 (apparatus is under test), OL-306 (a shared
parameter name is not a shared contract). backend=all.
