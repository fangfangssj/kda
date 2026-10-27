---
type: Runbook
title: "An NPU correctness sweep can reach a wrong kernel-corruption verdict from harness defects while the corrupt observations stay unattributed"
description: "An NPU correctness sweep can reach a wrong kernel-corruption verdict from harness defects while the corrupt observations stay unattributed"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-424"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-424 — OL-424: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B150; bisheng=n/a; op_class=any custom pybind kernel adjudicated against a torch/vendor reference。
- 验证来源：a5_ops 4_GroupedMatmulSwigluQuant, 2026-09-08 — non-author sweep vs author re-measurement, both on Ascend950PR_9579 / CANN 9.1.0.B150。

**OL-424** — **An NPU correctness sweep can reach a wrong "kernel corruption" verdict and publish misleading numbers from harness defects, while the corrupt observations themselves stay unattributed: (1) running the reference on NPU while the recorded contract runs it on CPU; (2) relative error divided by an unmasked `x + 1e-12` denominator against scales that legitimately reach `1e-4`; (3) a truth adjudication covering 2 cases labelling a 60-case sweep "truth-verified". A missing `torch.npu.synchronize()` between the custom kernel call and the `.cpu()` read is hygiene, not the explanation: unsynchronised candidate-only scripts ran 840 invocations with 0 corrupt.** Symptom set: intermittent whole-group-sized garbage, bad-case sets in 2 of 3 sweep runs, per-token **scale** values inside the int8 range (`[-13, 127]`), and relative errors of `1e11`/`5.6e4`/`2153`.

- Former A5Ops ID: `OL-424`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150; bisheng=n/a; op_class=any custom pybind kernel adjudicated against a torch/vendor reference`
`verified_on: a5_ops 4_GroupedMatmulSwigluQuant, 2026-09-08 — non-author sweep vs author re-measurement, both on Ascend950PR_9579 / CANN 9.1.0.B150`
`unverified_on: other SoCs and CANN versions — the defects are harness-level and plausibly arch-neutral, but were measured only on Ascend950PR_9579 / CANN 9.1.0.B150; re-check before relying on them elsewhere`

**Related canonical entries — read first**: PB-115（A5Ops 绑定外部引用；本语料未收录目标） predicts this entry's surviving discriminator: a still-queued torch_npu vendor op can silently overwrite a raw-launched kernel's output, stream-level sync does not drain the host queue, and acceptance verifiers that run a vendor reference and a raw-launch candidate in one process are among its predicted instances. PB-106（A5Ops 绑定外部引用；本语料未收录目标）: torch_npu dispatcher ops and a raw launch are not stream-ordered against each other. PB-38（A5Ops 绑定外部引用；本语料未收录目标）: an async release racing kernel execution after launch (V220). PB-115 is verified on Ascend310P3 only; this entry does **not** lift its `unverified_on` for Ascend950PR, because the overwrite signature was not checked here.

**Principle**: before believing a correctness failure, prove the *instrument*. Every number above came through a harness. The author's re-measurement — fixed dataset materialised to disk and reloaded, artifact identified by loaded digest, core count exported and printed from inside the process, `synchronize()` immediately after the candidate call — returned 60/60 under the official comparator on two passes, under three verifier versions, with known-bad `null` and `y+2` both rejected.

**The cheapest detector is a structural invariant, not a tolerance.** Here `out_scale = max|swiglu|/127` is **non-negative by construction**, so `out_scale < 0` needs no reference, no tolerance, and no truth definition to argue about. It settled in one pass (720 executions, zero violations) what truth-comparison arguments could not.

**⚠ Do not stop at the first cause that fits.** A missing `synchronize` was present in 10 of 12 scripts, but it does not explain the corruption: two of those unsynchronised scripts ran the candidate alone and were clean over 840 invocations, and the one correctly synchronised script (`call → synchronize() → .cpu()`) still showed `|dy|` 253/255. Partitioning every observation by configuration gave the discriminator that survives:

    candidate alone                                   840 invocations   0 corrupt
    candidate + reference on CPU                     >720 executions    0 corrupt
    candidate + vendor NPU reference co-resident      every corrupt observation, synchronised or not

⇒ co-residency with vendor NPU ops in one process is a real axis. Status `UNATTRIBUTED / LATENT`: it does not negate a CPU-reference P, and it does not prove a co-resident consumer safe. Accepting "harness artifact" wholesale would have been an over-correction in the author's own favour — the mirror of accepting the failure uncritically.

**Settling experiment**: candidate-alone arm vs co-resident-NPU-reference arm, both correctly synchronised, same fixed dataset, N passes each. Add a third arm that drains the device's host task queue before the raw launch (PB-115's fix), and compare every corrupt value against the vendor reference's own outputs — equality is PB-115's signature.

**Other instances (predicted)**: any op whose reference is a vendor NPU op driven in the same process as the candidate (PB-115's predicted set) — GroupedMatmul family, MoE experts, fused attention. Also any quantised op with a per-token scale output, where the scale carries a sign or range invariant that a comparison-based check will never exploit.

**Corollaries worth carrying**:
- **Bind a bit-exact count to its build.** Two builds gave 38/60 and 42/60 bit-exact while both satisfied `|dy| <= 1` on 60/60 (only the rebuild's source digest was recorded, so "same source" is not established). The pass verdict is a property of the operator; the bit-exact count is a property of the artifact.
- **A dispatch you reason about may decide nothing.** A long argument for why `input_type` resolves to `float` was correct but not load-bearing: forcing it to `float`/`int`/`no_tensor`/a bogus value gave identical pass counts, because both arms reach `delta <= 1`. Test the branch instead of narrating it.
- **Regenerating inputs per pass is not the same experiment.** Materialise once, publish the dataset SHA, reload everywhere — otherwise author and reviewer adjudicate different populations and the disagreement is unresolvable by construction.
