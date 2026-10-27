---
type: Runbook
title: "Pass B bit-exact comparison must normalize stored high-precision CPU truth to the declared output dtype before comparing"
description: "Pass B bit-exact comparison must normalize stored high-precision CPU truth to the declared output dtype before comparing"
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
# OL-273 — OL-273: Pass B bit-exact comparison must normalize stored high-precision CPU truth to the declared output dtype before comparing

> 关联：同一 OL-273 主题已由既有卡 [ol_273_dtype_templated_ub_widest_dtype_507035.md](../compilation/ol_273_dtype_templated_ub_widest_dtype_507035.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all; harness=edge_dataset; phase=PassB。
- 验证来源：soc=Ascend910_9382; cann=9.0.0。

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all; harness=edge_dataset; phase=PassB`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 evidence not yet attached; this is a harness-schema rule and expected to transfer when the same edge_dataset contract is used)`

### Principle

When Phase O2.5 stores a CPU-truth edge output at higher precision than the op's declared output dtype, Pass B must compare the kernel output against the oracle cast back to the declared output dtype. A direct `torch.equal(out_declared_dtype, ref_high_precision)` is a schema mismatch, not a precision failure, when the numeric diff is already zero after dtype normalization. The Pass B report should state the cast in its method so future readers know that bit-exactness was checked against the user-visible output contract, not against the storage dtype of the truth bundle.

### Concrete anchor

```python
# edge_dataset.pt may store CPU truth as fp64 even when the op returns fp32.
ref = case["outputs"]["y"].detach().cpu().to(declared_output_dtype)
out = model_new(*npu_inputs).detach().cpu()
status = "PASS" if torch.equal(out, ref) else "FAIL"
```

### Evidence

- codex_e2e_add_a3 (2026-07-06/07, A3 Ascend910_9382, CANN 9.0.0): `edge_dataset.pt` stored CPU-truth outputs as fp64 while the benchmark signature returned fp32. Direct equality against fp64 failed 8/9 with `max_abs=0.0`; `pass_b_runner.py` cast `outputs["y"]` to the declared fp32 dtype before `torch.equal`, and Pass B then reported 9/9 bit-exact. Evidence cross-check: `workspace/codex_e2e_add_a3/pass_b_runner.py` contains the cast and `workspace/codex_e2e_add_a3/verification.json` records `method: pass_b_runner.py edge_dataset.pt CPU truth cast to declared fp32 output dtype`.

### Other instances (predicted)

- Any edge-dataset verifier where the truth bundle stores fp64 CPU values for a declared fp32, fp16, bf16, int, or bool output.
- Multi-output ops where only some outputs are stored high-precision; cast each reference output independently to the declared dtype before equality.
- Cross-backend verifiers that reuse the same Phase O2.5 edge bundle contract.

Cross-ref: [OL-97](ol_97_precision_target_cpu_truth_vs_cann_bit_match.md) (CPU-truth vs peer-reference precision target), OL-262 (check the truth source before blaming the kernel), OL-270 (harness case-schema contract).
