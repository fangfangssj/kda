---
type: Runbook
title: "A model signature, declared JSONL, generated edge manifest, forwarded-argument adapter, and one-case ref_runnable certify different slices — bind every coverage claim to the exact artifact SHA"
description: "A model signature, declared JSONL, generated edge manifest, forwarded-argument adapter, and one-case ref_runnable certify different slices — bind every coverage claim to the exact artifact SHA"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-502"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-502 — OL-502: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all; scope=benchmark-fixture-analysis。
- 验证来源：host-only exact-byte audit of one benchmark fixture, 2026-08-14; this entry makes no kernel-correctness, performance, or product-completion claim。

**OL-502** — A model signature, declared JSONL, generated edge manifest, forwarded-argument adapter, and one-case `ref_runnable` certify different slices. Bind every coverage or runnable claim to the exact artifact SHA, case selector, and arguments actually forwarded; missing coverage is `UNMEASURED`, not inherited from a neighboring artifact. Crossformer exact-byte anchor: the bundled helper drops five optional/mode arguments; the retained generated manifest covers 60 cases while the preflight proves one small output only.

- Former A5Ops ID: `OL-502`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; scope=benchmark-fixture-analysis`
`verified_on: host-only exact-byte audit of one benchmark fixture, 2026-08-14; this entry makes no kernel-correctness, performance, or product-completion claim`

**Principle.** A model signature, a declared-case file, a generated edge manifest, and a
reference-runnability preflight answer four different questions. None is evidence that the
others were exercised. Before turning a green result into a coverage claim, bind it to the
exact artifact SHA, case selector, and arguments actually forwarded.

**Concrete anchor.** The Crossformer attention model declares ten arguments, including
`num_heads`, `mask`, `scale`, `pos_bias`, and `feature_shape`
(`input/level4_cv/problem_sets/l4_36_80_20260807/problems/45_CrossformerAttention/model.py:20-29`).
Its bundled `get_input_groups()` reads every JSONL record but forwards only
`[x, group_size, q, k, v]`
(`input/level4_cv/problem_sets/l4_36_80_20260807/problems/45_CrossformerAttention/model.py:127-158`).
Therefore values declared for the other five arguments do not reach that helper's model call;
the defaults select `num_heads=1` and `feature_shape=None`, including the single-group branch
(`input/level4_cv/problem_sets/l4_36_80_20260807/problems/45_CrossformerAttention/model.py:91-96`).

The 48th JSONL record (zero-based case 47) declares `x=[8,20449,512]`, fp16,
`feature_shape=[143,143]`, and `num_heads=8`
(`input/level4_cv/problem_sets/l4_36_80_20260807/problems/45_CrossformerAttention/45_CrossformerAttention.json:48`).
If consumed through the five-argument helper, the dropped `feature_shape` and `num_heads` make
the score tensor contain `8 * 1 * 20449 * 20449` fp16 elements: 6,690,585,616 bytes. This is a
static consequence of the cited bytes, not a retained runtime measurement.

**Evidence boundaries.** Exact artifacts from the audited workspace were:

- in-repo model SHA-256
  `acf73bbe32757a981554dc50b9db68fbdbb8b2dbef86d8ebeee85d49ca0fd0b0` and 50-line
  JSONL SHA-256 `5a8c4911cf39e34211ffd14335f2745b56936636cd41a4ecda04cf84b8947709`;
- generated manifest SHA-256
  `3654fcc65f19197656136e3ebe895f6900d9f791bd48ef7f27547418320b4b4e`
  (内部路径（不公开）workspace/45_CrossformerAttention/manifest.json:1), which reports 60 cases = 50
  declared + 10 augmentations, with 22 multi-group, 23 mask-present, 16 position-bias, and
  13 explicit-scale cases. Those counts describe this generated manifest, not the bundled
  five-argument helper;
- reference preflight SHA-256
  `f56adc018e4b806ad0f4b3873a008c98a98e9b62ecdb7b4afb963d653a9d3194`
  (内部路径（不公开）workspace/45_CrossformerAttention/ref_runnable.json:1), whose retained output is only
  `torch.Size([1, 196, 96])`, matching the first declared JSONL record. It proves one reference
  call returned; it does not certify all 50 declared cases, all 60 generated cases, or every
  optional-argument arm.

**Rule.** Report each layer independently: declared-case count and SHA; generated-case count,
augmentation count, and structural coverage; the exact adapter/helper and forwarded argument
set; and the preflight case identity plus output. A missing field is `UNMEASURED`, never
inherited from a neighboring artifact with a broader name.

**Other instances (predicted).** Any benchmark whose model has optional attributes, mode-selecting
arguments, broadcast/rank-polymorphic inputs, or a separate edge-dataset generator and one-case
reference preflight.

**Cross-ref:** OL-321 (a reassuring result must cover the claim) and OL-323 (record the exact
denominator artifact SHA).
