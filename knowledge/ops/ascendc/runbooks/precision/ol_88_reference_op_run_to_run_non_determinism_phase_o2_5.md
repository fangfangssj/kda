---
type: Runbook
title: "Reference-op run-to-run non-determinism — Phase O2.5 pre-flight + scope-by-construction"
description: "Reference-op run-to-run non-determinism — Phase O2.5 pre-flight + scope-by-construction"
tags: [task.debugging, topic.implementation.pipeline, topic.data_movement, topic.memory, topic.performance.analysis, topic.precision]
status: draft
sources: []
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-88"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Reference-op run-to-run non-determinism — Phase O2.5 pre-flight + scope-by-construction

> 原 OKF v1 卡号：**OL-88**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - any benchmark/opgen op whose reference is torch_npu.npu_X or any CANN fused op

## 冻结验收合同与重复性诊断边界

重复运行同一原 forward、保存输入、参数 state、dtype 与 grader，记录所有输出和差异；如果输出不稳定，结果是一个**带具体病例/环境的诊断局限**。不能把 returned 的病例数当稳定 oracle，也不能据一次漂移宣告所有 torch_npu/CPU、全部 dtype 或整个算子不可解。CPU provider、随机状态和算法选择只是待检机制；未证明根因时不能写成 race/OOB。

冻结输入/输出合同下，不裁输出、不零填参考、不换 oracle、不放宽阈值、不重新生成索引。替代 Python 分解、唯一索引生成或新的 reference 可作为分开的诊断/未应用提案；只有当前验收 authority 明确采用新合同时才能产生新资格链，旧数值账和来源保持。参见[替代 reference 权限](ol_89_when_the_active_reference_is_non_deterministic_loo.md)。

后续观察边界：一次原 CPU grid_sample 固定50返回后，在5病例×3次未改 forward 复跑中，病例37/49的原首次输出→第1次复跑、第1次→第2次复跑均不通过原 grader，第2次→第3次复跑通过；其余3个被测病例0/3/5的这3次相邻比较均通过；另一次原 CPU roi_pooling 六次定向复跑在病例43/47不一致。两者没有新候选、编译或 NPU/P/Q，不能推断通用数值天花板或已证 int16 OOB。保留其原 dtype、输入及所有重复结果。没有测得冷消费的迭代节省。

该5病例重复比较的团队私有保留记录不能匿名读取；已认证 Wiki clone 中以提交 `fc8acb1b2a11d55f4c6837a4fab26071fd68de55`、路径 `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/CPU_REPEAT_ORIGINAL_GRADER_COMPARISON.json`、blob `81a3e082a33f4f7be84bf38ff4d5470b715ac153`、SHA256 `39488cdb8ed5051d0038ae7f232edd7c8997094f057c44780d8df45187a23b5d` 复核（`git show <commit>:<path>`、`git rev-parse <commit>:<path>`、`sha256sum`）。0为原首次50输出，1..3为独立复跑；这是既有证据勘误，没有新运行，也不建立稳定 oracle。上游公开晋升前须替换该私有来源或改写此实测范围。

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: pipeline / preflight / reference-determinism
- **Loaded by**: orchestrator (Phase O2.5 step 4 mandatory), aog-kernel-worker (Phase A KB load when `ref_determinism.json` exists in workspace)
- **Trigger**: any benchmark/opgen op whose reference is `torch_npu.npu_X` or any CANN fused op
- **Mandatory action**: orchestrator MUST run `python3 src/scripts/check_ref_determinism.py --op {slug} --workspace workspace/{op}/ --edge-inputs ... --model-py ...` AFTER edge_runner produces edge_dataset.pt and BEFORE spawning aog-kernel-worker. The script fires the reference 3× on identical inputs and emits `workspace/{op}/ref_determinism.json` listing per-output deterministic-vs-not verdicts.

- **Why this exists**: torch_npu CANN ops on Ascend950PR are not always run-to-run deterministic. If the worker writes a deterministic kernel against a non-deterministic reference, the verifier's bit-exact compare will FAIL no matter how correct the kernel is — and the worker can spend hours debugging "why doesn't my kernel match" before realizing the reference is the moving target.

- **Three confirmed non-det classes** (op#12 KvRmsnormRopeCache, 2026-04-24, 3-run probes):
  1. **Paged-attention block layouts** (`PA_BLK_BNSD`, `PA_BLK_NZ`): scatter races at the block-position level. 3-run max_diff ≈ 10-15.
  2. **`is_output_kv=False` semantic violation**: docstring says "returns None" for k_embed_ret/y_ret but CANN returns 4-tuple with uninitialized buffers — sometimes zero, sometimes garbage from prior NPU memory. 3-run pattern: r0=0, r1=6.78, r2=0 (clearly cold-buffer leak).
  3. **Norm cache_mode with duplicate indices**: `k_cache_out[b,n,idx,:] = k[b,n,s,:]` races when multiple (b,s) target same idx. 3-run max_diff ≈ 8.

- **What worker does with `ref_determinism.json`**:
  1. Phase A: read it before designing the kernel
  2. If `summary_per_output[<output_name>].verdict == "NON_DETERMINISTIC"`: preserve that output and every original comparison. Record the scoped instability; propose any alternative contract separately and apply it only when the current acceptance authority explicitly adopts it.
  3. If all outputs deterministic: proceed normally. Worker brief should still mention `ref_determinism.json` was checked (audit trail).

- **Pattern enforcement**: `case_gen.py` `index_range:<N>` invariant now defaults to **unique-by-default** (uses `randperm` + slice when n ≤ upper). Opt-in to duplicates via `tensor_inputs[i]["allow_dup_indices"] = True` only when the op semantically tests scatter-race behavior. This is a pre-freeze generation policy; it must not change already-frozen saved inputs.

- **Cross-reference**:
  - OL-83 (torch_npu vs pytorch-native drift): static algorithm difference. OL-88 covers DYNAMIC run-to-run non-determinism — orthogonal class.
  - OL-87 (preflight benchmark reference run): companion preflight; OL-87 catches schema/crash issues, OL-88 catches non-det issues.
  - OL-66 (torch::zeros not stream-ordered): different but related — non-determinism from kernel side.

- **Evidence**: op#12 cold-start without OL-88 → worker spent 1 hour iterating on PARTIAL 18/50 + 7/27 before identifying CANN ref non-det as root cause. The historical 30-second pre-flight estimate was an expectation, not a measured controlled saving.

- **2026-04-27 empirical refinement — DO NOT extrapolate class-1 to all `torch_npu` inplace-scatter ops**: op#19 IndexPut Track 1 ref-determinism probe (18 cases of `torch.index_put_(accumulate=True)` at fp16/bf16, M ∈ {31..65536}) found **0/18 non-deterministic**, max_diff=0.0 across 5 runs. This contradicts an earlier extrapolation in `output/npukernelbench/src/kernels/19_IndexPut/verification_HODSA_attempt.json` which framed `index_put_` failures as OL-88 class-1. Lesson: class-1 was confirmed only on PA_BLK_BNSD/PA_BLK_NZ at op#12; do not generalize the verdict to other torch_npu inplace-scatter ops without a fresh `check_ref_determinism.py` probe. Probe report: `workspace/indexput/track1_falsification_report.md`.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-88（category=pipeline / preflight / reference-determinism，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
