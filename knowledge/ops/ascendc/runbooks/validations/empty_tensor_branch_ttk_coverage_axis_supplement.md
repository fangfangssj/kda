---
type: "Runbook"
title: 空 tensor 分支 TTK 结构覆盖轴补齐配方
description: 为空 tensor 分支生成 TTK 白盒用例时，空 shape 的 elems 恒为 0 → size 档恒 S，结构上无法覆盖 coverage-audit 固定期望的 S/M/L 与 prime_nonaligned；用 scenario_dimensions 收窄结构用例 + 非空 shape_boundaries（[2048,2048] M、[8192,8192] L、[0,97] prime_nonaligned）补齐结构轴。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T04:06:11Z'
updated_at: '2026-09-17T04:06:11Z'
platforms: ["950"]
---

# 空 tensor 分支 TTK 结构覆盖轴补齐配方

## 现象

在一个 softmax 基准算子（opbuild 自定义算子包，Ascend 950，CANN 9.1.0）空 tensor 分支（EmptyTensor，进入条件 = 输入任一维 size==0）的白盒用例生成任务中，用白盒设计工具（`whitebox_designer.py` 的 `design-to-ttk` 流水线）从设计事实（design_facts.json）+ 算子接口描述（opspec.json）生成 TTK 用例集，再用同工具的三项校验把关：`validate-ttk`（CSV 格式）、`coverage-audit`（结构覆盖轴）、`validate-shapes`（shape 合法性）。

遇到的结构性矛盾：coverage-audit 的期望模型（`build_model(dims)`）**固定**要求结构轴全覆盖——size 档 `S/M/L`、align 档 `aligned/nonaligned/prime_nonaligned`、dtype 全集（本算子 fp16/fp32/bf16），期望签名总数 2607。但空 tensor 分支的主体用例（设计文档 §9 测试点：`[0,4]`、`[0]`、`[0,0,0]`、`[4,0,8]`、`[0,2097152]`）的元素总数 `elems = prod(shape) = 0`，按档位函数恒映射为 size="S"——**结构上不可能产生 M/L 档的空 tensor**（任一维为 0 则乘积必为 0）。若只提交空 shape 用例，coverage-audit 的 size 轴必然缺口。

解法（本轨迹验证）：design_facts 的 `scenario_dimensions` 把"结构性用例"收窄到分支能产生的最小集（避免生成器自行铺开非空结构矩阵），再在 `shape_boundaries` 里显式追加**非空补充用例**覆盖 M/L/prime_nonaligned 轴——这些补充用例运行时会路由到其他分支（非本分支测试点），仅用于满足结构覆盖审计。生成 19 条用例后三项校验全过（coverage-audit exit 0，size 轴 actual [L,M,S] = expected [L,M,S]），下游 TTK kernel 模式实测 19/19 PASS。

## 适用场景

- 工程形态：opbuild 算子包白盒用例设计（`tests/whitebox/<branch>/`，`design-to-ttk` 生成 → 三项校验 → 下游 `ttk kernel` 执行）；为**退化分支**（空 tensor / 恒定小规模 / 单 dtype 等"结构轴先天受限"的分支）生成用例时。
- API 路径：`whitebox_designer.py` 的 `design-to-ttk` / `validate-ttk` / `coverage-audit` / `validate-shapes` 子命令；design_facts.json 的 `scenario_dimensions`（dtype/rank/size/align/valclass/attrs 收窄）与 `semantic_extract.shape_boundaries`（显式 shape 补充）；opspec.json 的 `numerical_tolerance.per_dtype`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（SOC Ascend950PR_9579）+ CANN 9.1.0 + TTK 白盒执行（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`workflow_only`（白盒用例设计工作流；不宣称 kernel 行为）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证场景：非退化分支（自身能覆盖结构轴）无需此配方；coverage-audit 期望模型若改为支持轴裁剪（读 design_facts 限制 expected 集合），本配方的作用降为历史做法（工具版本演进未验证）。

## 根因/解决配方

**机制（单句）**：coverage-audit 的期望集来自固定结构模型（SIZES=["S","M","L"]、ALIGNS=["aligned","nonaligned","prime_nonaligned"] 等常量笛卡尔积），不随分支的实际可达 shape 空间裁剪，而空 tensor 的 elems=prod=0 恒落 S 档、末维 0 或 32 的倍数只产生 aligned/nonaligned——期望与可达的差只能靠显式补充用例填平。

配方由 design_facts.json 关键字段（`scenario_dimensions` 收窄 + `shape_boundaries` 显式补充 + 配套开关）与生成/校验命令组成，完整内容见下节。

## 完整代码

**1. `scenario_dimensions` 收窄结构性用例生成为最小集**（生成器只产 1 条非空结构用例，避免铺开不可达组合）：

```json
  "scenario_dimensions": {
    "dtype": ["float16"],
    "rank": [2],
    "size": ["S"],
    "align": ["aligned"],
    "valclass": ["sym-small"],
    "attrs": {"dim": [-1]}
  },
```

**2. `shape_boundaries` 承载分支主体测试点 + 结构轴补充**（完整清单）：

```json
    "shape_boundaries": [
      {"tag": "B3-empty-A-axis-r2",      "shape": [0, 4],      "attrs": {"dim": -1}, "dtypes": ["float16", "float32", "bfloat16"], "value_range": [-1, 1]},
      {"tag": "empty-rank1-unique-axis", "shape": [0],         "attrs": {"dim": -1}, "dtypes": ["float16", "float32", "bfloat16"], "value_range": [-1, 1]},
      {"tag": "empty-all-zero-r3",       "shape": [0, 0, 0],   "attrs": {"dim": -2}, "dtypes": ["float16", "float32", "bfloat16"], "value_range": [-1, 1]},
      {"tag": "empty-R-axis-r3-dim1",    "shape": [4, 0, 8],   "attrs": {"dim": 1},  "dtypes": ["float16", "float32", "bfloat16"], "value_range": [-1, 1]},
      {"tag": "empty-large-vocab-r2",    "shape": [0, 2097152],"attrs": {"dim": -1}, "dtypes": ["float16"],                      "value_range": [-1, 1]},
      {"tag": "empty-prime-last-r2",     "shape": [0, 97],     "attrs": {"dim": -1}, "dtypes": ["float16", "float32", "bfloat16"], "value_range": [-1, 1]},
      {"tag": "M-supplement-structural-r2", "shape": [2048, 2048], "attrs": {"dim": -1}, "dtypes": ["float16"], "value_range": [-1, 1]},
      {"tag": "L-supplement-structural-r2", "shape": [8192, 8192], "attrs": {"dim": -1}, "dtypes": ["float16"], "value_range": [-1, 1]}
    ],
```

档位判定依据（档位函数行为，本轨迹核对源码得出）：`_size(elems)`：0 ≤ elems < 4194304 → "S"；`[2048,2048]` = 4.19M → "M"；`[8192,8192]` = 67.1M → "L"。`_align(last)`：`last % 32 == 0` → "aligned"，非 32 倍数且非素数 → "nonaligned"，末维为素数（97）→ "prime_nonaligned"。空 shape 的末维为 0（0%32==0 → aligned）或 4/8（nonaligned），故 align 轴的 prime_nonaligned 也需 `[0,97]` 显式补。

**3. 配套开关**（防生成器注入不属本分支的用例形态）：

```json
  "op_dir": "operators/BenchHarnessSoftmax_package/tests/whitebox/branch-2",
  "golden_path": null,
  "allow_scalar": false,
  "allow_empty": false,
  "allow_min_shape": false,
  "max_shape_elems": 4096,
  "mode_struct_k": 1
```

（`allow_empty=false`：空 shape 全部由 shape_boundaries 显式提供，不靠生成器随机注入；`golden_path=null`：避免 helper 对 `[0,2097152]` 类大末维空 tensor 做探测，空输出 shape 由 opspec 的 `shape_rule: y.shape = x.shape` 派生。）

**4. 生成与校验命令**（verbatim）：

```bash
PY=skills/ascendc-whitebox-design/references/whitebox_designer.py
OUT=operators/BenchHarnessSoftmax_package/tests/whitebox/branch-2
python3 $PY design-to-ttk $OUT/_design_facts.json $OUT/_opspec.json \
    $OUT/design_doc_cases.json $OUT/ttk_design_doc_cases.csv

python3 $PY validate-ttk    $OUT/ttk_design_doc_cases.csv
python3 $PY coverage-audit  $OUT/design_doc_cases.json $OUT/_design_facts.json $OUT/_opspec.json
python3 $PY validate-shapes $OUT/design_doc_cases.json $OUT/_design_facts.json $OUT/_opspec.json
```

`_design_facts.json`/`_opspec.json` 为临时文件，用后删除；产物结构与决策依据写入 `WORKFLOW_PROVENANCE.json`（验收方据其重建临时文件复跑审计）。

## 设计决策

- **为何补充用例选 `[2048,2048]`/`[8192,8192]`/`[0,97]`**：M/L 阈值（4194304）两侧的最小可读形状，末维 2048/8192 均 32 对齐（不再引入新的 align 缺口）；prime_nonaligned 用空 shape 本体（`[0,97]`）补——空 tensor + 素数末维，既是本分支合法输入又补齐 align 轴，一石二鸟。
- **为何 M/L 补充用例接受"路由到其他分支"**：coverage-audit 审的是**用例集的结构轴覆盖**，不审单用例的路由归属；M/L 非空 shape 必然路由到 FullloadR/ChunkedR（R=2048/8192 ≤ rFullloadThresh≈10K），它们在下游执行中同时充当跨分支回归，无额外代价。这一取舍在 provenance 的 gaps 里显式记录（"结构性限制，非设计缺陷"）。
- **为何收窄 scenario_dimensions 而不是放开让生成器铺**：放开会在非空结构矩阵上生成大量不属于空分支的用例（dtype×rank×size×align 全组合），偏离任务边界；收窄后结构用例 1 条（(63,64) fp16 S-aligned），主体由 shape_boundaries 精确控制。
- **当前方案路线档位**：`recommended_fastpath`。

## 系统信息

- 工具链：白盒设计 `skills/ascendc-whitebox-design/references/whitebox_designer.py`（`design-to-ttk`/`validate-ttk`/`coverage-audit`/`validate-shapes`）；TTK 执行（ops-test-kit 源码仓，`ttk kernel --compare close --pc 1`，SOC Ascend950PR_9579）；CANN 9.1.0。
- 相关文件：`operators/BenchHarnessSoftmax_package/tests/whitebox/branch-2/design_doc_cases.json`、`ttk_design_doc_cases.csv`、`WORKFLOW_PROVENANCE.json`、`whitebox_constraints.yaml`；临时 `_design_facts.json`/`_opspec.json`（用后删）；下游 `tests/whitebox/ttk/whitebox_all.csv`、`result.csv`。

## 预防

- 为退化分支做白盒设计前，先推一遍"分支可达 shape 空间在 size/align/dtype 轴上的像"——凡恒定值（本例 elems=0→S）就是必缺口，提前在 shape_boundaries 备好补充项，避免生成-审计-返工一轮。
- design_facts 的 gaps 字段显式记录"哪些轴靠非空补充、为什么结构性不可达"——验收方重建审计时不至于把补充用例误判为越界生成。
- 临时 `_design_facts.json`/`_opspec.json` 删除前，把完整内容摘要进 WORKFLOW_PROVENANCE.json（本轨迹验收环节即靠它重建临时文件复跑了两项审计）。

## 验证信号

问题与验证来源：一次空 tensor 分支白盒用例生成任务，按上述配方生成后跑三项校验并交付下游执行。以下均为本会话实测：

- 生成：`design-to-ttk` 产出 19 条用例（1 条结构用例 + 16 条空 tensor + 2 条 M/L 补充），§9 测试点 13/13 全部 HIT（来源平台：950；证据等级：tool_observed）。
- `validate-ttk`：PASS（26 列表头正确、19 行、用例名唯一、tolerance per-dtype 齐全），EXIT=0（来源平台：950；证据等级：tool_observed）。
- `coverage-audit`：EXIT=0，`num_cases=19 / num_expected_sigs=2607 / num_actual_sigs=18`，size 轴 expected [L,M,S] = actual [L,M,S]，dtype 3/3，align expected [aligned,nonaligned,prime_nonaligned] = actual 全覆盖（来源平台：950；证据等级：tool_observed）。
- `validate-shapes`：EXIT=0，0 违规（来源平台：950；证据等级：tool_observed）。
- 下游执行：TTK kernel 模式（`--compare close --pc 1`）`total=19 pass=19 fail=0 skip=0`（GATE PASS），空 tensor 用例与 M/L 非空补充用例全 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 档位阈值（S/M 以 4194304 为界等）来自所用版本 `wb_scenario_sig.py` 的常量；工具版本更新后阈值或轴定义变化，补充 shape 需按新档位重算（未验证，推断需重算）。
- 补充用例走其他分支意味着其正确性依赖那些分支的实现质量——若被补分支本身有缺陷，补充用例失败会被误读为本分支问题；排查时先看用例 shape 与路由。
- 本配方针对"审计期望固定全轴"的工具行为；若工具后续支持期望集裁剪，收窄 scenario_dimensions 即可，无需非空补充。
- 平台边界：本配方在 Ascend 950（SOC Ascend950PR_9579）+ CANN 9.1.0 实测；其他平台未验证（生成与审计为主机侧工具行为，推断可迁移）。

**未决项**：

- `build_model` 固定全轴期望是否会在工具后续版本改为可裁剪，未跟踪；若可裁剪，本卡配方应随之简化。
## 坏实践

- 为过 coverage-audit 把空分支的 shape_boundaries 塞进实际不可达/无意义的组合（如给空 tensor 强行配超大非零维试图凑 M/L——elems 仍为 0，凑不出）。
- 把 M/L 补充用例藏起来不记录——验收方重建审计时对不上 expected 轴，反而触发返工；正确做法是 provenance gaps 里写明补充理由。

## 误诊

- 无重大误诊往返；唯一自纠正是对 `num_expected_sigs=2607 vs num_actual_sigs=18` 的解读——审计按"轴维度覆盖"判过而非"签名全等"，18 个实际签名覆盖全部轴值即 exit 0，初读时曾担心签名缺口会 FAIL，核对 axes 输出后确认判据是逐轴 expected⊆actual。
