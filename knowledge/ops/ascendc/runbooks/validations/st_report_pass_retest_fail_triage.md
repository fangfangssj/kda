---
type: "Runbook"
title: ST报告通过复测失败的四步分诊
description: ST 验收复测与集成报告矛盾（报告称通过、实测失败且指标逐次波动）时的四步分诊：单用例复跑统计通过率→固定种子核对→其他用例指标与报告逐字对照裁决报告真伪→失败用例特征相关定位专属代码路径；按此定位全屏蔽行路径并修复后十连跑确定性通过。
tags: [task.debugging, topic.precision, topic.attention]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# ST报告通过复测失败的四步分诊

## 现象

在 Ascend950PR（arch35，CANN 9.2.0）上对自定义稀疏 top-K 注意力算子（BenchNoHarnessSparseFlashAttention，GQA + topK 稀疏索引，BSND/BNSD 双布局，fp16/bf16）做迭代二测试验收：开发会话刚出完集成报告（声称 ST 通过率 100%，含 L0 基线 24 条，其中方形因果用例记录为 `MERE=1.75e-04` 通过），测试工程师重装算子包后复测，L0 回归 24 条中出现 1 条失败：

```
[ST] [L0-024] FAIL  MERE=2.09e-02 MARE=1.51e+04 (thr=9.77e-04) smallVal=0/33742  | S1=S2 causal square fp16 BSND causal
[INFO] Per-case results: PASS=23 FAIL=1 TOTAL=24
[ERROR] Failed cases: L0-024
```

同一用例在集成报告里是通过（1.75e-04）、在验收复测里是失败（MARE 高达 1.51e+04 且逐次波动），两者直接矛盾。当时代码状态为集成验证通过、已提交的版本；待裁决的问题有三个互相竞争的解释：集成报告造假/未真实执行、环境或安装包不一致、kernel 存在非确定性（heisenbug）缺陷。后续分诊确认是第三种：该用例（S1=S2 因果方阵，存在全屏蔽行）的非确定性精度缺陷，根因为全屏蔽行路径的标量/向量流水缺少同步屏障，由后续修复会话另行闭环；本卡只沉淀"报告通过、实测失败"矛盾出现时的分诊方法本身。

## 适用场景

- 工程形态：算子仓开发的 ST 黑盒验收/回归复测（C++ aclnn 评测，逐用例独立进程跑 NPU，测试二进制支持 `--case-id` 单用例模式）。
- API 路径：`tests/st/run.sh` + `--csv`/`--case-id`；测试用例 CSV（含 shape/dtype/causal 等特征列）；集成/验收报告 markdown。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——分诊流程知识，不宣称任何 kernel 行为结论。
- 路线优先级：recommended_fastpath（遇到矛盾先分诊再动代码，避免对着错误假设修 kernel）。
- 明确不适用 / 未验证场景：
  - 确定性失败（每次复跑同值失败）——步骤 1 即短路，直接进入常规失败定位，不需要全套分诊。
  - 报告部分造假的极端情形——本方法只对照了 4 条代表性用例指标（见失效边界），未做全量逐条核对。
  - 与 AscendC ST 评测无关的通用测试报告矛盾（方法可借鉴但未在本仓验证）。

## 判别方法

四步分诊，每步都有可执行探针与原始输出；顺序有意安排：先确认"非确定性"再裁决"报告真伪"，最后才做代码路径定位。

**第 1 步：单用例复跑统计（判定确定性）。** 对失败用例做 ≥7 次单跑，统计通过率并观察指标是否双态聚集：

```bash
CSV=<abs>/tests/st/testcases/L0_test_cases.csv; cd <abs>/tests/st/build
echo "=== L0-024 flake rate (7 runs) ==="; P=0; F=0
for i in $(seq 1 7); do
    R=$(./test_aclnn_bench_no_harness_sparse_flash_attention "$CSV" --case-id L0-024 2>&1 | grep -oE "L0-024.*(PASS|FAIL)" | head -1)
    echo "run $i: $R"; echo "$R" | grep -q PASS && P=$((P+1)) || F=$((F+1))
done; echo "L0-024: PASS=$P FAIL=$F"
```

实测输出（7 连跑 `PASS=3 FAIL=4`）：

```
run 1: L0-024] PASS
run 2: L0-024] FAIL
...
L0-024: PASS=3 FAIL=4
```

通过态恒 `MERE=1.75e-04 MARE=4.96e-04`，失败态 `MERE=1.95e-02~3.89e+04`——两态各自聚集、无中间值，判定为**非确定性缺陷**（同输入不同输出）。

**第 2 步：输入确定性核对（排除数据随机）。** grep 测试源码的种子/随机源：

```bash
cd <abs>/tests/st && grep -n -E "seed|srand|rand|generator|manual_seed|torch::rand|at::rand|Philox| randint" test_aclnn_bench_no_harness_sparse_flash_attention.cpp | head -50
```

实测命中：`query/key 固定 seed = 42 + B*1000 + S1*100 + S2`（由用例参数推导）、`sparseIndices 固定 seed=0、无重复、值域 [0,S2)`——输入完全确定，排除"随机输入导致的双态"。

**第 3 步：报告真伪裁决（区分 heisenbug 与报告造假）。** 复跑若干**未失败**用例，把实测 MERE/MARE 与集成报告声称值**逐字对照**：

```bash
echo "=== My L1 run: L1-025/026/122/151 MERE vs integration report ===" && grep -E "L1-025|L1-026|L1-122|L1-151" /tmp/iter2_l1_run.log | grep -E "PASS|FAIL"
echo "=== Integration report claimed values ==="
echo "L1-025: MERE=1.76e-04 MARE=4.92e-04"; echo "L1-026: MERE=1.42e-03 MARE=3.89e-03"
echo "L1-122: MERE=1.75e-04 MARE=4.84e-04"; echo "L1-151: MERE=1.76e-04 MARE=4.94e-04"
```

实测四条的指标与报告声称值**逐字一致**——报告是真实执行的结果，不是造假；失败用例在报告生成时恰好落在通过态。矛盾解释收敛为"间歇性缺陷当时未被复现"。

**第 4 步：失败用例特征相关（定位专属代码路径）。** 用用例 CSV 的特征列做相关性筛选，找"只有失败用例具备"的特征：

```bash
cd <abs>/tests/st/testcases
echo "=== L0 中 S1==S2 的用例 ==="; awk -F, 'NR>1 && $3==$4{print $1": S1="$3" S2="$4" causal="$11}' L0_test_cases.csv
echo "=== L1_iter2 中 S1==S2 且 causal=true 的用例（潜在 flaky）==="; awk -F, 'NR>1 && $3==$4 && $11=="true"' L1_iter2_test_cases.csv; echo "(空则无)"
```

实测：L0 表里唯一 `S1==S2 且 causal` 的用例就是该失败用例（S1=S2=512，因果方阵，存在全屏蔽行）；L1_iter2 无此类用例。再用对照组复跑排除"causal 本身"：

```bash
for CID in L0-005 L0-007 L0-010 L0-013; do echo "=== $CID (3 runs) ==="
    for i in 1 2 3; do R=$(./test_aclnn_bench_no_harness_sparse_flash_attention "$CSV" --case-id $CID 2>&1 | grep -oE "$CID\].*(PASS|FAIL)" | head -1); echo "  run $i: $R"; done; done
```

实测对照组（causal 但 S1<S2、无全屏蔽行）各 3/3 稳定通过。**解释**：失败特征 = "S1=S2 且 causal"（全屏蔽行）→ 定位到 kernel 的全屏蔽行处理路径（online softmax 的全屏蔽 chunk 检测与跳过分支）——后续在该路径补标量/向量流水屏障后，同用例十连跑 10/10 确定性通过，证实分诊指向正确。

## 验证信号

以下信号来自该算子迭代二验收会话（Ascend950PR，CANN 9.2.0）：测试工程师在验收复测中发现报告与实测矛盾，按上述四步分诊后由修复会话在同一用例上闭环验证。

- 验收复测失败用例 FAIL `MERE=2.09e-02 MARE=1.51e+04`，与集成报告声称的通过（1.75e-04）矛盾（来源平台：950；证据等级：tool_observed）。
- 第 1 步探针：7 连跑 PASS=3 FAIL=4，通过态恒 1.75e-04/4.96e-04、失败态 1.95e-02~3.89e+04 双态聚集 → 非确定性（来源平台：950；证据等级：tool_observed）。
- 第 2 步探针：种子 grep 确认 query/key seed=42+B*1000+S1*100+S2、sparseIndices seed=0 → 输入确定（来源平台：950；证据等级：tool_observed）。
- 第 3 步探针：4 条未失败用例实测 MERE/MARE 与报告声称值逐字一致 → 报告真实（来源平台：950；证据等级：tool_observed）。
- 第 4 步探针：L0 唯一 S1==S2 causal 用例即失败用例；对照组 4 条 causal（S1<S2）用例各 3/3 稳定 → 全屏蔽行路径（来源平台：950；证据等级：tool_observed）。
- 分诊结论的闭环验证：指向的全屏蔽行路径补标量/向量流水屏障后，同用例十连跑 10/10 通过且指标逐次一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 方法的前提是"同输入可重复运行"：第 1/2 步依赖单用例模式与确定性种子；若测试框架本身引入随机（如时间做种子），双态可能来自数据而非 kernel，分诊会在第 2 步改道。
- 第 3 步只对照了 4 条代表性用例指标即裁决"报告真实"，未做全量逐条对照；若报告只部分造假（挑用例贴数），本方法漏检。
- 第 4 步的特征相关只是定位**候选路径**，不构成根因证明；根因确认须回到代码级证据（本例由屏障修复 + 十连跑闭环）。
- 分诊流程在 C++ aclnn ST 评测形态下验证；PyTorch 适配路径的同类矛盾未跑过全套四步（其逐用例子进程隔离结构与本方法前提一致，未验证，推断可迁移）。

**未决项**：

- 报告真伪裁决的自动化对照脚本未沉淀（本次为临时 grep + 人工比对；可固化为"复测 N 条 + 与报告 diff"的验收步骤）。
- "部分造假报告"的检测能力缺失：全量逐条对照的成本与收益未评估（未尝试）。
## 坏实践

- 用 `grep -c "FAIL"` 统计失败数——会把每条用例 Summary 行里的 `FAIL=0` 汇总字样也计入（本会话实测该统计给出假阳性计数），必须用 `grep -c "] FAIL"` 或对照 Per-case results 行。
- 见到"报告通过、复测失败"就直接怀疑报告造假并要求重跑集成——本例报告完全真实，矛盾源是间歇性缺陷；先做第 3 步逐字对照再下结论。
- 对非确定性用例只复跑一两次就下"已复现/无法复现"结论——本例 7 连跑才给出 3/4 的通过率，1 次通过毫无信息量。

## 误诊

- 分诊前出现过的三个竞争假设均被探针排除：输入随机（第 2 步固定种子排除）、集成报告造假（第 3 步指标逐字一致排除）、安装包与源码不一致（核对 run 包与 vendors 目录时间戳排除）——最终保留的"kernel 非确定性缺陷"由第 1 步双态聚集与第 4 步特征相关共同支撑。
- 验收报告最初的修复建议"把归约 API 的掩码参数改为 true"基于参数误读（该末参实为是否输出索引），与分诊无关但同属该轮误诊记录。
