---
type: "Runbook"
title: "黑盒 bf16-only 精度失败：dtype 分层判别"
description: "TTK aclnn 黑盒全量跑批出现部分 FAIL 时，先按输入 dtype 分层统计通过率：fp16/fp32 100% 而 bf16 85% 即锁定 bf16 特有精度问题；再按失败用例的 shape/range 聚簇，小动态范围也失败可证伪单纯 exp 溢出假设。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["950"]
---

# 黑盒 bf16-only 精度失败：dtype 分层判别

## 现象

softmax 基准算子（Ascend 950，CANN 9.1.0，kernel 内部 fp32 累加）完成三分支开发后做 TTK aclnn 模式黑盒精度验收：golden 插件修正后全量复跑 520 条（L0 17 + L1 500 + L2 3，串行 `--pc 1 --proc-no-reuse --no-memory-check`），跑批中段开始出现零星 FAIL，终局：

```
total=520 pass=494 fail=26 skip=0
```

26 条失败里 3 条是 L2 异常用例（PARAM_COUNT_MISMATCH，预期失败，另卡），其余 23 条全部 `precision=FAIL / precision_status=FAIL`（GOLD 正常——golden 生成成功，是**比较失败**而非 golden 崩）。逐条提取失败用例的 dtype/shape/属性/输入范围后，全部 23 条都是 bfloat16 输入，且聚成两簇：

| shape | dim | input_data_range | 失败数 |
|-------|-----|------------------|--------|
| (2,6,8,8,5,6,5) | 2 | (-100, 100) | 18（含 L0_017 一条，其余 L1） |
| (31,11,6,9) | -1 | (-0.01, 0.01) | 5 |

而同一跑批里 float16 192/192、float32 172/172 全过。

## 适用场景

- 工程形态：TTK aclnn 模式黑盒精度验收（vendor 部署算子）的结果判读与失败分流。
- API 路径：`ttk aclnn` 的 `result.csv`（`testcase_name/precision/precision_status` 列）+ 用例 CSV（`tensor_dtypes/tensor_view_shapes/attributes/input_data_ranges` 列）。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR_9579），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：判别方法 operator_family 级可复用（任何多 dtype 黑盒跑批的失败分流）；bf16 失败结论本身 same_operator（该算子该实现实测）。
- 路线优先级：recommended_fastpath（跑批结果判读的第一步）。
- 明确不适用 / 未验证场景：单一 dtype 用例集（无分层可言）；TTK kernel 模式结果判读（字段不同）；fp16/fp32 的失败模式未在本轨迹出现。

## 判别方法

1. **触发**：全量跑批部分用例 FAIL，失败零散分布在多个用例名之间（非整段整块失败——整段失败先怀疑 harness/竞态，参见 warmup 竞态类知识）。
2. **探针**（可执行，verbatim 取自本会话）：按输入 dtype 分层统计通过率，并把失败用例明细从用例 CSV 拉出来对齐：

```python
import csv, ast

PKG = "<package_root>"
OUTDIR = PKG + "/tests/blackbox/ttk"
TC = PKG + "/tests/blackbox/testcases"

results = {}
with open(OUTDIR + "/result.csv") as f:
    for r in csv.DictReader(f):
        results[r['testcase_name']] = r

all_cases = {}
for layer in ['l0','l1','l2']:
    with open(f"{TC}/aclnn<Op>_{layer}_test_cases.csv") as f:
        for r in csv.DictReader(f):
            all_cases[r['testcase_name']] = r

def get_input_dtype(case):
    dtypes = ast.literal_eval(case.get('tensor_dtypes','()'))
    return dtypes[0] if dtypes else 'unknown'

print("=== Precise pass rate by input dtype (excluding L2 exception) ===")
dtype_stats = {}  # dtype -> [pass, total]
for name, case in all_cases.items():
    if 'exception' in name:
        continue
    dt = get_input_dtype(case)
    if dt not in dtype_stats:
        dtype_stats[dt] = [0, 0]
    dtype_stats[dt][1] += 1
    if results.get(name,{}).get('precision_status','').strip().upper()=='PASS':
        dtype_stats[dt][0] += 1

for dt in sorted(dtype_stats):
    p, t = dtype_stats[dt]
    print(f"  {dt}: {p}/{t} pass ({100*p/t:.1f}%)")
```

   注意 `tensor_dtypes` 形如 `('bfloat16','bfloat16',)`（输入+输出），取 `dtypes[0]` 才是输入 dtype；用 `awk -F','` 切列会因引号内逗号错位，必须用 csv 模块 + `ast.literal_eval`。
3. **观测**（本会话实测值）：`bfloat16: 130/153 pass (85.0%) / float16: 192/192 pass (100.0%) / float32: 172/172 pass (100.0%)`；23 条 bf16 失败聚成 (2,6,8,8,5,6,5) dim=2 (-100,100) ×18 与 (31,11,6,9) dim=-1 (-0.01,0.01) ×5 两簇；层分布 L0 16/17、L1 478/500、L2 0/3。
4. **解释**：fp16/fp32 全过排除 golden/TTK 侧系统性问题（golden 独立于 kernel，同一 golden 服务全部 dtype）；失败集中在单一 dtype + 特定 shape/range 组合 → kernel 侧 bf16 路径的精度特性问题，不是环境或 harness。两簇里**小动态范围 (-0.01,0.01) 也失败**，证伪"单纯大范围 exp 溢出"假设（softmax 稳定实现先减行最大值，理论上 ±100 输入不应溢出；±0.01 输入更无溢出可能）——指向更底层的 bf16 数值口径差异（kernel fp32 累加后回写 bf16 vs golden 在原生 bf16 下 max→sub→exp→sum→div 的逐级舍入），机制未追到底。

## 设计决策

- 无设计取舍（诊断分析卡）；分析顺序选择"dtype 分层 → shape/range 聚簇 → 机制假设排除"，从最便宜的统计开始，未上单用例逐元素复现（归修复阶段）。

## 系统信息

- 硬件/软件：Ascend 950（Ascend950PR_9579），CANN 9.1.0，TTK=ops-test-kit。
- 相关文件：`tests/blackbox/ttk/result.csv`（520 行结果）、`tests/blackbox/testcases/aclnn<Op>l[0-2]_test_cases.csv`（用例定义）、`tests/blackbox/ttk/blackbox_results_summary.md`（失败分类汇总）、`docs/<Op>/develop/golden.py`（CPU golden，原生 bf16 口径）。

## 预防

- 黑盒跑批结果先做 dtype 分层统计再下结论，不要把 26 条失败笼统当"精度不行"；L2 异常用例（PARAM_COUNT_MISMATCH）与真实精度 FAIL 必须分开计数，否则污染通过率。
- bf16 算子的用例设计阶段就应包含小动态范围用例——本会话正是小范围簇证伪了溢出假设。

## 验证信号

以下信号来自该算子黑盒验收会话（golden 插件修正后全量跑批，Ascend950PR_9579 上板）：

- 终局统计：total=520 pass=494 fail=26 skip=0（来源平台：950；证据等级：tool_observed）。
- dtype 分层（排除 L2 异常）：bfloat16 130/153（85.0%）、float16 192/192（100.0%）、float32 172/172（100.0%）（来源平台：950；证据等级：tool_observed）。
- 失败明细：23 条 bf16 全部 precision=FAIL（GOLD 正常），双簇分布 18×(2,6,8,8,5,6,5)/dim=2/(-100,100) + 5×(31,11,6,9)/dim=-1/(-0.01,0.01)（来源平台：950；证据等级：tool_observed）。
- 层分布：L0 16/17（1 条 bf16 失败 L0_017）、L1 478/500（22 条 bf16 失败）、L2 0/3（异常用例预期失败）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 23 条 bf16 失败的机制层根因未追到底：kernel fp32 累加 vs golden 原生 bf16 逐级舍入的口径差异是候选假设（golden 源码 docstring 声明"输入按实际 dtype 原样送入、不额外提升到 fp32；kernel 内部 accumulator_dtype=float32 是其实现细节"），未做逐元素对比验证。
- 黑盒跑批用 TTK 默认比较口径（未加 `--compare close`）；23 条 bf16 FAIL 在宽松口径下是否仍失败未对照（未验证）。
- bf16 通过率 85% 是该算子该实现的 single_run 实测值，不构成跨算子预期。
- 其他算子/平台未验证（未验证，判别方法推断可迁移）。

**未决项**：

- **优先项：复现并追根因到机制层**——单条失败用例（建议小范围簇 (31,11,6,9) dim=-1）逐元素对比 kernel 输出与 CPU golden，定位差异来自 bf16 精度位宽、fp32 累加后的舍回、还是比较容差口径。
- 对照 `--compare close`（np.isclose）口径重跑 23 条失败用例，区分"真实精度差异"与"口径过严"。
- 修复与回归归 kernel 实现阶段（本会话为执行与汇报任务，未修复）。
## 坏实践

- 把 26 条失败混在一个失败率里汇报（3 条 L2 异常 + 23 条 bf16 性质完全不同）。
- 只看失败用例的 shape 不看 input_data_ranges，把两簇混为一谈（大范围簇和小范围簇的机制含义不同）。

## 误诊

- 首判"大范围 (-100,100) 涉及 bf16 exp 溢出"（对首条失败用例的第一反应）：随后 (31,11,6,9) (-0.01,0.01) 小范围簇同样失败，该假设被证伪——±0.01 输入的 exp 无溢出可能（agent_reported 级假设修正，如实记录；失败分布本身为 tool_observed）。
