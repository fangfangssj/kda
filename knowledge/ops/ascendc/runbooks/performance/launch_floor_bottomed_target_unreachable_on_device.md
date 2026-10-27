---
type: "Runbook"
title: launch floor触底目标不可达：pipe body标定偏低
description: 理论评估 pipe body 全部低于设备 launch floor 使目标一律触底 1.6μs；上板实测 pipe body 全部超 floor 标定区间上沿、达成率 19%–58% 全未达标——触底口径仅在真实 body < floor 时有效。
tags: [task.testing, topic.performance.analysis, topic.architecture]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# launch floor触底目标不可达：pipe body标定偏低


## 现象

对一个 AscendC 算子（门梯度逐元素链 + bias 沿 batch 归约，纯 Vector 形态；成品 kernel 黑盒已全绿）做上板性能采集，与设计文档登记的理论性能目标逐条对照。设计期理论评估（`ascendc-performance-evaluation` 引擎）的结论是：7 条关键训练 shape 的理论 pipe body 估算（0.10–1.50μs）**全部低于设备 launch floor（平台标定 1.6μs，上板标定区间 1.5–1.77μs）**，因此理论目标时延一律触底取 launch floor = 1.6μs；达标阈值按 `target_efficiency=0.60` 折算 = 1.6μs ÷ 0.60 = **2.667μs**（实测 ≤ 2.667μs 即达标）。用 TTK kernel 模式开 msprof 上板采集（`--task-prof true --run 5`，取去首跑后 4 次 Task Duration 中位数）后逐条对照：**7/7 全部未达标**——达成率（理论目标 ÷ 实测）19.2%–57.7%，全部低于 60% 阈值；多核 6 条的实测 Task Duration 为理论目标的约 4.3–5.2×，单核 1 条约 1.7×；msprof 的 `aiv_time(us)`（pipe body 实测代理）全部超过 launch floor 标定区间的上沿（1.77μs）——**理论「pipe body < floor 所以触底」的前提在 7/7 条上都不成立**，最接近达标的单核半精度 case 也差一步（实测比阈值高约 4%）。

## 适用场景

- **工程形态**：ACLNN/AscendC ops-registry 算子仓成品 kernel 的性能目标设定与达标验收（TTK kernel 模式 + msprof 上板采集）；用理论评估引擎（ascendc-performance-evaluation 类）产出的目标值作验收线之前的适用性检查。
- **API 路径**：理论侧为评估引擎的 `theoretical_time_us = max(各 pipe body + fill/drain, launch floor)` 口径；实测侧为 TTK kernel 模式 `bin_perf_us`（msprof Task Duration 去首跑中位数）与 `op_summary` 的 `aiv_time(us)`。
- **验证过的平台与 CANN 版本**：Ascend950PR（frontmatter `platforms: ["950"]`）+ CANN 9.0.0-beta.2；launch floor 标定常数来自该平台（1.6μs，标定区间 1.5–1.77μs）。
- **适用粒度（transfer_scope）**：`operator_family`——「指令密集的逐元素 + 归约形态小 kernel」这一族（多链乘加 + 归约段，GM 流量 MB 级以下）。
- **路线优先级**：`banned`——把 launch-floor 触底的理论目标直接当达标验收线（不作实测 pipe body 校验）是被本轮实测证伪的路线。
- **明确不适用 / 未验证场景**：真实 pipe body 确实低于 floor 的极小 kernel（触底口径有效）；body ≫ floor 的大 kernel（目标由 body 主导、触底不参与）；其他平台（launch floor 常数不同）未验证。

## 判别方法

判断触底口径是否适用，**不要看理论 body 估算值，要看实测**：

1. 上板采集后读 msprof `op_summary_*.csv` 的 `aiv_time(us)`（kernel 行，去 warmup 首跑）——它是 pipe body 的实测代理；
2. 与本平台 launch floor 标定区间（950PR 为 1.5–1.77μs）比较：实测 `aiv_time` 全部高于区间上沿 ⇒ 「body < floor」前提失效，触底目标不可作为验收线；
3. 快速判据：实测 Task Duration ÷ 理论目标时延 > 1.5× 且 `aiv_time` 已超 floor 上沿，基本可判定目标失真，须重定目标或先做 body 校准。

## 根因

理论评估引擎的 pipe body 标定系统性偏低（估算口径未计入实际发射效率损失与 Scalar pipe 开销），使「body < launch floor」的触底判断在指令密集小 kernel 上失效。

机制链（本轮实测证据）：

1. **触底机制**：理论时延 = max(各 pipe body + fill/drain, launch floor)。7 条的理论 body 估算 0.10–1.50μs 全部 < floor 1.6μs ⇒ 目标一律 = 1.6μs，阈值 = 1.6 ÷ 0.60 = 2.667μs。
2. **实测推翻前提**：实测 `aiv_time`（pipe body 代理）为平台 launch floor 标定常数（1.6μs）的 **1.5×–5.1×**，7/7 超过 floor 标定区间上沿（1.77μs）；实测 Task Duration ÷ 理论目标 = 1.7×（单核）–5.2×（多核最大）。
3. **达成率算术**：达成率 = 目标(1.6μs) ÷ 实测 Task Duration × 100% ∈ [19.2%, 57.7%]，7/7 < 60%；单核 case 57.7% 最接近（实测比阈值 2.667μs 高约 4%），多核 6 条 19%–23%。
4. **body 低估的来源**（推断层归因，实测差距为客观）：估算口径按峰值频率折算指令时间，未计入归约段指令的实际发射效率低于纯乘加、以及 Scalar pipe（Cast/地址/循环开销）的时间——该算子 fp32 路径实测 Vector 占比最高（0.323–0.435）而理论预测写侧 MTE3，半精度路径实测 Scalar 最高（详见同批[理论 MTE3 bound 落空卡](theory_bound_missing_scalar_pipe_candidate.md)）；叠加 group 分支 SyncAll 栅栏 + Phase 2 workspace 往返（理论评估已登记未建模）的固定开销。

## 设计决策

- **验收口径决策**：本轮采集任务按「只采集与归因、不以达标为门禁」执行——7/7 未达标如实记录，未尝试改 kernel 凑数，也未放宽阈值；达标判定留给后续优化任务以修正后的目标执行。
- **对照口径决策**：达成率用「理论目标 ÷ 实测中位数（去首跑）」，噪声带按 ±10%（floor 标定区间波动幅度）考虑；避免用单跑值对照。
- **本卡路线定位**：`banned`（触底目标直接当验收线）的负知识；替代路线见判别方法（先实测 body 再定/校准目标）。

## 系统信息

- 设备：Ascend950PR（128GB HBM），npu-smi 25.7.rc1.6；CANN 9.0.0-beta.2（innerversion V100R001C10SPC001B214）；平台配置 Ascend950PR_9589（vector_core_cnt=64、ub_size=253952）。
- 采集框架：TTK kernel 模式（`--task-prof true --run 5 -b release --pc=1` + golden 插件）；msprof 产物 `msprof/op/<testcase>/binary/PROF_*/mindstudio_profiler_output/op_summary_*.csv`。
- 相关文件：性能用例 CSV、结果 CSV（`result_perf.csv`：`bin_perf_us`/`bin_tiling_key`/`bin_block_dim`）、理论目标与标定常数（设计文档 Validation.md「性能目标」：floor 1.6μs、阈值 2.667μs、达成率算法）、性能报告（PERF_REPORT.md）。

## 完整代码

本卡无代码修复（采集归因任务，生产代码零改动，已 git 核验）。判别方法的可执行 probe：TTK kernel 模式采集命令 + `op_summary` 的 `aiv_time(us)` 读取，参数化自采集会话（判读逻辑逐字）：

```bash
source <env.sh> && cd $TTK && python3 -m ttk kernel \
  -i <perf_csv> -d false -b release --pc=1 --task-prof true --run 5 \
  --plugin <golden.py> -o <outdir>/result_perf.csv
```

```python
# 判别触底前提是否成立：读代表 run 的 aiv_time（pipe body 实测代理）
import csv, glob
f = sorted(glob.glob(f'/tmp/ops-test-kit/msprof/op/{tcname}/binary/PROF_*/mindstudio_profiler_output/op_summary_*.csv'))[-1]
rows = [r for r in csv.DictReader(open(f)) if r.get('Task Type','') in ('MIX_AIV','AI_VECTOR_CORE','KERNEL_MIX_AIV','KERNEL_AIVEC')]
steady = rows[1:]  # 去 warmup 首跑
aiv_time_us = float(steady[len(steady)//2]['aiv_time(us)'])   # 稳态行取中位
## 与本平台 launch floor 标定区间比较（950PR：1.5–1.77μs）：超上沿 ⇒ 触底口径失效
```

## 性能影响

本卡不引入优化手段。相对参照（来源平台：Ascend950PR；证据等级：tool_observed）：实测 Task Duration / 理论目标时延 = 1.7×（单核 case）–5.2×（多核 case），即按触底目标折算的「预期余量」实际不存在——对指令密集小 kernel，用触底目标排期或判收会系统性低估 4–5 倍量级的真实耗时（多核形态）。

## 预防

- 理论评估输出「目标触底 launch floor」时，**先抽查 1–2 条上板实测 `aiv_time`** 是否真的低于 floor 标定区间，再决定目标是否可用作验收线。
- 指令密集（多链乘加 + 归约段）的小 kernel，对理论 body 估算保守看待：发射效率损失与 Scalar 开销不在估算内。
- 设计文档登记性能目标时，把「触底口径」与「body 支配口径」分列，触底目标标注「未实测校验」。

## 验证信号

来源：某 AscendC 算子（门梯度逐元素链 + bias 归约形态）成品 kernel 的上板性能采集与目标对照会话（TTK kernel 模式 + msprof，Ascend950PR / CANN 9.0.0-beta.2，`--run 5` 去首跑中位数口径）；目标值与标定常数取自该算子设计文档 Validation.md「性能目标」。

- 7/7 条达成率 19.2%–57.7%，全部低于 60% 阈值（2.667μs = 1.6μs ÷ 0.60）（来源平台：Ascend950PR；证据等级：tool_observed——实测中位数与理论目标均为工具观测值，比率为算术）。
- 实测 `aiv_time(us)`（pipe body 代理）为平台 launch floor 标定常数（1.6μs）的 1.5×–5.1×，7/7 超过 floor 标定区间上沿（1.77μs）——触底前提失效（来源平台：Ascend950PR；证据等级：tool_observed——实测值为 msprof 工具输出、floor 为设计文档登记的平台标定常数，比为算术）。
- 实测 Task Duration / 理论目标时延 = 1.7×（单核）–5.2×（多核最大）（来源平台：Ascend950PR；证据等级：tool_observed，同上为实测与文档值的算术比）。
- 理论 body 估算区间 0.10–1.50μs（全部 < floor 1.6μs 故目标触底）取自设计文档登记（来源平台：Ascend950PR；证据等级：tool_observed）。
- 单核半精度 case 达成率 57.7% 为最接近达标的一条（实测比阈值高约 4%），多核 6 条 19%–23%（来源平台：Ascend950PR；证据等级：tool_observed）。

## 失效边界

- 触底口径本身**并非永远错误**：真实 body < floor 的极小 kernel（近纯搬运、指令极少的短路 kernel）目标触底是合理的——本卡结论只覆盖「指令密集逐元素 + 归约小 kernel」形态。
- launch floor 标定常数（1.6μs / 区间 1.5–1.77μs）是 950PR 平台值；其他平台常数不同，判别方法（比 `aiv_time` 与本平台 floor 区间）可迁移，具体数值不可搬运（未验证，推断可迁移）。
- 单算子、单次上板采集；理论 body 低估的引擎内部机制（峰值频率折算口径）为推断层归因，未做引擎内部核验。
- group 分支 SyncAll 栅栏 + Phase 2 workspace 往返的未建模开销只是叠加因素之一，量值未隔离。

## 坏实践

- 把「理论目标一律触底 1.6μs」当保守下界使用——本轮实测证明触底目标反而**高估了可达性**（真实耗时高 4–5 倍量级），「触底=保守」的直觉在指令密集小 kernel 上是反的。
- 未达标时反向「凑达标」：放宽阈值、换宽松口径或挑单跑最好值对照——本会话明确只记录不修门禁。

## 误诊

- 误把「未达标」直接归因为 kernel 实现质量差：本轮归因指向**目标设定失真**（触底前提失效）而非实现低效——区分「实现慢」与「目标不可达」须先核 `aiv_time` 与 floor 的关系，再谈优化。
- 误用冒烟单跑值（`--run 1`，含 warmup）作达标对照——首跑系统性偏慢（详见同批[上板首跑 warmup 偏慢卡](task_prof_warmup_first_run_median_of_rest.md)），会放大未达标程度。

## 未决事项

1. **理论 body 标定偏差的机制层归因**——优先项：拆分「归约段发射效率损失」「Scalar pipe 开销」「SyncAll 栅栏 + Phase 2 往返」三项各占多少（需受控实验），才能给评估引擎提出可落地的标定修正。
2. group vs base 受控对照（同 shape/dtype）缺失，栅栏/往返开销量未隔离。
3. 修正后的目标值重定（body 实测校准口径）未做——本轮只登记未达标事实。
