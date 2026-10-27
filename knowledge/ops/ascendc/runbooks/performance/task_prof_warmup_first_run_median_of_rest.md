---
type: "Runbook"
title: 上板首跑warmup偏慢：TTK取后N-1次中位数
description: TTK kernel 模式 --task-prof 上板实测中首跑系统性偏慢（相对稳态中位数 +24%～+69%）；bin_perf_us 的口径=丢弃首跑、取其余 N-1 次 Task Duration 的中位数——单跑或含首跑会高估耗时。
tags: [task.testing, topic.performance.analysis]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 上板首跑warmup偏慢：TTK取后N-1次中位数


## 现象

对一个 AscendC 算子（门梯度逐元素链 + bias 归约形态）的成品 kernel 做 TTK kernel 模式上板性能采集（`--task-prof true --run 5 -b release --pc=1` + golden 插件）时，逐 case 解析 msprof `op_summary_*.csv` 发现：每条用例采集后 profiling 目录里有 **5 行 kernel 记录**（对应 `--run 5` 的 5 次执行），其中**第一次 run 的 Task Duration 系统性显著偏慢**——7 条用例的首跑相对各自稳态中位数的偏移为 **+24.0% ～ +69.4%**（多核 group case 约 +24%～+58%，单核 case 最高 +69.4%）；丢弃首跑后的 4 次之间波动很小（相对稳态中位数约 ±2%）。同时核对 TTK 结果 CSV 的 `bin_perf_us` 列发现：它**不是** 5 次的均值或全量中位数，而是**丢弃首次（warmup）后其余 4 次 Task Duration 的中位数**——该口径经两条用例算术核验逐字一致（以稳态中位数归一：一条五次为 1.313/0.996/0.981/1.004/1.013，去首跑后中位数 = 1.000 == `bin_perf_us`；另一条五次为 1.309/0.988/1.006/0.994/1.013，去首跑后中位数 = 1.000 == `bin_perf_us`）。若误用单跑值（如 `--run 1` 冒烟）或自行把首跑计入统计，会把耗时高估约四分之一到近七成。

## 适用场景

- **工程形态**：TTK kernel 模式的上板性能采集与达标对照（`--task-prof true --run N`）；任何需要读取/复算 TTK `bin_perf_us` 或 msprof Task Duration 稳态值的场景。
- **API 路径**：TTK kernel 模式结果 CSV 的 `bin_perf_us` / `bin_tiling_key` / `bin_block_dim` 列；msprof `op_summary_*.csv` 的 `Task Duration(us)`；TTK 框架 `OnlineRtsProfiling._process_msprof_cycles`（口径实现处）。
- **验证过的平台与 CANN 版本**：Ascend950PR（frontmatter `platforms: ["950"]`）+ CANN 9.0.0-beta.2；TTK kernel 模式 binary 路线（`-b release`）。
- **适用粒度（transfer_scope）**：`platform_wide`——采集/统计口径类知识，跨算子适用于该平台上 TTK kernel 模式采集。
- **路线优先级**：`recommended_fastpath`——直接采信 `bin_perf_us`（N≥3），自算稳态时**必须**去首跑取中位数。
- **明确不适用 / 未验证场景**：aclnn 模式的 `op_perf_us` / `api_perf_us` 列口径不同（本 trace 仅读其列名，未采集验证）；其他 NPU 平台未验证；`--run 1` 单跑无稳态可言（只作连通性检查）。

## 判别方法

- **读数**：稳态耗时直接取 TTK 结果 CSV 的 `bin_perf_us`（TTK 已内建去首跑中位数口径），不要用单跑 Task Duration。
- **自算复核**（判口径是否被满足/排查异常值）：从 `msprof/op/<testcase>/<run_mode>/PROF_*/mindstudio_profiler_output/op_summary_*.csv` 取全部 kernel 行的 `Task Duration(us)`，**丢弃第一行（warmup）**，其余取中位数；与 `bin_perf_us` 对照应逐字一致。
- **判别信号**：若「全部 N 次的中位数」与 `bin_perf_us` 明显不一致而「去首跑后 N-1 次的中位数」一致，即证明首跑被剔除；首跑/稳态比 > 1.2 即存在显著 warmup 效应。

## 根因

上板首跑含冷启动开销（推断为 cache 冷/首次 launch 路径，硬件级根因未验证），而 TTK 框架的 Task 级 profiling 口径设计为「丢弃首个 Task Duration、取其余的 numpy.median」，二者叠加使单跑或含首跑的统计口径系统性高估 kernel 耗时。

TTK 口径的机制证据（框架源码 `_process_msprof_cycles`，逐字内联见「完整代码」）：行数 > 1 时 `results = op_prof[1:]`（丢弃首行）→ `total_cycle = numpy.round(numpy.median(results), 3)`。本轮 7 条用例的实测分布与该口径完全吻合（现象一节的归一化核验）。

## 设计决策

- **采集次数选择**：任务要求 N≥3，实际取 `--run 5`——去首跑后剩 4 次，中位数对偶发抖动更稳（4 个值的中位数 = 中间两值均值）；`--run 3` 去首跑后仅 2 次，鲁棒性差一档。
- **冒烟与全量分离**：冒烟显式用 `--run 1` 只验连通性（编译/精度/profiling 链路），其耗时值**不进入**对照；全量才取稳态口径——避免把冒烟值误当性能数。
- **串行采集**：`--pc=1` 串行执行，避免多进程并发干扰 profiling 产物归属。
- **本卡路线定位**：`recommended_fastpath`（`bin_perf_us` + N≥3 直接可用）。

## 系统信息

- 设备：Ascend950PR（128GB HBM），npu-smi 25.7.rc1.6；CANN 9.0.0-beta.2；TTK 框架（kernel 模式，`-b release` binary 路线，`DYN_*`/`CST_*` 列均 OFF、`bin_*` 列为实测）。
- 相关文件：结果 CSV（`bin_perf_us` 列）、msprof 产物（`op_summary_*.csv` 的 `Task Duration(us)` 列）、TTK 框架源码 `ttk/core_modules/npu/op/rts_sequence.py`（`_process_msprof_cycles`，约 L576–625）。

## 完整代码

TTK 口径实现（框架源码逐字摘录，`ttk/core_modules/npu/op/rts_sequence.py`，`_process_msprof_cycles`——bin_perf_us/DYN_PERF 的取数逻辑；`KERNEL_TYPE` 为 kernel 行过滤集）：

```python
    def _process_msprof_cycles(self):
        KERNEL_TYPE = (
            "AI_CORE",
            "AIV_SQE",
            "AI_VECTOR_CORE",
            "MIX_AIC",
            "MIX_AIV",
            "KERNEL_MIX_AIC",
            "KERNEL_MIX_AIV",
            "KERNEL_AIVEC",
            "KERNEL_AICORE",
        )
        total_cycle = "UNKNOWN"
        prof_result_path = pathlib.Path(self._prof_result_path)
        if not prof_result_path.is_dir():
            return total_cycle
        csv_files = list(prof_result_path.glob("**/*.csv"))
        if not csv_files:
            return total_cycle

        op_prof = []
        for item in csv_files:
            if op_prof:
                break
            if item.name.startswith("op_summary_"):
                op_prof = extract_csv_cells(
                    item,
                    ("Task Duration(us)",),
                    col_name_mapping={"Task Duration(us)": "duration"},
                    cmp=lambda row: row["Task Type"] in KERNEL_TYPE,
                )
            elif item.name.startswith("task_time_"):
                op_prof = extract_csv_cells(
                    item,
                    ("task_time(us)",),
                    col_name_mapping={"task_time(us)": "duration"},
                    cmp=lambda row: row["kernel_type"] in KERNEL_TYPE,
                )
            else:
                pass

        if op_prof:
            op_prof = list(x["duration"] for x in op_prof)
            results = [op_prof[0]]
            if len(op_prof) > 1:
                results = op_prof[1:]
            logging.debug(f"MsProf Task Profiling result: {results}")
            total_cycle = numpy.round(numpy.median(results), 3)

        return total_cycle
```

口径复核 probe（采集会话实跑逻辑，源算子用例清单改为从结果 CSV 读取——判读算法逐字保留）：

```python
import csv, glob, statistics
# 从 TTK 结果 CSV 读每条用例的 bin_perf_us（TTK 报告值）
for row in csv.DictReader(open('result_perf.csv')):
    tcname, ttk_perf = row['testcase_name'], float(row['bin_perf_us'])
    f = sorted(glob.glob(f'/tmp/ops-test-kit/msprof/op/{tcname}/binary/PROF_*/mindstudio_profiler_output/op_summary_*.csv'))[-1]
    rows = [r for r in csv.DictReader(open(f)) if r.get('Task Type','') in ('MIX_AIV','AI_VECTOR_CORE','KERNEL_MIX_AIV','KERNEL_AIVEC')]
    durs = [float(r['Task Duration(us)']) for r in rows]
    steady = durs[1:]                                    # 丢弃首跑 warmup
    assert round(statistics.median(steady), 3) == ttk_perf  # 应与 bin_perf_us 逐字一致
    print(f'{tcname}: first_run/steady_median = {durs[0]/statistics.median(steady):.3f}')
```

## 性能影响

本卡为测量口径知识，无优化手段。相对参照（来源平台：Ascend950PR；证据等级：tool_observed）：含首跑的统计相对去首跑稳态中位数高估 +24.0%～+69.4%（7 条用例实测分布；单核 case 偏移最大）。

## 预防

- 上板性能对照一律 `--run N≥3`（建议 5）并采信 `bin_perf_us`；不用 `--run 1` 值作性能数。
- 自算 Task Duration 统计时先画/看逐 run 分布，识别首跑离群再选统计量。
- 报告中注明取值口径（「去首跑后 N-1 次中位数」），避免下游按别的口径复算出「对不上」的疑问。

## 验证信号

来源：某 AscendC 算子成品 kernel 的 TTK kernel 模式上板性能采集会话（Ascend950PR / CANN 9.0.0-beta.2，`--run 5`，7 条用例）；以下信号均取自该会话 msprof `op_summary_*.csv` 逐 run 实测与 TTK 结果 CSV。

- 每用例 5 次 run 的 Task Duration 中首跑系统性最慢：相对稳态中位数偏移 +24.0%/+30.9%/+30.9%/+24.0%/+69.4%/+58.1%/+26.2%（7 条用例逐条，来源平台：Ascend950PR；证据等级：tool_observed）。
- 去首跑后 4 次之间相对稳态中位数波动约 ±2%（来源平台：Ascend950PR；证据等级：tool_observed）。
- `bin_perf_us` == median(去首跑后各次 Task Duration)：两条用例算术核验逐字一致（以稳态中位数归一：五次 1.313/0.996/0.981/1.004/1.013 → 去首跑中位数 1.000；五次 1.309/0.988/1.006/0.994/1.013 → 去首跑中位数 1.000）（来源平台：Ascend950PR；证据等级：tool_observed——原始逐 run 值为工具输出，中位数为算术）。
- TTK 口径实现（丢弃首行 + numpy.median）取自框架源码读取（来源平台：Ascend950PR；证据等级：tool_observed）。

## 失效边界

- 口径结论限定 TTK kernel 模式（task 级 msprof、binary 路线）；aclnn 模式 `op_perf_us`/`api_perf_us` 的统计口径未验证（推断有类似 warmup 效应，但取数逻辑不同）。
- 首跑偏移幅度（+24%～+69%）为本轮 7 条用例的实测分布，非通用常数——不同算子/shape/核数分布会不同；但「首跑显著偏慢、须剔除」的定性结论在该平台本轮全部成立。
- 其他 NPU 平台未验证（推断可迁移但未实测）。
- 首跑偏慢的硬件级根因（冷 cache / 首次 launch 路径）未验证。

## 坏实践

- 用 `--run 1` 的单跑 Task Duration 当性能实测值做达标对照（本会话冒烟即 `--run 1`，其值仅作连通性检查）。
- 自算统计量时把全部 N 次一起取均值/中位数（含 warmup 首跑），再反过来质疑 TTK 报告值「对不上」。
- 不同口径的数字混进同一张对照表（如冒烟单跑值与全量中位数值并列比较）。

## 误诊

- 会话初期曾把 `DYN_PERF: DYN_OFF` 误当异常——`-b release`（binary 路线）下 DYN/CST 列设计为 OFF，`bin_perf_us` 才是实测口径；读结果 CSV 先确认哪条路线的列有值。
- aclnn 模式结果 CSV 有 `op_perf_us`/`api_perf_us` 列而 kernel 模式结果 CSV 没有（列集不同）——两模式的性能列不可互相比对口径。

## 未决事项

1. **首跑偏慢的硬件级根因**（冷 cache / 首次 launch 路径 / 驱动侧初始化）——优先项：定位后可判断是否存在可消除的固定开销（影响极小 kernel 的实测下限）。
2. 首跑偏移幅度与核数/shape 的关系仅有 7 点样本（单核 +69.4% 最高、多核 +24%～+58%），无拟合规律——样本不足，不下定量结论。
3. aclnn 模式 `op_perf_us` 的统计口径（是否同样去 warmup）未验证。
