---
type: "Runbook"
title: 理论MTE3 bound落空：半精度链Scalar pipe被低估
description: ascendc-performance-evaluation 预测 MTE3 写侧 bound，上板 msprof 实测 7/7 落空——fp32 实测 Vector 主导、半精度 Cast 链 Scalar 主导（占比最高 53%），Scalar 是理论遗漏的 bound 候选。
tags: [task.debugging, topic.performance.analysis, topic.programming_model.execution_unit.scalar, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 理论MTE3 bound落空：半精度链Scalar pipe被低估


## 现象

对一个 AscendC 算子（门梯度逐元素链 + bias 沿 batch 归约的纯 Vector 形态，对标 torch 的 GRU Cell 反向算子；成品 kernel 黑盒已全绿）做上板性能采集与目标对照。设计期理论评估（`ascendc-performance-evaluation` 引擎，flow 模版机制）对该算子 7 条关键训练 shape 的 bound 归属给出预测：6 条 MTE3（写侧耗时 > 读侧，如某条「写侧 0.60μs > 读侧 0.51μs」）、1 条 Vector（单核小 shape）。随后用 TTK kernel 模式开 msprof profiling（`--task-prof true --run 5`）上板采集，逐条解析 msprof `op_summary_*.csv` 的 pipe 时间占比字段（`aiv_vec_ratio` / `aiv_mte2_ratio` / `aiv_mte3_ratio` / `aiv_scalar_ratio`，占比 = 各 pipe time / aiv_time），取占比最高者为实测 bound。结果与理论预测 **7/7 全部不一致**：实测 4 条 **Vector 主导**（全部是 fp32 多核 case，vec_ratio 0.323–0.435）、3 条 **Scalar 主导**（全部是半精度 case：fp16×2 + bf16×1，scalar_ratio 0.308–0.531，其中单核 fp16 最高 0.531）；理论预测的 MTE3 在全部 7 条上占比仅 0.116–0.215，**从未成为最高 pipe**。若按理论预测去优化写侧搬运（MTE3），优化方向将完全落空。

## 适用场景

- **工程形态**：ACLNN/AscendC ops-registry 算子仓成品 kernel 的上板性能归因（TTK kernel 模式 + msprof task 级 profiling 采集）；性能优化方向决策前的瓶颈确认。
- **API 路径**：msprof `op_summary_*.csv` 的 `aiv_*_time(us)` / `aiv_*_ratio` 列；TTK kernel 模式 `--task-prof`；理论评估侧为 `ascendc-performance-evaluation` 的 bound 归属输出。
- **验证过的平台与 CANN 版本**：Ascend950PR（frontmatter `platforms: ["950"]`）+ CANN 9.0.0-beta.2；TTK kernel 模式采集。
- **适用粒度（transfer_scope）**：`operator_family`——「含 dtype 升降精度 Cast 链的逐元素 + 归约形态 AscendC kernel」这一族通用。
- **路线优先级**：`banned`——把理论评估的 bound 归属预测直接当实测瓶颈指导优化方向（不作上板 pipe 占比核验）属于被本轮实测证伪的路线。
- **明确不适用 / 未验证场景**：纯搬运型（memcpy 类）或无 Cast 的 kernel 未验证（其 MTE 侧理论预测可能仍然有效）；950DT 等其他核数平台未验证；Cube/Matmul 形态（aico* 列）未涉及。

## 判别方法

对每条性能用例，从 msprof `op_summary_*.csv` 中过滤 kernel 行（`Task Type` ∈ AIV 类：`MIX_AIV` / `AI_VECTOR_CORE` / `KERNEL_MIX_AIV` / `KERNEL_AIVEC`），读代表 run（去 warmup 首跑后最接近 TTK 报告中位数的 run）的四个 pipe 占比，**占比最高者即实测 bound**：

- fp32 路径：先看 `aiv_vec_ratio`（本轮实测 0.323–0.435，四条全为 Vector 主导）；
- fp16/bf16 半精度路径：先看 `aiv_scalar_ratio`（本轮实测 0.308–0.531，三条全为 Scalar 主导，单核形态占比最高）；
- `aiv_mte3_ratio`（写侧）本轮 0.116–0.215、`aiv_mte2_ratio`（读侧）0.165–0.272，均未出现过最高。

最小 probe（Python，参数化自采集会话（判读逻辑逐字）的解析逻辑）：

```python
import csv, glob

f = sorted(glob.glob(f'/tmp/ops-test-kit/msprof/op/{tcname}/binary/PROF_*/mindstudio_profiler_output/op_summary_*.csv'))[-1]
with open(f) as fh:
    reader = csv.DictReader(fh)
    rows = [r for r in reader if r.get('Task Type','') in ('MIX_AIV','AI_VECTOR_CORE','KERNEL_MIX_AIV','KERNEL_AIVEC')]
runs = rows[1:]; rep = min(runs, key=lambda r: abs(float(r['Task Duration(us)']) - ttk_perf))   # 代表 run（首跑为 warmup，取稳态）
pipes = {'Vector': float(rep['aiv_vec_ratio']), 'MTE2': float(rep['aiv_mte2_ratio']),
         'MTE3': float(rep['aiv_mte3_ratio']), 'Scalar': float(rep['aiv_scalar_ratio'])}
bound = max(pipes, key=pipes.get)   # 占比最高者为实测 bound
```

## 根因

理论评估引擎的 bound 归属候选集只覆盖读写搬运（MTE2/MTE3）与 Vector 计算体，未将 Scalar pipe 列为 bound 候选，导致半精度 Cast 链落在 Scalar pipe 的开销被系统性遗漏。

机制链（本轮实测证据）：

1. **指令流差异是 dtype 分层的输入**：理论评估的指令流定义文件中，fp32 模版只有 `Sub×5 + Mul×11 + ReduceSum×6(+尾段)`、**0 条 Cast**；fp16/bf16 模版因为「输入升 fp32（H2F）→ fp32 乘加 → 输出收窄（F2H）→ bias 输出收窄」的数值口径，额外含三类 Cast 链（该算子为 6+7+6 条）。
2. **实测占比按 dtype 干净分层**：scalar_ratio——fp32 四条 0.211–0.237，半精度三条 0.308–0.531；两者区间完全不重叠。vec_ratio 反向：fp32 多核 0.323–0.435 > 半精度 0.258–0.307。
3. **半精度 → Scalar 主导、fp32 → Vector 主导**与指令流中「有无 Cast 链」一一对应：多出的 Cast 类开销把 Scalar pipe 抬到占比第一（单核半精度 case 达 0.531，即 Scalar pipe 时间占 aiv_time 过半）。
4. **理论侧从未输出 Scalar**：7 条预测里 MTE3×6 + Vector×1，与实测 7/7 不一致——不是测量误差（非 warmup run 间占比波动仅 ±2% 左右），而是候选集缺项。

（注：「Cast 指令开销落在 Scalar pipe 执行」是对 dtype 分层相关性的机制解释，属推断层归因；指令级 profiling 直接验证见遗留问题。）

## 设计决策

- **bound 判读口径**：取四个 AIV pipe（vec/mte2/mte3/scalar）中 `aiv_*_ratio` 最高者为实测 bound——四 pipe 全纳入候选，而不是只看理论预测的 MTE3/Vector 两项；本轮若只按理论候选集读数，会得出「都不是」的僵局。
- **代表 run 选取**：去 warmup 首跑后取最接近 TTK 报告中位数的 run——占比在非 warmup run 间稳定（±2%），单 run 取值即可代表；无需平均多 run 占比。
- **本卡路线定位**：`banned` 路线（跳过上板核验直接采信理论 bound）的负知识；正确路线见判别方法（先核验占比再定优化方向）。

## 系统信息

- 设备：Ascend950PR（128GB HBM），npu-smi 25.7.rc1.6；CANN 9.0.0-beta.2（innerversion V100R001C10SPC001B214，x86_64-linux）。
- 采集框架：TTK kernel 模式（`--task-prof true --run 5 -b release --pc=1` + golden 插件），msprof 产物位于 `msprof/op/<testcase_name>/binary/PROF_*/mindstudio_profiler_output/op_summary_*.csv`。
- 相关文件：性能用例 CSV（`tests/perf/perf_key_shapes.csv`）、结果 CSV（`result_perf.csv` 的 `bin_perf_us`/`bin_tiling_key`/`bin_block_dim` 列）、理论侧 bound 预测表（设计文档 Validation.md「性能目标」）与指令流定义（roofline 用例 jsonl）。

## 完整代码

判别方法一节的 probe 即本卡核心可复用代码（参数化自采集会话（判读逻辑逐字）的 op_summary 解析逻辑，源算子用例清单已参数化为 `tcname`）。完整工具链配方（含产物定位、Task Type 过滤全集、代表 run 选取与逐 case 批量判读脚本）已内联于上文判别方法的 op_summary 解析逻辑。

## 性能影响

本卡不引入优化手段，无成对路线对比。给优化决策的相对参照：半精度 case 的 Scalar pipe 时间占 aiv_time 比例 0.308–0.531（来源平台：Ascend950PR；证据等级：tool_observed）——即在这些 case 上，砍掉一半 Cast 开销理论上限也只能收回约 aiv_time 的一至五成占比的时间，且以 Scalar 确为 Cast 开销为前提（未做指令级验证）。

## 预防

- 用理论评估（ascendc-performance-evaluation 类引擎）的 bound 归属预测作优化方向时，**必须先上板核验 pipe 占比**再动手；预测值只当假设。
- 半精度（fp16/bf16）数值口径的算子（升 fp32 计算 + 收窄输出），性能归因时把 **Scalar pipe 列入 bound 候选集**，与 MTE2/MTE3/Vector 并列判读。
- 理论评估结论写入设计文档时标注「未上板核验」，避免后续任务当实测事实引用。

## 验证信号

来源：某 AscendC 算子（门梯度逐元素链 + bias 归约形态）成品 kernel 的上板性能采集会话（TTK kernel 模式 + msprof task 级 profiling，Ascend950PR / CANN 9.0.0-beta.2，`--run 5` 去首跑中位数口径）；以下信号均取自该会话 msprof `op_summary_*.csv` 与结果 CSV 的实测输出。

- 实测 bound 与理论预测 7/7 不一致：实测 Vector×4（fp32 多核）+ Scalar×3（fp16×2 + bf16×1），理论预测 MTE3×6 + Vector×1（来源平台：Ascend950PR；证据等级：tool_observed）。
- MTE3 占比 0.116–0.215，7 条中从未最高（来源平台：Ascend950PR；证据等级：tool_observed）。
- scalar_ratio 按 dtype 分层：fp32 四条 0.217/0.237/0.211/0.235，半精度三条 0.308/0.531/0.374，区间不重叠（来源平台：Ascend950PR；证据等级：tool_observed）。
- 指令流定义：fp32 模版 0 条 Cast，半精度模版含三类 Cast 链（来源平台：Ascend950PR；证据等级：tool_observed）。
- 非 warmup run 间 pipe 占比稳定（约 ±2%），排除测量噪声解释（来源平台：Ascend950PR；证据等级：tool_observed）。

## 失效边界

- 「Scalar 是被低估 bound 候选」的结论边界：适用于含 Cast 链的半精度逐元素/归约形态；fp32 无 Cast 路径实测为 Vector 主导。纯搬运型 / Cube 主导 / 无 Cast 的 kernel 未验证。
- 950DT 等其他核数平台未验证（推断可迁移但未实测）。
- 单算子、单次上板采集（`--run 5` 重复 5 次，但同一会话同一次部署）；跨轨迹复现待后续采集。
- 「Cast 开销落在 Scalar pipe」为推断层机制解释，未做指令级 profiling 验证；若实际是地址计算/循环控制主导 Scalar，则「砍 Cast」的优化收益推算不成立，但「Scalar 占比按 dtype 分层、理论候选集缺 Scalar」的实测结论不受影响。

## 坏实践

- 按理论 MTE3 预测直接优化写侧（如调写合并、MTE3 双缓冲、写侧对齐），不先上板看 `aiv_*_ratio`——本轮 7 条的 MTE3 占比全部垫底级别，优化方向完全落空。
- 理论评估的 bound 输出未标注「未上板核验」就写入设计文档性能目标表，被后续任务（如优化任务）当实测瓶颈引用。

## 误诊

- 采集会话中发现 msprof 根目录残留**前次采集的旧产物**（testcase 命名与本轮不同、时间戳更早）。若不过滤直接读，会把旧 run 的占比当本轮实测。本轮按文件时间（`find -newer`）过滤规避，未造成误判——采集时注意先排除残留目录。
- 把理论 pipe body 估算值（如「写侧 0.60μs」）与实测占比混在一起对照——两者口径不同（前者是估算绝对量、后者是实测占比），须分别与同口径数据比。

## 未决事项

1. **指令级验证 Scalar 高占比来源**（Cast vs 地址计算 vs 循环控制）——优先项：用指令级 profiling 或逐类消除实验定位，决定「减 Cast / Cast 向量化 / 换收窄策略」是否为有效优化方向。
2. group vs base 受控对照（同 shape/dtype 各跑一组）缺失，无法把 SyncAll 栅栏 + Phase 2 workspace 往返开销从 dtype 效应中剥离。
3. 950DT 等其他平台复测，确认 dtype 分层模式的平台无关性。
