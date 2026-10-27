---
type: "Runbook"
title: "scatter 逐元素 RMW 实测偏离 MTE2 目标 4–5 量级"
description: "单核逐元素标量 GM read-modify-write 的 scatter 类 kernel，实测 AiCore 耗时随 update 数线性增长、每元素数百 ns，比按向量化 MTE2-bound roofline 估的理论目标慢约 65×–2.1×10⁴×；性能归因时用每元素时延判别实际瓶颈，不要拿向量化 roofline 当达标线。"
tags: [task.optimization, topic.performance.analysis, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# scatter 逐元素 RMW 实测偏离 MTE2 目标 4–5 量级

## 现象

在 Ascend950PR（CANN 9.1.0）上为一个自定义 scatter/scatter-reduce 算子采集上板性能：该算子的性能目标表按"向量化、MTE2-bound（GM 读侧带宽瓶颈）"的 roofline 模型估了 6 条关键 shape 的理论时延。用 TTK aclnn 模式 `--task-prof` 实测设备侧 AiCore 任务时长（`--run 3`、计时调用 Min≈Max 稳态）后对照发现：

- 5 条可测 shape 的达成率（理论时延 / 实测时延）全部远低于 1，最低约 5×10⁻⁵、最高约 1.5×10⁻²——实测比理论目标慢约 **65× 到 2.1×10⁴×**；
- 实测耗时随 update 元素总数（`prod(indices.shape)`）**线性增长**，折算每元素时延稳定在 **约 149–355 ns** 区间，与 update 数、dtype、reduce 模式基本无关；
- 精度全部 PASS（输入合法、数据分布正常），设备频率正常（AiCore 1650MHz），排除"跑错了/降频了"。

当时代码状态：kernel 的 `Process()` Phase 2 为**单核（仅 core 0）逐元素 GM read-modify-write**——每个 update 元素执行一次"MTE2 读 y[targetOff] → S-pipe 标量 reduce → MTE3 写 y[targetOff]"并夹显式事件同步；这是为保证重复索引聚合确定性而做的正确性驱动设计（kernel 注释自述"single-core eliminates the multi-core RMW race"）。

## 适用场景

- 工程形态：scatter/scatter-add/gather-scatter 类随机写算子的性能评测与归因；性能目标以向量化 roofline（MTE2/带宽 bound）为参照的对照报告。
- API 路径：TTK aclnn `--task-prof` 的 `<Op>_AiCore` 任务时长口径；kernel 侧 S-pipe 标量 `GetValue/SetValue` + 逐元素 `DataCopyPad` RMW 路径。
- 验证过的平台与 CANN 版本：Ascend950PR（950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——"逐元素标量 GM RMW 每元素数百 ns、与向量化带宽 roofline 差 2–3 个量级"是平台级机制特征；具体倍率区间依赖该算子实现。
- 路线优先级：recommended_fastpath（归因判别规则本身；是否改造 kernel 是后续决策）。
- 明确不适用 / 未验证场景：向量化/批量搬运实现的 scatter kernel（每元素时延应低 2–3 个量级，本卡签名不适用）；msprof 逐 pipe 直测（本轨迹不可得，见失效边界）；其他芯片的每元素时延量级（未验证，推断量级特征可迁移）。同一 RMW 路径的精度侧风险（2 字节元素大规模丢写）见[2 字节逐元素 GM RMW 大规模丢写判别卡](../precision/two_byte_gm_rmw_large_scale_lost_write.md)；逐元素随机读场景的替代路线（改 SIMT 直访 GM）见[随机 Gather 每元素 DMA 过慢卡](random_gather_per_element_dma_should_switch_to_simt_direct_gm.md)。

## 根因

kernel 实际计算链是**单核逐元素标量 GM read-modify-write**，实测耗时由标量 RMW 串行时延主导，而非 GM 读带宽：

- 机制链：每元素一次 GM 读（MTE2）→ S-pipe 标量 reduce → 一次 GM 写（MTE3），且每步之间夹显式事件同步（MTE2_S / S_MTE3 / MTE3_MTE2），单核串行。总耗时 ≈ update 数 × 单次标量 GM RMW 时延。
- 量级推导：非缓存标量 GM 访存时延为数百 ns 量级；实测每元素 149–355 ns 与之吻合。而向量化 MTE2-bound roofline 按带宽（TB/s 级）折算每元素应 <1 ns——两者差 2–3 个数量级，与实测/理论 65×–2.1×10⁴× 的总差距一致（总差距还含 data 全量拷贝等非 RMW 项，故区间更宽）。
- 目标侧根因：性能目标表用"向量化 MTE2-bound 实现"的 roofline 上限估理论时延，未对"单核逐元素 RMW"这一实际计算链建模——目标值系统性偏乐观 4–5 个量级。设计文档自身已注明"随机访存建模精度低于规则计算，目标值需结合实测校准"，本实测即该校准结论。

## 判别方法

对"scatter 类算子实测远低于 roofline 目标"的归因判别（无需 msprof）：

1. 取实测 AiCore 平均时延（TTK aclnn `--task-prof` 结果 CSV 的 `op_perf_us`，或 run.log profiling 表 `<Op>_AiCore` Avg；注意 `api_perf_us` 是 host 侧 aclnn API 时长，不要混用）。
2. 折算每元素时延：`每元素 ns = op_perf_us × 1000 / prod(indices.shape)`。
3. 判别：
   - **数百 ns 量级且随 update 数线性增长 ⇒ 标量逐元素 RMW bound**（本卡签名）；
   - <1 ns 量级、随字节数增长 ⇒ 接近带宽 bound（MTE2/MTE3）；
   - 与 update 数无关、与 data 体量相关 ⇒ data 全量拷贝主导。
4. 交叉验证线性性（同代码路径内对比）：取两条同 reduce/dtype 的 shape，总时延比 ≈ update 数比、每元素时延几乎相等，则"时延 = update 数 × 单元素时延"成立。

本例 5 条 shape 的每元素折算值全部落在 149–355 ns；线性性用同路径对照验证：两条同为 fp16 add 的 shape update 数 4:1，实测总时延比约 3.99:1，每元素时延 355 ns vs 356 ns 几乎相等——"时延 = update 数 × 单元素 RMW 时延"在同路径内严格成立，据此把"未达标"归因为设计路径而非噪声/环境。

## 设计决策

- kernel 为何选单核逐元素 RMW：scatter 的重复索引聚合（add/multiply/amax/amin）在多核并发写同一 `y[target]` 时存在竞态；单核串行 RMW 用确定性换性能。这是正确性驱动的取舍，不是实现失误——性能归因时应把它当"已知代价"而非 bug。
- 归因方法为何用每元素时延而不是总时延对比：总时延混入 data 全量拷贝（与 data 体量相关）等非 RMW 项；每元素折算能剥离出 RMW 串行项的机制特征。
- 为何不用 `--sim-report` 佐证：仿真报告不是 msprof 实测，用它支撑"实测 bound"会误导；本卡把 bound 归因明确标注为"设计 + 时延特征推断"。
- 当前方案路线档位：recommended_fastpath（判别规则零成本；改造 kernel 属后续优化决策，不在本卡范围）。

## 性能影响

- 相对代价（成对表述）：相对性能目标表的向量化 MTE2-bound 理论路线，单核逐元素 RMW 实测慢约 **65×–2.1×10⁴×**（参照路线：§4.4 理论 MTE2 roofline 目标；来源平台：950）；差距下界出现在 update 数极小、data 拷贝占比高的 shape，上界出现在 update 数百万级的大 shape。
- fp16/bf16 的 add/multiply 走 fp32 workspace 三阶段路径（data 上采 fp32 → 逐元素 RMW → 回写原生 dtype），比原生路径更多一轮全量拷贝与两次类型转换，同 update 数下每元素时延落在区间上段（静态代码阅读推断，未单独实测拆分）。

## 预防

- 为随机写类算子定性能目标时，先确认实现路径：逐元素标量 RMW 设计的目标应按"update 数 × 数百 ns"建模，不要直接套向量化带宽 roofline——两者差 2–3 个数量级。
- 性能对照报告里区分三类耗时口径：host aclnn API 时长（`api_perf_us`）、设备 AiCore 任务时长（`op_perf_us`）、理论 roofline；混用口径会得出错误达成率。
- 评测"未达标"先做每元素折算与线性性检查，再下"实现低效"结论。

## 验证信号

以下信号来自该 scatter 算子上板性能采集会话：aclnn 模式 `--task-prof true --run 3` 实测 5 条可测 shape 的设备侧 AiCore 任务时长并折算。

- 5 条可测 shape 实测/理论达成率分别约为 5.17×10⁻⁵、1.54×10⁻²、4.75×10⁻⁵、4.94×10⁻⁵、5.16×10⁻⁵（参照：设计文档 §4.4 理论 MTE2-bound 目标；来源平台：950；证据等级：tool_observed）。
- 每元素时延折算（实测 AiCore 时长 ÷ update 元素数）：5 条 shape 分别约 355、216、250、149、356 ns，全部落在数百 ns 量级（来源平台：950；证据等级：tool_observed）。
- profiling 表显示计时调用 Min≈Max（如某 shape Min/Max=371372.04/373548.13），测量稳态非噪声（来源平台：950；证据等级：tool_observed）。
- 同批 5 条 shape 精度全部 PASS（binary_equal / stat_rel_err），排除输入非法导致的假慢（来源平台：950；证据等级：tool_observed）。
- 线性性对照：两条同 fp16 add 路径的 shape update 数 4:1，实测 AiCore 总时延比约 3.99:1，每元素时延 355 ns vs 356 ns 几乎相等（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 每元素数百 ns 的签名基于 950 + CANN 9.1.0 单卡实测；其他芯片/版本未验证（机制上非缓存标量 GM RMW 的量级特征推断可迁移，具体数值未验证）。
- 本卡归因（标量 RMW bound）是"kernel 设计 + 每元素时延特征"的推断，无 msprof 逐 pipe 占比直测佐证——msprof 直测在 aclnn `--task-prof` 路径下不产出（见遗留问题）。
- 若 kernel 混合了向量化与标量路径（如小 update 数走批量、大数走逐元素），每元素折算会失真；需先按 shape 分组再折算。
- 达成率区间（65×–2.1×10⁴×）依赖该算子的理论目标口径（模板化 roofline）；换目标口径需重算，倍率本身不是通用预期。

**未决项**：

- msprof 逐 pipe 占比（`aiv_mte2_ratio` / `aiv_vec_ratio` 等）未取得：aclnn `--task-prof` 不产出 PROF_* 目录、kernel 模式因 VEC_ERROR 不可用——获得逐 pipe 直测的可行采集路径待查（优先项：补 msprof 直测以把"标量 RMW bound"从推断升级为实测）。
- fp32 workspace 三阶段路径中各阶段（上采/RMW/回写）的耗时拆分未实测。
- 多核 + 原子/分桶去重等替代设计的性能-正确性权衡未在本轨迹验证。
## 坏实践

- 把向量化 roofline 目标直接当 scatter 类随机写算子的达标线，实测差 4 个量级就反复查设备/频率/数据分布——应先折算每元素时延识别串行 RMW 设计。
- 用 `api_perf_us`（host 侧 API 时长）当 kernel 耗时报达成率——它与设备侧 AiCore 时长是两个口径，异步 launch 下前者可远小于后者。
- 用仿真报告（`--sim-report`）的 pipe 占比冒充 msprof 实测 bound。

## 误诊

- 无（本卡归因一次到位；同会话曾考虑"设备降频/数据分布异常/采集噪声"均已用频率读数、精度 PASS、Min≈Max 稳态排除）。
