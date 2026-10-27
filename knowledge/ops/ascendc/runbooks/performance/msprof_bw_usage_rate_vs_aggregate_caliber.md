---
type: "Runbook"
title: msprof带宽利用率与聚合带宽口径差约10倍
description: "msprof Memory.csv 的 GM_to_UB_bw_usage_rate 与“总搬运量/耗时/芯片峰值”的手动聚合带宽利用率对同一次运行可差约一个数量级（1.69% vs 16.2%）；带宽验收必须写明口径并同时报两个数。"
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T04:39:31Z'
updated_at: '2026-09-17T04:39:31Z'
platforms: ["950"]
---

# msprof带宽利用率与聚合带宽口径差约10倍

## 现象

在 Ascend950PR 上对 BenchNoHarnessStridedSlice 做带宽验收（目标"内存带宽利用率 > 60%"）时，同一个场景、同一次 msprof 采集得到两个相差悬殊的"带宽利用率"：

- msprof `Memory.csv` 汇总的 `GM_to_UB_bw_usage_rate`：场景1（CONTIGUOUS）**1.69%**、场景2 **0.40%**、场景3 **0.41%**；
- 手动聚合口径（总搬运量 ÷ Task Duration ÷ 芯片峰值带宽，峰值取 1.8TB/s 参考值）：场景1 **16.2%**、场景2 **4.1%**、场景3 **4.4%**。

两者相差 ~10x。如果验收只引用其中一个数，结论表述会完全不同（"利用率 1.69%"与"利用率 16%"），且都无法与 60% 目标直接对齐。

## 适用场景

- 工程形态：msprof op 单算子性能采集 + 带宽达标判定（算子性能验收、优化前后对比）。
- API 路径：msprof `Memory.csv` 字段 `GM_to_UB_bw_usage_rate(%)` / `aiv_gm_to_ub_bw(GB/s)` / `GM_to_UB_datas(KB)` / `UB_to_GM_datas(KB)` / 主存读写带宽；`OpBasicInfo.csv` 的 Task Duration。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——msprof 指标解读方法（与算子无关）。
- 路线优先级：safety_fallback——本卡是报告口径规则而非算子行为修复：主判口径（msprof `bw_usage_rate`）保持不变，手动聚合口径作为并列参照的安全回退，防止单一口径把全芯片利用率系统性低估约一个数量级。
- 明确不适用 / 未验证场景：Cube/L2/L0C 等其他层级带宽字段的口径未逐一验证；`GM_to_UB_bw_usage_rate` 的官方定义未从文档核实（见误诊/遗留），当前解释为推断。

## 判别方法

触发条件：用 msprof 带宽利用率做达标判定时，先算一遍手动聚合口径，两数差距在一个数量级左右（本例 1.69% vs 16.2%、0.40% vs 4.1%、0.41% vs 4.4%）即命中本口径差异，不要直接拿单核口径报"全芯片利用率"。（若两口径同低且 STRIDED 场景比 CONTIGUOUS 差一个数量级，则另查搬运量放大根因，见 [逐元素 DataCopyPad 搬运放大 25~64 倍判别](per_element_datacopypad_amplification_diagnosis.md)。）

可执行探针（聚合口径计算，与 msprof 汇总并列输出）：

```bash
python3 -c "
# 输入：OpBasicInfo.csv 的 Task Duration(us) + Memory.csv 主存读/写 KB 合计折算的 total_mb + 参考峰值(TB/s)
duration_us, total_mb, peak_tb = DURATION_US, TOTAL_MB, PEAK_TB
bw = total_mb/1024/(duration_us*1e-6)
print(f'实际聚合带宽 {bw:.1f} GB/s')
print(f'峰值{peak_tb}TB/s 利用率: {bw/1024/peak_tb*100:.1f}%')"
```

原始观测（tool_observed）：
- msprof 汇总（summary.txt）：场景1 `GM→UB: 2081.2KB total (37.2KB/core), BW usage: 1.69%`；场景2 `BW usage: 0.40%`；场景3 `BW usage: 0.41%`。
- 手动聚合：对应 1.8TB/s 参考基准的利用率 16.2% / 4.1% / 4.4%。

解释（推断）：`GM_to_UB_bw_usage_rate` 是**单核 MTE2 活动期间**的带宽利用率（相对单核理论峰值；56 核各自统计再取平均），而手动聚合是**全芯片**实际搬运量 ÷ 总时长的平均带宽 ÷ 芯片总峰值。单核活动占比低（MTE2 ratio ~52-61%）+ 分母口径不同，导致 msprof 字段系统性小于聚合口径约一个数量级。

## 设计决策

- 验收报告口径选择：主指标用 msprof `bw_usage_rate`（与 msprof 字段参考文档的达标标准 >60% 一致），但必须并列给出手动聚合口径作参照，并写明两个口径含义——单一数字无法回答"离 60% 目标还差多远"的量级问题。
- 峰值带宽取值：1.8TB/s 为 Atlas A2 参考值，Ascend950PR 实际峰值未查到官方数（报告注明"实际利用率可能更低"）——聚合口径的绝对百分数因此只作量级参考，跨平台比较无效。

## 系统信息

- 环境：Ascend950PR（56 VectorCore，1650/1650MHz），CANN 9.1.0，msprof op --warm-up=10。
- 采集物：`docs/perf/round_001/scene{1,2,3}_*/`（8 CSV × 3 场景）+ 各场景 summary.txt + 总 summary.txt；验收报告 `docs/performance-report.md`。
- 参考文档：msprof 字段参考 `csv_fields_reference.md`（字段表：`GM_to_UB_bw_usage_rate(%)` 仅标注"GM→UB 带宽利用率，达标标准 >60%"，未给出单核/聚合口径定义）。

## 验证信号

来源说明：性能验收 round_001 三场景的 msprof 汇总与手动计算并列对照（同机同日）。

- 口径差观测：场景1 1.69%（msprof）vs 16.2%（手动）；场景2 0.40% vs 4.1%；场景3 0.41% vs 4.4%（均为同一次采集的两个口径）（来源平台：950；证据等级：tool_observed）。
- 聚合利用率：三场景总搬运 4.0 / 64.0 / 256.0 MB，Task Duration 取自 OpBasicInfo.csv、峰值取 1.8TB/s 参考值，折算 16.2% / 4.1% / 4.4%（python 计算）（来源平台：950；证据等级：tool_observed）。
- 字段参考事实：csv_fields_reference.md 对 `GM_to_UB_bw_usage_rate(%)` 的说明只有"GM→UB 带宽利用率 / >60% 良好"（来源平台：950；证据等级：tool_observed——文档读取事实；口径差异的机制解释为推断，见「误诊」一节）。

## 失效边界

- "~10x 差距"是本三场景的观测量，不是恒定换算系数；核数、MTE2 活动占比、搬运突发度都会改变差距大小。不要用 10x 反推任一口径。
- 聚合口径的百分数依赖峰值带宽取值（本例 1.8TB/s 为 A2 参考值）；峰值取错，聚合口径全体失真。
- 未验证，推断可迁移：`GM_to_UB_bw_usage_rate` 的"单核活动期"口径解释在其他 CANN 版本/芯片上是否同样成立未核实。

**未决项**：

- 向 msprof/CANN 官方文档核实 `GM_to_UB_bw_usage_rate` 的精确定义与分母口径（优先项）。
- Ascend950PR 官方峰值带宽未查到，聚合口径百分数待官方数修正。
## 坏实践

- 验收报告只写一个"带宽利用率"不写口径——评审者无法判断是单核口径还是聚合口径，数字之间不可比。
- 拿 msprof 单核口径与"全芯片 60% 带宽目标"直接对齐判定——口径不匹配会系统性低估约一个数量级。

## 误诊

- "msprof 的 bw_usage_rate 是单核 MTE2 活动期间带宽利用率（相对单核理论峰值）"这一机制解释为推断（依据两口径数值差与 mte2_active_bw 字段交叉验证），**未从官方文档核实**（csv_fields_reference.md 未定义口径）——引用时按推断对待；两口径数值本身均为 tool_observed。
