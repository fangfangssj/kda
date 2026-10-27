---
type: "Runbook"
title: "逐元素 DataCopyPad 搬运放大 25~64 倍判别"
description: "STRIDED 场景 kernel 逐元素 DataCopyPad，每条 MTE2 指令平均搬 0.13KB（32B 对齐块）却只取 2B 有效数据，实际搬运量为理论需求的 25.6~64 倍，带宽利用率被放大项压死；用 msprof Memory.csv 放大倍数 + 指令数/元素数 1:1 + Avg transfer 三探针判别。"
tags: [task.optimization, topic.data_movement.data_copy, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# 逐元素 DataCopyPad 搬运放大 25~64 倍判别

## 现象

在 Ascend950PR（56 VectorCore，满频 1650/1650MHz）上对 BenchNoHarnessStridedSlice 做性能验收（msprof op，warm-up 10 + 100 循环），三场景带宽利用率全部远低于 >60% 的目标：msprof `GM_to_UB_bw_usage_rate` 1.69% / 0.40% / 0.41%，手动聚合口径也只有 ~16% / ~4% / ~4%。其中两个 STRIDED 场景出现惊人的搬运量放大（Memory.csv 实测）：

- 场景2 STRIDED_POS（[1024,1024] fp16，strides=[2,2]，输出 [512,512]）：理论需搬运 2.5MB（读 2MB + 写 0.5MB），实测主存读 32796KB + 写 32768KB ≈ 64MB——**数据放大 25.6x**；MTE2 指令 262,312 条 ≈ 输出元素数 262,144（1:1 对应），Avg transfer 0.13KB/指令。
- 场景3 STRIDED_NEG（[1048576] fp16，strides=[-1]，全逆序）：理论 4MB（读 2MB + 写 2MB），实测读 128MB + 写 128MB = 256MB——**数据放大 64x**；MTE2 指令 1,048,744 条 ≈ 输出元素数 1,048,576（1:1 对应），Avg transfer 0.13KB/指令。

对照场景1 CONTIGUOUS（strides=[1,1]）：理论 4MB、实测 4.03MB（放大 1x），Avg MTE2 transfer 1.67KB/指令——证明放大只在"逐元素搬运"的 STRIDED 模式出现。

## 适用场景

- 工程形态：AscendC kernel 性能验收/瓶颈定位（msprof op 单算子采集 + Memory/PipeUtilization CSV 解读）。
- API 路径：kernel 侧 `DataCopyPad`（MTE2/MTE3 搬运）；msprof `Memory.csv`（`GM_to_UB_datas`/`UB_to_GM_datas`/主存读写/Avg MTE2 transfer）、`PipeUtilization.csv`（MTE2 指令数与占比）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"按索引/步长逐元素搬运"的 kernel（strided slice、gather/scatter 类）在向量核上的搬运粒度放大规律。
- 路线优先级：recommended_fastpath（带宽验收不达标时的第一排查方向）。
- 明确不适用 / 未验证场景：Cube/MIX 路线、通信搬运未验证；数据放大倍数随 dtype/步长变化（本证据为 fp16、stride=2/-1），套用时应按"最小搬运粒度/有效字节"重算。

## 判别方法

触发条件：访存密集型算子带宽利用率远低于目标（msprof bw_usage <2% 或手动聚合 <20%），且 STRIDED/索引类场景比 CONTIGUOUS 场景差一个数量级。

可执行探针（用 msprof Memory.csv/PipeUtilization.csv 三个量直接判定）：

1. 放大倍数 = (主存读取 KB + 主存写入 KB) / 理论搬运字节（读输入 + 写输出）。理论值按用例 shape×dtype 手算。
2. MTE2 指令数 vs 输出元素数：两者 1:1（262,312 ≈ 262,144；1,048,744 ≈ 1,048,576）即逐元素搬运。
3. Avg MTE2 transfer：0.13KB/指令（≈133B，32B 对齐块）而元素只有 2B（fp16）。

```bash
# 每场景单独采集（一次 msprof op 只出单 kernel 数据）
msprof op --warm-up=10 --output=./msprof_scene2 ./perf_demo 2 100
# 判定：汇总 Memory.csv 主存读写总量，与理论值对比
python3 -c "
# 输入：OpBasicInfo.csv 的 Task Duration(us) + Memory.csv 主存读/写 KB（场景2 实测读 32796KB / 写 32768KB）
duration_us, rd_kb, wr_kb = DURATION_US, RD_KB, WR_KB
total_mb = (rd_kb + wr_kb) / 1024
print(f'实际搬运 {total_mb:.1f}MB, 耗时 {duration_us}us, 实际带宽 {total_mb/1024/(duration_us*1e-6):.1f} GB/s')
print(f'理论需搬运 2.5MB, 数据放大 {total_mb/2.5:.1f}x')"
```

解释：kernel 对 STRIDED_POS/NEG 模式逐元素搬运（每个输出元素一次 DataCopyPad），搬运硬件最小粒度 32B 对齐，每条指令实际搬 ~133B 却只取 2B 有效数据，有效数据率 2/133 ≈ 1.5%——读侧与写侧同时放大，总搬运量膨胀 25~64 倍，聚合带宽被压到 CONTIGUOUS 场景的约 0.25x（参照路线=场景1 CONTIGUOUS 全量拷贝；来源平台：950；同机同日成对对比）。

## 设计决策

（本卡为诊断规则，无代码修复落地；优化方向按验收报告记录如下，供下个迭代接续。）

- [P0] STRIDED_POS：改用带步长的批量搬运（按行批量读取间隔数据），预期 MTE2 指令数减少 ~100x、消除放大。
- [P0] STRIDED_NEG：分段连续逆序搬运（把逆序切成正向连续段），或使用支持反向访问的 GatherDMA。
- 两条均为未实施建议，本开发会话 未验证收益。

## 系统信息

- 环境：Ascend950PR（DAV_3510，56 VectorCore，1650/1650MHz 满频），CANN 9.1.0；msprof `--warm-up=10`。
- 采集物：`docs/perf/round_001/scene{1,2,3}_*/` 各 8 个 CSV + summary.txt（OpBasicInfo/PipeUtilization/Memory/ResourceConflict/L2Cache 等）。
- 相关文件：`tests/perf/perf_demo.cpp`（三场景构造，含 expected_tiling_key）；kernel 侧 STRIDED_POS/NEG 逐元素搬运实现（op_kernel）。

## 预防

- 任何按索引/步长访问的 kernel，先按"最小搬运粒度 / 每元素有效字节"估算放大上界再选搬运形态，不写逐元素 DataCopyPad 循环。

## 验证信号

来源说明：性能验收 round_001 的 msprof CSV 实测与汇总计算（同机同日，三场景各自单独采集）。

- 场景2：主存读 32796KB / 写 32768KB，理论 2.5MB，放大 25.6x；MTE2 262,312 条、Avg transfer 0.13KB（来源平台：950；证据等级：tool_observed）。
- 场景3：主存读 131100KB / 写 131072KB，理论 4MB，放大 64x；MTE2 1,048,744 条、Avg transfer 0.13KB（来源平台：950；证据等级：tool_observed）。
- 对照场景1：读 2081.2KB / 写 2053.2KB ≈ 理论 4MB（放大 1x）；Avg transfer 1.67KB（来源平台：950；证据等级：tool_observed）。
- 聚合带宽成对对比（总搬运/耗时）：STRIDED 两场景（场景2/3）相对 CONTIGUOUS 参考路线（场景1，同机同日成对对比）约 0.25x（来源平台：950；证据等级：tool_observed）。
- 验收结论：带宽利用率未达标（msprof 0.40%~1.69% / 手动 4%~16%，目标 >60%），报告状态失败、不阻塞流程并记录优化建议（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 放大倍数不是常数：25.6x/64x 绑定本用例的 fp16 + stride 2/-1 与 32B 对齐粒度；换 dtype/步长应按 `最小搬运粒度 / 每元素有效字节` 重算上界（fp16 stride=2 时相邻元素间隔 4B，32B 块内有效密度更高，故读侧 16x、写侧 64x，混合后 25.6x）。
- 本判定假设"理论搬运量 = 读输入一遍 + 写输出一遍"；若算法本身需要多遍读取，放大倍数口径要相应调整。
- 优化建议（批量带步长搬运/分段逆序）未实施未验证，不能当作已验证收益引用。
- 推断可迁移（未验证）：其他向量核平台的最小搬运粒度若同为 32B 量级，规律可参照。

**未决项**：

- "每条指令 ~133B"的精确构成（32B 对齐 × burst 组织方式）未在硬件文档层核实，为对 Avg transfer 观测值的解释；实施批量搬运优化后应复测放大倍数验证本机制（优先项）。
## 坏实践

- 只看 msprof `bw_usage_rate` 一个数就下"带宽不行"的结论——不看搬运总量与指令数就无法区分"放大型低带宽"（本卡）与"串行型低带宽"（读写无流水重叠，见 [mte2/mte3 wait 双高示读写串行无重叠](mte2_mte3_wait_double_high_serial_no_overlap.md)），优化方向完全不同。

## 误诊

- 无（本闭合从 CSV 数据直接判定，无错误归因轮次）。
