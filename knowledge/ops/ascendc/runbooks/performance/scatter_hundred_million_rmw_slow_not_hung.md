---
type: "Runbook"
title: "scatter 亿级元素 RMW 灾难性慢（非挂死）"
description: "逐元素 GM 读-改-写的 scatter reduce kernel 在 ≥1 亿元素用例上单例 7~25 分钟且 AI Core util 100%——是慢不是挂死；用 npu-smi util + TTK 阶段行 + result 行数三信号判活，按规模分桶预算全量回归时长，勿盲目杀进程或提并行。"
tags: [task.debugging, topic.performance.analysis, topic.compute.indexing, topic.data_movement]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# scatter 亿级元素 RMW 灾难性慢（非挂死）

## 现象

在 Ascend950PR 单卡（CANN 9.2.0）上对自定义算子 BenchHarnessScatter（scatter/scatter-reduce）做 TTK 黑盒全量精度验收。合并后正向 667 例（L0=76 + L1=591）+ 负向 25 例；启动前规模分桶显示：最大单张量 2,147,483,648 元素（2.1B），≥1 亿元素用例 220 个，中位数 35,393,280 元素。全量以下列命令 nohup 后台运行：

```bash
python3 -m ttk aclnn -i blackbox_all.csv --pc=1 --task-prof false --run 1 \
    --warmup false --device-whitelist 0 --plugin golden.py -o result.csv
```

随后进度长时间停在同一例，很容易被当成"挂死"：

- 一个 7D shape `(4,4,2256,414,4,4)` 用例（约 2.39 亿元素）：OnProfiling 阶段 291s → 301s → 311s 仍在跑，kernel 段约 7 分钟后才进入比对；
- fp32 data 269,290,592 元素 + 8,415,331 个索引的 amax 用例（dim=-1）：OnProfiling 828s（约 14 分钟）仍在推进，单例全程约 24 分钟（含数分钟 host 侧 golden 比对）；
- 682,134,048 元素用例：OnProfiling 961s（16 分钟以上）未结束。

轮询 `npu-smi` 时这些时段 AI Core 利用率 100%、功耗约 285W、HBM 7840/131072 MB——设备满负荷计算，不是挂死。91 分钟仅完成 55/667（进度行 `Progress: 8% 55 / 667 ET: 5464s`）。同一套件里的小用例（<1000 万元素）每例只要 17~26 秒。按实测速率外推全量约需 17~25 小时，本次验收只能按部分结果如实汇报（43 例统计点：14 PASS / 29 FAIL）。

## 适用场景

- 工程形态：TTK（ops-test-kit）aclnn 黑盒全量精度回归 / 大规模用例联调，kernel 侧为逐元素 GM 读-改-写（RMW）型 scatter/index-put 实现。
- API 路径：`npu-smi` 利用率/功耗/HBM 监控；TTK `result.csv` 增量行数与进度行（`OnProfiling`/`OnComparison` 阶段 + 单例耗时）；用例 CSV 的元素规模统计。
- 验证过的平台与 CANN 版本：Ascend950PR 单卡，CANN 9.2.0，TTK（/tmp/ops-test-kit，aclnn 模式）（与 frontmatter `platforms` 一致）。
- 适用粒度：判活与时间预算方法 platform_wide（950 实测，NPU 上任何"进度停滞怀疑挂死"的场景）；"亿级元素单例分钟级"的量化特征 operator_family（绑定逐元素 RMW 型 scatter 实现）。
- 路线优先级：recommended_fastpath（判活与预算方法本身）。
- 明确不适用 / 未验证场景：真挂死（util=0 且阶段停滞且行数不增）本轨迹未遇到，判据方向为反向排除；msprof 级瓶颈分解未做；其他 NPU 型号的 util 口径未验证。

## 判别方法

**触发**：TTK/harness 大规模用例长时间停留在同一例——`result.csv` 行数不涨、进度行 `RUNNING` 的单例耗时持续增长，需要判断"kernel 挂死"还是"就是慢"，以及全量还要多久。

**探针**（三信号判活 + 规模预算，命令取自实际轮询）：

1. `npu-smi` 利用率/功耗/HBM 轮询 + 进程存活检查：

```bash
npu-smi info           # 看 AI Core util、功耗、HBM 用量
ps -o pid,etime,stat,cmd -p <TTK_PID>
```

2. TTK 进度行与阶段推进（`tail` 运行日志的进度表，看当前用例处于 `OnProfiling`（kernel 执行）还是 `OnComparison`（host 侧 golden 比对），以及单例 RUNNING 秒数）：

```bash
tail -50 <运行日志> | grep -E "RUNNING|Progress"
wc -l result.csv       # 已完成用例数（增量）
```

3. 用例规模分桶统计（全量启动前做，用于时间预算）：

```bash
python3 - <<'PYEOF'
# 读用例 CSV，解析 tensor_view_shapes 逐例求元素数，输出 max/median/分桶计数
PYEOF
```

**观测**（本会话实测）：

- 判活两态对比（同一用例前后两次轮询）：

```text
# kernel 执行段（OnProfiling）
| 0      | Ascend950PR      | OK            | 285.3       63                0     / 0             |
|        |                  | 0000:E1:00.0  | 100         0    / 0          7840  / 131072        |
# 进入 host 侧比对（OnComparison）后
| 0      | Ascend950PR      | OK            | 212.1       52                0     / 0             |
|        |                  | 0000:E1:00.0  | 0           0    / 0          6535  / 131072        |
```

- 进度序列（ET=已用秒）：3/667@ET121s → 12/667@ET842s → 19/667@ET972s → 24/667@ET1092s → 25/667@ET2472s → 39/667@ET3573s → 43/667@ET4173s → 55/667@ET5464s。
- 规模分布：total 667、max 2,147,483,648、≥100M 220 例、median 35,393,280；L0（76 例）中 ≥100M 26 例、L1（591 例）中 ≥100M 194 例——慢用例全程分布，不会"熬过开头就快"。
- 边界用例 45 例中 6 例为 2.1B 元素（int64 下单例四张量约 67GB）、23 例 ≥500M。

**解释**：判活规则——**util>0（本例 100%）+ 阶段推进（OnProfiling→OnComparison）+ 行数最终 +1 ⇒ 活跃慢，不是死锁**；只有 util=0 且阶段停滞且行数长期不增才按疑似挂死处理。机制层：该算子 reduce 模式对每个索引做一次 GM 读-改-写（amax 大用例 = 8,415,331 次逐元素 RMW + 2×269,290,592 元素搬运），GM 访问次数随索引数/元素数线性放大，亿级元素下单例 kernel 段 7~25 分钟属"预期慢"（此归因在本会话由设计文档记载与实测速率推断，未读 kernel 源码验证；判活规则本身不依赖该归因成立）。时间预算：同一全量运行内（同 kernel、同 `--pc=1`、同平台）成对对比——≥1 亿元素用例单例 7~25 分钟 vs <1000 万元素用例 17~26 秒，相差约 30~90 倍；全量时长必须按规模桶分别估算，不能按小用例速率线性外推。

## 设计决策

- 判活选"util + 阶段行 + 行数"三信号而不是直接杀进程重启：TTK 无自动断点续跑，重启会丢弃已完成结果（本会话 55 例即约 91 分钟）。
- 并行度保持 `--pc=1`：工具指引明确警告单卡并行过高会 OOM 卡死；按算术，2.1B int64 单例四张量约 67GB，`--pc=4` 四例并行约 268GB > 128GB HBM 必然 OOM，且 OOM 可能表现为整跑挂死而非单例 crash。放弃 `--pc=2`/`--pc=4` 提速方案（理论提速 2~3.6 倍，但风险与重启代价不成比例）。
- 负向 L2（25 个小用例）与正向并行先收口：HBM 竞争小，先锁定 L2 结果，正例慢跑不受影响。
- 当前方案路线档位：recommended_fastpath（判活与预算方法）；kernel 性能优化本身不在本方法范围内。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（device 0），CANN 9.2.0，TTK=/tmp/ops-test-kit（aclnn 模式）；运行参数 `--pc=1 --task-prof false --run 1 --warmup false --device-whitelist 0`。
- 相关文件（与本次运行实际读写一一对应）：`tests/blackbox/testcases/` 下的分级用例 CSV（源用例）、`tests/blackbox/ttk/blackbox_all.csv` / `blackbox_l2_neg.csv`（合并后输入）、`tests/blackbox/ttk/result.csv`（增量结果）、`docs/develop/golden.py`（TTK golden plugin）。

## 预防

- 大规模黑盒回归启动前先跑规模分桶统计（一条 Python 脚本），≥1 亿元素用例单独按"单例分钟级"量级预算（以本套件实测为参照重估；绝对时长绑定该算子实现与本套件用例分布，不可跨实现套用）。
- 全量放 nohup 后台 + 长间隔轮询（注意轮询 sleep 不要超过工具超时上限，本会话 sleep 180 曾触发 120s 超时）。
- 评测计划中与用例设计方确认超大 stress 用例（2.1B 级）的必要性与超时策略；一次全量的墙上时间要写进交付预期。

## 验证信号

以下信号来自该算子 TTK 黑盒全量回归会话的实测轮询（单次全量运行内的观测，非通用预期收益；绝对时长绑定该算子实现与本套件用例分布）：

- 判活观测 A：kernel 执行段（OnProfiling）npu-smi AI Core util 100%、功耗 285.3W、HBM 7840/131072 MB（来源平台：950；证据等级：tool_observed）。
- 判活观测 B：同用例进入 OnComparison 后 util 0%、212.1W、HBM 6535 MB——host 侧比对阶段设备空闲，阶段推进可观测（来源平台：950；证据等级：tool_observed）。
- 超大用例时长：fp32 269,290,592 元素 + 8,415,331 索引 amax 用例 OnProfiling 828s 仍活跃，单例 kernel 段约 14 分钟、全程约 24 分钟；682,134,048 元素用例 OnProfiling 961s 未结束（来源平台：950；证据等级：tool_observed）。
- 进度序列：91 分钟完成 55/667（`Progress: 8% 55 / 667 ET: 5464s`），慢用例期间 result 行数最终均 +1（未丢失）（来源平台：950；证据等级：tool_observed）。
- 成对对比（同 kernel、同 `--pc=1`、同平台）：≥1 亿元素用例单例 7~25 分钟 vs <1000 万元素用例 17~26 秒（约 30~90 倍）；套件规模 max 2,147,483,648、≥100M 220/667 例、median 35,393,280（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判活规则在 Ascend950PR + CANN 9.2.0 + 该版本 TTK 实测；其他 NPU 型号的 util 口径/功耗读数未验证。
- "亿级元素 7~25 分钟/例"绑定该算子当时的逐元素 RMW 实现与本套件用例分布；换 kernel 实现（如批量搬运/向量化改造）或平台后须重测。
- "逐元素 GM RMW 是瓶颈"为机制推断（设计文档 + 速率对应），未做源码级验证与 msprof 分解；不排除其他因素（如同步密度）贡献。
- 全量最终结果（55/667 之后）在本次会话截断处未完成，本卡不含最终通过率结论。
- 本卡只提供判别与预算，不提供 kernel 性能修复；同算子家族"2 字节逐元素 RMW 大规模丢写"的正确性边界另见 [2 字节逐元素 GM RMW 大规模丢写判别卡](../precision/two_byte_gm_rmw_large_scale_lost_write.md)。

**未决项**：

- 该 kernel 的性能优化（瓶颈分解、向量化/批量搬运改造、同步合并）属 kernel 开发职责，本卡只提供判别与预算，不提供修复。
- 逐元素 GM RMW 机制的源码级验证未做：接续点为读 `op_kernel` 的 reduce 分支确认每索引 GM 访问模式，并用 msprof 做带宽/指令分解。
- 2.1B 极限用例（6 例）的完整单例时长未观测到（会话截断）；全量最终通过率待补跑。
## 坏实践

- 见进度停滞就杀进程重启：TTK 无断点续跑，已完成结果全部作废。
- 单卡盲目提高 `--pc` 提速：2.1B int64 用例 4 例并行约 268GB > 128GB HBM，OOM 可能挂死整跑而非单例 crash；提速收益（2~3.6 倍）远小于挂死风险。
- 按小用例速率（约 20 秒/例）线性外推全量时长——会低估一个数量级以上（667×20s≈3.7h vs 实测外推 17~25h）。

## 误诊

- 轮询早期"ET 只涨、rows 不增"曾被当作疑似挂死；三信号（util 100% + OnProfiling→OnComparison 推进 + rows 最终 +1）证明是活跃慢。（agent_reported：会话中曾多次考虑切换 `--pc=4` 提速并被反复否决，最终按工具指引保持 `--pc=1`——该决策无实测对照，仅风险论证。）
- "AI Core util 100%"曾被默认等价于"性能没问题/快"——本例说明 util 满载只证明设备在算，不代表有效吞吐高；逐元素 RMW 型 kernel 可以 100% util 地慢。
