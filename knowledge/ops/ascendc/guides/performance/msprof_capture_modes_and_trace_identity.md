---
type: Guide
title: msprof 采集模式与结果身份
description: 保存真机、仿真和CSV时间线交叉分析，限制示例CLI与空字段解释。
tags:
- task.optimization
- topic.performance.analysis
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/msprof_usage_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/references/msprof-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/references/msprof-op-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/msopprof-visualization/references/collection-workflow.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/msopprof-visualization/references/environment-and-failure.md
  role: supporting
created_at: '2026-09-11T04:58:58Z'
updated_at: '2026-10-08T14:20:00Z'
platforms: []
aliases:
- PipeTimeline
- PCSampling
- PcSampling
- instrTimeLine
- sink_file_mutex
---

# msprof 采集模式与结果身份

msprof op真机采集用于当前设备的执行时间、流水活动与内存计数；simulator路径可提供指令时间线、热点及访存波形，但其目标配置和模拟成本不同，不能把仿真时延直接当真机性能。代码热点映射通常需要合适的调试/行号信息，Ascend C编译-g是来源线索；Triton环境变量不作为Ascend C配置迁入。

来源示例为msprof op --kernel-name=<name> <test-command>，仿真再指定simulator与--soc-version，另给--export-timeline、--output-dir。具体子命令、参数名和可采指标依工具版本/设备支持，原文件未给每项对应的help或执行记录，故这些是待核验用法，不能宣称所有CANN发行都支持。源码函数名是查找kernel的起点，实际编译符号、模板实例和多次launch可能改变采集对象；应绑定运行中实际任务身份。

来源8类CSV为OpBasicInfo、ArithmeticUtilization、PipeUtilization、ResourceConflictRatio、Memory、MemoryL0、MemoryUB、L2Cache，另有insight_view、pipeline与hotspot等视图。op_summary可能由多个表合并，并非每个采集模式天然导出；合并应按device/process/task/kernel/block及时间身份，不能直接按行号拼接。某文件或字段缺失可能为未采、不可用或不适用，不代表值0。

Task Type决定可解释的执行侧，但不能由纯AI_CORE字样机械断言永远没有aiv字段，需结合实际核型、混合模式与导出schema。流水图空白可能是筛选对象、采集支持、目标配置或调试映射问题，来源<1us无法捕获只是一种推测而非固定硬件阈值；-g缺失也不能统一解释所有指令时间线为空。

Wait-Wait、Starvation、Transfer-Compute Gap、Sync Overhead是时间线候选形态，可与Idle Cycles、Overlap Ratio、Sync Density及CSV计数交叉。重叠比须定义分母和参与资源，等待长并不唯一指向同步错误，可能是生产者计算、访存或负载不平衡。热点行号需对应同一binary/源码版本，CSV活动比不能直接替代因果时间线。

优化前后及不同shape数据应隔离保存采集配置、输入和制品身份；热身、重复数、频率及测量范围保持可比，不能只看一个Task Duration字段。来源工作目录、Phase升级与复制文件命令属于Skill流程，未迁入执行指令。

## task-based、sample-based 与周期聚合

固定来源区分任务型与采样型观测；具体 CLI、可用表和字段随版本变化。aicore.db 的 task_cyc、逐核周期与 op_summary 的任务聚合时延不能只按名字视为同一窗口。采样到的活动比例也不等同于某次任务的连续执行时间，跨轮采集不同计数组需要同输入、制品、任务和稳定执行条件。

用周期/时长反推频率之前，先确认周期属于同一核型与任务、时间已扣除或包含哪些边界，以及频率是否变化；不拿全部核的周期和除以单任务墙钟。多轮多 rank 汇总中 max(mean(rank)) 与 mean(max(rank)) 含义不同，前者不能代替每轮最慢 rank 的平均时间。

来源的 3/5/10 次热身、固定时长阈值、MC2 判别和同地址访问必然串行均需对应版本和执行证据，不提升为硬规则。验证应保留原始单位、单次样本、任务身份和聚合公式，并分别报告采集范围及不可用项。

## 指标名称、采集范围与实际产物

固定 reference 描述的一套 msprof op 能力对应如下。可用项以本次安装版本的 `msprof op --help` 为准，保留大小写；本表不保证所有产品/版本均支持，也不把可视化页面名当作原生 CLI 参数。

| 观测对象 | 来源指标或材料 | 使用边界 |
|---|---|---|
| 核/子核占用 | Occupancy | 用实际 Details/占用数据解释，不能由 Kernel 名推导 |
| 计算、内存、缓存计数 | MemoryDetail | Roofline 只有确含所需完整 CSV 才能复用，不由标签推断 |
| Roofline | Roofline | 可能同时包含标准 CSV，实际检查字段、单位与任务身份 |
| 流水时间线 | PipeTimeline | 来源要求独立 replay，不能假装与普通计数器来自同一次运行 |
| 源码映射 | Source | 所选二进制的实际构建树必须有对应 .debug_line |
| Warp Stall | PCSampling 或 PcSampling | 拼写以当前 CLI 为准；须有对应 SIMT/采样材料 |
| 指令时间线 | instrTimeLine、TimelineDetail,Default 或 TimelineDetail | 是来源列出的不同可用入口形态，不能无条件互换 |
| 片上内存分配 | memory_info.json | 该工具从已有材料取得，不为此执行 msprof replay；不是 MemoryDetail 计数 |
| 定向 KernelScale 采集 | KernelScale | 只说明来源列出该项，不从名称补造指标计算公式 |

来源在没有 Default 入口时还列出标准计数器组合 ArithmeticUtilization、L2Cache、Memory、MemoryL0、MemoryUB、PipeUtilization、ResourceConflictRatio；仅能选择当前 CLI 实际提供的成员，不能保证全组合一定可同时采集。Raw Data 的实质是已有标准 CSV；重复执行另一轮采集会引入 replay 差异，并不天然提供更“原始”的同一次结果。复用需材料完整、任务身份一致和解释范围成立；超时、非零、partial 或 empty 产物不能补成一次完整采集。

计数、分配生命周期、PC 采样与指令事件的解释复用[不同观测层](../diagnostics/profiler_memory_source_and_sampling_evidence.md)。独立 replay 即使使用相同输入/制品也不是同一次时序记录，跨轮比较应保留采集轮次与波动，不能按行号拼出因果链。

## Source 的调试信息属于哪个制品

`-g` 属于实际 Kernel/可执行文件的编译命令，Source 才是 msprof 的指标；不能把 -g 当采集参数补传。检查 .debug_line 应限制在所选可执行文件所属的精确构建树，旁边另一个 build 目录有符号不证明当前制品有对应行号。重新构建后 binary、源码快照和采集材料必须共同更新，不能用同名文件代替匹配。

记录 app 路径、实际工作目录和输入解析位置；相对输入依赖 cwd，另一目录下同名可执行文件或相对文件解析成功也不证明采到了用户指定的算子。目录评分、默认路径选择和条件重建命令是调用方机制，不是 msprof 的通用选择合同。

## 返回码、目录权限与共享设备锁

退出码 0 不等于已有有效产物。固定来源报告过输出目录对 group/other 可写时 msprof 返回 0 但不生成产物的情形；本卡未复现这一现象，因此它是需要按工具版本核对的诊断线索，不是所有 msprof 版本的必然行为。结合输出目录权限、日志和本次生成的 OpBasicInfo/实际可解析 payload 判断，不能用旧文件掩盖新一轮空采集，也不应未经确认盲目改变目录权限。

缓存的环境/能力探测信息依赖主机身份、环境来源脚本的路径/时间戳/内容以及 msprof 的绝对路径和文件身份。这些改变后，应重新确认实际环境与能力；主机曾成功采集不证明另一个算子的 binary、输入与权限均已检查。

共享设备或锁故障的证据包括本次 npu-smi 状态、/proc/locks、Ascend `sink_file_mutex_*` 文件身份，以及相关 holder/waiter 的 PID、状态、wchan、父进程和 cwd。锁文件存在或进程等待本身不唯一证明死锁；要关联占用者、本次采集进程和时间。超时或父进程结束不证明所有后代进程及设备活动已经清理，残留可能影响下一次采集，诊断需保留结束方式与进程身份。

上述补充为固定 reference 的合同与观测范围，未执行采集、权限修改或进程清理。

## 补充证据（2026-10-08，DS §110/§113 集合通信 replay 实测）

本节只补充一次实测观测，不修改上文任何结论，也不把本次倍率上升为通用定律。

**观测绑定**（倍率只在此窗口内成立）：

    工具    msopprof，CANN 9.1.0 厂商安装（未测其他构建）
    算子    MegaMoE arch35（Ascend950DT），**两卡集合通信**算子
    负载    H=7168，2*FFN=1024，topk=8，bs=512，epWorldSize=2
    采集    --launch-count=3 --warm-up=2
    样本    每个 replay 模式 4 次 capture
    对照    A-A 空对照（同一 binary 两臂）+ 完全不挂 profiler 的时延 A/B

**一、在集合通信算子上，上文的"replay 差异"在本次表现为换挡而非扰动**

上文已指出「独立 replay 即使使用相同输入/制品也不是同一次时序记录」。本次实测中该差异的量级为：

    profiled 总时长（vector1）   44 / 44 / 644 / 905 ms（同一臂 4 次）
    真实不挂 profiler 的同形负载  约 1.4 ms
    ⇒ 本次观测到的总时长相对未插桩值高出约 24x~628x，且呈**双峰**：
       4 次中 2 次落在约 34~44 ms，2 次落在约 290~905 ms

**同一臂重复之间即相差十几倍**，因此在本次窗口内，总时长及其派生比率无法承载臂间比较。
⚠ 该倍率是**本次实测记录**，不是工具常数；双峰的成因未查明，故连本窗口内的量级也不可预测。
⚠ 本节**不主张**"只有 work 时间可比"这一普遍结论；本次 work 时间仅被验证为**重复稳定**
（臂内离散 0.8%），其数值未与任何非 profiler 来源交叉验证，故只有方向获得旁证。

**二、`--replay-mode=kernel|application` 不是本次问题的调节旋钮**

`msopprof --replay-mode` 在本次安装中默认 `kernel`。曾假设：重放单卡 kernel 而对端不同步
会放大跨卡自旋等待。**该假设在跑之前被写成可证伪判据**（换 `application` 后总时长应塌向
约 1.4 ms、rank 不对称应收向无 profiler 的约 1.2%），**实测结果为否**：

    kernel       ctrl 44.09 / 44.14 / 644.15 / 904.62 ms
    application  ctrl 44.31 / 44.51 / 554.44 / 745.18 ms

两个分布形状一致，rank 不对称同为 2x~17x（无 profiler 时为 1.0005x~1.0122x）。
⇒ 在本次窗口内，切换 replay 模式不改变该现象；**成因仍未确定**。
（application 模式 4/4 首次成功、kernel 模式有 2 次失败，属可用性差异，非测量差异。）

**三、让该差异现形的控制是 A-A 空对照**

本次另一组实测中，仅比较 ctrl 与候选两臂时，`Memory` 组曾给出 -15.23% 的表观差异；补上
**同一 binary 的 A-A 空对照**后，该差异的真值为 +0.06%——原差异来自单次 capture 中单个
device 的异常抬高。⇒ 在用 profiler 判定任何臂间差异前，**A-A 空对照是必要控制**；
仅有 A-B 两臂时，无法区分"效应"与"该指标在本窗口内的轮间波动"。
