---
type: Guide
title: 通信与通算融合的计时边界和带宽口径
description: 对齐 SHMEM 与基准实现的数据边界，区分多卡统计、任务时长与调用跨度，并明确带宽分子的单位和含义。
tags:
- task.debugging
- topic.performance.analysis
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/timing-and-metrics-standard.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/profiling-tools.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/baseline-selection.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/hccl-expansion-mode.md
  role: supporting
created_at: '2026-09-10T13:50:53Z'
updated_at: '2026-09-10T13:50:53Z'
platforms: []
---

# 通信与通算融合的计时边界和带宽口径

比较 SHMEM 与 HCCL/ACLNN 或串行拼接实现前，应确认它们完成相同的数据转换与通信语义，使用相同的 shape、dtype、PE 数和路由分布。算子名称相似不证明语义等价，例如 Dispatch/GMM/Combine 还包含路由、布局、量化及结果合并，不能仅以两个 AllToAll 的存在证明拼接基准完整。

计时区间需要从相同输入状态开始，到相同输出可用状态结束。若一种实现先将普通 GM 数据暂存到对称内存，另一种实现把同类工作包含在 Kernel 内，只比较后续 Kernel 时长会遗漏前者成本。是否需要暂存由具体引擎和地址支持决定；外部暂存耗时也不一定恰好等于两种计时的差值，因为还可能有调度、等待及重叠。

Host 墙钟、stream event、profiler task duration 和一次完整调用的时间跨度是不同观测。应记录每个值的起止点、同步条件、采集工具及版本，不因字段名含 kernel 就将其解释为纯设备执行。融合与拼接的比较应保留实际调用跨度和组成任务；多个任务有重叠时，耗时之和不能代替跨度。独立进程或设备的时间戳也需确认具有可比时钟基准。

多 PE 统计应保留 rank 和迭代身份。`max_r mean_i(T[r,i])` 表示平均耗时最大的卡，`mean_i max_r(T[r,i])` 表示逐轮最慢卡耗时的平均，两者不同。例如两卡两轮耗时分别为 `[10,1]` 和 `[1,10]`，前者为 5.5，后者为 10。前者可描述持续较慢的卡，却可能掩盖在不同卡间移动的长尾；二者都不能自动替代跨卡调用的真实完成跨度。

warmup 与正式测量应按调用轮次区分。一轮有多个 task 时，跳过固定数量的 CSV 行不等于跳过固定轮数。过滤应使用实际的设备、stream、task 和调用关联信息；不能无条件把 memcpy/memset 作为噪声移除，它们是否属于被比较语义需要单独判断。

令 B 为声明口径的逻辑数据字节数、t 为秒，则 `algo_GBps = B / t / 1e9`。若时间为微秒，则是 `B / t_us / 1e3`。B 必须说明是单 PE 输入、输出还是全局数据量，不能将微秒直接代入以秒为单位的公式。

约定的 bus bandwidth 可写为 `algo_GBps * f`，其中 f 来自给定通信算法和 B 的定义。均匀 AllToAll 使用单 PE 总发送数据为 B 时，排除 self 后的远端比例可为 `(P-1)/P`；不均匀路由需按实际远端字节推导。AllGather 输入与输出的数据量不同，不能脱离分子定义机械套相同系数。这种归一化带宽也不是链路上真实字节计数。

与峰值比较时，方向、链路或聚合范围、拓扑、PE 数和单位应一致。来源中的 28/196 GB/s 及 80%/20% 通过线不具有全平台效力；低 payload 场景也可能以时延为主。HCCL 的展开模式、确定性设置和实际生效路径会影响基准，应记录它们，不能把一次初始化失败直接解释为某模式不受支持。

验证报告应能由原始事件与路由数据复算耗时和字节数，同时保留正确性结果及测量扰动范围。实测是否改善由同条件对照确定，不能从更换引擎、增加并发或减少 task 数本身推断。
