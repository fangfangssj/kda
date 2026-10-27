---
type: Guide
title: MC2 计时、轮次身份与抖动统计
description: 提炼多卡时空波动分析，纠正按CSV行序配对和固定预热剔除。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/MC2_JitterDiagnose.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/index.md
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# MC2 计时、轮次身份与抖动统计

MC2可从同卡多次执行的时间变化和同轮不同rank的空间差异两个方向分析。op_summary常见Device_id、Op Name、Task Duration(us)、aicore_time/aiv_time及各自MTE2/MTE3字段，列名、单位与语义按实际版本确认；应先过滤同一kernel、shape、stream和采集模式。每卡CSV第j行不自动属于全局第j轮，缺记录、额外kernel或异步启动会错配，需要运行序号/相关ID等可信身份。

跨卡同步路径关心最慢参与者，但max(rank各轮均值)与每轮max(rank耗时)再取均值是不同统计量，后者更接近对齐轮次的耗时分布；二者都不能取代包含跨rank起止关系的端到端关键路径。单独aiv_time可诊断通信/等待，整体择优应使用匹配范围的Task Duration或完整collective计时，不能只降低aiv时间便认为整体更快。

Count/min/max/mean/std/CV=std/mean及range可描述抖动，mean0、缺失值、样本数1和std样本/总体分母应明确。来源CV2%/5%分级、首次超过后续均值3倍判warmup、10轮丢前5留后5都是案例统计政策，不是通用稳定性阈值；保留原始序号、全量与剔除后结果，不能因慢点拉高坐标就删除。

预热需按真实cache、编译、分配、频率和通信初始化证据区分，所有rank使用一致的测量轮次集；若分别去掉不同行再重编号会制造空间匹配。采集目录每设备独立产物需核验全部参与rank完整，msprof包裹fork程序并不免除验证采集是否覆盖所有子进程。

Device profiling可隔离kernel执行，Host计时可用于含launch/同步的端到端开销，两者量测对象不同；来源全面禁止Host计时属于Skill工作流限制，不能提升为技术定律。纯通信/纯计算隔离版本必须有合法输入及闭合事件协议，不能直接注释Process留下Wait。时间线上AIC/AIV/MTE可重叠，阶段时长不可简单相加当总时间。

掩盖率(Tserial-Tfused)/min(Tcompute,Tcomm)与加速比Tserial/Tfused须输入正且同shape/拓扑/计时边界。掩盖率≤1仅在serial=compute+comm且工作量/效率不变的理想模型成立；切分改变cache、计算量或额外开销时可超1或负，不能截断掩盖异常。tileCnt1也不自动等于完全串行，LocalMatmul或其他重叠仍需检查。
