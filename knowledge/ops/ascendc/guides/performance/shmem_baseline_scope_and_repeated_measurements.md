---
type: Guide
title: SHMEM 基线范围与重复测量的可比性
description: 补充 fused/stitched/proxy 基线、inline 与 msprof 关联及状态复位成本。
tags:
- task.testing
- topic.performance.analysis
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/baseline-compare-workflow.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/perf-chat-output-spec.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/perf-compare-spec.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/perf-workflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/performance-eval-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/skill-performance-review.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/fused-compute-phase-gates.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/agent-execution-contract.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/readme-spec.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-review/references/review-report-template.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM 基线范围与重复测量的可比性

性能基线要完成同一local compute、通信和finalize，并使用相同shape、dtype、PE/team、sendcounts、路由及输入状态。融合Matmul+collective不能仅与单HCCL通信比较。可用同语义fused ACLNN，或计算+HCCL串行拼接，并标明后者无overlap；API名相似、存在GetWorkspaceSize或调用成功都不证明语义一致。支持范围与弃用状态按版本核对，来源P1/P2/P3优先级是选型流程。

Host memcpy或模拟通信proxy只能观察其自身行为，不能作为真实HCCL/ACLNN性能证据。Round0是内部改进参照，与外部基线分开。Python或C++ runner均需明确实际执行及计时范围，来源强制C++只是控制测量环境的工程约定。

来源推荐分进程/会话、分profile目录测HCCL与SHMEM以避免上下文、设备负载和残留会话互扰；“绝不可同shell”与固定等待30秒不构成库兼容性定理，真正要检查进程退出、资源回收和环境一致。HCCL展开模式AIV/AI_CPU应记录实际生效值，模式差异可能影响小消息启动成本，不能拿不相同路线隐藏基线差异。

inline_event的e2e/kernel值与msprof task_time值有不同边界。通信主指标可选同口径kernel，而融合与stitched完整流程通常需完整输入就绪到输出可用跨度；不能把某个task_time与另一侧完整e2e混为比值。来源部分文件强制kernel、另一些要求融合e2e，按真实比较合同决定，保留工具与区间标签。

同一调用可含多个task/stream。warmup=10不是任意丢弃前10条CSV，按实际调用轮次/PE/stream匹配正式区间；输出缺行、未匹配或0次不能记0耗时。msprof --hccl、--task-time=l2等选项是来源版本命令，解析应按实际输出schema和单位。tee管道需保留真实runner退出码，输出一行[PERF]不证明完整采集成功。

固定payload口径时带宽比B_current/B_base等于T_base/T_current；延迟降低百分比为1-T_current/T_base，与带宽提高百分比T_base/T_current-1不同。例如性能≥基线80%对应T_current≤T_base/0.8，是最多25%更慢，不能误写≤0.8*T_base。来源80%/20%与5%分支收益门槛是验收策略，不是通用硬件效率要求。

小大消息体现不同启动/稳态机制，稀疏或延迟主导用例可以在相同数据量上有效比较，无须先打满带宽。计算性能op_count/time与通信payload/time采用各自单位和范围，通算融合e2e归一化带宽不等于纯链路吞吐，细分frame百分比也可能重叠。

重复测量保持每轮功能输入及累加初态。H2D、memset、staging是否纳入由合同定义；把它们移出循环仅当同一轮不再需要且基线等价时成立。仅warmup清零可能把重复归约变为逐轮累加；反之每轮不必要的memset也会测到额外工作。量化、调度或buffer改变后仍需关联正确性结果，性能文件存在不能覆盖失败/跳过case。
