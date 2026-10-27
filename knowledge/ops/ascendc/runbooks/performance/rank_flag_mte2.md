---
type: "Runbook"
title: "逐 rank 通信路径的 MTE2 主导诊断"
description: "ReduceScatter 远程读采用逐 rank DataCopy 与同步时，msprof 显示 aiv_mte2_time 主导 T_comm；三段注释法可排除 vec/reduce，但现有证据未单独验证 flag 是带宽缺口的唯一原因。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.data_movement.data_copy", "topic.implementation.pipeline", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
aliases: ["aclshmemx_udma_get_nbi", "aclshmemx_udma_quiet", "per-PE dispatch"]
---

# 逐 rank 通信路径的 MTE2 主导诊断

## 现象

matmul_reducescatter（MC2 通算融合，4P SHMEM ReduceScatter）的 AIV 侧 `ReduceSegmentParallel` 对每个远程 rank 逐次 `DataCopy`，前后带 `V_MTE2/MTE2_V` 同步。msprof 显示 `aiv_mte2_time` 占 T_comm 绝对主导；现有实验可排除 vec/reduce 为主要耗时，但没有提供移除或合并 flag 后的受控带宽对照。

## 适用场景

- 工程形态：kernel 直调（ops-direct-invoke），MC2 多卡 ReduceScatter/AllReduce 通信层带宽诊断。
- API 路径：`DataCopy`（GM→UB）+ `SetFlag<HardEvent::V_MTE2>`/`WaitFlag` + `SetFlag<HardEvent::MTE2_V>`/`WaitFlag`（逐 rank flag 同步）；诊断用 msprof `ai_core_metrics` 取 `aiv_mte2_time`/`aiv_vec_time`。
- 验证平台与版本：Atlas A2（ascend910b，4 卡），4P M=2048 N=3584 K=5120。
- 适用粒度：`operator_family`（所有 SHMEM 被动拉取的 ReduceScatter 通信算子，凡用逐 rank DataCopy + flag 同步读远程 GM 的写法共享此瓶颈）。
- 路线优先级：`diagnostic_only`（本卡只定位 MTE2 主导，不给出同步改写结论）。
- 不适用或未验证场景：通信数据有依赖（需分段同步）时逐 rank flag 本是必需、非瓶颈；Ascend950（950）与 A3（a3）的 MTE2 流水语义推断一致但未实测。

## 判别方法

触发（tool_observed）：ReduceScatter 远程读采用逐 rank `DataCopy` + 前后 `V_MTE2/MTE2_V` flag 同步时，msprof task-based 显示 `aiv_mte2_time` 占 T_comm 绝对主导、通信带宽打不满。

探针：① msprof task-based 采集 4 卡最后 5 轮平均，取 `aiv_mte2_time` 与 `T_comm`，看 MTE2 占比；② 三段注释法——在 `ReduceSegmentParallel` 内分别注释 `Cast`/`Add` 段、注释 MTE2 `DataCopy` 段，对比 `aiv_vec_time`/`aiv_mte2_time` 变化，分解 T_compute/T_comm/T_comm_no_reduce；③ 注释 `Cast`/`Add` 后看 `aiv_vec_time` 是否归零，交叉验证 vec 是否瓶颈。

观测（tool_observed）：逐 rank 同步搬运时 `aiv_mte2_time` 占 T_comm 绝对主导；注释 `Cast`/`Add` 后 `aiv_vec_time=0.00`；三段注释法测得 `T_reduce` 相对 T_comm 约 3.4%。三者共同确认当前实现的 T_comm 主要耗在 MTE2 路径。

解释边界（interpretation）：逐 rank flag 是值得验证的候选因素，因为它位于每段 DataCopy 边界；但 `aiv_mte2_time` 主导也可能包含远程访问本身、搬运粒度和拓扑开销。必须固定数据量与 rank 数，对“原同步方式”和“合法的合并同步方式”做成对 profile，才能把带宽缺口归因于 flag。

## 验证信号

- 逐 rank 同步搬运时 `aiv_mte2_time` 占 T_comm 绝对主导（来源平台：a2；证据等级：tool_observed）
- 三段注释法测得 `T_reduce` 相对 T_comm 约 3.4%，reduce vec 非瓶颈（来源平台：a2；证据等级：tool_observed）
- 注释 `Cast`/`Add` 后 `aiv_vec_time=0.00`，交叉验证 vec 非瓶颈（来源平台：a2；证据等级：tool_observed）

## 失效边界

- 已验证：Atlas A2（ascend910b）4 卡，4P M=2048 N=3584 K=5120，tileCnt=1。
- 未验证，推断可迁移：Ascend950（950）与 A3（a3）的 MTE2 流水与 flag 同步语义推断一致，未实测。
- 通信数据有依赖（需分段同步）时逐 rank flag 本是必需，本判别不适用。

## 坏实践

- 只凭 `aiv_mte2_time` 主导就断言 flag 是唯一根因，不做合法同步改写的成对 profile。
- 只看 `T_comm` 总时长归因"通信慢"而不分解 `aiv_mte2_time`/`aiv_vec_time` 占比，漏掉 MTE2 被同步切碎的真因。
- 用 reduce vec 耗时或 vector 计算耗时解释 T_comm 主导项而不经三段注释法交叉验证——本卡探针已证 reduce 仅约 3.4%、vec 可注释归零。