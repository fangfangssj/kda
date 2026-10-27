---
type: "Runbook"
title: "CrossCore 反压 flag 方向错致通算流水死锁"
description: "MC2 通算融合算子（Pattern B 先计后通、AIC 写大容量 GM、AIV 读）从数据流反向的基底工程照搬 AIC WaitFlag(t-buf)+AIV SetFlag(t-buf) 反压，在 tileCnt>bufferSize 时形成 AIC t↔AIV t 循环等待死锁。判别=画出 t 与 t-bufferSize 的 flag 依赖图，出现 AIC t↔AIV t 互等即循环依赖。根因是反压方向与数据流不匹配，且 cGm_ 容量足以一次性容纳所有 tile 时本不需要反压。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.programming_model.cooperative_groups"]
status: stable
sources: []
created_at: "2026-08-18T07:59:48Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["a2"]
---

# CrossCore 反压 flag 方向错致通算流水死锁

## 现象

为 matmul_reducescatter（MC2 通算融合，4P SHMEM ReduceScatter，Pattern B 先计后通）实现通算流水时，按改造指令给 `MatmulProcess`（AIC）和 `ReduceScatterProcess`（AIV）各加 mLoop 循环，并用 ping-pong 反压：AIC 在 `t >= bufferSize` 后 `CrossCoreWaitFlag<0x2, PIPE_FIX>(t - bufferSize)` 等 AIV 释放旧 buffer，AIV 在 reduce 后 `CrossCoreSetFlag<0x2, PIPE_MTE3>(t - bufferSize)` 通知 AIC。bufferSize=2 时 tileCnt=1/2 精度 PASS、不挂；**tileCnt=4 时 `bash run.sh ... precision 4` 挂起超时**，bash 在 300s timeout 被杀死，无返回，疑似死锁。

被诊断的反压 flag（改造指令中实际应用的两个完整方法，verbatim 取自 trace 中 seq 129 改造指令 user_message）：

```cpp
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void MatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::MatmulProcess()
{
  Params params;
  SetupParams(&tilingData_->tileQbmmTilingData, params);

  params.mmadParams.aGmAddr = aGm_;
  params.mmadParams.scaleAGmAddr = scaleAGm_;
  params.mmadParams.bGmAddr = bGm_;
  params.mmadParams.scaleBGmAddr = scaleBGm_;

  for (int32_t t = 0; t < static_cast<int32_t>(tileCnt_); ++t) {
    if (t >= static_cast<int32_t>(bufferSize_)) {
      CrossCoreWaitFlag<0x2, PIPE_FIX>(t - static_cast<int32_t>(bufferSize_));
    }

    uint64_t mOffset = static_cast<uint64_t>(t) * headMSize_;
    params.mmadParams.cGmAddr = cGm_ + mOffset * axisN_ * sizeof(CType);
    params.localParams.localAGmAddr = aGm_ + mOffset * axisKa_;
    params.localParams.localScaleAGmAddr = scaleAGm_ + mOffset * scaleKaSize_;

    quantMatmulKernelImpl_(params);

    CrossCoreSetFlag<0x2, PIPE_FIX>(t);
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void MatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::ReduceScatterProcess()
{
  for (int32_t t = 0; t < static_cast<int32_t>(tileCnt_); ++t) {
    CrossCoreWaitFlag<0x2, PIPE_MTE3>(t);

    reduceScatterComm_.BarrierAll();

    uint64_t mOffset = static_cast<uint64_t>(t) * headMSize_;
    ReduceSegmentParallel(mOffset, headMSize_);

    if (t >= static_cast<int32_t>(bufferSize_)) {
      CrossCoreSetFlag<0x2, PIPE_MTE3>(t - static_cast<int32_t>(bufferSize_));
    }
  }
}
```

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke），MC2 多卡通算融合算子的通算流水（mLoop）改造。
- **API 路径**：`CrossCoreSetFlag<0x2, PIPE_FIX/PIPE_MTE3>` / `CrossCoreWaitFlag<0x2, PIPE_FIX/PIPE_MTE3>`（核间 flag 同步）。
- **验证过的平台与版本**：Atlas A2（ascend910b，4 卡），4P M=2048 N=3584 K=5120；a3/950 未验证。
- **适用粒度**：`operator_family`——所有 Pattern B“先计后通”、AIC 写大容量 GM、AIV 读的 MC2 算子共享此反压方向判别。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；去掉反向 flag 的逐字 after 方法与修后死锁消除的完整闭环信号未从 trace 恢复。
- **不适用/未验证场景**：基底工程 all_to_all_matmul（AIV 写 Win Area、AIC 读、需要 ping-pong）原反压方向（AIV 等 AIC 释放）正确，不得照搬本卡“去掉反压”结论；cGm_ 容量不足以容纳全量 tile（需 buffer 复用）时仍需正确方向的 ping-pong 反压，本卡判别方法仍适用但“去反压”修复不适用。

## 判别方法

**触发条件**（tool_observed 失败症状）：通算流水改造后，tileCnt 较小（tileCnt ≤ bufferSize）时精度 PASS、不挂；当 tileCnt 增大到超过 bufferSize 时，`bash run.sh ... precision <tileCnt>` 挂起超时、无返回，bash 进程被 timeout 杀死（本 trace 为 bufferSize=2、tileCnt=4 触发）。

**可执行探针**：对被怀疑的反压 flag 配置，按 tile 索引 `t` 画出 AIC 与 AIV 两侧每个 `CrossCoreWaitFlag(t - buf)` / `CrossCoreSetFlag(t - buf)` 的依赖图，逐 `t` 标出“谁等谁的 SetFlag、那个 SetFlag 在哪个 t 之后才发”。具体步骤：
1. 读 `MatmulProcess`（AIC）与 `ReduceScatterProcess`（AIV）循环体，列出 AIC 每个 `t` 的 `WaitFlag(t-buf)` 与 `SetFlag(t)`、AIV 每个 `t` 的 `WaitFlag(t)` 与 `SetFlag(t-buf)`。
2. 对每个 `t ≥ buf` 的 tile，追依赖链：AIC `WaitFlag(t-buf)` 等 AIV `SetFlag(t-buf)`；AIV `SetFlag(t-buf)` 在 AIV 跑完 `t` 的 reduce 之后才发；AIV 跑 `t` 的 reduce 前要先 `WaitFlag(t)` 等 AIC `SetFlag(t)`；AIC `SetFlag(t)` 在 AIC 跑完 `t` 的 matmul 之后才发；AIC 跑 `t` 的 matmul 前要先过 `WaitFlag(t-buf)`——回到起点。
3. 若依赖图出现“AIC 的某个 `WaitFlag` 与 AIV 的某个 `WaitFlag` 互相在等对方的 `SetFlag`”，即循环依赖，判为反压方向致死锁。
4. 旁证：核对数据流方向（AIC 写 GM 还是 AIV 写 GM）与 GM 容量是否真的需要 buffer 复用（cGm_ 能否一次容纳所有 tile）。若 GM 容量足够且数据流是“AIC 写、AIV 读”，则“AIC 等 AIV 释放”的反压方向与数据流相反，是多余且致死的。

**原始观测**（Atlas A2，4 卡，bufferSize=2）：
- tileCnt=1 / 2 精度 PASS、不挂；tileCnt=4 `bash run.sh 2048 5120 3584 4 precision 4` 挂起超时，bash 在 300s timeout 被杀死、无返回。
- 依赖图（tileCnt=4, bufferSize=2）：AIC t=2 执行 `WaitFlag(0)` 等 AIV `SetFlag(0)`；AIV `SetFlag(0)` 在 AIV t=2 reduce 之后才发；AIV t=2 reduce 前要先 `WaitFlag(2)` 等 AIC `SetFlag(2)`；AIC `SetFlag(2)` 在 AIC t=2 matmul 之后才发；AIC t=2 matmul 前要先过 `WaitFlag(0)`——AIC t=2 ↔ AIV t=2 互相等待，循环依赖。
- 数据流核对：本算子数据流为 AIC 写 cGm_（matmul 结果）、AIV 读 cGm_（reduce scatter），与基底工程 all_to_all_matmul（AIV 写 Win Area、AIC 读）方向相反；且 cGm_ 是全量大 GM（`[M,N]` 一次容纳所有 tile，无 buffer 复用），AIC 不需要等 AIV 释放任何 buffer。

**机制解释**（根因分析，属 interpretation）：反压 flag 方向与数据流方向不匹配。基底工程 all_to_all_matmul 是 AIV 写 Win Area、AIC 读，所以 AIC `WaitCommComplete` 等 AIV 释放正确；本算子数据流反向（AIC 先算 matmul 写 cGm_，AIV 后 reduce scatter 读 cGm_），且 cGm_ 是全量大 GM（一次容纳所有 tile、无 buffer 复用），AIC 不需要等 AIV 释放。照搬基底工程的“AIC 等 AIV 释放”反压，在 `tileCnt > bufferSize` 时必然形成“AIC t ↔ AIV t”循环等待：AIC 的 `WaitFlag(t-buf)` 等 AIV 的 `SetFlag(t-buf)`，而 AIV 的 `SetFlag(t-buf)` 又依赖 AIV 跑到 tile t，AIV 跑 tile t 又要 `WaitFlag(t)` 等 AIC `SetFlag(t)`，AIC `SetFlag(t)` 又要 AIC 先过 `WaitFlag(t-buf)`——闭环。tileCnt ≤ bufferSize 时 `WaitFlag(t-buf)` 不触发（`t ≥ buf` 条件不成立），故不挂，这也解释了为何小 tileCnt 不暴露。

## 验证信号

本卡验证信号来自 trace 中对 tileCnt=4 死锁现象的工具观测与去反压后回归（来源平台：a2）。

- tileCnt=4 改造后 `bash run.sh ... precision 4` 挂起超时、无返回，bash 进程被 300s timeout 杀死（来源平台：a2；证据等级：tool_observed）
- 去掉 AIC `WaitFlag(t-buf)` 与 AIV `SetFlag(t-buf)` 两个反向 flag 后，tileCnt=1/2/4 精度全 PASS、无死锁（max_diff=0.5）（来源平台：a2；证据等级：tool_observed）

## 坏实践

- 从数据流反向的基底工程照搬反压 flag 方向而不核对“谁写谁读、buffer 是否真的被复用（容量是否不足）”。
- 在 cGm_ 足够大时仍加 ping-pong 反压，制造本不需要的循环依赖。
- 用 `bufferSize_ = min(bufferSize_, tileCnt_)` 这种上限钳制来“绕过”死锁——只掩盖 `tileCnt <= bufferSize` 的情形，tileCnt 一旦超过仍死锁。

## 预防

- 加反压 flag 前先判数据流方向：AIC 写还是 AIV 写？被等的是谁释放？buffer 是否真的被复用（容量是否不足）？
- 当 GM 容量足以容纳全量 tile 时，AIC 不需要等 AIV，只保留“AIV 等 AIC 算完该 tile”的单向 flag 即可。
- 死锁排查：画出 `t` 与 `t-bufferSize` 的 flag 依赖图，若出现 AIC t ↔ AIV t 互相等待即循环依赖。

## 失效边界

- 已验证：Atlas A2（ascend910b）4 卡，bufferSize=2、tileCnt=1/2/4，去反压后不挂。
- 未验证，推断可迁移：Ascend950（950）与 A3（a3）的 CrossCore flag 语义应一致，但本 trace 未在这些平台实测。
- 当 cGm_ 容量不足以容纳全量 tile（需 buffer 复用）时，“去掉反压”会写覆盖未读数据，本卡“去反压”修复不适用——此时需正确方向的 ping-pong 反压（参考基底工程 all_to_all_matmul 的 AIV 写/AIC 读方向），但本卡的判别方法（画依赖图找循环等待）仍适用。