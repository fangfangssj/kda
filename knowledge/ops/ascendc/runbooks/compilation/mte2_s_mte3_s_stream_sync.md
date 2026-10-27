---
type: "Runbook"
title: "探针末尾 MTE2_S/MTE3_S 同步不可由 stream sync 替代"
description: "DMA 带宽探针 kernel 末尾须保留 PipeBarrier<PIPE_MTE2/MTE3> + MTE2_S/MTE3_S SetFlag/WaitFlag，确保 MTE 管线 DataCopy 完成后 Scalar 才退出；aclrtSynchronizeStream 不能替代该 in-kernel 同步，去掉后小数据量带宽虚高约 1%。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-15T10:30:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 探针末尾 MTE2_S/MTE3_S 同步不可由 stream sync 替代

## 适用场景

- 工程形态：Kernel 直调（ops-direct-invoke），Python/aclrt 调 `.so` kernel，用 NPU event 或 msprof/TTK CST_PERF 计时。
- API 路径：`PipeBarrier<PIPE_MTE2|PIPE_MTE3>`；`GetTPipePtr()->FetchEventID(HardEvent::MTE2_S|MTE3_S)`；`SetFlag`/`WaitFlag`；Host 侧 `aclrtSynchronizeStream`。
- 验证过的平台与 CANN 版本：Ascend950PR；CANN 9.1.0。
- 适用粒度：operator_family（所有 DMA 带宽探针 / 需精确计量 MTE2/MTE3 搬运完成时机的 kernel）。
- 路线优先级：recommended_fastpath（末尾保留 `PipeBarrier<PIPE_MTE*>` + `MTE2_S/MTE3_S` SetFlag/WaitFlag 作为正确性同步；不要为压开销去掉它）。
- 不适用或未验证场景：aclnn op_host 流程下 stream sync 的语义未在本卡验证；仅覆盖 Kernel 直调带 NPU event/引擎级计时的带宽探针。

## 判别方法

触发（tool_observed）：在 GM→UB 纯读带宽探针 kernel（`DataCopy` 循环 + 末尾同步）上，为压同步开销去掉末尾 `PipeBarrier<PIPE_MTE2>` + `MTE2_S` 的 `SetFlag`/`WaitFlag`，改由 Host 侧 `aclrtSynchronizeStream` 兜底，用 NPU event best-of-5 计时后，小数据量带宽相对偏高约 1%。

探针：对同一 kernel 做"保留 vs 去掉末尾 `MTE2_S` 同步"A/B（小数据量 112MB + 大数据量 1.75GB，best-of-5 NPU event），恢复同步后另用 best-of-50 NPU event 与 TTK `CST_PERF` 交叉校验计时口径。

原始观测（verbatim 同步块，保留态，取自 `test/read_only/probe_read.asc`）：

```
    PipeBarrier<PIPE_MTE2>();
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}
```

去掉态（verbatim，edit 的 OLD→NEW，删去末尾同步块只剩 `}`）：

OLD:
```
    }

    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}
```

NEW:
```
    }
}
```

写侧对称同步（verbatim，取自 `test/write_only/probe_write.asc`）：

```
    PipeBarrier<PIPE_MTE3>();
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId);
    WaitFlag<HardEvent::MTE3_S>(eventId);
}
```

观测：去掉同步后小数据量 best-of-5 NPU event 带宽相对约 +1.1%（参考路线：保留末尾 MTE2_S/MTE3_S 同步），大数据量基本不变；恢复同步后 best-of-50 NPU event 与 TTK `CST_PERF` 仅差约 0.7%。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：`aclrtSynchronizeStream` 等待 stream 上所有任务完成，但 kernel 内 MTE2/MTE3 管线的 `DataCopy` 是异步发出的——Scalar 发出 DataCopy 描述符后即可退出 kernel。末尾若无 `PipeBarrier<PIPE_MTE*>` + `MTE2_S/MTE3_S` 的 `SetFlag`/`WaitFlag`，Scalar 可能在 MTE 管线 `DataCopy` 还没搬完时退出 kernel，使 stream sync 返回时部分搬运未完成，小数据量下带宽测得值虚高（相对差异约 1%）。该机制解释属 agent 推断，未单独验证"搬运未完成"的确定性数据证据。结论：保留末尾 `PipeBarrier<PIPE_MTE*>` + `MTE2_S/MTE3_S` 同步，改用引擎级计时采集纯 MTE 时间以获得精确带宽。

## 验证信号

来源：本 trace 在 Ascend950PR（CANN 9.1.0）上对 `test/read_only/probe_read.asc` 做"保留 vs 去掉末尾 `MTE2_S` 同步"A/B，用 NPU event best-of-5 计时；恢复同步后用 best-of-50 NPU event 与 TTK `CST_PERF` 交叉校验。

- 去掉末尾 `MTE2_S` 同步后，小数据量(112MB) best-of-5 NPU event 带宽相对约 +1.1%（参考路线：保留末尾 MTE2_S/MTE3_S 同步）；大数据量(1.75GB) 基本不变。（来源平台：950；证据等级：tool_observed）
- 恢复同步后 best-of-50 NPU event 与 TTK `CST_PERF` 仅差约 0.7%（参考路线：保留末尾 MTE2_S/MTE3_S 同步）。（来源平台：950；证据等级：tool_observed）

## 失效边界

- Ascend950PR / CANN 9.1.0 实测；仅覆盖 Kernel 直调带 NPU event/引擎级计时的带宽探针。
- aclnn op_host 框架下 `aclrtSynchronizeStream` 的语义未验证，推断可迁移（保留末尾同步的推荐路线应跨流程有效，但未在 op_host 流程实测）。
- "去掉同步导致搬运未完成"目前仅有带宽相对差异的间接证据，缺确定性数据校验。

## 坏实践

- 为压同步开销直接删末尾 `MTE2_S`/`MTE3_S` 的 `SetFlag`/`WaitFlag`，靠 `aclrtSynchronizeStream` 兜底——会测到未完成搬运，带宽虚高。该路线标记为 `banned`。