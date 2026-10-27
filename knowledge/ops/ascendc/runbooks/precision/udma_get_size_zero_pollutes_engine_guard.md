---
type: "Runbook"
title: "UDMA GET size=0 污染后续 GET 致精度失败"
description: "matmul_reducescatter 通算流水 tileCnt=2 时 Rank 2/3 因 hasOverlap=false 致 tileSegByteSize=0，对零大小 UDMA GET 引擎异常污染后续 GET，98.9% 元素错。判别要点是 GET 前缺 size>0 守卫。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.implementation.synchronization", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# UDMA GET size=0 污染后续 GET 致精度失败

## 现象

在 matmul_reducescatter（MC2 通算融合，4P SHMEM ReduceScatter，Pattern B）实现通算流水后跑精度回归：M=2048 N=3584 K=5120、4 卡、tileCnt=2。Rank 0/1 精度 PASS，**Rank 2/3 FAIL**，约 98.9% 元素错误（1815424/1835008），max_diff=205。

场景还原：4 卡 rankSize=4，M=2048，每卡 segRows=M/rankSize=512。每个 rank 的 segment 是 `[rankId*512, rankId*512+512)`：
- Rank 0: [0, 512)
- Rank 1: [512, 1024)
- Rank 2: [1024, 1536)
- Rank 3: [1536, 2048)

tileCnt=2 时 headMSize=1024，Tile 0 覆盖 mOffset=0（M 行 [0,1024)），Tile 1 覆盖 mOffset=1024（M 行 [1024,2048)）。Rank 2/3 的 segment [1024,2048) 只在 **Tile 1** 有 overlap，**Tile 0 hasOverlap=false**（overlapStart=overlapEnd，overlapRows=0）。

`ReduceSegmentParallel` 的 Phase 1（UDMA GET）中，`tileSegByteSize = tileOverlapRows * axisN_ * sizeof(bfloat16_t)`；当 hasOverlap=false 时 tileSegByteSize=0。代码未对此守卫，直接对 `UdmaGetNbi(dstStaging, srcSymGm, 0, r)` 下发零大小 GET，UDMA 引擎对 size=0 的请求行为异常，污染了同一 AIV 后续对其他 PE 的 GET，导致 Tile 1（Rank 2/3 真正有 overlap 的 tile）的 reduce 数据错乱。

## 判别方法

### 触发

UDMA GET（`aclshmemx_udma_get_nbi`）的 `size` 参数为 0（`tileSegByteSize=0`，由 `hasOverlap=false` 引起）→ 引擎异常污染同一 AIV 后续 GET → 下游 reduce 数据错乱、特定 Rank 精度失败。触发信号：tileCnt>1 且失败集中在某些 Rank，与 overlap 分布吻合。

### 探针

1. 枚举所有 (rank, tile) 组合的 `hasOverlap` 状态：若某些 rank-tile 组合 `hasOverlap=false` → `tileSegByteSize = tileOverlapRows * axisN_ * sizeof(T) = 0`。
2. 检查 `ReduceSegmentParallel` Phase 1 的 `UdmaGetNbi(...)` 调用前是否有 `if (tileSegByteSize > 0)` 守卫——无守卫即确诊。
3. 精度失败按 Rank 分组看：若失败集中在特定 Rank 且其 segment 仅在部分 tile 有 overlap，优先查通信层零大小搬运。

before（Phase 1 UDMA GET，无 size 守卫，verbatim 取自改造指令中实际应用的 Phase 1 片段）：

```cpp
  uint32_t remoteIdx = 0;
    for (uint32_t r = 0; r < rankSize_; r++) {
      if (r == rankId_) { continue; }
      if (globalAivIdx == remoteIdx) {
        GM_ADDR dstStaging = stagingGm_ + stagingBase + static_cast<uint64_t>(r) * segRows_ * axisN_ * sizeof(bfloat16_t);
        GM_ADDR srcSymGm = cGm_ + tileRemoteOffsetBytes;
        reduceScatterComm_.UdmaGetNbi(dstStaging, srcSymGm,
          static_cast<uint32_t>(tileSegByteSize), static_cast<int32_t>(r));
        reduceScatterComm_.UdmaQuiet(static_cast<int32_t>(r));
      }
      remoteIdx++;
    }
```

### 观测

- tileCnt=2 Rank 2/3 FAIL，约 98.9% 元素错（1815424/1835008），max_diff=205。
- Rank 2/3 失败集中在 Tile 1（其 segment 仅在 Tile 1 有 overlap），Tile 0 hasOverlap=false → tileSegByteSize=0 是触发点。

### 判别

UDMA GET 对 size=0 的请求未在引擎层做无操作（no-op）短路，触发异常并污染同 AIV 后续 GET。`hasOverlap=false` 是 ReduceScatter 按 tile 分段的合法状态（某些 rank-tile 组合天然无 overlap），调用方必须在 GET 前用 `if (tileSegByteSize > 0)` 守卫，不能假设引擎对零大小静默——不加守卫则进 banned。

## 适用场景

- 工程形态：kernel 直调（ops-direct-invoke），MC2 多卡 ReduceScatter 通信层使用 UDMA GET 的 `ReduceSegmentParallel`。
- API 路径：`aclshmemx_udma_get_nbi`（SHMEM UDMA GET，size 参数为字节计数当 T=uint8_t）。
- 验证平台与版本：Atlas A2（ascend910b，4 卡），4P M=2048 N=3584 K=5120。
- 适用粒度：`operator_family`（所有按 tile 分段、某些 rank-tile 组合 hasOverlap=false 的 ReduceScatter/AllReduce 通信算子共享此边界）。
- 路线优先级：`recommended_fastpath`（size>0 守卫是 UDMA GET 的正确写法；不加守卫则进 banned）。
- 不适用或未验证场景：`DataCopy` 对 size=0 的行为未在本 trace 核验（可能不同于 UDMA GET）；其它 SHMEM 版本 UDMA 引擎对 size=0 的行为未验证。

## 验证信号

- tileCnt=2 Rank 2/3 FAIL，约 98.9% 元素错（1815424/1835008），max_diff=205（来源平台：a2；证据等级：tool_observed）
- Rank 2/3 失败集中在 Tile 1，Tile 0 hasOverlap=false → tileSegByteSize=0 是触发点（来源平台：a2；证据等级：tool_observed）

## 坏实践

- 假设 UDMA/DMA 引擎对 size=0 静默无操作而不加守卫。
- 用"调整 tileCnt/segRows 让所有 rank-tile 都 hasOverlap"来绕过——只是掩盖，换一组 M/rankSize/tileCnt 仍会触发。
- 把失败标注为"环境问题/偶发"放行通过（反投机取巧：只改测试不改实现）。

## 预防

- 任何 UDMA GET / DMA 搬运调用，调用前用 `if (byteSize > 0)` 守卫零大小请求；这是通信层的通用防御。
- ReduceScatter 按 tile 分段时，先枚举所有 (rank, tile) 组合的 hasOverlap 状态，确认零大小 GET 被守卫。
- 精度失败按 Rank 分组看：若失败集中在特定 rank 且与 overlap 分布吻合，优先查通信层零大小搬运。

## 失效边界

- 已验证：Atlas A2（ascend910b）4 卡，tileCnt=1/2 加守卫后精度 PASS。
- 未验证，推断可迁移：Ascend950（950）与 A3（a3）的 UDMA 引擎对 size=0 行为应一致（同源设备 API），但本 trace 未实测，推断可迁移。
- 本卡只解决 size=0 GET 污染；tileCnt>1 即使精度通过，性能仍反降（见同 trace 的 banned 路线卡）。