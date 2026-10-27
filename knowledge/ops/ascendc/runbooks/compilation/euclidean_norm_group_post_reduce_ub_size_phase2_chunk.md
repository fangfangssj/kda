---
type: "Runbook"
title: "postReduceUbSize 不足致 Phase2 读破坏"
description: "branch-3/4 group 路径 postReduceUbSize 只装每 chunk A 大小，Phase 2 把 aTotal 拆多 chunk 且 srcStride 非零，blockLen 非 32B 对齐时 workspace 读破坏；修复 enlarge postReduceUbSize 至 aTotal*dtypeSize 对齐强制单 chunk。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["EuclideanNorm"]
---

# postReduceUbSize 不足致 Phase2 读破坏

## 现象

`euclidean_norm` 算子 Branch-3 / Branch-4 (group-tail-A / group-tail-R) kernel 上板跑 ACLNN 黑盒（`operators/euclidean_norm_package/tests/aclnn`，`bash run.sh`）：BB15 (fp32, shape `(2,3,2,132800)`, axes=`[-1]`, group-tail-R) `actual=1149.1 expected=257.7`（偏差 4.46×）；int32 group-tail-A BB05/07/10/12 `bitwise mismatch at elem[0]`，diff 13–248；bf16 group BB06 crash。workspace dump 显示 partial sum 正确（host 同步后读 workspace 数据正确），但 kernel 输出错。

## 适用场景

- **工程形态**：ACLNN 算子仓 kernel 直调，group 路径两阶段 reduce（Phase 1 各 core 写 workspace partial sum，Phase 2 跨 core 读 workspace 再 reduce）。
- **API 路径**：`AscendC::DataCopyPad`（MTE2 读 workspace / MTE3 写 workspace，`DataCopyExtParams.srcStride`）；`AscendC::ReduceSum`；host `ComputeBranch3Tiling / ComputeBranch4Tiling` 设 `postReduceUbSize`。
- **验证过的平台与 CANN 版本**：Ascend 950 (dav-3510)，CANN 9.1.0。
- **适用粒度**：`operator_family`（任何 group 路径两阶段 reduce + workspace 中转的 Reduction 算子）。
- **路线优先级**：`recommended_fastpath`（postReduceUbSize 必须能装下 aTotal 全量 A 输出）。
- **明确不适用或未验证场景**：normal 路径（无 group）不经 Phase 2 workspace 中转，不在此列；BB15 Heisenbug（PRINTF 后通过）的部分另有根因。

## 根因

group 路径 Phase 2 从 workspace 读 `[rGroupCnt_][aTotal_]` fp32 partial sum 再 `ReduceSum<RA>` 跨 rGroupCnt_ 维 reduce。`Phase1OutputToWorkspace` 按 `UnravelALoop` 给的 `chunkOutOff` 把每个 A chunk 的 partial sum 写到 `wsGm_[rChunkIdx*aTotal_ + wsColOff]`；`Phase2Process` 用 `DataCopyExtParams` 按 `aUbFactorP2` 分 chunk 读 workspace：

```cpp
ext.blockLen = aUbFactorP2 * static_cast<int64_t>(sizeof(float));
ext.srcStride = static_cast<uint32_t>(aTotal_ * sizeof(float) - ext.blockLen);
```

`aUbFactorP2` 由 Phase 2 反推：`aUbFactorP2 = min(postReduceUbSize / dtypeSize, aTotal_)`。当 `postReduceUbSize < aTotal_*dtypeSize` 时 `aUbFactorP2 < aTotal_`，Phase 2 把 A 拆成多个 chunk，每个 chunk `srcStride = aTotal_*4 - blockLen > 0`（workspace 行间距）。

问题：当 `blockLen`（= `aUbFactorP2*4`）非 32B 对齐（如 `aUbFactorP2` 非 8 倍数）时，`DataCopyPad` 的 `srcStride` 单位语义（3510 上为 bytes，但 Phase1 写侧用 `/kBlockBytes` 32B 块单位）两侧不一致 + 非 32B 对齐 chunk 跨 stride 读，破坏 workspace 读取，partial sum 错位累加。

## 解决配方

修复 `operators/euclidean_norm_package/op_host/arch35/euclidean_norm_tiling_common.cpp` 的 `ComputeBranch3Tiling` 与 `ComputeBranch4Tiling`，在 `ctx.isGroup = true` 之后 enlarge `postReduceUbSize` 至能装下全量 A 输出，强制 `aUbFactorP2 = aTotal_`（单 chunk，`srcStride = 0`）。

**修复前**（branch-3 末尾，branch-4 同构）：

```cpp
    ctx.rGroupCnt = (ctx.aLoopCntTotal > 0) ? (numBlocks / ctx.aLoopCntTotal) : 0;
    ctx.isGroup = true;

    return ge::GRAPH_SUCCESS;
}
```

**修复后**（commit ff4dbf4f）：

```cpp
    ctx.rGroupCnt = (ctx.aLoopCntTotal > 0) ? (numBlocks / ctx.aLoopCntTotal) : 0;
    ctx.isGroup = true;

    // Phase 2 fix: postReduceUbSize must hold all A output elements at once.
    //   Phase 2 reverse-derives aUbFactorP2 from preReduceUbSize / rGroupCnt,
    //   clamped by postReduceUbSize / dtypeSize and aTotal_.  When postReduceUbSize
    //   < aTotal_ * dtypeSize, Phase 2 splits A into multiple chunks with non-zero
    //   workspace srcStride, which can corrupt reads when blockLen is not 32B-aligned.
    //   Fix: enlarge postReduceUbSize to hold aTotal_ elements -> single Phase 2 chunk.
    if (ctx.isGroup) {
        int64_t aTotalAll = 1;
        for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
            if (!ctx.isReduceAxis[i]) aTotalAll *= ctx.axisShape[i];
        }
        int64_t outNeeded = (aTotalAll * ctx.dtypeSize + ctx.blockSize - 1) / ctx.blockSize * ctx.blockSize;
        ctx.postReduceUbSize = std::max(ctx.postReduceUbSize, outNeeded);
    }

    return ge::GRAPH_SUCCESS;
}
```

branch-4 用 `CeilAlign(aTotal*dtypeSize, blockSize)` 同语义。强制 `postReduceUbSize >= CeilAlign(aTotal_*dtypeSize, blockSize)` → `aUbFactorP2 = aTotal_` → `srcStride = aTotal_*4 - aTotal_*4 = 0`，Phase 2 单 chunk 连续读 workspace，消除 stride 错位。

## 验证信号

本卡验证来源为 dav-3510/CANN 9.1.0 的黑盒跑与 git diff ff4dbf4f。

- BB15 group-tail-R fp32 修复前 `actual=1149 vs expected=257`，修复后 partial sum 读取对齐（来源平台：950；证据等级: tool_observed）
- int32 group-tail-A bitwise mismatch diff 由 13–248 收敛（来源平台：950；证据等级: tool_observed）
- git commit ff4dbf4f `feat(kernel-developer): 黑盒失败用例修复` 含 branch-3/4 两处同构 fix（来源平台：950；证据等级: artifact_verified）

## 设计决策

- 选 enlarge `postReduceUbSize` 而非修 `srcStride` 单位：`postReduceUbSize` 是 host tiling 量，单点修改即强制 `srcStride=0`，避免在 kernel 侧动 `DataCopyExtParams` 单位语义（3510 上 bytes vs 32B 块的歧义是更深问题）。
- 放弃的候选：在 Phase 2 用 `CeilAlign(aUbFactorP2, 8)` 强制 32B 对齐 chunk——会丢尾块 A 元素，需加 tail 处理，复杂度高，未被采用。
- 当前方案属 `recommended_fastpath`：单 chunk 连续读，零 stride。

## 预防

- group 路径两阶段 reduce 算子，host tiling 必须保证 `postReduceUbSize >= CeilAlign(aTotal * dtypeSize, blockSize)`，让 Phase 2 `srcStride=0`。
- Tiling UT 加用例：`aTotal * dtypeSize` 非 32B 对齐时断言 `postReduceUbSize` 已 enlarge 且 `aUbFactorP2 == aTotal`。

## 失效边界

- 推断可迁移：a2 / a3 平台 group 路径同构 `DataCopyPad` + `ReduceSum`——未验证，推断可迁移。
- 未验证：`srcStride` 在 3510 上的单位（bytes vs 32B 块）歧义未在本卡闭环，仅用 `srcStride=0` 绕开；若未来需多 chunk 读，必须先确认单位语义。