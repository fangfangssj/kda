---
type: "Runbook"
title: "tiling扩A超UB致507035：UT小核数掩盖"
description: ExpandAIfRFullyLoaded 在大 coreNum 下按单槽 residual 高估预算过度扩展 aUbFactor，4 个 tile 槽超出 UB 触发 507035；UT 用 coreNum=8 不触发该分支故全绿，须在 glue 加 UB 安全帽。
tags: [task.debugging, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# tiling扩A超UB致507035：UT小核数掩盖

## 现象

在 ascend950（arch35/dav-3510，CANN 9.2.0，48 AIV 核，UB 253952 B）softmax 类算子分支 kernel 的验收中：小 shape 用例多数 PASS，但两个中大型 shape 的 **fp32** 用例稳定报 `[ACLNN_FAIL]` + `ACL error 507035 at aclrtSynchronizeStream`——attention `[2,8,64,512]`（totalA=1024, R=512）与 classification `[32,1000]`（totalA=32, R=1000），而同 shape 的 fp16/bf16 只报精度失败不崩溃。tiling 逻辑：R 能整装进 UB 时（`rUbFactor == axisShapeR`），调用 `ExpandAIfRFullyLoaded` 把剩余 UB 用于扩展 A 方向的 `aUbFactor`；kernel 侧按 4 个 tile 槽（2×preInBuf 双缓冲位 + 2×tmpBuf）+ cacheBuf(16384 B) + postReduce 分配 UB。单元测试（tiling UT，51 例，**coreNum=8**）全部通过。手工推演真实硬件参数：`[2,8,64,512]` fp32 在 48 核下 `atmp=min(128, ceil(1024/48)=22)=22 → aUbFactorAlign=24`，`rUbFactor=512`（R 整装），随后 ExpandA 把 `aUbFactor` 扩到 111——`4 × 111对齐后 × 512 × 4B` 远超 253952 B UB，kernel 上板即崩。

## 适用场景

- 工程形态：算子仓式自定义算子的 **op_host tiling** 开发与上板 ST；tiling 公式与 kernel UB 分配的联合预算。
- API 路径：`op_host/arch35/*_tiling_arch35.cpp`（TilingFunc glue）与 `op_host/*_tiling_host.cpp`（ComputeXxxTiling 共享实现）；`ExpandAIfRFullyLoaded` 共享 helper；kernel 侧 `TPipe` UB 分配。
- 验证过的平台与 CANN 版本：Ascend 950（48 核，UB 253952 B）+ CANN 9.2.0。与 frontmatter `platforms` 一致。
- 适用粒度：operator_family（任何"R 整装后用残余 UB 扩 A"的 tiling 模式）。
- 路线优先级：recommended_fastpath（UB 预算安全帽是 tiling 正确性的必要组成）。
- 明确不适用/未验证场景：R 不整装（触发 group/三遍路径）时 ExpandA 不触发，本卡不适用；不同 UB 大小的平台未验证。

## 根因/解决配方

根因：`ExpandAIfRFullyLoaded` 的 `ubResidual` 按单槽口径估算（`ubSize - CACHE_BUF_SIZE - postReduceUbSize - rUbFactor * sizeof(float) * 4` 里对 4 槽的扣减不足），在 **coreNum 大**（真实 48 核）时 `atmp` 变小 → `rUbFactor` 变大 → residual 被高估 → `aUbFactor` 过度扩展；而 kernel 实际按 **4 个 tile 槽**（2 preInBuf + 2 tmpBuf）分配，`4 × aUbFactorAlign × rUbFactorAlign × sizeof(float)` 超出 253952 B UB → 上板 507035。UT 不暴露该 bug 是因为 UT 用 coreNum=8：`[2,8,64,512]` 在 8 核下 `atmp=128 → rUbFactor=115 < 512`，R 不整装、ExpandA 根本不触发——**同一公式在 UT 参数域内正确、在真实硬件参数域内越界**。

解决配方：在 ExpandA 之后追加 UB 预算安全帽（不改 ExpandAIfRFullyLoaded 公式本身、不改 UT oracle，避免"改用例"），clamp 到 4 槽预算内并按 dataBlockElems 向下取整（verbatim，glue 与共享实现两处同步加）：

```cpp
// Step 3: if R fully loaded, spend residual UB on expanding A.
int64_t axisShapeR = ps.axisShape[rSplitAxisIdx];
int64_t ubResidual = static_cast<int64_t>(ubSize) - CACHE_BUF_SIZE
                     - postReduceUbSize
                     - rUbFactor * sizeof(float) * 4;  // 4 slots × rUbFactor rows
ExpandAIfRFullyLoaded(rUbFactor, axisShapeR, ubResidual, sizeofDT, totalA,
                      aUbFactor, aUbFactorAlign);

// UB budget safety cap: ensure 4 tile slots (2 preInBuf + 2 tmpBuf) +
// cacheBuf + postReduce fit in UB.  ExpandAIfRFullyLoaded may over-expand
// when coreNum is large (small atmp → large rUbFactor → overestimated
// residual).  Cap aUbFactorAlign so the actual allocation stays in-budget.
{
    int64_t perSlotBytes = aUbFactorAlign * rUbFactorAlign * static_cast<int64_t>(sizeof(float));
    int64_t budgetForTiles = static_cast<int64_t>(ubSize) - CACHE_BUF_SIZE - postReduceUbSize;
    if (perSlotBytes > 0 && 4 * perSlotBytes > budgetForTiles) {
        int64_t maxAUbAlign = (rUbFactorAlign > 0)
            ? (budgetForTiles / (4 * rUbFactorAlign * static_cast<int64_t>(sizeof(float))))
            : aUbFactorAlign;
        maxAUbAlign = (maxAUbAlign / dataBlockElems) * dataBlockElems;  // floor-align
        if (maxAUbAlign < dataBlockElems) maxAUbAlign = dataBlockElems;
        if (aUbFactorAlign > maxAUbAlign) {
            aUbFactorAlign = maxAUbAlign;
            if (aUbFactor > aUbFactorAlign) aUbFactor = aUbFactorAlign;
        }
    }
}
```

判别方法：fp32 大 shape 崩溃（507035）而 fp16/bf16 同 shape 不崩（tile 减半）+ tiling UT 全绿 + 手工用**真实 coreNum**（48）重推 aUbFactor/rUbFactor 并对 4 槽算 UB 账——三项合起来即指向 tiling 越界而非 kernel 逻辑。

## 设计决策

- 为何在 glue 加 clamp 而不是修 `ExpandAIfRFullyLoaded` 公式：UT oracle 与该 helper 共用同一公式，改公式必须同步改 UT 期望值，逼近"改用例"红线；在调用方（arch35 glue 与 ComputeBranch0Tiling 两处）加安全帽保持 helper 与 UT 不动，语义上"扩张上限"与"扩张公式"也确实是两层职责。
- 为何保守用扩张前的 postReduceUbSize：它随 aUbFactorAlign 增大会变大，用旧值略保守（多扣一点），只会让 cap 更紧，安全。
- 放弃的候选：直接改 ExpandA 公式 + UT oracle——被否（改用例风险）；kernel 侧减缓冲槽（2 preInBuf → 1）——改动面大且牺牲流水。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.2.0；算子仓 `operators/build.sh`；tiling UT（`tests/ut/test_branch0_tiling.cpp` 等，coreNum=8、ubSize=253952）。
- 目标平台：ascend950（48 AIV 核，UB 253952 B，`CACHE_BUF_SIZE=16384`）。
- 相关文件：`op_host/arch35/bench_harness_softmax_tiling_arch35.cpp`、`op_host/bench_harness_softmax_tiling_host.cpp`（两处同步加 cap）；`op_kernel/arch35/bench_harness_softmax_kernel.h`（4 槽 UB 分配）。

## 验证信号

问题与验证来自同一次验收-修复循环（改 tiling 后重编译并全量重跑 run.sh）：

- 修复前：`[2,8,64,512]` 与 `[32,1000]` 的 fp32 用例 `ACLNN_FAIL`（507035），验收判定 FAILED（来源平台：950；证据等级：tool_observed）。
- 参数推演：48 核下 ExpandA 将 aUbFactor 从 22/24 域扩至 111（`[2,8,64,512]` fp32），4 槽超 UB（来源平台：950；证据等级：artifact_verified——推演基于 tiling 源码与硬件参数，非上板直测）。
- 修复后：run.sh 中 507035 全部消失（0 ACLNN_FAIL，46 aclnn-OK），两个崩溃 shape 的 fp32 转 PASS（来源平台：950；证据等级：tool_observed）。
- UT 回归：tiling UT 51/51 不退化（cap 不改变 UT 参数域内的输出）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 安全帽按"4 槽 + cacheBuf + postReduce"的口径 clamp；若 kernel 改缓冲布局（槽数或 cacheBuf 大小），cap 公式须同步改。
- `[1,131072]`/`[1,2097152]` 等 R 超长用例走 group/三遍路径（ExpandA 不触发），其精度失败与本卡无关。
- postReduce 用扩张前值的保守近似在极端 shape 下可能略紧（多 clamp 一点），未观察到副作用。

**未决项**：

- `ExpandAIfRFullyLoaded` 公式本身的单槽 residual 口径未修正（只在调用方 cap）；后续如改公式须同步 UT oracle 并加 coreNum=48 的用例。
- UT 未补真实 coreNum 的回归用例（避免再犯"参数域外越界"）。
## 坏实践

- tiling UT 只用一个 coreNum（8）覆盖——coreNum 是该公式的敏感分支参数（atmp=min(…, ceil(totalA/coreNum))），单值覆盖等于不覆盖真实硬件域。
- UB 账本只算单槽、不算 kernel 实际分配的全部槽位。
- 见"UT 全绿 + ST 崩溃"就先怀疑测试环境——本例是公式在 UT 参数域外越界的经典形态。

## 误诊

- 曾按"group 切分 SyncAll 核间同步"方向排查（同为 507035 家族现象），并先修了 ReduceMax 隐式 tmpBuffer（该修复独立有效但未消掉这两例，见库内 `reduce_work_buf_ub_507035` 的 复现增量）——最终由参数推演定位到 tiling 越界。
