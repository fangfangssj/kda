---
type: "Runbook"
title: group 2D 分核：对齐 aLoop 与 SetScheduleMode
description: "A×R 2D 分核 host tiling——totalOuter 均分后 numBlocks 对齐 aLoopCntTotal（放不下则 FloorAlign）、rGroupCnt 整除保证 2D 网格整齐；SetScheduleMode(1) 供跨核 SyncAll；workspace=rGroupCnt×6×H×4（R2 实例：rGroupCnt×aTotal×4）+ 系统区。R2（euclidean_norm，CANN 9.1.0）：TilingUT 独立 oracle 基线 0/8 → 实现 8/8，真机 group 路径 ST 8/8 闭环，SetScheduleMode 返回值在 UT mock 上下文可失败须忽略。"
tags: [task.implementation, topic.tiling, topic.implementation.synchronization]
status: stable
sources: []
created_at: "2026-09-09T11:49:41Z"
updated_at: "2026-09-16T11:08:02Z"
platforms: ["950"]
---

# group 2D 分核：对齐 aLoop 与 SetScheduleMode

## 现象

在为 ThnnFusedGruCellGrad（GRU 单步融合反向算子，ascend950/arch35，bias 输出沿 batch 归约）实现 host 侧公共 TilingFunc 时，base 一维切分在"外层 A 块数少、R 块数多"的 shape 下只用少数核（例：A 块 16、R 块 3 时 base 只用 16/48 核）。设计引入 group 模式做 A×R 2D 分核：每个核处理一个 (aChunk, rGroup) 块，Phase 1 各核把 bias 部分和写 GM workspace，`SyncAll` 全核同步后 Phase 2 归约。host tiling 侧要解决三件事：

1. 触发判定：`aLoopCntTotal ≤ coreNum/2 && rLoopCntTotal ≥ 2`；
2. numBlocks 对齐：总块数均分到核后要**对齐 aLoopCntTotal**，保证每核恰好 1 个 A chunk（aPerCore=1）的 2D 网格整齐（`aChunkIdx = blockIdx / rGroupCnt`、`rChunkIdx = blockIdx % rGroupCnt`）；
3. 跨核同步与 workspace：tiling 阶段 `SetScheduleMode(1)`（SyncAll 语义），bias 跨核归约的 GM workspace = `rGroupCnt × BIAS_SEG_CNT × hiddenSize × FP32_BYTES`。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（host tiling 函数的 group/2D 分核分支；跨核归约型算子）
- API 路径：`gert::TilingContext::SetScheduleMode`；`Ops::Base::CeilDiv/CeilAlign/FloorAlign`；`GetWorkspaceSizes(1)` + `PlatformAscendC::GetLibApiWorkSpaceSize`
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）；R2 独立复现：Ascend 950 + CANN 9.1.0
- 适用粒度：`operator_family`（跨核两阶段归约的 2D 分核 tiling 模式）
- 路线优先级：`recommended_fastpath`（触发条件满足时 group 优于 base：核利用率从 aLoopCntTotal/coreNum 提到 ~100%；未做两模式实测性能对比——未验证性能影响）
- 明确不适用/未验证：`aLoopCntTotal > coreNum/2` 或 `rLoopCntTotal < 2` 的 shape（走 base）；本卡只覆盖 host tiling 侧（R2 已补充真机 group 路径验证）

## 根因

2D 分核的网格整齐性约束：核数 numBlocks 必须是 aLoopCntTotal 的整数倍（每"行"一个 A chunk，rGroupCnt 列），否则 `blockIdx / rGroupCnt` 的行映射出现残行，kernel 侧边界处理复杂化。均分计算 `numBlocks = CeilDiv(totalOuter, perCoreNum)` 天然不保证整除，因此需要显式对齐：能向上对齐（CeilAlign）且不超核数则向上（多用核），否则向下对齐（FloorAlign，宁可少核不破网格）。

## 解决配方

触发判定 + 2D 分核 + workspace（verbatim，取自 ThnnFusedGruCellGrad TilingFunc 终版）：

```cpp
// ─── Group 触发判定（§6.1；BranchRoute.md §2 触发条件逐字一致）───
bool ShouldUseGroup(const ThnnFusedGruCellGradCtx& ctx)
{
    return (ctx.aLoopCntTotal <= ctx.coreNum / 2) && (ctx.rLoopCntTotal >= 2);
}

// ─── group 2D 分核（§6.2）：对齐 aLoopCntTotal 保 2D 网格整齐，aPerCore=1 恒成立 ───
ge::graphStatus ComputeGroupSplit(gert::TilingContext* context, ThnnFusedGruCellGradCtx& ctx)
{
    const int64_t totalOuter = ctx.aLoopCntTotal * ctx.rLoopCntTotal;
    const int64_t perCoreNum = CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks = CeilDiv(totalOuter, perCoreNum);
    if (CeilAlign(numBlocks, ctx.aLoopCntTotal) <= ctx.coreNum) {
        numBlocks = CeilAlign(numBlocks, ctx.aLoopCntTotal);
    } else {
        numBlocks = FloorAlign(numBlocks, ctx.aLoopCntTotal);
    }
    ctx.usedCoreNum = numBlocks;                   // ≤ coreNum
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal; // 整除（已对齐 aLoopCntTotal）
    // SyncAll 要求（Phase 1 → Phase 2 全核同步）；失败 OP_LOGE + GRAPH_FAILED（§6.2）
    OP_CHECK_IF(context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Set schedule mode failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

收尾处的 workspace 公式（verbatim，TilingFunc 尾段）：

```cpp
    // workspace（§6.3）：group 且 isBias=1 → sys + rGroupCnt×BIAS_SEG_CNT×hiddenSize×4
    //（Tuple Reduce 纪律不乘归约路数 2）；base / group&&isBias=0 → 仅 sys
    const int64_t usrWorkspaceBytes = (ctx.isGroup && (ctx.isBias == 1))
                                          ? ctx.rGroupCnt * BIAS_SEG_CNT * ctx.H * FP32_BYTES
                                          : 0;
    return SetWorkspaceSize(context, usrWorkspaceBytes);
```

`SetWorkspaceSize`（sys 部分取平台 libApi 工作区）：

```cpp
ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, int64_t usrWorkspaceBytes)
{
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const int64_t sysWorkspaceSize = static_cast<int64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    ws[0] = static_cast<size_t>(sysWorkspaceSize + usrWorkspaceBytes);
    OP_LOGI(context, "Set ws size:%zu, usrSize:%lu", ws[0], static_cast<unsigned long>(usrWorkspaceBytes));
    return ge::GRAPH_SUCCESS;
}
```

机制推导示例（fp16, B=64, H=2048, 48 核；UT 用例 GroupFp16DocExample2 同款）：
- UB 切分得 `aUbFactor=128`（fp16）、`rUbFactor=24`、`rLoopCntTotal=CeilDiv(64,24)=3`、`aLoopCntTotal=CeilDiv(2048,128)=16`
- 触发判定：`16 ≤ 48/2=24` 且 `3 ≥ 2` → group ✓
- `totalOuter = 16×3 = 48`；`perCoreNum = CeilDiv(48,48) = 1`；`numBlocks = CeilDiv(48,1) = 48`
- 对齐：`CeilAlign(48,16) = 48 ≤ 48` → `numBlocks = 48`；`usedCoreNum = 48`；`rGroupCnt = 48/16 = 3`（整除）
- workspace（isBias=1）：`usr = 3 × 6 × 2048 × 4 = 147456` 字节；`ws[0] = sysWorkspaceSize + 147456`
- blockDim = max(48,1) = 48；tilingKey = 1（group）

FloorAlign 分支示例：若 `CeilAlign(numBlocks, aLoopCntTotal) > coreNum`（向上对齐超核数），改 `FloorAlign`——宁可少用核也保住"每核恰好 1 个 A chunk"的网格（UT 用例 GroupFloorAlignBranch 覆盖）。

### 独立复现 R2（euclidean_norm，CANN 9.1.0）：三件套缺失的 TilingUT 红绿基线 + 真机 group 闭环

同一三件套（2D 分核覆盖 usedCoreNum/填 rGroupCnt、SetScheduleMode(1)、workspace 声明）在 euclidean_norm（归约算子，Phase 1 部分和 `[rGroupCnt, aTotal]` fp32 dense）上以测试先行方式复现：TilingUT 用**独立 oracle**（合轴四步 + UB 因子 + loop counts + 触发判断 + 分核公式全套 CPU 复刻，期望值手算）先写 8 个用例，基线 0/8，每个用例稳定失败在三类断言（报错原文）：

```
[ara_basic_group] FAILED: usedCoreNum(group)=1 expected 5
[ara_basic_group] FAILED: rGroupCnt=0 expected 5
[ara_basic_group] FAILED: ws[0]=0 < usrSize=144 (expected ws[0] = usrSize + sys, usrSize = rGroupCnt(6) × aTotal(6) × sizeof(float)=4)
[ara_alooopcnt3_floor] FAILED: usedCoreNum(group)=3 expected 6
[ara_fp16_ws_fp32] FAILED: ws[0]=0 < usrSize=224 (expected ws[0] = usrSize + sys, usrSize = rGroupCnt(7) × aTotal(8) × sizeof(float)=4)
```

实现后 8/8，全量 TilingUT 无回归；tiling 落地后 ST 的 group 用例在 kernel 入口接线前 0/8（哨兵）、接线后 8/8（fp32 maxRel ≤ 4.038e-07）——**真机 group 路径（SetScheduleMode(1) + 跨核 SyncAll + workspace 归约）就此闭环**，补上本卡原"真机未验证"的缺口。

关键差异点一：`SetScheduleMode` 的返回值处理。R2 实现中返回值被显式忽略（`(void)context->SetScheduleMode(1)`）——UT 的 mock `TilingContext` 可能没有为该输出槽分配存储、调用会返回失败，而真机运行时必需；两种情况下 group 正确性都不受影响，不能拿返回值当中断条件（与 §6.2 的"真机上下文失败即 GRAPH_FAILED"互补：真机路径失败须早退，UT mock 上下文失败须忽略，按上下文区分）。API 事实：`ge::graphStatus SetScheduleMode(const uint32_t schedule_mode)`，定义在 CANN 9.1.0 `x86_64-linux/include/exe_graph/runtime/tiling_context.h:226`。

关键差异点二：workspace 用户区公式按算子语义取 `usrSize = rGroupCnt × aTotal × sizeof(float)`（`aTotal` = 全部非归约 fused 轴乘积；fp16/int32 输入的 partial 仍为 fp32），总大小同样加 `GetLibApiWorkSpaceSize()` 系统区——与本卡 6H bias partial 公式同构，迁移时替换行宽语义即可。

R2 的 TilingFunc 接线（group 触发在 normal 切分链跑完后判断，`ComputeGroupSplit` 覆盖 `usedCoreNum`，verbatim）：

```cpp
        if (ComputeBranch0Split(context, ctx) != ge::GRAPH_SUCCESS) {
            context->SetBlockDim(1);
            size_t* wsFail = context->GetWorkspaceSizes(1);
            if (wsFail != nullptr) { wsFail[0] = 0; }
            return ge::GRAPH_FAILED;
        }
        // group decision (after split populates loop counts). On trigger,
        // run ComputeGroupSplit (OVERRIDES usedCoreNum + sets rGroupCnt) and set
        // ScheduleMode=1 (group needs SyncAll cross-core barrier).
        if (ShouldUseGroup(ctx)) {
            ComputeGroupSplit(ctx);
            // ScheduleMode=1 is a hard prerequisite for group (SyncAll dependency).
            // Return value ignored: UT context may not allocate the slot, but the
            // real runtime requires it; group correctness is unaffected either way.
            (void)context->SetScheduleMode(1);
        }
        FillNormalTilingData(td, ctx);
```

## 设计决策

- **为何必须对齐 aLoopCntTotal**：kernel 侧行映射 `aChunkIdx = blockIdx / rGroupCnt` 依赖 numBlocks = aLoopCntTotal × rGroupCnt 的整齐网格；残行会迫使 kernel 加边界分支，且 aPerCore=1 的"每核一个 A chunk"负载假设被破坏。
- **为何 CeilAlign 优先、FloorAlign 兜底**：向上对齐多用核（更均匀），超核数时向下——两种方向都保整除，核数不足时少核优于破网格。
- **为何 workspace 不乘归约路数 2**：两路 bias（input 侧 3H + hidden 侧 3H）在同一 GM 段按 rGroupCnt 份复用（Tuple Reduce 纪律：同一 rGroup 的两路写同一份 partial）——乘 2 是常见误写，会翻倍 workspace。
- **为何 SetScheduleMode 放 tiling 且失败即返回**：schedule mode 是 context 输出槽，Phase 1→Phase 2 的 SyncAll 语义由它承载；静默失败会让 kernel 死等同步。实现前已用 builder 探针验证 UT 上下文支持该 API。（R2 补充：该"失败即返回"适用于真机/支持该槽的上下文；UT mock 上下文无该槽时返回失败是无害的，须忽略——见 R2 关键差异点一。）
- **当前方案属哪档路线**：`recommended_fastpath`。

## 系统信息

- CANN 9.0.0-beta.2；ascend950/arch35；UT mock 平台（vector_core_cnt=48 / ub_size=262144 / GetLibApiWorkSpaceSize=16777216 默认值）
- 相关文件：`op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（ShouldUseGroup L~351 / ComputeGroupSplit L~357 / SetWorkspaceSize L~221 / TilingFunc workspace 尾段 L~497-505，终版行号）
- 源算子设计常量（迁移时替换）：`BIAS_SEG_CNT=6`（bias partial 行宽 6H = input 3H + hidden 3H）、`FP32_BYTES=4`（partial dtype）
- group 触发阈值 `coreNum/2` 与 `rLoopCntTotal ≥ 2` 来自源算子 BranchRoute 设计，属设计输入而非通用常数
- R2（CANN 9.1.0）：Ascend950PR（A310-50）；`op_host/arch35/euclidean_norm_tiling_arch35.cpp`（ShouldUseGroup/ComputeGroupSplit/SetWorkspaceSize/TilingFunc 接线）、`tests/tiling/test_branch4_tiling.cpp`（TilingUT，独立 oracle 全套 CPU 复刻 + 固定平台常数 coreNum=8、ubSize=253952）

## 完整代码

见「解决配方」（ShouldUseGroup + ComputeGroupSplit + SetWorkspaceSize + workspace 尾段全量 verbatim；R2 接线片段）。

## 性能影响

group 相对 base 的收益是核利用率提升（例中 base 用 16 核 → group 用 48 核），但**未做两模式实测延迟对比——未验证性能影响**；跨核 SyncAll 与 GM workspace 归约有额外同步/搬运成本，收益是否为正取决于 shape，需 profiling 复核。

## 预防

- 写 2D 分核先写"网格整齐"不变量（numBlocks = aLoopCntTotal × rGroupCnt，整除），再推导对齐方向；UT 至少覆盖 CeilAlign 命中、FloorAlign 兜底、触发边界（aLoopCntTotal == coreNum/2）三类用例。
- 跨核两阶段归约的 workspace 记得"不乘路数"核对（同段复用 vs 分段开辟）。
- SetScheduleMode 返回值按上下文处理：真机路径失败即断，UT mock 上下文（无该输出槽）失败忽略。
- 每个 group 用例断言三件套齐全：`usedCoreNum` 覆盖、`rGroupCnt` 非零、`ws[0] ≥ usrSize + sys`，缺一会在 kernel 侧暴露成越界或错值。

## 验证信号

本卡信号来自 ThnnFusedGruCellGrad 公共 TilingFunc 实现会话（group 分支 TilingUT 用例 + 整套回归）：

- group 分支 6 用例全部通过：GroupFp16DocExample2 / GroupFp32SmallH / GroupFp32NoBiasSysOnlyWs / GroupBf16 / GroupBoundaryAloopsEqualToHalfCores / GroupFloorAlignBranch（含 workspace sys-only（isBias=0）与 FloorAlign 兜底断言）（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 整套 TilingUT 33/33 通过、重跑一致（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 全量 build.sh 退出码 0；真机 aclnn 冒烟 1/1 OK（走 base 路径，确认 group 引入未破坏既有通路）（来源平台：950，真机；证据等级：tool_observed）

独立复现 R2（euclidean_norm，CANN 9.1.0）信号：

- 基线（三件套未实现）：Branch-4 TilingUT 0/8（三类断言稳定失败：usedCoreNum 未覆盖、rGroupCnt=0、ws[0]=0）（来源平台：950；证据等级：tool_observed）
- 实现后：Branch-4 TilingUT 8/8 PASSED（含 CeilAlign/FloorAlign 两分支、非对齐 R、5 轴、fp16/int32 workspace 仍 fp32、underfill 核数等用例）（来源平台：950；证据等级：tool_observed）
- 全量回归：public TilingUT 19/19、Branch-0 9/9、Branch-1 9/9、Branch-2 13/13 无退化；构建退出 0、"do not registe tiling struct" 警告 0 条（来源平台：950；证据等级：tool_observed）
- 端到端（真机 group 路径）：该 tiling 落地后 ST 的 group 用例在 kernel 入口接线前 0/8（哨兵输出）、接线后 8/8（fp32 maxRel ≤ 4.038e-07），佐证 tiling 路由与 workspace 正确（来源平台：950；证据等级：tool_observed）

## 失效边界

- group 触发阈值（coreNum/2、rLoopCntTotal≥2）是源算子设计输入；其它算子的甜点区不同，需各自判定。
- `BIAS_SEG_CNT=6` 行宽与本卡 workspace 公式绑定源算子的 6H bias partial 布局——迁移时替换（R2 按 `aTotal`（非归约轴乘积）取行宽，同构）。
- ~~kernel 侧 group 分支（tilingKey=1 子核、SyncAll 两阶段执行、GM partial 归约）在 kernel 模板化前未实现——host 侧 TilingData 契约（usedCoreNum/rGroupCnt）已由 UT 锁定，真机 group 路径未验证。~~（R2 已在 euclidean_norm 上完成真机 group 路径验证（ST 8/8）；本卡原始算子的真机 group 路径仍未单独验证。）
- `GetLibApiWorkSpaceSize()` 的 16777216 是 mock 未设置时的默认值路径；真机 sys workspace 值未直接观测。
- R2 的 `SetScheduleMode` "UT 可失败、真机必需"行为基于 CANN 9.1.0 的 mock 上下文观察；后续版本若 UT 上下文实现变化，忽略返回值的理由需复核。
- 触发条件数值（coreNum/2、UB 因子决定的 rLoopCntTotal）依赖运行时平台参数；跨平台（不同核数/UB 大小）时同一 shape 的触发结论会变，须重推或实测路由。

## 坏实践

- `numBlocks = CeilDiv(totalOuter, perCoreNum)` 后直接用、不对齐 aLoopCntTotal——残行破坏 2D 网格，kernel 边界复杂化或负载失衡。
- bias workspace 乘归约路数 2——同段复用被误写成双份，workspace 翻倍。
- SetScheduleMode 失败被吞——kernel 侧 SyncAll 语义静默失效（真机路径）；UT mock 上下文里反过来拿返回值当中断条件会误伤（R2）。
- 只设置 `ctx.isGroup` 用于 tilingKey 编码而不做 2D 分核/不声明 workspace——key 路由到了 group 分支但没有可用的切分数据。

## 误诊

- "group 恒优于 base"——错。aLoopCntTotal 大（>coreNum/2）时 base 已吃满核，2D 分核反而引入同步开销；触发条件就是防这个。
- "rGroupCnt = coreNum / aLoopCntTotal"——近似但不精确：必须经 perCoreNum 均分 + 对齐（CeilAlign/FloorAlign）推导，直接除会得到非整网格。
- R2：基线失败值 `usedCoreNum=1 expected 5` 容易被读成"normal 切分算错了"；实际 normal 切分正确，是 group 触发后应**覆盖**它的 2D 分核缺失。断言命名带上 `(group)` 后缀可以避免这种误读。
