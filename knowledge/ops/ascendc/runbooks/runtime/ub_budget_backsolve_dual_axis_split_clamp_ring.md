---
type: "Runbook"
title: UB 预算反解切分：R 因子反解、R 驻留扩 A、钳制环
description: 双轴 tiling 的 UB 预算反解环——先定 aUbFactor，从剩余预算反解 rUbFactor；R 全驻时反向扩 A；cacheBuf 容量超限时 aUbFactor 减半 FloorAlign 重算；装不下一个 R 元素时显式失败。
tags: [task.implementation, topic.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# UB 预算反解切分：R 因子反解、R 驻留扩 A、钳制环


## 现象

在为 ThnnFusedGruCellGrad（GRU 单步融合反向算子，ascend950/arch35）实现 host 侧公共 TilingFunc 时，核心难题是双轴（A=特征轴 H（切分轴）、R=批轴 B（循环轴））的 UB 切分：`aUbFactor`（单 chunk A 元素数）与 `rUbFactor`（单 chunk R 元素数）共享 UB 预算，且与 preBuf/postBuf/cacheBuf 三块 buffer 的字节量耦合——直接联立求解不可行。33 条 TilingUT 覆盖了该环的全部行为分支（含"UB 太小必须显式失败"的负向用例 BaseFp16UbTooSmallTilingFail）。

设计采用的迭代环：**先按 cache line 定 A 因子 → 用剩余 UB 预算反解 R 因子 → R 全驻时反向扩 A → cacheBuf 容量校验不过则 A 因子减半重算**，任何一步预算装不下一个 R 元素即 `OP_LOGE` + `GRAPH_FAILED`。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（host tiling 函数的双轴 UB 切分）
- API 路径：`Ops::Base::CeilDiv/CeilAlign/FloorAlign`（`op_common/op_host/util/math_util.h`）；`OP_CHECK_IF`/`OP_LOGE`（`op_common/log/log.h`）；平台粒度经 `Ops::Base::GetUbBlockSize/GetCacheLineSize` + `PlatformAscendC::GetCoreMemSize(UB)`
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）
- 适用粒度：`operator_family`（任何双轴（切分轴×循环轴）共享 UB 的 tiling）
- 路线优先级：`recommended_fastpath`（预算反解环是此类算子 tiling 的标准解法；替代方案"固定小因子"浪费 UB、"全量驻留"放不下）
- 明确不适用/未验证：单轴 tiling（无 R 反解/扩 A 步骤）；share 常量与 buffer 布局不同的算子需替换预算公式中的份额常量

## 根因

UB 预算被三块 buffer 联动占用，A 因子每变一次，所有 buffer 字节量线性变化，因此 R 因子没有闭式解，只能"先定 A、反解 R、必要时回缩 A"迭代。预算恒等式（本算子设计）：

- `ubAvailable = ubSize − 16KB（cacheBuf 固定预留）`
- `postBufSize = CeilAlign(BIAS_SEG_CNT × aUbFactor × maxDtypeSize, blockSize)`（post 段，6 份 × A 因子，按 block 对齐）
- `bytesPerRElem = (P_PRE + P_PRE_EXT) × aUbFactor × maxDtypeSize`（pre 主段 + pre 尾段 = 13+6=19 份额 × A 因子）
- R 反解：`rIMax = (ubAvailable − postBufSize) / bytesPerRElem`（同批 R 元素每个占 19 份额×A 因子字节）
- cacheBuf 容量：`cacheCount × CeilAlign(BIAS_SEG_CNT × aUbFactor, 8) × 4 ≤ 16KB`（二分缓存树 depth × 每 level 行宽 × fp32）

## 解决配方

完整切分环（verbatim，取自 ThnnFusedGruCellGrad TilingFunc 终版；`P_PRE/P_PRE_EXT/BIAS_SEG_CNT/CACHE_BUF_BYTES/TREE_LANE_ALIGN/FP32_BYTES` 为源算子设计常量——份额数 13/6、段数 6、16KB cacheBuf、lane 8 对齐，迁移时替换为各自 buffer 语义）：

```cpp
// ─── Step 1（§4.2）：aUbFactor = min(cachelineTmp, CeilAlign(H, bsElem))───
// 联合爬坡在本 pattern 退化：尾轴 H 先 CeilAlign 再与 cachelineTmp 比较；无 aUbFactorAlign。
void ComputeAUbFactor(ThnnFusedGruCellGradCtx& ctx)
{
    ctx.bsElem = ctx.blockSize / ctx.dtypeSize;                                   // fp32:8 / fp16,bf16:16
    ctx.cachelineTmp = FloorAlign(ctx.cacheLineSize / ctx.dtypeSize, ctx.bsElem); // fp32:64 / b16:128
    ctx.aUbFactor = std::min(ctx.cachelineTmp, CeilAlign(ctx.H, ctx.bsElem));
}

// ─── Step 2-4 环（§4.3 反解 rUbFactor → §4.4 R 全驻扩 A → §4.5 cacheBuf 容量校验，
//     违反时 aUbFactor 减半 FloorAlign 重算；§9 pseudocode「loop:」结构）───
ge::graphStatus ComputeUbSplit(gert::TilingContext* context, ThnnFusedGruCellGradCtx& ctx)
{
    ctx.maxDtypeSize = std::max(ctx.dtypeSize, FP32_BYTES);  // 恒 4（fp32 中间域）
    ctx.ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;          // 先扣固定 16KB cacheBuf
    for (;;) {
        const int64_t aUnit = ctx.aUbFactor;
        // Step 2（§4.3）：UB 预算反解 rUbFactor；rIMax<1 → OP_LOGE + TILING_FAIL
        const int64_t postBufSize = CeilAlign(BIAS_SEG_CNT * aUnit * ctx.maxDtypeSize, ctx.blockSize);
        const int64_t bytesPerRElem = (P_PRE + P_PRE_EXT) * aUnit * ctx.maxDtypeSize;  // 19 份额
        const int64_t rIMax = (ctx.ubAvailable - postBufSize) / bytesPerRElem;
        OP_CHECK_IF(rIMax < 1,
                    OP_LOGE(context, "UbSize is too small: rIMax=%ld < 1 (ubAvailable=%ld, postBufSize=%ld,"
                            " bytesPerRElem=%ld)", rIMax, ctx.ubAvailable, postBufSize, bytesPerRElem),
                    return ge::GRAPH_FAILED);
        ctx.rUbFactor = std::max(std::min(rIMax, ctx.B), int64_t(1));
        // Step 3（§4.4）：唯一 R 轴全驻 UB 后扩 A（cacheBuf 六段树根 682 钳制）
        if (ctx.rUbFactor == ctx.B) {
            int64_t aUnitMax = ctx.ubAvailable /
                (ctx.maxDtypeSize * ((P_PRE + P_PRE_EXT) * ctx.B + BIAS_SEG_CNT));
            aUnitMax = std::min(aUnitMax, ctx.H);  // TotalAProd = 1×H（前导 A=1 不贡献）
            aUnitMax = std::min(aUnitMax, CACHE_BUF_BYTES / (BIAS_SEG_CNT * FP32_BYTES));  // 16384/24 = 682
            aUnitMax = FloorAlign(aUnitMax, ctx.bsElem);  // tail-A：Min(∏A) 之后 FloorAlign
            if (aUnitMax > aUnit) {
                ctx.aUbFactor = std::min(aUnitMax, CeilAlign(ctx.H, ctx.bsElem));  // A 端重爬坡退化式
            }
        }
        // Step 4（§4.5）：cacheCount × CeilAlign(6×aUbFactor, 8) × 4 ≤ 16KB，违反迭代钳制
        ctx.rLoopCntTotal = CeilDiv(ctx.B, ctx.rUbFactor);
        const int64_t levelStride = CeilAlign(BIAS_SEG_CNT * ctx.aUbFactor, TREE_LANE_ALIGN);
        if (CacheCount(ctx.rLoopCntTotal) * levelStride * FP32_BYTES <= CACHE_BUF_BYTES) {
            break;
        }
        ctx.aUbFactor = std::max(ctx.bsElem, FloorAlign(ctx.aUbFactor / 2, ctx.bsElem));
    }
    return ge::GRAPH_SUCCESS;
}
```

二分缓存树深度辅助（verbatim）：

```cpp
// 树深（§4.5）：FindNearestPower2 = 严格小于入参的最大 2 幂（入参 2 → 1）
inline int64_t LargestPow2StrictlyBelow(int64_t n)
{
    if (n <= 1) {
        return 0;
    }
    int64_t p = 1;
    while (p * 2 < n) {
        p *= 2;
    }
    return p;
}

// CalLog2 = floor(log2(x))（x ≥ 1）
inline int64_t FloorLog2(int64_t x)
{
    int64_t r = 0;
    while (x > 1) {
        x >>= 1;
        ++r;
    }
    return r;
}

// cacheCount = CalLog2(FindNearestPower2(rLoopCntTotal)) + 1；
// rLoopCntTotal == 1 时树退化为单层根（cacheCount = 1）
inline int64_t CacheCount(int64_t rLoopCntTotal)
{
    if (rLoopCntTotal <= 1) {
        return 1;
    }
    return FloorLog2(LargestPow2StrictlyBelow(rLoopCntTotal)) + 1;
}
```

机制推导示例（3 个关键分支，48 核 / UB=262144 / blockSize=32 / cacheLine=256 / cacheBuf=16KB）：

**例 1 常规反解（fp32, B=64, H=2048）**：`bsElem=8, cachelineTmp=FloorAlign(64,8)=64` → `aUbFactor=min(64, 2048)=64`；`ubAvailable=262144−16384=245760`；`postBufSize=CeilAlign(6×64×4=1536,32)=1536`；`bytesPerRElem=19×64×4=4864`；`rIMax=(245760−1536)/4864=244224/4864=50`；`rUbFactor=min(50,64)=50`（R 未全驻，不扩 A）；`rLoopCntTotal=CeilDiv(64,50)=2`；`cacheCount(2)=FloorLog2(LargestPow2StrictlyBelow(2)=1)+1=0+1=1`；`levelStride=CeilAlign(384,8)=384`；`1×384×4=1536 ≤ 16384` ✓ 收敛。

**例 2 R 全驻扩 A（fp16, B=1, H=2048）**：`bsElem=16, cachelineTmp=128` → `aUbFactor=128`；`postBufSize=CeilAlign(6×128×4=3072,32)=3072`；`bytesPerRElem=19×128×4=9728`；`rIMax=(245760−3072)/9728=24`；`rUbFactor=min(24,1)=1==B` → 扩 A：`aUnitMax=245760/(4×(19×1+6))=245760/100=2457` → `min(2457, H=2048)=2048` → `min(2048, 682)=682` → `FloorAlign(682,16)=672 > 128` → `aUbFactor=min(672, CeilAlign(2048,16)=2048)=672`；`rLoopCntTotal=1`、`cacheCount=1`、`levelStride=CeilAlign(6×672=4032,8)=4032`、`4032×4=16128 ≤ 16384` ✓（刚好装下）。注意 682 钳制来自 cacheBuf 六段树根：`16384/(6×4)=682.67→682`，再 FloorAlign 到 bsElem=16 得 672。

**例 3 cacheBuf 钳制减半（fp16, B=792, H=2048）**：首轮 `aUbFactor=128`、`rUbFactor=24`、`rLoopCntTotal=CeilDiv(792,24)=33`；`cacheCount(33)=FloorLog2(32)+1=5+1=6`；`levelStride=CeilAlign(768,8)=768`；`6×768×4=18432 > 16384` ✗ → `aUbFactor=max(16, FloorAlign(64,16))=64` 重算：`postBufSize=CeilAlign(1536,32)=1536`、`bytesPerRElem=4864`、`rIMax=50`、`rUbFactor=50`、`rLoopCntTotal=CeilDiv(792,50)=16`、`cacheCount(16)=FloorLog2(8)+1=4`、`levelStride=CeilAlign(384,8)=384`、`4×384×4=6144 ≤ 16384` ✓ 收敛（aUbFactor 128→64，用例断言"ClampShrinksAUbFactor"）。

**负向（UB 过小）**：`rIMax<1` 时（`ubAvailable − postBufSize` 装不下一个 R 元素的 19 份额×A 因子字节）显式 `OP_LOGE("UbSize is too small: rIMax=%ld < 1 …")` + `GRAPH_FAILED`——TilingUT 用例 BaseFp16UbTooSmallTilingFail 断言该失败路径（设计要求 tiling fail 而非产出非法切分）。

## 设计决策

- **为何"先定 A、反解 R"而非联立**：postBuf 只依赖 A 因子，反解 R 是单变量除法；联立两因子无闭式解。
- **为何 R 全驻才扩 A**：R 未全驻时 pre 段按 R 元素线性吃预算，扩 A 会挤掉 R；R 全驻（rUbFactor==B）后 pre 段不再增长，剩余预算全部让给 A——此时 `aUnitMax = ubAvailable / (maxDtypeSize × (19×B + 6))`（每 R 元素 19 份额 + post 6 份额）。
- **为何扩 A 有 682 硬钳制**：cacheBuf（16KB）树根是 6 段 fp32——A 因子超 682 时树根自身超 cacheBuf，扩了也白扩；随后 FloorAlign 到 bsElem 保证搬运粒度。
- **为何减半用 FloorAlign(bsElem) 且下限 bsElem**：保持 block 对齐（搬运粒度），且 aUbFactor 不得低于一个 block 的元素数，否则减半环无终止下限。
- **为何 cacheCount 用"严格小于的最大 2 幂"**：二分树每层归约一半，rLoopCntTotal=2 时树只有 1 层（largest pow2 strictly below 2 = 1，log2=0，+1=1）——用"≤ 的最大 2 幂"会算成 2 层，postBuf 虚高。
- **当前方案属哪档路线**：`recommended_fastpath`。

## 系统信息

- CANN 9.0.0-beta.2；ascend950/arch35；UT mock 平台（vector_core_cnt=48 / ub_size=262144）
- 相关文件：`op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（ComputeAUbFactor L~292 / ComputeUbSplit L~301 / 树深辅助 L~64-96，终版行号）
- 源算子设计常量（迁移时替换）：`P_PRE=13`（pre 阶段份额）、`P_PRE_EXT=6`（pre 尾段）、`BIAS_SEG_CNT=6`（post 段数）、`CACHE_BUF_BYTES=16KB`、`TREE_LANE_ALIGN=8`（fp32 lane 对齐）、`FP32_BYTES=4`（中间域 dtype）
- 验证载体：tests/tiling TilingUT（base 扩 A/钳制/UB 过小等用例）+ 设计文档 worked examples

## 完整代码

见「解决配方」（ComputeAUbFactor + ComputeUbSplit + 三个树深辅助函数全量 verbatim）。

## 性能影响

无性能故事（本卡为 tiling 正确性算法；R 全驻扩 A 减少外层循环数属于设计动机，未做两路线实测对比——未验证性能影响）。

## 预防

- 写双轴切分前先写预算恒等式（各 buffer 字节 = 份额 × 因子 × dtype），确认哪块 buffer 只依赖 A、哪块依赖 A×R——反解方向由依赖结构决定。
- 迭代环必须有：显式失败条件（rIMax<1）+ 单调回缩（减半）+ 对齐下限（bsElem）——三者缺一会死循环或产出非法因子。
- cacheBuf 树深公式（严格小于的 2 幂）配一个 rLoopCntTotal=2 的 UT 用例——边界最易写错。

## 验证信号

本卡信号来自 ThnnFusedGruCellGrad 公共 TilingFunc 实现会话（实现后 TilingUT 全绿，含各分支专测用例）：

- 反解/收敛主路径用例通过：BaseFp32DocExample1、BaseFp32AloopsJustAboveHalfCores、BaseFp32ReduceAxisLengthOne、BaseFp32HiddenOne、BaseFp32SmallShapeNoBias、BaseFp32UnalignedSmallH（来源平台：950，TilingUT mock；证据等级：tool_observed）
- R 全驻扩 A 用例通过（例 2 机制）：BaseFp16ExpandAWhenRLoaded、BaseBf16ExpandAWhenRLoaded（aUbFactor 128→672/384 扩张断言）（来源平台：950，TilingUT mock；证据等级：tool_observed）
- cacheBuf 钳制减半用例通过（例 3 机制）：BaseFp16CacheBufClampShrinksAUbFactor（来源平台：950，TilingUT mock；证据等级：tool_observed）
- UB 过小显式失败用例通过：BaseFp16UbTooSmallTilingFail（tiling 返回失败而非产出非法切分）（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 核饱和/少核用例通过：BaseFp32LargeHCoreSaturation、BaseFp32FewCoresClamp（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 整套 33/33 通过、重跑一致；真机 aclnn 冒烟 1/1 OK（base 路径走本环）（来源平台：950，TilingUT mock + 真机；证据等级：tool_observed）

## 失效边界

- 份额常量（13/6/6/16KB/8lane）是源算子 buffer 布局设计输入——跨算子迁移必须替换为各自的 pre/post/cache 份额，公式骨架（反解/扩 A/钳制环）可迁移。
- 本环假设单一 R 轴（pattern [1(A), B(R), H(A)] 退化式）；多 R 轴嵌套时反解目标变为内积，需重推导。
- 例 2 的"16128 ≤ 16384 刚好装下"依赖 16KB cacheBuf 常量；改 cacheBuf 容量会改变扩 A 上限与钳制行为。
- 未验证 kernel 侧实际 UB 消耗与该预算的一致性（host 侧预算模型来自设计文档；kernel 联调在后续任务）。

## 坏实践

- `aUbFactor` 减半不做 `FloorAlign(bsElem)` 或不设 `max(bsElem, …)` 下限——环可能无限递减到 0 或产出非 block 对齐因子。
- UB 装不下时静默夹到 1 而非显式失败——下游拿到 rUbFactor=1 的"合法"切分，错误被推迟到运行时。
- cacheCount 用"≤ 的最大 2 幂"——rLoopCntTotal=2 时多算一层，postBuf 虚高导致过早钳制。

## 误诊

- "扩 A 应该无条件做（A 越大越好）"——R 未全驻时扩 A 会挤掉 R 因子（pre 段按 R 元素吃预算），必须等 rUbFactor==B 后再扩。
- "cacheBuf 校验只看行宽不看树深"——树深随 rLoopCntTotal 对数增长（33 块→6 层），行宽×层数才超限（例 3）。


