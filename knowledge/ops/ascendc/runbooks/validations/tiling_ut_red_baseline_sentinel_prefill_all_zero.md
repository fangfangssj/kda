---
type: "Runbook"
title: TilingUT 红基线预填哨兵垃圾值防全 0 巧合命中
description: 期望输出全 0 的 tiling 字段重映射用 TDD 红基线时，若 td 预置 0 则 no-op stub 与真实现不可区分；预填哨兵垃圾值（-123456789）使 stub 必 FAIL、实现清零后必 PASS。
tags: [task.testing, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-17T04:11:06Z'
updated_at: '2026-09-17T04:11:06Z'
platforms: ["950"]
---

# TilingUT 红基线预填哨兵垃圾值防全 0 巧合命中

## 现象

在一个 softmax 基准算子（opbuild 自定义算子包，Ascend 950，CANN 9.1.0）空 tensor 分支（EmptyTensor）的测试先行开发中，需要为 host 侧 tiling 函数 `optiling::FillEmptyTilingData(BenchHarnessSoftmaxTilingData& td)` 先写 UT（红基线，此阶段函数是 no-op 桩），下一任务再实现它。该分支的语义是"EMPTY 重映射：全字段清零"——即**期望输出恰好在全 0 附近**（`usedCoreNum=0`、`aTotal=0`、`rTotal=0`、各 UB buffer size 全 0、`cacheBufUbSize=0` 区别于常规路径的 16KB 等 19 个字段）。

这里有个测试先天的盲区：TilingData 结构体 `BenchHarnessSoftmaxTilingData td;` 在栈上声明后若按零初始化处理（或测试者顺手 `memset(&td, 0, ...)`），那么 **no-op 桩什么都不做时字段已经是 0，与"真实现显式清零"的输出完全相同**——红基线会"巧合全绿"，测试先行失去对角线验证能力，桩阶段的实现缺陷（比如实现者以为调用方会清零、只清了部分字段）完全探测不到。

本轨迹采用哨兵垃圾值预填方案，实测闭环：桩阶段 0/15 通过（红）、实现（memset 全清零）后 15/15 转绿且既有 99 例回归不退化。

## 适用场景

- 工程形态：opbuild 算子包 host 侧 tiling 函数的 gtest UT（`tests/ut/`，链接 tiling_host.cpp 后在 host 上跑）；TDD 测试先行（先红后绿）工作流。
- API 路径：`<op>_tiling_host.cpp` 的字段重映射/填充函数（如 `FillEmptyTilingData`），被测对象为 host 纯函数，无设备依赖。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.1.0（与 frontmatter `platforms` 严格一致；本技术为 host 侧 gtest 行为，随算子包 UT 构建运行验证）。
- 适用粒度：`workflow_only`（UT 设计技术；对"期望输出与未初始化/清零态同形"的被测函数通用）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证场景：期望输出非全 0 的常规 tiling 用例（自身可区分，无需哨兵）；kernel 侧上板用例；未验证被测函数内部自带完整 memset 时哨兵的必要性（此时哨兵仅验证实现存在，不验证实现方式）。

## 根因/解决配方

**机制（单句）**：被测函数期望输出与"输入保持不动+全 0 初值"同形时，no-op 桩与真实现不可区分，红基线失效；把输入预填为非 0 哨兵垃圾值后，"不清零（桩/缺陷）"与"清零（实现）"在输出上可判定地区分开。

**判别要点**：桩阶段失败输出里应出现哨兵原值（如 `td.tmpBufUbSize Which is: -1234567890123 / kB2OracleEmpty.tmpBufUbSize Which is: 0`），这是"红基线真的在测清零行为"的直接证据；若桩阶段意外全绿，说明预填未覆盖到字段或被测路径另有清零，需要先查测试自身。完整可复用代码三件套（预填函数 / 用例体断言 / 实现侧 before-after）见下节。

## 完整代码

**1. 哨兵常量与预填函数**——覆盖 TilingData 全部字段：

```cpp
// ===== 哨兵垃圾值（让 stub no-op 可被检测，避免 memset 巧合命中）=====
constexpr int32_t GARBAGE_SENTINEL_I32 = -123456789;       // int32 哨兵
constexpr int64_t GARBAGE_SENTINEL_I64 = -1234567890123LL; // int64 哨兵

// ===== 预填哨兵垃圾值（让 stub no-op 可被检测）=====
// 覆盖 BenchHarnessSoftmaxTilingData 全字段（与设计取值表一一对应）。
void B2FillGarbage(BenchHarnessSoftmaxTilingData& td)
{
    td.axisNum = GARBAGE_SENTINEL_I32;
    for (int i = 0; i < MAX_PATTERN_RANK; ++i) {
        td.axisShape[i] = GARBAGE_SENTINEL_I64;
        td.axisStride[i] = GARBAGE_SENTINEL_I64;
    }
    td.rTotal = GARBAGE_SENTINEL_I64;
    td.rUbFactor = GARBAGE_SENTINEL_I64;
    td.rUbFactorAlign = GARBAGE_SENTINEL_I64;
    td.rChunkCnt = GARBAGE_SENTINEL_I64;
    td.aTotal = GARBAGE_SENTINEL_I64;
    td.aLoopCntTotal = GARBAGE_SENTINEL_I64;
    td.aBigCoreLoopCnt = GARBAGE_SENTINEL_I64;
    td.aSmallCoreLoopCnt = GARBAGE_SENTINEL_I64;
    td.aBigCoreCnt = GARBAGE_SENTINEL_I32;
    td.usedCoreNum = GARBAGE_SENTINEL_I32;
    td.xUbSize = GARBAGE_SENTINEL_I64;
    td.eUbSize = GARBAGE_SENTINEL_I64;
    td.yUbSize = GARBAGE_SENTINEL_I64;
    td.tmpBufUbSize = GARBAGE_SENTINEL_I64;
    td.cacheBufUbSize = GARBAGE_SENTINEL_I64;  // ★ 须被清零（EMPTY=0，非 16KB）
    td.rFullloadThresh = GARBAGE_SENTINEL_I64;
}
```

**2. 用例体**：预填 → 调被测函数 → 对照全 0 独立 oracle 逐字段断言：

```cpp
    // ── FillEmptyTilingData EMPTY 字段重映射 ──
    // 预填哨兵垃圾值：stub no-op 不清零 → 字段≠0 → FAIL（红基线）
    BenchHarnessSoftmaxTilingData td;
    B2FillGarbage(td);
    optiling::FillEmptyTilingData(td);

    // 设计取值表逐字段断言（全 0）
    EXPECT_EQ(td.axisNum, kB2OracleEmpty.axisNum);
    for (int i = 0; i < MAX_PATTERN_RANK; ++i) {
        EXPECT_EQ(td.axisShape[i], kB2OracleEmpty.axisShape[i]) << "axisShape[" << i << "]";
        EXPECT_EQ(td.axisStride[i], kB2OracleEmpty.axisStride[i]) << "axisStride[" << i << "]";
    }
    EXPECT_EQ(td.rTotal, kB2OracleEmpty.rTotal);
    EXPECT_EQ(td.rUbFactor, kB2OracleEmpty.rUbFactor);
    EXPECT_EQ(td.rUbFactorAlign, kB2OracleEmpty.rUbFactorAlign);
    EXPECT_EQ(td.rChunkCnt, kB2OracleEmpty.rChunkCnt);
    EXPECT_EQ(td.aTotal, kB2OracleEmpty.aTotal);
    EXPECT_EQ(td.aLoopCntTotal, kB2OracleEmpty.aLoopCntTotal);
    EXPECT_EQ(td.aBigCoreLoopCnt, kB2OracleEmpty.aBigCoreLoopCnt);
    EXPECT_EQ(td.aSmallCoreLoopCnt, kB2OracleEmpty.aSmallCoreLoopCnt);
    EXPECT_EQ(td.aBigCoreCnt, kB2OracleEmpty.aBigCoreCnt);

    // ★ 不变量：usedCoreNum==0（EMPTY 早退，SetBlockDim(1) 的 tiling-level 使能）
    EXPECT_EQ(td.usedCoreNum, kB2OracleEmpty.usedCoreNum);
    EXPECT_EQ(td.usedCoreNum, 0);

    // buffer 全 0（含 cacheBufUbSize=0，≠ 16KB）
    EXPECT_EQ(td.xUbSize, kB2OracleEmpty.xUbSize);
    EXPECT_EQ(td.eUbSize, kB2OracleEmpty.eUbSize);
    EXPECT_EQ(td.yUbSize, kB2OracleEmpty.yUbSize);
    EXPECT_EQ(td.tmpBufUbSize, kB2OracleEmpty.tmpBufUbSize);
    EXPECT_EQ(td.cacheBufUbSize, kB2OracleEmpty.cacheBufUbSize);
    EXPECT_EQ(td.cacheBufUbSize, 0) << "EMPTY 不分配 cacheBuf（≠16KB）";

    EXPECT_EQ(td.rFullloadThresh, kB2OracleEmpty.rFullloadThresh);
```

（`kB2OracleEmpty` 为独立 oracle：从设计文档取值表手抄的全 0 结构体，不调被测函数。）

**3. 实现侧（下一任务落地时）**：显式清零、不依赖调用方初始化状态——桩注释里"td 已由调用方零初始化"的假设正是哨兵要防的缺陷：

```cpp
// ── before（stub，含错误假设）──
// Branch-2 EmptyTensor（tilingKey=2）字段重映射。td 已由调用方
// 零初始化，stub 不填即符合 EMPTY 重映射（usedCoreNum=0 / aTotal=0 / rTotal=0 / buffer 全 0）。
void FillEmptyTilingData(BenchHarnessSoftmaxTilingData& td)
{
    (void)td;
    // 实现（待下一任务）。
}

// ── after（实现：显式全字段清零）──
//   td 由调用方可能未清零（UT 预填哨兵垃圾值检测红基线），故本函数显式清零全字段。
void FillEmptyTilingData(BenchHarnessSoftmaxTilingData& td)
{
    // 显式清零全字段（不依赖调用方 memset）：符合设计取值表 EMPTY 重映射口径。
    (void)memset(&td, 0, sizeof(BenchHarnessSoftmaxTilingData));
}
```

（实现文件头需补 `#include <cstring>`。）

## 设计决策

- **为何选非 0 哨兵而不是随机值**：哨兵要求（a）与任何合法期望值（本例全 0，含 `cacheBufUbSize=0` 这类"非常规 0"）都不相等；（b）可读、可 grep——失败输出里一眼认出 `-1234567890123` 即知"没清零"；随机值失败时还要先确认是不是数据污染。
- **为何 int32/int64 各设一个**：字段宽度不同（`axisNum`/`aBigCoreCnt`/`usedCoreNum` 是 int32），窄字段填宽哨兵会被截断成别的值，仍非 0 但失去"一眼认出"性质；分宽度定义保证每个字段的失败输出都是完整哨兵。
- **放弃的候选**：`memset(&td, 0xFF, ...)`（0xFF... 哨兵）——效果等价，但失败输出是 `-1`/大负数不如十进制哨兵好认；依赖 `td = {}` 零初始化——正是要防的反模式。
- **实现侧配套决策**：真实现用 `memset(&td, 0, sizeof(...))` 整体清零而非逐字段赋 0——与"全字段=0"的取值表口径天然一致，且未来 TilingData 加字段时不会漏清（哨兵 UT 会兜住漏清字段）。
- **当前方案路线档位**：`recommended_fastpath`。

## 系统信息

- 工具链：GCC 13.3.0 + googletest、CANN 9.1.0、Ascend 950 环境；UT 经 `cd operators && ./build.sh <OP>_package` 构建（tiling_host.cpp 直接链入 UT binary，host 纯函数测试，无需设备）。
- 相关文件：`operators/BenchHarnessSoftmax_package/tests/ut/test_bench_harness_softmax_branch2_tiling.cpp`（哨兵 UT，15 例）、`op_host/bench_harness_softmax_tiling_host.cpp`（被测 `FillEmptyTilingData`，stub→memset 实现）、`op_host/bench_harness_softmax_tiling_host.h`、`op_kernel/bench_harness_softmax_tiling.h`（TilingData 结构与 `MAX_PATTERN_RANK`）。

## 预防

- 为"输出全 0/常量"类分支写红基线 UT 前，先问一句"no-op 桩的输出和真实现差在哪"——差不在，就必须预填哨兵。
- 被测填充函数的实现注释里禁止出现"td 已由调用方零初始化"类假设；清零责任在实现内闭合。
- 评审红基线时检查失败输出是否出现哨兵原值，作为"红得有道理"的验收点。

## 验证信号

问题与验证来源：一次空 tensor 分支 tiling 函数的测试先行开发，桩阶段与实现阶段各跑一轮 UT。以下均为本会话实测：

- 红基线（桩 no-op）：Branch-2 子集 0 passed / 15 failed，失败输出含哨兵原值 `td.tmpBufUbSize Which is: -1234567890123` vs oracle `0`、`td.cacheBufUbSize Which is: -1234567890123` vs `0`（来源平台：950；证据等级：tool_observed）。
- 同轮回归：公共+Branch-0+Branch-1 共 99/99 通过，无退化（来源平台：950；证据等级：tool_observed）。
- 实现（memset 全清零）后：`[ PASSED ] 114 tests`（Branch-2 15/15 转绿 + 既有 99 回归全过）（来源平台：950；证据等级：tool_observed）。
- 既有 aclnn ST 回归（Branch-0 32/32 + Branch-1 17/17）不受影响（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 技术前提是被测语义"清零/写常量"与哨兵值域不交；若某字段合法期望恰为哨兵值（如某字段合法取 `-123456789`），该字段断言失效，需换哨兵。
- 哨兵只证明"实现写了这些字段"，不证明写对了业务语义（语义正确性由独立 oracle 断言承担，两者在此配方中成对出现）。
- 本例在 host 侧纯函数 UT 验证；kernel 侧 device 内存哨兵化（如 GM 预填 pattern 检测 kernel 未写）未验证，推断可迁移（未验证，推断可迁移）。

**未决项**：

- 未尝试"实现逐字段赋 0 + 人为漏清 1 个字段"的缺陷注入实验来进一步证明哨兵对部分清零缺陷的捕获力（当前由逻辑推论 + 桩阶段实测覆盖"全不清零"情形）。
## 坏实践

- 对全 0 期望的用例把 `td` 零初始化（或实现注释假设调用方清零）——桩与实现不可区分，红基线形同虚设。
- 红基线"全绿"时不追查为什么绿——本配方下桩阶段全绿只有两种解释（预填漏字段/另有清零路径），都说明测试没测到目标行为。

## 误诊

- 无（本闭环内未出现基于错误假设的往返；桩注释"td 已由调用方零初始化"是被测试设计预先防住的坏假设，未实际造成误修）。
