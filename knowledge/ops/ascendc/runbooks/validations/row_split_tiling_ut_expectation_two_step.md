---
type: "Runbook"
title: 行切分 tiling UT 期望须含反推 blockDim 步
description: 按行切分多核 tiling 采用「rowsPerCore=ceilDiv(M,min(coreNum,M))，再反推 blockDim=ceilDiv(M,rowsPerCore)」两步公式；M≤coreNum 时结果是 blockDim=M、rowsPerCore=1（每核一行），不是"单核包揽全部行"。UT 期望值按单步直觉推导会在 M<coreNum 用例上失败。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# 行切分 tiling UT 期望须含反推 blockDim 步

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子的按行切分多核 tiling 补充尾核边界 UT（`TailRowsUnderflowTest`，覆盖 M 略大于/小于 coreNum 的组合）时，运行 `tests/ut` 下 GoogleTest 套件出现失败：

```
[  PASSED  ] 35 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] TailRowsUnderflowTest.no_underflow_when_m_slightly_larger_than_corenum
```

先修掉一处 M=57 的 tailRows 期望（3→0，均匀分布时 tailRows 归零）后重跑仍然失败；用 `--gtest_filter=TailRowsUnderflowTest.*` 单跑拿到精确断言差异：

```
tiling_test.cpp:490: Failure
Expected equality of these values:
  blockDim
    Which is: 2
  tc.expectBlockDim
    Which is: 1
M=2 coreNum=32

tiling_test.cpp:492: Failure
Expected equality of these values:
  rowsPerCore
    Which is: 1
  tc.expectRowsPerCore
    Which is: 2
M=2 coreNum=32
（M=3 同样：actual blockDim=3/rowsPerCore=1，expect 1/3）
```

同时测试打印的实测值给出正确答案：`[TailRows] M=2 coreNum=32 → blockDim=2 rowsPerCore=1 tailRows=0`、`M=3 → blockDim=3 rowsPerCore=1`。当时代码状态：tiling 实现已含「反推 blockDim」修复（M=29/28→blockDim=15/rowsPerCore=2/tailRows=1 等实测值正确），失败的只是 UT 期望值——按"M 小于核数就压到 1 个核"的直觉写成 `{2, 32, 1, 2, 0}`、`{3, 32, 1, 3, 0}`（blockDim=1、rowsPerCore=M）。

## 适用场景

- 工程形态：算子仓开发（op_host TilingFunc）的 tiling UT 编写/评审；按行（M 维）切分的多核 tiling。
- API 路径：host 侧 TilingFunc 的 blockDim/rowsPerCore/tailRows 计算；UT 侧 gert::TilingContextPara + ExecuteTestCase 期望值推导。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35，coreNum=32），CANN 9.1.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：operator_family——任何按行切分的行并行 tiling 实现，不限于本算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：按列切分（ColSplit）/混合切分的 blockDim 语义；coreNum 与 M 关系以外的 tiling 字段（NAlign/chunk 等）推导；其他核数的具体数值表需按公式重算。

## 根因

UT 期望值按单步直觉 `blockDim = min(coreNum, M)` 推导，漏掉了 tiling 实现的第二步「反推」：实现为避免尾核空转，会先算 `rowsPerCore = ceilDiv(M, min(coreNum, M))`，再**反推** `blockDim = ceilDiv(M, rowsPerCore)`（只启动有工作的核）。当 M≤coreNum 时：`rowsPerCore = ceilDiv(M, M) = 1`，`blockDim = ceilDiv(M, 1) = M`——即每核分 1 行、启动 M 个核，而不是 1 个核处理 M 行。

推导链（与实测一致）：

- M=2, coreNum=32：initBlockDim=min(32,2)=2 → rowsPerCore=ceilDiv(2,2)=1 → blockDim=ceilDiv(2,1)=2；tailRows=2−(2−1)×1=1=rowsPerCore → 归 0。
- M=3, coreNum=32：initBd=3 → rowsPerCore=1 → blockDim=3；tailRows=3−2×1=1=rowsPerCore → 归 0。
- M=29, coreNum=28：initBd=28 → rowsPerCore=ceilDiv(29,28)=2 → blockDim=ceilDiv(29,2)=15；tailRows=29−14×2=1。
- M=57, coreNum=28：initBd=28 → rowsPerCore=3 → blockDim=19；tailRows=57−18×3=3=rowsPerCore → 归 0（均匀）。
- M=256, coreNum=32：rowsPerCore=8 → blockDim=32；tailRows=256−31×8=8=rowsPerCore → 归 0。

## 解决配方

修复前（UT 期望值按单步直觉推导——错误）：

```cpp
    std::vector<TailRowsCase> tailCases = {
        {29, 28, 15, 2, 1},
        {33, 32, 17, 2, 1},
        {53, 28, 27, 2, 1},
        {57, 28, 19, 3, 3},   // ← tailRows 期望错：均匀时应为 0
        {256, 32, 32, 8, 0},
        {1, 32, 1, 1, 0},
        {2, 32, 1, 2, 0},     // ← blockDim/rowsPerCore 期望反了：M≤coreNum 时 blockDim=M、rowsPerCore=1
        {3, 32, 1, 3, 0},     // ← 同上
    };
```

修复后（按两步公式完整推导——正确）：

```cpp
    std::vector<TailRowsCase> tailCases = {
        {29, 28, 15, 2, 1},
        {33, 32, 17, 2, 1},
        {53, 28, 27, 2, 1},
        {57, 28, 19, 3, 0},
        {256, 32, 32, 8, 0},
        {1, 32, 1, 1, 0},
        {2, 32, 2, 1, 0},
        {3, 32, 3, 1, 0},
    };
```

期望值推导检查清单（按行切分 tiling 通用）：

1. `initBlockDim = min(coreNum, M)`；
2. `rowsPerCore = ceilDiv(M, initBlockDim)`；
3. `blockDim = ceilDiv(M, rowsPerCore)`（反推，仅启动有工作的核——**不要**停在第 1 步）；
4. `tailRows = M - (blockDim-1)*rowsPerCore`；若 `tailRows == rowsPerCore` 则归 0（均匀分布）；
5. M≤coreNum 特例快速校验：期望必为 `blockDim=M、rowsPerCore=1、tailRows=0`。

## 设计决策

- 修 UT 期望值而不是改 tiling 实现：实测输出（M=29→15/2/1、M=57→19/3/0、M=2→2/1/0）与两步公式完全自洽，且 M=29/33/53/256/1 等既有期望全部命中，只有 M=2/M=3/M=57 三条是按单步直觉写错的——错在测试推导，不在实现。
- 期望表把推导过程写成逐行注释（每个 case 一行算式），而不是只留结果数字：失败时可直接对照断言差异定位是哪一步推导错了，本卡修复即受益于此格式。
- 当前方案路线档位：recommended_fastpath（正确推导期望是 UT 的本职；无替代路线）。

## 预防

- 写行切分 tiling 的 UT 期望前，先从实现里抄出完整的两步公式（含反推与 tailRows 归零规则），再手推每个 case；M<coreNum 的小 M 用例是单步直觉推导的高发错误区，务必单独覆盖。
- UT 期望与实现公式同源：若实现注释里有「反推 blockDim=ceil(M/rowsPerCore)」字样，期望推导必须包含该步。

## 验证信号

以下信号来自该算子尾核边界 UT 的开发会话：期望值两轮修正后全量回归。

- 修复前：UT 套件 36 tests 中 1 FAILED（TailRowsUnderflowTest），M=2 期望 blockDim=1/rowsPerCore=2 与实际 2/1 相反，M=3 期望 1/3 与实际 3/1 相反（来源平台：950；证据等级：tool_observed）。
- 修复中：首轮只修 M=57 tailRows（3→0）后重跑仍 FAILED，verbose 单跑定位到 M=2/M=3 断言差异（来源平台：950；证据等级：tool_observed）。
- 修复后：`[TailRows]` 实测链 M=29/28→15/2/1、M=33/32→17/2/1、M=53/28→27/2/1、M=57/28→19/3/0、M=256/32→32/8/0、M=1/32→1/1/0、M=2/32→2/1/0、M=3/32→3/1/0；UT 36/36 全部 PASS（来源平台：950；证据等级：tool_observed）。
- 无回归：随后 ST Real 模式 40/40（11 L0 + 29 L1，含 M=29/31/33 尾核用例）、Golden 自测 7/7、确定性验证 3/3 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡数值表基于 coreNum=28/32 的推导，其他核数值需按公式重算；公式本身（两步推导 + tailRows 归零）为该 tiling 实现的既定语义，跨核数成立。
- 「反推 blockDim」是该算子 tiling 的实现选择（防尾核 rowStart>M 空转），并非平台强制语义；其他算子的行切分实现若不反推，M<coreNum 时可能确实输出 blockDim=1/rowsPerCore=M——写 UT 前先读目标实现的公式，本卡给的是推导方法而非通用期望值。
- ColSplit（按列切分）路径的 blockDim 语义不受本卡覆盖。

**未决项**：

- 无（推导规则已闭环验证）。
## 坏实践

- UT 期望值按 `blockDim=min(coreNum,M)` 单步直觉写，不读实现的反推逻辑。
- 期望表只写结果数字不写推导算式，失败时无从定位是哪一步错。
- 期望值对不上就先怀疑实现、改实现迁就测试——本例实现是对的，错的是测试推导（该方向已在「设计决策」中用实测链排除）。

## 误诊

- 首轮修复只改了 M=57 的 tailRows 期望，误以为失败点是"均匀分布时 tailRows 不归零"一类单点笔误；重跑失败后才用单用例 filter 拿到 M=2/M=3 的断言差异，发现是 blockDim/rowsPerCore 整体推导模型错了（两次修正均有工具输出佐证，证据等级：tool_observed）。
