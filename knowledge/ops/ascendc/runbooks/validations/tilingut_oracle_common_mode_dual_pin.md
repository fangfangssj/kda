---
type: "Runbook"
title: TilingUT独立oracle双钉：全字段比对+字面常数防同源漂移
description: TilingUT 独立 oracle 与被测 TilingFunc 同源于同一设计文档，全字段一致性比对无法发现共同错误；每例再以设计文档手算 worked example 的字面常数钉死实际捕获值，并用 grep 门禁强制 oracle 不调被测实现命名空间。
tags: [task.testing, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingUT独立oracle双钉：全字段比对+字面常数防同源漂移


## 现象

在为 GRU Cell 反向算子的 base 分支（条带分核，tilingKey=0）写 gtest 形态的 TilingUT 时，断言结构面临一个结构性风险：UT 的「独立 oracle」是按设计文档（HostTiling.md/DESIGN-BRANCH-0.md）公式手写的参考实现（`OracleComputeCommon`，纯整数算术重写 UB 切分/大小核均衡/缓存树公式），而被测 TilingFunc（`op_host/arch35/..._tiling_arch35.cpp`，optiling 命名空间）也是按**同一份设计文档**实现的——两者同源。若设计文档公式本身有错，或 oracle 抄录公式时与实现犯了同一个错，oracle 与实现会「一起错」：全字段一致性比对（`ExpectCommonMatchesOracle` 逐字段 EXPECT_EQ）照样全绿，错误被共同模式掩盖，UT 给出虚假信心。本会话的解法是**双钉**：每条用例先走 `RunCommonCase`（独立 oracle 全字段比对，公共机制不动），再按设计文档「具体切分计算」节 worked example 的**手算数字**，把实际捕获的 TilingData 字段以字面常数钉死（如 `EXPECT_EQ(act->preBufSize, 12800)`）——字面常数独立于被测实现与 oracle 代码，构成第三源；另用 grep 门禁机械强制 oracle 独立性（Oracle* 函数体不得出现 `optiling::` 调用）。14 条双钉用例全绿且钉子全部命中，实现、oracle、文档手算三方一致。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓 TilingUT（gtest）的断言结构设计；任何「oracle 按设计文档重写参考实现」的宿主侧单测。
- API 路径：gtest `EXPECT_EQ`/`ASSERT_*`；被测宿主侧 TilingFunc 输出的 TilingData 结构（`reinterpret_cast` 捕获缓冲后逐字段断言）；被测实现命名空间（如 `optiling::`）作为独立性门禁的检索目标。
- 验证过的平台与 CANN 版本：950（ascend950 run 包、宿主 mock 平台 coreNum=48/ubSize=256KB 执行，非上板）+ CANN cann-9.0.0-beta.2。与 frontmatter `platforms` 严格一致。
- 适用粒度（transfer_scope）：operator_family——任何带设计文档 worked example（手算数值表）的算子 tiling/切分类单测均可套用；不限于 tiling。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：无设计文档手算数值可比对的场景（缺第三源，双钉退化为单钉，只剩 oracle 比对）；非确定性输出（依赖随机/时间的 tiling）未验证；harness 本身的搭建见同主题 [TilingFunc 无设备 gtest 单测卡](tilingfunc_no_device_gtest_registry_mock_oracle.md)（本卡只覆盖断言层）。

## 根因

oracle 一致性比对的盲区是**共同模式失效（common-mode failure）**：oracle 与实现都从同一设计文档推导，文档错 → 两者同错 → 比对恒绿。这不是本会话发生的实际故障（本会话为绿基线，三方全一致），而是写断言时识别的结构性风险；本卡为 validated_pattern——验证的是双钉结构可执行且钉住真实数值（14 用例 × 每例 4~12 个字面钉子全部命中实际捕获值）。

## 解决配方

双钉断言结构（每条分支用例的固定形态）：

1. `RunCommonCase(c)`：走公共机制——InvokeTiling（调用被测 TilingFunc 并捕获 TilingData/blockDim/tilingKey/workspace）+ `OracleComputeCommon`（独立 oracle 按文档公式重算）+ `ExpectCommonMatchesOracle`（逐字段全比对）。公共机制不改（保护既有 33 条用例零回归）。
2. `EXPECT_EQ(tilingKey_, 0U)`：直接断言实际捕获的分支路由值（不经 oracle 中转）。
3. **字面钉子**：`reinterpret_cast` 捕获缓冲为 TilingData 结构后，按设计文档 worked example 的手算数字逐项 `EXPECT_EQ` 字面常数（切分因子、循环计数、大小核三元组、buffer 字节数），并断言可由 TilingData 字段独立推导的形态量（如尾 R 块有效长度 `B − (rLoopCntTotal−1)×rUbFactor`）。
4. **独立性门禁**（执行后自查）：`grep -E 'Oracle(Compute|ChooseTilingKey)' -A 40 <ut>.cpp | grep -E 'optiling::[A-Za-z]'` 必须无命中——机械保证 oracle 函数体不调被测实现命名空间（防 oracle 直接复用被测代码而失去独立性）。

字面钉子的数值来源必须是**第三源**：设计文档「具体切分计算」节的手算 worked example（或写用例前的独立预核算脚本对文档数值的复核），不能从 oracle 代码或被测实现输出抄录——否则钉子与比对同源，失去防漂移作用。

## 设计决策

- **为何双钉而非只留 oracle 比对**：oracle 与实现同源（同一设计文档），一致性比对对共同错误盲视；字面常数来自文档手算，独立于两份代码，任一方漂移即钉子爆红。
- **为何不修改公共 helper**：任务约束「不改动公共 TilingUT」；新增用例只在自身 TEST_F 体内追加钉子（`reinterpret_cast` 读 fixture 捕获缓冲），公共 `RunCommonCase`/oracle 零改动，33 条公共用例零回归风险。曾考虑让 `RunCommonCase` 返回 oracle 结构以便钉 oracle 值——被弃（改公共基础设施，违背回归不退化要求）。
- **钉实际捕获值还是钉 oracle 值**：钉实际捕获值（`act->字段`）——`RunCommonCase` 已证实际==oracle，再钉字面常数即同时锁住「实际==文档手算」；比钉 oracle 变量多一层传递性证明。
- **为何加 grep 门禁**：oracle 独立性是双钉成立的前提（oracle 若调被测实现，比对退化为自比对）；人工约定易腐化，grep 机械门禁进验收。
- 当前方案属 recommended_fastpath 路线。

## 系统信息

- 工程形态：ops-registry-invoke 算子仓 `operators/ThnnFusedGruCellGrad_package/tests/tiling/test_tiling_gtest.cpp`（单一 UT 源文件，公共用例区 + 分支用例区同文件）。
- 被测对象：run 包安装树 `libcust_opmaster_rt2.0.so` 内注册的 TilingFunc（加载/调用机制见 [TilingFunc 无设备 gtest 单测卡](tilingfunc_no_device_gtest_registry_mock_oracle.md)）。
- 相关文件清单：`tests/tiling/test_tiling_gtest.cpp`（读 + 编辑追加 14 条双钉用例）、`docs/.../design/HostTiling.md` 与 `design/branches/DESIGN-BRANCH-0.md`（oracle 公式与 worked example 数值来源，读）、`op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（被测实现，读）。
- 平台口径：宿主 x86_64，mock 平台 coreNum=48/ubSize=262144/blockSize=32/cacheLineSize=256（设计文档 worked example 平台值）；CANN cann-9.0.0-beta.2。

## 完整代码

分支用例节注释（设计意图 verbatim，含平台口径与防漂移说明）：

```cpp
// ─── Branch-0（tilingKey=0 · base 条带分核）分支形态（DESIGN-BRANCH-0.md
//     §0「分支定位」进入条件：校验通过 + 非空（B>0 且 H>0）+ Group 触发条件不成立
//     （aLoopCntTotal > coreNum/2 或 rLoopCntTotal <= 1）；覆盖 §9「分支验证」登记
//     全部形态：E01–E06 / S1–S6 / N1 + B0-a..B0-e）───
//     每例先走 RunCommonCase（独立 oracle 全字段比对，公共机制不动），再按 §2
//     「具体切分计算」worked example 数字以字面常数钉死实际捕获值（独立于被测
//     实现与 oracle 代码，防两者同源漂移）。平台口径：fixture 默认 coreNum=48 /
//     ubSize=256KB（§2 worked example 平台值；实机 64 核下 H=2048/fp32 形态会漂移
//     落 group，§9 注——UT 以 mock 平台钉死分支归属，复算以运行时值为准）。
```

最简形态用例（E01，双钉的最小样例，verbatim）：

```cpp
// E01（Validation.md 正向 #1 / DESIGN-BRANCH-0.md §9 行 1）：B=4, H=8, fp32,
// has_bias=true——标准形状：aUbFactor=8、R 全驻 rUbFactor=4、rLoopCntTotal=1 →
// base（Group 否定路径二 rLoopCntTotal <= 1）。
TEST_F(ThnnFusedGruCellGradTilingTest, Branch0E01Fp32StandardShape)
{
    CaseIn c;
    c.B = 4;
    c.H = 8;
    c.dtype = ge::DT_FLOAT;
    c.hasBias = true;
    RunCommonCase(c);
    ASSERT_NE(rawTd_, nullptr);
    EXPECT_EQ(tilingKey_, 0U);
    const auto *act = reinterpret_cast<const ThnnFusedGruCellGradTilingData *>(rawTd_);
    EXPECT_EQ(act->aUbFactor, 8);
    EXPECT_EQ(act->rUbFactor, 4);
    EXPECT_EQ(act->rLoopCntTotal, 1);
    EXPECT_EQ(act->usedCoreNum, 1);
}
```

满配形态用例（B0-a，钉子最密集的样例：切分因子/循环计数/大小核三元组/派生形态量/buffer 字节，verbatim）：

```cpp
// B0-a（§9 B0-a / §2 worked example）：B=64, H=2048, fp32, has_bias=true——多 A
// 块条带 + 大小核：aUbFactor=64、rUbFactor=50（rIMax=50）、rLoopCntTotal=2（Phase A
// 配对 P=FindNearestPower2(2)=1、T=1）+ 尾 R 块 rLen=64−50=14 < 50 清零；
// aLoopCntTotal=32 > coreNum/2=24 → base（Group 否定路径一）；aSmallCoreLoopCnt=0 →
// usedCoreNum=aBigCoreCnt=32、aBigCoreLoopCnt=1，其余 16 核 idle。
TEST_F(ThnnFusedGruCellGradTilingTest, Branch0B0aFp32MultiStripeBigSmallCore)
{
    CaseIn c;
    c.B = 64;
    c.H = 2048;
    c.dtype = ge::DT_FLOAT;
    c.hasBias = true;
    RunCommonCase(c);
    ASSERT_NE(rawTd_, nullptr);
    EXPECT_EQ(tilingKey_, 0U);
    const auto *act = reinterpret_cast<const ThnnFusedGruCellGradTilingData *>(rawTd_);
    EXPECT_EQ(act->aUbFactor, 64);
    EXPECT_EQ(act->rUbFactor, 50);
    EXPECT_EQ(act->rLoopCntTotal, 2);
    EXPECT_EQ(act->aLoopCntTotal, 32);      // > coreNum/2=24 → base
    EXPECT_EQ(act->aSmallCoreLoopCnt, 0);
    EXPECT_EQ(act->aBigCoreCnt, 32);
    EXPECT_EQ(act->aBigCoreLoopCnt, 1);
    EXPECT_EQ(act->usedCoreNum, 32);
    EXPECT_EQ(OracleLargestPow2StrictlyBelow(act->rLoopCntTotal), 1);  // P=1、T=1
    EXPECT_EQ(c.B - (act->rLoopCntTotal - 1) * act->rUbFactor, 14);    // 尾 R 块 rLen=14
    EXPECT_EQ(act->preBufSize, 12800);      // 64×50×4
    EXPECT_EQ(act->postBufSize, 1536);      // CeilAlign(6×64×4, 32)
}
```

公共比对入口（`RunCommonCase`，双钉第一钉，verbatim）：

```cpp
    // Run a non-empty success scenario end-to-end: invoke, then oracle-compare.
    void RunCommonCase(const CaseIn &c)
    {
        InvokeTiling(c);
        ASSERT_TRUE(retOk_) << "tiling unexpectedly FAILED";
        const auto e = OracleComputeCommon(c.B, c.H, DtypeSizeOf(c.dtype), c.hasBias ? 1 : 0, c.coreNum,
                                           c.ubSize);
        ASSERT_TRUE(e.ok) << "oracle says this input must be tileable";
        ExpectCommonMatchesOracle(e);
    }
```

oracle 独立性门禁（执行后自查命令，verbatim；无命中即 PASS）：

```bash
grep -E 'Oracle(Compute|ChooseTilingKey)' -A 40 <tests/tiling 目录>/*.cpp | grep -E 'optiling::[A-Za-z]' && echo "FAIL: oracle calls optiling::" || echo "PASS: independent"
```

## 预防

- 写「独立 oracle」型 UT 时先问：oracle 的公式来源与被测实现是否同一份文档？是 → 必须补第三源字面钉子（文档手算数值）。
- 钉子数值只能来自文档手算/独立预核算，禁止从被测实现输出或 oracle 代码抄录。
- oracle 函数命名加统一前缀（如 `Oracle*`），让 grep 门禁可机械检索独立性。
- 评审分支用例时核对：每例是否同时具备 oracle 全比对 + 至少一个不经 oracle 的字面钉子。

## 验证信号

本卡信号来自一次 GRU Cell 反向算子 base 分支（tilingKey=0）TilingUT 开发会话：追加 14 条双钉用例后完整执行 run.sh 与验证门禁。「950」指 ascend950 run 包在宿主以 mock 平台参数（coreNum=48/ubSize=256KB）执行，非上板。

- run.sh 全链：`[  PASSED  ] 47 tests.`（33 公共 + 14 新增双钉用例全部 OK，含 B0-a 的 preBufSize=12800/postBufSize=1536 等字面钉子全部命中实际捕获值），EXIT=0（来源平台：950，宿主侧 mock 平台参数执行；证据等级：tool_observed）。
- oracle 独立性门禁：`PASS: independent`（Oracle* 函数体无 `optiling::` 调用）（来源平台：950，宿主侧 mock 平台参数执行；证据等级：tool_observed）。
- 公共用例零回归：47 条 OK 中含全部 33 条公共用例（来源平台：950，宿主侧 mock 平台参数执行；证据等级：tool_observed）。

## 失效边界

- 双钉防的是「oracle 与实现共同漂移」；若设计文档 worked example 本身算错且预核算也按同一错公式复算，三源仍可能同错——第三源的独立性以人工手算为前提。
- 字面钉子绑定平台口径：钉死数值按 mock 平台（coreNum=48/ubSize=256KB）计算，换平台参数需整套重算。
- 派生量断言（如尾 R 块 rLen）依赖 TilingData 字段语义稳定；字段增删时钉子需同步维护。
- 本卡验证的是断言结构在 14 用例上三方一致；未发生过实际漂移被钉子捕获的实例（绿基线会话，无反例验证）。

## 坏实践

- 只写 oracle 全字段比对就收工：对 oracle 与实现的共同错误盲视（同源盲区）。
- 钉子数值从被测实现运行输出抄录：钉子与比对同源，防漂移作用归零。
- oracle 内部调用被测实现（`optiling::`）省事：比对退化为自比对，独立性门禁必须拦截。
- 为让用例过而改 oracle 公式/放松钉子常数：掩盖真因，属测试侧放松（反投机取巧红线）。

## 误诊

- 无（本会话未出现误诊路径）。


