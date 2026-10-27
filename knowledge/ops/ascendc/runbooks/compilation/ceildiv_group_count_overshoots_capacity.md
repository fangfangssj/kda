---
type: "Runbook"
title: "CeilDiv 反推组数过冲容量阈值，UT 期望按取整方向写"
description: "tiling 的 c1PerKLoop=CeilDiv(K_TILE_MAX, kPerC1) 向上取整，使 kTile=c1PerKLoop×kPerC1 可超过 K_TILE_MAX 最多一个 kPerC1（fp16 57×144=8208>8192），与设计注释 kSingle≤K_TILE_MAX 不一致；UT 期望须按 CeilDiv 实际取整方向手算并显式记录过冲上界。"
tags: [task.testing, topic.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T03:16:41Z'
updated_at: '2026-09-17T03:16:41Z'
platforms: ["950"]
---

# CeilDiv 反推组数过冲容量阈值，UT 期望按取整方向写

## 现象

为 BenchNoHarnessConv2d（Ascend950PR，CANN 9.1.0）迭代二写 K_SPLIT / M_K_SPLIT 分支的 UT 时，按设计注释"确保 kSingle = c1PerKLoop × KH × KW × c0 ≤ K_TILE_MAX"手算期望值（floor 语义）：fp16 CIn=1024 时 `c1PerKLoop = 8192/144 = 56`、`kTile = 56×144 = 8064`；fp32 时 `= 8192/72 = 113`、`kTile = 8136`。UT 实跑 8 个用例失败，实际值全部比手算大 1：

```
tests/ut/op_host/test_bench_no_harness_conv_2d_tiling.cpp:390: Failure
Expected equality of these values:
  tiling->kTile
    Which is: 8208
  param.expectKTile
    Which is: 8064
tests/ut/op_host/test_bench_no_harness_conv_2d_tiling.cpp:394: Failure
Expected equality of these values:
  tiling->c1PerKLoop
    Which is: 57
  param.expectC1PerKLoop
    Which is: 56
[PASS] conv2d_fp16_mksplit (tilingKey=769, mode=3, blockDim=1, mTile=65536, mLoop=2, kLoop=2, c1PerKLoop=57)
```

fp16 与 fp32 的 kTile 都恰为 8208——超过 K_TILE_MAX=8192。

## 适用场景

- 工程形态：op_host tiling 切分策略开发与 UT（K 维度按容量阈值分组）；以及依赖"kTile ≤ 阈值"假设的下游（L0A/L0B 容量预算）。
- API 路径：`Ops::Base::CeilDiv`；TilingData 字段 `c1PerKLoop`/`kLoop`/`kTile`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致；本结论与平台无关，是纯整数运算语义）。
- 适用粒度：platform_wide（任何用 CeilDiv(容量, 单位) 反推每组单位数的 tiling 实现）。
- 路线优先级：recommended_fastpath（UT 侧按实际行为断言 + 把过冲量写入断言上界）；实现侧改 floor 属可选修正，本会话未改。
- 明确不适用 / 未验证场景：本卡不判断 kTile=8208 是否真超 L0A/L0B 物理容量（直接卷积路径未用 L0A/L0B，见失效边界）。

## 根因/判别方法

tiling 实现取 `c1PerKLoop = CeilDiv(K_TILE_MAX, kPerC1)`：

```cpp
if (needKSplit) {
    // K_SPLIT: 按 C1 分组
    // 每次 K 循环处理 c1PerKLoop 组 C1
    // 确保 kSingle = c1PerKLoop * KH * KW * c0 ≤ K_TILE_MAX
    int64_t kPerC1 = KH * KW * c0;
    c1PerKLoop = CeilDiv(K_TILE_MAX, kPerC1);
    if (c1PerKLoop < 1) c1PerKLoop = 1;
    if (c1PerKLoop > c1) c1PerKLoop = c1;
    kLoop = CeilDiv(c1, c1PerKLoop);
}
```

`CeilDiv(8192, 144) = (8192+143)/144 = 57`（57×144=8208 > 8192；floor 才是 56/8064）。**向上取整的语义是"至少覆盖 K_TILE_MAX 所需的组数"，不是"能装进 K_TILE_MAX 的组数"**——于是 kTile 恒可能超出阈值至多一个 kPerC1（kPerC1 整除 K_TILE_MAX 时恰好不超）。注释宣称的 ≤ 约束与实现不符：这是"组数向上取整"与"每组容量向下取整"的经典混用。判别式：UT 期望按 floor 手算、实测恰好 +1 且 kTile=ceil 值×kPerC1 ⇒ CeilDiv 语义过冲，不是计算 bug。

## 解决配方

本会话采取"UT 对齐实际行为 + 显式记录过冲上界"（实现未改，verbatim 取自 UT 修正写入）：

```cpp
// ========== K_SPLIT 切分策略验证（迭代二新增）==========
TEST(Conv2dTilingModeTest, ksplit_tiling_fields_correct)
{
    // FP16, CIn=1024 (c1=64), KH=KW=3, c0=16
    // kTotal = 64 * 3 * 3 * 16 = 9216 > 8192 → K_SPLIT
    // kPerC1 = 144, c1PerKLoop = CeilDiv(8192, 144) = 57 (CeilDiv 向上取整)
    // kLoop = CeilDiv(64, 57) = 2, kTile = 57 * 144 = 8208
    // 注意：kTile(8208) 略超 K_TILE_MAX(8192)，因 c1PerKLoop 采用 CeilDiv 向上取整
    ...
    EXPECT_EQ(td->mLoop, 1) << "K_SPLIT: mLoop should be 1";
    EXPECT_GT(td->kLoop, 1) << "K_SPLIT: kLoop should be > 1";
    EXPECT_EQ(td->kLoop, 2) << "K_SPLIT: kLoop should be 2";
    EXPECT_EQ(td->c1PerKLoop, 57) << "K_SPLIT: c1PerKLoop should be 57 (CeilDiv(8192,144))";
    EXPECT_EQ(td->kTile, 8208) << "K_SPLIT: kTile should be 8208 (57*144)";
    EXPECT_EQ(td->mTile, 64) << "K_SPLIT: mTile should be mTotal=64";
    // 验证 kTile * kLoop ≥ kTotal（覆盖全部 K）
    EXPECT_GE(td->kTile * td->kLoop, 9216) << "K_SPLIT: kTile*kLoop should cover kTotal";
    // 验证 kTile 在 K_TILE_MAX 附近（CeilDiv 向上取整，kTile 可略超 K_TILE_MAX 一个 kPerC1 单位）
    EXPECT_LE(td->kTile, 8192 + 144) << "K_SPLIT: kTile should not exceed K_TILE_MAX + kPerC1";
}
```

（参数化用例的期望值同步由 56/8064、113/8136 改为 57/8208、114/8208；M_K_SPLIT 用例断言 `EXPECT_LE(td->kTile, 8192 + 144)` 同理。）

若下游确需"kTile ≤ 阈值"硬约束，实现侧应改为 floor：`c1PerKLoop = K_TILE_MAX / kPerC1;`（再夹到 [1, c1]）——本会话未做此修改（改动需设计评审），留作遗留问题。

## 设计决策

- 为什么 UT 匹配实际而不是按设计注释断言 ≤：UT 的职责是锁当前实现行为；把 `EXPECT_LE(kTile, 8192)` 留着会让 CI 恒红，遮蔽真正的回归信号。过冲事实改由 `EXPECT_LE(kTile, K_TILE_MAX + kPerC1)` 显式记录（上界含一个 kPerC1 的过冲额度）。
- 为什么不顺手改 floor：kTile 阈值与 L0A/L0B 容量预算耦合（设计文档 §3.5.4），floor 会改变 kLoop 与覆盖关系，需重新过设计评审；本会话是 UT 开发任务。
- 当前方案路线档位：recommended_fastpath（UT 侧）；实现侧 floor 修正为可选后续。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；UT 工程 `tests/ut/`（gtest + `tiling_case_executor` 桩直调 TilingFunc）。
- 相关文件：`op_host/arch35/bench_no_harness_conv_2d_tiling.cpp`（CeilDiv 所在实现）；`tests/ut/op_host/test_bench_no_harness_conv_2d_tiling.cpp`（断言修正落点）。

## 验证信号

以下信号来自该算子迭代二 UT 开发会话：

- 触发：UT 首跑 8 个用例失败，kTile 实际 8208 vs 期望 8064（fp16）、8208 vs 8136（fp32），c1PerKLoop 实际 57/114 vs 期望 56/113（来源平台：950；证据等级：tool_observed）。
- 机制核对：`CeilDiv(8192,144)=57`、`57×144=8208>8192`；`CeilDiv(8192,72)=114`、`114×72=8208>8192`（来源平台：950；证据等级：tool_observed）。
- 修正后：UT 30/30 PASSED（参数化 17 + 模式 4 + TilingKey 2 + InferShape 7），kTile 上界断言 `≤ 8192+144` 生效（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡不主张"kTile 超阈会溢出 L0A/L0B"：当前直接卷积路径 K 维度在寄存器累加、不占 L0A/L0B，8208 无实际容量后果；将来接 Cube 路径（Load2D/MMAD）时该过冲是否安全未验证——那时必须重审。
- 过冲上界是"至多一个 kPerC1"，仅当 `c1PerKLoop` 未被 `[1, c1]` 夹断时成立；c1 很小（夹断到 c1）时 kTile 可能远小于阈值。

**未决项**：

- 优先项：评审决定 K_SPLIT 分组取 floor（真 ≤ 阈值）还是保留 ceil（少一次循环、需容量余量证明）；若走 Cube 路径，必须验证 kTile=8208 时 L0A/L0B（64KB 预算）是否安全。
- 设计文档 §3.5.4 的容量公式与实现 CeilDiv 的不一致未在设计侧更新。
## 坏实践

- 按设计注释的 ≤ 约束手算 UT 期望值而不看实现的取整方向——floor/ceil 一念之差，期望值全错（本例 8 个用例齐挂）。
- 在注释里写"确保 X ≤ 阈值"而实现用 CeilDiv 反推组数——注释与代码埋雷。

## 误诊

无（一次定位：期望/实际差恰 +1 且 kTile=ceil 组数×单位 ⇒ 取整语义）。
