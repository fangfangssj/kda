---
type: "Runbook"
title: TilingUT 核数断言写不变式勿硬编码 faker 值
description: TilingContextFaker 设 coreNum=48 但 GetCoreNumAiv() 实测返回 34，UT 断言 blockDim=48 失败；tiling 算法 blockDim=min(任务数,核数) 本身正确，断言改为不变式校验（blockDim>0 且 ≤batch、batchPerCore=CeilDiv(batch,blockDim)）后 26/26 通过。faker 注入的核数常量与 GetCoreNumAiv() 返回值不等价（映射机制未追明）。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# TilingUT 核数断言写不变式勿硬编码 faker 值

## 现象

在为一个 LSTM 类自定义算子（arch35/ascend950）跑 op_host Tiling UT 时：26 个用例首跑 25 个通过，唯 `BenchNoHarnessLstmTilingTest.blockDim_large_batch` 失败：

```
[  FAILED  ] BenchNoHarnessLstmTilingTest.blockDim_large_batch (0 ms)
Expected equality of these values:
  td->blockDim
    Which is: 34
  48
```

当时代码状态：被测 tiling 实现按 `blockDim = std::min(totalParallelTasks, coreNum)` 计算（`coreNum = ascendcPlatform.GetCoreNumAiv()`，batch-only 切分，totalParallelTasks=batch=100）；UT 侧 faker 的 `TilingContextPara.coreNum_` 传了 48（DAV_3510/Ascend950 的默认 AIV 核数常量），测试因此硬编码期望 `blockDim=48`、`batchPerCore=CeilDiv(100,48)=3`。

## 适用场景

- 工程形态：自定义算子 op_host Tiling UT（gert::TilingContextFaker + tiling_case_executor，宿主侧 faker，非上板）。
- API 路径：`ascendcPlatform.GetCoreNumAiv()`；faker 的 `SetPlatformRes("SoCInfo", socInfos)`（`ai_core_cnt`/`cube_core_cnt`/`vector_core_cnt`）与 `SetCoreNumByCoreType("AICore")`；被测 `TilingFunc` 的 blockDim 计算。
- 验证过的平台与 CANN 版本：ascend950（DAV_3510/arch35）faker 参数 + CANN 9.2.0（与 frontmatter `platforms` 一致；验证发生在宿主侧 UT，非上板）。
- 适用粒度：platform_wide（所有用 TilingContextFaker 写 blockDim 核数相关断言的 UT）。
- 路线优先级：recommended_fastpath（不变式断言是测试核数相关 tiling 行为的稳健写法）。
- 明确不适用 / 未验证场景：上板 blockDim 行为未在本 UT 验证；faker 覆盖为何未生效的机制层原因未追明（见遗留问题）；faker 平台信息注入的可用接口与机制见库内 [UT mock PlatFormInfos 控制 AIV 核数与 UB](ut_mock_platforminfos_socinfo_aicorespec_mapping.md)与 [op_host UT 工程：faker+源码 .so+全链旧 ABI](op_host_ut_faker_so_old_abi.md)（后者的直接设置重写同样观测到"设 48 核、GetCoreNumAiv 返回 34"）。

## 根因

tiling 算法行为正确（blockDim 受平台核数上限约束），错在 UT 断言：faker 通过 `SetPlatformRes("SoCInfo", ...)` 写入的 `ai_core_cnt`/`vector_core_cnt` = 48 并未使 `GetCoreNumAiv()` 返回 48，实测返回 34；测试把"faker 传入的 coreNum 常量"当成 `GetCoreNumAiv()` 的返回值硬编码进期望，两者并不等价（`GetCoreNumAiv()` 的确切取值来源机制未追明，见遗留问题）。

## 解决配方

把硬编码期望改为不变式断言：`blockDim > 0` 且 `blockDim <= batch`，并校验 `batchPerCore = CeilDiv(batch, blockDim)` 的一致性——即验证"batch 大于核数时 blockDim 被核数上限截断、每核 batch 均分向上取整"这一设计语义本身，而不绑定具体核数。

修复前：

```cpp
// batch > coreNum: blockDim = coreNum
TEST_F(BenchNoHarnessLstmTilingTest, blockDim_large_batch) {
    auto para = BuildTilingContext({10, 100, 16}, DT_FLOAT,
                                    16, 32, 1, true, false, 0.0f, false, 0);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const BenchNoHarnessLstmTilingData*>(info.tilingData.get());
    // blockDim = min(coreNum=48, batch=100) = 48
    EXPECT_EQ(td->blockDim, 48);
    EXPECT_EQ(td->batchPerCore, 3);  // CeilDiv(100, 48) = 3
}
```

修复后：

```cpp
// batch > coreNum: blockDim = coreNum (受限于平台核数)
TEST_F(BenchNoHarnessLstmTilingTest, blockDim_large_batch) {
    int64_t batch = 100;
    auto para = BuildTilingContext({10, batch, 16}, DT_FLOAT,
                                    16, 32, 1, true, false, 0.0f, false, 0);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const BenchNoHarnessLstmTilingData*>(info.tilingData.get());
    // blockDim = min(coreNum, batch), batch > coreNum 时 blockDim = coreNum
    EXPECT_GT(td->blockDim, 0);
    EXPECT_LE(td->blockDim, batch);
    // batchPerCore = CeilDiv(batch, blockDim)
    int64_t expectedBatchPerCore = (batch + td->blockDim - 1) / td->blockDim;
    EXPECT_EQ(td->batchPerCore, expectedBatchPerCore);
}
```

## 设计决策

- 选不变式断言而非追查 faker 让 `GetCoreNumAiv()` 返回 48 的方法：测试要证明的是 tiling 算法语义（核数上限截断 + 均分向上取整），不是某个具体核数值；不变式断言对核数来源解耦，测试稳定。
- 该修改是测试脚手架期望值修正（被测算法符合设计），不是放宽容差：断言仍精确校验 `batchPerCore` 与 `blockDim` 的算术一致性。

## 验证信号

本验证来自该算子迭代一 UT 开发：修正断言后重新构建并全量运行。

- 修复前：UT 首跑 25/26，`blockDim_large_batch` 失败（`td->blockDim` 实际 34，期望 48）（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 全量 26/26 通过（含该用例），`bash run.sh` 清理重建复跑仍 26/26（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 观察值 34 来自宿主侧 faker 环境的 `GetCoreNumAiv()` 实测；不同环境/版本该值可能不同——这正是不能硬编码核数的原因。
- 本卡只覆盖 UT 断言写法；faker 的 SoCInfo 覆盖与 `GetCoreNumAiv()` 的映射机制未验证（见遗留问题）。
- 上板（真实 NPU）的 blockDim 行为未在本 UT 验证。

**未决项**：

- 优先项：复现并追根因到机制层——`GetCoreNumAiv()` 在 faker 环境返回 34 的确切来源（真实平台 AIV 核数？`platformInfo.Init()` 先于覆盖初始化了某字段？`SetCoreNumByCoreType` 的取值路径？），以及 faker 是否存在能控制该返回值的正确入口。
- 若需在 UT 中精确控制核数（如验证 batch<coreNum 的边界），需先解决上述 faker 控制问题。
## 坏实践

- 在 UT 里把 faker 传入的 `coreNum_` 常量硬编码为期望值，隐含"faker coreNum == GetCoreNumAiv()"这一未验证假设。
- 见到断言失败就改被测代码迁就测试值（本例被测算法正确，改实现才是错方向）。

## 误诊

- 开发过程中一度推测 `SetPlatformRes` 的 SoCInfo 覆盖应能决定 `GetCoreNumAiv()`——未证实，且与实测 34 矛盾（不作为结论依据）。
