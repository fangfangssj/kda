---
type: "Runbook"
title: TilingUT 悬垂指针—CreateCap 缓冲需 fixture 持有
description: TilingUT 把 GetRawTilingData 裸指针存成员，CreateCap 局部缓冲随 InvokeTiling 返回释放，断言读前被堆复用成乱码；应在调用后立即快照进 fixture 向量。
tags: [task.testing, topic.memory, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingUT 悬垂指针—CreateCap 缓冲需 fixture 持有


## 现象

在自定义算子的公共 tiling 单元测试（gtest、测试先行红基线阶段）首次全量运行后检查输出：33 个用例按预期全部失败（红基线 0 passed / 33 failed 本身正常——被测 TilingFunc 还是桩，不填任何字段），但**失败消息里的字段值不对劲**。TilingContext 的 `GetTilingKey()` 读到 `94534513905616` 这类指针样的值（桩没有 set tilingKey，非 0 可以理解，但该值不稳定）；更严重的是 EMPTY 路由用例的 TilingData 字段读出了 ASCII 形状的垃圾：

```
ThnnFusedGruCellGradTilingTest.EmptyRAllCores:
  act->aSmallCoreLoopCnt  Which is: 8019896578607376233   ← 期望 0
  act->postBufSize        Which is: 2314861703650832495   ← 期望 0
  act->hiddenSize         Which is: 2335225711164858400   ← 期望 0
  act->isBias             Which is: 540701545            ← 期望 0
```

（这些十进制值按小端拆字节是 `o`/`l`/`e`/`d`/`i`/`L`/`i`/空格 等 ASCII——缓冲区里被写入了文本。）而 harness 明明在调用前对 tiling data 缓冲做过 memset 清零，且结构体前 40 字节（usedCoreNum/aTotal/aUbFactor/aBigCoreCnt/aBigCoreLoopCnt）确实读到了干净的 0，只有偏移 40 之后的字段变垃圾。乱码不确定、逐次运行不同——这会污染后续"由红转绿"阶段断言的可信度：失败可能来自实现 bug，也可能来自框架自身的悬垂读取。

## 适用场景

- 工程形态：算子仓自定义算子的 host 侧 tiling gtest（模式 B：dlopen 安装 run 包 .so，经 V2 注册表解析 TilingFunc 并在 UT 进程内调用）。
- API 路径：`gert::TilingData::CreateCap(cap)`（返回管理堆缓冲的局部智能指针）→ `OpTilingContextBuilder::TilingData(reinterpret_cast<gert::TilingData*>(buf.get()))` → 被测函数返回后 `ctx->GetRawTilingData()->GetData()` 取裸指针。
- 验证过的平台与 CANN 版本：Ascend 950（编译目标 ascend950；host 侧 gtest 运行，平台参数经 mock）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（gert::TilingData 的所有权语义属框架 API，不绑定算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：在调用被测函数的同一作用域内立即消费 raw tiling data 的写法不受影响（缓冲尚存活）；本卡针对"把裸指针存进 fixture 成员、跨函数边界延后读取"的 harness 结构。

## 根因

`gert::TilingData::CreateCap(kTilingDatumSize)` 分配的堆缓冲由**函数局部**的 `std::unique_ptr` 持有；harness 在 `InvokeTiling()` 里创建它、传给 builder、调被测 TilingFunc，然后把 `ctx->GetRawTilingData()->GetData()` 的**裸指针**存进 fixture 成员 `rawTd_` 就返回了——unique_ptr 析构，缓冲回到堆池。后续 `Expect*` 比较与 gtest 失败消息格式化（`Which is: ...` 的字符串构造）会做堆分配，复用这块内存并把文本字节写进去，于是悬垂的 `rawTd_` 读到的不是桩留下的 0，而是"谁最后用了这块内存"的内容——这解释了"前 40 字节干净、后面 ASCII 乱码、逐次不稳定"的全部观测（清零是调用前做的；释放后的覆写只发生在被重新分配到的区域）。

## 解决配方

在 `InvokeTiling()` 内、被测函数返回后**立即**把 raw tiling data 字节快照进 fixture 拥有的存储，`rawTd_` 改指快照。before/after 逐字取自事件流：

```cpp
    // ── before：存裸指针（悬垂）──
    void InvokeTiling(const CaseIn &c)
    {
        // ...（组装输入/输出/CompileInfo/PlatformInfo/workspace，略同 after）...
        auto tilingDataBuf = gert::TilingData::CreateCap(kTilingDatumSize);
        ASSERT_NE(tilingDataBuf, nullptr);
        // Deterministic zeroed tiling data region (fields the tiling function
        // does not write must read back 0; header lives at offset 0).
        std::memset(tilingDataBuf.get() + sizeof(gert::TilingData), 0, kTilingDatumSize);
        // ... builder.Build() → ctx → tilingFunc_(ctx) ...
        auto ret = tilingFunc_(ctx);
        retOk_ = (static_cast<uint32_t>(ret) == static_cast<uint32_t>(ge::GRAPH_SUCCESS));
        blockDim_ = ctx->GetBlockDim();
        tilingKey_ = ctx->GetTilingKey();
        const auto *rawTd = ctx->GetRawTilingData();
        rawTd_ = (rawTd != nullptr) ? reinterpret_cast<const uint8_t *>(rawTd->GetData()) : nullptr;   // ← 悬垂：缓冲随本函数返回释放

        size_t *wsSizes = ctx->GetWorkspaceSizes(1);
        ASSERT_NE(wsSizes, nullptr);
        ws0_ = wsSizes[0];
    }

    // fixture 成员（before）
    gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc_ = nullptr;
    bool retOk_ = false;
    const uint8_t *rawTd_ = nullptr;
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;
    size_t ws0_ = 0;
    int64_t sysWs_ = 0;
```

```cpp
    // ── after：调用后立即快照进 fixture 持有的向量 ──
        auto ret = tilingFunc_(ctx);
        retOk_ = (static_cast<uint32_t>(ret) == static_cast<uint32_t>(ge::GRAPH_SUCCESS));
        blockDim_ = ctx->GetBlockDim();
        tilingKey_ = ctx->GetTilingKey();
        const auto *rawTd = ctx->GetRawTilingData();
        const uint8_t *tdData = (rawTd != nullptr) ? reinterpret_cast<const uint8_t *>(rawTd->GetData()) : nullptr;

        // Snapshot the raw tiling data into fixture-owned storage: the CreateCap
        // buffer is a local unique_ptr that dies with this function, and later
        // heap reuse (e.g. gtest failure-message formatting) would corrupt the
        // dangling pointer before the Expect* comparisons read it.
        if (tdData != nullptr) {
            rawTdBytes_.assign(tdData, tdData + kTilingDatumSize);
            rawTd_ = rawTdBytes_.data();
        } else {
            rawTdBytes_.clear();
            rawTd_ = nullptr;
        }

        size_t *wsSizes = ctx->GetWorkspaceSizes(1);
        ASSERT_NE(wsSizes, nullptr);
        ws0_ = wsSizes[0];
    }

    // fixture 成员（after）
    gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc_ = nullptr;
    bool retOk_ = false;
    std::vector<uint8_t> rawTdBytes_;   // fixture-owned snapshot of the raw tiling data
    const uint8_t *rawTd_ = nullptr;    // points into rawTdBytes_ after InvokeTiling
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;
    size_t ws0_ = 0;
    int64_t sysWs_ = 0;
```

要点：快照必须在 `InvokeTiling()` 返回**前**完成（此时 CreateCap 缓冲仍存活）；`blockDim_`/`tilingKey_`/`ws0_` 是值拷贝，本就不受影响。

## 设计决策

- 为何选"字节快照进 `std::vector<uint8_t>`"而不是延长 CreateCap 缓冲的生命周期（如把 unique_ptr 存成 fixture 成员）：① 快照语义正是 UT 需要的——"被测函数返回那一刻的 tiling data 定格"，与后续内存活动解耦，红/绿两阶段都确定；② 不依赖 `gert::TilingData` 对象本身的拷贝/移动语义（框架类型，语义不透明）；③ 改动最小，只动指针来源，不改断言逻辑。
- 备选与放弃原因：把 `tilingDataBuf`（unique_ptr）提升为 fixture 成员——放弃：context/builder 也是局部对象，`GetRawTilingData()` 指向的归属链不清晰，且延长缓冲只保证"这块内存不被复用"，不能保证 `GetRawTilingData()` 在 context 析构后仍合法；在 `InvokeTiling` 内直接做完所有断言——放弃：会破坏"一用例一 TEST_F + 集中式 Expect* 助手"的结构。
- 当前方案属哪档路线：recommended_fastpath。

## 系统信息

- CANN：`/home/<user>/Ascend/cann-9.0.0-beta.2`；`gert::TilingData::CreateCap`/`GetRawTilingData` 定义于 `${CANN}/x86_64-linux/include/exe_graph/runtime/`（tiling_data.h / tiling_context.h）。
- 目标平台：ascend950（host 侧 gtest，平台参数 mock）。
- 相关文件：`operators/<Op>_package/tests/tiling/test_tiling_gtest.cpp`（唯一改动文件；fixture `InvokeTiling()` 与成员声明两处修改）。

## 完整代码

修复涉及的完整函数与成员声明即上文「## 解决配方」before/after 两段（`InvokeTiling` 的组装部分在另一张 TilingUT 配方卡中完整内联；本卡聚焦生命周期修复的最小完整因果单元：`CreateCap` 局部持有 → 裸指针外存 → 快照替代）。

## 预防

- 任何"把框架 context 里的裸指针存成员、跨函数延后读"的 UT harness，先问一句：**这块内存归谁、活多久**。`CreateCap` 返回智能指针即提示所有权在调用方局部。
- 调试期发现"字段值像 ASCII/指针、逐次运行不同、memset 过仍非 0"三联征，直接怀疑 use-after-free，先用快照法排除，不要先怀疑被测实现。

## 验证信号

来源说明：以下信号来自本次自定义算子公共 TilingUT 红基线的两次全量运行（修复前后各一次），同一套 33 用例、同一桩实现。

- 修复前：失败消息中 EMPTY 用例字段读到 ASCII 形垃圾值（如 `aSmallCoreLoopCnt=8019896578607376233`、`isBias=540701545`），且 `GetTilingKey()` 读到 `94534513905616`——非确定、非 memset 的 0（来源平台：950（host 侧 gtest）；证据等级：tool_observed）。
- 修复后（快照进 `rawTdBytes_`）：重跑全链，33 用例失败消息全部为**确定性的干净值**——桩未写字段读回 0（如 EmptyRAllCores：`act->usedCoreNum` 读 0 对期望 48、`act->aTotal` 读 0 等），无任何 ASCII 乱码；`[==========] 33 tests from 1 test suite ran. / [  PASSED  ] 0 tests. / [  FAILED  ] 33 tests`，红基线稳定（来源平台：950（host 侧 gtest）；证据等级：tool_observed）。
- 修复后门禁复核：改动仅限 `tests/tiling/` 三文件（本卡只动 test_tiling_gtest.cpp），实现文件零改动，红基线仍 0/33（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定"`CreateCap` 缓冲由局部智能指针持有"的使用形态；若未来 CANN 改为 context 拥有 tiling data 缓冲，快照法仍然安全（只是不再必需）。
- 快照长度取 `kTilingDatumSize`（CreateCap 容量）；若被测实现把 dataSize 填为更大值再越界写，快照不覆盖该场景（`GetTilingData<T>` 会 `SetDataSize(sizeof(T))`，正常实现不超过容量）。
- 未在其他 CANN 版本验证（推断可迁移——悬垂指针与堆复用是通用机制，不依赖版本）。

## 坏实践

- 在 UT harness 里保存框架返回的裸指针并跨函数使用（本卡 before 形态）：内存已释放，读取结果取决于后续堆活动，测试变成"非确定性"；红基线阶段它表现为乱码失败消息，绿基线阶段可能表现为**间歇性假失败**——更危险。
- 见到乱码先怀疑被测实现或 memset 逻辑：memset 是调用前做的，释放后的覆写与它无关；先修 harness 再判实现。

## 误诊

- 排查初期曾沿着"谁在 tiling 调用后写了缓冲区"的方向怀疑 CANN plog/日志系统或 builder 析构写入（垃圾值确为文本字节）；定位到"值只在**本函数返回之后**才被读取"这一事实后，确认不是"有人越界写活缓冲"，而是"缓冲已死、读者悬垂"——覆写者是 gtest 失败消息格式化等正常堆活动。`GetTilingKey()` 的 `94534513905616` 属另一现象（builder 未初始化字段的残留值，桩未 set 时非 0），与悬垂无关、无需修复。

## 未决事项

- 无（根因已到内存生命周期机制层，修复后信号确定）。
