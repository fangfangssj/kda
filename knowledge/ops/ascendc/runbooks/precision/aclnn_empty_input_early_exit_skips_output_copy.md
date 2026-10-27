---
type: "Runbook"
title: aclnn空输入早退过宽：data非空indices空y未写
description: aclnn 适配层的空 tensor 快速返回用 HasEmptyTensor（data/indices/updates 任一为空即返回 SUCCESS），scatter 类算子 data 非空 + indices/updates 空时 y 语义上应等于 data（Kernel Phase1 拷贝），却因早退完全未写入（保持调用方内存原值）。修复为仅 data 为空才早返回（IsDataEmpty），并新增"y 初始化为全 0 验证 y==data"的回归测试。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# aclnn空输入早退过宽：data非空indices空y未写

## 现象

BenchNoHarnessScatter 算子（scatter 语义：`y = data.clone()` 为基底，再把 updates 按 indices 聚合进 y）完成全量验收后，做交付前代码检视时在 aclnn 适配层 `op_api/aclnn_bench_no_harness_scatter.cpp` 里发现一个正确性缺陷（检视定级 P0）：第一段接口里的"空 tensor 快速返回"判断写的是**任一**输入为空就返回：

```cpp
// 空 tensor 快速返回（y 视为空拷贝）
if (HasEmptyTensor(data, indices, updates)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

而 `HasEmptyTensor` 的实现是 `data->IsEmpty() || indices->IsEmpty() || updates->IsEmpty()`。问题场景：**data 非空、indices/updates 为空（0 元素）**——此时 y 与 data 同 shape、语义上必须等于 data（Kernel 的 Phase1 负责 data→y 拷贝、Phase2 因 indicesTotalNum=0 自然跳过），但这条早退路径既不下发 Kernel 也不做任何拷贝，第二段 `aclnnBenchNoHarnessScatter` 执行的是一个空 executor，y 保持调用方传入内存的原始值（未初始化/脏数据）。此缺陷在全部 875 条 ST 用例中都没有暴露——既有空张量用例是"全空"（data 与 indices 同时空，y 也是空，早退结果碰巧正确），缺"data 非空 + indices 空"组合的用例。修复前该失败现象未在运行时复现（由静态检视发现）；修复后补了专门回归用例在 NPU 上验证修复行为。

## 适用场景

- 工程形态：ascendc-registry-invoke 自定义算子的 aclnn 适配层（op_api）正确性；也适用于任何"输出=输入拷贝再修改"语义（scatter/scatter_like/out-place update 类）算子的空输入短路逻辑设计。
- API 路径：`aclnn<Op>GetWorkspaceSize` 第一段里的空输入快速返回分支；`aclTensor::IsEmpty()`；Kernel 侧 Phase1/Phase2 两阶段结构。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——空输入短路语义与平台无关；判别规则对同结构算子通用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：输出与输入无拷贝关系（纯函数式覆盖输出）的算子——任一输入空即可安全早退，本卡判别不适用；updates 空但 indices 非空的非法组合（参数校验应先拦截，本轨迹走 CheckShape 前置校验路径）。

## 根因

空输入早退条件把"输出必为空"（只有 data 为空时成立，因 y shape=data shape）与"部分输入为空"混为一谈：`indices/updates` 为空只意味着**无更新操作**，不意味着**无输出内容**——y 仍须包含 data 的拷贝。早退分支跳过了 Kernel 下发，也就跳过了 Kernel Phase1 的 data→y 拷贝，y 成为未写入内存。

## 解决配方

修复前（`op_api/aclnn_bench_no_harness_scatter.cpp`，函数与调用点）：

```cpp
static bool HasEmptyTensor(const aclTensor* data, const aclTensor* indices,
                           const aclTensor* updates)
{
    return data->IsEmpty() || indices->IsEmpty() || updates->IsEmpty();
}

/**
 * @brief 第一段接口
 */
extern "C" aclnnStatus aclnnBenchNoHarnessScatterGetWorkspaceSize(
    const aclTensor* data,
    const aclTensor* indices,
    const aclTensor* updates,
    int64_t dim,
    int64_t reduce,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBenchNoHarnessScatter, DFX_IN(data, indices, updates), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(data, indices, updates, y, dim, reduce);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 tensor 快速返回（y 视为空拷贝）
    if (HasEmptyTensor(data, indices, updates)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Contiguous 处理（Kernel 输入需连续）
    auto dataContiguous = l0op::Contiguous(data, uniqueExecutor.get());
    ...
```

修复后（早退条件收紧为仅 data 空；data 非空 + indices/updates 空时正常下发 Kernel，由 Phase1 拷贝、Phase2 零元素自动跳过）：

```cpp
// 仅 data 为空时输出 y 必为空（y shape = data shape），可直接返回无需调用 Kernel。
// 注意：data 非空但 indices/updates 为空时不能早返回——仍需 Kernel Phase1 拷贝 data→y，
//       Phase2 因 indicesTotalNum=0 自动跳过，保证 y=data（ISSUE-002 修复）。
static bool IsDataEmpty(const aclTensor* data)
{
    return data->IsEmpty();
}

/**
 * @brief 第一段接口
 */
extern "C" aclnnStatus aclnnBenchNoHarnessScatterGetWorkspaceSize(
    const aclTensor* data,
    const aclTensor* indices,
    const aclTensor* updates,
    int64_t dim,
    int64_t reduce,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    ...
    // 空 data 快速返回：data 为空时 y 必为空（y shape = data shape），无需调用 Kernel。
    // data 非空但 indices/updates 为空时不早返回，由 Kernel Phase1 拷贝 data→y（ISSUE-002）。
    if (IsDataEmpty(data)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    ...
```

配套回归测试（`tests/st/test_aclnn_scatter.cpp` 新增，要点：**y 显式初始化为全 0 而非 data 值**，专门检测 Phase1 拷贝是否发生）：

```cpp
// ============================================================================
// ISSUE-002 回归测试：data 非空 + indices/updates 空 → y 必须等于 data（Phase1 拷贝生效）
// ============================================================================
// 关键：y 初始化为 0（非 data 值），验证 Kernel Phase1 确实拷贝了 data→y。
// 旧代码 HasEmptyTensor 任一空即早返回，y 未写入（保持 0）→ 测试失败。
// 新代码仅 data 空才早返回，Kernel Phase1 拷贝 data→y → y == data → 测试通过。
bool RunIssue002Regression(aclrtStream stream) {
    LOG_PRINT("\n[Real/ISSUE-002] data 非空 + indices 空: 验证 y=data 拷贝生效");

    // data [4,5] float32, 值 1..20
    std::vector<int64_t> dataShape = {4, 5};
    std::vector<int64_t> idxShape  = {0, 5};  // 空 indices（0 元素）
    DType dataDt = DType::FLOAT32;
    DType idxDt  = DType::INT32;

    size_t dataNum = static_cast<size_t>(GetShapeSize(dataShape));  // 20
    size_t dataBytes = dataNum * DTypeSize(dataDt);
    std::vector<uint8_t> dataHost(dataBytes);
    for (size_t i = 0; i < dataNum; ++i) {
        WriteFromDouble(dataHost.data(), i, dataDt, static_cast<double>(i + 1));
    }

    // y 初始化为全 0（关键：不是 data 值，用于检测 Phase1 拷贝是否生效）
    std::vector<uint8_t> yHost(dataBytes, 0);

    // indices/updates 空（0 字节数据）
    std::vector<uint8_t> idxHost(1, 0);
    std::vector<uint8_t> updHost(1, 0);

    void* data_dev = nullptr; void* idx_dev = nullptr; void* upd_dev = nullptr; void* y_dev = nullptr;
    aclTensor* data_t = nullptr; aclTensor* idx_t = nullptr; aclTensor* upd_t = nullptr; aclTensor* y_t = nullptr;
    bool success = false;
    auto cleanup = [&]() {
        if (data_t) aclDestroyTensor(data_t);
        if (idx_t)  aclDestroyTensor(idx_t);
        if (upd_t)  aclDestroyTensor(upd_t);
        if (y_t)    aclDestroyTensor(y_t);
        if (data_dev) aclrtFree(data_dev);
        if (idx_dev)  aclrtFree(idx_dev);
        if (upd_dev)  aclrtFree(upd_dev);
        if (y_dev)    aclrtFree(y_dev);
    };

    if (CreateAclTensor<uint8_t>(dataHost, dataShape, dataDt, &data_dev, &data_t) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 data tensor 失败"); cleanup(); return false;
    }
    if (CreateAclTensor<uint8_t>(idxHost, idxShape, idxDt, &idx_dev, &idx_t) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 indices tensor 失败"); cleanup(); return false;
    }
    if (CreateAclTensor<uint8_t>(updHost, idxShape, dataDt, &upd_dev, &upd_t) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 updates tensor 失败"); cleanup(); return false;
    }
    // y 用全 0 初始化（非 data 值）
    if (CreateAclTensor<uint8_t>(yHost, dataShape, dataDt, &y_dev, &y_t) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 y tensor 失败"); cleanup(); return false;
    }

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    auto ret = aclnnBenchNoHarnessScatterGetWorkspaceSize(
        data_t, idx_t, upd_t, 0, 0, y_t, &ws, &ex);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWorkspaceSize 失败: %d", static_cast<int>(ret)); cleanup(); return false;
    }

    void* workspace = nullptr;
    if (ws > 0) {
        if (aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] workspace 分配失败"); cleanup(); return false;
        }
    }
    ret = aclnnBenchNoHarnessScatter(workspace, ws, ex, stream);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnBenchNoHarnessScatter 失败: %d", static_cast<int>(ret));
        if (workspace) aclrtFree(workspace); cleanup(); return false;
    }
    aclrtSynchronizeStream(stream);

    // 拷贝结果回 Host
    std::vector<uint8_t> resultHost(dataBytes, 0);
    ret = aclrtMemcpy(resultHost.data(), dataBytes, y_dev, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("  [FAIL] D2H 拷贝失败: %d", static_cast<int>(ret));
        if (workspace) aclrtFree(workspace); cleanup(); return false;
    }

    // 验证 y == data（Phase1 拷贝生效）
    bool pass = true;
    for (size_t i = 0; i < dataNum; ++i) {
        float actual = static_cast<float>(ReadAsDouble(resultHost.data(), i, dataDt));
        float expected = static_cast<float>(i + 1);
        if (actual != expected) {
            pass = false;
            LOG_PRINT("  [FAIL] y[%zu]=%.1f != data=%.1f (Phase1 拷贝未生效)", i, actual, expected);
        }
    }
    if (pass) {
        LOG_PRINT("  [PASS] y == data（data 非空 + indices 空，Phase1 拷贝正确生效）");
    }

    if (workspace) aclrtFree(workspace);
    cleanup();
    return pass;
}
```

修复后重打包（`build.sh --soc=ascend950 -j8`）+ `--force` 安装再跑全量。

## 设计决策

- 为什么早退条件只看 data：y 的 shape 与 dtype 由 data 决定（y=data.clone() 语义），data 空 ⟺ y 空，此时无任何输出内容可写，早退安全；indices/updates 空只影响 Phase2 聚合量（indicesTotalNum=0），Kernel 自身可处理，无须 host 侧短路。
- 为什么回归测试把 y 初始化为全 0：若 y 初值取 data（与一些既有用例的做法相同），即使早退路径漏写，比对 y==data 也碰巧通过——全 0 初值让"未写入"直接显形为 `y[i]=0 != data[i]`。
- 备选方案（未采用）：在早退分支里 host 侧补一次 data→y 拷贝——引入第二条数据通路，重复 Kernel Phase1 职责且要处理非连续/异步语义，不如放行给 Kernel。
- 当前方案属 recommended_fastpath。

## 验证信号

来源说明：问题与验证来源：交付前代码检视发现的 P0 正确性问题（缺陷代码与修复均从事件流中的文件读取/编辑恢复）；修复后在 NPU 上以新增回归用例与全量套件验证。说明：修复前的错误输出（y 为脏数据）未在运行时采集——既有用例集没有覆盖该输入组合，本卡"现象"依据为检视结论与缺陷代码的静态语义；以下信号均为修复后实测。

- 回归用例（Real NPU）：ISSUE-002 回归 1/1 通过，日志 `[PASS] y == data（data 非空 + indices 空，Phase1 拷贝正确生效）`（y 初值全 0）（来源平台：950；证据等级：tool_observed）。
- 全量回归：ST Real 876/876（875 原有 + 1 新回归）、Mock ST 876/876（来源平台：950；证据等级：tool_observed）。
- UT 17/17（op_host 2 InferShape + 15 Tiling）（来源平台：950；证据等级：tool_observed）。
- PyTorch ST 795/795（含 bfloat16/int64 等 dtype 组合）（来源平台：950；证据等级：tool_observed）。

## 预防

- 写"空输入快速返回"前先写判别式：**哪个输入为空才能推出"输出无内容"**（本例只有 data）；对"输出=输入基底再修改"类算子，任何只空"操作数"不空"基底"的组合都不能短路。
- 空张量边界用例按输入集合的**子集组合**枚举（data 空×indices 空 / data 非空×indices 空 / 全空），只测"全空"会漏掉这类缺陷。
- 回归测试的输出 tensor 一律用非期望值（如全 0）初始化，让"未写入"可检测（与 [ST 红基线输出 poison 防 stub 巧合命中](../validations/st_red_baseline_poison_prefill_exact_zero.md) 同一原则）。

## 失效边界

- 本卡判别依赖"y shape=data shape、y 内容=data 基底"的算子结构；输出 shape 与所有输入都相关的算子（如 reduction 输出）早退条件需另行推导。
- 修复走 Kernel 路径意味着 data 非空 + indices 空时会真实下发 Kernel（多一次 dispatch）；若该组合在热点路径上高频出现，可评估 host 侧拷贝短路，但须处理非连续 y（本轨迹未做，未验证）。
- 未验证、推断可迁移：indices 非空 + updates 空等非法组合在 CheckShape 的拦截行为。

**未决项**：

- 优先项：在旧版本算子包上运行 `RunIssue002Regression` 采集修复前的失败输出（y=全 0 vs data），把"现象"从静态推断补成运行时实测证据——本轮修复先于用例落地，未留存失败态。
- data 非空 + indices 空组合的性能影响（多一次 Kernel dispatch）未量化。
## 坏实践

- `HasEmptyTensor(任意输入)` 式一把梭早退——把"操作数为空"误当"输出为空"。
- 回归测试把 y 预填为期望值——掩盖未写入缺陷。
- 因"全部现有用例通过"而认为空输入逻辑无需专测——本缺陷正是藏在全绿的用例集下。

## 误诊

- 无（本缺陷由代码检视直接定位，修复一次通过；中间确认过 Kernel/tiling 对 indicesTotalNum=0 的 Phase2 跳过行为无需改动，早退条件是唯一修改点）。
