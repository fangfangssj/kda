---
type: "Runbook"
title: "多输出 TensorList 地址经 op 属性传 kernel"
description: REG_OP 单 OUTPUT 的 aclnn 算子多输出（TensorList）只有 y[0] 进 kernel，其余组越界写致精度失败；E 个输出设备地址改为装进 aclIntArray 作第 4 个 op 属性 → TilingData.outputAddrs 传给 kernel。workspace 传址两条路线（GetWorkspaceSize 阶段写 host workspace、Execute 预写 device workspace）均客观失败，见同批两张负结果卡。
tags: [task.implementation, topic.precision, topic.implementation.api_layer, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# 多输出 TensorList 地址经 op 属性传 kernel

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）上开发自定义 aclnn 算子 BenchNoHarnessGroupedMatmul（分组矩阵乘：x[M,K] × E 个专家权重，`split_item` 取 0/1 时输出为 E 个独立 tensor 组成的 TensorList，取 2/3 时输出为单个连续 [M,N] tensor）。跑 `tests/st` 的 C++ 原生 ST（Real 模式，`bash run.sh --real`）时，37 条 L0 用例中 14 条 split_item=0/1 用例精度失败：输出[0]、输出[1] 数值达标，从第 2~3 个输出起 mismatch 逐组放大，部分组合还伴随段错误。翻看 `op_api/bench_no_harness_grouped_matmul.cpp`（L0 API），调度 Kernel 的代码是：

```cpp
const aclTensor* yOut = (y != nullptr && y->Size() > 0) ? (*y)[0] : nullptr;
auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessGroupedMatmul,
    OP_INPUT(x, weight, bias), OP_OUTPUT(yOut),
    OP_ATTR(group_list, split_item, transpose_weight));
```

REG_OP 里 OUTPUT(y) 声明为单个 Tensor，因此 L0 API 只把 TensorList 的**第一个** tensor 传给 kernel；kernel 侧两条输出分支实际是同一段代码 `mm.IterateAll(yGm[gs.yOffset])`——所有组都往以 y[0] 为基址的连续缓冲里按 `start_g * N` 偏移写。当 `start_g * N` 超出 y[0] 的实际大小（m_0 × N）后就是越界写：前几组碰巧落在 y[0] 范围内所以"看起来对"，后面的组写进了相邻 device 内存，表现为后面输出的精度 mismatch / 崩溃。

## 适用场景

- 工程形态：aclnn 自定义算子开发（L2 `aclnn*GetWorkspaceSize/Execute` + L0 `ADD_TO_LAUNCHER_LIST_AICORE` + TilingFunc + AscendC kernel），C++ 原生 ST 黑盒验证。
- API 路径：`ADD_TO_LAUNCHER_LIST_AICORE`/`OP_ATTR`/`aclOpExecutor::AllocIntArray`、`gert::TilingContext::GetAttrs()->GetListInt(idx)`、`aclTensor::GetStorageAddr()`、TilingData 自定义结构、`GlobalTensor::SetGlobalBuffer`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"REG_OP 单 OUTPUT 但 aclnn 层需要多输出 TensorList"的算子（动态输出个数、split/分组输出类）。
- 路线优先级：recommended_fastpath（workspace 传址两条路线在本轨迹均客观失败，见本卡链接的两张负结果卡）。
- 明确不适用 / 未验证场景：
  - 输出个数静态已知：应直接在 REG_OP 声明多个 OUTPUT，不走本配方。
  - graph 模式（图构图）未验证：第 4 个属性未写进 def.cpp/REG_OP 声明，图模式下无 L0 API 注入该属性，TilingFunc 会走"属性缺失"回退分支。
  - E > 128 未验证：TilingData 内嵌数组按 E ≤ 128 设计。

## 根因

REG_OP 的 OUTPUT(y) 是单 Tensor，L0 API 只把 TensorList 第一个 tensor 的地址传进 kernel；split_item=0/1 时 kernel 拿不到其余 E-1 个输出 tensor 的 GM 地址，只能按单连续缓冲偏移写，越界后精度失败。单句归因：**多输出地址没有从 aclnn 层传到 kernel**。

## 解决配方

机制链：device 输出地址在 L2/L0 API（host 侧）已知 → 以 `int64_t` 装进 `aclIntArray` 作为第 4 个 op 属性随 `OP_ATTR` 下发 → TilingFunc 在 tiling 阶段用 `GetAttrs()->GetListInt(3)` 读出 → 写入 TilingData 的 `outputAddrs[128]` 数组（TilingData 会由框架搬运到 device）→ kernel 每组用 `SetGlobalBuffer` 造独立 GlobalTensor 输出。属性通道全程走框架标准搬运，绕开了 workspace 的两段式分配问题。

**修复前**（L0 API 只传 y[0]、3 个属性）：

```cpp
/*!
 * \brief AICore Kernel 调度
 *
 * 将输入、属性和预分配输出传递给 Kernel。
 * 属性顺序：group_list (ListInt), split_item (Int), transpose_weight (Bool)
 *           → 对应 tiling 函数中 attrs index 0/1/2
 *
 * 注意：REG_OP 声明 OUTPUT(y) 为单 Tensor，故从 TensorList 中取第一个 tensor
 * （split_item=2/3 时 TensorList 仅含 1 个 [M, N] tensor）。
 */
static const aclTensorList* BenchNoHarnessGroupedMatmulAiCore(
    const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
    const aclIntArray* group_list, int64_t split_item, bool transpose_weight,
    aclTensorList* y, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessGroupedMatmulAiCore, x, weight, bias, y);

    // 从 TensorList 提取第一个 tensor（REG_OP 定义为单 OUTPUT）
    const aclTensor* yOut = (y != nullptr && y->Size() > 0) ? (*y)[0] : nullptr;

    // ADD_TO_LAUNCHER_LIST_AICORE 调度 Kernel
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessGroupedMatmul,
        OP_INPUT(x, weight, bias), OP_OUTPUT(yOut),
        OP_ATTR(group_list, split_item, transpose_weight));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessGroupedMatmulAiCore failed."),
             return nullptr);
    return y;
}
```

**修复后**（L0 API 构造 output_addrs 第 4 属性；新增 `#include <vector>`）：

```cpp
/*!
 * \brief AICore Kernel 调度
 *
 * 将输入、属性和预分配输出传递给 Kernel。
 * 属性顺序：group_list (ListInt), split_item (Int), transpose_weight (Bool),
 *           output_addrs (ListInt, 可选) → 对应 tiling 函数中 attrs index 0/1/2/3
 *
 * split_item=0/1 时，output_addrs 携带 E 个输出 tensor 的设备地址（cast 为 int64_t），
 * TilingFunc 将其写入 TilingData，Kernel 直接从 TilingData 读取。
 */
static const aclTensorList* BenchNoHarnessGroupedMatmulAiCore(
    const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
    const aclIntArray* group_list, int64_t split_item, bool transpose_weight,
    aclTensorList* y, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessGroupedMatmulAiCore, x, weight, bias, y);

    // 从 TensorList 提取第一个 tensor（REG_OP 定义为单 OUTPUT）
    const aclTensor* yOut = (y != nullptr && y->Size() > 0) ? (*y)[0] : nullptr;

    // split_item=0/1: 构造 output_addrs 属性，携带 E 个输出 tensor 的设备地址
    // TilingFunc 读取此属性并写入 TilingData，Kernel 从 TilingData 读取
    const aclIntArray* outputAddrsArray = nullptr;
    std::vector<int64_t> outputAddrsVec;
    if (split_item <= 1 && y != nullptr && y->Size() > 0) {
        outputAddrsVec.reserve(y->Size());
        for (uint64_t i = 0; i < y->Size(); i++) {
            const aclTensor* yTensor = (*y)[i];
            if (yTensor == nullptr) {
                outputAddrsVec.push_back(0);
            } else {
                outputAddrsVec.push_back(
                    static_cast<int64_t>(reinterpret_cast<uintptr_t>(yTensor->GetStorageAddr())));
            }
        }
        outputAddrsArray = executor->AllocIntArray(outputAddrsVec.data(), outputAddrsVec.size());
    } else {
        // split_item=2/3: 传空数组占位
        outputAddrsArray = executor->AllocIntArray(nullptr, 0);
    }

    // ADD_TO_LAUNCHER_LIST_AICORE 调度 Kernel
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessGroupedMatmul,
        OP_INPUT(x, weight, bias), OP_OUTPUT(yOut),
        OP_ATTR(group_list, split_item, transpose_weight, outputAddrsArray));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessGroupedMatmulAiCore failed."),
             return nullptr);
    return y;
}
```

**TilingData**（把 workspace 偏移方案换成内嵌地址数组）：

```cpp
    // ===== 多核分组调度（内嵌数组，E ≤ 128）=====
    GroupSchedule groupSchedule[128];  // 每组的调度信息

    // ===== 多输出地址（split_item=0/1 时有效）=====
    // E 个输出 tensor 的 GM 设备地址
    // 由 L0 API 通过属性传递，TilingFunc 写入 TilingData
    // split_item=2/3: 不使用（单 tensor 连续输出）
    uint64_t outputAddrs[128];  // 输出 tensor GM 地址数组
    int64_t outputAddrCount = 0;  // 有效地址数量（= E for split_item=0/1, = 0 otherwise）
```

**TilingFunc**（读第 4 属性填 TilingData；workspace 恢复为仅系统 workspace）：

```cpp
    // split_item=0/1: 从属性读取 E 个输出 tensor 的设备地址，写入 TilingData
    // 属性 index 3: output_addrs (ListInt)，由 L0 API 传入
    if (splitItem <= 1) {
        auto tilingAttrs = context->GetAttrs();
        if (tilingAttrs != nullptr) {
            auto outputAddrsVec = tilingAttrs->GetListInt(3);  // 第 4 个属性
            if (outputAddrsVec != nullptr && outputAddrsVec->GetSize() == static_cast<size_t>(E)) {
                const int64_t* addrData = outputAddrsVec->GetData();
                for (int64_t g = 0; g < E; g++) {
                    tiling->outputAddrs[g] = static_cast<uint64_t>(addrData[g]);
                }
                tiling->outputAddrCount = E;
            } else {
                OP_LOGW(context, "split_item<=1 but output_addrs attribute missing or size mismatch");
                tiling->outputAddrCount = 0;
            }
        } else {
            tiling->outputAddrCount = 0;
        }
    } else {
        tiling->outputAddrCount = 0;
    }

    // 7. 设置 BlockDim 和 WorkspaceSize
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // Workspace: 仅需 Matmul API 系统 workspace
    // DAV_3510 上 GetSysWorkSpacePtr() 返回独立的 KFC workspace，与用户 workspace 分离
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    uint32_t sysWsSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = static_cast<size_t>(sysWsSize);
```

**kernel Process**（修复后按组取地址；修复前两条分支都是 `mm.IterateAll(yGm[gs.yOffset])`，仅注释不同）：

```cpp
        // 执行矩阵乘，输出到 GM
        if (tilingData_->splitItem <= 1 && tilingData_->outputAddrCount > 0) {
            // split_item=0/1: 多输出路径
            // 从 TilingData 读取第 g 个输出 tensor 的 GM 地址，创建独立 GlobalTensor
            uint64_t yGmAddr = tilingData_->outputAddrs[g];
            GlobalTensor<D_T_X> yGmOut;
            yGmOut.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(yGmAddr));
            mm.IterateAll(yGmOut);
        } else {
            // split_item=2/3: 单 tensor 连续输出，yOffset = start_g * N
            mm.IterateAll(yGm[gs.yOffset]);
        }
```

L2 API（`aclnn_bench_no_harness_grouped_matmul.cpp`）不需要任何改动：`Execute` 直接 `return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);`，全部地址传递都在 L0 API 一次完成。

## 设计决策

- 为什么选属性通道而不是 workspace：先试了"GetWorkspaceSize 阶段写 executor host workspace"，`GetWorkspaceAddr()` 返回 null 直接报 561103（见 [host workspace 负结果卡](../compilation/561103_host_workspace_null_getworkspacesize.md)）；再试"全局变量暂存 + Execute 里 `aclrtMemcpy` 预写 device workspace"，memcpy 返回 0 但 kernel 读不到（流同步失败 507015，见 [device workspace 预写负结果卡](../compilation/507015_device_workspace_prewrite_invisible.md)）。两条 workspace 路线都客观失败后，属性通道是唯一不依赖 workspace 分配时机的框架标准通道——`OP_ATTR` 的数据由框架负责搬运到 tiling/kernel 可见的位置。
- 放弃的候选：改 REG_OP 为多 OUTPUT（输出个数 E 动态，静态声明不可行）；TensorList 整体作为 OP_OUTPUT（launcher 宏不按 TensorList 语义展开，kernel 签名里 y 仍是单 GM_ADDR）；TilingFunc 里直接取输出 tensor 地址（gert::TilingContext 只有 GetOutputShape，没有输出 tensor 地址 API）。
- 地址以 `int64_t` 装进 ListInt 属性（device 地址 `uintptr_t` 强转），kernel 侧再 `static_cast<uint64_t>` 还原——数值在 host/device 两侧一致，只是位模式搬运。
- split_item=2/3 时仍传 `AllocIntArray(nullptr, 0)` 空数组占位，保持属性个数恒为 4，避免按 index 读取错位。
- 属性缺失/大小不匹配时 TilingFunc 走 `outputAddrCount = 0` + OP_LOGW 警告回退单输出路径（不是 GRAPH_FAILED），保证图模式等未注入属性的场景不炸。
- 当前方案属 recommended_fastpath：对本轨迹的 ACLNN 直调形态是标准做法。

## 预防

- 写"单 OUTPUT + 多输出 TensorList"型 aclnn 算子前，先确认 kernel 能拿到每个输出的 GM 地址；拿不到就是本卡场景。
- review 检查点：L0 API 的 `OP_OUTPUT(...)` 是否只传了 `(*y)[0]`；kernel 输出分支是否对 split/分组模式真正区分了基址。
- 属性按 `OP_ATTR` 声明顺序索引读取（index 0/1/2/3），新增属性必须两端（L0 API 注入 + TilingFunc 读取）同步，UT 侧 faker 也要按名补该属性。

## 验证信号

来源说明：以下信号来自该算子迭代二开发会话——多输出精度修复落地后重编重装算子包、在 NPU 上以 C++ 原生 ST（Real 模式、逐用例 fork）实测得到；此前同套用例 14 条 split_item=0/1 失败。

- 属性传址修复后 L0 全量 37/37 通过，其中多输出用例（split_item=1 + 转置 + bias，fp16，E=4）四个输出全部达标：输出[0] MERE=5.65e-07, MARE=8.31e-04；输出[1] MERE=3.55e-07, MARE=8.42e-04；输出[2] MERE=8.67e-07, MARE=9.09e-04；输出[3] MERE=8.33e-07, MARE=8.48e-04（阈值 9.77e-04）（来源平台：950；证据等级：tool_observed）。
- 首条多输出用例（split_item=0, E=4, fp16）四个输出精度全部达标（来源平台：950；证据等级：tool_observed）。
- 迭代一回归子集 13/13 通过，无回归；后续迭代二联调 154/154（含 62 条 split_item=0、8 条 split_item=1）与 L0 37/37 全过，验收累计 191/191（来源平台：950；证据等级：tool_observed）。
- 迭代三全量 687/687（L0 37 + L1 650，含全部多输出与空组多输出组合）通过（来源平台：950；证据等级：tool_observed）。

## 性能影响

无性能故事：本修复只改地址传递路径（每用例多一次 host 侧 E 元素数组构造与 tiling 拷贝），轨迹内无前后 profiling 对比，未验证性能影响。

## 失效边界

- E ≤ 128：TilingData 内嵌 `outputAddrs[128]`，超出需改动态方案（未验证）。
- 属性缺失或 `GetSize() != E` 时回退单输出路径并打警告——此时 split_item=0/1 的多输出语义实际是错的，调用方必须保证 L0 API 正常注入。
- graph 模式未验证（第 4 属性未在 def.cpp/REG_OP 声明，仅 ACLNN 直调路径注入）。
- 其他 NPU 平台未验证，推断可迁移（属性/tiling 机制与平台无关，但 DAV_3510 的 KFC workspace 分离行为是本卡背景之一）。
- 地址经 int64 属性位搬运，隐含 device 地址 < 2^63 的假设（当前 NPU 地址空间满足）。

**未决项**：

- `CommonOpExecutorRun` 内部对 device workspace 的确切搬运行为未追到框架源码层（只从 507015 失败信号推断"预写被覆盖/不可见"），复现并追根因到机制层列为下个 agent 的优先项。
- UT 侧已覆盖 output_addrs 属性缺失/大小不匹配的 tiling 回退分支，但 kernel 侧"outputAddrCount=0 时多输出用例的错误输出形态"没有专门断言（ST 侧靠精度比对兜底）。
## 坏实践

- 把 kernel 两条输出分支写成同样代码、只靠注释区分（修复前的状态）——编译器不会提醒，测试一跑多输出就现形。
- 依赖 executor host workspace 在 GetWorkspaceSize 阶段可写（561103）或在 Execute 阶段预写 device workspace（507015）——见同批两张负结果卡，勿再走。

## 误诊

无（本根因一次定位；两次失败尝试是方案探索而非误诊，已单独立卡）。
