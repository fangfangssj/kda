---
type: "Runbook"
title: "arch35 SyncAll 无 HardEvent 模板形式"
description: "CANN 9.1.0 arch35（Ascend950）上 kernel 入口写 AscendC::SyncAll<HardEvent::MTE3_MTE2>() 编译报 no matching function——该平台 SyncAll 是 bool isAIVOnly 模板参数形式或需要 GlobalTensor gmWorkspace + LocalTensor 同步缓冲的多参形式；骨架期移除 SyncAll 并将 tiling 置单核规避。"
tags: [task.implementation, topic.compilation, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# arch35 SyncAll 无 HardEvent 模板形式

## 现象

BenchNoHarnessUnique 算子（radix sort 去重）迭代一 kernel 入口按常见的跨核同步写法，在多核阶段之间插入事件同步：

```cpp
AscendC::SyncAll<AscendC::HardEvent::MTE3_MTE2>();
```

面向 ascend950 编译（`bash build.sh --soc=ascend950 -j8`）时，kernel 入口文件两处调用（Phase 1 后、每趟排序后）报错，报错原文：

```
op_kernel/bench_no_harness_unique_arch35.cpp:41:5: error: no matching function for call to 'SyncAll'
    AscendC::SyncAll<AscendC::HardEvent::MTE3_MTE2>();
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
note: candidate template ignored: invalid explicitly-specified argument for template parameter 'isAIVOnly'
__aicore__ inline void SyncAll()
                       ^
note: candidate function template not viable: requires at least 2 arguments, but 0 were provided
__aicore__ inline void SyncAll(const GlobalTensor<int32_t> &gmWorkspace,
```

查 CANN 头文件（`impl/basic_api/kernel_operator_block_sync_intf_impl.h`）后确认：arch35 的 `SyncAll` 是 `template <bool isAIVOnly>` 的无参形式（模板参数是布尔而不是 HardEvent），以及带 `const GlobalTensor<int32_t> &gmWorkspace, const LocalTensor<int32_t> &ub, ...` 的多参形式——跨核同步需要 GM workspace 与 UB 同步缓冲，不存在 `SyncAll<HardEvent::MTE3_MTE2>()` 这种单模板参数形式。

## 适用场景

- 工程形态：AscendC kernel 开发，多核（多 block）kernel 的阶段间跨核同步。
- API 路径：`AscendC::SyncAll`（`x86_64-linux/asc/impl/basic_api/kernel_operator_block_sync_intf_impl.h`）；tiling 侧 `context->SetBlockDim(usedCoreNum)`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510）、CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该平台 + 版本上所有多核 kernel 的同步写法。
- 路线优先级：骨架期 recommended_fastpath（单核规避）；多核化为性能需求时的前置知识点。
- 明确不适用 / 未验证场景：A2/A3 等其他架构的 SyncAll 形态；arch35 多参 SyncAll（gmWorkspace/UB 布局与生命周期）在本轨迹未实际使用与验证。

## 根因

CANN 9.1.0 arch35 的 `SyncAll` 签名是 `template <bool isAIVOnly>` 的无参形式或要求 `GlobalTensor<int32_t> gmWorkspace + LocalTensor<int32_t> ub` 同步缓冲的多参形式，不存在 `SyncAll<HardEvent::MTE3_MTE2>()` 单模板参数形式（显式指定的 HardEvent 实参与 `isAIVOnly` 布尔模板参数不匹配）。

## 解决配方

迭代一按"骨架优先"规避：kernel 入口移除全部 SyncAll 调用，tiling 侧把核数固定为 1（排序与去重阶段本来就只由 core 0 执行，单核下无跨核同步需求），编译通过。

修复前（kernel 入口 v1，Phase 1 多核后同步 + 每趟排序同步）：

```cpp
template <typename D_T_X, int HAS_INVERSE>
__global__ __aicore__ void bench_no_harness_unique(
    GM_ADDR x, GM_ADDR y, GM_ADDR inverse, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessUniqueTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessUniqueTilingData, tilingData, tiling);

    NsBenchNoHarnessUnique::BenchNoHarnessUnique<D_T_X, HAS_INVERSE> op;
    op.Init(x, y, inverse, workspace, &tilingData);

    // Phase 1: 键编码（多核并行）
    op.EncodeKeys();
    AscendC::SyncAll<AscendC::HardEvent::MTE3_MTE2>();

    // Phase 2: Radix 排序（多趟，迭代一仅 core 0 执行）
    for (uint32_t passIdx = 0; passIdx < tilingData.sortPasses; passIdx++) {
        op.RadixSortPass(passIdx);
        AscendC::SyncAll<AscendC::HardEvent::MTE3_MTE2>();
    }

    // Phase 3: 去重紧凑 + inverse 生成（迭代一仅 core 0 执行）
    op.DedupAndInverse();
}
```

修复后（kernel 入口 v2，移除同步；tiling 同步改单核）：

```cpp
template <typename D_T_X, int HAS_INVERSE>
__global__ __aicore__ void bench_no_harness_unique(
    GM_ADDR x, GM_ADDR y, GM_ADDR inverse, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessUniqueTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessUniqueTilingData, tilingData, tiling);

    NsBenchNoHarnessUnique::BenchNoHarnessUnique<D_T_X, HAS_INVERSE> op;
    op.Init(x, y, inverse, workspace, &tilingData);

    // Phase 1: 键编码（多核并行）
    op.EncodeKeys();

    // Phase 2: Radix 排序（迭代一仅 core 0 执行）
    op.RadixSort();

    // Phase 3: 去重紧凑 + inverse 生成（迭代一仅 core 0 执行）
    op.DedupAndInverse();
}
```

配套 tiling 修改（多核切分处固定单核）：

```cpp
    // 5. 多核切分
    // 迭代一简化：单核执行（blockDim=1），避免跨核同步复杂性
    // 后续迭代将启用多核 Phase 1 并行 + SyncAll 同步
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t blockFactor = CeilAlign(totalNum, ubBlockSize);
    int64_t usedCoreNum = 1;
```

arch35 正确签名（头文件 grep 所得，供多核化时使用，本轨迹未验证）：

```cpp
template <bool isAIVOnly>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t> &gmWorkspace,
                               const LocalTensor<int32_t> &ub, ...)
```

即多核同步需要：GM 侧 int32 workspace（同步计数/标志）+ UB 侧同步缓冲，且以 `bool isAIVOnly` 作模板参数。

## 设计决策

- 为什么移除 SyncAll 而不是改写成新签名：迭代一排序/去重阶段只由 core 0 执行（其余核在 Phase 1 后自然退出），跨核同步在单核语义下是冗余调用；直接实现多参 SyncAll 需要设计 GM workspace 布局与 UB 同步缓冲的分配，超出骨架迭代目标。移除 + 单核是最小正确改动。
- 单核化的代价：Phase 1 键编码失去多核并行（后续迭代的性能接续点），`usedCoreNum` 恢复多核时必须同步补上正确形式的 SyncAll，否则排序阶段会读到未写完的 keysGM。
- 当前方案路线档位：骨架期 recommended_fastpath；多核化为性能迭代时，本卡记录的签名差异是前置知识。

## 系统信息

- 硬件/软件：Ascend950PR（arch35 / DAV_3510）、CANN 9.1.0、ccec kernel 编译。
- 相关文件：`op_kernel/bench_no_harness_unique_arch35.cpp`（入口，v1→v2）、`op_host/arch35/bench_no_harness_unique_tiling.cpp`（usedCoreNum=1）、`op_kernel/arch35/bench_no_harness_unique.h`（单核 `if (coreId_ != 0) return;` 守卫）。

## 预防

- 写跨核同步前先 grep 目标平台的 `kernel_operator_block_sync_intf_impl.h` 确认 SyncAll 模板签名，不从旧代码/旧文档照抄 `SyncAll<HardEvent::...>()` 写法到 arch35 工程。

## 验证信号

以下信号来自该算子骨架的编译闭环（来源说明：同会话两轮构建输出）：

- 修复前：`AscendC::SyncAll<AscendC::HardEvent::MTE3_MTE2>()` 编译报 `error: no matching function for call to 'SyncAll'`，note 原文 `invalid explicitly-specified argument for template parameter 'isAIVOnly'` 与 `requires at least 2 arguments, but 0 were provided`（候选为 `SyncAll(const GlobalTensor<int32_t> &gmWorkspace, ...)`）（来源平台：950；证据等级：tool_observed）。
- 根因核对：CANN 头文件 grep 确认 arch35 SyncAll 为 `template <bool isAIVOnly>` + `gmWorkspace/ub` 多参形式（来源平台：950；证据等级：tool_observed）。
- 修复后：kernel 编译通过，生成 kernel 二进制与算子包（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 签名结论限于 CANN 9.1.0 arch35 头文件；其他版本/架构以各自 `kernel_operator_block_sync_intf_impl.h` 为准（未验证，机制推断可迁移）。
- 多参 SyncAll 的运行行为（gmWorkspace/UB 布局、isAIVOnly 取值语义）本轨迹未验证，仅为头文件事实。
- 单核规避只对"阶段间同步可省略"的骨架成立；任何真正的多核数据依赖（Phase 1 多核写 → Phase 2 读）都必须补正确形式的同步，否则是正确性缺陷而非性能问题。
- 本卡未覆盖 arch35 上 `SyncAll` 的等价替代（如 SetFlag/WaitFlag 事件同步）在该场景的适用性。

**未决项**：

- arch35 多核 SyncAll 的正确用法（gmWorkspace 布局、UB 同步缓冲分配、isAIVOnly 语义）未验证——多核化迭代的优先接续点。
## 坏实践

- 从旧代码/旧文档照抄 `SyncAll<HardEvent::MTE3_MTE2>()` 写法到 arch35 工程。
- 为"过编译"把同步删掉但 tiling 仍配多核——多核写、单核读之间没有屏障，正确性不保（本轨迹配套把 usedCoreNum 固定为 1，避免了这一陷阱）。

## 误诊

- 无（报错 note 直接给出 isAIVOnly 模板参数与两参候选，一次定位）。
