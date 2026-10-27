---
type: Runbook
title: "事件 id 预算 QUE_MAX_EVENT=8：每 HardEvent 池、每核，超额不报错直接挂死"
description: "arch35 上每核每个 HardEvent 有一个独立 id 池，池深 QUE_MAX_EVENT=8；ops_transformer 的一个 Buffer<>::Init<INTERNAL>() 一次吃掉两个 id（EventP2C + EventC2P），池按 HardEvent 而非 BufferType 划分（L0A/L0B/C2 共用 MTE1_M/M_MTE1）。超额时 AllocEventID 的边界检查是 ASCENDC_ASSERT，device 构建下被定义为空，于是返回越界 id=8，SetFlag/WaitFlag 等一个永不到来的事件，kernel 100% util 挂死、无 error code、无编译告警。判别=刚加完一个 buffer 就挂死先查事件预算再查数据通路，按 HardEvent 池数已有 buffer 对照 8。修复=只要地址范围时用 BufferManager::AllocBuffer+GetTensor 不调 Init，不分配任何事件。"
tags:
- task.debugging
- topic.implementation.synchronization
- topic.implementation.pipeline
- topic.memory
- topic.programming_model.execution_unit.cube
status: draft
sources: []
created_at: '2026-10-10T02:26:24Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases:
- QUE_MAX_EVENT
- AllocEventID 耗尽
- event id budget
- 我加了一个 buffer 之后 kernel 卡死
- kernel hang after adding L1 buffer
- added a buffer and the kernel hangs
- buffer 挂死 kernel hang
- L1 buffer 数量上限 event id 分配
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。

# 事件 id 预算 QUE_MAX_EVENT=8：每 HardEvent 池、每核，超额不报错直接挂死

## 现象

Ascend950DT / arch35 上的 MXFP8 FA 反向 kernel，在 AIC 侧加入第 9 个 L1 `Buffer<>` 之后 kernel 不再返回：

- NPU 利用率停在 100%，最终被超时杀掉，`rc=124`；
- **没有 error code，没有 plog 报错，没有编译期告警，没有运行期断言**；
- 同一 kernel 退到 5 个 buffer 时正常跑完，`rc=20`。

失败信号与数据内容、shape、dtype 无关，只随 buffer 个数出现——这是它和普通数据通路 bug 的第一个分界点。

## 适用场景

- **工程形态**：`ops_transformer` attention 系的 `fa_base_matmul::Buffer<>` / `BuffersPolicySingleBuffer` / `BuffersPolicyDB` 封装；以及任何直接用 `TPipe::AllocEventID<HardEvent>()` 管理核内同步 id 的 kernel。
- **平台与版本**：Ascend950DT，arch35（`__NPU_ARCH__ == 3510`）。本卡的常量与源码行读自本机 CANN **9.1.0-beta.1** 头文件；运行期 bisect 在 **9.2.0** 上完成（版本边界见 `## 失效边界`）。
- **适用粒度**：`platform_wide`——arch35 上任何在单核内持有多个同向 `Buffer<>` 或手工 `AllocEventID` 的 kernel。
- **不适用**：AIV 侧池集合不同（见 `## 失效边界`）；非 ops_transformer 的 buffer 封装不适用"每 buffer 2 个 id"这条计数规则，但池深规则仍适用。

## 根因

四条事实叠起来就是这个静默挂死：

**1. 池深 = 8，且这个 8 是条件编译出来的。**

`asc/impl/basic_api/kernel_macros.h:111-117`：

```cpp
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201) ||           \
    (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510) ||           \
    (__NPU_ARCH__ == 5102)) // Available for V200 and V210
constexpr int32_t QUE_MAX_EVENT = 8;
#else
constexpr int32_t QUE_MAX_EVENT = 4;
#endif
```

arch35 → `__NPU_ARCH__ == 3510` → 命中 8。**不在这个列表里的目标预算是 4，不是 8。**
（常量定义在 `kernel_macros.h`，不在 `tikcpp/tikcfw/impl/kernel_tpipe_impl.h`——后者只是指向 `asc/impl/basic_api/kernel_tpipe_impl.h` 的符号链接，且只**使用**该常量。`tikcfw/` 整个是符号链接农场，`grep -r` 默认不跟进，查这个常量要用 `grep -R` 或直接 grep 解析后的 `asc/` 实体路径。）

**2. 一个 `Buffer<>::Init<INTERNAL>()` 吃掉两个 id，分属两个不同的池。**

`attention/common/op_kernel/attn_buffer.h:195-206`：

```cpp
template <IdSource idSource = IdSource::INTERNAL>
__aicore__ inline void Init()
{
    if ASCEND_IS_AIC {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC && idSource == IdSource::INTERNAL) {
            if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>();  // :201
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();  // :202
                SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
```

注意三个前置条件：只有 `ASCEND_IS_AIC`、`INNER_CORE_SYNC`、`SET_WAIT_FLAG` 这条路径才分配。跨核同步（`CROSS_CORE_SYNC_*`）和 `LOCK_UNLOCK` 走别的资源，不占事件池。

**3. 池按 HardEvent 分，不是按 BufferType 分。**

`attn_buffer.h:68-110`（`BufferInfo<Type>::ConsWaitProdStatus()` / `ProdWaitConsStatus()`）：

| BufferType | EventP2C 池 | EventC2P 池 |
|---|---|---|
| L1 | `MTE2_MTE1` | `MTE1_MTE2` |
| L0A | `MTE1_M` | `M_MTE1` |
| L0B | `MTE1_M` | `M_MTE1` |
| C2 | `MTE1_M` | `M_MTE1` |
| L0C | `M_FIX` | `FIX_M` |
| GM | `MTE2_S` | `S_MTE2` |

**L0A、L0B、C2 三者共用同一对池。** 按 BufferType 分别数到 8 会漏算——这是计数时最容易犯的错。
（`kernel_event.h:233-243` 的 `EventToIndex` 在 arch35 上走 `EventToIndexAic`/`EventToIndexAiv` 的紧凑索引空间，每个 HardEvent 一个独立 `TEventPool`。）

**4. 超额不报错，因为唯一的检查在 device 构建里被编掉了。**

`asc/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h:438-450`：

```cpp
template <HardEvent evt> __aicore__ inline TEventID TPipe::AllocEventID()
{
    auto ptr = this->g_tpipeImpl.eventPool_ + EventToIndex(evt);
    auto lastId = sff0(ptr->eventOccupy);           // scan-for-first-zero
    ASCENDC_ASSERT((lastId < QUE_MAX_EVENT && lastId >= 0), { ... });   // 唯一的边界检查
    ptr->eventOccupy = sbitset1(ptr->eventOccupy, lastId);
    return lastId;                                   // 没有返回值校验
}
```

而 `asc/impl/basic_api/kernel_log.h:64-69`，在 `__NPU_DEVICE__` / `__ASCC_DEVICE__`（即真实 device 构建）下：

```cpp
#ifndef ASCC_ASCENDC_ASSERT
#define ASCC_ASCENDC_ASSERT
#define ASCENDC_ASSERT(cond, behavior)      // 空
#endif
```

于是在 device 上：`eventOccupy` 是 `uint64_t`（`kernel_tpipe_base.h:97`），低 8 位占满后 `sff0` 返回 **8**，`sbitset1` 照常把第 8 位置上，函数返回 id=8 —— 落在硬件合法区间 `[0, 8)` **之外**。随后的 `SetFlag<EVT>(8)` / `WaitFlag<EVT>(8)` 在等一个硬件上不存在的事件，核就停在那里，利用率 100%。

**没有任何一层会告诉你。** 编译期不知道运行期的分配序列，运行期的断言被编掉了，返回值没人校验。

## 判别方法

**触发条件**：加完一个 buffer（或一个带 `Init()` 的 buffer 策略）之后 kernel 立刻挂死，100% util，无 error code。

**可执行探针**：

1. **先问事件预算，再问数据通路。** 这是本卡最可迁移的一条：*挂死紧跟在"新增 buffer"之后* 这个时序本身就是信号。先花两分钟数池，再去查 DataCopy 的 stride 和 tiling。
2. **按 HardEvent 池数，不按 BufferType 数。** 对着上面的映射表，把该核上所有走 `INNER_CORE_SYNC + SET_WAIT_FLAG + AIC` 的 `Buffer<>` 按池归类：
   - 每个 `BuffersPolicySingleBuffer` 贡献 1 个 buffer = 每池 1 个 id；
   - 每个 `BuffersPolicyDB` 贡献 2 个 buffer = 每池 2 个 id（`buffers_policy.h:72-81`）；
   - L0A 与 L0B 的计数要**加在一起**。
   任何一个池的总数 > 8 就是本卡。
3. **二分 buffer 数反向证明。** 砍到明显安全的数量（如 5）跑通，再加回去，确认挂死随 buffer 数单调出现、与功能改动无关。本次即用两组不相关特性各关一次做了两遍，结论一致。

**原始观测**（Ascend950DT / arch35 / CANN 9.2.0，MXFP8 FA 反向）：

- 5 个 L1 buffer：`rc=20`，kernel 跑完；
- 9 个 L1 buffer：`rc=124`，100% util 挂死；关掉两组互不相关的特性后各复现一次，共两次；
- 预测写在测量之前，三次全中。

**计数核对**：挂死时该核的 8 个 L1 buffer = 4 个单 buffer（不变量槽）+ 2 组 ping/pong，正好把 `MTE2_MTE1` 与 `MTE1_MTE2` 两个池各占满 8 个 id（0..7）。第 9 个 buffer 的 `Init()` 必然拿到越界 id。

## 解决配方

按代价从低到高：

**A. 只要地址范围，就不要调 `Init()`。**（本次采用）

第 9 个 buffer 的 `Wait()` / `Set()` 从未被调用过——它纯粹是一段地址。`BufferManager::AllocBuffer<>(size)`（`attn_buffer_manager.h:57-63`）只做地址算术，**不碰事件池**：

```cpp
template <SyncType syncType = ..., SyncMode syncMode = ...>
__aicore__ inline Buffer<bufferType, syncType, syncMode> AllocBuffer(uint32_t size)
{
    TensorType temp = mem_[offset_];     // :59
    offset_ += size;
    return Buffer<bufferType, syncType, syncMode>(temp, size);
}
```

配 `Buffer::GetTensor()`（`attn_buffer.h:443`）取 tensor 即可。

**关键陷阱**：不能用 `BuffersPolicySingleBuffer` 来做这件事——它的 `Init` 在 `AllocBuffer` 之后**紧跟**一次 `buffer_.Init<idSource>()`（`buffers_policy.h:29-37`），事件照样被分配。要直接持有 `BufferManager` 并调 `AllocBuffer`。

**B. 共享一个 id。**

`Buffer::Init<IdSource::EXTERNAL>(uint32_t id)`（`attn_buffer.h:214-231`）不分配，直接用调用方给的 id；`BuffersPolicySingleBuffer<..., IdSource::EXTERNAL>` 会把 `id` 透传下去。适用于两个生命周期不重叠、可以安全共用一个事件的 buffer。

**C. 及时归还。**

`UnInit<IdSource::INTERNAL>()`（`attn_buffer.h:234-245`）会调 `ReleaseEventID` 把两个 id 还回池。若 buffer 是分阶段使用的，阶段结束就 `UnInit`，预算即可复用。
（注意 `BufferManager::FreeBuffer` 是**空实现**，只还事件、不回收地址 `offset_`。）

## 预防

- 加 buffer 之前先数池，不要加完再调试。一个 `BuffersPolicyDB` 是 2 个 id，不是 1 个。
- 把 L0A 与 L0B 的 buffer 数加在一起对照 8。
- 需要的只是一块地址时，默认走 `AllocBuffer` + `GetTensor`，不要顺手套 `BuffersPolicySingleBuffer`——后者的 `Init` 会悄悄花掉两个 id。
- 不要指望断言。device 构建下 `ASCENDC_ASSERT` 是空的；只有 CPU 孪生调试（`ASCENDC_CPU_DEBUG=1`）或定义 `ASCENDC_DEBUG` 才会触发这条越界检查——**这也给出一条排查手段：把同一 kernel 放到 CPU 孪生调试下跑，越界会直接报出来。**

## 误诊

挂死且无 error code 时，第一反应通常是去查 DataCopy 的地址/stride、tiling 边界或跨核握手。本次挂死紧跟在"加了一个 buffer"之后，这个时序比任何数据通路假设都更有信息量。**挂死 ≠ 数据错**：事件预算问题不会污染数据，它只是让核停下。

## 验证信号

- 5 buffer：`rc=20`，kernel 跑完（来源平台：950；证据等级：tool_observed）
- 9 buffer：`rc=124`，100% util 挂死（来源平台：950；证据等级：tool_observed）
- 9 buffer 挂死在两组分别关闭不相关特性的配置下各复现一次，共两次（来源平台：950；证据等级：tool_observed）
- 改用 `AllocBuffer` + `GetTensor`（不调 `Init`）后 kernel 恢复正常返回（来源平台：950；证据等级：tool_observed）

## 失效边界

- **精确阈值是推断，不是实测。** "8 通过 / 9 挂死"由 `QUE_MAX_EVENT=8` 与"每 buffer 2 个 id、分属两池"推出；实测只把阈值夹在 `(5, 9]` 区间，6/7/8 未逐个测。
- **版本**：常量、宏展开与源码行号读自本机 CANN 9.1.0-beta.1（`kernel_macros.h` sha256 `3e48fb57…`，`dav_3510/kernel_tpipe_impl_c310.h` sha256 `c95631256e…`）；运行期 bisect 在 CANN 9.2.0 上。两个版本的 `QUE_MAX_EVENT` 与断言展开**未逐字节比对**。
- **ops_transformer 侧行号**来自 `ops-transformer` master checkout commit `1461cca857ff3b2018404c0275e80e7aa13a3fc9`（`attn_buffer.h` sha256 `9d433f48f0…`），**不是**知识仓 Registry 登记的固定提交 `be33d95b…`；该固定提交的源码树本机未恢复，行号未在登记版本上核对。
- **`__NPU_ARCH__` 不在 {2002,2201,3002,3102,3510,5102} 时预算是 4**，本卡的全部计数要减半。a2/a3 未验证。
- **AIV 侧未覆盖**：`kernel_event.h:198-230` 的 `EventToIndexAiv` 是另一组池（13 个事件），本卡的映射表只覆盖 AIC（`EventToIndexAic`，24 个事件）。
- 本卡不涉及 `TPipe::Reset()` 对事件池的影响；Reset 会释放**整个全局池**，那是另一个失败面，见下方相关卡片。

## 相关卡片

- [AllocEventID](../../apis/simd_api/basic_api/resource_management/t_pipe/alloc_event_id.md) —— 官方 API 卡，也是"事件 id 怎么分配"最容易落到的那张。它的"约束说明"只写了"TEventID 有数量限制……防止 TEventID 耗尽"，**没有数字 8、没有说池按 HardEvent 划分、也没有说耗尽后是静默挂死**。只读那张卡会以为自己已经读过权威定义，从而把挂死归到别处——本卡补的正是这三点。
- [PB-45 `TPipe::Reset()` 释放全局 `g_tpipeImpl` 事件池](../compilation/pb_45_tpipe_reset_frees_the_global_g_tpipeimpl_event_poo.md) —— 同一资源的**生命周期**失败面（Reset 把整个全局池释放），本卡是**容量**失败面。漏掉它，带 `Reset()` 的多阶段 MIX kernel 即使事件预算没超也会挂死，按本卡数池会得出"没超额，不是这个问题"的错误结论；该卡还给出 `507015` 这一条本卡不覆盖的失败形态。
- [Reduce work buf 额外占 UB 致 507035](../compilation/reduce_work_buf_ub_507035.md) —— **同一套方法用在另一种资源上**：逐 `InitBuffer` 累加字节对照 UB 上限 ⇄ 逐 `Buffer<>` 累加事件 id 对照 `QUE_MAX_EVENT`。两者共同形态是"片上资源静态划分超额、编译期不报、运行期才炸"；**区别是 UB 超额给 error code 507035，事件 id 超额什么都不给**。漏掉这条不对称，就会因为"没看到 507035"而排除资源超额这一整类。

<!-- 本卡原 draft 另链 ol_223（Reset-safe FetchEventID）与 catlass_resource_memory_model_event_ids；按「每卡至多 2-3 条相关链接」裁到 3 条，两者经 PB-45 仍可达。 -->
