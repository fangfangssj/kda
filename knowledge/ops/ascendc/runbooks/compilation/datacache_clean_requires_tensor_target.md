---
type: "Runbook"
title: "DataCacheCleanAndInvalid 无参调用导致编译失败"
description: "在同一 SIMT 函数内完成初始化与更新时，误把 DataCacheCleanAndInvalid 当作无参全缓存操作会触发接口匹配失败；删除无效调用并依靠同一执行域内的读写闭环。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api", "topic.implementation.synchronization"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DataCacheCleanAndInvalid 无参调用导致编译失败

## 现象

在构建一个 Ascend C 直接调用工程的 `operators/scatter/scatter.asc` 时，SIMT 函数已经把输出初始化和后续标量更新合并到同一个函数中。为了让后续执行者看见标量写回，函数尾部又增加了一个不带 Tensor 实参的“全 DCache 清理”调用。通过远端构建入口执行 CMake/Make 后，毕昇编译器在该行同时报告：

```text
error: no member named 'ENTIRE_DATA_CACHE' in 'AscendC::HardEvent'
error: no matching function for call to 'DataCacheCleanAndInvalid'
```

编译器列出的可用候选都是接收 `LocalTensor<T>` 或 `GlobalTensor<T>` 的重载；当前写法既使用了不存在的 `HardEvent` 成员，也没有传入目标 Tensor。

## 适用场景

- 工程形态：Ascend C kernel 直调工程，SIMT 函数通过 GM 标量访问完成输出初始化和原地更新。
- API 路径：`__simt_vf__` 函数中的 `AscendC::DataCacheCleanAndInvalid`。
- 验证过的平台与版本：Ascend 950（实测设备名为 Ascend950PR 系列），CANN 9.0.1。
- 适用粒度：`same_operator`；证据只覆盖当前 Scatter 实现中“同一 SIMT 函数完成初始化与更新”的路线。
- 路线优先级：`recommended_fastpath`；这是该实现最终通过构建和功能探针的调用形态。
- 明确不适用或未验证场景：不同 kernel、不同流水阶段或不同缓存域之间确实需要显式可见性操作时，不能照搬“直接删除”；A2、A3 以及其他 CANN 版本未验证。

## 根因

根因是把 `DataCacheCleanAndInvalid` 错当成“用 `HardEvent` 模板参数选择整个缓存、无需目标 Tensor”的接口，而当前 CANN 头文件暴露的是以 `LocalTensor<T>` 或 `GlobalTensor<T>` 为目标的函数重载，因此模板枚举解析和实参数量同时不满足契约。

这里的原函数已经在同一 SIMT 执行域内先写入输出初值、再读取并更新同一 GM 输出；编译错误不是缺少另一个模板枚举拼写，而是调用形态本身不属于已暴露的 API 契约。

## 解决配方

先确认初始化与后续读改写是否在同一 SIMT 函数中完成。若是当前这种同一执行域闭环，不构造不存在的“全缓存”事件；删除无效调用，并用构建与上板结果验证该路线。若程序真正跨 kernel 或跨缓存域，则应重新选择有明确 Tensor 目标和同步语义的 API，不能把本配方外推为普遍免同步结论。

### before

```cpp
template <typename T, typename IdxT>
__simt_vf__ __aicore__ inline void scatter_simd(
    __gm__ const T *dataGm, __gm__ T *yGm, __gm__ const IdxT *idxGm, __gm__ const T *updGm,
    __gm__ const ScatterTilingData *t, __gm__ const int64_t *myStartPtr)
{
    if (threadIdx.x != 0) return;                 // single thread per core
    const int32_t dim = t->dim;
    const int32_t rank = t->rank;
    const int32_t reduceMode = t->reduceMode;
    const int64_t yNumel = t->yNumel;
    const int64_t dataDimSize = t->dataDimSize;
    const int64_t idxNumel = t->idxNumel;
    const int64_t outBlockLen = t->outBlockLen;
    const int64_t outStart = *myStartPtr;          // per-block start via GM pointer
    if (outStart >= yNumel) return;
    int64_t outEnd = outStart + outBlockLen;
    if (outEnd > yNumel) outEnd = yNumel;
    int64_t yStride[8];
    int64_t idxShape[8];
    for (int32_t d = 0; d < rank; ++d) { yStride[d] = t->yStride[d]; idxShape[d] = t->idxShape[d]; }

    // Phase 1: initialize this core's owned y range from data (SIMT scalar, same DCache
    // as the scatter writes below => coherent, no cross-kernel DMA/scalar stale issue).
    for (int64_t pos = outStart; pos < outEnd; ++pos) {
        yGm[pos] = dataGm[pos];
    }

    // Phase 2: scan ALL indices in gi order; for hits in this core's owned range,
    // overwrite (None) or read-modify-write (reduce). Single thread => deterministic,
    // last-highest-gi wins, matching torch CPU scatter_.
    for (int64_t gi = 0; gi < idxNumel; ++gi) {
        IdxT idxVal = idxGm[gi];
        if ((int64_t)idxVal < 0 || (int64_t)idxVal >= dataDimSize) continue;   // clamp (E15)
        uint32_t r = static_cast<uint32_t>(gi);
        int64_t dstOff = 0;
        for (int32_t d = rank - 1; d >= 0; --d) {
            uint32_t s = static_cast<uint32_t>(idxShape[d]);
            uint32_t coord = (s > 0) ? (r % s) : 0u;
            r = (s > 0) ? (r / s) : r;     // 32-bit division (dav-3510 supports 32-bit)
            int64_t c = (d == dim) ? (int64_t)idxVal : (int64_t)coord;
            dstOff += c * yStride[d];
        }
        if (dstOff < outStart || dstOff >= outEnd || dstOff >= yNumel) continue;  // not my range
        if (reduceMode == REDU_NONE) {
            yGm[dstOff] = updGm[gi];
        } else {
            T cur = yGm[dstOff];
            T upd = updGm[gi];
            T res;
            if (reduceMode == REDU_ADD) {
                res = cur + upd;
            } else if (reduceMode == REDU_MUL) {
                res = cur * upd;
            } else if (reduceMode == REDU_MIN) {
                res = (cur != cur) ? cur : ((upd != upd) ? upd : ((cur < upd) ? cur : upd));
            } else {  // REDU_MAX
                res = (cur != cur) ? cur : ((upd != upd) ? upd : ((cur < upd) ? upd : cur));
            }
            yGm[dstOff] = res;
        }
    }
    // Flush this core's DCache so the host/subsequent kernels see the scalar writes.
    AscendC::DataCacheCleanAndInvalid<AscendC::HardEvent::ENTIRE_DATA_CACHE>();
}
```

### after

```cpp
template <typename T, typename IdxT>
__simt_vf__ __aicore__ inline void scatter_simd(
    __gm__ const T *dataGm, __gm__ T *yGm, __gm__ const IdxT *idxGm, __gm__ const T *updGm,
    __gm__ const ScatterTilingData *t, __gm__ const int64_t *myStartPtr)
{
    if (threadIdx.x != 0) return;                 // single thread per core
    const int32_t dim = t->dim;
    const int32_t rank = t->rank;
    const int32_t reduceMode = t->reduceMode;
    const int64_t yNumel = t->yNumel;
    const int64_t dataDimSize = t->dataDimSize;
    const int64_t idxNumel = t->idxNumel;
    const int64_t outBlockLen = t->outBlockLen;
    const int64_t outStart = *myStartPtr;          // per-block start via GM pointer
    if (outStart >= yNumel) return;
    int64_t outEnd = outStart + outBlockLen;
    if (outEnd > yNumel) outEnd = yNumel;
    int64_t yStride[8];
    int64_t idxShape[8];
    for (int32_t d = 0; d < rank; ++d) { yStride[d] = t->yStride[d]; idxShape[d] = t->idxShape[d]; }

    // Phase 1: initialize this core's owned y range from data (SIMT scalar, same DCache
    // as the scatter writes below => coherent, no cross-kernel DMA/scalar stale issue).
    for (int64_t pos = outStart; pos < outEnd; ++pos) {
        yGm[pos] = dataGm[pos];
    }

    // Phase 2: scan ALL indices in gi order; for hits in this core's owned range,
    // overwrite (None) or read-modify-write (reduce). Single thread => deterministic,
    // last-highest-gi wins, matching torch CPU scatter_.
    for (int64_t gi = 0; gi < idxNumel; ++gi) {
        IdxT idxVal = idxGm[gi];
        if ((int64_t)idxVal < 0 || (int64_t)idxVal >= dataDimSize) continue;   // clamp (E15)
        uint32_t r = static_cast<uint32_t>(gi);
        int64_t dstOff = 0;
        for (int32_t d = rank - 1; d >= 0; --d) {
            uint32_t s = static_cast<uint32_t>(idxShape[d]);
            uint32_t coord = (s > 0) ? (r % s) : 0u;
            r = (s > 0) ? (r / s) : r;     // 32-bit division (dav-3510 supports 32-bit)
            int64_t c = (d == dim) ? (int64_t)idxVal : (int64_t)coord;
            dstOff += c * yStride[d];
        }
        if (dstOff < outStart || dstOff >= outEnd || dstOff >= yNumel) continue;  // not my range
        if (reduceMode == REDU_NONE) {
            yGm[dstOff] = updGm[gi];
        } else {
            T cur = yGm[dstOff];
            T upd = updGm[gi];
            T res;
            if (reduceMode == REDU_ADD) {
                res = cur + upd;
            } else if (reduceMode == REDU_MUL) {
                res = cur * upd;
            } else if (reduceMode == REDU_MIN) {
                res = (cur != cur) ? cur : ((upd != upd) ? upd : ((cur < upd) ? cur : upd));
            } else {  // REDU_MAX
                res = (cur != cur) ? cur : ((upd != upd) ? upd : ((cur < upd) ? upd : cur));
            }
            yGm[dstOff] = res;
        }
    }
}
```

## 设计决策

- 选择删除无效全缓存调用，是因为当前函数已经把输出初始化和标量更新放在同一 SIMT 执行域内；修复目标是遵守已暴露 API 契约，而不是猜测另一个事件枚举。
- 放弃“改一个枚举名继续无参调用”的候选，因为编译器同时表明重载需要 `LocalTensor<T>` 或 `GlobalTensor<T>` 实参，单改模板参数仍不满足签名。
- 当前方案标为 `recommended_fastpath`，仅表示它是这条同函数读写路线中最终通过验证的实现；不宣称跨 kernel 场景无需同步。

## 预防

- 遇到 `no matching function` 时同时核对候选重载的目标 Tensor 类型和参数个数，不只修正枚举命名。
- 在加入缓存清理前先画清执行边界：初始化、读取、更新是否位于同一 kernel、同一 SIMT 函数和同一缓存访问路径。
- 对跨 kernel 可见性单独做最小探针；不要把一个假想的全缓存 API 当作同步屏障。

## 验证信号

验证来源是删除无效调用后的同一次远端构建与上板功能探针；设备随后由运行时探测确认为 Ascend 950。

- 动态库完成链接，构建输出出现 `Built target op_scatter`，原有两个 `DataCacheCleanAndInvalid` 编译错误消失。（证据等级: tool_observed；来源平台: 950）
- 覆盖无规约、加法、乘法、最小值和最大值五类路径的两组规模探针全部 `match=True`，且每组比较的 `maxdiff=0.0`。（证据等级: tool_observed；来源平台: 950）
- 后续完整 PyTest 回归得到 35 项通过、6 项跳过；本卡不把跳过项计作通过。（证据等级: tool_observed；来源平台: 950）

## 失效边界

- 本结论只验证了 Ascend 950、CANN 9.0.1 和同一 SIMT 函数内的 GM 初始化/更新闭环。
- 若生产者与消费者位于不同 kernel、不同流水阶段或不同缓存域，应按对应 API 的目标 Tensor 和同步契约重新设计；“删除调用”未在这些场景验证。
- A2、A3 和其他 CANN 版本未验证，推断可迁移前必须重新编译并做上板可见性探针。

## 坏实践

- 凭枚举名臆造 `DataCacheCleanAndInvalid<HardEvent::...>()` 的无参形式。
- 只看到“缓存可见性”字样就加入全局 flush，而不检查函数签名与实际 kernel 边界。
- 把“删除后当前算例通过”外推成所有跨 kernel 场景都无需同步。

## 误诊

最初把问题理解为“还缺一个正确的全缓存事件枚举”。编译器候选重载表已经否定了这一点：该版本接口的关键契约是显式 Tensor 目标，而不只是枚举拼写。