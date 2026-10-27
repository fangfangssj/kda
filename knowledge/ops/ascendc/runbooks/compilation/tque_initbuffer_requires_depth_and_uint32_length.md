---
type: "Runbook"
title: "TQue InitBuffer 须传深度和 uint32 长度"
description: "将 TBuf 的两参数 InitBuffer 写法套到 TQue 会无匹配重载；按 queue、bufferNum、uint32 len 三参数初始化后恢复构建。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# TQue InitBuffer 须传深度和 uint32 长度

## 现象

在 A3 / `dav-2201` 的经典 AscendC Gather kernel 中，三个队列以 `pipe_.InitBuffer(queue, bytes)` 初始化时，编译器对 `srcQueue_` 首先报 `no matching member function for call to 'InitBuffer'`。错误不是 UB 容量不足，而是调用形态与 TQue 重载不匹配。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`TQue`、`TPipe`、`InitBuffer`、`bufferNum` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖当前 TQue 类型、深度和字节长度重载。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

代码误用了 TBuf 风格的两参数调用。该环境中的 TQue 重载要求显式提供队列 buffer 数量，且单 buffer 长度形参为 `uint32_t`；原调用同时缺少 `bufferNum` 并传入 `int64_t` 长度，因此没有匹配重载。

## 解决配方

### before

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                               int64_t numRows, int64_t rowLenSrc, int64_t rowLenOut,
                               int64_t blockLength, uint32_t outTile, int64_t srcFits)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        idxGm_.SetGlobalBuffer((__gm__ IdxT *)index);
        yGm_.SetGlobalBuffer((__gm__ T *)y);
        rowLenSrc_ = rowLenSrc;
        rowLenOut_ = rowLenOut;
        outTile_ = outTile;
        srcFits_ = srcFits;

        // srcQueue holds a full source row when it fits; a 32B placeholder otherwise.
        // srcFits is uniform across cores (depends only on fixed rowLenSrc), so the
        // conditional Init/AllocTensor never diverges between cores (no ACL 507035).
        int64_t srcRowBytes = rowLenSrc * static_cast<int64_t>(sizeof(T));
        int64_t srcBufBytes = srcFits ? srcRowBytes : 32;
        pipe_.InitBuffer(srcQueue_, srcBufBytes);
        pipe_.InitBuffer(idxQueue_, outTile * static_cast<int64_t>(sizeof(IdxT)));
        pipe_.InitBuffer(outQueue_, outTile * static_cast<int64_t>(sizeof(T)));

        int64_t rowStart = blockLength * AscendC::GetBlockIdx();
        int64_t rowEnd = rowStart + blockLength;
        if (rowEnd > numRows) rowEnd = numRows;
        if (rowStart > numRows) rowStart = numRows;
        rowStart_ = rowStart;
        rowEnd_ = rowEnd;
    }
```

### after

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                               int64_t numRows, int64_t rowLenSrc, int64_t rowLenOut,
                               int64_t blockLength, uint32_t outTile, int64_t srcFits)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        idxGm_.SetGlobalBuffer((__gm__ IdxT *)index);
        yGm_.SetGlobalBuffer((__gm__ T *)y);
        rowLenSrc_ = rowLenSrc;
        rowLenOut_ = rowLenOut;
        outTile_ = outTile;
        srcFits_ = srcFits;

        // srcQueue holds a full source row when it fits; a 32B placeholder otherwise.
        // srcFits is uniform across cores (depends only on fixed rowLenSrc), so the
        // conditional Init/AllocTensor never diverges between cores (no ACL 507035).
        int64_t srcRowBytes = rowLenSrc * static_cast<int64_t>(sizeof(T));
        uint32_t srcBufBytes = static_cast<uint32_t>(srcFits ? srcRowBytes : 32);
        pipe_.InitBuffer(srcQueue_, static_cast<uint8_t>(PIPELINE_DEPTH), srcBufBytes);
        pipe_.InitBuffer(idxQueue_, static_cast<uint8_t>(PIPELINE_DEPTH),
                         static_cast<uint32_t>(outTile * sizeof(IdxT)));
        pipe_.InitBuffer(outQueue_, static_cast<uint8_t>(PIPELINE_DEPTH),
                         static_cast<uint32_t>(outTile * sizeof(T)));

        int64_t rowStart = blockLength * AscendC::GetBlockIdx();
        int64_t rowEnd = rowStart + blockLength;
        if (rowEnd > numRows) rowEnd = numRows;
        if (rowStart > numRows) rowStart = numRows;
        rowStart_ = rowStart;
        rowEnd_ = rowEnd;
    }
```

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：BUILD_FAIL，编译器报 InitBuffer 无匹配重载。（来源平台：a3；证据等级: tool_observed）。
- 修复后：补齐 depth 与 uint32 长度后输出 BUILD_OK。（来源平台：a3；证据等级: tool_observed）。

## 预防

- TQue 固定写成 `InitBuffer(queue, depth, uint32_bytes)`，TBuf 再使用对应两参数形式；
- shape/numel 计算可保留 `int64_t`，在确认单 buffer 字节数受 UB 上限约束后再显式收窄到 `uint32_t`；
- 编译错误优先核对 queue/buffer 类型与重载，不要用强转掩盖缺失的参数。

## 失效边界

本卡不证明任意 `int64_t` 字节值都可安全收窄；调用者仍须先保证长度不超过 UB 和 `uint32_t`。也不覆盖 TBuf、TQueBind 或其他架构的不同重载。