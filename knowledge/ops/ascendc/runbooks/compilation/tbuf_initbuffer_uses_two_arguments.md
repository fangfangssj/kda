---
type: "Runbook"
title: "TBuf 三参数 InitBuffer 触发 isTQue 静态断言"
description: "TBuf 误用带队列深度的三参数 InitBuffer 会选择仅支持 TQue 的重载；改用两参数缓冲区重载后恢复编译。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# TBuf 三参数 InitBuffer 触发 isTQue 静态断言

## 现象

在为 Ascend950 / `dav-3510` 构建 Flash kernel 直调算子时，`operators/rms_norm/rms_norm.asc` 已经声明两个流水队列和一个常驻计算缓冲区。远端执行构建命令后，Bisheng 编译器在 `KernelRmsNorm<T>::Init` 中实例化 `gammaBuf_` 的 `InitBuffer` 调用，并报出原文：

```text
error: static assertion failed due to requirement
'TBuf<AscendC::TPosition::VECCALC>::isTQue':
TPipe::InitBuffer(T& que, uint8_t num, uint32_t len) not supports T as TBuf
```

调用栈继续指向：

```text
pipe_.InitBuffer(gammaBuf_, 1, rowStride_ * sizeof(T));
```

随后 `gmake` 以 `Error 2` 结束，没有生成目标共享库。此时 `xQueue_`、`yQueue_` 是 `TQue`，而 `gammaBuf_` 是 `TBuf<VECCALC>`；三者不能套用同一个带深度参数的重载。

## 适用场景

- 工程形态：Ascend950 Flash kernel 直调工程中的 `.asc` 编译；
- API 路径：`AscendC::TPipe::InitBuffer` 为 `AscendC::TQue` 与 `AscendC::TBuf` 分配 UB；
- 验证过的平台与 CANN 版本：`950` / `dav-3510`，CANN 9.0.1，与 frontmatter `platforms` 一致；
- 适用粒度：`platform_wide`，适用于同一工具链中混合使用 `TQue` 和 `TBuf` 的 AscendC kernel；
- 路线优先级：`recommended_fastpath`，对 `TBuf` 直接使用其合法的两参数重载；该标注表示首选契约正确写法，不声明性能收益；
- 明确不适用或未验证：不适用于 `TQue`，因为队列仍需用三参数形式表达队列深度；A2、A3 以及其他 CANN 版本的重载集合未在本次开发中验证。

## 根因

`gammaBuf_` 的静态类型是 `AscendC::TBuf`，却传入了队列深度参数，导致重载解析选中只允许 `T::isTQue == true` 的三参数 `TPipe::InitBuffer(T& que, uint8_t num, uint32_t len)` 并触发静态断言。

## 解决配方

保留 `TQue` 的 `(queue, depth, bytes)` 三参数调用；仅对 `TBuf` 删除队列深度参数，改用 `(buffer, bytes)` 两参数调用。以下 before/after 均由事件流中的完整首次写入与精确 edit 机械恢复，代码未做改写。

### before

revision: 未提交工作区快照（首次远端构建前）；path: `operators/rms_norm/rms_norm.asc`；symbol: `KernelRmsNorm<T>::Init`；line segment: 46–72。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t numRow, int64_t numCol,
                                int64_t blockRowLength, int64_t tileRows,
                                float epsilon, float avgFactor)
    {
        rowStride_ = (numCol + ELEMS_ALIGN - 1) / ELEMS_ALIGN * ELEMS_ALIGN;
        int64_t coreRowOffset = blockRowLength * AscendC::GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ T *)x + coreRowOffset * numCol);
        yGm_.SetGlobalBuffer((__gm__ T *)y + coreRowOffset * numCol);
        gammaGm_.SetGlobalBuffer((__gm__ T *)gamma);

        int64_t currentBlockRows = numRow - coreRowOffset;
        if (currentBlockRows > blockRowLength) currentBlockRows = blockRowLength;
        if (currentBlockRows < 0) currentBlockRows = 0;

        numCol_ = numCol;
        rowStride_ = (numCol + ELEMS_ALIGN - 1) / ELEMS_ALIGN * ELEMS_ALIGN;
        epsilon_ = epsilon;
        avgFactor_ = avgFactor;
        tileRows_ = tileRows;
        tileNum_ = currentBlockRows / tileRows_;
        tailTileRows_ = currentBlockRows - tileNum_ * tileRows_;

        pipe_.InitBuffer(xQueue_, PIPELINE_DEPTH, tileRows_ * rowStride_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, PIPELINE_DEPTH, tileRows_ * rowStride_ * sizeof(T));
        pipe_.InitBuffer(gammaBuf_, 1, rowStride_ * sizeof(T));
    }
```

这段代码把 `gammaBuf_` 当成队列处理；中间的 `1` 被解析为 `num`，因而进入仅接受 `TQue` 的重载。

### after

revision: 未提交工作区快照（精确 edit 后、成功重建前）；path: `operators/rms_norm/rms_norm.asc`；symbol: `KernelRmsNorm<T>::Init`；line segment: 46–72。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t numRow, int64_t numCol,
                                int64_t blockRowLength, int64_t tileRows,
                                float epsilon, float avgFactor)
    {
        rowStride_ = (numCol + ELEMS_ALIGN - 1) / ELEMS_ALIGN * ELEMS_ALIGN;
        int64_t coreRowOffset = blockRowLength * AscendC::GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ T *)x + coreRowOffset * numCol);
        yGm_.SetGlobalBuffer((__gm__ T *)y + coreRowOffset * numCol);
        gammaGm_.SetGlobalBuffer((__gm__ T *)gamma);

        int64_t currentBlockRows = numRow - coreRowOffset;
        if (currentBlockRows > blockRowLength) currentBlockRows = blockRowLength;
        if (currentBlockRows < 0) currentBlockRows = 0;

        numCol_ = numCol;
        rowStride_ = (numCol + ELEMS_ALIGN - 1) / ELEMS_ALIGN * ELEMS_ALIGN;
        epsilon_ = epsilon;
        avgFactor_ = avgFactor;
        tileRows_ = tileRows;
        tileNum_ = currentBlockRows / tileRows_;
        tailTileRows_ = currentBlockRows - tileNum_ * tileRows_;

        pipe_.InitBuffer(xQueue_, PIPELINE_DEPTH, tileRows_ * rowStride_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, PIPELINE_DEPTH, tileRows_ * rowStride_ * sizeof(T));
        pipe_.InitBuffer(gammaBuf_, rowStride_ * sizeof(T));
    }
```

修复后，两个 `TQue` 继续声明 `PIPELINE_DEPTH`，`TBuf` 只声明字节长度；缓冲区角色与重载契约一一对应。

## 设计决策

1. 选择两参数 `TBuf` 重载，是因为编译器静态断言已经把三参数重载限定为 `TQue`，不是为了规避诊断或放松检查。
2. 放弃“给所有 UB 对象统一传 depth”的写法；`TQue` 的队列槽位数与 `TBuf` 的单块缓冲区语义不同，统一封装会掩盖这个类型边界。
3. 当前方案归为 `recommended_fastpath`：直接遵守类型契约，不引入额外队列或拷贝，也没有 fallback 分支。

## 系统信息

- toolchain：CANN 9.0.1，Bisheng/Clang 15.0.5，`dav_c310` 的 `kernel_tpipe_impl_c310.h`；
- 目标平台：Ascend950PR_957c / `dav-3510`；
- 工程形态：PyTorch 自定义算子的 Flash kernel 直调构建；
- 相关文件：`operators/rms_norm/rms_norm.asc`；
- 相关类型：`AscendC::TPipe`、`AscendC::TQue`、`AscendC::TBuf<VECCALC>`。

## 预防

- 读用 `InitBuffer` 前先按对象静态类型区分调用形态：`TQue` 使用 `(queue, depth, bytes)`，`TBuf` 使用 `(buffer, bytes)`；
- 审查 `TPipe` 初始化函数时，不要只看所有对象都位于 UB；同时核对它们是队列还是常驻缓冲区；
- 遇到包含 `T::isTQue` 的静态断言，先检查是否给 `TBuf` 传入了队列深度，而不是先改 UB 字节数或模板类型。

## 验证信号

来源是一次 Ascend950 Flash 算子开发中的首次远端骨架构建：编译器先对三参数 `TBuf` 调用给出静态断言；精确删除 `gammaBuf_` 的深度参数后立即执行相同构建流程。

- 修复前：Bisheng 输出 `not supports T as TBuf` 与 `TBuf<VECCALC>::isTQue` 静态断言，`gmake` 以 `Error 2` 结束（来源平台: 950 / dav-3510；证据等级: tool_observed）。
- 修复后：相同远端构建完成 `Linking ASC shared library libop_rms_norm.so` 和 `Built target op_rms_norm`（来源平台: 950 / dav-3510；证据等级: tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R4：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R5：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R6：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 本配方只解决 `TBuf` 误选 `TQue` 三参数重载的编译失败，不解释 UB 容量不足、队列深度不足或运行时同步问题；
- 如果对象本身是 `TQue`，删除 depth 会改变队列初始化语义，不能套用本修复；
- A2、A3 和 CANN 9.0.1 之外版本未验证；不得假设其重载声明与本次目标环境完全一致。

## 坏实践

- 为了“接口统一”给 `TBuf` 硬塞一个值为 `1` 的 depth；这个值不会把缓冲区变成单槽队列，只会选择错误重载并触发编译期断言。
- 看到断言后改动模板类型、注释静态断言或降低编译检查；这些做法绕开类型契约，不能修复调用错误。

## 误诊

本 trace 没有把该静态断言误判为 UB 大小、dtype 或 Host tiling 问题；这些方向没有失败证据，不能作为本卡结论。