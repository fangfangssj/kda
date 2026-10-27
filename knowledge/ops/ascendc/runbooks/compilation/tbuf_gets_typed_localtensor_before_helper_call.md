---
type: "Runbook"
title: "TBuf 不能直接传给 LocalTensor 形参"
description: "Ascend C helper 以 LocalTensor 为形参时，调用点必须先从 TBuf 取得带类型的 LocalTensor，不能依赖隐式转换。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# TBuf 不能直接传给 LocalTensor 形参

## 现象

在 Ascend 950 / `dav-3510` 上构建一个 Reg 原生 Softmax 直调算子时，`operators/softmax/softmax.asc` 已经把若干计算 helper 定义成接收 `const AscendC::LocalTensor<T>&`，但 `KernelSoftmax::Process` 仍直接把 `TBuf` 成员传给它们。Bisheng 在实例化 `ComputeLoadX`、`ComputeMax`、`ComputeSubExp`、`ComputeSum` 和 `ComputeDivCast` 时连续报错，代表性原文是：

```text
error: no matching function for call to 'ComputeLoadX'
note: candidate function template not viable: no known conversion from
'AscendC::TBuf<AscendC::TPosition::VECOUT>' to
'const AscendC::LocalTensor<half>' for 1st argument
```

这不是 dtype 推导失败；当时代码把 buffer 管理对象当成了可寻址 Tensor 对象。

## 适用场景

- 工程形态：Ascend C kernel 直调工程，类内用 `TPipe` 管理 `TBuf`，计算函数用 `LocalTensor` 接口。
- API 路径：`TPipe::InitBuffer` → `TBuf::Get<T>()` → `LocalTensor<T>` → Reg wrapper。
- 验证过的平台与版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致。
- 适用粒度：`operator_family`，适用于用同一 buffer 对象串联多个 Ascend C helper 的 Vector/Reg 算子。
- 路线优先级：`recommended_fastpath`，在进入行循环前一次取得各个有类型的 `LocalTensor`，再在循环中复用。
- 明确不适用：helper 本身接收 `TBuf`，或采用 `TQue::AllocTensor/DeQue` 管理生命周期的路径；后者须遵循队列生命周期，不能照搬本卡的持久局部句柄。

## 根因

`TBuf` 表示由 `TPipe` 管理的缓冲区对象，`LocalTensor<T>` 才是带元素类型和地址语义、可传给计算 API 的 Tensor 视图。二者没有满足这些 helper 形参的隐式转换。模板报错中的目标类型已经具体化为 `LocalTensor<half>`，说明模板参数 `T` 已推导成功，失败发生在实参对象层级。

## 解决配方

在 `Process` 入口对每个 `TBuf` 调用一次 `Get<T>()`，把得到的 `LocalTensor` 传给计算和搬运 helper；同时让 `CopyIn`/`CopyOut` 显式接收同一个局部 Tensor。

### before

revision: trace snapshot before the LocalTensor fix
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmax::Process`
lines: 272-291

```cpp
    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < numRows_; ++i) {
            int64_t row = startRow_ + i;
            int64_t gmOff = row * int64_t(rLen_);
            CopyIn(gmOff);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeLoadX<T>(fBuf_, xBuf_, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeMax<T>(maxSlot_, fBuf_, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeSubExp<T>(expBuf_, fBuf_, maxSlot_, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeSum<T>(sumSlot_, expBuf_, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeDivCast<T>(yBuf_, expBuf_, sumSlot_, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            CopyOut(gmOff);
        }
    }
```

revision: trace snapshot before the LocalTensor fix
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmax::CopyIn`
lines: 294-300

```cpp
    __aicore__ inline void CopyIn(int64_t gmOff)
    {
        auto xLocal = xBuf_.Get<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(rLen_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[gmOff], copyParams, padParams);
    }
```

revision: trace snapshot before the LocalTensor fix
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmax::CopyOut`
lines: 302-307

```cpp
    __aicore__ inline void CopyOut(int64_t gmOff)
    {
        auto yLocal = yBuf_.Get<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(rLen_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[gmOff], yLocal, copyParams);
    }
```

### after

revision: trace snapshot after the LocalTensor fix
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmax::Process`
lines: 272-297

```cpp
    __aicore__ inline void Process()
    {
        auto xLocal = xBuf_.Get<T>();
        auto fLocal = fBuf_.Get<float>();
        auto expLocal = expBuf_.Get<float>();
        auto yLocal = yBuf_.Get<T>();
        auto maxLocal = maxSlot_.Get<float>();
        auto sumLocal = sumSlot_.Get<float>();
        for (int64_t i = 0; i < numRows_; ++i) {
            int64_t row = startRow_ + i;
            int64_t gmOff = row * int64_t(rLen_);
            CopyIn(gmOff, xLocal);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeLoadX<T>(fLocal, xLocal, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeMax<T>(maxLocal, fLocal, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeSubExp<T>(expLocal, fLocal, maxLocal, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeSum<T>(sumLocal, expLocal, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeDivCast<T>(yLocal, expLocal, sumLocal, rLen_);
            AscendC::PipeBarrier<PIPE_V>();
            CopyOut(gmOff, yLocal);
        }
    }
```

revision: trace snapshot after the LocalTensor fix
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmax::CopyIn`
lines: 300-305

```cpp
    __aicore__ inline void CopyIn(int64_t gmOff, AscendC::LocalTensor<T> &xLocal)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(rLen_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[gmOff], copyParams, padParams);
    }
```

revision: trace snapshot after the LocalTensor fix
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmax::CopyOut`
lines: 307-311

```cpp
    __aicore__ inline void CopyOut(int64_t gmOff, AscendC::LocalTensor<T> &yLocal)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(rLen_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[gmOff], yLocal, copyParams);
    }
```

## 设计决策

- 选择在 `Process` 的行循环外取得六个 `LocalTensor`，因为这些 `TBuf` 的生命周期覆盖整个 kernel 对象，循环内无需重复构造视图。
- 搬运 helper 改为接收同一份 `LocalTensor`，使搬入、计算和搬出明确操作同一 buffer 视图。
- 该修复只解决对象类型契约；后来输入/输出 DMA 的跨流水同步另由队列机制处理，不能把两种问题合并。

## 预防

- 审查 helper 签名时区分 buffer 管理对象与 Tensor 视图；编译前检查每个 `TBuf` 使用点是否先调用 `Get<T>()`。
- 模板错误若已显示具体 `LocalTensor<half>` 或 `LocalTensor<float>`，先排除实参对象层级，不要继续改 dtype 模板参数。

## 验证信号

来源是同一次 Ascend 950 远端构建：修改前 Bisheng 明确报 `TBuf` 到 `LocalTensor` 无转换；修改后再次构建不再出现这组模板匹配错误，并完成共享库链接。

- `TBuf<VECOUT>` 无法转换为 `LocalTensor<half>` 的编译错误可稳定触发（证据等级: tool_observed；来源平台: 950）。
- 修复后构建输出出现 `[100%] Built target op_softmax`（证据等级: tool_observed；来源平台: 950）。

## 失效边界

若 buffer 改成 `TQue`，Tensor 必须通过 `AllocTensor`/`DeQue` 获取并按 `FreeTensor` 生命周期释放，不能把本卡的循环外 `Get<T>()` 方案机械迁移。A2/A3 未验证，推断可迁移。

## 坏实践

把 `TBuf` 视作 `LocalTensor` 的透明别名，依赖模板隐式转换；这会把真正的对象层级错误淹没在大量模板实例化信息中。

## 误诊

看到 `LocalTensor<half>` 就继续调整 `T` 或 Cast 路径是误诊；错误文本已经表明模板实例化完成，失败的是 `TBuf` 实参。
