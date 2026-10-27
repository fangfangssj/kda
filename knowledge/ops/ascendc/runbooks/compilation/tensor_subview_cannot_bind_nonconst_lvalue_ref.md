---
type: "Runbook"
title: Tensor operator[] 临时值不能绑定非 const 引用
description: gm[off]/ub[off] 的 operator[] 返回按值临时子 tensor，传给非 const 左值引用形参报 non-const lvalue reference cannot bind to a temporary；GlobalTensor 按值传参或改传 base+offset 修复。
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T04:12:25Z'
updated_at: '2026-09-17T04:12:25Z'
platforms: ["950"]
---

# Tensor operator[] 临时值不能绑定非 const 引用

## 现象

在 Ascend950PR（CANN 9.2.0）上为 bench_no_harness_gru 编写 kernel 辅助函数时，把 tensor 偏移视图直接作为实参传给自定义函数，构建算子包时报两类同根因错误：

GlobalTensor 版（第一次，把 workspace 子张量传给 VectorMatmul）：

```
op_kernel/arch35/bench_no_harness_gru.h:376:22: error: non-const lvalue reference to type 'GlobalTensor<...>' cannot bind to a temporary of type 'GlobalTensor<...>'
```

LocalTensor 版（第二次重写，把 UB 子张量传给 VectorMatmul）：

```
op_kernel/arch35/bench_no_harness_gru.h:385:26: error: non-const lvalue reference to type 'LocalTensor<...>' cannot bind to a temporary of type 'LocalTensor<...>'
```

两次的调用形态分别是 `VectorMatmul(gmGiAll_[giOffset], ...)` 与 `VectorMatmul(ghUb[b * gateSize_], hPrevUb[b * hiddenSize_], ...)`——`operator[]` 返回的是按值的子 tensor（临时对象），而函数形参声明成了非 const 左值引用 `Tensor<T>&`。

## 适用场景

- 工程形态：AscendC kernel 内自定义辅助函数的参数设计（GlobalTensor/LocalTensor 视图传递）。
- API 路径：`GlobalTensor<T>::operator[]` / `LocalTensor<T>::operator[]`（按值返回偏移视图）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0，bisheng clang 15.0.5。
- 适用粒度：platform_wide——所有 AscendC kernel 的 tensor 视图传参，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：只需只读访问时 `const Tensor<T>&` 形参可绑定临时值（C++ 语言规则），本卡针对需要写回或在函数内多次取视图的场景。

## 根因/解决配方

**根因（单句）**：`gmTensor[off]`/`ubTensor[off]` 的 `operator[]` 返回**按值**的子 tensor 临时对象，C++ 不允许非 const 左值引用绑定到临时值。

**解决配方（两种，trace 中先后使用）**：

修法 A（GlobalTensor）——形参改按值传递（GlobalTensor 是 `__gm__` 指针的轻量包装，拷贝廉价）：

修复前 → 修复后：

```cpp
    __aicore__ inline void VectorMatmul(GlobalTensor<float>& gmC, GlobalTensor<D_T>& gmA,
                                        GlobalTensor<D_T>& gmB, int64_t M, int64_t N, int64_t K);
```

```cpp
    __aicore__ inline void VectorMatmul(GlobalTensor<float> gmC, GlobalTensor<D_T> gmA,
                                        GlobalTensor<D_T> gmB, int64_t M, int64_t N, int64_t K);
```

（实现处同步改：）

```cpp
__aicore__ inline void BenchNoHarnessGru<D_T, HAS_BIAS>::VectorMatmul(
    GlobalTensor<float>& gmC, GlobalTensor<D_T>& gmA, GlobalTensor<D_T>& gmB,
    int64_t M, int64_t N, int64_t K)
```

```cpp
__aicore__ inline void BenchNoHarnessGru<D_T, HAS_BIAS>::VectorMatmul(
    GlobalTensor<float> gmC, GlobalTensor<D_T> gmA, GlobalTensor<D_T> gmB,
    int64_t M, int64_t N, int64_t K)
```

修法 B（LocalTensor，最终版）——形参改为"base tensor + offset"，函数体内按 `base[off + i]` 访问，写回用 `resultBase.SetValue(resultOff + n, sum)`：

```cpp
    // 纯 Vector 矩阵乘法: result[off:off+N] = xVec[xoff:xoff+K] @ wMatrix[N,K]^T
    __aicore__ inline void VectorMatmul(
        LocalTensor<float>& resultBase, int64_t resultOff,
        const LocalTensor<float>& xVecBase, int64_t xVecOff,
        const GlobalTensor<D_T>& wGm,
        int64_t N, int64_t K);
```

调用侧（TimeStepLoop 内，不再产生临时实参）：

```cpp
            VectorMatmul(ghUb, b * gateSize_, hPrevUb, b * hiddenSize_,
                         gmWhh_, gateSize_, K);
```

修法 B 同样适用于 GlobalTensor（按 base+offset 传），并可让"只读输入"形参保持 `const&`。

## 设计决策

- 修法 A（按值）改动最小，适合"函数只包一层指针语义"的 GlobalTensor；
- 修法 B（base+offset）语义更清晰且对 LocalTensor 必须用它（结果要写回，按值形参的 SetValue 不会反映到调用者视图）——最终重写版统一采用 B。
- 函数体内 `wGm[n * K]` 这类只读临时视图可继续用（传给 `DataCopyPad` 的 const 引用形参可绑定临时值）。

## 预防

- kernel 辅助函数需要写回子视图时，形参一律"base + offset"，不要声明 `Tensor<T>&` 再指望调用方传 `ub[off]`。
- 看到本报错先定位实参里的 `xxx[expr]` 子视图调用，再决定改按值（只读/轻量）还是 base+offset（写回）。

## 验证信号

以下信号来自该算子两轮 kernel 编译循环：

- GlobalTensor 版：修复（按值传参）后构建输出 3 个 kernel 全部 `Generating BenchNoHarnessGru_*.o Done`、`[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。
- LocalTensor 版：base+offset 修复后 `rm -rf build && bash build.sh` 干净构建无该错误，产物含 `libcust_opmaster_rt2.0.so` 与 ascend950 二进制（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 在 950 + CANN 9.2.0 + bisheng clang 15.0.5 实测；本结论属 C++ 语言规则 + AscendC tensor API 返回值约定，跨平台稳定（其他平台未验证，推断可迁移）。
- 按值传 GlobalTensor 的前提是它确为轻量包装（本开发会话 推断 + 编译通过，未做布局级验证）。

**未决项**：

- 无。
## 坏实践

- 把 host 侧"传引用再改"的习惯带进 kernel：声明 `LocalTensor<float>& result` 并以 `ub[off]` 调用。
- 报错后盲目给所有形参加 `const`——需要写回的形参加 `const` 后 SetValue 直接编译不过。
