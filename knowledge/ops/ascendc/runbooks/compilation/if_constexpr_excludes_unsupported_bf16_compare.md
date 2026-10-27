---
type: "Runbook"
title: "bf16 未编译剔除触发 backend Cannot select"
description: "运行时 dtype 分支仍实例化 bf16 标量比较，BiSheng backend Cannot select；用 if constexpr 编译期剔除不支持分支。"
tags: ["task.debugging", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# bf16 未编译剔除触发 backend Cannot select

## 现象

在 Ascend950 的 kernel 直调工程中编译一个按 `AccT`/`SrcT` 分发的随机写 Kernel。bf16 实例只会在运行时走覆盖写分支，归约模式则由 Host 先上转 fp32；但 `Process()` 仍用普通 `if (reduceMode_ == UPDATE)` 隔开覆盖写与归约算术。

通过 CMake 调用 BiSheng 编译 `scatter.asc` 时，编译在 bf16 Kernel 实例化阶段失败，原始关键输出为：

```text
fatal error: error in backend: Cannot select: ... i64 = setcc ... setuo:ch,
operators/scatter/scatter.asc:64:11
  ... bf16,ch = CopyFromReg ... Register:bf16 ...
In function: _Z14scatter_kernelIu6__bf16u6__bf16iEvPhS0_S0_S0_lllli
bisheng: error: clang frontend command failed with exit code 70
```

报错行落在 NaN 传播 helper 的 `a != a`，调用链来自 `Process()` 的极值归约分支。即使 bf16 的运行时 `reduceMode_` 永远是覆盖写，普通 `if` 也没有阻止该分支参与模板实例化。

## 适用场景

- 工程形态：Ascend C kernel 直调工程的模板化 Kernel 构建。
- API 路径：`__aicore__` C++ 模板、`bfloat16_t`/`half` 实例、BiSheng 后端标量比较或算术 lowering。
- 验证过的平台与版本：Ascend950PR_957b / dav-3510，CANN 9.0.1，BiSheng clang 15.0.5；与 frontmatter `platforms: [950]` 一致。
- 适用粒度：`operator_family`。适用于“某些 dtype 只允许覆盖写，另一些 dtype 才允许归约算术”的模板化随机写或索引算子。
- 路线优先级：`recommended_fastpath`。对不需要且后端不支持的模板分支，直接用 `if constexpr` 编译期剔除；本轨迹未比较性能路线。
- 明确不适用：若 bf16/half 确实必须在设备侧执行归约，不能直接剔除算术分支，必须先提供受支持的计算 dtype/API 路线。其他 CANN 版本及非 950 平台未验证。

## 根因

C++ 普通 `if` 只控制运行时执行，不控制模板实例化：编译器实例化 `Kernel<AccT=bfloat16_t>` 时，仍会编译 `else` 中的 `nanmax_(cur, v)`/`nanmin_(cur, v)`；helper 里的 `a != a` 被 lowering 为 unordered compare `setuo`，需要 bf16 `setcc`，而该版本 dav-3510 BiSheng 后端没有可选指令，于是以 `Cannot select` 终止。改为 `if constexpr` 后，`AccT` 不是 fp32/int32 时算术分支在编译期被丢弃，bf16/half/int64 实例只保留覆盖写；Host 对 half/bf16 归约先上转 fp32，因此必需语义仍由 fp32 实例承担。

## 解决配方

把“某 dtype 运行时不会进入不支持分支”的假设改成编译期约束。允许算术的 dtype 放进 `if constexpr`；其他 dtype 的实例只生成纯覆盖写代码。

### before

以下是失败版本中事件流真实写入的完整 `KernelScatter::Process()`：

```cpp
    __aicore__ inline void Process()
    {
        for (int64_t p = pStart_; p < pEnd_; ++p) {
            int64_t idx = static_cast<int64_t>(iGm_.GetValue(static_cast<uint64_t>(p)));
            int64_t dstOff = baseOuter_ + idx * yStride_[dim_];
            SrcT u = uGm_.GetValue(static_cast<uint64_t>(p));
            if (reduceMode_ == SCATTER_UPDATE) {
                yGm_.SetValue(static_cast<uint64_t>(dstOff), static_cast<AccT>(toAcc<AccT>(u)));
            } else {
                AccT cur = yGm_.GetValue(static_cast<uint64_t>(dstOff));
                AccT v = toAcc<AccT>(u);
                AccT r;
                if (reduceMode_ == SCATTER_ADD) {
                    r = cur + v;
                } else if (reduceMode_ == SCATTER_MULTIPLY) {
                    r = cur * v;
                } else if (reduceMode_ == SCATTER_AMAX) {
                    r = nanmax_(cur, v);
                } else {  // SCATTER_AMIN
                    r = nanmin_(cur, v);
                }
                yGm_.SetValue(static_cast<uint64_t>(dstOff), r);
            }
            advance();
        }
        // Flush this core's dirty y cache lines back to GM so the host sees them.
        AscendC::DataCacheCleanAndInvalid<AccT, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    }
```

### after

以下是两次真实 edit 合并恢复出的完整修复后函数；第二次 edit 又把仅覆盖写的 int64 从算术白名单移除：

```cpp
    __aicore__ inline void Process()
    {
        for (int64_t p = pStart_; p < pEnd_; ++p) {
            int64_t idx = static_cast<int64_t>(iGm_.GetValue(static_cast<uint64_t>(p)));
            int64_t dstOff = baseOuter_ + idx * yStride_[dim_];
            SrcT u = uGm_.GetValue(static_cast<uint64_t>(p));
            // aicore forbids half/bf16 scalar arithmetic/comparison (backend cannot
            // select bf16 setcc/fmul/fadd). half/bf16 are only ever dispatched on the
            // UPDATE (overwrite) path — reduce modes for half/bf16 are host-upcast to
            // fp32 (AccT=float) before launch. Guard the arithmetic with if constexpr
            // so it is not instantiated for half/bf16/int64-update-only types.
            if constexpr (std::is_same_v<AccT, float> ||
                          std::is_same_v<AccT, int32_t>) {
                if (reduceMode_ == SCATTER_UPDATE) {
                    yGm_.SetValue(static_cast<uint64_t>(dstOff), static_cast<AccT>(toAcc<AccT>(u)));
                } else {
                    AccT cur = yGm_.GetValue(static_cast<uint64_t>(dstOff));
                    AccT v = toAcc<AccT>(u);
                    AccT r;
                    if (reduceMode_ == SCATTER_ADD) {
                        r = cur + v;
                    } else if (reduceMode_ == SCATTER_MULTIPLY) {
                        r = cur * v;
                    } else if (reduceMode_ == SCATTER_AMAX) {
                        r = nanmax_(cur, v);
                    } else {  // SCATTER_AMIN
                        r = nanmin_(cur, v);
                    }
                    yGm_.SetValue(static_cast<uint64_t>(dstOff), r);
                }
            } else {
                // half/bf16 UPDATE-only instance: pure overwrite, no arithmetic.
                yGm_.SetValue(static_cast<uint64_t>(dstOff), static_cast<AccT>(u));
            }
            advance();
        }
        // Flush this core's dirty y cache lines back to GM so the host sees them.
        AscendC::DataCacheCleanAndInvalid<AccT, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    }
```

## 设计决策

1. 选择 `if constexpr`，因为问题发生在模板实例化而不是运行时分支选择；普通 `if`、提前设置 `reduceMode_` 或依赖 Host 永不分发都无法阻止后端编译该表达式。
2. 放弃在 bf16 `nanmax_` 上继续改写比较表达式：此实例本来只需 UPDATE，生成任何 bf16 归约算术都是多余的；需要归约时已由 Host 路由到 fp32 实例。
3. 白名单只保留本轨迹实际需要算术的 fp32/int32；half/bf16/int64 进入纯覆盖写分支。当前方案属 `recommended_fastpath`。

## 系统信息

- 目标：Ascend950PR_957b / dav-3510。
- 工具链：CANN 9.0.1；BiSheng clang 15.0.5。
- 工程入口：kernel 直调 CMake 构建。
- 相关文件：`scatter.asc`；修复符号为模板类的 `KernelScatter::Process()`。

## 预防

- 对每个模板 dtype 列出“需要生成的操作集合”，不要只列运行时会走的分支。
- 遇到 `Cannot select` 先沿模板实例化栈回看：报错 helper 可能只是被一个运行时不可达分支实例化。
- 对仅覆盖写 dtype，用 `if constexpr` 或模板特化从语法树中移除比较、乘加、极值等不支持操作。
- 剔除分支前确认 Host 已为被剔除的语义提供受支持路线；不能用编译通过掩盖功能缺失。

## 验证信号

来源于一次 Ascend950 scatter kernel 开发会话：首次远程构建真实触发 bf16 `setcc` backend 错误，应用编译期 guard 后在同一目标平台重建并执行受支持用例与完整评测。

- 修复后重建输出 `[100%] Linking ASC shared library` 与 `[100%] Built target`（来源平台: 950 / Ascend950PR_957b；证据等级: tool_observed）。
- guard 生效后的真机参数化测试报告 `211 passed`；合同明确不支持的 int64 reduce 单独 skip，未以放宽容差通过（来源平台: 950 / Ascend950PR_957b；证据等级: tool_observed）。
- 保持该 guard 的最终实现通过 cann-bench 20/20 正确性用例（来源平台: 950 / Ascend950PR_957b；证据等级: artifact_verified）。

## 失效边界

- 若 half/bf16 归约必须直接在设备侧执行，本配方不能只保留覆盖写；需改用该平台支持的计算 dtype/API，并重新上板验证。
- 本结论只在 CANN 9.0.1、dav-3510 上实测。其他 CANN 版本和平台是否已支持 bf16 标量比较未验证，推断可迁移但不得据此跳过编译探针。
- `if constexpr` 的 dtype 白名单必须与 Host 分发契约同步；两侧不一致会把功能缺失伪装成构建成功。

## 坏实践

- 用普通 `if` 隔离模板中后端不支持的表达式，并认为“运行时不会走”就不会编译。
- 为了消除编译错误直接删除归约功能或跳过对应受支持用例；这不构成实现修复。
- 把 `Cannot select` 只归因于报错行的 NaN 公式，而不检查是哪一个 dtype 实例和上游调用分支触发实例化。

## 误诊

错误栈指向 `nanmax_` 的 `a != a`，但根因不是 NaN 传播公式写错，而是 UPDATE-only bf16 实例仍生成了本不需要的归约 helper。