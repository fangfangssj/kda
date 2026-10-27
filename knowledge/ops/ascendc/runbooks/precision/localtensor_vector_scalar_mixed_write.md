---
type: "Runbook"
title: LocalTensor 混用 Vector/Scalar 写致输出全错
description: LocalTensor 标量 exp 辅助函数先 Duplicate 清零再 SetValue 写入，Vector 写与 Scalar 写无排序保证使 exp 输入恒 0，online softmax rescale 恒为 1 导致输出整体错误；改纯 Vector 路径（Duplicate 直接填充目标值）修复。
tags: [task.debugging, topic.precision, topic.attention, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# LocalTensor 混用 Vector/Scalar 写致输出全错

## 现象

在 Ascend950PR（arch35，CANN 9.2.0）上开发稀疏 top-K 注意力算子（Vector 主路径 GM→UB→GM，online softmax + 逐行 Mul/ReduceSum 打分），ST Real 模式首轮 24 个用例仅 2 个通过。典型失败用例（B=1, S1=1024, S2=8192, N1=32, N2=8, Dk=Dv=128, topK=512, BSND, fp16, 非 causal）输出 `MERE=6.71 MARE=4.29e+05 (thr=9.77e-04)`，bf16、MLA（Dk≠Dv）等用例错误量级相同（MERE 6.4~14.9，MARE 最高 6.8e+06），nan=0；只有单头小 topK 用例（topK=64、topK=1）通过且 topK=1 用例 MERE=0。这是正确性错误而非精度抖动。当时代码状态：kernel 中标量 exp 通过一个 8 元素 fp32 LocalTensor（expBuf）实现——先 `Duplicate` 把 8 个元素清零，再 `SetValue(0, x)` 写入第 0 个元素，然后对该 buffer 调 Vector `Exp`，取 `GetValue(0)` 返回。

## 适用场景

- 工程形态：AscendC kernel（aclnn 两段式调用的自定义算子）的 ST 精度联调；也适用于任何 kernel 内标量辅助计算。
- API 路径：AscendC Vector API（`Duplicate` / `Exp` / `Muls` / `SetValue` / `GetValue`）与 LocalTensor 标量读写混用的写法。
- 验证过的平台与 CANN 版本：Ascend950PR（含于 `platforms: [950]`）+ CANN 9.2.0。
- 适用粒度：operator_family——所有在 UB 上用小 LocalTensor 做标量计算（rescale 系数、逐元素系数）的 Vector kernel。
- 路线优先级：recommended_fastpath（纯 Vector 路径无额外代价）。
- 明确不适用 / 未验证场景：纯 Vector 数据流、无 Scalar/Vector 写交错的 kernel；其它 NPU 代卡未验证（推断可迁移）。

## 根因/解决配方

根因（单句）：标量 exp 辅助函数对同一 8 元素 LocalTensor 先用 `Duplicate`（Vector 写）清零、再用 `SetValue`（Scalar 写）写入目标值，两类写之间无硬件排序保证，Vector 清零可能晚于 Scalar 写落盘，覆盖掉刚写入的 x，使 exp 的输入恒为 0。

机制链（为什么输出会全错）：online softmax 每遇到更大的 chunk max 需要缩放旧累加，`float rescale = ScalarExp(oldMax - newMax)`（应为小于 1 的系数），旧 sum 与 outAcc 都乘它；若 ScalarExp 因上述混写恒返回 `exp(0)=1`，则旧累加永不缩放，等价于 max 迁移失效，最终 `y = outAcc / sum` 系统性偏离 golden——表现为全部用例 MERE 在个位数量级的整体错误，且与 causal/GQA/shape 因素无关。

修复（before/after，函数级 verbatim）：

修复前：

```cpp
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline float BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ScalarExp(float x)
{
    auto expTmp = expBuf.Get<float>();
    Duplicate<float>(expTmp, 0.0f, 8);  // 清零避免 NaN（exp(0)=1 无害）
    expTmp.SetValue(0, x);
    Exp<float>(expTmp, expTmp, 8);
    return expTmp.GetValue(0);
}
```

修复后（纯 Vector 路径，Duplicate 直接填充目标值）：

```cpp
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline float BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ScalarExp(float x)
{
    // ⚠️ 禁止 Duplicate + SetValue 混用：Duplicate 是 Vector 写，SetValue 是 Scalar 写，
    // 两者写同一地址时无硬件排序保证，Vector 写可能覆盖 Scalar 写。
    // 正确做法：纯 Vector 路径，Duplicate 直接填充目标值，避免 Scalar 写。
    auto expTmp = expBuf.Get<float>();
    Duplicate<float>(expTmp, x, 8);      // Vector 写：全部填充 x
    Exp<float>(expTmp, expTmp, 8);       // Vector 计算：exp(x)
    return expTmp.GetValue(0);           // Scalar 读（Vector 写后读，有隐式同步）
}
```

## 设计决策

- 选择"纯 Vector 写 + Scalar 读"而非"纯 Scalar 计算"：Scalar 读紧跟 Vector 写在同序依赖下有隐式同步（修复后实测正确），而纯 Scalar 循环无法利用 Vector Exp。
- 8 元素（32B）buffer 是该平台 Vector 指令的最小舒适粒度，保持原设计不变，只改写入通道。

## 预防

- LocalTensor 上的标量计算保持单一写入通道：要么纯 Vector（`Duplicate` 直接填目标值），要么纯 Scalar（`SetValue`）。
- review 检查点：凡出现 `Duplicate(...,0,...)` 后紧跟 `SetValue` 的模式，一律视为可疑。

## 验证信号

问题与验证来源：在一次算子开发会话的 ST 联调中，先全量跑出 2/24 的失败基线，随后仅修改 ScalarExp 这一处代码并重建、重装算子包后重跑同一 CSV。

- 修复前典型用例 `MERE=6.71 MARE=4.29e+05`（来源平台：950；证据等级：tool_observed）。
- 修复后同一用例 `MERE=1.81e-04 MARE=1.90e-01`（来源平台：950；证据等级：tool_observed），MERE 下降约 4 个数量级、进入阈值内，单点改动与结果强绑定。
- 修复后 topK=1 用例仍为 `MERE=0.00e+00 MARE=0.00e+00`，无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 单轨迹单次验证（single_run），未在其它平台/代卡复现。
- 该结论针对"小 LocalTensor 上 Vector/Scalar 混写"这一模式；大数据量下 Vector 与 Scalar 交错访问的通用排序规则未做系统验证（未验证，推断可迁移）。

**未决项**：

- Vector 写与 Scalar 写的确切硬件排序语义未在文档层核实（本卡按"混写无排序保证"处理并实测有效）；可用最小 kernel 探针（混写后读回）进一步隔离，列为接续方向。
## 坏实践

- 用 `Duplicate` 清零 + `SetValue` 改单元素的方式"安全化"标量计算——正是本问题的直接来源。

## 误诊

- 首轮失败曾按失败用例分布怀疑 GQA 头映射或大 topK 处理错误：非 causal、G=1、小 topK 用例同样全错，与这些因素无关，排除。
- 同期 causal 用例的持续失败是另一个独立问题（同进程 tiling 缓存），勿与本根因混并。
