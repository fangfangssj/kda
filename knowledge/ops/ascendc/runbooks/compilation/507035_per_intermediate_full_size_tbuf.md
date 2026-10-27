---
type: "Runbook"
title: "507035：每中间量全尺寸 TBuf 总量超 UB"
description: "AscendC kernel 为每个中间张量各开一个全尺寸 TBuf，总 UB 达 1867KB 远超 192KB 上限，向量核异常 507035；消除全部 TBuf 仅留 2 队列并改逐元素标量计算后恢复。"
tags: [task.debugging, topic.memory, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 507035：每中间量全尺寸 TBuf 总量超 UB

## 现象

在开发 BesselI1e 算子（算子仓自定义包工程，AscendC kernel + aclnn 白盒 ST）的 Branch-0（FP32）kernel 时，向量版实现编译安装均成功，但运行白盒 ST（14 个用例，oracle 为 scipy.special.i1e）出现大规模崩溃：14 个用例中 12 个在 `aclrtSynchronizeStream(stream)` 处报

```text
ACL error 507035 at .../tests/aclnn/test_aclnn.cpp:351: aclrtSynchronizeStream(stream)
```

（507035 = ACL_ERROR_RT_VECTOR_CORE_EXCEPTION，向量核异常）。只有 2 个用例通过——它们是空输入和零轴用例（dim0=0，kernel 早退，实际不执行计算）。当时 kernel 的 `Init` 中除了 `queIn`/`queOut` 两个队列 buffer 外，还为算法的每个中间张量各分配了一个全尺寸 TBuf（`bufA`、`bufYLe`、`bufYGt`、`bufT1`、`bufChA`、`bufRA`、`bufChB`、`bufRB`、`bufMask`、`bufOnes`、`bufTmp`、`bufZeroT`、`bufB0`、`bufB1`、`bufB2`、`bufBT`、`bufScratch` 共 17 个），每个都按一个 tile 的字节数（98304B）分配。

## 适用场景

- 工程形态：算子仓自定义算子开发（AscendC `op_kernel/*.cpp`，TPipe/TQue/TBuf 内存规划），aclnn 白盒 ST 验证。
- API 路径：`AscendC::TPipe::InitBuffer`（TQue 与 TBuf）、`AscendC::TQue`、向量二进制 API（Abs/Mul/Div/Select 等）。
- 验证过的平台与 CANN 版本：ascend950（编译目标与 ST 执行）；CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（UB 容量是平台行为，与具体算子无关；192KB ubSize 由 tiling 平台参数给出）。
- 路线优先级：recommended_fastpath（正确性修复，实测通过；性能代价未测量）。
- 明确不适用/未验证：性能敏感场景（本轨迹无 profiling 信号，标量路径代价未测）；其他 ubSize 不同的平台需按本文算式重算（未验证，推断可迁移）。

## 根因

kernel 的 UB 分配总量超出平台上限。机制链与算式：

- tiling 给出 `ubSize = 196608` B（192KB，TilingUT 实测确认）；
- FP32：`elemBytes = 4`、`bufferNum = 2` → 每队列元素数 `ubFormer = 196608 / (2×4) = 24576`；
- 每个 buffer 字节 `bufBytes = align(24576 × 4, 32) = 98304` B；
- kernel `Init` 中共 19 个 buffer（queIn 1 槽 + queOut 1 槽 + 17 个 TBuf），每个 98304 B：
  `19 × 98304 = 1867776 B ≈ 1824KB ≫ 192KB`（约 9.7 倍超限）。

UB 物理容量只有 192KB，超额分配在向量核执行时触发异常，host 侧表现为 `aclrtSynchronizeStream` 返回 507035。空输入/零轴用例因 `dim0 == 0` 提前返回、不触碰 UB，故幸免。

失败前 `Init` 的分配代码（每个中间张量一个全尺寸 TBuf）：

```cpp
        // Align buffer size to ONE_BLK_SIZE=32 for DataCopy DMA alignment
        constexpr uint32_t ONE_BLK_SIZE = 32;
        uint32_t rawBytes = static_cast<uint32_t>(this->ubFormer * sizeof(T));
        uint32_t bufBytes = (rawBytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        pipe.InitBuffer(queIn, 1, bufBytes);
        pipe.InitBuffer(queOut, 1, bufBytes);
        pipe.InitBuffer(bufA, bufBytes);
        pipe.InitBuffer(bufYLe, bufBytes);
        pipe.InitBuffer(bufYGt, bufBytes);
        pipe.InitBuffer(bufT1, bufBytes);
        pipe.InitBuffer(bufChA, bufBytes);
        pipe.InitBuffer(bufRA, bufBytes);
        pipe.InitBuffer(bufChB, bufBytes);
        pipe.InitBuffer(bufRB, bufBytes);
        pipe.InitBuffer(bufMask, bufBytes);
        pipe.InitBuffer(bufOnes, bufBytes);
        pipe.InitBuffer(bufTmp, bufBytes);
        pipe.InitBuffer(bufZeroT, bufBytes);
        // Chbevl internal temporaries
        pipe.InitBuffer(bufB0, bufBytes);
        pipe.InitBuffer(bufB1, bufBytes);
        pipe.InitBuffer(bufB2, bufBytes);
        pipe.InitBuffer(bufBT, bufBytes);
        pipe.InitBuffer(bufScratch, bufBytes);  // dedicated scratch for Chbevl
```

## 解决配方

消除全部 17 个 TBuf，只保留 `queIn` + `queOut`（`2 × 98304 = 196608 B = 192KB`，恰好用满预算），`Compute` 改为 `LocalTensor::operator()(offset)` 逐元素标量计算——所有中间量都在标量寄存器里，不再需要任何 UB 中间 buffer。修复后的 `Init` 与 `Compute`：

```cpp
        // 2-Queue design: only queIn + queOut, each holds ubFormer elements.
        // Total UB = bufferNum(2) × ubFormer × sizeof(T) = 196608 bytes = 192KB.
        // No TBufs — all intermediate computation uses scalar registers.
        constexpr uint32_t ONE_BLK_SIZE = 32;
        uint32_t rawBytes = static_cast<uint32_t>(this->ubFormer * sizeof(T));
        uint32_t bufBytes = (rawBytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        pipe.InitBuffer(queIn, 1, bufBytes);
        pipe.InitBuffer(queOut, 1, bufBytes);
```

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        // Element-by-element scalar computation — no UB TBufs needed
        for (uint32_t i = 0; i < c; ++i) {
            yLocal(i) = BesselI1eScalar<T>(xLocal(i));
        }

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

其中单元素标量实现（Chebyshev 级数 Clenshaw 递推 + Newton 迭代开方，替代向量版 Abs/Mul/Div/Select 链）：

```cpp
// Scalar Chbevl: evaluates Chebyshev polynomial of degree N-1 at point y
template <typename T>
__aicore__ inline T ChbevlScalar(T y, const float* coeffs, int N)
{
    T b0 = T(coeffs[0]);
    T b1 = T(0.0f);
    T b2 = T(0.0f);
    for (int j = 1; j < N; ++j) {
        T t = y * b0 - b1 + T(coeffs[j]);
        b2 = b1;
        b1 = b0;
        b0 = t;
    }
    return (b0 - b2) * T(0.5f);
}

// Scalar sqrt with Newton's method (FP32 precision, ~8 iterations)
template <typename T>
__aicore__ inline T ScalarSqrt(T x)
{
    if (x <= T(0.0f)) return T(0.0f);
    T guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = (guess + x / guess) * T(0.5f);
    }
    return guess;
}

// Scalar Bessel I1e computation for a single element
template <typename T>
__aicore__ inline T BesselI1eScalar(T x)
{
    // Handle special values
    if (x != x) return x;          // NaN → NaN
    if (x == T(0.0f)) return x;    // ±0 → ±0 (i1e is odd)

    T ax = (x < T(0.0f)) ? -x : x;  // |x|

    T result;
    if (ax <= T(8.0f)) {
        T y = ax * T(0.5f) - T(2.0f);
        T ch = ChbevlScalar<T>(y, CHEBYSHEV_A, 17);
        result = ax * ch;
    } else {
        T y = T(32.0f) / ax - T(2.0f);
        T ch = ChbevlScalar<T>(y, CHEBYSHEV_B, 7);
        result = ch / ScalarSqrt<T>(ax);
    }

    if (x < T(0.0f)) result = -result;
    return result;
}
```

## 设计决策

- 为什么选"消除 TBuf + 标量重写"而不是其他方案：向量算法至少需要 3 个同时活跃的中间量（输入 x、|x|、分段变量 yLe），2 个队列 buffer 无法同时容纳；把队列 buffer 切成子区域 overlay 需要 `SetBuffer` 手工管理生命周期，复杂且易错；VF（寄存器）全链重写当时评估复杂度过高。标量路径零 UB 中间量、正确性最容易保证。
- 当前方案路线：safety_fallback（正确性优先的可靠路径；性能代价未测量，见失效边界）。
- 队列每队列只开 1 槽（`InitBuffer(queIn, 1, bufBytes)`）而不是双缓冲：双缓冲会把每槽数量减半，总字节不变但单 tile 元素数下降，本修复保持 tile 大小不变。

## 预防

- 写 AscendC kernel 的 `Init` 时先做 UB 预算清单：`Σ(所有 InitBuffer 的字节数) ≤ ubSize`。"每个中间张量各开一个全尺寸 TBuf"的直觉向量写法，在中间量多的算法（Chebyshev 递推需要 b0/b1/b2/t，分段判定还需要掩码/分支结果）下必然超额——本例 19 × 98304B ≈ 9.7 倍于 192KB 上限。tiling 侧 `ubSize` 是平台参数（本例 196608B），不要按 GM 思维估算 UB。

## 验证信号

来源说明：问题与验证来自 BesselI1e 算子 Branch-0（FP32）白盒 ST 的一轮崩溃修复：向量版 kernel 运行 14 例中 12 例在流同步报 507035，消除 TBuf 改标量后重建安装重跑全量通过。

- 修复前：12/14 用例 ERROR（`ACL error 507035 at ...:351: aclrtSynchronizeStream(stream)`），仅空输入/零轴 2 例 PASS（来源平台：950；证据等级：tool_observed）。
- 根因量化：ubSize=196608B、ubFormer=24576、bufBytes=98304B，19 buffer 共 1867776B≈1824KB（来源平台：950；证据等级：tool_observed——TilingUT 与 kernel 源码共同确认）。
- 修复后：重建（`rm -rf build && cmake .. && make -j binary package`）+ 安装 + 重跑 ST，`PASS: 14 | SKIP: 0 | FAIL: 0 | ERROR: 0 | TOTAL: 14`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本算式基于 ubSize=196608B（192KB）的平台；其他平台按 `ubSize/(bufferNum×elemBytes)` 重算（未验证，推断机制可迁移）。
- elemBytes=2（FP16/BF16）时 `ubFormer=49152`，同样只够 2 个 buffer——多 TBuf 写法同样会超额。
- 标量路径的性能代价未测量（本轨迹无 profiling 数据，不能给出相对倍数）；性能敏感场景需另行评估。
- 未验证：队列 buffer 子区域 overlay（`SetBuffer`）等"保留向量化但压缩中间量"的替代方案。

**未决项**：

- 标量逐元素路径相对向量路径的性能代价未测量（下一测量点：同 shape 双路径对比）。
- "2 队列 buffer + 子区域 overlay 保留向量化"的方案未尝试，可作为性能恢复方向。
## 坏实践

- 为算法每个中间张量无脑开一个全尺寸 `TBuf`，不做 `Σ InitBuffer ≤ ubSize` 预算。
- 只跑非空用例之外的特殊用例（空输入/零轴）就认为 kernel 正常——这两类用例 dim0=0 早退，恰好绕过 UB 缺陷。
