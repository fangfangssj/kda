---
type: "Runbook"
title: "TQue AllocTensor 作临时分配致 kernel 挂起"
description: "Compute 里把 queIn.AllocTensor<T>()/FreeTensor 当通用临时内存用（AllocTmp/FreeTmp 封装）会使 kernel 在首个用例挂起（构建 SUCCESS + 600s 超时无输出）；加队列深度、换独立 TQue 都无效——临时张量必须用 TBuf（每临时一块 InitBuffer + Get<T>()）。"
tags: [task.debugging, topic.memory, topic.programming_model]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# TQue AllocTensor 作临时分配致 kernel 挂起

## 现象

bessel_i1e 类算子（Branch-0 FP32）kernel 的 Compute 需要十几个中间张量（a/yLe/yGt/chA/rA/chB/rB/mask/ones/tmp/zeroT + Chbevl 内部 b0/b1/b2/t）。实现时按"从输入队列借一块"的思路封装了临时分配：

```cpp
    // Allocate a temp tensor from the input queue; must be freed after use
    __aicore__ inline AscendC::LocalTensor<T> AllocTmp()
    {
        return queIn.template AllocTensor<T>();
    }

    __aicore__ inline void FreeTmp(AscendC::LocalTensor<T>& t)
    {
        queIn.FreeTensor(t);
    }
```

流水线本身已验证健康（CopyIn→Compute→CopyOut，pass-through/Abs 全 14 用例无崩溃）。一旦 Compute 使用 AllocTmp——哪怕最简单的三行版本——run.sh 在第一个用例就挂起：

```
=== BesselI1e Branch-0 (FP32) Whitebox ST ===
[ 1/14] 0-A_f32_main_block_aligned
<shell_metadata>
shell tool terminated command after exceeding timeout 600000 ms. ...
```

（构建 SUCCESS 但测试首用例无任何 Mismatch 输出、shell 600 秒超时；aclrtSynchronizeStream 永不返回。）二分定位：

- `Abs(yLocal, xLocal, c)`（不 AllocTmp，直接写输出队列张量）→ 14/14 完成；
- `a = AllocTmp(); Abs(a, xLocal, c); Muls(yLocal, a, 0.5f, c); FreeTmp(a);` → 挂起；
- 恢复 AllocTmp 且把 queIn 深度 1→2、tiling 同步扣除 → 仍挂起；
- 换独立的 `TQue<VECIN, 1> queTmp` 专供临时 → 仍挂起；
- 改用 `TBuf`（`pipe.InitBuffer(tmpBuf, bytes)` + `tmpBuf.Get<T>()`）→ 全部用例完成。

## 适用场景

- 工程形态：算子仓 kernel 开发，AscendC 标准 API 流水线（TPipe/TQue/TBuf），Compute 需要多个中间张量的算子（数学函数链、Clenshaw 递推等）。
- API 路径：`AscendC::TQue::AllocTensor/FreeTensor` vs `AscendC::TBuf` + `pipe.InitBuffer`；`TPosition::VECIN/VECOUT`。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——ascend950 上所有 AscendC 流水线 kernel 的临时内存组织。
- 路线优先级：recommended_fastpath（临时内存一律 TBuf；TQue Alloc/Free 仅服务 CopyIn/CopyOut 流水线）。
- 明确不适用 / 未验证场景：输入/输出本身经队列 Alloc/EnQue/DeQue/FreeTensor 的标准用法（CopyIn/CopyOut 路径）不受影响——出问题的是"把队列 AllocTensor 当通用临时分配器"这一用法。"DeQue 输入未释放导致后续 AllocTensor 卡死"是另一形态（槽位耗尽，加 depth 可推迟），见 [TQue 输入未释放导致循环 AllocTensor 卡死卡](tque_dequeue_must_end_with_freetensor.md)。

## 根因

机制未完全明确（无 plog/硬件级证据）。经验结论：`TQue::AllocTensor/FreeTensor` 是流水线队列管理语义（与 EnQue/DeQue 的事件配对），在 Compute 内对同一个（甚至独立的）VECIN 队列做"DeQue 输入之后再 AllocTensor 若干次当临时、再 FreeTensor"的用法会让 kernel 挂起（疑似队列事件等待死锁——depth 2 与独立队列仍挂起说明不是槽位耗尽）；`TBuf` 才是通用临时内存的正确抽象（静态划分，一次 InitBuffer，Get<T>() 随取）。

## 判别方法

1. 症状：构建 SUCCESS + 测试首用例无输出 + shell 超时（不是 507035 崩溃、不是 mismatch）。
2. 把 Compute 换成不使用临时分配的单操作（直接 xLocal→yLocal）——若恢复完成，即锁定临时分配方式。
3. 逐个变量排除：队列深度、独立队列、TBuf。

## 解决配方

修复前（AllocTmp 走 queIn，verbatim）：

```cpp
    // Allocate a temp tensor from the input queue; must be freed after use
    __aicore__ inline AscendC::LocalTensor<T> AllocTmp()
    {
        return queIn.template AllocTensor<T>();
    }

    __aicore__ inline void FreeTmp(AscendC::LocalTensor<T>& t)
    {
        queIn.FreeTensor(t);
    }
```

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a = AllocTmp();
        AscendC::Abs(a, xLocal, c);
        AscendC::Muls(yLocal, a, T(0.5f), c);
        FreeTmp(a);

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

修复后（一）：单个 TBuf 顶替 AllocTmp（过渡形态，verbatim 取自修复编辑的完整 Init）：

```cpp
    __aicore__ inline void Init(
        __gm__ T* xGmBase, __gm__ T* yGmBase, const BesselI1eTilingData& td)
    {
        int64_t dim0 = td.dim0;
        if (dim0 == 0) { this->isEmpty = true; return; }

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->isLastBlock = (static_cast<int64_t>(blockIdx) == td.blockNum - 1);
        this->loopNum = this->isLastBlock ? td.ubLoopOfTailBlock : td.ubLoopOfFormerBlock;
        this->tailNum = this->isLastBlock ? td.ubTailOfTailBlock : td.ubTailOfFormerBlock;
        this->ubFormer = td.ubFormer;

        int64_t offset = static_cast<int64_t>(blockIdx) * td.blockFormer;
        this->xGm.SetGlobalBuffer(xGmBase + offset, td.blockFormer);
        this->yGm.SetGlobalBuffer(yGmBase + offset, td.blockFormer);

        pipe.InitBuffer(queIn, 1, this->ubFormer * sizeof(T));
        pipe.InitBuffer(queOut, 1, this->ubFormer * sizeof(T));
        // TBuf for temp tensors: enough space for ~8 concurrently live FP32 tensors
        pipe.InitBuffer(tmpBuf, static_cast<uint32_t>(this->ubFormer * 8 * sizeof(T)));
    }
```

```cpp
    // Allocate a temp tensor from tmpBuf; must be freed after use
    __aicore__ inline AscendC::LocalTensor<T> AllocTmp()
    {
        return tmpBuf.template Get<T>();
    }

    __aicore__ inline void FreeTmp(AscendC::LocalTensor<T>& t)
    {
        // TBuf tensors don't need explicit free, but kept for API compatibility
    }
```

修复后（二）：终版每个临时一块 TBuf（互不别名，verbatim 取自终版重写的 Init）：

```cpp
        uint32_t bufBytes = static_cast<uint32_t>(this->ubFormer * sizeof(T));
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
```

配合取用封装：

```cpp
    // Helper to get a tensor from a TBuf
    __aicore__ inline AscendC::LocalTensor<T> BufTensor(...)
    {
        return bufX.template Get<T>();  // 每个专用 TBuf 各自 Get
    }
```

（注意：单个大 TBuf 上多次 `Get<T>()` 返回的是**同一块**缓冲（别名），并发中间量会互相覆盖——终版因此按"一临时一 TBuf"展开。）

## 设计决策

- 为什么按"一临时一 TBuf"而不是单个大 TBuf 切片：`TBuf::Get<T>()` 返回整块缓冲的视图，多次 Get 之间无偏移机制，LocalTensor 也不支持指针算术切片；别名会让 Clenshaw 这类多活中间量的链式计算互相践踏。
- 为什么不用 `std::array` + 手工偏移的 RawBuf：会话内未验证；TBuf 展开虽然啰嗦但每块都是编译期独立的 UB 分配。
- 队列深度 1→2 与独立 queTmp 的失败尝试说明问题不在"槽位不够"，而在用法语义本身。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

以下信号来自挂起问题的二分循环（ascend950 + CANN 9.0.0-beta.2，每轮构建 SUCCESS）：

- 修复前（queIn.AllocTensor 作临时）：首用例挂起（600 秒 shell 超时，无任何用例输出）（来源平台：950；证据等级：tool_observed）。
- 对照（Abs-only，不 AllocTmp）：14/14 完成，输出 |x| 数值正确（来源平台：950；证据等级：tool_observed）。
- 反证（恢复 queIn.AllocTensor + depth 2；独立 queTmp 队列）：两种变体均再次挂起（来源平台：950；证据等级：tool_observed）。
- 修复后（TBuf）：简化 Compute（AllocTmp→TBuf）14/14 完成；终版 16-TBuf 全计算链重写后 14/14 用例全部完成、无崩溃无挂起（数值正确性属另一问题）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 16 块 TBuf + 双队列的 UB 总占用 = 18 × ubFormer × 4 字节，依赖 ubFormer cap（本会话 ubFormer=1024，共 72KB）；ubFormer 变大时须重新核对 UB 预算。
- 挂起的确切机制（队列事件死锁假设）未验证；其他 CANN 版本行为未知（推断可迁移，须实测）。
- `TBuf::Get<T>()` 同块别名是设计事实，依赖它的"复用同一 TBuf 当多个临时"的写法必然数据竞争。

**未决项**：

- 【优先项】复现并追根因到机制层：queIn.AllocTensor 临时用法的确切挂起机制（队列事件等待/深度语义），用最小复现工程固化。
- TBuf 数量上限与 InitBuffer 分配粒度（是否 32B 对齐、有无 per-buffer 开销）未验证。
## 坏实践

- 把 `TQue::AllocTensor/FreeTensor` 当 malloc/free 用（DeQue 输入后在同一队列上反复 Alloc 临时）。
- 用单个大 TBuf 上多次 `Get<T>()` 冒充多个独立临时（别名覆盖）。
- 独立 TQue 专供临时（实测同样挂起，不是解法）。

## 误诊

- 挂起初期先后怀疑：ubFormer 过大（继续降 cap 无效）、Process 迭代间流水冒险（PipeBarrier 无效）、计算链中某个算术指令（Abs-only 排除）——直到"最简 AllocTmp 版也挂、Abs-only 不挂"的对照才锁定分配方式。
- 曾把 TBuf 过渡版的成功误读为"流水线修好了"，随后又把 AllocTmp 换回 queIn（depth 2）验证，再次挂起才彻底放弃队列临时分配。
