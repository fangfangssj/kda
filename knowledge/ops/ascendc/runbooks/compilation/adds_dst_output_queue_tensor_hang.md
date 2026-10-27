---
type: "Runbook"
title: "Adds 写输出队列张量挂起 — 改写 TBuf"
description: "AscendC::Adds(dst=queOut 的 yLocal, src, 标量) 使 kernel 挂起（构建 SUCCESS + 首用例超时）；同目标 Muls/Abs 正常、同指令写 TBuf 正常——判别式是指令 × 目标缓冲类别组合。规避：标量加法先落 TBuf 再用向量指令拷出到 yLocal。"
tags: [task.debugging, topic.memory, topic.compute.elementwise.arithmetic]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# Adds 写输出队列张量挂起 — 改写 TBuf

## 现象

bessel_i1e 类算子（Branch-0 FP32）kernel 在临时内存改用 TBuf 之后（挂起问题已解，见 [TQue AllocTensor 作临时分配致 kernel 挂起卡](tque_alloctensor_temp_alloc_hangs.md)），为计算 `y = |x|*0.5 - 2.0` 做逐指令二分时发现一条诡异约束。Compute 用 `yLocal = queOut.template AllocTensor<T>()` 作输出、`a/t/t2` 为 TBuf 张量：

版本 1（两步，Muls 写 yLocal）——正常：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a = BufTensor(bufA);
        AscendC::Abs(a, xLocal, c);
        AscendC::Muls(yLocal, a, T(0.5f), c);

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

run.sh：`0-A got 0.163986206`（= |0.32797|×0.5 ✓），14 用例全部完成。

版本 2（三步，Adds 写 yLocal，无论源是谁）——挂起：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a = BufTensor(bufA);
        AscendC::Abs(a, xLocal, c);
        // y = a * 0.5 - 2.0
        AscendC::Muls(yLocal, a, T(0.5f), c);
        AscendC::Adds(yLocal, yLocal, T(-2.0f), c);      // 挂起

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a = BufTensor(bufA);
        auto t = BufTensor(bufTmp);
        AscendC::Abs(a, xLocal, c);
        AscendC::Muls(t, a, T(0.5f), c);
        AscendC::Adds(yLocal, t, T(-2.0f), c);           // 独立源，同样挂起

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

两版均为构建 SUCCESS + 首用例无输出 + shell 超时。

版本 3（对照，Abs 写 yLocal）——正常：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a = BufTensor(bufA);
        auto t = BufTensor(bufTmp);
        AscendC::Abs(a, xLocal, c);
        AscendC::Muls(t, a, T(0.5f), c);
        // Just copy t to yLocal using Duplicate
        AscendC::Abs(yLocal, t, c);

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

版本 4（修复，Adds 写 TBuf 再拷出）——正常且数值正确：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a  = BufTensor(bufA);
        auto t  = BufTensor(bufTmp);
        auto t2 = BufTensor(bufOnes);
        AscendC::Abs(a, xLocal, c);
        AscendC::Muls(t, a, T(0.5f), c);
        // Write result to t2 (TBuf), then copy to yLocal
        AscendC::Adds(t2, t, T(-2.0f), c);
        AscendC::Abs(yLocal, t2, c);

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

run.sh：`0-A got 1.83601379`（= ||0.32797|×0.5 − 2.0| ✓）。

## 适用场景

- 工程形态：算子仓 kernel 开发（AscendC 标准 API 计算链，TBuf 中间量 + TQue 输出）。
- API 路径：`AscendC::Adds(dst, src, scalar, count)`（标量加）与 `AscendC::Muls`/`AscendC::Abs` 的目标操作数差异。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——ascend950 上 AscendC 标准 API kernel 的输出写入模式。
- 路线优先级：safety_fallback——机制未明，规避写法多一条拷贝指令（相对代价：一次额外向量读改写，未测量，直觉可忽略）。**fallback 声明**：本卡是规避路线，不是根因修复；根因定位后应以官方说明为准。
- 明确不适用 / 未验证场景：`Adds` 写 TBuf/写 queIn 张量的其他组合未逐一枚举；VF（regbase）路径的 Adds 受 hoisted 约束另行处理。

## 根因

机制未追到（无 plog 细化信息）。经验事实（二分矩阵）：`Adds(标量)` 以 **queOut 队列张量（AllocTensor 得到的 yLocal）为目标** 时 kernel 挂起；同目标下 `Muls`/`Abs` 正常；同指令 `Adds` 改以 TBuf 为目标正常。判别式：指令 × 目标缓冲类别的组合，而非"输出队列不可写"或"标量指令不可用"这类单因子。

## 判别方法

怀疑该问题时，按如下矩阵各跑一轮（构建 SUCCESS + 首用例超时 = 挂起）：

| 指令 | dst=TBuf | dst=yLocal(queOut) |
|---|---|---|
| Muls(标量) | 正常 | 正常 |
| Abs(向量) | 正常 | 正常 |
| Adds(标量) | 正常 | **挂起** |

## 解决配方

修复 = "标量 `Adds` 先落 TBuf，再用向量指令拷出到输出队列张量"（版本 4，verbatim 见现象节）。拷出指令选 `Abs` 是因为源恒非负场景下的零代价搬运；需要保号时可用 `Muls(yLocal, t2, T(1.0f), c)` 恒等乘。

## 设计决策

- 为什么用"多一条指令的绕行"而不是深挖：会话目标是恢复全链可测；挂起无日志可查，二分矩阵已给出可复用规避模式。
- 为什么判定不是自读自写（RAW）问题：独立源形态（`Adds(yLocal, t, -2.0f)`）同样挂起，排除别名因子。
- 当前方案路线档位：safety_fallback（额外一次向量搬运，代价未测量）。

## 验证信号

以下信号来自逐指令二分循环（ascend950 + CANN 9.0.0-beta.2，每轮构建 SUCCESS）：

- `Muls(yLocal, a, 0.5)`：14/14 完成，`0-A got 0.163986206` = |0.32797|×0.5 ✓（来源平台：950；证据等级：tool_observed）。
- `Adds(yLocal, yLocal, -2.0)`：首用例挂起（无输出、shell 超时）（来源平台：950；证据等级：tool_observed）。
- `Adds(yLocal, t, -2.0)`（独立 TBuf 源）：同样挂起（来源平台：950；证据等级：tool_observed）。
- `Abs(yLocal, t)`：14/14 完成（来源平台：950；证据等级：tool_observed）。
- 修复 `Adds(t2, t, -2.0)` + `Abs(yLocal, t2)`：14/14 完成，`0-A got 1.83601379` = ||0.32797|×0.5−2.0| ✓（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只枚举了 Muls/Abs/Adds × {TBuf, yLocal} 的最小矩阵；`Subs`（标量减）、其他标量指令写 yLocal 的行为未验证（推断同类风险，须实测）。
- 机制（标量加指令对输出队列缓冲的目标约束）未定位；其他 CANN 版本/平台不可直接搬用（未验证，推断可迁移）。
- 绕行指令假设源数据可被 Abs/Muls(1.0) 无损搬运（fp32 成立；带 NaN 语义时 Muls(1.0) 保号但 Abs 改符号，按需选择）。

**未决项**：

- 【优先项】复现并追根因到机制层：定位 Adds×queOut 缓冲组合的挂起点（指令发射/队列事件），向工具链报障或找到官方说明。
- `Subs/Maxs/Mins` 等其他标量指令写 yLocal 的行为枚举（未尝试）。
## 坏实践

- 标量 `Adds` 直写输出队列张量（本会话两种源形态均挂起）。
- 把"某指令挂起"泛化成"输出队列不能写"（Muls/Abs 写 yLocal 正常）或"标量指令不能用"（Adds 写 TBuf 正常）。

## 误诊

- 第一轮以为是 `Adds(yLocal, yLocal, ...)` 的自读自写引发 RAW 挂起；换独立源后仍挂起，才把因子收敛到"指令 × 目标类别"。
