---
type: "Runbook"
title: torch fp32 sum(dim=0)逐列序：大N稳定为casc16级联
description: torch CPU fp32 sum(dim=0)的逐列累加序随work与列数N变化：work≤~2048纯顺序、≤~16K为16行块级联、大N实际shape（work 20万+）下逐列恒为casc16且跨线程数稳定；预判golden误差或评估bitwise对齐可行性前必读。
tags: [task.testing, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# torch fp32 sum(dim=0)逐列序：大N稳定为casc16级联


## 现象

在排查 Ascend950PR 上某融合反向算子 bias 归约的 mare 超差时（golden 为 torch fp32 `t.sum(dim=0)`，t 是 (B,N) fp32 连续张量），需要回答"golden 沿 batch 归约时每一列到底按什么顺序累加"。初期用 1 线程 + 小 N（64 列）探针得到：B≤17 匹配纯顺序、B=24~256 匹配"16 行块内顺序+块间顺序合并"、B≥2048（work=B·N≥131072）任何候选都不匹配——一度得出"大 work 序未破解、疑似线程依赖"的印象。随后改用**真实 ST golden 位流**（实际 (B,N)=(32,6144)/(64,6528)/(96,12480)/(256,6528)，work 19.7 万~167 万）回验，发现逐列序就是 **casc16**（16 行顺序块 + 块间顺序合并），且 1/8/64 线程下位级不变——初期小 N 探针的结论不能外推。

## 适用场景

- 工程形态：AscendC 算子 ST/精度调试的 golden 侧分析（aclnn 算子仓或 kernel 直调的数值验收）。
- API 路径：torch CPU `sum(dim=0)` / ATen SumKernel（golden 生成脚本通常直接调用）；对 NPU 侧的意义在于预测 golden 误差、评估 kernel bitwise 对齐可行性、设计 golden。
- 验证过的平台与 CANN 版本：golden 在 x86 host（torch 2.7.1+cpu，默认 64 线程，未 pin）生成；被比对对象是 Ascend950PR 实机 kernel 输出（与 frontmatter `platforms` 一致；host 侧探针在失效边界中单独说明）。
- 适用粒度（transfer_scope）：platform_wide（golden 侧行为与 NPU 型号无关；只要 golden 用 torch CPU fp32 sum 就适用）。
- 路线优先级：recommended_fastpath（做三方对比/评估对齐可行性时的标准前置知识）。
- 明确不适用/未验证：torch ≠ 2.7.1（SumKernel 实现可能变）；GPU torch；work>131072 且 N 很小（~64 列级）的行分裂序（未破解、线程依赖）；mean/prod 等其他归约。

## 判别方法

**探测原理**：对给定 (B,N) 构造随机 fp32 矩阵，比较 torch `sum(dim=0)` 结果与候选序的**位级**相等性（不是数值接近——求和序判定只有 0/1）。必须用 golden 的真实 (B,N) 形态与真实 golden 位流做最终确认，不能只用自造小 shape。

**观测结果（全部位级判定）**：

| 条件 | 逐列累加序 |
|---|---|
| work=B·N ≤ ~2048（N=64，B≤17，1 线程） | 纯顺序 (((x0+x1)+x2)+…) |
| ~2048 < work ≤ ~16384（N=64，B=24~256，1 线程） | 16 行块内顺序、块间顺序合并（casc16） |
| work ≥ 131072 且 N=64（B≥2048，1 线程） | 上述候选全不匹配（行分裂，序未破解、线程相关） |
| 实际 golden 形态 N=6144~12480（work 19.7 万~167 万），线程 1/8/64 | **casc16，位级不变**（与真实 golden 位流逐位相等，四形态全对） |
| B∈{2,3,5,6}，N=64 | 纯顺序 |

**机制解释**（对应 ATen `aten/src/ATen/native/cpu/SumKernel.cpp` v2.7.1）：`sum(dim=0)` 走 TensorIterator reduction 的 `cascade_sum` + `multi_row_sum`（`num_levels=4` 的自适应级联，`level_power = max(4, CeilLog2(size)/num_levels)`，块内顺序累加、各级部分和顺序合并）并经 `parallel_reduce` 派发。列数 N 大时线程并行拆的是**输出列**（每列仍由单线程对全部 B 行跑完整级联），对 B≤256、级联块 16 行的情形级联退化为 casc16 且与线程数无关；N 小而 work 大时 reduction 沿**行**分裂成 chunk（两遍式归约），逐列序变为 chunk 边界/线程数依赖。golden 生成脚本不 pin 线程时用默认线程（本例 64），大 N 下不影响逐列序。

**判别含义**：
- 想预判 golden 的 fp32 噪声水平：把 golden 建模为 casc16（大 N 场景），其 vs fp64 的 mare 可在 host 离线复算；
- 想让 kernel bitwise 对上 golden：唯一途径是实现 casc16（16 行顺序块+顺序合并），任何树形/分块拓扑都到不了（相消列上差异 ≥ 数 ULP）；
- golden 复现性风险点在小 N 大 B 形态（序线程依赖），不在大 N 形态。

## 完整代码

work 阈值与块序探测（可复用；位级判定候选序）：

```python
#!/usr/bin/env python3
"""Probe torch sum(dim=0) order: find where sequential stops matching, test block schemes."""
import numpy as np
import torch

torch.set_num_threads(1)

def seq_sum(x):
    acc = x[0].copy()
    for i in range(1, x.shape[0]):
        acc = (acc + x[i]).astype(np.float32)
    return acc

def blocks_seq(x, bs):
    """Sum each aligned block of bs rows sequentially, then combine blocks sequentially."""
    n = x.shape[0]
    accs = [seq_sum(x[s:s+bs]) for s in range(0, n, bs)]
    out = accs[0]
    for a in accs[1:]:
        out = (out + a).astype(np.float32)
    return out

def blocks_halves(x):
    n = x.shape[0]
    if n <= 1:
        return x[0] if n else None
    h = n // 2
    return (blocks_halves(x[:h]) + blocks_halves(x[h:])).astype(np.float32)

print("== work threshold probe (N=64, varying B) ==")
for B in [2,3,4,5,6,7,8,12,16,17,24,31,32,33,48,64,96,256,2048,2049,4096,8192,16384]:
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(B, 64, generator=g) * 4 - 2).numpy().astype(np.float32)
    t = torch.from_numpy(x).sum(dim=0).numpy()
    s = seq_sum(x)
    tag = "SEQ" if np.array_equal(t, s) else "???"
    extra = ""
    if tag == "???":
        for bs in [2,4,8,16,32,64,128,256]:
            if np.array_equal(t, blocks_seq(x, bs)):
                extra = f" blocks_seq({bs})"
                break
        if not extra and np.array_equal(t, blocks_halves(x)):
            extra = " halves"
    print(f"B={B:5d} work={B*64:7d}: {tag}{extra}")

print("== B=32 with varying N (work threshold) ==")
for N in [64, 256, 1024, 4095, 4096, 4097, 6528, 12480]:
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(32, N, generator=g) * 4 - 2).numpy().astype(np.float32)
    t = torch.from_numpy(x).sum(dim=0).numpy()
    s = seq_sum(x)
    print(f"N={N:6d} work={32*N:7d}: {'SEQ' if np.array_equal(t, s) else '???'}")
```

线程稳定性探测（对 golden 的真实 (B,N) 形态验证 casc16 跨线程不变）：

```python
import numpy as np, torch

def seq(X):
    acc = X[0].copy()
    for i in range(1, X.shape[0]):
        acc = (acc + X[i]).astype(np.float32)
    return acc

def casc16(X):
    B = X.shape[0]
    parts = [seq(X[s:s+16]) for s in range(0, B, 16)]
    out = parts[0]
    for p in parts[1:]:
        out = (out + p).astype(np.float32)
    return out

for B, N in [(32, 6144), (64, 6528), (96, 12480), (256, 6528)]:
    g = torch.Generator().manual_seed(42)
    x = (torch.rand(B, N, generator=g) * 4 - 2).numpy().astype(np.float32)
    ref = casc16(x)
    line = f"B={B:3d} N={N:5d}: "
    for nt in [1, 8, 64]:
        torch.set_num_threads(nt)
        t = torch.from_numpy(x).sum(dim=0).numpy()
        line += f"threads={nt}:{'casc16' if np.array_equal(t, ref) else 'DIFFER'}  "
    print(line)
```

## 设计决策

- 判定"kernel 能否 bitwise 对上 golden"时，用**真实 golden 位流**（ST 的 exp_*.bin）回验候选序，而不是只用自造随机数+pin 线程的探针——探针条件（线程数、N）与 golden 实际生成条件不一致时会得出错误结论（本例初期即如此，险些误判"无法对齐"）。
- golden 侧分析属于验收/评测层（solution_layer: test）；不据此改 kernel——bitwise 复刻 golden 库实现属于被禁止的偏离路径（见 bias mare 定界相关知识）。

## 预防

- golden 生成脚本要么固定并记录 torch 版本（求和序是实现细节），要么 bias 归约直接用 fp64（如 `x.double().sum(dim=0)` 再单次收窄）从根上消掉序噪声。
- ST shape 家族若含小 N 大 B 的归约用例，注意 golden 序线程依赖（跨机重生成 golden 可能不可复现）。

## 验证信号

来源说明：以下为对真实 ST golden 位流（torch 2.7.1+cpu、默认 64 线程生成）与 host 探针的位级比对；casc16 判定同时以"与 NPU 实机 kernel 输出的三方对比"为语境。

- 实际 golden 形态 (32,6144)/(64,6528)/(96,12480)/(256,6528)：casc16 与 golden 位级相等（mare=0，四形态全对）（来源平台：950 ST 全链路，golden 于 x86 host 生成；证据等级：tool_observed）
- 线程稳定性：上述四形态在 `set_num_threads(1/8/64)` 下均为 casc16（来源平台：x86 host golden 侧探针，非上板信号；证据等级：tool_observed）
- work 阈值：N=64 时 B≤17 SEQ、B=24~256 blocks_seq(16)、B≥2048（work≥131072）无候选匹配（来源平台：x86 host golden 侧探针，非上板信号；证据等级：tool_observed）
- B∈{2,3,5,6}、N=64：纯顺序位级相等（来源平台：x86 host golden 侧探针，非上板信号；证据等级：tool_observed）
- ATen 源码核对：`multi_row_sum` 为 num_levels=4 自适应级联（level_power=max(4, CeilLog2(size)/4)）、`cascade_sum` 经 `parallel_reduce` 派发（v2.7.1 原文获取）（来源平台：golden 生成环境 torch 2.7.1 源码；证据等级：tool_observed）

## 失效边界

- 绑定 torch 2.7.1+cpu（ATen SumKernel 的 multi_row_sum 级联 + parallel_reduce）；torch 升级可能改级联参数（历史上变过），换版本须重跑探针。
- 小 N（~64 列级）大 work 的行分裂序未破解且线程依赖；B 非 16 倍数时 casc16 的尾块处理未单独验证（本组 B 全为 16 倍数）。
- 探针在 x86 host 执行；逐列序由 multi_row_sum 结构决定的结论有源码级解释支撑，但未在其他架构 host 上验证。GPU torch 完全未验证。

## 坏实践

- 假设 torch sum 是纯顺序（对 work>2048 直接错）或假设"pin 了线程数序就固定"（小 N 行分裂仍随线程变）。
- 用与 golden 生成条件不一致的探针（1 线程+小 N）外推 golden 序并下"无法对齐"结论。
- golden 与 kernel 的 mare 差异未定界前就动手改 kernel 归约序。

## 误诊

- 初期结论"work≥131072 的 torch 序未破解、疑似线程相关"只在小 N 探针下成立；对实际大 N golden 形态回验后推翻（casc16 稳定）。教训：探针矩阵必须覆盖 golden 的真实 (B,N) 形态，最终判定要用真实 golden 位流。

## 未决事项

- work>131072 且小 N 的行分裂分块公式（TensorIterator two_pass_reduction 的触发条件与 chunk 边界）未做源码级确认。
- torch 求和序跨版本稳定性未系统跟踪；B 非 16 倍数形态的 casc16 尾块语义未验证。
