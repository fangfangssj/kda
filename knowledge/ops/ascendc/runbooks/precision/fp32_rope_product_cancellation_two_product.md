---
type: "Runbook"
title: "fp32 乘积相消 MARE 超阈值 — 用补偿乘积保留低位"
description: "两个大幅值 fp32 乘积先各自舍入再相减会放大近零结果的相对误差；对 fp32 路径使用 Split/TwoProduct 补偿后恢复精度。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.arithmetic"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# fp32 乘积相消 MARE 超阈值 — 用补偿乘积保留低位

## 现象

经典 Ascend C 的 RoPE kernel 对 fp32 输入计算 `lo*cos - hi*sin` 与 `hi*cos + lo*sin`。大多数正式用例通过，只有极值输入中的少量相消点失败：整体绝对误差指标仍约为 `1e-7`，但 query/key 的 MARE 分别达到 `0.073104` 和 `0.284128`，超过 `0.05` 阈值。

判别信号是“绝对误差很小、失败集中在接近零的加减结果、朴素 CPU fp32 也能复现高相对误差”，而不是广泛随机 mismatch。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`Dekker`、`TwoProduct`、`Veltkamp`、`catastrophic cancellation` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

朴素路径先把 `lo*cos` 与 `hi*sin` 各自舍入为 fp32，再做减法。当两个乘积幅值大且接近时，主要有效位在相减中消失，两个乘积已经丢失的低位却无法恢复；最终结果接近零，因而很小的绝对误差会变成很大的相对误差。

同输入上的 CPU fp32 模拟也复现了这一模式，排除了 Ascend C 搬运或索引错误。修复切面是 fp32 数值策略：把每个乘积分解为舍入主项与误差项，再把两组主项和误差项分别合并。

## 解决配方

### before

```cpp
    __aicore__ inline void RopeCompute(const AscendC::LocalTensor<float>& xd,
                                       const AscendC::LocalTensor<float>& cF,
                                       const AscendC::LocalTensor<float>& sF,
                                       const AscendC::LocalTensor<float>& outF)
    {
        AscendC::LocalTensor<float> t1 = t1Buf_.Get<float>();
        AscendC::LocalTensor<float> t2 = t2Buf_.Get<float>();
        AscendC::Mul(t1, xd, cF, h_);          // lo*cos
        AscendC::Mul(t2, xd[h_], sF, h_);      // hi*sin
        AscendC::Sub(outF, t1, t2, h_);        // out[0:h]
        AscendC::Mul(t1, xd[h_], cF, h_);      // hi*cos
        AscendC::Mul(t2, xd, sF, h_);          // lo*sin
        AscendC::Add(outF[h_], t1, t2, h_);    // out[h:D]
    }
```

### after

```cpp
constexpr float VELTKAMP_K = 4097.0f;  // 2^12 + 1 for fp32 (24-bit mantissa) split

    __aicore__ inline void Split(const AscendC::LocalTensor<float>& src,
                                 const AscendC::LocalTensor<float>& hi,
                                 const AscendC::LocalTensor<float>& lo)
    {
        AscendC::LocalTensor<float> tw = twBuf_.Get<float>();
        AscendC::Muls(tw, src, VELTKAMP_K, h_);  // tw = K*src
        AscendC::Sub(hi, tw, src, h_);           // tw - src
        AscendC::Sub(hi, tw, hi, h_);            // hi = tw - (tw - src)
        AscendC::Sub(lo, src, hi, h_);           // lo = src - hi
    }

    // p + e == a*b exactly (fp32). p = round(a*b), e = error.
    __aicore__ inline void TwoProduct(const AscendC::LocalTensor<float>& a,
                                      const AscendC::LocalTensor<float>& b,
                                      const AscendC::LocalTensor<float>& p,
                                      const AscendC::LocalTensor<float>& e)
    {
        AscendC::LocalTensor<float> ah = ahBuf_.Get<float>();
        AscendC::LocalTensor<float> al = alBuf_.Get<float>();
        AscendC::LocalTensor<float> bh = bhBuf_.Get<float>();
        AscendC::LocalTensor<float> bl = blBuf_.Get<float>();
        AscendC::LocalTensor<float> tw = twBuf_.Get<float>();
        AscendC::Mul(p, a, b, h_);
        Split(a, ah, al);
        Split(b, bh, bl);
        // e = al*bl - (((p - ah*bh) - al*bh) - ah*bl)
        AscendC::Mul(tw, ah, bh, h_);
        AscendC::Sub(e, p, tw, h_);
        AscendC::Mul(tw, al, bh, h_);
        AscendC::Sub(e, e, tw, h_);
        AscendC::Mul(tw, ah, bl, h_);
        AscendC::Sub(e, e, tw, h_);
        AscendC::Mul(tw, al, bl, h_);
        AscendC::Sub(e, tw, e, h_);
    }

    // out = u*v - w*z   (length h, accurate)
    __aicore__ inline void AccMulSub(const AscendC::LocalTensor<float>& out,
                                     const AscendC::LocalTensor<float>& u,
                                     const AscendC::LocalTensor<float>& v,
                                     const AscendC::LocalTensor<float>& w,
                                     const AscendC::LocalTensor<float>& z)
    {
        AscendC::LocalTensor<float> p1 = p1Buf_.Get<float>();
        AscendC::LocalTensor<float> e1 = e1Buf_.Get<float>();
        AscendC::LocalTensor<float> p2 = p2Buf_.Get<float>();
        AscendC::LocalTensor<float> e2 = e2Buf_.Get<float>();
        AscendC::LocalTensor<float> hi = hiBuf_.Get<float>();
        AscendC::LocalTensor<float> lo = loBuf_.Get<float>();
        TwoProduct(u, v, p1, e1);
        TwoProduct(w, z, p2, e2);
        AscendC::Sub(hi, p1, p2, h_);
        AscendC::Sub(lo, e1, e2, h_);
        AscendC::Add(out, hi, lo, h_);
    }

    // out = u*v + w*z   (length h, accurate)
    __aicore__ inline void AccMulAdd(const AscendC::LocalTensor<float>& out,
                                     const AscendC::LocalTensor<float>& u,
                                     const AscendC::LocalTensor<float>& v,
                                     const AscendC::LocalTensor<float>& w,
                                     const AscendC::LocalTensor<float>& z)
    {
        AscendC::LocalTensor<float> p1 = p1Buf_.Get<float>();
        AscendC::LocalTensor<float> e1 = e1Buf_.Get<float>();
        AscendC::LocalTensor<float> p2 = p2Buf_.Get<float>();
        AscendC::LocalTensor<float> e2 = e2Buf_.Get<float>();
        AscendC::LocalTensor<float> hi = hiBuf_.Get<float>();
        AscendC::LocalTensor<float> lo = loBuf_.Get<float>();
        TwoProduct(u, v, p1, e1);
        TwoProduct(w, z, p2, e2);
        AscendC::Add(hi, p1, p2, h_);
        AscendC::Add(lo, e1, e2, h_);
        AscendC::Add(out, hi, lo, h_);
    }

    __aicore__ inline void RopeComputeAcc(const AscendC::LocalTensor<float>& xd,
                                          const AscendC::LocalTensor<float>& cF,
                                          const AscendC::LocalTensor<float>& sF,
                                          const AscendC::LocalTensor<float>& outF)
    {
        AscendC::LocalTensor<float> lo = xd;
        AscendC::LocalTensor<float> hi = xd[h_];
        AccMulSub(outF, lo, cF, hi, sF);       // out[0:h] = lo*cos - hi*sin
        AccMulAdd(outF[h_], hi, cF, lo, sF);   // out[h:D] = hi*cos + lo*sin
    }
```

## 设计决策

补偿路径只用于 fp32；fp16/bf16 仍先提升到 fp32 后走朴素乘加。这把额外 UB 缓冲和向量指令集中到实际发生相消精度风险的 dtype，避免把昂贵路线无条件扩散。

## 性能影响

补偿路径增加多组临时张量、拆分和乘加指令，预期比朴素 fp32 路径更慢、占用更多 UB。本轨迹后续又同时改变了分组和搬运结构，不能把最终性能变化单独归因于补偿算术；采纳时必须在同平台、同负载下重新做配对性能测量。

## 预防

- 对 `a*b ± c*d` 且结果可能接近零的表达式，除全局 MERE 外单独检查相消点的相对误差；
- 先用同输入的 CPU fp32 与 fp64 对照：若朴素 fp32 已复现，优先查数值策略，不先改搬运和索引；
- 把补偿路线按 dtype 或明确幅值条件门控，并同时核算 UB 容量。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：定向失败用例的 query MARE=0.073104、key MARE=0.284128。（来源平台：a3；证据等级: tool_observed）。
- 修复后：AFTER 评测 exit=0，20 passed。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

本卡只证明该 Split/TwoProduct 组合在本轨迹 RoPE 表达式、输入范围与 a3 路径上通过。常量选择、溢出范围、次正规数、不同编译器收缩语义，以及 a2/950 均未验证；不得将代码注释中的“exact”解释成所有 fp32 输入上的平台无关证明。若误差来自地址、广播或尾块，应回到相应 kernel 因果链。

## 坏实践

- 只看 MERE 很小就把失败归为评测器噪声；相消点的 MARE 可能仍真实超阈；
- 为通过单个用例放宽容差或修改随机输入；这不会恢复被乘法舍入丢失的低位；
- 对全部 dtype 无条件启用补偿路径而不评估 UB 和性能成本。

## 误诊

若朴素 CPU fp32 不能复现，或 mismatch 分布广泛而非集中在近零结果，应优先排查 kernel 的布局、广播、搬运和未初始化问题，而不是直接套用补偿算术。
