---
type: "Runbook"
title: golden 归约顺序与 kernel 不一致放大 near-zero MARE
description: PyTorch golden 用 torch.nn.functional.conv2d（im2col+GEMM 归约顺序），与 kernel/C++ golden 的 (ci,kh,kw) 逐项累加顺序不同：输入数据完全一致时，near-zero 输出因 fp32 舍入路径不同产生 ~3e-5 绝对差，MARE 放大到 3.1e-2 超阈。显式按 kernel 累加顺序重写 golden 后同一批用例全部通过。
tags: [task.testing, topic.precision, topic.compute.neighborhood.convolution]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# golden 归约顺序与 kernel 不一致放大 near-zero MARE

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子 BenchNoHarnessConv2d 跑 PyTorch ST 批量测试。输入随机流（CppRNG 移植）、比较器自适应 epsilon、golden fp32 计算全部对齐后，L0 单用例 L0-001 已通过；但 L0 全量 60 条仍有 **10 条 fp16 大计算量用例失败**（L0-007/008/009/010/021/032/033/034/043/056——共同特征是大 C_in/大空间/大 batch，即累加次数多的用例）。以 L0-007（x[4,128,32,32] filter[256,128,3,3] fp16，每输出元素累加 128×3×3=1152 次）为例：

```
MERE: 1.768238e-04 (阈值: 9.765625e-04) PASS
MARE: 3.103226e-02 (阈值: 9.765625e-03) FAIL
Top 5 MARE positions (epsilon=9.765625e-04):
  [370259] golden=0.00004143, actual=0.00000983, abs_diff=3.15904617e-05, rel_err=3.10322624e-02
  [221486] golden=0.00030541, actual=0.00034165, abs_diff=3.62396240e-05, rel_err=2.82685515e-02
  [436798] golden=0.00149059, actual=0.00144577, abs_diff=4.48226929e-05, rel_err=1.81677621e-02
```

失配位置全部是 **near-zero 输出**（golden≈4e-5~3e-3），绝对差只有 3e-5~5e-5（fp32 累加噪声量级），MERE（均值）1.77e-4 与 C++ ST 完全一致——即数据一致、平均精度一致，只有"谁的最大相对误差"不一致。C++ ST 同用例 `[PASS] MERE=1.76e-04, MARE=4.88e-04`。

## 适用场景

- 工程形态：aclnn 算子 ST 测试的 golden 移植（C++ 手写卷积 golden → PyTorch），以及任何"参考实现与被测实现归约顺序可不同"的精度比对场景。
- API 路径：`torch.nn.functional.conv2d`（内部 im2col+GEMM，归约顺序未定义）vs 手写 (ci,kh,kw) 逐项累加；MARE（最大相对误差）对 near-zero 元素的敏感性。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）；golden 在 CPU 侧计算，kernel 在 NPU 侧。
- 适用粒度：operator_family——凡"多项乘加的算子 + near-zero 输出 + 最大相对误差判据"的组合都适用（卷积/归约/点积类）；本会话在 950 上验证。
- 路线优先级：recommended_fastpath（golden 按 kernel 累加顺序显式重写，判定即对齐）。
- 明确不适用 / 未验证场景：阈值判据只用均值误差（MERE）或绝对误差的场景不受累加顺序影响；fp32 dtype 用例（本批 fp32 7/7 全过，因 fp32 精度余量大 + fp32 比较 epsilon=0.1 吸收）未触发。

## 根因

浮点加法不结合：`(a+b)+c ≠ a+(b+c)`。同一组完全相同的输入，golden 的归约顺序决定每个输出元素的舍入轨迹。本例三条实现路径的累加顺序：

- kernel（NPU）：按 (ci, kh, kw) 顺序逐项累加进标量 sum；
- C++ golden（`ComputeConv2dGolden`）：同样 (ci, kh, kw) 三重循环逐项累加（`float sum = bp[co]; for ci { for kh { for kw { sum += xv * fv; } } }`）——与 kernel 顺序一致，故 C++ ST 的 MARE 只有 4.88e-4；
- PyTorch golden（`torch.nn.functional.conv2d`）：内部 im2col + GEMM，归约按 BLAS 分块顺序进行——与 kernel 顺序不同。

对量级正常（|golden|≈O(1)~O(100)）的输出元素，~1e-5 的绝对差产生 ≤1e-3 的相对误差，不超阈；但 near-zero 输出（golden≈4.14e-5）本身就是大数相消的产物，两条舍入路径的差 3.16e-5 在分母 (|golden|+epsilon≈9.8e-4) 下被放大为 rel_err=3.16e-5/9.8e-4≈3.2e-2——正是观测的 MARE=3.10e-2。累加次数越多（L0-007 每元素 1152 次），相消出 near-zero 的概率越大，所以失败集中在计算量大的用例。

与库内相邻根因的区分：[大归约维 fp32 golden 顺序累加致假精度失败卡](fp32_golden.md)是 golden 自身累加精度不足（顺序 fp32 vs 树形，需 double 累加修复）；本卡是两条"各自都足够准"的路径因**归约顺序不同**在 near-zero 处分歧（需按 kernel 顺序重写修复）。[fp16 次正规 1-ULP 差被 mare 放大判别卡](fp16_subnormal_1ulp_mare_amplification_diagnosis.md)则是硬件 1-ULP 差被 FLOOR 分母放大，属非缺陷；本卡的差异来自 golden 实现选择，可消除。

## 解决配方

golden 按 kernel 的 (ci, kh, kw) 顺序显式重写（PyTorch 向量化实现，每个输出元素累加顺序与 C++/kernel 逐项一致）：

```python
def compute_golden_conv2d_ordered(
    x: torch.Tensor,
    filter: torch.Tensor,
    bias: torch.Tensor,
    strides: list,
    pads: list,
    dilations: list = [1, 1],
) -> torch.Tensor:
    """
    计算 2D 卷积的 golden 结果（与 C++ ST 测试完全一致的累加顺序）

    C++ ComputeConv2dGolden 按 (ci, kh, kw) 顺序逐个累加：
      for ci: for kh: for kw: sum += x[ci,hi,wi] * filter[co,ci,kh,kw]

    PyTorch conv2d 内部使用 im2col + GEMM，累加顺序不同，
    导致 near-zero 输出位置的 golden 值有差异（~1e-5 级别），
    在 MARE 比对时被放大为较大的相对误差。

    本函数使用 PyTorch 张量操作按相同的 (ci, kh, kw) 顺序累加，
    确保与 C++ golden 在 bit 级别一致。
    """
    # 在 fp32 下计算
    x_f32 = x.to(torch.float32)
    f_f32 = filter.to(torch.float32)
    b_f32 = bias.to(torch.float32)

    N, C_in, H, W = x_f32.shape
    C_out, _, K_h, K_w = f_f32.shape
    sh, sw = strides[0], strides[1]
    pt, pb, pl, pr = pads[0], pads[1], pads[2], pads[3]
    dh, dw = dilations[0], dilations[1]

    H_out = (H + pt + pb - dh * (K_h - 1) - 1) // sh + 1
    W_out = (W + pl + pr - dw * (K_w - 1) - 1) // sw + 1

    # 使用 F.pad 进行 padding（对称/非对称统一处理）
    x_padded = torch.nn.functional.pad(x_f32, (pl, pr, pt, pb))

    # 按 (ci, kh, kw) 顺序累加（与 C++ ComputeConv2dGolden 一致）
    y = b_f32.view(1, C_out, 1, 1).expand(N, C_out, H_out, W_out).clone()
    for ci in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # 提取 x patch: x_padded[n, ci, ho*sh+kh*dh, wo*sw+kw*dw]
                ho_idx = torch.arange(H_out, device=x_f32.device) * sh + kh * dh
                wo_idx = torch.arange(W_out, device=x_f32.device) * sw + kw * dw
                # x_patch: [N, 1, H_out, W_out]
                x_patch = x_padded[:, ci:ci+1, ho_idx, :][:, :, :, wo_idx]
                # f_val: [C_out, 1, 1, 1] -> broadcast
                f_val = f_f32[:, ci:ci+1, kh, kw].view(C_out, 1, 1, 1)
                y = y + x_patch * f_val

    return y
```

要点：① 累加从 bias 初值开始（`y = bias.expand(...).clone()`），与 C++ `float sum = bp[co]` 对应；② 外层循环 ci→kh→kw、每轮一次全量张量加，保证每个输出元素经历与 C++ 标量循环**完全相同顺序**的 1152 次加法；③ 计算全程 fp32。

对照的 C++ golden（同顺序的标量三重循环，摘其核心）：

```cpp
for (int n = 0; n < N; ++n) {
    for (int co = 0; co < C_out; ++co) {
        for (int ho = 0; ho < Ho; ++ho) {
            for (int wo = 0; wo < Wo; ++wo) {
                float sum = bp[co];
                for (int ci = 0; ci < C_in; ++ci) {
                    for (int kh = 0; kh < K_h; ++kh) {
                        int hi = ho * sh - pt + kh * dh;
                        if (hi < 0 || hi >= H) continue;
                        for (int kw = 0; kw < K_w; ++kw) {
                            ...
                            sum += xv * fv;
                        }
                    }
                }
                yp[...] = sum;
```

## 设计决策

- 为何不改成"放宽 MARE 判据"：MERE 已与 C++ 一致，说明算子与数据都正常；MARE 差异是 golden 自身舍入路径造成的伪差异，修 golden 才是修因，动判据是掩盖。
- 为何用张量向量化循环而不是 Python 标量六重循环：向量化版每个输出元素只做与 C++ 同序的加法，但按整图并行执行，C_in=2048 的用例也可接受；标量版等价但慢几个量级。
- 为何不用 im2col+按序 matmul"优化"：任何改变归约分块的写法都会重新引入顺序差异。
- 性能代价：循环次数 C_in×K_h×K_w（最大约 4608 次），大 C_in 用例 golden 计算分钟级——测试工程可接受；若成瓶颈，备选是保存 C++ golden 二进制离线比对（未采用，破坏自包含性）。

## 验证信号

来源说明：以下信号来自该算子 PyTorch ST 的 golden 顺序对齐闭环。

- 修复前：L0 全量 60 条中 10 条 fp16 失败；L0-007 单跑 MARE=3.10e-2 超阈，Top5 失配位置 golden=4.14e-5~3.0e-3、abs_diff=3.2e-5~4.9e-5（near-zero + fp32 累加噪声特征）（来源平台：950；证据等级：tool_observed）。
- 修复后（ordered golden）：L0-007 单跑 PASS（1/1，fp16 通过率 100%）（来源平台：950；证据等级：tool_observed）。
- 修复后 L0 全量 58/60（剩余 L0-021/L0-056 为另一独立根因——同 shape 不同属性的批跑互踩，非 golden 问题）；该根因修复后 L0+L1 全量 187/187（来源平台：950；证据等级：tool_observed）。
- C++ 侧参照（同用例）：`[Real] L0-007 ... [PASS] MERE=1.76e-04, MARE=4.88e-04`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "与 C++ bit 级一致"的前提是 C++ golden 与 kernel 累加顺序一致；若被测 kernel 内部采用分块/树形归约，则应按 kernel 实际顺序写 golden（本卡方法论不变，顺序目标因算子而异）。
- ordered golden 循环 O(C_in×K_h×K_w)，超大 C_in 的 CPU golden 耗时显著上升。
- bf16 用例（L1 39 条）在本批同样通过，但 bf16 量化噪声远大于顺序差异，顺序对齐对 bf16 的边际收益小（未单独消融）。

**未决项**：

- torch 的 `F.conv2d` 在不同版本/后端的归约顺序是否稳定未验证；若 golden 保留高层 API 实现，升级 torch 可能改变 MARE。
## 坏实践

- golden 直接用框架高层 API（conv2d/matmul）并假设"反正都是 fp32"——归约顺序是隐式实现细节，near-zero + MARE 判据下必然暴露。
- 见 near-zero 超阈就调大 epsilon/阈值——掩盖的是 golden 的伪差异，同时会放过真实精度退化。

## 误诊

- RNG 未对齐残余论：MARE 3.10e-2 出现在随机流已对齐之后，一度怀疑 CppRNG 仍有微差；但 MERE 与 C++ 一致（1.77e-4≈1.76e-4）说明输入数据一致，排除（随机流对齐的前置过程见[PyTorch 移植 ST 随机流不一致卡](st_port_rng_stream_mismatch.md)）。
- 算子精度不足论：C++ ST 同用例 MARE=4.88e-4 通过，同一 kernel 不可能只在 PyTorch 测试里精度差——判定差异必在测试侧。
