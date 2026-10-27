---
type: "Runbook"
title: "多维 Gather 部分 case 错 — LUT 维序反了"
description: "C-contiguous 多维坐标必须从末维逆序解码；host LUT 是正确性回退而非快路，SIMT 可用同一逆序规则在线解码。"
tags: ["task.debugging", "topic.compute.indexing", "topic.implementation"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["reverse_decode", "simt_coordinate_decode"]
---

# 多维 Gather 部分 case 错 — LUT 维序反了

## 现象

在 Ascend950 Flash 工程中实现 `torch.gather` 语义时，单维和部分二维 NPU 冒烟通过，但涉及多个 leading/trailing 维的 `dim>0` case 报 `FAIL`。失败输出里前五个结果值与参考相同，完整 `allclose` 仍为 false，说明不是 kernel 全体空跑，而是某些多维坐标被映射到了错误的 GM 地址。

当时代码在 host 侧构造 `outerBase` / `innerBase` LUT，再由 SIMT kernel 使用 `outerBase[outerI] + idx * strideK + innerBase[innerI]` 取数；错误集中在 LUT 的扁平下标反解循环。

## 适用场景

- 工程形态：Flash 单文件算子，host 侧预计算地址 LUT，kernel 侧按索引随机访问 GM；
- API 路径：PyTorch 自定义算子 → host `build_offset_luts` → SIMT `__gm__` 指针索引；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于把 C-contiguous 子空间扁平序号反解为多维坐标后再乘真实 stride 的索引访问算子；
- 路线优先级：`safety_fallback`；这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。
- 明确不适用/未验证场景：非 C-contiguous 输入、列主序布局、device 侧使用不同坐标编码的实现，以及 A2/A3 均未验证。

## 根因

C-contiguous 扁平下标的最后一维才是最低有效维，但 LUT 代码从子空间首维开始执行 `% size` / `/= size`，把首维误当成最低有效维，导致多维 `outerBase` / `innerBase` 坐标反解顺序颠倒。 机制链为：对维度大小 `[s0, s1, ..., sm]`，C-contiguous 下标满足 `j = (...((c0 * s1 + c1) * s2 + c2)...) * sm + cm`；因此必须先取 `cm = j % sm`，再令 `j = j / sm` 继续向前，而不能先取 `c0 = j % s0`。反解后的坐标还要乘输入张量真实 `xStride[d]`，维序一旦颠倒，只有恰好对称或单维子空间会偶然正确。

## 解决配方

保留 `xStride` 的计算和 LUT 结构，只把 leading 子空间的解码改为 `dim-1 → 0`，把 trailing 子空间的解码改为 `n-1 → dim+1`。以下代码均为轨迹中 `gather.asc` 的函数级完整写入内容。

### before

代码证据：revision = 未提交的轨迹内 pre-fix 完整写入快照；path = `gather.asc`；symbol = `build_offset_luts`；lines = 138–194；支撑结论 = 正序取模把首维当成最低有效维。轨迹未提供可公开 blob URL。

```cpp
static void build_offset_luts(const at::Tensor &x, const at::Tensor &index,
                               int64_t dim,
                               at::Tensor &outerLutNpu, at::Tensor &innerLutNpu,
                               int64_t &outer, int64_t &inner, int64_t &strideK)
{
    int64_t n = static_cast<int64_t>(x.dim());
    // C-contiguous strides of x (in elements).
    std::vector<int64_t> xStride(n);
    {
        int64_t acc = 1;
        for (int64_t d = n - 1; d >= 0; --d) {
            xStride[d] = acc;
            acc *= x.size(d);
        }
    }
    strideK = xStride[dim];

    auto idxSizes = index.sizes();
    outer = 1;
    for (int64_t d = 0; d < dim; ++d) { outer *= idxSizes[d]; }
    inner = 1;
    for (int64_t d = dim + 1; d < n; ++d) { inner *= idxSizes[d]; }
    if (outer < 1) { outer = 1; }
    if (inner < 1) { inner = 1; }

    // ---- outerBase LUT (leading dims, decomposed by index sizes) ----
    torch::Tensor outerLutCpu = torch::empty({outer}, torch::kInt64);
    int64_t *op = outerLutCpu.data_ptr<int64_t>();
    for (int64_t j = 0; j < outer; ++j) {
        int64_t rem = j;
        int64_t off = 0;
        for (int64_t d = 0; d < dim; ++d) {
            int64_t sz = idxSizes[d];
            int64_t coord = rem % sz;      // (sz>0 guaranteed: d<dim, size>=1)
            off += coord * xStride[d];
            rem /= sz;
        }
        op[j] = off;
    }
    outerLutNpu = outerLutCpu.to(x.device());

    // ---- innerBase LUT (trailing dims, decomposed by index sizes) ----
    torch::Tensor innerLutCpu = torch::empty({inner}, torch::kInt64);
    int64_t *ip = innerLutCpu.data_ptr<int64_t>();
    for (int64_t j = 0; j < inner; ++j) {
        int64_t rem = j;
        int64_t off = 0;
        for (int64_t d = dim + 1; d < n; ++d) {
            int64_t sz = idxSizes[d];
            int64_t coord = rem % sz;
            off += coord * xStride[d];
            rem /= sz;
        }
        ip[j] = off;
    }
    innerLutNpu = innerLutCpu.to(x.device());
}
```

### after

代码证据：revision = 轨迹内 post-fix 完整重建快照；path = `gather.asc`；symbol = `build_offset_luts`；lines = 139–200；支撑结论 = 末维优先取模恢复 C-contiguous 坐标。该函数由完整写入快照加两次逐块 edit 原样重建；轨迹未提供可公开 blob URL。

```cpp
static void build_offset_luts(const at::Tensor &x, const at::Tensor &index,
                               int64_t dim,
                               at::Tensor &outerLutNpu, at::Tensor &innerLutNpu,
                               int64_t &outer, int64_t &inner, int64_t &strideK)
{
    int64_t n = static_cast<int64_t>(x.dim());
    // C-contiguous strides of x (in elements).
    std::vector<int64_t> xStride(n);
    {
        int64_t acc = 1;
        for (int64_t d = n - 1; d >= 0; --d) {
            xStride[d] = acc;
            acc *= x.size(d);
        }
    }
    strideK = xStride[dim];

    auto idxSizes = index.sizes();
    outer = 1;
    for (int64_t d = 0; d < dim; ++d) { outer *= idxSizes[d]; }
    inner = 1;
    for (int64_t d = dim + 1; d < n; ++d) { inner *= idxSizes[d]; }
    if (outer < 1) { outer = 1; }
    if (inner < 1) { inner = 1; }

    // ---- outerBase LUT (leading dims, decomposed by index sizes) ----
    // Decompose the flat outer index in C-contiguous order: dim 0 is the
    // MOST-significant (largest stride) of the leading dims, so iterate from
    // dim-1 down to 0 (least-sig first) to recover per-dim coords correctly.
    torch::Tensor outerLutCpu = torch::empty({outer}, torch::kInt64);
    int64_t *op = outerLutCpu.data_ptr<int64_t>();
    for (int64_t j = 0; j < outer; ++j) {
        int64_t rem = j;
        int64_t off = 0;
        for (int64_t d = dim - 1; d >= 0; --d) {
            int64_t sz = idxSizes[d];
            int64_t coord = rem % sz;      // (sz>0 guaranteed: d<dim, size>=1)
            off += coord * xStride[d];
            rem /= sz;
        }
        op[j] = off;
    }
    outerLutNpu = outerLutCpu.to(x.device());

    // ---- innerBase LUT (trailing dims, decomposed by index sizes) ----
    // C-contiguous: the first trailing dim (dim+1) is most-significant, so
    // iterate from n-1 down to dim+1 (least-sig first).
    torch::Tensor innerLutCpu = torch::empty({inner}, torch::kInt64);
    int64_t *ip = innerLutCpu.data_ptr<int64_t>();
    for (int64_t j = 0; j < inner; ++j) {
        int64_t rem = j;
        int64_t off = 0;
        for (int64_t d = n - 1; d > dim; --d) {
            int64_t sz = idxSizes[d];
            int64_t coord = rem % sz;
            off += coord * xStride[d];
            rem /= sz;
        }
        ip[j] = off;
    }
    innerLutNpu = innerLutCpu.to(x.device());
}
```

## 设计决策

- 选择 host 侧 LUT，是为了让 SIMT kernel 的每个线程只做三项 LUT/stride 合成与一次 GM 读取，同时支持非 gather 维上 `index.size(d) <= x.size(d)` 的真实 stride；
- 保留 LUT 方案，只修正扁平下标反解顺序，因为失败信号与代码 diff 已把根因定位到坐标解码；轨迹未尝试把多维除法全部搬回 device 侧；
- 当前方案是 `safety_fallback`：精度闭环完整，但最终相对任务基线性能较低，不应被误当成性能最优 gather 路线。

## 系统信息

- 平台：Ascend950PR_957c / dav-3510；
- 工具链：CANN 9.0.1；
- 工程形态：单文件 `gather.asc` + `libop_gather.so`；
- 相关实现文件：`gather.asc`；验证脚手架：`test_gather.py`。

## 性能影响

最终固定 LUT 的 SIMT 路线在 950 上相对同任务 baseline 约为 `0.01–0.06x`（参照路线：任务 baseline；证据等级: tool_observed）。该结果说明本配方是正确性 fallback；轨迹没有分离测量“逆序解码”本身的性能增量。

## 预防

- 对任意 C-contiguous 坐标反解，先写出 Horner 形式的线性化公式，再从最后一维向前做 `%` 与 `/`；
- 至少准备一个 leading 子空间含两维、一个 trailing 子空间含两维的验证 case，单维 case 无法暴露维序颠倒；
- 冒烟测试不要只打印前几个值，必须对完整输出做比较；
- LUT 用 `index` 的子空间尺寸恢复坐标，用输入 `x` 的真实 stride 合成 GM offset，二者不要混用。

## 验证信号

这条知识来自 Ascend950 Gather 开发中“多维冒烟失败 → 逆序重建两个 LUT → 重建与全量 NPU 评测”的修复步骤。

- 修复前：低维 case 报 `OK`，一个多维 `dim>0` case 报 `FAIL`，且前五个打印值相同（来源平台：950；证据等级：tool_observed）；
- 修复后：多维、四维、五维及多种 dtype 的定向 NPU 冒烟均报 `OK`（来源平台：950；证据等级：tool_observed）；
- 完整 pytest 汇总为 `217 passed`（来源平台：950；证据等级：tool_observed）；
- cann-bench 全量评测 `20/20` case 通过，浮点输出 `MERE=0`、`MARE=0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该修复假定输入按 C-contiguous stride 解释；非连续输入必须先明确 stride 契约或直接读取运行时 stride；
- 若只有一个 leading/trailing 维，正序和逆序循环等价，无法靠该 case 证明实现普遍正确；
- A2/A3 未验证，推断可迁移，但不得视为已支持；
- 该卡不覆盖越界 index 的语义，也不覆盖更高性能的硬件 gather / SIMD 路线。

## 坏实践

- 从第一个维度开始 `% size`，再把所得坐标乘该维 stride：这只适用于首维恰好最低有效的布局，不适用于 C-contiguous；
- 看到前几个值相同就判定整个输出正确：维序错误可能只在跨维边界后出现；
- 为了让失败 case 通过而缩小 shape、限制 index 分布或放宽比较条件：这些都是测试侧掩盖，不修复地址映射。

## 误诊

剩余的 int8 随机输入异常和后续 `torch.Size` 比较 TypeError 都来自测试脚手架，不是本卡的 kernel 根因；它们已从候选知识中剔除，不能拿来解释多维 GM 地址错误。