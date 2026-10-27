---
type: "Runbook"
title: "每核最少行数地板压低 AIV 并行度—按可用核数切分"
description: "行级归约 tiling 不应先用每核最少行数限制 block 数；按可用 AIV 核数上限切分可恢复并行度。"
tags: ["task.optimization", "topic.tiling", "topic.compute.reduction", "topic.performance.optimization.optimization_overview"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 每核最少行数地板压低 AIV 并行度—按可用核数切分

## 现象

在一个每行独立完成末轴归约的 Ascend C kernel 中，正确性已经全部通过，tile 内队列、VF 和 DMA 也都按 tile 合并，但 cann-bench 的各类 workload 仍普遍慢。目标 Ascend950 设备报告 56 个 vector core；检查 host tiling 后发现，`numBlocks` 不是从可用 AIV 核数出发，而是先用自设的“每核最少行数”地板计算，导致中等行数 workload 只发射少量 block。移除该地板后，在其余实现保持不变的受控实验里，平均性能相对原路线约提高 6.2 倍，正确性保持不变。

## 适用场景

- 工程形态：Ascend C 行级并行 kernel，经 host 侧 tiling 计算 block 数并在真实 NPU 上评测；
- API 路径：`platform_ascendc::PlatformAscendCManager::GetCoreNumAiv()`、host launch grid、`GetBlockIdx()` 行段切分；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于各行独立、无跨核归约或跨核同步的归一化/行归约算子；
- 路线优先级：`recommended_fastpath`，先让 `numBlocks` 覆盖可用 AIV 核，再用 `min(coreNumAiv,numRow)` 避免空 block；
- 明确不适用：需要跨核同步、单行可拆为多核、每核固定开销大于行计算量或行间工作量高度不均的算子；这些场景未验证。

## 根因

Host tiling 额外引入每核最少行数地板，使 `numBlocks` 按人为行数阈值而非可用 AIV 数量上限计算，导致大量 vector core 没有收到工作。

## 解决配方

代码证据来自两个固定 Git 状态；为遵守跨算子纪律，after 中仅省略了含源任务绝对性能数字和特定 shape 的注释，可执行代码行逐字取自固定 revision：

- before revision：`9355dd1ccc266a47c6812c8a8c6abdb06bf57b92`；path：`operators/rms_norm/rms_norm.asc`；symbol：`calc_rms_norm_tiling_params`；lines：304–332；取证：`git show 9355dd1ccc266a47c6812c8a8c6abdb06bf57b92:operators/rms_norm/rms_norm.asc`。
- after revision：`3d6c851510934d027bbbaf10bc08c5edcd2b9759`；path：`operators/rms_norm/rms_norm.asc`；symbol：`calc_rms_norm_tiling_params`；lines：317–351；取证：`git show 3d6c851510934d027bbbaf10bc08c5edcd2b9759:operators/rms_norm/rms_norm.asc`。

### before

```cpp
std::tuple<int64_t, int64_t, uint32_t, uint32_t>
calc_rms_norm_tiling_params(int64_t numRow, int64_t D, int64_t sizeofT)
{
    constexpr int64_t MIN_ROWS_PER_CORE = 1024;
    constexpr uint32_t MAX_TILE_ROWS = 256;
    constexpr int64_t UB_SIZE = 262144;   // 950 UB = 256KB
    constexpr int64_t HEADROOM = 8192;     // ACC-002 headroom

    // Dalign = align_up(D, VL=64). Inline arithmetic (no host helper in device).
    int64_t dalign = ((D + 63) / 64) * 64;
    if (dalign < 64) dalign = 64;

    int64_t availUB = UB_SIZE - HEADROOM;
    int64_t gammaBytes = dalign * sizeofT;
    int64_t perRowBytes = 4 * dalign * sizeofT;   // xQueue depth2 + yQueue depth2
    int64_t mT = (availUB - gammaBytes) / perRowBytes;
    if (mT < 1) mT = 1;
    if (mT > MAX_TILE_ROWS) mT = MAX_TILE_ROWS;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = (numRow + MIN_ROWS_PER_CORE - 1) / MIN_ROWS_PER_CORE;
    if (numBlocks > coreNum) numBlocks = coreNum;
    if (numBlocks < 1) numBlocks = 1;
    int64_t blockLength = (numRow + numBlocks - 1) / numBlocks;

    return std::make_tuple(numBlocks, blockLength, static_cast<uint32_t>(mT),
                           static_cast<uint32_t>(dalign));
}
```

### after

```cpp
std::tuple<int64_t, int64_t, uint32_t, uint32_t>
calc_rms_norm_tiling_params(int64_t numRow, int64_t D, int64_t sizeofT)
{
    constexpr uint32_t MAX_TILE_ROWS = 256;
    constexpr int64_t UB_SIZE = 262144;   // 950 UB = 256KB
    constexpr int64_t HEADROOM = 8192;     // ACC-002 headroom

    // Dalign = align_up(D, VL=64). Inline arithmetic (no host helper in device).
    int64_t dalign = ((D + 63) / 64) * 64;
    if (dalign < 64) dalign = 64;

    int64_t availUB = UB_SIZE - HEADROOM;
    int64_t gammaBytes = dalign * sizeofT;
    int64_t perRowBytes = 4 * dalign * sizeofT;   // xQueue depth2 + yQueue depth2
    int64_t mT = (availUB - gammaBytes) / perRowBytes;
    if (mT < 1) mT = 1;
    if (mT > MAX_TILE_ROWS) mT = MAX_TILE_ROWS;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = coreNum;
    if (numBlocks > numRow) numBlocks = numRow;
    if (numBlocks < 1) numBlocks = 1;
    int64_t blockLength = (numRow + numBlocks - 1) / numBlocks;

    return std::make_tuple(numBlocks, blockLength, static_cast<uint32_t>(mT),
                           static_cast<uint32_t>(dalign));
}
```

## 设计决策

- 选择 `numBlocks=min(coreNumAiv,numRow)`：前者保证尽量覆盖可用 AIV，后者保证每个激活 block 至少有一行，避免空核。
- 放弃“每核最少行数”地板：它没有来自硬件能力或实测的依据，并会在行数没有大到阈值倍数时静默压低发射 block 数。
- 先尝试 batched DMA 的路线只有约 1.18 倍相对收益，证明搬运提交并非主因；受控修改跨核切分后才得到约 6.2 倍相对提升。
- 当前方案是 `recommended_fastpath`；若算子存在跨核通信或每核固定开销高，应重新做核数扫描，不能机械照搬。

## 系统信息

- toolchain：CANN 9.0.1；
- 目标硬件：Ascend950PR_957c / dav-3510，设备报告 56 个 vector core；
- 相关文件：`operators/rms_norm/rms_norm.asc`；
- 验证入口：远端构建、完整 pytest、cann-bench 正确性与性能评测。

## 性能影响

- full-core 路线相对“每核最少行数地板”路线的平均性能约为 **6.2x**（来源平台：950，证据等级：tool_observed）。
- batched DMA 单独相对逐行 DMA 只有约 **1.18x**，因此本次性能恢复的主因是跨核并行度而不是 DMA 合并（来源平台：950，证据等级：tool_observed）。
- 两组对比均为同一算子内部路线的成对相对值；不把该比例外推为其他算子的通用收益。

## 预防

- Host tiling 同时打印或检查 `coreNumAiv`、`numBlocks`、每核工作量，不能只看 tileRows/UB 是否装得下。
- 对“各行独立”的算子，先以 `min(coreNumAiv,numRow)` 为并行度基线，再用真实 Profiling 证明降低核数更好；不要先凭经验加入每核工作量地板。
- 当端到端性能普遍低且正确性稳定时，把“发射了多少 AIV block”列为与 DMA、VF、同步同级的首轮诊断项。
- 每次改核数后重新跑完整正确性，特别检查尾核 `rowWork` 和 `blockLength=ceilDiv(numRow,numBlocks)`。

## 验证信号

本知识来自一次 Ascend950 行归约 Flash 算子性能闭环：先固定 batched DMA 代码状态，再只修改 `numBlocks` 计算并重新完成构建、pytest 和 cann-bench。

验证实现：revision `9355dd1ccc266a47c6812c8a8c6abdb06bf57b92`，path `operators/rms_norm/test_rms_norm.py`，symbols `test_rms_norm_case`（106–130）与 `test_rms_norm_generic`（145–154）；它以 CPU RMS 参考和 NPU 自定义算子输出作逐项比较，支撑“跨核切分改动未破坏正确性”的结论。

- 设备属性工具报告 56 个 vector core（来源平台：950，证据等级：tool_observed）。
- full-core 修复后远端构建成功，完整 pytest 45/45 通过（来源平台：950，证据等级：tool_observed）。
- full-core 修复后 cann-bench 正确性 20/20 通过（来源平台：950，证据等级：tool_observed）。
- full-core 路线相对每核行数地板路线平均性能约 6.2x（来源平台：950，证据等级：tool_observed）。

- 独立复现 R1：修改前后正确性均为 20/20；仅调整启核公式后，相对原每核最少行数路线提升 about 6.2x（950，tool_observed）。
- 独立复现 R2：修改前后正确性均为 20/20；仅调整启核公式后，相对原每核最少行数路线提升 about 5x（950，tool_observed）。

## 失效边界

- 只在 Ascend950 / CANN 9.0.1、每行独立且无跨核同步的归约型 kernel 上验证。
- 若 `numRow < coreNumAiv`，实现会自然限制为 `numRow` 个 block；不应发射没有工作行的空 block。
- 对跨核归约、原子更新、全局同步或单行计算量极小的算子，全核路线可能被同步/启动开销反噬，需要重新扫描核数。
- A2、A3 未验证，不能把 950 的核数策略直接视为跨平台事实。

## 坏实践

- 在没有设备核数和受控性能数据时加入 `MIN_ROWS_PER_CORE` 一类经验阈值；它可能让代码正确但绝大多数 AIV 不工作。
- 只优化 tileRows、UB 或 DMA，而不核对 host launch grid；这些局部优化无法弥补跨核并行度缺失。
- 把单个特殊 workload 的低并行度当成整体慢的解释；本轨迹的多数 workload 同时变慢，指向的是系统性发射不足。

## 误诊

最初把端到端低性能主要归因于逐行 DMA；受控 batched DMA 实验只有约 1.18 倍相对改善，随后核数探测与 full-core 实验才确认主因。