---
type: "Runbook"
title: "多核非对齐 DataCopyPad 写回边界互相覆盖"
description: "相邻核以非 32B 对齐列块执行 DataCopyPad GM 写回时，尾块 RMW 可覆盖邻核边界数据；按 dtype 的 32B 粒度对齐非尾 colTile 可消除竞态。"
tags: ["task.debugging", "topic.precision", "topic.alignment", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 多核非对齐 DataCopyPad 写回边界互相覆盖

## 现象

在 Ascend950 上运行 `softmax.asc` 的非末轴 ARA 路径时，AR 路径修复后完整 NPU pytest 已缩小到两个低精度失败。正确的 Softmax 输出应为非负小值，但失败输出只在相邻列块的边界尾部突然出现数量级很大的正负随机值，其余大部分列仍正确。先把 UB 的 `ceilW` padding 显式清零后，完整测试仍是 2 failed、51 passed，说明坏值不是 padding 残留，而是多核列块写回 GM 时发生在块边界的数据破坏。

## 适用场景

- 工程形态：AscendC kernel 直调，非末轴 reduction 把外层列维切成多核相邻 column block。
- API 路径：host tiling 的 `colTile/numColTiles` 与 kernel `DataCopyPad` 的 GM 写回。
- 已验证平台与版本：Ascend950PR_957c（dav-3510），CANN 9.0.1；与 frontmatter 的 `platforms: [950]` 一致。
- 适用粒度：`same_operator`。证据只覆盖 Softmax ARA 的 float16/bfloat16 多核列切分。
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。最终组合实现平均速度约为评测参考实现的 0.29 倍，且没有隔离测出对齐 tiling 自身的性能影响。
- 不适用或未验证：单核写回、只有一个列块、每个核写完全不相邻 GM 区域、A2/A3 和其他 CANN 版本均未验证。

## 根因

多核 ARA 的相邻列块起点和非尾长度没有按 32B 对齐，使 `DataCopyPad` 的非对齐 GM 写对尾部 32B 块执行 read-modify-write 时，与邻核针对同一边界块的 RMW 互相覆盖刚写入的数据。 机制链是：对 dtype 大小 `sizeofT`，一个 32B block 对应 `minElems = 32 / sizeofT` 个元素；若非尾 `colTile % minElems != 0`，第 `j` 块起点 `j * colTile * sizeofT` 不是 32B 的整数倍，并且其尾写会触及相邻块所在的 32B 区域。两个核并发读取旧 block、各自合并局部字节、再写回整 block，其中后完成的 RMW 会把另一个核的新字节还原成旧值。只有最终尾块后方没有相邻 writer，因此可以保留非对齐尾部。

## 解决配方

先按目标并行度计算 `colTile`，再用 `minElems = 32 / sizeofT` 把所有非尾块向上对齐。这样第 `j` 个非尾块的起点 `j * colTile * sizeofT` 和长度 `colTile * sizeofT` 都是 32B 的整数倍；不足 32B 的最后尾块并回前一个块，保证非对齐写只可能发生在张量最终尾部。

### before：仅保证块不小于 32B，未保证每个非尾块对齐

- revision: trace-before-coltile-alignment
- path: `softmax.asc`
- symbol: `calc_softmax_tiling_params`
- line_range: `L645-L681`（修复前快照）

```cpp
std::tuple<int64_t, int64_t, int64_t> calc_softmax_tiling_params(
    int64_t a1, int64_t r, int64_t a0, int64_t sizeofT)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    (void)r;

    if (a0 == 1) {
        int64_t numBlocks = std::min(coreNum, std::max(int64_t(1), a1));
        return std::make_tuple(numBlocks, int64_t(0), int64_t(0));
    }

    int64_t minElems = 32 / sizeofT;  // 8 (fp32) or 16 (fp16/bf16): 32B minimum
    int64_t colTile = std::min(a0, COL_TILE_MAX);
    int64_t numColTiles;
    if (a1 >= coreNum) {
        numColTiles = (a0 + colTile - 1) / colTile;
    } else {
        int64_t desired = (coreNum + a1 - 1) / a1;  // ceil(coreNum / a1)
        int64_t ct = (a0 + desired - 1) / desired;  // ceil(a0 / desired)
        if (ct < minElems) ct = minElems;
        if (ct > colTile) ct = colTile;
        if (ct < 1) ct = 1;
        colTile = ct;
        numColTiles = (a0 + colTile - 1) / colTile;
        // Merge the tail block into the previous one if it would be sub-32B.
        int64_t tail = a0 - (numColTiles - 1) * colTile;
        while (numColTiles > 1 && tail > 0 && tail < minElems) {
            numColTiles--;
            tail = a0 - (numColTiles - 1) * colTile;
        }
    }
    int64_t workUnits = a1 * numColTiles;
    int64_t numBlocks = std::min(coreNum, std::max(int64_t(1), workUnits));
    return std::make_tuple(numBlocks, colTile, numColTiles);
}
```

### after：按 dtype 的 32B 元素粒度对齐非尾块

- revision: trace-after-coltile-alignment
- path: `softmax.asc`
- symbol: `calc_softmax_tiling_params`
- line_range: `L645-L697`

```cpp
std::tuple<int64_t, int64_t, int64_t> calc_softmax_tiling_params(
    int64_t a1, int64_t r, int64_t a0, int64_t sizeofT)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    (void)r;

    if (a0 == 1) {
        int64_t numBlocks = std::min(coreNum, std::max(int64_t(1), a1));
        return std::make_tuple(numBlocks, int64_t(0), int64_t(0));
    }

    int64_t minElems = 32 / sizeofT;  // 8 (fp32) or 16 (fp16/bf16): 32B minimum
    auto alignUp = [](int64_t v, int64_t m) -> int64_t { return ((v + m - 1) / m) * m; };

    // Target column-tile count: enough work units to occupy all cores.
    int64_t desiredTiles;
    if (a1 >= coreNum) {
        desiredTiles = (a0 + COL_TILE_MAX - 1) / COL_TILE_MAX;  // cap block size
    } else {
        desiredTiles = (coreNum + a1 - 1) / a1;  // ceil(coreNum / a1)
    }
    if (desiredTiles < 1) desiredTiles = 1;
    int64_t colTile = (a0 + desiredTiles - 1) / desiredTiles;  // ceil(a0 / desiredTiles)
    if (colTile > COL_TILE_MAX) colTile = COL_TILE_MAX;

    // Align colTile up to minElems so every NON-tail block has a 32B-aligned GM
    // start (c0 = j*colTile -> byte offset j*colTile*sizeofT, a multiple of 32B)
    // and a 32B-aligned length. A non-32B-aligned GM DataCopyPad write performs a
    // read-modify-write on the trailing 32B block; when two adjacent cores each
    // issue such a write to neighbouring column-blocks, one core's RMW reads the
    // other's just-written 32B region and writes stale data back, corrupting the
    // boundary columns. Aligning non-tail blocks removes the spillover; the last
    // block owns the tensor tail (no neighbour beyond it), so its non-aligned
    // trailing RMW cannot race another core.
    if (colTile > minElems) {
        int64_t a = alignUp(colTile, minElems);
        if (a <= a0 && a <= COL_TILE_MAX) colTile = a;
    }
    if (colTile < 1) colTile = 1;
    int64_t numColTiles = (a0 + colTile - 1) / colTile;
    // Merge a sub-32B tail into the previous block so only the final block can
    // be sub-32B (it still owns the tensor tail).
    int64_t tail = a0 - (numColTiles - 1) * colTile;
    while (numColTiles > 1 && tail > 0 && tail < minElems) {
        numColTiles--;
        tail = a0 - (numColTiles - 1) * colTile;
    }
    int64_t workUnits = a1 * numColTiles;
    int64_t numBlocks = std::min(coreNum, std::max(int64_t(1), workUnits));
    return std::make_tuple(numBlocks, colTile, numColTiles);
}
```

## 系统信息

- 硬件：Ascend950PR_957c，dav-3510。
- 软件：CANN 9.0.1，Bisheng/AscendC 编译链。
- 代码文件：`softmax.asc`。
- 关键符号：`calc_softmax_tiling_params`、`DataCopyPad`、`colTile`、`numColTiles`。

## 设计决策

选择按 `32 / sizeofT` 对齐非尾 `colTile`，而不是对所有块强制等长，是为了同时满足相邻 writer 不共享 32B 边界块和最终尾块可覆盖任意剩余长度。先尝试的“只清零 `ceilW` padding”保留了 GM 边界写回竞态，上板后仍然失败，因而被放弃。当前方案属于 `safety_fallback`：它给出正确性所需的保守 tiling，不宣称在并行度和尾块负载均衡上最优。

## 性能影响

向上对齐 `colTile` 可能减少列块数、改变核间负载，合并短尾还会让最后一个 work unit 更大。轨迹没有隔离该变化的成对性能数据；最终组合实现的平均速度约为评测参考实现的 0.29 倍，因此只能记录为正确性兜底，不能把对齐策略写成性能优化收益。

## 预防

- 多核把连续 GM 区间切成相邻块时，不只检查 `len >= 32B`，还要同时检查所有非尾块的 `start % 32B == 0` 与 `len % 32B == 0`。
- 将对齐条件写成 dtype 无关公式：`minElems = 32 / sizeofT`，`alignedColTile = ceil(colTile / minElems) * minElems`。
- 若最终尾块少于 32B，把它并入前块；不要让一个 sub-32B 中间块夹在两个并发 writer 之间。
- 精度测试应覆盖相邻多核块边界，并检查负概率或数量级突变，不能只看整体均值误差。

## 验证信号

来源说明：本卡由单条 AscendC 开发轨迹中的文件编辑、构建输出、NPU pytest 与 cann-bench 工具结果重建；内部事件锚点只保存在同目录 `manifest.json`，正文不暴露会话标识。

- 修复前，AR 已通过而非末轴低精度多核 case 仍输出大幅正负随机尾值，完整测试为 2 failed、51 passed。（证据等级: tool_observed；来源平台: Ascend950）
- 显式清零 UB padding 后仍为 2 failed、51 passed，排除了“仅 padding 残留”的候选解释。（证据等级: tool_observed；来源平台: Ascend950）
- 按 32B 元素粒度对齐非尾 `colTile` 并合并短尾后，工程构建成功，完整 NPU pytest 首次 53/53，同一轨迹随后连续两次 53/53，清理构建后仍为 53/53。（证据等级: tool_observed；来源平台: Ascend950）
- 最终新评测进程覆盖的 20 个用例全部精度通过；最终组合实现平均速度约为参考实现的 0.29 倍。（证据等级: tool_observed；来源平台: Ascend950）

## 失效边界

- 证据只覆盖 Ascend950 与 CANN 9.0.1；A2/A3 和其他版本未验证，推断可迁移但不得直接采用。
- 只有相邻核可能触及同一 GM 32B 边界块时才应用该根因；单核或物理地址不相邻的写回不能据此归因。
- 对齐值仍必须服从 UB 容量与 `COL_TILE_MAX`，不能无界向上取整。
- 底层 RMW 微序列没有被硬件 trace 直接捕获；根因由边界信号、失败对照和单变量对齐修复共同支持。
- 最终性能是两个修复和整条 kernel 的组合结果，不能归因到本 tiling 单点。

## 坏实践

- 只保证每块至少 32B，却允许非尾块起点或长度不是 32B 整数倍。
- 看到尾部随机值就只清 UB padding；这不会改变并发 GM writer 的地址重叠。
- 为了增加并行度把任意列数均分给核，却不检查 `j * colTile * sizeofT` 的块起点对齐。
- 通过放宽低精度容差掩盖数量级错误；本轨迹没有采用这种捷径。

## 误诊

“UB padding 未清零”是一个可检验但被上板结果否定的假设：清零后失败数和坏值模式不变。真正有区分度的改动是只改变 host tiling 的非尾 32B 对齐，随后所有原失败用例稳定通过。
