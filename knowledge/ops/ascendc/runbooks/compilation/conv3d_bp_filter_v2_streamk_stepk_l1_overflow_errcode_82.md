---
type: "Runbook"
title: "errcode 82 — streamk stepK 致 L1 越界"
description: "Conv3DBackpropFilterV2 在 950 streamk 切分后 stepKb 被 min 限制为不整除 wo 的值，hoCal 非线性跳变使 bl1Bound 反增、L1 溢出报 errcode 82；DoStreamkByHWout 末尾缺 L1 校验是判别要点。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["errcode_82", "L1_overflow"]
---

# errcode 82 — streamk stepK 致 L1 越界

## 现象

在 Ascend 950 (arch35 / DAV_3510) 上运行 `Conv3DBackpropFilterV2` 算子，输入 shape `[3, 2668, 1, 11, 168]` (NCDHW, fp32)，filter shape `[5]`，out_backprop `[3, 1, 1, 11, 56]`，输出 `[1, 2668, 1, 1, 3]`。tiling 日志显示走 streamk 切分 `streamkType=2 (STREAMK_HWOUT)`，关键 tiling 参数：`baseM=16, baseK=32, baseN=240, stepKa=6, stepKb=6, al1Bound=3072, bl1Bound=67200, al1Pbuffer=2, bl1Pbuffer=2, a_dtype_bytes=4, singleCoreK=168 (被 streamk 从 616 切分), coreStreamK=4, singleCoreHo=3, splitWo=56, singleCoreCin=80, usedCoreNum=28`。运行时报错：

```
errcode:(82) errorStr: The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds.
```

errcode 82 是 MTE2 (GM→L1/L0 搬运) 写 UB / MTE3 (L0C→GM/UB) 写 L1/UB 的地址越界。本案例是 L1 buffer 分配超出物理容量后，TPipe 给 buffer 分配的地址超出 L1 硬件地址范围，MTE2 搬运写入 L1 时硬件检测到地址非法。

## 判别方法

### 触发

`Conv3DBackpropFilterV2` streamk 切分（`streamkType=STREAMK_HWOUT`）后运行报 errcode 82（MTE2/MTE3 地址越界）。触发信号：tiling 日志 `streamkType=2` 且 `stepKb` 被 `min` 限制为不整除 `wo` 的值。

### 探针

1. 从 tiling 日志提取 `stepKb`、`bl1Bound`、`al1Bound`、`al1Pbuffer`、`bl1Pbuffer`、`a_dtype_bytes`，按 `al1Bound×al1Pbuffer×a_dtype_bytes×2 + bl1Bound×bl1Pbuffer×a_dtype_bytes×2` 估算 L1 总分配，对比平台 L1 容量（Ascend 950 = 524288 字节）。
2. 检查 `DoStreamkByHWout` 在 `min` 限制 `stepKa/stepKb` 之后是否调用 `IsCurBlockL1Invalid` 重新校验 L1——原始实现在 `else` 块内修改 `singleCoreK/blockBaseK/stepKa/stepKb` 后**无 `IsCurBlockL1Invalid` 调用**。
3. 检查 `kBl1Size = blockBaseK × stepKb` 是否整除 `wo`：不整除时 `GetHiCal` 走 `hoCal = floor(kBl1Size/wo) + NUM_HALF(2)` 非线性分支，`bl1Bound` 可能反增。

`DoStreamkByHWout` 原始实现（verbatim 取自 trace file_read 事件，`op_host/op_tiling/arch35/conv3d_backprop_filter_v2_stream_k_tiling.cpp:275-296`）——诊断时确认 `else` 块末尾是否缺 L1 校验：

```cpp
void Conv3DBackpropFilterV2StreamKTiling::DoStreamkByHWout()
{
    blockTiling_.singleCoreBatchDout = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout;

    // 当前切HWout只对H方向切块
    uint64_t singleShapeK = GetSingleShapeKByStreamK();
    uint64_t streamkHWoutDim = Ops::Base::CeilDiv(blockTiling_.singleCoreK, singleShapeK);
    if (streamkHWoutDim == ONE_DIM) {
        blockTiling_.streamkType = NO_STREAMK_CALC;
    } else {
        blockTiling_.streamkType = STREAMK_HWOUT;
        blockTiling_.singleCoreK = singleShapeK;
        blockTiling_.coreStreamK = streamkHWoutDim;
        uint64_t blockBaseK = std::min(static_cast<uint64_t>(blockTiling_.blockBaseK), blockTiling_.singleCoreK);
        blockBaseK = Ops::Base::CeilAlign(blockBaseK, static_cast<uint64_t>(BLOCK_CUBE));
        blockTiling_.blockBaseK = blockBaseK;
        blockTiling_.stepKa =
            std::min(blockTiling_.stepKa, static_cast<uint32_t>(Ops::Base::CeilDiv(singleShapeK, blockBaseK)));
        blockTiling_.stepKb =
            std::min(blockTiling_.stepKb, static_cast<uint32_t>(Ops::Base::CeilDiv(singleShapeK, blockBaseK)));
    }
}
```

关键缺陷：`else` 块内修改 `singleCoreK/blockBaseK/stepKa/stepKb` 后**无 `IsCurBlockL1Invalid` 调用**，stepK 被 `min` 截断为不整除 `wo` 的值时无人拦住 `bl1Bound` 反增。

### 观测

- 运行报 errcode 82：`The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds`。
- tiling 日志关键参数（tool_observed）：`stepKb=6, bl1Bound=67200, al1Bound=3072, al1Pbuffer=2, bl1Pbuffer=2, a_dtype_bytes=4`。

### 判别

机制（**由 tiling 日志参数算术推演，标记为 inferred-in-interpretation，非直接 tool_observed**）：`DoStreamkByHWout` 在 streamk 切分后用 `min` 限制 `stepKb`，但限制后未重新校验 L1 容量。限制后的 `stepKb` 使 `kBl1Size = blockBaseK × stepKb` 不再整除 `wo`，触发 `GetHiCal` 的非整除分支 `hoCal = floor(kBl1Size/wo) + NUM_HALF(2)`，`hoCal` 非线性跳变（比整除时的 `CeilDiv` 多算 1 行），`bl1Bound = hiCal × splitWi × singleCoreCin` 反增，L1 总分配溢出。

算式链（推演）：
1. `SetStepK4SplitMN` 阶段 `stepKb=7`：`kBl1Size=32×7=224`，`224%56==0` → 整除分支 `hoCal=CeilDiv(224,56)=4`，`bl1Bound=4×168×80=53760`，L1 总分配 `3072×4×2 + 53760×4×2 = 454656 < 524288` ✓。
2. `DoStreamkByHWout` 把 `singleCoreK` 切为 168（`coreStreamK=4`），`stepKb = min(7, CeilDiv(168,32)) = min(7,6) = 6`，**此后无 `IsCurBlockL1Invalid`**。
3. `stepKb=6`：`kBl1Size=32×6=192`，`192%56=24≠0` → 非整除分支 `hoCal=floor(192/56)+2=3+2=5`（`CeilDiv(192,56)=4`，多算 1 行），`bl1Bound=5×168×80=67200`（反增 13440），L1 总分配 `3072×4×2 + 67200×4×2 = 562176 > 524288` ✗，溢出 37888 字节。

`NUM_HALF=2` 语义：`+1` 向上取整（最后一个 ho 行不完整需补齐）`+1` load3d 跨行余量。整除时 ho 行边界与 K 数据边界对齐，无需余量，走精确 `CeilDiv`；`kBl1Size` 略低于整除点时 `floor+2` 比整除时的 `CeilDiv` 还大——这就是 stepKb 减小反而 L1 增大的非直觉根因。

## 适用场景

- **工程形态**：op_host tiling 计算（`op_host/op_tiling/arch35/`），非 kernel 直调、非 aclnn spec。
- **API 路径**：`Conv3DBackpropFilterV2StreamKTiling::DoStreamkByHWout` + `Conv3DDWV2BasicBlockTilingArch35::GetHiCal` / `CalBL1BoundSplitWo` / `IsCurBlockL1Invalid`。
- **验证过的平台与 CANN 版本**：Ascend 950 (arch35 / DAV_3510)，CANN ops-nn 私仓（用户工作目录），无固定上游 commit/blob。
- **适用粒度**：`same_operator`（根因机制 `kBl1Size % wo 整除性→hoCal 非线性跳变` 对 Conv3DBackpropFilterV2 的 `CalBL1BoundSplitWo` 成立；`DoStreamkByHWout stepK 限制后无 L1 校验` 的修复模式对带 streamk 切分的算子可参考，但具体函数名与 `streamkType` 枚举为本算子特有）。
- **路线优先级**：`recommended_fastpath`（在 DoStreamkByHWout 末尾加 L1 校验 + 递减 stepK + 降级还原，是本算子 streamk 路线的首选修复方向）。
- **不适用或未验证场景**：`a2`/`a3` 平台未验证；非 streamk 切分场景（`streamkType=NO_STREAMK_CALC`）不触发本根因；其他算子的 `CalBL1Bound` 若不经过 `hoCal` 的 `floor+NUM_HALF` 非线性分支则不适用。

## 验证信号

- 运行报 errcode 82：`The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds`（来源平台：950；证据等级：tool_observed）
- tiling 日志关键参数：`streamkType=2, stepKb=6, bl1Bound=67200, al1Bound=3072, al1Pbuffer=2, bl1Pbuffer=2, a_dtype_bytes=4, singleCoreK=168, coreStreamK=4, splitWo=56, singleCoreCin=80`（来源平台：950；证据等级：tool_observed）

## 坏实践

- **`DoStreamkByHWout` 修改 stepK 后不重新校验 L1**：原始代码假设 `SetStepK4SplitMN` 已校验过 L1，但 `DoStreamkByHWout` 改了 `singleCoreK/blockBaseK/stepK` 后 `bl1Bound` 会重算（非单调），旧校验失效。任何在 `SetStepK4SplitMN` 之后修改 stepK 的路径都必须重新 `IsCurBlockL1Invalid`。
- 用 `baseN × baseK × stepKb` 线性估算 L1B 占用：B 侧 L1B 是 `hiCal × splitWi × singleCoreCin` 的 hi×wi 展开空间，含 ho→hi 与 wo→wi 的 stride/kernel 重叠，比 L0B 级紧凑大小大；线性估算会漏判越界。

## 误诊

- agent 初版分析中误称"修改前 stepKb=7"——实际 tiling 日志是 `DoStreamkByHWout` 之后的最终值 `stepKb=6`，7 是 `SetStepK4SplitMN` 阶段的中间值。经用户纠正后重新梳理 `SetStepK4SplitMN(stepKb=7) → DoStreamkByHWout(stepKb=6)` 的变化链（agent_reported，非实测；已由代码 read 与 tiling 日志交叉验证纠正）。
- 中间方案曾尝试"ShrinkBaseBlock + UpdateStepMNK + 重限制 blockBaseK/stepK + 二次 L1 校验降级"，被用户否决（不愿改 blockBaseK）。
- 中间方案曾尝试"临时改 `mmInfo_.kValue=singleShapeK` 调 `SetStepK4SplitMN` 再恢复"，被用户否决（副作用大、不直观）。

## 预防

- 在 tiling 代码中，任何修改 `stepKa/stepKb/blockBaseK/singleCoreK` 的路径之后，若后续 kernel 侧 L1 buffer 按 `al1Bound/bl1Bound` 分配，必须重新 `IsCurBlockL1Invalid`。
- `CalBL1Bound`/`GetHiCal` 的非整除分支 `floor(kBl1Size/wo)+NUM_HALF` 是非单调的，stepKb 减小可能使 bl1Bound 反增；写 tiling 测试用例时覆盖 `kBl1Size % wo == 0` 与 `kBl1Size % wo ≠ 0` 两种边界。

## 失效边界

- `a2`/`a3` 平台：未验证，推断可迁移（L1 容量校验机制相同，但 L1 物理大小不同，需用对应平台 `l1_size`）。
- 非 streamk 切分场景（`streamkType=NO_STREAMK_CALC`）：不触发本根因，`DoStreamkByHWout` 不执行 stepK 限制。
- 其他算子：`DoStreamkByHWout stepK 限制后无 L1 校验` 的修复模式可参考，但 `GetHiCal` 的 `floor+NUM_HALF` 非整除分支与 `bl1Bound = hiCal×splitWi×cin` 的展开公式是 `Conv3DBackpropFilterV2` 特有；其他算子的 `CalBL1Bound` 若不经过 `hoCal` 非线性分支则 stepKb 减小不会使 bl1Bound 反增。
- `bl1Bound` 的 `floor+NUM_HALF` 保守估计在 `kBl1Size` 略低于整除点时过于保守；若未来 `GetHiCal` 改用精确 `CeilDiv`，本根因的"反增"现象会消失，但"stepK 限制后无 L1 校验"的缺陷仍在。
