---
type: "Runbook"
title: "广播 CopyIn tile 跨段越界读判别"
description: "多核线性切分下广播输入 tile 起点不落广播段(innerContig)边界时，单次 1D DataCopyPad 从段内偏移越界读 gradOutput，自 32B 边界起全 NaN/错值；判别=核起点 offset%innerContig≠0 即定位跨段越界读。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-18T05:04:22Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 广播 CopyIn tile 跨段越界读判别

## 现象

在 Ascend950 / CANN 9.1.0 上实现 `softplus_v2_grad`（Softplus 反向梯度）Branch-1 kernel（tilingKey=1, RANK=4 模板，fp16/bf16/fp32）。计算链为 `Cast→Muls(βx)→Compares(mask=βx>τ)→Muls(-βx)→Exp→Adds(1+exp)→Div(g/(1+exp))→Select(mask: g | sigmoid)`。Branch-1 ST 跑 `./run.sh`：绝大多数子运行通过，仅 gradOutput 广播（含 1 维，fp32）与 rank 不等左补 1 两类用例 FAIL。

run.sh 的 DBG-PROBE 输出显示广播用例的分歧起点：

```
BUILD=0
[FAIL] B1-3/fp32: matches_oracle (n=262144, beta=1, thr=20)
    [DBG-PROBE] i=4608 actual=0.008 golden=-0.481418
    [DBG-PROBE] i=4681 actual=0 golden=0.184
    [DBG-PROBE] i=4864 actual=-1.884 golden=-0.126521
```

自 `i=4681` 起实际值全错（0 / NaN / 负大数），golden 为正常 sigmoid 值。`i=4681` 是 32B 边界（8 个 fp32），且 `4681 % 256 = 73`（256 = 该用例广播输入 gradOutput 的 innerContig 段长）。非广播用例与标量广播用例均 PASS，只有 gradOutput 广播到多行 FAIL。

## 适用场景

- **工程形态**：aclnn kernel 直调（op_kernel），多核线性切分 + UB tile 循环 + TQue double buffer 三级流水（CopyIn→Compute→CopyOut）。
- **API 路径**：`AscendC::DataCopyPad(ub, gm[offset], DataCopyExtParams, DataCopyPadExtParams)`；广播输入搬运（srcOff 由 N 维坐标 `CoordToInputOffset` 计算）。
- **验证过的平台与 CANN 版本**：Ascend950，CANN 9.1.0。
- **适用粒度**：`operator_family`（任何“多核线性切分 + 广播输入搬运”的算子，只要广播输入 innerContig < 多核 tileLen 就会踩同一坑）。
- **路线优先级**：`diagnostic_only`（本卡只给判别方法与根因解释；修复实现的终版代码未从 trace 逐字恢复，不在此内联完整 after 代码）。
- **不适用或未验证场景**：a2 / a3 未实测（推断可迁移，见 ## 失效边界）；非广播输入（broadcastFlag 全 0）走 1D 快速路径，本坑不触发；广播输入 innerContig ≥ ubTileLength 时天然不跨段，本坑不触发。

## 判别方法

**触发条件**（tool_observed 失败症状）：kernel 编译通过、非广播与标量广播用例全 PASS，但「广播输入沿最内非 1 维广播到多行」的用例自某个 32B 边界起实际值全错（0 / NaN / 负大数），golden 为正常值；DBG-PROBE 显示分歧首点 `i` 恰为 32B 对齐点（fp32 下 8 的倍数）。

**可执行探针**：
1. 在 host/kernel 侧加 DBG-PROBE，逐元素打印 `actual` 与 `golden`，定位分歧首点 `i`。检查 `i` 是否为 32B 边界（fp32：`i % 8 == 0`）——若是，强烈提示 DMA 搬运越界。
2. 对广播输入计算 `innerContig`（广播输入沿最内非 1 维的连续元素数），再算分歧首点所在核的首 tile 起点 `offset`，取 `offset % innerContig`。若余数 ≠ 0，即 tile 起点不落广播段边界。
3. 反查 `CopyInOne` 广播分支：若用单次 1D `DataCopyPad(ub, gm[srcOff], blockLen=tileLen)` 且 `tileLen > innerContig - (offset % innerContig)`，则 `gm[srcOff .. srcOff+tileLen)` 必然读出广播输入 GM 缓冲边界（广播输入在 GM 里只有 innerContig 个元素）之外的内存——越界读成立。
4. 对照实验：把同一用例的 broadcastFlag 置全 0（关广播，输入按全 shape 分配）重跑，若分歧消失，确认根因在广播搬运而非计算链。

**原始观测**（Ascend950，fp32，softplus_v2_grad Branch-1）：
- DBG-PROBE 输出 `i=4681 actual=0 golden=0.184`、`i=4608 actual=0.008 golden=-0.481418`、`i=4864 actual=-1.884 golden=-0.126521`，分歧首点 `i=4681` 为 32B 边界。
- `4681 % 256 = 73`：核首 tile 起点不在广播段（innerContig=256）边界上。
- 非广播用例与标量广播（innerContig=1，srcOff 恒=0）用例均 PASS，仅 gradOutput 广播到多行 FAIL。

**机制解释**（根因分析）：多核线性切分把每个核的起始 offset 按 `blockLength/formerNum/tailBlockLength` 切，与广播输入的段边界 `innerContig` 无关。当 `offset % innerContig ≠ 0` 时，单次 1D `DataCopyPad(ub, gm[srcOff], blockLen=tileLen)` 从 `gm[srcOff]` 读 `tileLen` 个元素，但广播输入在 GM 里只有 `innerContig` 个元素，`gm[srcOff .. srcOff+tileLen)` 越界 → 读到相邻 GM 垃圾/未分配区 → NaN 或错值。分歧恰在 32B 边界显现，因 DataCopyPad 最小搬运粒度按 32B 切齐。机制链（不跳步）：`offset → 段内偏移 p=offset%innerContig ≠ 0 → srcOff=CoordToInputOffset=p → 单次 1D DataCopyPad(ub, gm[p], tileLen) → 读 gm[p..p+tileLen] 越界（输入仅 [0..innerContig-1]） → GM 越界读 → NaN/垃圾 → 自 32B 边界起全错`。

## 验证信号

- DBG-PROBE 输出 `i=4681 actual=0 golden=0.184`、`i=4608 actual=0.008 golden=-0.481418`、`i=4864 actual=-1.884 golden=-0.126521`，分歧首点为 32B 边界（来源平台：Ascend950；证据等级：tool_observed）。
- 非广播用例与标量广播用例 PASS、仅 gradOutput 广播到多行用例 FAIL 的对照（来源平台：Ascend950；证据等级：tool_observed）。
- LOG.md artifact 概述最终方案为“广播 CopyIn（N 维坐标 srcOff + tile 限不跨段）+ V_MTE2 同步”，ST 全 PASSED（来源平台：Ascend950；证据等级：artifact_verified）。

## 失效边界

- a2 / a3 平台：未实测，未验证，推断可迁移（DataCopyPad 的 blockLen/srcStride 语义、32B 最小搬运粒度、GM 越界读行为属 AscendC UB API 通用机制）。迁移前须重跑 ST 全矩阵确认 innerContig/对齐参数。
- 当广播输入 innerContig 极小（如 1，标量广播）时，`min(ubTileLength, 1)` = 1，限段后退化为逐元素 tile，性能极差但正确；本卡不优化此场景（innerContig=1 时 srcOff 恒=0 不越界，原本 PASS）。
- 若广播输入 innerContig 不是 32B 对齐的整数倍，限段对齐可能让 tileLen < innerContig，需保证 CopyOut 侧 offset 仍按原 tileOff 写回（不要用 srcOff 写输出）。

## 坏实践

- 广播 CopyIn 用 `DataCopyExtParams.blockCount>1 + srcStride=0` 想“批量重复搬运”——`srcStride=0` 对 `blockCount>1` 是“块间无步进=覆盖”，不是“源连续前进”，会越界读 GM。须逐段或限段后单次 1D。
- 广播 CopyIn 用段循环多次 `DataCopyPad` 不加 `PipeBarrier<PIPE_MTE2>`——多次 MTE2 之间可能竞争/乱序，导致段间数据错乱。
- 对广播输入直接用 `gm[tileOff]`（不计算 srcOff）——tileOff 是输出空间偏移，广播输入的源偏移须由 `CoordToInputOffset`（广播维→0）重映射。

## 误诊

调试期出现过多个被证伪的根因假设，记录如下以防后续 agent 重蹈：
- **误诊 1**：`Adds(tempX, tempX, 1.0f)` in-place 在 arch 3510 对某些值产生 NaN → 后续证伪：非 in-place `Adds(tempR, tempX, 1.0f)` 同样出问题，真因是广播搬运越界 + double buffer MTE2/V 数据竞争，与 Adds in-place 无关。
- **误诊 2**：`Compares` 破坏 `tempX`（maskBuf 与 tempXBuf 物理重叠）→ 后续增大 maskBuf 无效，真因是 gIn 在 V 读时被下一 tile 的 MTE2 覆盖（数据竞争），非 Compares 破坏。
- **误诊 3**：`Duplicate(mask, 0)` 测试时误以为 mask 已清零，实际 `Duplicate(mask,0)` 自身破坏 tempX（buffer 重叠），导致“mask 清零后 Div 正确”的假象。
- **真发现**：double buffer 下 V 异步未完成时下一 tile 的 MTE2 覆盖 gIn slot（数据竞争），需 V_MTE2 同步或限段单次 1D + 三缓冲规避——本卡判别方法的根因即由此澄清得出。