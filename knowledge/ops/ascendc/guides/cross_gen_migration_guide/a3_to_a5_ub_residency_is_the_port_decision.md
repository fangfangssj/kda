---
type: Guide
title: A3→A5 移植先看驻留量，不是先看算法
description: A3 内核常把整块数据设为 UB 常驻；A5 UB 仅约 248 KB，逐行移植会在大 case 上直接不可表示。判据与改法。
tags:
- task.migration
- topic.implementation.memory_management
- topic.performance.analysis
status: draft
sources: []
created_at: '2026-09-25T08:28:47Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases: ["UB 常驻"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# A3→A5 移植先看驻留量，不是先看算法

## 症状

A3(arch22) 内核的算法可以逐行搬到 A5(arch35)，但**数据搬运方式搬不过来**。典型形态：
A3 kernel 把整幅输入设为 UB 常驻（`CopyGmToUbExact(imgF, xIn[n*hwc], hwc)`），host 侧
只用一句 `assert ub <= 192*1024` 兜底。

## 为什么会炸

A5 可用 UB 约 **248 KB**（`ub_size=253952`）。在 row34 `crop_and_resize` 的 owner 固定
50 例上，`h*w*c` 最大到 **12,582,912**（case 46：1×2048×2048×3 ≈ **50 MB**），
**50 例中有 17 例超过 A3 那条 assert**。也就是说**逐行移植根本表示不了这个 benchmark**——
这不是性能问题，是可行性问题。

## 判据（移植前就能算）

1. 找到 A3 kernel 的**驻留集**表达式（常见是 `h*w*c` 或 `m*n`）。
2. 代入目标 benchmark 的**最大** case，和 A5 的 `ub_size` 比。
3. 超了就**不要**先写 kernel：先决定流式切分轴。

## 改法

把驻留从 `O(h*w*c)` 降到 `O(w*c)`：只驻留**一个输出行可能触及的至多两行输入**
（bilinear 需要 top+bottom，nearest 需要一行），用 2 槽行缓存，键为绝对行号，跨 batch 时失效。
row34 实测峰值 UB 约 **64.5 KB**（约为上限的 26%）。

## 附带效果：性能来源也在这里

参考实现（PyTorch）会物化 `xb = x[box_index]`（case 46 约 **3.2 GB**），而流式 kernel
只读真正需要的行。该 case 实测 **113×**；整体 50 例逐例几何均值 **45.5×**（参考腿为绑定的
PyTorch/torch_npu `model.py`，**非 CANN 融合实现**，读数时须带此限定）。

## 边界

- 数字来自单一算子（`crop_and_resize`）与单一固定 50 例分母，**不是跨算子普适常数**；
  可迁移的是**"先算驻留集再决定移植方式"这条顺序**，不是 64.5 KB 这个值。
- 未在其它 A3 算子上复现过该阈值。
