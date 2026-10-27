---
type: "Runbook"
title: "Cube tile k/n 超 L1/L0 上限报 INVALID_TILE"
description: "PyPTO cube tile 的 k/n 分量超过 L1a(256)/L1(64) 上限且不整除时，编译 tile 校验阶段报 ErrCode F0000C INVALID_TILE，调大 tile 前须先确认上限。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.programming_model.execution_unit.cube"]
status: stable
sources: []
created_at: "2026-08-18T05:06:54Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["ErrCode_F0000C"]
---

# Cube tile k/n 超 L1/L0 上限报 INVALID_TILE

## 现象

在 950（DAV_3510，Ascend950）CANN 9.1.0 上调优 `mla_prolog_quant_d` decode 算子时，为减少 K 方向迭代次数把 `tile_config.pre_quant_cube_tile` 的 k 分量从 256 调大到 512（K=3584，k=256 需 14 次迭代，k=512 需 7 次），编译立即在 tile 校验阶段报错：

```
ERROR .../mla_prolog_quant_impl.py:887:4: CHECK FAILED: ErrCode: F0000C! Enum: ExternalError::INVALID_TILE
Invalid L1/L0 relation: kL0=512, kL1a=256, require kL0 <= kL1a && kL1a % kL0 == 0
```

报错点在 `mla_prolog_quant_compute(...)` 调用处。同一调优会话中另一次把某 cube tile 的 n 分量调到 128 时触发同类错误：

```
pypto.error.RenderedParserError: CHECK FAILED: ErrCode: F0000C! Enum: ExternalError::INVALID_TILE
Invalid L1/L0 relation: nL0=128, nL1=64, require nL0 <= nL1 && nL1 % nL0 == 0
```

## 适用场景

- 工程形态：kernel 直调（PyPTO `@pypto.frontend.jit` 算子，通过 `MlaTileConfig.pre_quant_cube_tile` / `cube_qb_tile` / `cube_wuk_tile` 等 `tile_config` 字段设置 cube matmul 的 m/k/n 分块）。
- API 路径：`tile_config.<cube_tile>` = `[m, m, k, n, base_n, base_k]`（6 元组，第 3 元 k = kL0，第 4 元 n = nL0）；校验在 `pypto.set_cube_tile_shapes` / 算子 compile 阶段。
- 验证过的平台与 CANN 版本：Ascend950（950/DAV_3510），CANN 9.1.0。
- 适用粒度：`platform_wide`（L1/L0 容量是 950 硬件级约束，kL1a=256、nL1=64 是 DAV_3510 cube 单元上限，本平台所有 cube matmul tile 都受此约束）。
- 路线优先级：`banned`（k/n 超过 L1a/L1 上限必然报错，调大 tile 前必须先确认上限）。
- 不适用/停止边界：非 950 平台（a3/a2）的 L1/L0 上限数值不同，不可直接外推 256/64；vector tile（`q_vec_tile1`/`k_vec_tile1`）不受此 cube L1/L0 约束。只有平台硬件上限或 `pass_options`（如 `cube_l1_reuse_setting`）改变并有新对照证据时才重新评估。

## 根因

950（DAV_3510）cube 单元的 L1a 缓冲上限为 kL1a=256，L1 缓冲的 n 方向上限为 nL1=64。`tile_config` 里 cube tile 的 k 分量（kL0）必须 `kL0 <= kL1a` 且 `kL1a % kL0 == 0`，n 分量（nL0）必须 `nL0 <= nL1` 且 `nL1 % nL0 == 0`。把 k 调到 512 > 256 或 n 调到 128 > 64 都违反上限，编译器在 tile 校验阶段（`pypto.set_cube_tile_shapes` / 算子 compile 入口）直接 CHECK FAILED 并抛 `ErrCode: F0000C ExternalError::INVALID_TILE`，不进入后续编译，也不会自动降到合法值。

## 失败路线

路线优先级：`banned`。cube tile 的 k/n 分量超过 L1a(256)/L1(64) 上限必然编译失败，不应作为减少迭代次数的调优手段。

- 调大 `pre_quant_cube_tile` k=512（目标：减少 K 迭代次数 14→7）→ 违反 `kL0 <= kL1a(256)`，编译 tile 校验阶段报 `ErrCode F0000C INVALID_TILE`，未进入后续编译。
- 调大某 cube tile n=128（目标：扩大 n 分块）→ 违反 `nL0 <= nL1(64)`，报同类 `INVALID_TILE`。

950 上 cube tile k 的有效上限就是 256（kL1a），n 的有效上限是 64（nL1）；超出即报 INVALID_TILE。若 K 较大想减少迭代次数，k 顶到 256 已是极限（本算子 K=3584 时 k=256 使 14 次迭代），再大直接报错；n 方向同理顶到 64。把 k/n 设为上限的约数（如 256/64/32）可同时满足整除约束。

## 验证信号

以下信号来自本次开发会话对 tile_config 调大的实际编译输出。

- `pre_quant_cube_tile` k=512 → `CHECK FAILED: ErrCode: F0000C! Enum: ExternalError::INVALID_TILE, Invalid L1/L0 relation: kL0=512, kL1a=256, require kL0 <= kL1a && kL1a % kL0 == 0`（来源平台：950；证据等级：tool_observed）
- 某 cube tile n=128 → `Invalid L1/L0 relation: nL0=128, nL1=64, require nL0 <= nL1 && nL1 % nL0 == 0`（来源平台：950；证据等级：tool_observed）

## 失效边界

- 已验证：950/DAV_3510 + CANN 9.1.0，kL1a=256、nL1=64。
- 未验证，推断可迁移：950 上其他 cube matmul 算子受同一上限约束（platform_wide），但各算子的 L1a/L1 实际上限可能因 `pass_options`（如 `cube_l1_reuse_setting`）而变化，未实测。
- 未验证：a3/a2 平台的 L1/L0 上限数值不同，不可直接外推 256/64。