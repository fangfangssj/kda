---
type: "Runbook"
title: "RoPE DIST_INTLV_B16 store mask 致零值"
description: "MLA 融合算子 rms_norm_rope_k_store_kv_cache（dav-3510/Ascend950/CANN 9.1.0）的 DoRope64 用 DIST_INTLV_B16 interleave store 把 even/odd 配对 RoPE 结果交错写回 bf16，bs=64 的 k_rope_cache 出现 index 27+ 起的稳定零值模式（golden 非零、npu=0.000000 FAIL，每 4 元素出现 2 个零）；DIST_INTLV_B32 在 fp32 域直接产生垃圾值。判别=用 debug_rope.py 逐元素对比 NPU RoPE 输出与 golden，定位零值起始 index 与每 4 元素 2 零的重复模式，结合 store 路径（DIST_INTLV_B16 interleave）确认是 store mask 与 64 维 interleaved 布局不兼容，非 V 流水线 race（加 PipeBarrier<PIPE_V> 仍失败）。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.implementation.api_layer.basic_api", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T07:59:41Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# RoPE DIST_INTLV_B16 store mask 致零值

## 适用场景

- **工程形态**：kernel 直调（AscendC vector API + UB `LocalTensor` 标量访问），MLA Prefill 融合算子 `rms_norm_rope_k_store_kv_cache` 的 64 维 RoPE 路径。
- **API 路径**：`DataCopy<DIST_INTLV_B16>` / `DataCopy<DIST_INTLV_B32>` interleave store；`Cast` / `GetValue` / `SetValue` / `PipeBarrier<PIPE_V>` / `LocalTensor::ReinterpretCast`。
- **验证平台与版本**：Ascend950（dav-3510，Ascend950PR V100）/ CANN 9.1.0，`qk_rope_dim=64`，cos/sin 为 interleaved 布局（`cos[2i]=cos[2i+1]=C_i`）。
- **适用粒度**：`operator_family`——所有对小维（≤64）interleaved RoPE 用 `DIST_INTLV_B16` 做 store 而踩零值坑的算子。
- **路线优先级**：`safety_fallback`——标量配对计算（`Cast`→fp32 workspace + `GetValue`/`SetValue` 配对 + `Cast` 回 bf16）是 correctness baseline，相对向量化 interleave store 路线更慢但稳定；可行向量化路线。
- **不适用/未验证场景**：大维 RoPE（>64）标量循环过慢，应走向量化或 half-split 布局（参考算子 `fused_k_rms_norm_rope_store_kv_cache_mx_quant` 的 `DoRope` 即用 half-split 前半/后半分开 store，不踩 interleave store 坑）；a2/a3 未验证（vector API `Cast`/`GetValue`/`SetValue` 跨架构一致，推断可迁移但未上板）。

## 判别方法

**触发**（tool_observed 失败）：在 `rms_norm_rope_k_store_kv_cache`（MLA Prefill 融合算子，dav-3510 / Ascend950 / CANN 9.1.0）RMSNorm 修复后，bs=4/32 通过、仅 bs=64 的 `k_rope_cache` 断言失败。`pytest -k "65536-128-64"` 输出 `k_rope_cache` FAIL，NPU RoPE 输出呈"每隔一个元素为零"的零值模式（golden 全非零，npu 多处 0.0000）。

**探针**（可执行检查）：

1. 跑 bs=64 单测隔离失败：`python3 -m pytest ops/test/numerical_test/ascend/test_rms_norm_rope_k_store_kv_cache.py -x -v -k "65536-128-64" 2>&1 | tail -30`，确认 `cprss_kv_cache` 通过、仅 `k_rope_cache` 断言失败（排除 RMSNorm/Phase2 回读路径）。
2. 跑 `debug_rope.py` 逐元素对比 NPU RoPE 输出与 golden：`python3 debug_rope.py 2>&1 | tail -30`，逐行打印 `[i] golden=... npu=... FAIL/PASS`，定位零值起始 index 与重复周期。
3. 在 `DoRope64` 前后加 `PipeBarrier<PIPE_V>()` 重跑，观察零值是否消失——若仍失败，排除 V 流水线 race，定位到 store 路径本身。
4. 将 store 从 `DIST_INTLV_B16` 换成 fp32 域 `DIST_INTLV_B32` 重跑，观察是否仍异常——若 B32 产生垃圾值，进一步坐实 interleave store 路径与该布局不兼容。

**观测**（tool_observed）：

- bs=64 `k_rope_cache` 断言失败；NPU RoPE 输出呈 `[非零, 非零, 0.0000, ..., 非零, 0.0000, 0.0000]` 的零值模式，golden 全非零。
- `debug_rope.py` 逐元素输出：`[27] golden=0.373047 npu=0.000000 FAIL`、`[29] golden=0.005280 npu=0.039062 FAIL`、`[30] golden=0.761719 npu=0.000000 FAIL`、`[31] golden=0.964844 npu=0.000000 FAIL`；零值 index 集合 = {27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63}（自 index 27 起每 4 元素出现 2 个零）。
- `DoRope64` 前后加 `PipeBarrier<PIPE_V>()` 后零值仍持续（排除 V 流水线 race）。
- fp32 域 `DIST_INTLV_B32` interleave store 产生垃圾值（非零但错）。

**判读**（根因解读，agent 对 debug 输出与多次替换尝试的归因）：`DoRope64` 用 `DataCopy<DIST_INTLV_B16>` 把 `[-2.0... 偶, 2.0... 奇]` 配对结果以 interleave 模式交错写回 bf16 tensor，该 interleave store 的 mask 在 64 维 / 当前 UB 布局下行为异常，从 index 27 起按"每 4 元素 2 零"的周期写零——不是偶/奇全零，而是 mask 边界与 64 元素 interleaved 布局错配。`DIST_INTLV_B32`（fp32 域 interleave）则直接产生垃圾值，进一步坐实 interleave store 路径与小维 interleaved 布局不兼容。加 `PipeBarrier<PIPE_V>` 不解，故非 V 流水线 race。参考算子 `fused_k_rms_norm_rope_store_kv_cache_mx_quant` 的 `DoRope` 用 half-split 布局（前半/后半分开 store），不踩此坑；本算子 cos/sin 是 interleaved，布局与 RegBase `LoadAlign`/`LoadTensorForDtypeT` 直读不匹配，放大了 interleave store 的 mask 问题。修复方向=放弃 `DIST_INTLV_*` interleave store，改用 vector API `Cast` 到 fp32 workspace + 标量 `GetValue`/`SetValue` 配对计算 RoPE 再 `Cast` 回 bf16（correctness baseline）。

## 验证信号

来源：`rms_norm_rope_k_store_kv_cache` 算子 bs=64 精度联调，Ascend950（dav-3510，Ascend950PR V100）/ CANN 9.1.0。

- bs=64 `k_rope_cache` 断言失败：NPU RoPE 输出呈"每隔一个元素为零"模式，golden 全非零、npu 多处 0.0000（来源平台：950；证据等级：tool_observed）
- `debug_rope.py` 逐元素对比输出 `[27] golden=0.373047 npu=0.000000 FAIL` 等，零值 index 集合 {27, 30-31, 34-35, …, 62-63}，自 27 起每 4 元素 2 零（来源平台：950；证据等级：tool_observed）
- `DoRope64` 前后加 `PipeBarrier<PIPE_V>()` 后零值仍持续，排除 V 流水线 race（来源平台：950；证据等级：tool_observed）