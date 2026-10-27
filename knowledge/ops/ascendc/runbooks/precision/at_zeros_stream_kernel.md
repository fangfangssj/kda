---
type: "Runbook"
title: "at::zeros 异步零初始化跨 stream 覆盖 kernel 写出"
description: "kernel 直调算子的 host 层用 allocate_xpu_zeros_like（at::zeros）在 torch 默认 stream 异步零初始化输出 tensor，kernel launch 走 XpuRuntimeManager::get_ins<aclrtStream> 的 xpu_ops stream，两 stream 无同步；批量高负载下零初始化在 kernel 写入之后完成，整段 GM 写入覆盖 kv_a_out new 区。判别=BUGDIAG 按 region 分桶报 hist_zr=0 new_zr 非零 + 源码审查命中 allocate_xpu_zeros_like 且 kernel 在异 stream。"
tags: ["task.debugging", "topic.precision", "topic.implementation.synchronization", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-18T07:46:55Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# at::zeros 异步零初始化跨 stream 覆盖 kernel 写出

## 现象

在 dav-3510（Ascend950，CANN 9.1.0）上对 `rms_norm_rope_k_store_kv_cache` 算子跑数值 UT（`test/numerical_test/ascend/test_rms_norm_rope_k_store_kv_cache.py`，`DropRate=0.2 python -u -m pytest ... -v -s --no-header --tb=no`）。perf_3 优化版本合入后批量 pytest 仍 12 failed / 50 passed。批量 pytest 的 `[BUGDIAG]` tool 输出显示 `zero_rows=254244 hist_zr=0 new_zr=254244`——全部 0 行都在 new 区（`Process` 应写的 `[outBase, outBase+nowSeqLen)`），hist 区（`ProcessCache` 回写历史 KV）全对。

`kv_a_out`（RMSNorm 归一化输出，shape `[totalLen, kv_lora_dim]`，`totalLen = seqLenSum + multi_kv_cache_len_sum`）的失败特征：new 区尾部整行为 0，gold 正常；同一次迭代里写 `k_rope_cache` 的 cache 断言始终通过，只有写 `kv_a_out` 的输出被覆盖（cache 信号为 agent 观测，未作为 tool_result 落地）。

## 适用场景

- **工程形态**：kernel 直调工程（`kernels::RmsNormRopeKStoreKvCache::compute` 委托 + `<<<>>>` launch）；host 侧 layer `.cpp` 负责输出 tensor 分配与 kernel 启动。
- **API 路径**：`allocate_xpu_zeros_like` / `allocate_xpu_tensor_like`（`ops/common/xpu_allocator.h`，分别走 `at::zeros` / `at::empty`）；kernel launch 用 `runtime::XpuRuntimeManager::get_ins<aclrtStream>()` 获取的 stream。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510），CANN 9.1.0；a2/a3 未验证。
- **适用粒度**：`operator_family`——所有"host 预零初始化输出 + kernel 在另一 stream 写入"的 kernel 直调算子通用，不限于本算子。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；`allocate_xpu_zeros_like → allocate_xpu_tensor_like` 的逐字 after 代码与修后 62/304 passed 闭环信号未从 trace 作为 tool_result 恢复。
- **不适用/未验证场景**：若 kernel 实际不覆盖某些输出行（"那些行为 0"是算子语义一部分），不能用 `at::empty`，须显式同步两条 stream 或在 kernel stream 上做零初始化；a2/a3 未验证。

## 判别方法

**触发条件**（tool_observed 失败症状）：kernel 直调算子在批量 pytest 下出现 `kv_a_out` new 区整行为 0，BUGDIAG 报告 `hist_zr=0` 且 `new_zr` 随批量/seqLenSum 增大而膨胀（全部 0 行落 new 区，hist 区全对）。失败非确定性：隔离单 case 通过、批量跑触发（此非确定性信号为 agent 观测，未作为 tool_result 落地，仅作判别佐证）。

**可执行探针**：
1. 在批量 pytest 入口加 BUGDIAG，按 region 分桶统计 zero_rows：`hist_zr`（hist 区 0 行数）与 `new_zr`（new 区 0 行数）。若 `hist_zr=0` 且 `new_zr` 随批量/seqLenSum 膨胀，排除 kernel 内 dispatch 遗漏（dispatch 漏写应为确定性边界丢失，而非整段 new 区丢失）。
2. 打开 layer `.cpp` 的 compute 函数，grep 输出 tensor 分配：若用 `allocate_xpu_zeros_like`（底层 `at::zeros`），即命中本判别模式。
3. 确认 kernel launch 所用 stream 与 `at::zeros` 所用 stream 是否同一 stream：`at::zeros` 在 torch 默认 NPU stream 异步执行零初始化；kernel launch 若用 `runtime::XpuRuntimeManager::get_ins<aclrtStream>()` 取到的 xpu_ops stream，两 stream 间无显式同步即命中。
4. A/B 对照：把 `allocate_xpu_zeros_like` 临时换成 `allocate_xpu_tensor_like`（`at::empty`，不做 GM 写入）重跑批量 pytest，观察 `new_zr` 是否归零（此步 after 代码与修后通过信号未作为 tool_result 恢复）。

**原始观测**（Ascend950，dav-3510，CANN 9.1.0）：
- TQue 改造后批量 pytest 的 `[BUGDIAG]` tool_result：`zero_rows=254244 hist_zr=0 new_zr=254244`，全部 0 行在 new 区，hist 区全对；0 行数量比 TBuf 版更大（254244 vs 202447）。
- 缺陷存在于源码：layer `.cpp` compute 输出分配处为 `allocate_xpu_zeros_like(kv_a_in, {totalLen, kv_lora_dim})` 与 `allocate_xpu_zeros_like(kv_a_in, {totalLen, qk_rope_dim})`。

**机制解释**（根因分析，属 interpretation）：`allocate_xpu_zeros_like` 底层是 `at::zeros(shape, options)`，在 **torch 默认 NPU stream** 上异步执行零初始化（一次整段 GM 写入）；kernel launch 走 `runtime::XpuRuntimeManager::get_ins<aclrtStream>()` 拿到的 **xpu_ops stream**，两条 stream 之间没有显式同步。竞争链：host 调 `at::zeros` → torch stream 排入零初始化（异步）；host 调 `<<<..., xpu_ops_stream>>>` → xpu_ops stream 排入 kernel；两条 stream 独立调度，NPU 负载低时零初始化先于 kernel 完成，NPU 负载高（批量跑、大 seqLenSum）时零初始化被推迟到 kernel 写入**之后**完成，整段 GM 零初始化写入覆盖 kernel 刚写出的 `kv_a_out` new 区 → 整行回到 0。hist 区由第二个 kernel `ProcessCache()` 写入，且发生在 `aclrtSynchronizeStream(xpu_ops_stream)` 之后，时序晚于零初始化，故不被覆盖；cache（`k_rope_cache`）是调用方传入的已初始化 tensor，不走 `allocate_xpu_zeros_like`，也不受影响。整段 new 区丢失（而非个别边界 token）本身就反证了"tiling 边界 / dispatch 遗漏"误诊——dispatch 逻辑经逐项核对 100% 覆盖 new 区每行。

## 预防

- kernel 直调工程里，凡是输出 tensor 由 host 预分配、kernel 在非 torch 默认 stream 上写入的，默认用 `at::empty` / `allocate_xpu_tensor_like`，不用 `at::zeros` / `allocate_xpu_zeros_like`；只有当 kernel 不覆盖某些行、且"那些行为 0"是算子语义时才预零初始化，且必须在 kernel launch 前显式同步 torch stream（`aclrtSynchronizeStream` 当前 NPU stream）或在 kernel 所在 stream 上做零初始化。
- 代码评审时把"host 输出分配 stream 与 kernel launch stream 是否一致 / 是否同步"作为常规检查项，优先于去查 tiling 边界与流水 fence。
- 参考同仓已通过的相似算子（如 `fused_qk_rms_norm_store_kv`，用 `allocate_xpu_tensor_like` 由 kernel 完全填充输出）作为基线。

## 坏实践

- 用 `at::zeros` 预初始化一个"kernel 会完整覆盖"的输出 tensor，且不在 kernel launch 前同步 torch stream——本 bug 的坏实践。
- 把零初始化当作"安全默认"无脑加在所有输出 tensor 上。
- 看到 new 区整行 0 就先去查 tiling 边界 / dispatch 遗漏 / 流水 fence（均误诊，见 `## 误诊`）。

## 误诊

调试过程中先后走过的弯路（均未修复，供下个 agent 避免）：
- **误诊 1：tiling 边界 / dispatch 遗漏**。逐项核对 dispatch 逻辑后确认 100% 覆盖 new 区，非根因。
- **误诊 2：TBuf 手动 fence 在并发下偶发写丢失**（见同 trace 的 build 域 negative_result 卡）。TQue（bufNum=1 / 输入 double / 全 double）4 个方案均未修复，全 double 反触发 507035 UB crash。
- **误诊 3：跨迭代 MTE2→V / V→MTE3 WAR hazard**。改 double buffer 后 0 行数量反而更大（254244 vs 202447），且 replay 不复现，与"同步 hazard"的确定性特征矛盾。
- 真正定位信号是"new 区整段丢失 + hist 区全对 + cache 不受影响 + 仅批量触发"——指向 host 侧 GM 写入竞争而非 kernel 内同步。

## 验证信号

- TQue 改造后批量 pytest 的 `[BUGDIAG]` tool_result：`zero_rows=254244 hist_zr=0 new_zr=254244`，全部 0 行在 new 区，hist 区全对（来源平台：950；证据等级：tool_observed）
- 缺陷存在于源码：layer `.cpp` compute 输出分配处为 `allocate_xpu_zeros_like(kv_a_in, {totalLen, kv_lora_dim})` / `allocate_xpu_zeros_like(kv_a_in, {totalLen, qk_rope_dim})`（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 本卡 `platforms` 只含 `950`（dav-3510），trace 内实测仅在该平台。a2/a3 未验证，推断可迁移：跨 stream 异步零初始化竞争是 torch_npu + xpu_ops stream 模型的通性，不依赖 950 特有硬件，但本卡未在 a2/a3 上复现或回归。
- 若目标平台的 kernel 直调框架改用 torch 默认 stream 启动 kernel（两 stream 合一），竞争自然消失，本卡判别不适用也无需。
- 若算子语义改变为"kernel 只写部分输出行、其余行必须为 0"，则 `at::empty` 不安全，需回到显式同步方案。