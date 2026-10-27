---
type: "Runbook"
title: "SIMT 切分未 coalesce 致 block 边界错位"
description: "多维大 shape 随机算子触发 SIMT split-block 切分时，op_host 侧 InitTensorSliceState 直接写原始多维 shape，未像 PyTorch TensorIterator 那样先 coalesce_dimensions，使 block0 numel 比参考实现少 stride[0]/2，block1 首元素 global idx 整体错位、counter 跳变点位置错。判别信号：block1 首 idx 与期望差恰好等于 stride[0]/2。"
tags: ["task.debugging", "topic.programming_model.execution_model.simt", "topic.implementation.tiling", "topic.precision"]
status: draft
sources: []
created_at: "2026-08-18T07:59:33Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
aliases: ["GetMaxOffsetBytes", "split_for_32bit"]
---

# SIMT 切分未 coalesce 致 block 边界错位

## 现象

在调 `sim_thread_exponential`（arch35）随机数算子对齐 PyTorch/CUDA curand Philox4_32_10 指数分布输出时，给定 8 维 C-contiguous shape `[81989,7,76,1,1,2,1,7]`（numel = 81989×7×76×1×1×2×1×7 = 610654072）、dtype fp32。因为 `(shape[0]-1)×strides[0]×elementSize = 81988×7448×4 ≈ 2.44 GB > 2³¹−1`，`Is32bitIndexable()` 判定 false，tiling 走 `CalcSplitBlocks` 二分堆栈把输出切成两块 block。

block0 末尾几个值（global idx 305323311、305323312、305323313）与 golden 参考对得上；但从 block1 的首元素（global idx 305327036，期望 1511.488，counter `[0x5AE,0,0,0]`）开始全部错位——旧 tiling 把 block1 首元素算成 305323312（即 `40994×7448`），实际期望是 305327036（即 `610654072/2`）。差值正好是 3724 = 7448/2 = stride[0]/2。

## 适用场景

- **工程形态**：op_host 侧 SIMT 模式 tiling 计算（`kernelMode=SIMT`、`enableSplitBlocks=true` 走 `FillUnifiedSimtTilingData`），kernel 直调随机数算子。
- **API 路径**：`InitTensorSliceState` → `CalcSplitBlocks` → `CalcExecutionPoliciesForBlocks`（`random_common/op_host/arch35/random_tiling_arch35.cpp`），喂给 `PhiloxSimtKernelDiscontinuous` SIMT kernel（`op_kernel/arch35/random_kernel_base.h` 的 `FlashCounter`/`SkipHi`/`SkipLo`）。
- **验证过的平台与 CANN 版本**：trace 内无 `npu-smi info`/SoCVersion 探测事件，平台未定位（arch35 目录暗示特定昇腾硬件，但无显式上板实测证据）；判别信号来自主机侧 golden 仿真（Python `exponential_golden.py` + `random_golden_base.py`），NPU 出包验证在本 trace 内未闭环。`platforms: []` 如实反映未定位。
- **适用粒度**：`operator_family`——通用于走 `random_common/op_host/arch35` 同一套 `TensorSliceState`+`CalcSplitBlocks` 切分逻辑的随机算子（exponential/uniform/normal 等），只要输出多维且触发 split-block。
- **明确不适用/未验证场景**：① 不触发 split-block 的小 shape（`Is32bitIndexable()` 为 true，单块直跑）不受此判别影响，coalesce 前后结果一致；② NPU 实际 fp16/fp32 出包在本 trace 内未对齐，其根因未确认是否与本卡同源。

## 判别方法

- **触发（tool_observed 失败）**：随机算子在多维大 shape（numel 使 `maxOffsetBytes > 2³¹−1`、`Is32bitIndexable()` 返回 false、走 `CalcSplitBlocks`）下，block1 首元素的 global idx 与 golden/PyTorch 参考不一致。本例：block1 首元素期望 global idx 305327036、值 1511.488、counter `[0x5AE,0,0,0]`；旧 tiling 给 block1 首 = 305323312，counter 跳变点错位 3724（来源平台：未定位；证据等级: tool_observed）。
- **探针（可执行核验）**：在主机侧 golden 仿真里同时算两种 block0 numel 并比对：
  1. 旧 tiling 路径：按原始多维 shape 切 dim0，`copy_size = shape[0]/2`，block0 numel = `copy_size × strides[0]`；
  2. PyTorch 参考路径：先 `coalesce_dimensions`（去 size=1 维 + 合并连续维）得 1D shape，`copy_size = numel/2`，block0 numel = `numel/2`；
  3. 计算两者差 `delta = numel/2 − copy_size × strides[0]`，并核验 `delta == strides[0]/2`。
  命令样例：`python exponential_golden.py <seed> --shape 81989 7 76 1 1 2 1 7 --rate <r> --offset <o> --idx 305323311 305323312 305327035 305327036 305327037 --show-blocks`，对比 block 边界 idx 与 counter 跳变点。
- **观测（tool_observed）**：仿真输出 `golden block0 numel = 305323312`、`实际 block0 numel = 305327036`、`差值 = 3724`，且 `3724 == 7448//2: True`（7448 = strides[0]）；block1 首 idx 305327036 的 golden 值 `gen(0x5AE, 0, 0) = 1511.487722` 对齐期望 1511.488，而旧 tiling 把该位置归给 block0 末段、counter 仍停在 `0x5AD`（来源平台：未定位；证据等级: tool_observed）。
- **判别（根因解释）**：delta 恒等于 `strides[0]/2` 这一特征量，说明切分发生在"未合并的 dim0 中点"而非"coalesce 后 1D 的中点"。根因是 `InitTensorSliceState` 把 `gert::Shape` 的原始多维 shape（含 size=1 维度）直接写入 `state.shape/state.strides`，没有像 PyTorch `TensorIterator::build` 那样先调 `coalesce_dimensions()` 合并：对本例 `[81989,7,76,1,1,2,1,7]`，coalesce 后应得 1D `[610654072]`、strides `[1]`，`split` 的 `copy_size` 应为 `610654072/2 = 305327036`；而旧 tiling 在原始 8 维上切 dim0 得 `copy_size = 81989/2 = 40994`、block0 numel = `40994×7448 = 305323312`。两者差 `305327036 − 305323312 = 3724 = 7448/2`，即"半个 stride"的语义差。block 边界错位喂给 `CalcExecutionPoliciesForBlocks` 的 `accumulatedOffset` 累加，使 block1 起始 counter 提前/错后，输出从 block 边界起全错。注意：`CalcSplitBlocks` 与 `CalcExecutionPoliciesForBlocks` 自身逻辑与 golden 一致，无需改动，唯一源头是喂入的 shape/strides 未 coalesce。

## 验证信号

本卡判别来源：本 trace 内主机侧 golden 仿真（`exponential_golden.py` + `random_golden_base.py`），未在 NPU 上板闭环。

- 旧 tiling 切分 dump：block0 形 `[40994,7,76,1,1,2,1,7]`、numel `305323312`；block1 起 `[40995,7,76,1,1,2,1,7]`、offset `305323312`——block0 numel 比 PyTorch coalesce 后期望 `305327036` 少 3724（来源平台：未定位；证据等级: tool_observed）。
- golden 仿真核验：`golden block0 numel = 305323312`、`实际 block0 numel = 305327036`、`差值 = 3724`，`3724 == 7448//2: True`，`7448 == strides[0]`（来源平台：未定位；证据等级: tool_observed）。
- block 边界采样点对齐：idx 0=355.477、305323311=733.278、305323312=412.752、305323313=695.538、305327035=1425.370（block0 末）、305327036=1511.487722（block1 首，counter 从 `0x5AD` 跳到 `0x5AE`、thread 归零）、305327037~039=405.268/1183.923/399.832，均匹配期望（来源平台：未定位；证据等级: tool_observed）。

## 失效边界

- 本判别在 `op_host/arch35/random_tiling_arch35.cpp` 的 `InitTensorSliceState` 上成立；同仓 `random_common` 下其他 arch 目录若同样缺 coalesce，判别信号（delta == stride[0]/2）应同样成立——未验证，推断可迁移。
- 非 C-contiguous 输入（`is_contiguous_` 为 false）：coalesce 的"合并连续维"判定 `shape[dim]×stride[dim] == stride[dim-1]` 对非连续 stride 不成立、不合并，只去 size=1 维。此场景在本 trace 未验证，判别信号形态可能变化（delta 不再恒为 stride[0]/2），未实测。
- 推断可迁移平台（a2/a3/Ascend950）：arch35 目录暗示某昇腾硬件，但本 trace 无 `npu-smi info`/SoCVersion 探测，未上板实测；`platforms: []` 如实反映未定位，不得据 arch35 推断填入。
- 不触发 split-block 的小 shape（`Is32bitIndexable()` true）：单块直跑，无 block 边界，本判别不触发。
- NPU 出包验证在本 trace 内未对齐（seq 413 用户报告"第0个值都不对"），其根因可能与本卡无关（疑似 transform 常数、printf 副作用、构建缓存或 blockDim 不一致）——判别信号仅在主机侧 golden 仿真闭环，不覆盖 NPU 实跑。

## 预防

- 任何"对齐 PyTorch 参考实现的算子"在做 tiling 切分/索引前，必须复刻 `TensorIterator::build` 的 `coalesce_dimensions`：先去 size=1 维，再合并 `shape[dim]×stride[dim] == stride[dim-1]` 的连续维。否则 split-block 边界、`can_use_32bit_indexing` 判定、`get_dim_to_split` 选维都会与 PyTorch 分歧。
- 调试随机算子 block 边界错位时，先用本卡判别信号——核对"block 首/末元素的 global idx 与 counter 跳变点"是否对齐参考实现的 coalesce 后 shape，且 `delta == strides[0]/2` 是否成立——确认后再怀疑 counter 布局或 transform 常数（后者在本 trace 是误诊）。

## 误诊

- **误诊一：transform 的 uniform 转换常数差异**。一度怀疑 NPU kernel `u = u32 × RAND_2POW32_INV + RAND_2POW32_INV_HALF`（+1.164e-10，半值）与 golden `u = u32 × RAND_2POW32_INV + RAND_2POW32_INV`（+2.328e-10，全值）差一倍是 fp16 "差很多"的根因（来源平台：未定位；证据等级: agent_reported）。机制核算：`u32×RAND_2POW32_INV` 的值（量级 0.3）远大于常数项差 1.164e-10，在 fp32 精度下被完全吸收，fp16 精度更低更被吸收；仿真显示 fp16 最大差 0.000584（精度级），非"差很多"。此假设被本 trace 否定。
- **误诊二：kernel blockDim=512（`DEFAULT_SIMT_THREAD_NUM`）vs tiling `SIMT_THREAD_GROUP_SIZE=256` 不一致**。用户假设这是仅 fp16 出错的根因（来源平台：未定位；证据等级: agent_reported）。机制分析：kernel grid-stride 用 `linearIndex < totalThreads` 守卫，`totalThreads` 来自 tiling（256 基），`blockDim=512` 时 threadIdx 256~511 的 `linearIndex` 起即 ≥ totalThreads 被守卫跳过，只多 256 个空闲线程（性能问题非正确性）；仿真显示 tiling 参数（grid/threads/rounds/kernelOffset）完全不依赖 dtype。本 trace 未能确认此为真 bug。