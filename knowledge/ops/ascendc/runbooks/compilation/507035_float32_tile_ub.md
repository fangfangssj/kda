---
type: "Runbook"
title: "507035 — 三队列 float32 tile 超 UB"
description: "Ascend950/dav-3510 上多 TQue+双缓冲+大固定 tile 的 float32 Reg kernel 运行期抛 ACL 507035；用同环境对照算子+缩 tile 探针二分定位为每核 UB 超额，而非 D2H 拷贝故障。"
tags: ["task.debugging", "topic.memory", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
aliases: ["TPipe::InitBuffer"]
---

# 507035 — 三队列 float32 tile 超 UB

## 现象

在 Ascend950PR_9579（NpuArch dav-3510）上为 masked_scale（`y = x * mask * scale`）写了一个**纯 float32、用 AscendC::Reg API** 的 kernel：3 个 `TQue`（inQueueX_ / inQueueMaskF_ / outQueueZ_，均 `PIPELINE_DEPTH=2`），tiling 固定 `FIXED_TILE_ELEMS=16384`。编译、`bash build.sh --install` 均成功；运行本地 `pytest tests/masked_scale/` 时**全量失败**，报：

```
FAILED ...::test_masked_scale_operator[shape5-dtype_combo1] - RuntimeError: copy_between_host_and_device_opapi:torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:57 NPU function error: device error type 3, error code is 507035
...
================== 80 failed, 1 passed, 14 warnings in 11.76s ==================
```

错误名 `copy_between_host_and_device_opapi` 极具误导性（看似 D2H 拷贝失败）。用 32 元素 / 8 元素最小用例复现，`torch.npu.synchronize()` 直接抛 507035；同进程同环境调用 `cann_bench.exp(x)`（同仓 exp 算子，1024×1024 float32）正常返回 `exp OK: tensor([[1.6654, 3.2601], ...])`，排除 NPU 硬件/驱动故障。最终把 `FIXED_TILE_ELEMS` 从 16384 改为 4096 后，32 元素与 1M 元素用例均 OK，本地 81 passed。

## 适用场景

- **工程形态**：ops-direct-invoke-flash 核函数构建（直接 Launch，非 aclnn 算子仓），`__global__ __aicore__ __vector__` kernel + `AscendC::Reg` 向量计算。
- **API 路径**：`AscendC::TQue<TPosition,DEPTH>` + `pipe_.InitBuffer(que, DEPTH, tileSize*sizeof(T))` + `AscendC::Reg::LoadAlign/Mul/Muls/StoreAlign`；`asc_vf_call<...>` 进 `__simd_vf__` 函数。
- **验证平台与版本**：Ascend950PR_9579 / NpuArch dav-3510（CANN 9.1.0，tikcpp impl 路径 `/home/<user>/Ascend/cann-9.1.0/x86_64-linux/tikcpp/tikcfw/impl/dav_3510/`）。仅此平台实测。
- **适用粒度**：`operator_family`——任何在 dav-3510 上用多 TQue + 双缓冲 + 大固定 tile 的元素级 elementwise Reg kernel 都可能踩同一 UB 超额。
- **路线优先级**：`recommended_fastpath`——把 tile 纳入 UB 预算约束（见判别方法机制解释），从源头避免 507035。
- **不适用/未验证场景**：a2/a3 未验证（推断可迁移，见失效边界）；非 dav-3510 平台的 UB 容量不同，算式需按目标平台重算。

## 判别方法

**触发条件**：在 dav-3510 上运行多 TQue + 双缓冲 + 大固定 tile 的 float32 Reg kernel，`build.sh` 编译成功、`pytest` 运行期抛 `RuntimeError: copy_between_host_and_device_opapi ... device error type 3, error code is 507035`，且最小用例（8/32 元素）`torch.npu.synchronize()` 同样抛 507035。错误名指向 D2H 拷贝路径，具有强误导性——需用下述探针二分定位真正根因。

**可执行探针**（三步二分）：
1. **同环境对照算子**：同进程同环境调用一个已知正常的同仓算子（如 `cann_bench.exp(x)`，1024×1024 float32），看是否正常返回。
   - 若 exp 正常 → 排除 NPU 硬件/驱动故障，问题在自家 kernel。
   - 若 exp 也报错 → 先查硬件/驱动，不在本卡范围。
2. **最小用例复现**：用 8/32 元素最小输入跑 `torch.npu.synchronize()`，确认 507035 仍复现（排除大 shape 才触发的边界）。
3. **tile 缩减对照**：把 tiling 函数的 `FIXED_TILE_ELEMS` 从 16384 改为 4096，重编重跑同一 pytest：
   - 若缩 tile 后 507035 消失、全量 passed → 确诊根因为 UB 容量超额。
   - 若缩 tile 后仍 507035 → 排除 UB 预算，改查 TBuf 生命周期 / GM 越界等其他 507035 路径。

```cpp
// 探针改动点（tiling 函数内，单行常量）：
//   constexpr static uint32_t FIXED_TILE_ELEMS = 16384;  // 触发 507035
constexpr static uint32_t FIXED_TILE_ELEMS = 4096;      // 探针：缩 tile 后通过
```

**原始观测**（Ascend950/dav-3510，masked_scale float32 Reg kernel，单 trace 多轮）：
- tile=16384 时 `pytest tests/masked_scale/` 80 failed/1 passed，全部 `RuntimeError: ... error code is 507035`；32/8 元素最小用例 `torch.npu.synchronize()` 同样抛 507035。
- 同进程 `cann_bench.exp(x)`（1024×1024 float32）返回 `exp OK: tensor([[1.6654, 3.2601], [0.3272, 1.2985]])`，证明 NPU 硬件正常。
- tile=4096 后 32 元素与 1M 元素用例均 OK，本地 `pytest tests/masked_scale/` `81 passed, 14 warnings in 19.22s`。
- 官方评测 `run_evaluation.sh --operator MaskedScale` 精度 20/20 通过。

**机制解释**：dav-3510 每核 UB 容量有限（本开发会话推算为 256 KiB；同批另一卡的会话估值为约 192KB/196608 字节，两口径均为会话估算、超容结论在两值下均成立，见 [507035 group 统计全载超限卡](507035_group_stats_fullload_colsplit.md)）。该 kernel 的 UB 占用为：

```
UB占用 = 队列数 × PIPELINE_DEPTH × FIXED_TILE_ELEMS × sizeof(元素)
       = 3        × 2               × 16384          × 4 B
       = 393216 B = 384 KiB  > 256 KiB（每核 UB 上限）
```

三个 TQue（x/mask/z 各一）× 双缓冲 × 16384 元素 × float32(4B) = 384 KiB，超过每核 UB；`pipe_.InitBuffer` 在编译期无法静态拒绝（tile 是 `constexpr` 但 UB 预算约束未在 API 层强制），运行期 kernel launch 后硬件访问越界 UB，最终以 `device error type 3, error code 507035` 经 D2H 拷贝路径抛出。把 tile 降到 4096 后：

```
3 × 2 × 4096 × 4 B = 98304 B = 96 KiB  < 256 KiB  → 通过
```

注意 `copy_between_host_and_device_opapi` 是错误呈现路径，不是根因——真正越界的是 kernel 对 UB 的访问，D2H 只是把 device 上已发生的 507035 传播回 host。这是 507035 在 dav-3510 上的多义性：同一错误码可由 UB 超额、TBuf 未初始化、GM 越界等多种根因触发，必须用上述「对照算子 + 最小用例 + 缩 tile」三步探针二分定位，不可凭错误名直判。

## 验证信号

- tile=16384 时 `pytest tests/masked_scale/` 80 failed/1 passed，全部 `RuntimeError: ... error code is 507035`；32/8 元素最小用例 `torch.npu.synchronize()` 同样 507035（来源平台：Ascend950/dav-3510；证据等级: tool_observed）
- 同进程 `cann_bench.exp(x)`（1024×1024 float32）返回 `exp OK: tensor([[1.6654, 3.2601], [0.3272, 1.2985]])`，证明 NPU 硬件正常（来源平台：Ascend950/dav-3510；证据等级: tool_observed）
- tile=4096 后 32 元素与 1M 元素用例均 `OK`，本地 `pytest tests/masked_scale/` `81 passed, 14 warnings in 19.22s`（来源平台：Ascend950/dav-3510；证据等级: tool_observed）
- 官方评测 `run_evaluation.sh --operator MaskedScale` 精度 20/20 通过（来源平台：Ascend950/dav-3510；证据等级: tool_observed）

## 失效边界

- 仅 Ascend950/dav-3510 实测；a2/a3 未验证，推断可迁移——但每核 UB 容量按目标平台规格重算后才能确定安全 tile。
- UB 预算算式随队列数、`PIPELINE_DEPTH`、dtype 字节数变化；若改双缓冲为单缓冲或减队列数，tile 上限可上调，需重新代入 `Nqueue×DEPTH×tile×sizeof(T) ≤ UB`。
- `copy_between_host_and_device_opapi` 错误名对本根因有强误导性；其他 507035 路径（如真 TBuf 未初始化、GM 越界）不归本卡，需独立判别。

## 误诊

agent 第一次命中 507035 时（kernel 含 `TBuf<VECCALC> tmpMaskFloat_` 用于 b16→float Cast），从 common-failure-modes 文档套用为「TBuf::Get<T>() 未在与 InitBuffer 相同的 `if constexpr` guard 内」，于是**移除整个 TBuf、把 b16→float 转换移到 plugin 侧 `x.to(torch::kFloat32)`**，重编后 507035 原样复现（80 failed/1 passed）。误判点：把 507035 等同于 TBuf 生命周期问题，忽略了 UB 预算这个更上游的约束。真正排障拐点是本卡判别方法的三步探针：同环境 `cann_bench.exp` 正常 → 问题在自家 kernel → 缩小 tile 立即通过，证明根因是 UB 容量而非 TBuf。

## 坏实践

- 把 507035 一律套用 common-failure-modes 的「TBuf Get() 未初始化」条目而盲改 TBuf，掩盖了 UB 预算这个真因，浪费一轮 rebuild+test。507035 在 dav-3510 上是多义信号，必须用最小用例 + 同环境对照算子 + 缩 tile 三步探针二分定位。
- 见 `copy_between_host_and_device_opapi` 错误名就归因为 D2H 拷贝故障——它只是 507035 的传播路径，真正越界发生在 kernel 对 UB 的访问。

## 预防

写多 TQue + 双缓冲 + 大固定 tile 的 elementwise Reg kernel 前，先代入 `Σ(队列数 × DEPTH × tileSize × sizeof(dtype)) ≤ 每核 UB` 选 tile；不要把 `copy_between_host_and_device_opapi`/507035 默认归因为 TBuf 生命周期或 D2H 拷贝，先用同环境已知正常算子（如 exp）排除硬件、再缩 tile 验证 UB 假设。

## 复现增量：推导型 ubFormer 恰满报告容量变体 + 保守 cap 回退（机制未验证）

另一独立轨迹并入 bessel_i1e 算子（ascend950/CANN 9.0.0-beta.2）的边界变体，与原实例（固定 tile 384KiB 明显超 256KiB）互补——本变体是 **tiling 按报告容量推导的 tile"恰好占满"报告值时崩溃**：

- 现象：尾块越界修复（另卡）后，小/中 shape 用例完成，大 dim0 用例（98336/16.7M/32768）持续 `ACL error 507035`，且 `01_f32_normal_128x256` 随 tiling 参数漂移时而复现时而消失。
- 判别事实：**崩溃与 Compute 内容无关**——把 Compute 换成 pass-through（CopyViaVF）或单条 `AscendC::Abs(yLocal, xLocal, c)` 都照样崩，崩溃用例集合一致；只与 dim0 量级相关（dim0<1024 全过）。tiling 推演：FP32 下 `ubFormer=12288`，`InitBuffer(queIn, 1, 12288*4)` + `InitBuffer(queOut, 1, 12288*4)` 共 98304 字节，**恰好等于按 96KB 口径报告的 UB 值，无任何余量**。
- 工作假设（未验证）：tiling 用 `GetCoreMemSize(UB)` 报告的 ubSize 推导 ubFormer，而 InitBuffer 实际可用容量小于该值（或另有保留），双队列恰好占满推导值时溢出触发 507035；真实平台 UB 大小在该会话未打印，无法闭合算术。

保守回退（safety_fallback，机制未追到根因）——多级防御 cap，核心是 1024 硬 cap：

```cpp
    // §5.2.2 UB 切分（统一字节量纲）
    // Use a conservative ubSize cap - platform may report unrealistic values on simulator.
    // Typical Ascend 950 UB is 96KB; cap at 256KB for headroom.
    const int64_t safeUbSize = std::min(ubSize, static_cast<int64_t>(256 * 1024));
    int64_t bufferDivisor = static_cast<int64_t>(bufferNum) * static_cast<int64_t>(elemBytes);
    int64_t maxElemNum    = safeUbSize / bufferDivisor;
    int64_t alignFactor   = UB_ALIGN_BYTES / static_cast<int64_t>(elemBytes);
    out.ubFormer = AlignDown(maxElemNum, alignFactor);
    if (out.ubFormer < 1) { out.ubFormer = alignFactor; }

    // DataCopy blockLen is uint16_t → count * elemBytes <= 65535
    const int64_t maxDcElements = 65535 / static_cast<int64_t>(elemBytes);
    out.ubFormer = std::min(out.ubFormer, maxDcElements);

    // Hard cap to prevent UB overflow on mis-reported platform info.
    out.ubFormer = std::min(out.ubFormer, static_cast<int64_t>(1024));
    out.ubFormer = std::min(out.ubFormer, out.blockFormer);
```

二分轨迹：12288 崩 → 8192 崩 → 1024 全过（真实可用上界介于 1024 与 8192 之间，未测定）。定位方法补充：用独立脚本复算 tiling 参数（dim0/coreNum/blockFormer/ubFormer/ubLoop/ubTail）并检查 queIn+queOut（含 depth）字节和与 UB 容量的关系；"崩溃只随 dim0 量级出现且与 Compute 内容无关"即可排除算术、锁定 tiling/内存布局。

**独立 trace 新增验证信号**（均为 tool_observed）：

- 修复前：0-C（dim0=98336）/0-E（dim0=16,777,216）/01（dim0=32768）持续 `ACL error 507035`；cap 8192 后 0-C/0-E/01 仍崩（来源平台：950；证据等级：tool_observed）。
- 判别信号：pass-through 与 Abs-only 两种 Compute 下崩溃用例集合一致（0-C/0-E/01），证明与计算内容无关（来源平台：950；证据等级：tool_observed）。
- 修复后（ubFormer cap 1024）：`ALL TESTS NOW COMPLETE (no crashes)`——14/14 用例完成（Abs-only 验证下输出 |x| 数值正确）（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界与坏实践**：cap 1024 后全量完成只证明"该 cap 下无崩溃"，不能当作根因结论写入设计文档；采用代价为 UB tile 变小、Process 迭代次数增多（相对 12288 约 12 倍迭代，性能未测量），机制层修复（探明平台真实 UB 可用容量与 InitBuffer 实际分配规则）后应回退此 cap。坏实践补充：用 `GetCoreMemSize(UB)` 的返回值无余量地推导双队列 tile 尺寸（bufferNum 除数只算队列数与元素宽度，不留 headroom）。关联：该 cap 后续残留在共享 ComputeTiling 引发公式型 TilingUT 大面积回归（去 caps 又暴露 kernel 侧 19 buffer 超 UB 的独立缺陷），见库内"共享 Tiling 仿真 caps 致公式 UT 回归"与"每中间量全尺寸 TBuf 总量超 UB"卡。
