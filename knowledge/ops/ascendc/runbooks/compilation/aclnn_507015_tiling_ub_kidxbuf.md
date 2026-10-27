---
type: "Runbook"
title: "ACLNN 507015 — tiling UB 漏算 kIdxBuf 下溢"
description: "cloud_sparse_inverse_conv3d_with_matmul ST 触发 507015 device error type 3；根因为 tiling 侧 CalcUbMaxTaskNum 未镜像 kernel 侧 kIdxBuf_/BLOCK_SIZE 且 uint64 无符号减法无下溢守卫，回绕产生垃圾 ubMaxTaskNum 致 kernel InitBuffer 超 UB。判别=tiling 打印 ubMaxTaskNum 为天文数字 + 核对 tiling 预算是否逐项镜像 kernel InitBuffer。"
tags: ["task.debugging", "topic.memory", "topic.tiling"]
status: draft
sources: []
created_at: "2026-08-18T07:49:55Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
aliases: ["unsigned_underflow"]
---

# ACLNN 507015 — tiling UB 漏算 kIdxBuf 下溢

## 现象

在调试 CANN `cloud-ops-turbo/driving/cloud_sparse_inverse_conv3d_with_matmul` 算子时，先在两个 tiling.cpp 的 `SetCommonTilingFields` 调用之前用 `fprintf(stderr, ...)` + `fflush(stderr)` 打印 12 个 tiling 字段，随后跑 ST 用例 `tests/st/test_cloud_sparse_inverse_conv3d_with_matmul.py::test_model_case0`。tiling 打印先输出：

```
[with_matmul tiling] usedVectorNum=2, usedCubeNum=1, kernelSize=27, totalTaskNum=19311, totalPointsCount=19497, startOffset=145635, inChannels=256, outChannels=192, ubMaxTaskNum=613566720, loopPointCount=20480, featureWspOffset=9175040
```

紧接着在 `getout` 把 device 上的 `conv3d_out.features` 拷回 host 时抛出：

```
RuntimeError: copy_between_host_and_device_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:57 NPU function error: device error type 3, error code is 507015
[ERROR] 2026-08-04-18:57:39 (PID:38109, Device:0, RankID:-1) ERR00100 PTA call acl api failed
```

即 tiling 阶段输出的 `ubMaxTaskNum=613566720` 是一个远超物理 UB 容量的垃圾值（with_matmul 的 `oneTaskSize = (inChannels+outChannels)*4 = (256+192)*4 = 1792 B`，613566720 行 × 1792 B ≈ 1.1 TB，不可能放进 UB），kernel 侧按此值 `InitBuffer` 必然溢出 UB，device 报 507015（device error type 3）。

## 适用场景

- **工程形态**：aclnn 算子开发（op_host tiling + op_kernel InitBuffer），ST 用例驱动。
- **API 路径**：`gert::TilingContext` → tiling 侧 `CalcUbMaxTaskNum` / `context->SetLocalMemorySize(availableUbSize)`；kernel 侧 `pipe_->InitBuffer(...)`（TPosition::VECCALC 的 `indexBuf_` / `kIdxBuf_` / `tmpFeaturesBuf_` / `tmpOutFeatureBuf_`）。
- **验证过的平台与 CANN 版本**：trace 内未探测 SoC/产品型号（无 `npu-smi info` / SoCVersion 输出），仅从 nn 版 `*_tiling_arch35.cpp` 参照推断可能为 arch35 系；**未定位平台，不写入 platforms**。CANN 版本未在 trace 内记录。
- **适用粒度**：`operator_family`（同一 `sparse_conv_tiling_common.h` 驱动的 with_matmul / grad 两个算子共用 `CalcUbMaxTaskNum`，根因与算子无关、与 UB 预算计算模式有关）。
- **路线优先级**：`recommended_fastpath`（在 tiling 侧精确镜像 kernel 的 `InitBuffer` 预算并加无符号下溢守卫，是首选正解；不是 fallback）。本卡只给判别方法与根因解释，逐字 after 代码与修后 ST 闭环未从 trace 恢复。
- **明确不适用/未验证场景**：修复后是否消除 507015 **未在本 trace 内复跑验证**；强制单 block（`usedCubeNum=1`）调试路线见 `## 失效边界`，未观测到运行结果。

## 判别方法

**触发**（tool_observed 失败症状）：ST 跑稀疏 inverse conv3d 算子，tiling 打印出现 `ubMaxTaskNum` 远大于物理 UB（with_matmul 此处为 613566720），随后 device 拷回 host 报 `error code 507015 / device error type 3`。`featureWspOffset = 20480 × (256+192) = 9175040` 自洽，说明 `loopPointCount` 与 channel 字段无误，异常孤立在 `ubMaxTaskNum`。

**可执行探针**（本 trace 实际执行的两步）：

1. 读 kernel 侧 `InitBuffer` 全部 UB 分配，列出每个 buffer 的字节算式。with_matmul kernel（`op_kernel/cloud_sparse_inverse_conv3d_with_matmul.h` L141-156，逐字）：

```cpp
    pipe_->InitBuffer(tmpFeaturesBuf_, ubMaxTaskNum_ * inChannels_ * INT32_SIZE);
    pipe_->InitBuffer(tmpOutFeatureBuf_, ubMaxTaskNum_ * outChannels_ * INT32_SIZE);
    pipe_->InitBuffer(indexBuf_, INDICES_BUFFER_COUNT * loopPointCount_ * INT32_SIZE + BLOCK_SIZE);

    tmpFeaturesLocal_ = tmpFeaturesBuf_.Get<float>();
    tmpOutFeatureLocal_ = tmpOutFeatureBuf_.Get<float>();

    inputIdxLocal_ = indexBuf_.Get<int32_t>();
    outputIdxLocal_ = inputIdxLocal_[loopPointCount_];
    outputIdxPtrLocal_ = outputIdxLocal_[loopPointCount_];
    idxInfoLocal_ = outputIdxPtrLocal_[loopPointCount_];

    uint32_t maskAlignLen_ = AlignUp(loopPointCount_, INT8_PER_LOOP);
    pipe_->InitBuffer(kIdxBuf_, maskAlignLen_ + maskAlignLen_ / BITS_PER_BYTE);
    kIdxLocal_ = kIdxBuf_.Get<uint8_t>();
    maskLocal_ = kIdxLocal_[maskAlignLen_];
```

   kernel 实际申请的固定 UB（不随 ubMaxTaskNum 变化部分）= `indexBuf_` + `kIdxBuf_`：
   - `indexBuf_ = INDICES_BUFFER_COUNT * loopPointCount * INT32_SIZE + BLOCK_SIZE`
   - `kIdxBuf_ = AlignUp(loopPointCount, INT8_PER_LOOP) + AlignUp(loopPointCount, INT8_PER_LOOP) / BITS_PER_BYTE`

2. 对照 nn 版 `cloud_sparse_inverse_conv3d_grad_tiling_arch35.cpp`（L34-48，逐字）看正确的 tiling 侧预算：

```cpp
    uint64_t ubSize = GetUbSize(platform);
    uint64_t dynamicUbSize = ubSize - DCACHE_UB_SIZE;
    context->SetLocalMemorySize(dynamicUbSize);
    uint32_t loopPointCount = 2048;
    uint32_t kIdxBufSize = AlignUp(loopPointCount, 256) + AlignUp(loopPointCount, 256) / 8;
    uint64_t availableUbSize = dynamicUbSize - INDICES_BUFFER * loopPointCount * INT32_SIZE - kIdxBufSize - 32;
    uint64_t oneTaskSize = (2 * shape.inChannels + shape.outChannels) * FEATURE_FLOAT_SIZE;
    uint32_t ubMaxTaskNum = availableUbSize / oneTaskSize;
    ubMaxTaskNum = ubMaxTaskNum < MINI_TASK_BLOCK ? ubMaxTaskNum : FloorAlign(ubMaxTaskNum, MINI_TASK_BLOCK);
    OP_CHECK_IF(ubMaxTaskNum == 0,
        OP_LOGE(context->GetNodeName(), "ubMaxTaskNum is 0"), return ge::GRAPH_FAILED);
```

   nn 版在算 `availableUbSize` 时**显式减去 `kIdxBufSize` 和 `+32` 的 BLOCK_SIZE padding**，并有 `ubMaxTaskNum == 0` 的守卫。

**原始观测**（tool_observed，修复前 `CalcUbMaxTaskNum` 的缺陷形态）：driving 版 `CalcUbMaxTaskNum`（`sparse_conv_tiling_common.h`，逐字 before）为：

```cpp
// 计算UB单次可承载的最大任务数（稀疏行数）
// UB布局：索引buffer区(inputIdx/outputIdx/outputIdxPtr) + 预留区 + 特征数据区
// oneTaskSize为单个稀疏行占用的UB字节数，不同算子不同：
//   with_matmul: (inChannels + outChannels) * 4  —— features + outFeatures
//   grad:        (2*inChannels + outChannels) * 4 —— gradOut + features + gradFeatures
inline uint32_t CalcUbMaxTaskNum(uint64_t availableUbSize, uint32_t loopPointCount, uint64_t oneTaskSize)
{
    // 扣除索引buffer区和预留区后，剩余空间全部分配给特征数据区
    uint64_t adjustedSize = availableUbSize - RESERVED_UB_SIZE - INDICES_BUFFER * loopPointCount * INT32_SIZE;
    uint32_t ubMaxTaskNum = adjustedSize / oneTaskSize;
    // 向16对齐以保证DataCopy对齐访问，不足16则不对其
    return ubMaxTaskNum < MINI_TASK_BLOCK ? ubMaxTaskNum : Ops::Base::FloorAlign(ubMaxTaskNum, MINI_TASK_BLOCK);
}
```

对照 kernel 侧 `InitBuffer` 清单，该预算**漏算 `kIdxBuf_`（`AlignUp(loopPointCount,256)+/8`）和 `indexBuf_` 的 `+BLOCK_SIZE` padding**，且 `availableUbSize - RESERVED_UB_SIZE - INDICES_BUFFER*loopPointCount*INT32_SIZE` 是 **uint64 无符号减法、无下溢守卫**。

**解释**（根因分析，属 interpretation；机制链含算式）：失败 case `loopPointCount=20480`、`inChannels=256`、`outChannels=192`。
- kernel 侧固定开销 `indexBuf_ = 3 * 20480 * 4 + 32 = 245792 B`（其中 `3 = INDICES_BUFFER_COUNT`，`32 = BLOCK_SIZE`）。
- kernel 侧 `kIdxBuf_ = AlignUp(20480, 256) + AlignUp(20480, 256)/8 = 20480 + 2560 = 23040 B`（20480 已 256 对齐）。
- 两者合计 `245792 + 23040 = 268832 B`，已超过这类芯片可用 UB 预算（`dynamicUbSize = ubSize - DCACHE_UB_SIZE(40*1024=40960)`，典型 UB 总量 ~192–256 KB，`dynamicUbSize` 仅 ~150–215 KB）。
- driving 版 `CalcUbMaxTaskNum` 只减 `RESERVED_UB_SIZE + INDICES_BUFFER*loopPointCount*INT32_SIZE`（未减 `kIdxBuf_`、未加 `BLOCK_SIZE`），且 `availableUbSize - RESERVED_UB_SIZE - 245760` 在 `245760 + RESERVED_UB_SIZE > availableUbSize` 时 **uint64 回绕到 ~2^64**，再 `÷ oneTaskSize(=(256+192)*4=1792)` 得到垃圾值 `613566720`（`613566720 × 1792 ≈ 1.1×10^12`，与回绕量级一致）。
- 垃圾 `ubMaxTaskNum` 被写进 TilingData，kernel `InitBuffer` 据此为 `tmpFeaturesBuf_` / `tmpOutFeatureBuf_` 申请 `613566720 × 1792 B` 级内存 → 远超物理 UB → device error type 3 / 507015。

判别要点：**tiling 打印的 `ubMaxTaskNum` 若是天文数字（远大于 UB 字节数 ÷ oneTaskSize 的合理上界），即为 UB 预算无符号下溢的标志**；接下来核对 tiling 侧预算是否逐项镜像了 kernel `InitBuffer` 的每个 buffer（含 `kIdxBuf_` 这类 mask buffer 与 `+BLOCK_SIZE` padding），以及 uint64 减法是否先做下溢守卫。

## 验证信号

本 trace 内可核的实测信号（均为失败侧/探针侧，**修复后未复跑**）：

- 507015 device error type 3 抛出于 `copy_between_host_and_device_opapi` 拷回 `conv3d_out.features` 时（来源平台:未定位；证据等级:tool_observed）。
- tiling 打印输出 `ubMaxTaskNum=613566720 @ loopPointCount=20480, inChannels=256, outChannels=192, featureWspOffset=9175040`（`featureWspOffset = 20480 × (256+192) = 9175040` 自洽，印证 `loopPointCount` 与 channel 字段无误，异常仅在 `ubMaxTaskNum`）（来源平台:未定位；证据等级:tool_observed）。
- kernel 侧 `InitBuffer` 4 个 buffer 的字节算式取自 `op_kernel/cloud_sparse_inverse_conv3d_with_matmul.h` L141-156 逐字读事件（来源平台:未定位；证据等级:artifact_verified）。
- nn 版 arch35 tiling 在 `availableUbSize` 中显式减 `kIdxBufSize` 与 `+32` 并有 `ubMaxTaskNum==0` 守卫，取自 `cloud_sparse_inverse_conv3d_grad_tiling_arch35.cpp` L34-48 逐字读事件（来源平台:未定位；证据等级:artifact_verified）。

无修后通过信号；"修复应消除 507015"属未验证推断，未作为实测信号写入。

## 系统信息

- 工程路径：`/mnt/workspace/gitCode/cann/cloud-ops-turbo/driving/`。
- 相关文件（与 trace 内读写一一对应）：
  - `driving/sparse_conv_common/op_host/sparse_conv_tiling_common.h`（缺陷点：`CalcUbMaxTaskNum`；常量 `INDICES_BUFFER=3`、`BLOCK_SIZE=32`、`INT8_PER_LOOP=256`、`BITS_PER_BYTE=8`、`DCACHE_UB_SIZE=40*1024`、`MINI_TASK_BLOCK=16`）。
  - `driving/cloud_sparse_inverse_conv3d_with_matmul/op_kernel/cloud_sparse_inverse_conv3d_with_matmul.h`（kernel 侧 `InitBuffer` 探针源，L135-156）。
  - `driving/cloud_sparse_inverse_conv3d_with_matmul/op_host/cloud_sparse_inverse_conv3d_with_matmul_tiling.cpp`（调用 `CalcUbMaxTaskNum` / `CalcCoreAlloc` 的 tiling host）。
  - `driving/cloud_sparse_inverse_conv3d_grad/op_host/cloud_sparse_inverse_conv3d_grad_tiling.cpp`（grad 侧同构调用）。
  - `nn/cloud_sparse_inverse_conv3d_grad/op_host/arch35/cloud_sparse_inverse_conv3d_grad_tiling_arch35.cpp`（正确预算参照，L34-48）。
  - `tests/st/test_cloud_sparse_inverse_conv3d_with_matmul.py`（ST 用例 `test_model_case0` / `getout`）。
- `RESERVED_UB_SIZE` 的确切数值未在 trace 的 `file_read` 事件中出现（constants 段只读到上述 6 个常量，未见 `RESERVED_UB_SIZE` 定义行）；不影响根因判别（无论其值多少，漏算 `kIdxBuf_` + 无下溢守卫即足以产生垃圾值），但量化精确下溢量级时需回源补读。

## 设计决策

- **为何 tiling 侧预算必须逐项镜像 kernel `InitBuffer`**：`ubMaxTaskNum` 既喂给 kernel 的 `tmpFeaturesBuf_` / `tmpOutFeatureBuf_`（按行 × channel × 4 字节申请），又受 UB 总量硬约束。kernel 侧还另申请 `indexBuf_` / `kIdxBuf_` 两块固定 UB。tiling 若只减"索引区 + 预留区"而漏掉 `kIdxBuf_` 与 `BLOCK_SIZE`，预算偏大 → `ubMaxTaskNum` 偏大 → kernel `InitBuffer` 溢出。nn 版 arch35 把 `kIdxBufSize` 与 `+32` 都减进去是正确范式。
- **为何加 `return 0` 守卫而不是直接修常量**：`availableUbSize - indicesSize - kIdxBufSize` 在大 `loopPointCount`（如 20480）下固定开销已超可用 UB，uint64 无符号减法会回绕到 ~2^64 → `÷oneTaskSize` 得到天文数字。守卫把这种 case 显式失败（`return 0` → 调用方 `return ge::GRAPH_FAILED`），而不是让一个垃圾 `ubMaxTaskNum` 流到 kernel。这比"调大 RESERVED_UB_SIZE"更稳：后者只是把下溢阈值挪动，仍可能在更大 `loopPointCount` 下再次下溢。
- **被放弃的路线**：用户先提"修改 loopPoint 来强行单核需缩小用例，问题不好复现"，故改走"固定 `loopPointCount=2048` + 强制 `usedCubeNum=1`/`usedVectorNum=2`"调试路线（见 `## 失效边界`），以保留原大用例。该路线为调试手段，非正解；正解仍是修 `CalcUbMaxTaskNum`。

## 性能影响

无性能故事（本卡为 build/runtime 诊断，无 profiling 数据；相对性能代价不适用）。

## 预防

- 写 tiling 侧 UB 预算函数时，强制逐项列出 kernel 侧 `pipe_->InitBuffer(...)` 的每个 buffer 字节算式，并保持 tiling 减项与 kernel 申请项一一对应（含 `+BLOCK_SIZE` 这类 padding 与 mask buffer）。
- 任何 `availableUbSize - <fixed overhead>` 的 uint64 减法必须先 `if (availableUbSize < fixedOverhead) return 0;`，调用方对 `0` 返回失败；禁止裸减法。
- 对 `ubMaxTaskNum` 加 sanity 上界检查（例如 `ubMaxTaskNum > availableUbSize / oneTaskSize` 必为下溢），在 tiling 打印里把 `availableUbSize` 一起打出，便于一眼发现天文数字。

## 坏实践

- 用 `RESERVED_UB_SIZE` 这类不透明"预留常量"代替逐项 kernel `InitBuffer` 算式：常量与实际 buffer 脱节后无人能从 tiling 代码看出预算是否覆盖 `kIdxBuf_`。
- `uint64` 减法无下溢守卫：把"UB 不够"伪装成"UB 巨大"，垃圾 `ubMaxTaskNum` 流到 kernel 才以 507015 爆出，掩盖了 tiling 侧的真实缺陷位置。

## 误诊

- agent 先以为"换 `fprintf(stderr)` + `fflush(stderr)` 打印 tiling 字段"是排查主路径；实际打印本身只是把 `ubMaxTaskNum=613566720` 这个垃圾值暴露出来，真正根因仍在预算函数而非打印通路（打印相关调试属 C stdio 缓冲行为，不单独成卡）。
- 用户一度想靠"缩小用例 + 改 loopPoint 强行单核"复现；该路线被否决（会改变问题形态），改用"固定 loopPoint + 强制单 block"保留原用例——但该改动本身不修复 507015，仅辅助调试。

## 失效边界

- **未验证，推断可迁移**：同一 `sparse_conv_tiling_common.h` 驱动的 `cloud_sparse_inverse_conv3d_grad` 算子（`oneTaskSize = (2*inChannels+outChannels)*4`）受同一 `CalcUbMaxTaskNum` 缺陷影响，推断同样需要此修复，但 trace 内仅 with_matmul 触发并观测到 507015，grad 侧未跑出失败。
- **未验证，推断可迁移**：nn 版参照文件命名为 `*_tiling_arch35.cpp`，只足以表明目标实现属于 arch35/Ascend950 路径；但 trace 无 SoC 探测，不写入 `platforms`。其他 UB 容量不同的平台（a2/a3）是否在同一 `loopPointCount=20480` 下下溢，取决于其 `dynamicUbSize`，未验证。
- **调试路线（强制单 block）未验证**：用户要求并 apply 的 `loopPointCount=2048; coreAlloc.usedCubeNum=1; coreAlloc.usedVectorNum=VECTOR_CUBE_RATIO*1;` 改动（覆盖 `CalcLoopPointCount` / `CalcCoreAlloc`）在 trace 内 **apply 后即 session_end，无运行结果**，不作为本卡的修后验证信号，仅作调试路线记录；该改动还原后才能用于生产。强制单 block 路线本身不解决 `CalcUbMaxTaskNum` 漏算 `kIdxBuf_` 的根因。
