---
type: "Runbook"
title: "多块 DataCopyPad dst 基址非 32B 对齐崩溃"
description: "多块 DataCopyPad(DataCopyExtParams) 把跨 slice 的 x0/x1 分段搬进 UB 时，若 head 块拷贝写入非 32B 整数倍字节长度，后续 middle/tail 块的 dst 基址随拷贝累积到非 32B 对齐地址，触发 MTE 地址未对齐 → vector core exception (EE9999 / 507035 rtDeviceSynchronizeWithTimeout)。判别方法：多块搬运上板即 EE9999，且 middle dst 基址字节偏移 firstLen*sizeof(T) 对 32 取余非零即命中；修复方向是按 alignUpS 步长重排 UB 为补齐布局，让每块 dst 基址 32B 对齐。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-18T16:53:02Z"
updated_at: "2026-09-17T01:47:15Z"
platforms: ["a2", "950"]
---

# 多块 DataCopyPad dst 基址非 32B 对齐崩溃

## 现象

在 Ascend 910B3（dav-2201，CANN 9.2.0，40 AIV，device 3）上对 `direct_launch_example/csrc/ops/swi_glu` 算子做性能优化。基线 kernel 单缓冲、每 tile 用 `toSliceEnd = slice - r` 截断强制落在单个 slice 内，搬运用**单块** `DataCopyPad`。优化版引入 slice 聚合切分：删除 `toSliceEnd` 截断，让一个 tile 跨多个 slice 边界，x0/x1 改用**多块** `DataCopyPad`（`DataCopyExtParams`，blockCount>1）一次搬多个 slice 的分段。构建通过，上板第一个 case（fp16，`[1024,2048]`，dim=-1，slice=1024）即崩：

```
EE9999[PID: 132750] 2026-08-06-18:52:10.227.360 (EE9999):  rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:69]
TraceBack (most recent call last):
 (function npuSynchronizeUsedDevices)
[Error]: The vector core execution is abnormal.
        Rectify the fault based on the error information in the ascend log.
EE9999: Inner Error!
[ERROR] 2026-08-06-18:52:10 (PID:132750, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

fp32 case 15（`[512,4096]`，dim=-1，slice=2048）同样崩。禁用 SwiGLU 高级 API（adv_api）后仍崩，排除 adv_api 内部因素。

## 适用场景

- **工程形态**：kernel 直调（`direct_launch_example`，`register_direct_launch_op`），bisheng `-xasc --npu-arch=dav-2201` 编译。
- **API 路径**：`AscendC::DataCopyPad(dst LocalTensor, src GlobalTensor, DataCopyExtParams, DataCopyPadExtParams)`，多块搬运（`blockCount>1`，`DataCopyExtParams.blockLen`/`srcStride`/`dstStride` 均以字节为单位）。
- **验证平台与 CANN 版本**：实测 Ascend 910B3 / dav-2201 / CANN 9.2.0（注册平台 token `a2`），`platforms: [a2, 950]`（a2 为主形态实测；950 见复现增量逐元素变体实测）。
- **适用粒度**：`operator_family`——任何用多块 DataCopyPad 把 GM 中分段/跨 stride 的数据搬进 UB 的 elementwise/激活算子。
- **路线优先级**：`recommended_fastpath`（修复后可用多块搬运做跨 slice 聚合，性能优于逐 slice 单块拷贝）。
- **不适用/未验证场景**：单块 DataCopyPad（`blockCount=1`）不受此 dst 基址对齐约束（单块 dst 基址即 buffer 基址，天然对齐）；非 dav-2201 架构未验证。

## 根因

多块 `DataCopyPad` 的**每个数据块的 dst 起始地址必须 32B 对齐**（`DataCopyPadGm2UBImpl` 断言 `TransUBAddr % 32 == 0`）。优化版把一个 tile 的 x0 分成 head(firstLen) + middle（逐 slice，`blockLen=slice*es`）+ tail 三段，head 拷贝写入 `dst[0..firstLen)` 后，middle 的 dst 基址取 `dst[firstLen]`。当 `firstLen * sizeof(T)` 不是 32B 整数倍时，middle 块的 dst 基址非 32B 对齐 → MTE 指令访问地址未对齐 → vector core exception。

具体算式（fp16 case1，slice=1024，tileElems=6144，firstLen=819）：
- `firstLen = 819`（输出偏移 r=819 距 slice 尾的剩余）
- middle dst 基址字节偏移 = `819 * sizeof(half) = 819 * 2 = 1638` 字节
- `1638 % 32 = 1638 - 51*32 = 1638 - 1632 = 6 ≠ 0` → 非 32B 对齐 → 崩溃。

fp32 case15（slice=2048，firstLen=819）：`819 * 4 = 3276`，`3276 % 32 = 3276 - 102*32 = 3276 - 3264 = 12 ≠ 0` → 同样崩溃。

机制链：多块 DataCopyPad 跨 slice 聚合 → head 块拷贝后 dst 偏移随非对齐长度累积 → middle/tail 块 dst 基址字节偏移 = `firstLen*sizeof(T)` 非 32B 倍数 → MTE 访问未对齐地址 → vector core exception。这不是"算子逻辑错"或"算力不够"，而是"多块 dst 基址必须 32B 对齐"的硬件级约束被 head 累积偏移破坏。

## 判别方法

- **触发（tool_observed）**：多块 `DataCopyPad`（`blockCount>1`）搬运上板即 `EE9999 rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception`，首个 case 即崩、fp16/fp32 多 dtype 同发；禁用 adv_api 后仍崩（排除 SwiGLU 高级 API 内部因素）。
- **探针（executable check）**：对崩溃路径的多块 CopyIn，核算 head 块写入长度 `firstLen` 与 middle 块 dst 基址字节偏移，做 32 取余判定：
  ```bash
  python3 - <<'EOF'
  es = 2  # fp16 sizeof(T); fp32 改 4
  firstLen = 819  # 输出偏移 r 距 slice 尾的剩余（tiling 打印或断点取值）
  off = firstLen * es
  print("middle dst byte offset =", off, " off % 32 =", off % 32,
        "=>" "ALIGNED" if off % 32 == 0 else "UNALIGNED -> crash")
  EOF
  ```
  若 `firstLen*sizeof(T) % 32 != 0` 即命中本判别。判别时须同时核验该路径确为多块（`DataCopyExtParams.blockCount>1`）且 dst 偏移随拷贝累积（非每块独立对齐布局），单块路径不触发。
- **观测（tool_observed，本开发会话 实测 a2）**：fp16 case1 `firstLen=819` → middle dst 基址字节偏移 `1638`，`1638 % 32 = 6 ≠ 0`，上板即 `EE9999 vector core exception`；fp32 case15 `firstLen=819` → `3276 % 32 = 12 ≠ 0`，同样 `EE9999` 崩溃。禁用 adv_api（手写 SiLU 链）后仍崩，锁定根因在多块 dst 基址对齐而非高级 API。
- **解释（interpretation）**：多块 `DataCopyPad` 的 MTE 硬件约束要求每个数据块 dst 起始地址 32B 对齐（`TransUBAddr % 32 == 0`）。当 CopyIn 拆 head/middle/tail 且 dst 偏移随非对齐长度累积时，head 写入 `firstLen` 个元素后 middle 基址落在 `firstLen*sizeof(T)` 字节处；该偏移非 32B 倍数即触发 MTE 地址未对齐异常。判别要点：多块搬运上板即 EE9999、且 `firstLen*sizeof(T) % 32 != 0` → 即为 dst 基址非 32B 对齐，优化方向是按 `alignUpS = AlignUp(slice*sizeof(T),32)/sizeof(T)` 步长重排 UB 为补齐布局（每块 dst 基址天然 32B 对齐），而非改 blockLen 单位或换 API。

## 验证信号

- 修复前：fp16 case1 / fp32 case15 多块 DataCopyPad 上板即 `EE9999 rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception`、`ERR99999 UNKNOWN application exception`（来源平台：a2；证据等级：tool_observed）。
- middle dst 基址字节偏移核算：fp16 case1 `819*2=1638`，`1638 % 32 = 6 ≠ 0`；fp32 case15 `819*4=3276`，`3276 % 32 = 12 ≠ 0`，与崩溃一一对齐（来源平台：a2；证据等级：tool_observed）。
- 禁用 SwiGLU 高级 API（adv_api）后仍崩，排除 adv_api 内部因素，锁定根因为多块 dst 基址对齐（来源平台：a2；证据等级：tool_observed）。

## 失效边界

- 本判别与根因在 a2（910B3 / dav-2201，CANN 9.2.0）实测；950 已由复现增量（逐元素变体）实测复现；a3 平台未实测，**未验证，推断可迁移**（DataCopyPad 的 32B 对齐约束是 MTE 硬件级要求，跨架构一致，但需重测确认）。
- 单块 DataCopyPad（`blockCount=1`）不受此 dst 基址约束（buffer 基址天然对齐），无需本判别。
- 修复版（alignUpS 补齐布局）的 verbatim before/after 完整 kernel 在归一化事件流中为占位（adapter 折叠为 `_see below_`），未从 trace 逐字恢复；判别与根因由 EE9999 实测信号 + dst 基址字节偏移核算 + 禁用 adv_api 对照三组 tool_observed 支撑，修复后 ALL PASS 信号。

## 误诊

- 初判怀疑 SwiGLU 高级 API（adv_api）内部 bug 导致崩——禁用 adv_api 后仍崩，排除该假设。
- 怀疑 UB 溢出（172032 vs 196608 字节）——核算后 UB 预算充足，排除。
- 怀疑 strided DataCopyPad 的 blockLen/stride 单位错——单位正确（字节），崩溃源于 dst 基址对齐而非单位。

## 坏实践

- 用多块 DataCopyPad 把 dst 偏移随拷贝累积到非 32B 倍数位置，却不显式对齐每块 dst 基址。
- 误以为 DataCopyPad"能处理非对齐长度"就等于"dst 基址可任意"——前者指 `blockLen` 任意，后者指每块 dst 起始必须 32B 对齐。

## 预防

- 用多块 DataCopyPad 搬运跨 stride 数据时，**每个块的 dst 基址字节偏移必须 = 块索引 × ceil(blockLen/32)*32**，不要让 dst 偏移随非对齐长度累积。
- tiling 阶段就算 `alignUpS = AlignUp(slice*es, 32)/es`，让 tile 步长与 dst 步长都按 alignUpS 取，从源头保证每块 dst 基址 32B 对齐。
## 复现增量：逐元素 GM→UB 的 dst 非对齐变体 + isPad 不放宽对齐的实测证明

另一独立轨迹并入 gather 算子的逐元素变体（arch35/DAV_3510，CANN 9.1.0），并给出两条关键澄清：

**1. isPad=true 不解决地址对齐（实测证明）**：把逐元素搬运的 `DataCopyPadParams` 从 `{false, 0, 0, 0}` 改为 `{true, 0, 0, 0}` 重编重装后**同样报** errcode 80（mte error info 仅从 0x...202cb 变为 0x...202d2）。头文件锚点（dav_3510 `kernel_operator_data_copy_impl.h` 内注释）：

```cpp
// DataCopy is in unit of 32 bytes, DataCopyPad is in unit of 1byte
uint32_t unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN;
```

即 DataCopyPad 的 "Pad" 只放宽**长度/填充**语义（blockLen 可按字节计），**不放宽地址对齐**要求；`isPad=true` 只是启用填充机制，同样不管地址对齐。

**2. 逐元素随机访存（gather 形态）的修复路线——32B 对齐 tempBuf 双段搬运**：多块布局重排（本卡原修复方向）对"每个元素的目的偏移由随机索引决定"的 gather 形态不适用；改经一个固定 32 字节对齐中转缓冲做 MTE2+MTE3 双段搬运——UB 侧只碰 `tempLocal`（缓冲头部，天然 32B 对齐），GM 侧允许元素对齐（`isPad=true` 处理 GM 源非 32B 对齐的填充）：

修复前（逐元素 DataCopyPad，UB 目的地址不对齐）：

```cpp
    // 逐元素 gather：读取 index 值 → 计算源地址 → DataCopyPad 单元素搬运
    for (int64_t j = 0; j < currentNum; j++) {
        int64_t outLinearIdx = globalStartIdx + j;
        int64_t indexVal = ReadIndexValueByDtype(indexLocal, j, indexDtypeSize);
        int64_t srcOffset = ComputeSrcOffset(outLinearIdx, indexVal);

        DataCopyParams elemCopyParams;
        elemCopyParams.blockCount = 1;
        elemCopyParams.blockLen = static_cast<uint32_t>(sizeof(D_T_X));
        elemCopyParams.srcStride = 0;
        elemCopyParams.dstStride = 0;

        DataCopyPad(yLocal[j], xGm[srcOffset], elemCopyParams, {true, 0, 0, 0});  // dst = yLocal + j*sizeof(T)
    }
```

修复后（经 32B 对齐 tempLocal 的 MTE2+MTE3 双段搬运，取自终版 ProcessBatch）：

```cpp
    // 逐元素 gather: MTE2 (GM→UB) + MTE3 (UB→GM) 双段搬运
    for (int64_t j = 0; j < currentNum; j++) {
        int64_t outLinearIdx = globalStartIdx + j;
        int64_t indexVal = ReadIndexValueByDtype(indexLocal, j);
        int64_t srcOffset = ComputeSrcOffset(outLinearIdx, indexVal);
        int64_t outOffset = globalStartIdx + j;

        // pipe_barrier(PIPE_ALL): 确保 S pipe 计算的地址对 MTE2/MTE3 可见
        pipe_barrier(PIPE_ALL);

        // MTE2: xGm[srcOffset] → tempLocal (32B 对齐 UB)
        DataCopyParams elemParams;
        elemParams.blockCount = 1;
        elemParams.blockLen = static_cast<uint32_t>(sizeof(D_T_X));
        elemParams.srcStride = 0;
        elemParams.dstStride = 0;
        DataCopyPad(tempLocal, xGm[srcOffset], elemParams, {true, 0, 0, 0});

        // MTE2 → MTE3 同步
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

        // MTE3: tempLocal → yGm[outOffset] (32B 对齐 UB → GM)
        DataCopyParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(sizeof(D_T_X));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm[outOffset], tempLocal, outParams);
    }
```

配套 UB 分配（`TBuf` 天然 32B 对齐；单元素中转缓冲固定 32 字节——一个元素最大 int64=8B，32B 同时满足对齐与容纳，不按 dtype 放大）：

```cpp
    int64_t indexUbSize = (ubFactor_ * indexDtypeSize + 31) & ~31;
    int64_t tempBufSize = 32;  // 32 字节，足够放一个元素
    pipe.InitBuffer(indexBuf, indexUbSize);
    pipe.InitBuffer(tempBuf, tempBufSize);
```

**独立 trace 新增验证信号**（均为 tool_observed）：

- 修复前：ST 用例 `[FAIL] 流同步失败: 507035`；plog `errcode:(80) errorStr: The MTE non-atomic instruction address is not aligned. subErrType: 0x4`（来源平台：950；证据等级：tool_observed）。
- `isPad=true` 尝试后：同一 errcode 80 报错复发（mte error info 0x...202d2），证明与填充语义无关（来源平台：950；证据等级：tool_observed）。
- 改为 32B 对齐 tempBuf 双段搬运后：507035/errcode 80 崩溃不再出现（后续失败均为数值错误，属另一根因；该数值问题闭环后最终 10/10 bitwise=YES）；该架构保留到终版 kernel（多 dtype × 双缓冲），ST Real 25/25 全部 bitwise 通过（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界与坏实践**：GM 侧仍需元素对齐（dtype 宽度），完全任意的字节级 GM 偏移未验证；双段搬运每元素 2 次 DMA + 同步，性能代价显著（正确性优先方案，批量 gather/向量 Gather 指令/按 32B 块对齐分组搬运为优化方向，本开发会话 未实施）。坏实践补充：认为 "DataCopyPad 支持 Pad 所以能搬不对齐地址"——Pad 只管长度；指望 `isPad=true` 修 errcode 80。
