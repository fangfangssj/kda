---
type: "Runbook"
title: "Ascend950 DMA 32B 块最小传输粒度"
description: "Ascend950 GM↔UB DMA 引擎最小传输单元为 32 字节（一个 block，fp32=8 元素）；单 float 标量 DMA 静默失败或报 507015，按 8 元素对齐的 DataCopy 会越界读 GM 致 Segfault。"
tags: ["task.debugging", "topic.data_movement.memory_access", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-16T11:20:30Z"
platforms: ["950"]
---

# Ascend950 DMA 32B 块最小传输粒度

## 现象

在 Ascend950 上开发 ApplyCamePart2 Branch-0 fp32 Kernel 时，需要从 GM 把标量输入（`beta2`、`sum_grad_rc`、`sum_r`）与短数组（`r`、`c`、`sum_grad_r`、`sum_grad_c`）搬入 UB 参与计算。多次尝试用 `DataCopy`/`DataCopyPad` 直接搬运单 float 标量或非 8 倍长数组，均触发硬件错误或静默失败：

- `DataCopy(dst, src, 1)`（单 float）：不报错但**静默不传输**，UB 局部缓冲残留上一轮旧值，下游计算用旧值算出错误结果。
- `DataCopyPad(tmp, gm[offset], {blockLen=4}, {isPad=false})`（4 字节）：`aclrtSynchronizeStream` 返回 **ACL error 507015**，用例 `[B0-S0] FAILED (execution error)`。
- `DataCopyPad(..., {blockLen=32})` 对 4B 标量 tensor：同样 507015（越界读）。
- `DataCopyPad(..., {isPad=true, rightPadding=7})`：507015。
- `tiling` 传标量再 `GetData<float>()`：返回设备指针而非标量值，解引用致 Segfault。

507015 错误典型输出（运行 `tests/aclnn && ./run.sh`）：

```
ACL error 507015 at .../test_aclnn.cpp:211: aclrtSynchronizeStream(stream)
ACL error 507015 at .../test_aclnn.cpp:453: aclrtSynchronizeStream(stream)
[B0-S0] FAILED (execution error)
[B0-S1] FAILED (execution error)
[B0-S2] FAILED (execution error)
```

另一失效形态：`CopyInArray` 为满足 32B 对齐，把搬运元素数向上取整到 8 的倍数 `alignedCount = (count + NUM_PER_BLK - 1) / NUM_PER_BLK * NUM_PER_BLK`（`NUM_PER_BLK = 8`）。对 `M=65` 的 GM 缓冲（260 字节），`alignedCount = 72`（288 字节）越界读 28 字节，导致大 shape 用例 Segfault：

```
DEBUG exec_b0: start N=33 M=65
DEBUG exec_b0: allocating gradBytes=8580 rBytes=132 cBytes=260
DEBUG exec_b0: dev mem allocated
DEBUG exec_b0: stream created
...
/bin/bash: line 1: 2733051 Segmentation fault      timeout 60 ./test_aclnn > /tmp/test_output9.txt 2>&1
Exit: 139
```

出问题时的 `CopyInArray`（verbatim，修复前）：

```cpp
static const int32_t NUM_PER_BLK = 8; // 32B / 4B

__aicore__ inline void CopyInArray(LocalTensor<float>& dst, GlobalTensor<float>& src,
                                    int64_t count, int64_t offset = 0)
{
    int64_t alignedCount = (count + NUM_PER_BLK - 1) / NUM_PER_BLK * NUM_PER_BLK;
    DataCopy(dst, src[offset], alignedCount);
}
```

## 适用场景

- **工程形态**：Kernel 直调（ops-registry-invoke，Ascend C device kernel），Kernel 内需从 GM 读标量或非 8 倍长 fp32 数组。
- **API 路径**：`DataCopy` / `DataCopyPad`（GM→UB DMA）；`GlobalTensor::GetValue`（标量直读，非 DMA）；`GlobalTensor::SetValue`（标量直写）；`ONE_BLK_SIZE` / `NUM_PER_BLK`（块长常量）。
- **验证平台与版本**：Ascend950（CANN 1.0.0, TimeStamp 202602），fp32。trace 内仅在此平台实测。
- **适用粒度**：`platform_wide`（凡 Ascend950 Kernel 内对 GM 做 sub-block 或非对齐 DMA 的算子均命中）。
- **路线优先级**：`recommended_fastpath`——标量走 `GetValue`/`SetValue` 直读直写；数组走 `DataCopy`/`DataCopyPad` 但 host 侧 GM 分配按 32B（8 fp32）取整对齐、kernel 侧搬运 count 取 8 倍数；尾块用 `DataCopyPad` 的 `isPad`/`rightPadding` 显式补齐。
- **不适用或未验证场景**：a2/a3 平台未验证（推断同 32B 块模型，但未实测，采用前需先在目标平台用单 float 标量 DataCopy 复现 507015 确认）；fp16/fp32 之外 dtype 的块长换算（fp16 块长 16 元素 = 32B）需按 `sizeof(T)*8` 等价推导，本卡仅 fp32 实测。

## 判别方法

**触发条件**：Kernel 编译通过，但 ST 运行出现两类现象之一——① 标量输入（`beta2`、`sum_grad_rc`、`sum_r` 等）参与计算的输出明显错误且 `aclrtSynchronizeStream` 返回 507015，或标量值像「未搬入」的旧值/零值；② 大 shape（任一维长度非 8 的倍数）用例 Segfault（Exit 139）而小 shape 用例正常。

**可执行探针**：
1. 对标量读取路径，分别试 `DataCopy(count=1)`、`DataCopyPad(blockLen=4)`、`DataCopyPad(blockLen=32, isPad=true, rightPadding=7)`、`GetValue(offset)`，逐个运行单 shape 用例并观察 `aclrtSynchronizeStream` 返回与输出值。预期：前三者 507015 或静默旧值，仅 `GetValue` 返回正确标量。
2. 对数组越界读，在 `exec_b0` 入口打印 `N`、`M`、各 GM 缓冲字节数（`gradBytes`/`rBytes`/`cBytes`），与 kernel 内 `alignedCount * sizeof(float)` 比对——若 `alignedCount * 4 > 实际分配字节数` 即为越界读，对应 shape 必 Segfault。
3. 用 PRINTF 同时兼作「强制加载新 kernel binary」开关（见相关卡 *NPU kernel binary 缓存*）——本平台存在 binary 缓存，源码改动可能不生效，PRINTF 改变 device hash 可验证新代码路径。

**原始观测**（Ascend950，fp32，ApplyCamePart2 Branch-0）：
- LOG 记录：`DataCopy(count=1)静默失败；DataCopyPad(blockLen=4)→507015；DataCopy(count=8)→507015(越界读)；DataCopyPad(isPad=true,rightPad=7)→507015；tiling传标量→GetData<float>()返回设备指针致segfault`。结论：标量值（`beta2`/`sum_grad_rc`/`sum_r`）无可行 DMA 读取路径。
- 第四轮重写改用「`GetValue` 读标量、padded `DataCopy` 读数组、`SetValue` 写数组」后，小 shape（B0-S0~S3）运行成功（不再 507015/Segfault），仅余精度与大 shape 挂起问题——即该 DMA 写法被验证为可行标量/数组搬运路径。
- `M=65` 用例 `alignedCount=72` 越界读 28 字节致 Segfault（Exit 139）。

**机制解释**：Ascend950 的 GM↔UB DMA 引擎以 32 字节（一个 block）为最小传输粒度，fp32 下即 8 个元素。小于一个 block 的传输请求不被支持：`DataCopy(count=1)` 静默不搬（UB 残留旧值），`DataCopyPad(blockLen<32)` 直接报 507015。把 count 向上取整到 8 倍数虽满足粒度，但若 GM 侧实际分配不足 32B 倍数（如 65 float=260B，取整 72 float=288B），DMA 会读出 GM 缓冲边界之外的 28 字节，轻则读到无关内存、重则触发 Segfault。标量值的正确读取路径是 `GlobalTensor::GetValue`（不走 DMA，走标量加载），写入对应 `SetValue`；`GetData<float>()` 在 tiling 上下文返回的是设备指针而非标量，解引用致 Segfault，不可用于读标量值。

## 解决配方

经 trace 验证可行的搬运写法（第四轮重写采此方案后小 shape 运行成功）：

1. **标量输入**（`beta2`、`sum_grad_rc`、`sum_r` 等单 float）：用 `GlobalTensor::GetValue(offset)` 直读，**不要**用 `DataCopy`/`DataCopyPad` 做单 float DMA。
2. **数组输入**：用 `DataCopy`，count 取 `alignedCount = (count + NUM_PER_BLK - 1) / NUM_PER_BLK * NUM_PER_BLK`（向上对齐到 8）；**同时 host 侧 GM 分配必须按 32B 取整**（`aclrtMalloc` 分配字节数取 `max(实际字节数, 32)` 且按 32 对齐），否则越界读致 Segfault。尾块可改用 `DataCopyPad` 配 `isPad=true`/`rightPadding` 显式补齐以避免越界。
3. **数组输出写回**：可用 `GlobalTensor::SetValue(offset, val)` 逐元素写（注意大 shape 逐元素写可能慢，trace 内 B0-S6 疑因此挂起）；或用 `DataCopy` 写回但同样需 GM 分配按 32B 对齐。

注意：本卡的「解决配方」给出经 LOG 与小 shape 运行验证的方向与写法要点；完整可执行的 after kernel 因 markdown_transcript 转写截断了 edit 工具 `newString` 参数，未在事件流逐字恢复，不在此内联完整 after 代码块。kernel.h 中保留的 `ReadScalarFloat` 辅助函数（verbatim，取自 apply_came_part2_kernel.h）记录了曾尝试的 DataCopyPad 通路，但其注释本身承认「DataCopyPad(blockLen=4) gives 507015」，故该辅助仅作历史尝试留档，**非推荐用法**——标量读取应以 `GetValue` 为准：

```cpp
// Helper: read a single float from GM.
// On Ascend950, DataCopy(count=1) fails silently and DataCopyPad(blockLen=4) gives 507015.
// Workaround: use DataCopyPad with the same pattern as Pre kernel (which works for N=1).
static __aicore__ inline float ReadScalarFloat(
    AscendC::GlobalTensor<float>& gm, int64_t offset,
    AscendC::TQue<AscendC::TPosition::VECCALC, 1>& que)
{
    AscendC::LocalTensor<float> tmp = que.AllocTensor<float>();
    AscendC::DataCopyPad(tmp, gm[offset],
        MakeCopyParams(sizeof(float)), MakePadParams());
    que.EnQue(tmp);
    tmp = que.DeQue<float>();
    float val = tmp.GetValue(0);
    que.FreeTensor(tmp);
    return val;
}
```

## 验证信号

本卡验证信号来自 ApplyCamePart2 Branch-0 fp32 Kernel 在 Ascend950 上的多轮 ST 运行与 LOG 记录（同一 trace）。

- `DataCopyPad(blockLen=4/32, isPad=false/true)` 对单 float 标量 tensor 均返回 ACL error 507015，用例 FAILED (execution error)（来源平台：Ascend950；证据等级: tool_observed）。
- `DataCopy(count=1)` 单 float 静默不传输，UB 残留旧值致下游计算错误（来源平台：Ascend950；证据等级: tool_observed）。
- `CopyInArray` 对 M=65 取 alignedCount=72 越界读 260B GM 缓冲致 B0-S6 Segfault Exit 139（来源平台：Ascend950；证据等级: tool_observed）。
- 改用「GetValue 读标量 + padded DataCopy 读数组 + SetValue 写数组」后，小 shape B0-S0~S3 运行成功（不再 507015/Segfault）（来源平台：Ascend950；证据等级: tool_observed）。
- kernel.h 源码注释明文记录「On Ascend950, DataCopy(count=1) fails silently and DataCopyPad(blockLen=4) gives 507015」（来源平台：Ascend950；证据等级: artifact_verified）。
## 失效边界

- 本结论仅在 Ascend950（CANN 1.0.0, 202602）fp32 实测。
- a2/a3 平台：未验证，推断可迁移（AscendC 块模型跨平台同源，块长按 `sizeof(T)*8` 换算），但未实测，采用前需先在目标平台用单 float 标量 DataCopy 复现 507015 确认。
- fp16（块长 16 元素 = 32B）、bf16、int64 等其他 dtype 的具体块长换算未在本开发会话 实测，需按 `32 / sizeof(T)` 等价推导后在目标 dtype 单测验证。
- 大 shape（B0-S6 [33,65]）的最终挂起机制未完全定位——SetValue 逐元素写慢 vs. PipeBarrier 死锁 vs. 24 核 ProcessU 协同，trace 内未给出确定性结论。

## 坏实践

- 用 `DataCopy(count=1)` 搬单 float 标量并假设它会传输——它静默不搬，UB 用旧值，是最隐蔽的陷阱。
- 把 `alignedCount` 向上取整到 8 倍数，却不相应把 host 侧 `aclrtMalloc` 的字节数取整到 32B 倍数——越界读致 Segfault。
- 用 `GetData<float>()` 在 tiling 上下文读标量值——返回设备指针，解引用 Segfault。

## 误诊

- 初判曾怀疑 workspace 布局或写入越界方向错误，逐一排除后定位到 DMA 32B 最小粒度（证据等级: agent_reported；仅记录调试路径）。
- 曾尝试 `aclrtMemcpy` 在 tiling 编译上下文读标量——build 失败（aclrtMemcpy 不可用于 device kernel 内），非可行方案。

## 预防

- 设计 Kernel 时，凡单标量输入默认用 `GetValue` 读、`SetValue` 写，不走 DMA。
- 凡数组走 `DataCopy`，host 侧 GM 分配字节数一律按 `32B`（fp32=8 元素）取整对齐，kernel 侧 count 同步取 8 倍数；尾块用 `DataCopyPad(isPad=true, rightPadding)` 显式补齐。
- 评审 Kernel 实现时，对 `DataCopy(dst, src, count)` 中 `count < NUM_PER_BLK` 或 `alignedCount * sizeof(T) > GM 实际分配字节数` 的写法直接打回。

## 后续增量（2026-09-16，并行策展批：SW>1 单元素 gather 垃圾值形态、GetValue/SetValue 修复路线与 CANN 9.2.0 复现）

DepthwiseConv2d（Ascend950PR/arch35，CANN 9.2.0，与原卡 202602 版本不同代）迭代三联调补充第三种失效形态与修复路线：

**1. 新失效形态（垃圾值而非静默旧值/507015）**：SW>1（列向跨步）分支用单元素 `DataCopyPad(xLocal[...], xGm_[...], DataCopyExtParams{1, sizeof(half), 0,0,0}, ...)` 把跨步输入列 gather 成连续行时，stride>1 用例全部失败且实际值为量级混乱的**垃圾值**（L0_002 stride=2 `MERE=8.06e+00, MARE=2.49e+02`；L1_024 实际=44.875 级异常大值；L1_032 bf16 实际=1.29e20 级垃圾）——fp16（blockLen=2B）、bf16（2B）、fp32（4B）三 dtype 均命中。这与原卡的"静默不搬（UB 残留旧值）"与"507015 报错"并列为本约束的第三种表现。

**2. 修复路线（标量通路 + S→V 同步）**：逐元素 DMA 换成 `xGm_.GetValue(...)`（S 流水线标量读 GM）+ `xLocal.SetValue(...)`（S 流水线标量写 UB）——标量通路不走 MTE 块搬运，逐元素语义确定；换原语时必须同步补 S→V 屏障（SetValue 走 S 流水线，EnQue 只同步 MTE 侧，V 计算（如 Axpy）读 xLocal 需显式 `PipeBarrier<PIPE_ALL>()` 后再 `EnQue`）：

```cpp
    for (int64_t ow = owStart; ow < owEnd; ow++) {
        int64_t iw = ow * SW_ + colOffset;
        half xVal = xGm_.GetValue(xOffset + inRow * W_ + iw);
        xLocal.SetValue(row * halfStride_ + ow, xVal);
    }
}
// S→V 同步：SW>1 路径用 SetValue(S) 写 xLocal，Axpy(V) 读 xLocal
// EnQue/DeQue 仅同步 MTE3→V，S→V 需显式 PipeBarrier
PipeBarrier<PIPE_ALL>();
xQueue_.EnQue(xLocal);
```

**3. 边界补充**：『blockLen<32B 行为未定义、DMA 最小传输单元 32B』仍是修复时的机制推断（未直接观测引擎行为），A/B 证据（换原语即翻转）只证明"该原语在此粒度不可用"；32B 对齐的块状 gather（把跨步列凑成 ≥32B 段搬）能否恢复 DMA 路径未验证。换原语后忘补 S→V 同步是本路线的伴生坑（坏实践）。

**4. 验证信号（来源：该算子迭代三联调，Ascend950PR 上板 ST；WAW 修复已就位后单独验证本修复）**：

- 修复前：stride>1 用例全失败——`L0_002 [FAIL] MERE=8.06e+00, MARE=2.49e+02`、L1_005 `MERE=7.55e+00, MARE=1.45e+03`、L1_024 实际=44.875、L1_032 实际=1.29e20 级垃圾（来源平台：950；证据等级：tool_observed）。
- 修复后（仅换 GetValue/SetValue + S→V 屏障）：全部 stride 用例转 PASS——`L0_002`、`L1_024 [PASS] MERE=2.38e-07, MARE=6.88e-04`、`L1_025 (stride 1,2) PASS`、`L1_026 (stride 3,3) PASS`、`L1_027 (stride 4,4) PASS`、`L1_032 stride=2+dilation=2 bf16 PASS`、`L1_033 多batch+stride2 fp32 PASS`；全量 53 用例 49 通过（92.5%），仅剩 4 条与 strided 无关的 L2 参数校验用例（来源平台：950；证据等级：tool_observed）。
- 归因隔离：同一轮 build 中 dilation 用例（SW=1 分支，未动）保持 PASS，仅 SW>1 用例翻转——改动唯一对应（来源平台：950；证据等级：tool_observed）。

**5. 坏实践补充**：换成 `SetValue` 后忘了 S→V 同步——EnQue/DeQue 只覆盖 MTE 侧，S 流水线写入对后续 V 流水线读取同样存在可见性时序。