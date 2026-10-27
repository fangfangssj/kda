---
type: "Runbook"
title: FullloadR softmax kernel AR/ARA 实现配方
description: "Ascend950/CANN 9.1.0 上归约轴整行可载入 UB 的 softmax kernel 完整实现：合轴后 AR/ARA 两类块均 GM 连续，单段 DataCopyPad + -inf 尾填充；Cast 升 fp32 后 ReduceMax/ReduceSum（Pattern::AR/RA）+ __simd_vf__ 元素链经 asc_vf_call 调用；AR 标量 GetValue 回读广播、ARA 按行向量对位；多核确定性 cyclic-by-block。"
tags: [task.implementation, topic.compute.elementwise.activation, topic.compute.reduction, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# FullloadR softmax kernel AR/ARA 实现配方

## 现象

在 Ascend950（CANN 9.1.0，regbase/ascendc 范式）上为一个 softmax 基准算子实现 Branch-0（FullloadR）kernel：进入条件是"非空 && rTotal ≤ rFullloadThresh"，即归约轴整行可一次性载入 UB。此前 kernel 是桩（只做 PipeBarrier，不写 y），aclnn ST 22 例全红，失败形态：正常用例 `rtol/atol fail`（如 `[S1_fp16] FAIL (rtol/atol fail at 128)`）、含 NaN/±inf 用例 `isnan mismatch`（桩输出 0 而非 NaN）、全相同值用例期望 1/n 得 0、`invariant sum_to_one: sum=0.000000`。要求同时满足：

- 数值稳定链：m = max_j x_j → e = exp(x−m) → s = Σe（fp32 累加）→ y = e/s；
- fp16/bf16 输入经 Cast 升 fp32 计算，出口 cast 回原 dtype；
- NaN 逐点传播、全 NaN 切片整切片 NaN、−inf+有限值时 −inf 位置精确 0、全相同值 1/n；
- 非对齐尾块（如 R=97，fp16 下行尾不落 32B 边界）正确；
- 范式硬门：元素级计算链必须以 `__simd_vf__` 函数经 `asc_vf_call` 调用。

输入经 host 侧合轴预处理后只有两类 pattern：AR（axisNum=2，[A, R]，每块 = 1 个归约切片 = R 个连续元素，沿内层 R 归约）与 ARA（axisNum=3，[A0, R, A1]，每块 = 1 个 a0 的 [R, A1] 连续块，沿外层 R 归约）。kernel 从 TilingData 读 axisNum/axisShape/axisStride/rTotal/usedCoreNum。

## 适用场景

- 工程形态：ACLNN 自定义算子 kernel（AscendC，arch35/regbase 范式，`asc_vf_call` + `__simd_vf__` 元素链），归约轴可整行载入 UB 的"保维归约"类算子（softmax 族与同构 max-sub-exp-sum-div）。
- API 路径：`DataCopyPad`（DataCopyExtParams + DataCopyPadExtParams 的 isPad/leftPadding/rightPadding/paddingValue）、`AscendC::ReduceMax/ReduceSum`（`Pattern::Reduce::AR`/`RA`，srcInnerPad，无 sharedTmpBuffer 的自动重载）、`Reg::LoadAlign/StoreAlign/Cast/Sub/Exp/Div/Duplicate/UpdateMask`、`asc_vf_call<Fn>`、`LocalTensor::GetValue`、`TQue`（VECIN DB=2 / VECOUT）、`TBuf`（VECCALC）。
- 验证过的平台与 CANN 版本：Ascend950 系列 + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——机制绑定"AR/ARA 合轴 + 单趟全载 + -inf 尾填充"的数据流形态，softmax 族与同构保维归约算子可迁移。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：rTotal 超过 UB 全载阈值的 ChunkedR 多 chunk 路径（本卡未实现，入口对 rChunkCnt != 1 早退）；空 tensor 分支（usedCoreNum == 0 早退）；需要跨块 gather 的非连续切片；性能调优（本卡为正确性首版，无 profiling 信号）。

## 根因/解决配方

核心机制链（每步都有为什么）：

1. **两类块在 GM 上均连续，搬运用单段 DataCopyPad（blockCount=1）**。AR 块 b 位于 GM 偏移 b·R；ARA 块 b（一个 a0 的全部 A1 个切片）位于 b·R·A1，块内 R·A1 元素连续。不要用 MTE2 的 blockCount>1 + dstStride=0 做"strided gather 进连续 UB"——UB 侧 stride=0 的语义是"下一块紧跟上一块 **32B 对齐后**的位置"，blockLen=2（fp16）时每块在 UB 占 CeilAlign(2,32)=32B，8 块散在 0/32/64/… 字节处，得到的是散射布局，直接破坏后续连续 reduce。按整 a0 块处理则天然连续，一次拷完。
2. **非对齐尾用 DataCopyPad rightPadding 以 −inf 填充，一个 padding 值同时喂两个归约**：ReduceMax 求行最大时 −inf padding 永远不会成为 max（被有效值压住）；exp(−inf − m) = 0 使 ReduceSum 的 padding 贡献为 0。若用 0 填充：全负输入行 max 会被 0 顶替、exp(0−m) ≠ 0 污染 sum——两个归约同时出错。
3. **AR 与 ARA 的归约 pattern 与广播方式不同**：AR 用 `Pattern::Reduce::AR` 对 [1, R_align] 沿内层 R 归约得 1 个标量（fp32），`GetValue(0)` 回读后在 VF 内 `Reg::Duplicate` 广播成寄存器标量；ARA 用 `Pattern::Reduce::RA` 对 [R, A1] 沿外层 R 归约得 A1 个值，按行（外层 r 循环）`LoadAlign` 对位向量减。Sub+Exp 与 Div+Cast 因此各写 AR/ARA 两个 `__simd_vf__` 变体（运行时无法在 Duplicate(scalar) 与 LoadAlign(vector) 之间切换，分函数最直接）。
4. **精度链全 fp32**：fp16/bf16 输入先 `Reg::Cast` 扩位到 fp32（`DIST_UNPACK_B16` 装载 + `CAST_NONE`），max/sum/exp/div 全 fp32，出口 `CAST_RINT` 缩位（`DIST_PACK_B32` 存储）；fp32 输入 `if constexpr` 直通无 cast。
5. **ReduceMax/ReduceSum 用无 sharedTmpBuffer 的自动重载**：`AscendC::ReduceMax<float, Pattern::Reduce::RA>(dst, src, srcShape, /*srcInnerPad=*/true)`，isReuseSource 取默认 false 保护源 tensor，scratch 由框架 PopStackBuffer 内部管理，无需手工排 3 个 tmpBuf slot。
6. **多核确定性分块**：`small = totalBlocks / usedCoreNum`，`bigCnt = totalBlocks % usedCoreNum`，前 bigCnt 核各处理 small+1 块，其余核 small 块——cyclic-by-block，无跨核依赖、无原子，输出确定。
7. **UB 对齐账**：AR 的 x 缓冲 padded 到 `CeilAlign(rTotal, 32B/elemBytes)` 个元素（容纳 −inf 尾填充）；ARA 的 x 缓冲按整块 `CeilAlign(R·A1, 32B/elemBytes)`；max/sum 缓冲按 `CeilAlign(A1, 8)` 个 fp32 且至少 32B；VF 单次循环 64 个 fp32（VL=256B）。

## 设计决策

- **AR 标量广播选 GetValue 而非寄存器级广播**：候选方案有 (a) `Duplicate(dstReg, srcReg, mask)` 寄存器间复制（语义是整寄存器复制而非 lane-0 广播，头文件中无匹配实现）、(b) `AscendC::Brcb` 把 1 元素 reduce 结果展开成 R 元素 tensor 再逐元素减（多一次显式广播搬运，且 Brcb 未在 regbase 白名单确认）、(c) `GetValue` 回读标量传给 VF、VF 内 `Reg::Duplicate`。最终选 (c)：正确性直接、API 标准；代价是 GetValue 的流水线 stall（未量化，见遗留问题）。ARA 无此问题（向量直接 LoadAlign）。
- **按整 a0 块处理而不是逐切片 gather**：ARA 若按"单个归约切片"处理需要 strided gather，而 MTE2 dstStride=0 的 32B 对齐散射语义使其不可直用；把 A1 个切片连成整块后 CopyIn/CopyOut 退化为单段连续拷贝，复杂度从"搬运特化"降为"reduce pattern 分支"。
- **VF 函数族 = CastIn + SubExp×2 + DivCast×2**：cast-in 单独一个 VF（含 padding 的全量转换），Sub+Exp 融合一个 VF（AR/ARA 两变体），Div+Cast 融合一个 VF（两变体）——融合减少 UB 往返，同时满足"`__simd_vf__`/`asc_vf_call` 真实代码"硬门（grep `asc_vf_call` 10 处、`__simd_vf__` 7 处）。
- **`TQue` DB=2 for x、DB=1 for y，`TBuf` for 中间量**：xCopyIn 与上一块的 VF 计算重叠；y 无重复利用不开 DB；fp32 中间量（xF32/e/max/sum）用 VECCALC TBuf 手工管理。
- 当前方案路线档位：recommended_fastpath（正确性首版，一次通过；性能未调）。

## 系统信息

- 硬件/软件：Ascend950 系列，CANN 9.1.0；kernel 以 DTYPE_X 编译期实例化 3 份二进制（fp16/fp32/bf16）。
- 验证框架：`tests/aclnn/test_aclnn.cpp`（自包含 ST：独立 CPU oracle 做 max→sub→exp→sum→div，不调 aclnn；22 例 = S1[32,1000]/S2[12,12,512,512]/S4[16,8]/B1 轴长 1/F3 dim0 非末轴/F4 负索引 dim−2/F5 非对齐尾 R=97 × fp16/fp32/bf16 组合 + E1-E6 特殊值：E1 含 NaN、E2 含 +inf、E3 全 NaN、E4 −inf+有限、E5 全相同值、E6 fp16 上溢边界；正常例 rtol/atol + 不变量兜底，E1-E3 逐点 isnan 判等，E4 −inf 位精确 0，E5 期望 1/n，E6 sum_to_one）。
- 相关文件：`op_kernel/bench_harness_softmax.h`（kernel 类 + 5 个 `__simd_vf__` 函数）、`op_kernel/bench_harness_softmax_kernel.cpp`（kernel 入口）、`op_kernel/bench_harness_softmax_tiling.h`、`op_host/*_tiling_host*`、`tests/aclnn/`。

## 完整代码

kernel 入口 `op_kernel/bench_harness_softmax_kernel.cpp`（完整，按运行时 TilingData 字段分发；dtype 走 DTYPE_X 编译期实例化）：

```cpp
#include "kernel_operator.h"
#include "bench_harness_softmax_tiling.h"
#include "bench_harness_softmax.h"

// Branch-0 (FullloadR) kernel 入口。
// 据运行时 TilingData 字段分发（dtype 走 DTYPE_X 编译期实例化）：
//   - usedCoreNum == 0   → EmptyTensor（早退，y 由框架预分配空 tensor）
//   - rChunkCnt == 1      → FullloadR（本实现）
//   - rChunkCnt > 1       → ChunkedR（未实现；此处早退）
extern "C" __global__ __aicore__ void bench_harness_softmax(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(BenchHarnessSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchHarnessSoftmaxTilingData, tilingData, tiling);

    if (tilingData.usedCoreNum == 0) {
        return; // EmptyTensor：早退不计算（returns_empty）
    }
    if (tilingData.rChunkCnt != 1) {
        return; // ChunkedR 未实现
    }

    NsBhs::BenchHarnessSoftmaxKernel<DTYPE_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
```

kernel 类实现 `op_kernel/bench_harness_softmax.h`（完整，verbatim 取自终版写入）：

```cpp
/*!
 * \file bench_harness_softmax.h
 * \brief BenchHarnessSoftmax Branch-0 (FullloadR) kernel 类实现。
 *
 * 数学（数值稳定版）：
 *   m  = max_j x_j
 *   e  = exp(x - m)
 *   s  = sum_j e_j   (fp32 累加)
 *   y  = e / s
 *
 * Branch-0 进入条件：非空 && rTotal ≤ rFullloadThresh，整行一次性载入 UB。
 * 单 chunk 单趟：CopyIn → (Cast→fp32) → ReduceMax → Sub → Exp → ReduceSum(fp32) → Div → Cast → CopyOut。
 *
 * 合轴 pattern 两类：
 *   AR  (axisNum=2, [A, R])：每块 = 1 个归约切片 = R 个连续元素，沿内层 R 归约（Pattern::AR）。
 *   ARA (axisNum=3, [A0, R, A1])：每块 = 1 个 a0 的 [R, A1] 连续块（A1 个切片），沿外层 R 归约（Pattern::RA）。
 * 两类块在 GM 上均连续，CopyIn/CopyOut 为单段连续 DataCopyPad；归约 Pattern 与广播方式不同。
 *
 * fp16/bf16 输入经 Cast 升 fp32 做 exp/累加，m/s 为标量/向量存 UB（reduce 输出）。
 * 元素级计算链（Cast/Sub+Exp/Div+Cast）以 __simd_vf__ 函数经 asc_vf_call 调用（regbase 范式）。
 */

#ifndef OP_KERNEL_BENCH_HARNESS_SOFTMAX_H
#define OP_KERNEL_BENCH_HARNESS_SOFTMAX_H

#include <limits>
#include <type_traits>
#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "bench_harness_softmax_tiling.h"

namespace NsBhs {
using namespace AscendC;

constexpr uint32_t kVlBytes = 256;                     // VL = 256B
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float); // = 64（fp32 单次循环元素数）
constexpr uint32_t kBlockBytes = 32;                   // UB 32B 对齐粒度

// ─── dtype 谓词 ───
template <typename D_T>
constexpr bool kIsFp32 = std::is_same_v<D_T, float>;
template <typename D_T>
constexpr bool kIsB16 = (sizeof(D_T) == 2); // fp16 / bf16

// CastTrait：b16 → fp32（扩位，UNKNOWN / CAST_NONE）
template <typename D_T>
constexpr Reg::CastTrait kCastB16ToF32{
    Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
// CastTrait：fp32 → b16（缩位，NO_SAT / CAST_RINT）
template <typename D_T>
constexpr Reg::CastTrait kCastF32ToB16{
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// 对齐工具
__aicore__ inline int64_t CeilAlignKernel(int64_t v, int64_t bs) { return ((v + bs - 1) / bs) * bs; }
__aicore__ inline int64_t CeilDivKernel(int64_t a, int64_t b) { return (a + b - 1) / b; }

// D_T 的 -inf（padding 值：使 ReduceMax 忽略 padding，且 exp(-inf-m)=0 使 ReduceSum 忽略 padding）
template <typename D_T>
__aicore__ inline D_T NegInfD()
{
    return static_cast<D_T>(-std::numeric_limits<float>::infinity());
}

// ════════════════════════════════════════════════════════════════════════════
// __simd_vf__ 函数（经 asc_vf_call 调用）：元素级计算链
// 约束：for 从 0 起、循环变量 uint16_t、禁数组、禁成员变量、禁运行时 if（用 if constexpr）。
// ════════════════════════════════════════════════════════════════════════════

// VF1：Cast D_T → fp32（b16 经 DIST_UNPACK_B16 扩位；fp32 直通复制）。
//   处理 count 个元素（含 padding，padding 经 -inf 扩位后供 reduce 忽略）。
template <typename D_T>
__simd_vf__ inline void CastInVf(
    __ubuf__ float* dstAddr, __ubuf__ D_T* srcAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> fReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        if constexpr (kIsFp32<D_T>) {
            Reg::LoadAlign(fReg, reinterpret_cast<__ubuf__ float*>(srcAddr) + off);
        } else {
            Reg::RegTensor<D_T> bReg;
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(bReg, srcAddr + off);
            Reg::Cast<float, D_T, kCastB16ToF32<D_T>>(fReg, bReg, mask);
        }
        Reg::StoreAlign(dstAddr + off, fReg, mask);
    }
}

// VF2：Sub + Exp（AR 扁平，标量 max 广播）。e = exp(x - max)，处理 count 个元素。
template <typename D_T>
__simd_vf__ inline void SubExpVfAr(
    __ubuf__ float* eAddr, __ubuf__ float* xAddr, float maxScalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> xReg, eReg, maxReg;
    Reg::MaskReg mask;
    Reg::Duplicate(maxReg, maxScalar);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(xReg, xAddr + off);
        Reg::Sub(xReg, xReg, maxReg, mask);
        Reg::Exp(eReg, xReg, mask);
        Reg::StoreAlign(eAddr + off, eReg, mask);
    }
}

// VF3：Sub + Exp（ARA 单行，向量 max）。处理一行 count 个元素，max 来自 maxPtr（count 同长）。
template <typename D_T>
__simd_vf__ inline void SubExpVfRow(
    __ubuf__ float* eAddr, __ubuf__ float* xAddr, __ubuf__ float* maxAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> xReg, eReg, maxReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(maxReg, maxAddr + off);
        Reg::LoadAlign(xReg, xAddr + off);
        Reg::Sub(xReg, xReg, maxReg, mask);
        Reg::Exp(eReg, xReg, mask);
        Reg::StoreAlign(eAddr + off, eReg, mask);
    }
}

// VF4：Div + Cast（AR 扁平，标量 s 广播）。y = (e / s) cast D_T，处理 count 个元素。
template <typename D_T>
__simd_vf__ inline void DivCastVfAr(
    __ubuf__ D_T* yAddr, __ubuf__ float* eAddr, float sScalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> eReg, sReg;
    Reg::MaskReg mask;
    Reg::Duplicate(sReg, sScalar);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(eReg, eAddr + off);
        Reg::Div(eReg, eReg, sReg, mask);
        if constexpr (kIsFp32<D_T>) {
            Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(yAddr) + off, eReg, mask);
        } else {
            Reg::RegTensor<D_T> bReg;
            Reg::Cast<D_T, float, kCastF32ToB16<D_T>>(bReg, eReg, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(yAddr + off, bReg, mask);
        }
    }
}

// VF5：Div + Cast（ARA 单行，向量 s）。处理一行 count 个元素，s 来自 sPtr（count 同长）。
template <typename D_T>
__simd_vf__ inline void DivCastVfRow(
    __ubuf__ D_T* yAddr, __ubuf__ float* eAddr, __ubuf__ float* sAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> eReg, sReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(sReg, sAddr + off);
        Reg::LoadAlign(eReg, eAddr + off);
        Reg::Div(eReg, eReg, sReg, mask);
        if constexpr (kIsFp32<D_T>) {
            Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(yAddr) + off, eReg, mask);
        } else {
            Reg::RegTensor<D_T> bReg;
            Reg::Cast<D_T, float, kCastF32ToB16<D_T>>(bReg, eReg, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(yAddr + off, bReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BenchHarnessSoftmaxKernel：Branch-0 FullloadR kernel 类
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
class BenchHarnessSoftmaxKernel {
public:
    using D_T = DType;

    __aicore__ inline BenchHarnessSoftmaxKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BenchHarnessSoftmaxTilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneBlock(int64_t blockIdx);
    __aicore__ inline void CopyIn(int64_t gmOff, int64_t elems, LocalTensor<D_T>& dst);
    __aicore__ inline void CopyOut(int64_t gmOff, int64_t elems, LocalTensor<D_T>& src);

    // ─── pattern / 几何 ───
    bool isAr_ = true;          // true=AR(axisNum=2)，false=ARA(axisNum=3)
    int32_t axisNum_ = 0;
    int64_t axisShape_[MAX_PATTERN_RANK] = {0};
    int64_t axisStride_[MAX_PATTERN_RANK] = {0};
    int64_t rTotal_ = 0;        // R 轴有效长（= axisShape[1]）
    int64_t aBundle_ = 1;       // AR=1，ARA=A1=axisShape[2]
    int64_t blockElems_ = 0;    // 单块元素数 = R * aBundle = axisStride[0]
    int64_t totalBlocks_ = 0;   // 总块数 = axisShape[0]
    int64_t rAlignElems_ = 0;   // AR：R padded 到 D_T 32B 对齐（元素）
    int64_t aBundleAlignElems_ = 1; // max/sum buf 元素数（fp32 32B 对齐）
    int64_t vfCount_ = 0;       // VF 处理元素数：AR=rAlignElems，ARA 块整体=blockElems（按行调用）
    int32_t elemBytes_ = 0;

    // ─── 多核 ───
    int32_t usedCoreNum_ = 0;

    // ─── GM + UB ───
    GlobalTensor<D_T> xGm_;
    GlobalTensor<D_T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> xQue_;     // xBuf DB=2
    TQue<QuePosition::VECOUT, 1> yQue_;    // yBuf 不开 DB
    TBuf<QuePosition::VECCALC> xF32Buf_;   // Cast 升 fp32 后的输入
    TBuf<QuePosition::VECCALC> eBuf_;      // exp(x-m) 中间结果（fp32）
    TBuf<QuePosition::VECCALC> maxBuf_;    // ReduceMax 输出（aBundleAlignElems_ fp32）
    TBuf<QuePosition::VECCALC> sumBuf_;    // ReduceSum 输出（aBundleAlignElems_ fp32）
};

// ════════════════════════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxKernel<DType>::Init(
    GM_ADDR x, GM_ADDR y, const BenchHarnessSoftmaxTilingData* td)
{
    axisNum_ = td->axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        axisShape_[i] = td->axisShape[i];
        axisStride_[i] = td->axisStride[i];
    }
    rTotal_ = td->rTotal;
    elemBytes_ = static_cast<int32_t>(sizeof(D_T));
    usedCoreNum_ = td->usedCoreNum;

    isAr_ = (axisNum_ == 2);
    aBundle_ = isAr_ ? 1 : axisShape_[2];
    blockElems_ = rTotal_ * aBundle_;        // = axisStride_[0]
    totalBlocks_ = axisShape_[0];

    const int64_t bsElemD = static_cast<int64_t>(kBlockBytes) / elemBytes_; // D_T 32B 元素数
    const int64_t bsElemF32 = static_cast<int64_t>(kBlockBytes) / 4;        // fp32 32B 元素数 = 8
    rAlignElems_ = CeilAlignKernel(rTotal_, bsElemD);                        // AR：D_T 对齐 padded
    aBundleAlignElems_ = CeilAlignKernel(aBundle_, bsElemF32);               // max/sum buf（fp32 对齐）
    if (aBundleAlignElems_ < bsElemF32) {
        aBundleAlignElems_ = bsElemF32; // 至少 32B
    }
    vfCount_ = isAr_ ? rAlignElems_ : blockElems_; // AR 处理 padded（含 -inf pad）；ARA 处理整块

    // GM
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y));

    // UB buffer 尺寸
    const int64_t xBufElems = isAr_ ? rAlignElems_ : CeilAlignKernel(blockElems_, bsElemD);
    const int64_t xBufBytes = xBufElems * elemBytes_;
    const int64_t xF32Elems = isAr_ ? rAlignElems_ : CeilAlignKernel(blockElems_, bsElemF32);
    const int64_t xF32Bytes = xF32Elems * 4;
    const int64_t reduceBufBytes = aBundleAlignElems_ * 4;

    pipe_.InitBuffer(xQue_, /*bufNum=*/2, xBufBytes);
    pipe_.InitBuffer(yQue_, /*bufNum=*/1, xBufBytes);
    pipe_.InitBuffer(xF32Buf_, xF32Bytes);
    pipe_.InitBuffer(eBuf_, xF32Bytes);
    pipe_.InitBuffer(maxBuf_, reduceBufBytes);
    pipe_.InitBuffer(sumBuf_, reduceBufBytes);
}

// ════════════════════════════════════════════════════════════════════════════
// CopyIn：GM 连续段 → UB（DataCopyPad，行内非对齐尾用 -inf pad 覆盖）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxKernel<DType>::CopyIn(
    int64_t gmOff, int64_t elems, LocalTensor<D_T>& dst)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(elems * elemBytes_);
    ext.blockCount = 1;
    ext.srcStride = 0;
    ext.dstStride = 0;
    ext.rsv = 0;
    // 行内非 32B 对齐尾：rightPadding 把 dummy 拉到 paddingValue 路径
    uint32_t misalign = ext.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = (misalign == 0) ? 0 : static_cast<uint8_t>((kBlockBytes - misalign) / elemBytes_);
    DataCopyPadExtParams<D_T> padParams{/*isPad=*/true, /*leftPadding=*/0, rPad, NegInfD<D_T>()};
    DataCopyPad(dst, xGm_[gmOff], ext, padParams);
}

// ════════════════════════════════════════════════════════════════════════════
// CopyOut：UB → GM 连续段（DataCopyPad，仅写有效字节）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxKernel<DType>::CopyOut(
    int64_t gmOff, int64_t elems, LocalTensor<D_T>& src)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(elems * elemBytes_);
    ext.blockCount = 1;
    ext.srcStride = 0;
    ext.dstStride = 0;
    ext.rsv = 0;
    DataCopyPad(yGm_[gmOff], src, ext);
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessOneBlock：单块完整 softmax（AR 单切片 / ARA 单 a0 块）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxKernel<DType>::ProcessOneBlock(int64_t blockIdx)
{
    const int64_t gmOff = blockIdx * blockElems_;
    const int64_t elems = blockElems_; // AR=R；ARA=R*A1

    // ── CopyIn（MTE2）──
    auto xLocal = xQue_.template AllocTensor<D_T>();
    CopyIn(gmOff, elems, xLocal);
    xQue_.EnQue(xLocal);
    auto xDeq = xQue_.template DeQue<D_T>();

    // ── Cast D_T → fp32（V，asc_vf_call）──
    auto xF32 = xF32Buf_.Get<float>();
    __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
    __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
    const uint32_t castCount = static_cast<uint32_t>(vfCount_);
    const uint16_t castReps = static_cast<uint16_t>(CeilDivKernel(castCount, kRepF32));
    asc_vf_call<CastInVf<D_T>>(xF32Ptr, xPtr, castCount, kRepF32, castReps);

    xQue_.FreeTensor(xDeq); // xBuf 可被下一块复用（DB=2）

    // ── ReduceMax（V）──
    auto maxDst = maxBuf_.Get<float>();
    auto eDst = eBuf_.Get<float>();
    if constexpr (true) {
        uint32_t srcShape[2];
        if (isAr_) {
            // AR：[1, R_align]，沿内层 R 归约 → 1 个标量
            srcShape[0] = 1;
            srcShape[1] = static_cast<uint32_t>(rAlignElems_);
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR>(
                maxDst, xF32, srcShape, /*srcInnerPad=*/true);
        } else {
            // ARA：[R, A1]，沿外层 R 归约 → A1 个值
            srcShape[0] = static_cast<uint32_t>(rTotal_);
            srcShape[1] = static_cast<uint32_t>(aBundle_);
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(
                maxDst, xF32, srcShape, /*srcInnerPad=*/true);
        }
    }

    // ── Sub + Exp（V，asc_vf_call）：e = exp(xF32 - max) ──
    __ubuf__ float* ePtr = reinterpret_cast<__ubuf__ float*>(eDst.GetPhyAddr());
    if (isAr_) {
        // AR：标量 max 广播，处理 rAlignElems_（含 -inf padding → exp=0，供 ReduceSum 忽略）
        float m = maxDst.GetValue(0);
        const uint32_t cnt = static_cast<uint32_t>(rAlignElems_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        asc_vf_call<SubExpVfAr<D_T>>(ePtr, xF32Ptr, m, cnt, kRepF32, reps);
    } else {
        // ARA：每行向量 max，循环 R 行
        __ubuf__ float* maxPtr = reinterpret_cast<__ubuf__ float*>(maxDst.GetPhyAddr());
        const uint32_t cnt = static_cast<uint32_t>(aBundle_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t off = r * aBundle_;
            asc_vf_call<SubExpVfRow<D_T>>(ePtr + off, xF32Ptr + off, maxPtr, cnt, kRepF32, reps);
        }
    }

    // ── ReduceSum（V，fp32 累加）──
    auto sumDst = sumBuf_.Get<float>();
    {
        uint32_t srcShape[2];
        if (isAr_) {
            srcShape[0] = 1;
            srcShape[1] = static_cast<uint32_t>(rAlignElems_);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
                sumDst, eDst, srcShape, /*srcInnerPad=*/true);
        } else {
            srcShape[0] = static_cast<uint32_t>(rTotal_);
            srcShape[1] = static_cast<uint32_t>(aBundle_);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(
                sumDst, eDst, srcShape, /*srcInnerPad=*/true);
        }
    }

    // ── Div + Cast（V，asc_vf_call）：y = (e / s) cast D_T ──
    auto yLocal = yQue_.template AllocTensor<D_T>();
    __ubuf__ D_T* yPtr = reinterpret_cast<__ubuf__ D_T*>(yLocal.GetPhyAddr());
    if (isAr_) {
        float s = sumDst.GetValue(0);
        const uint32_t cnt = static_cast<uint32_t>(rAlignElems_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        asc_vf_call<DivCastVfAr<D_T>>(yPtr, ePtr, s, cnt, kRepF32, reps);
    } else {
        __ubuf__ float* sPtr = reinterpret_cast<__ubuf__ float*>(sumDst.GetPhyAddr());
        const uint32_t cnt = static_cast<uint32_t>(aBundle_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t off = r * aBundle_;
            asc_vf_call<DivCastVfRow<D_T>>(yPtr + off, ePtr + off, sPtr, cnt, kRepF32, reps);
        }
    }

    // ── CopyOut（MTE3）──
    yQue_.EnQue(yLocal);
    auto yDeq = yQue_.template DeQue<D_T>();
    CopyOut(gmOff, elems, yDeq);
    yQue_.FreeTensor(yDeq);
}

// ════════════════════════════════════════════════════════════════════════════
// Process：按块循环多核分发（大小核均衡，确定性 cyclic-by-block）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return; // idle core
    }
    if (totalBlocks_ <= 0) {
        return;
    }
    const int64_t coreNum = static_cast<int64_t>(usedCoreNum_);
    int64_t small = totalBlocks_ / coreNum;            // floor
    int64_t bigCnt = totalBlocks_ % coreNum;
    int64_t big = small + 1;
    int64_t start, end;
    if (blockIdx < bigCnt) {
        start = blockIdx * big;
        end = start + big;
    } else {
        start = bigCnt * big + (blockIdx - bigCnt) * small;
        end = start + small;
    }
    for (int64_t b = start; b < end; ++b) {
        ProcessOneBlock(b);
    }
}

} // namespace NsBhs

#endif // OP_KERNEL_BENCH_HARNESS_SOFTMAX_H
```

## 预防

- 写"保维归约"kernel 前先做合轴归一：把任意 rank/dim 归一到 AR/ARA 两类 pattern，搬运与 reduce 的分支数立刻收敛（本例 22 例 ST 全部命中这两类）。
- DataCopyPad 的 paddingValue 显式选 −inf（对 max/sum 双归约安全），不要默认 0。
- reduce 前把输入 Cast 到 fp32：bf16/fp16 直接 exp/div 精度不达标，且 `srcInnerPad=true` 依赖 padded 值可安全参与归约。

## 验证信号

以下信号来自该算子 FullloadR 分支的实现会话：kernel 从桩（ST 0/22 红）替换为上述实现后，同一 ST 与 TilingUT 的实测结果。

- 实现前红基线：ST 0/22（正常例 rtol/atol fail、E1-E3 isnan mismatch、E5 期望 1/n 得 0、E6 sum=0.000000），run.sh 退出码 0（report 模式）（来源平台：950；证据等级：tool_observed）。
- 实现后：构建退出码 0，无 `do not registe tiling struct` 告警，3 个 dtype kernel 二进制生成（来源平台：950；证据等级：tool_observed）。
- 实现后：ST 22/22 全 PASS（S1/S2/S4/B1/F3/F4/F5 各 dtype + E1-E3 isnan pointwise ok + E4 ok + E5 ok + E6 sum_to_one ok）（来源平台：950；证据等级：tool_observed）。
- 回归：TilingUT 82/82 全 PASS（56 公共 + 26 分支例）（来源平台：950；证据等级：tool_observed）。
- 范式硬门：`grep -rn "asc_vf_call" op_kernel/` 10 处真实调用、`grep -rn "__simd_vf__" op_kernel/` 7 处真实定义（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅覆盖 rTotal ≤ rFullloadThresh 的全载分支；R 更大时走 ChunkedR（本卡未实现）。
- E6（fp16 全相同值 60000，exp 上溢边界）的 golden 自身可能 NaN，ST 靠 sum_to_one 不变量兜底判定——该场景的逐点数值一致性未单独验证。
- −inf 尾填充方案依赖"padding 值可被 max/exp 双归约忽略"这一数值性质；若算子语义改变（如 min 归约），padding 值须重选（+inf）。
- `Pattern::Reduce::AR/RA` 的 srcShape 上界、`GetValue` 在更深流水下的正确性边界未压测；GetValue 的 stall 代价未量化。

**未决项**：

- `GetValue` 标量回读的流水线 stall 代价未量化；候选替代（Brcb 预广播 / 寄存器 lane-0 广播）未验证——性能优化时优先测量点。
- ChunkedR（rChunkCnt > 1）与 EmptyTensor（usedCoreNum == 0）分支未实现（入口早退），多 chunk 跨块 carry 语义待后续闭环。
- E6 类上溢边界的逐点数值一致性未验证（仅不变量兜底）。
## 坏实践

- 用 MTE2 `blockCount>1 + dstStride=0` 期望 UB 内得到连续布局——实际每块落到 32B 对齐边界，得到散射布局，后续连续 reduce 直接算错。
- DataCopyPad 尾填充用 0 或不填——全负行的 max 被 0 顶替、exp(0−m) 污染 sum，双归约同时出错。
- 在 `__simd_vf__` 函数里写运行时 if / 成员变量 / 非 0 起始循环 / 数组——VF 约束（for 从 0 起、循环变量 uint16_t、禁数组、禁成员、禁运行时 if，用 `if constexpr`）。

## 误诊

- 曾考虑放弃高层 ReduceMax/ReduceSum、改用"in-register 树形归约"规避 GetValue——树形跨 lane 归约在 VF 里同样受限，最终回退高层 reduce + GetValue；教训：先确认约束（regbase 白名单）再换路线，避免在未验证的寄存器级 API 上空转。
