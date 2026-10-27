---
type: "Runbook"
title: dav-3510 混合 dtype 广播算子多核 kernel 骨架配方
description: TBuf 四缓冲全 FP32 域 + NDDMA 随路广播 CopyIn + DataCopyPad CopyOut + __simd_vf__/asc_vf_call 单函数 VF 链 + 三 dtype 模板 + 分阶段事件同步 + 核号守卫的完整多核 kernel 配方（ST 48/48 验证）。
tags: [task.implementation, topic.data_movement.data_copy, topic.compute.elementwise.activation, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# dav-3510 混合 dtype 广播算子多核 kernel 骨架配方

## 现象（任务形态）

为 softplus_v2_grad（SoftplusV2 反向梯度：`gradInput = gradOutput · (βx>τ ? 1 : sigmoid(βx))`，numpy broadcast，2 输入 1 输出，FP32/FP16/BF16 共 9 种 dtype 组合，0D~8D）实现 RANK_4 分支 kernel：多核切分、随路广播搬运、连续 Vector 计算链（Muls→Neg→Exp→Adds→Div→Compare→Select）。dav-3510 平台硬约束：连续 Vector 计算链必须用 `__simd_vf__` + `asc_vf_call` 实现（禁止拆成 `AscendC::` UB API 分离调用）；broadcast 范式禁止 TQue、CopyIn 用 NDDMA。该骨架在真实设备完成 48/48 ST 验证（含 NaN/±Inf/全零/FP16 上溢特殊值逐点对齐）。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（broadcast 范式、多核 AIV、混合 dtype、数值稳定分支）。
- API 路径：`AscendC::TBuf`/`TPipe`/`DataCopy<T,ND,cfg>`（NDDMA）/`DataCopyPad`/`Cast`/`AscendC::Reg::*`（VF 寄存器 API）/`asc_vf_call`/`SetFlag/WaitFlag`（MTE2_V、V_MTE2、V_MTE3、MTE3_MTE2）。
- 验证过的平台与 CANN 版本：ascend950（dav-3510，`ASCEND_COMPUTE_UNIT=ascend950`，`__NPU_ARCH__==3510`）+ cann-9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（broadcast/elementwise 类混合 dtype 算子的 kernel 骨架）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：含 Cube/AIC 的算子；RANK>5 的外层 Flat loop 多段搬运（本骨架 ND=min(RANK,5) 未实现外循环，该机制见[rank>5 广播缺外层 Flat loop 卡](../precision/rank_gt5_nddma_outer_loop_missing.md)）；其他平台未验证。

## 设计决策

1. **三 dtype 模板参数化**：构建系统按 dtype 组合生成多个 kernel binary，并对每个输入/输出注入 `-DDTYPE_<NAME>=<C++类型>` 宏（`DTYPE_MAP`: float32→`float`、float16→`half`、bfloat16→`bfloat16_t`，另有 `-DORIG_DTYPE_*`/`-DFORMAT_*`）。因此 kernel 类模板化为 `SoftplusV2GradKernel<TGrad, TFeat, TOut, RANK>`，用 `DTYPE_INPUT_GRADIENTS/DTYPE_INPUT_FEATURES/DTYPE_OUTPUT_BACKPROPS` 实例化——混合 dtype 组合（如 fp16,fp32→fp32）天然支持，无需运行时分派。kernel.h 必须 `#include "softplus_v2_grad_struct.h"`（TilingKey 模板机制才激活）。
2. **UB 全程 FP32 域 + TBuf 四缓冲**（`kPhysNodes=4`，禁止 TQue）：B0=staging（非 FP32 输入 CopyIn 源/非 FP32 输出 Cast 目的，可复用）、B1=feat FP32、B2=grad FP32、B3=result FP32。FP16/BF16 输入先 NDDMA 搬到 B0 再 `Cast(CAST_NONE)`→FP32；计算全程 FP32（x=60000 的 FP16 上溢用例走 threshold 分支不出中间溢出）；输出 `Cast(CAST_RINT)`→TOut 后 DataCopyPad 出去。
3. **CopyIn 用 NDDMA 随路广播**（`DataCopy<T,ND,cfg>` + `MultiCopyParams`）：广播轴 `loopSrcStride=0`，切分轴 `loopSize=a_i_seg` 运行时填、`loopDstStride` 按内层乘积填；**不要**用 1D 拷贝循环替代（含内层广播维时实测大面积错值）。CopyOut 统一 `DataCopyPad`（`blockLen=cnt×sizeof(T)`，无 32B 对齐要求）。
4. **VF 链单函数**：S1→S7 全链在一个 `__simd_vf__` 函数内（`__simd_vf__` 内只能调 `__simd_callee__`（`AscendC::Reg::*`）和 constexpr 函数），中间量 z/neg_z/exp/denom/sigmoid/cmp 全留寄存器不占 UB；`__aicore__` 侧用 `asc_vf_call<SoftplusV2GradVF<float>>` 调用，`oneRepeatSize = GetVecLen()/sizeof(float)`、`repeatTimes = CeilDivision(count, VL_F32)`。
5. **Reg:: API 签名先查 SDK 头**（`kernel_reg_compute_vec_*.h`）：`Muls/Adds(dstReg,srcReg,scalar,mask)`、`Neg/Exp(dstReg,srcReg,mask)`、`Div(dstReg,src0,src1,mask)`、`Compares<T,CMPMODE::GT>(maskDst,srcReg,scalar,mask)`、`Select(dstReg,src0,src1,mask)`、`LoadAlign(reg,ubufAddr,aReg)`、`StoreAlign(ubufAddr,reg,aReg,mask)`、`UpdateMask<T>(count)`、`CreateAddrReg<T>(i,oneRepeatSize)`。
6. **分阶段事件同步**：直读 FP32 输入时在 VF 前 MTE2_V 同步；Cast 路径每个 Cast 前后各同步（MTE2_V / V_MTE2）；CopyOut 前 V_MTE3；下轮 CopyIn 等上轮 CopyOut（MTE3_MTE2，buffer 复用）。
7. **多核入口守卫**：`Process()` 首行 `if (GetBlockIdx() >= num_cores) return;`（平台可能启动 BlockDim 外的核，越界分发崩溃的判别与守卫见[507035 多核分发未拦截多余核越界卡](../runtime/507035_unguarded_extra_cores_oob_dispatch.md)）。
8. **多核调度框架**：`GetCoreRange`（cores_tail 负载均衡）/`GetUBSplitRange`（尾块）/`FlatToEffectiveCoord`（flat tile → 多维坐标）/`CalcInput/OutputOffset`（广播 stride=0 求和）为 int64_t* 无 std::vector 版本（kernel 侧不可用容器）。

## 系统信息

- 工具链：CANN cann-9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装根），CMake `ASCEND_COMPUTE_UNIT=ascend950`，kernel 面向 `__NPU_ARCH__==3510`（dav-3510）。
- 算子工程：`operators/softplus_v2_grad_package/`（`build.sh <pkg>` → `make binary package` → 安装 .run）。
- 目录布局：`op_kernel/`（kernel.cpp 入口 + kernel.h 实现 + tiling_struct.h/struct.h）、`op_host/`（host.cpp TilingFunc + common/rank4/rank8 tiling）、`tests/aclnn/`（ST）、`tests/tiling_ut/`（TilingUT）。
- 相关文件：`op_kernel/softplus_v2_grad_kernel.cpp`、`op_kernel/softplus_v2_grad_kernel.h`、`op_kernel/softplus_v2_grad_tiling_struct.h`（`SoftplusV2GradTilingData<kRank>`、`kPhysNodes=4`、`kMaxNdDmaDims=5`）、`op_kernel/softplus_v2_grad_struct.h`（TilingKey `ASCENDC_TPL_*` 声明）、`build/ascendc_kernels/binary/dynamic/softplus_v2_grad.py`（构建系统 DTYPE 宏注入）。

## 完整代码

kernel 入口（`softplus_v2_grad_kernel.cpp`，RANK 来自 TilingKey、DTYPE 来自构建系统宏，`if constexpr` 编译期分派）：

```cpp
// softplus_v2_grad_kernel.cpp — Kernel 入口
// TilingData 按 RANK 模板化分两档实例化:
//   TilingData4 — RANK≤4: 数组维度 [4]
//   TilingData8 — RANK>4: 数组维度 [8]
// RANK 来自 TilingKey (ASCENDC_TPL_* in struct.h), DTYPE 来自 CANN 框架
// (def 注册的 Input 类型, build 系统 -DDTYPE_INPUT_GRADIENTS=... 等)。
// 三 dtype 独立: TGrad/TFeat/TOut (异 dtype 组合输出 FP32)。
#include "kernel_operator.h"
#include "softplus_v2_grad_kernel.h"
#include "softplus_v2_grad_tiling.h"

using TilingData4 = SoftplusV2GradTilingData<4>;  // RANK≤4: 数组维度 [4]
using TilingData8 = SoftplusV2GradTilingData<8>;  // RANK>4: 数组维度 [8]

template<int RANK>
__global__ __aicore__ void softplus_v2_grad(
    GM_ADDR input_gradients, GM_ADDR input_features,
    GM_ADDR output_backprops,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        SoftplusV2GradKernel<DTYPE_INPUT_GRADIENTS, DTYPE_INPUT_FEATURES,
                             DTYPE_OUTPUT_BACKPROPS, 4> kernel;
        kernel.Init(input_gradients, input_features, output_backprops, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        SoftplusV2GradKernel<DTYPE_INPUT_GRADIENTS, DTYPE_INPUT_FEATURES,
                             DTYPE_OUTPUT_BACKPROPS, 8> kernel;
        kernel.Init(input_gradients, input_features, output_backprops, &td);
        kernel.Process();
    }
}
```

kernel 实现（`softplus_v2_grad_kernel.h`，TBuf 四缓冲 + NDDMA CopyIn + VF 链 + 分阶段同步）：

```cpp
// softplus_v2_grad_kernel.h — SoftplusV2GradKernel<TGrad,TFeat,TOut,RANK>
//
// Broadcast 范式硬约束:
//   禁止 TQue，一律 TBuf；CopyIn 用 NDDMA DataCopy<T,ND,cfg>；CopyOut 用 DataCopyPad。
// dav-3510 平台连续 Vector 计算链必须用 __simd_vf__ + asc_vf_call 实现。
// VF 计算链 S1→S7: Muls→Neg→Exp→Adds→Div→Compares(GT)→Select，全程 FP32 域。
#pragma once
#include "kernel_operator.h"
#include "softplus_v2_grad_tiling_struct.h"
#include "softplus_v2_grad_struct.h"  // TilingKey 模板化 — 缺此行模板机制不激活

// ============================================================
// Type trait: FP32 判定 (避免 <type_traits> 依赖)
// ============================================================
template <typename T> struct Spv2gIsFp32 { static constexpr bool value = false; };
template <> struct Spv2gIsFp32<float> { static constexpr bool value = true; };

// ============================================================
// Kernel 侧辅助函数 (int64_t* 版本, 无 std::vector)
// ============================================================

__aicore__ inline void GetCoreRange(int64_t core_id, int64_t tiles_main, int64_t cores_tail,
    int64_t& start, int64_t& end)
{
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

__aicore__ inline int64_t GetUBSplitRange(
    int64_t a_o_off, int64_t a_o, int64_t a_i, int64_t a_i_tail)
{
    return (a_o_off == a_o - 1) ? a_i_tail : a_i;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* max_bro_shape,
    int64_t rank, int64_t split_axis, int64_t a_i, int64_t a_o, int64_t* eff_coord)
{
    for (int64_t d = 0; d < rank; d++)
        eff_coord[d] = 0;
    int64_t a_o_off = flat % a_o;
    int64_t outer = flat / a_o;
    for (int64_t d = split_axis - 1; d >= 0; d--) {
        eff_coord[d] = outer % max_bro_shape[d];
        outer /= max_bro_shape[d];
    }
    eff_coord[split_axis] = a_o_off * a_i;
    return true;
}

__aicore__ inline int64_t CalcInputOffset(
    const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;  // 元素个数，gmIn_[] 的 index
}

__aicore__ inline int64_t CalcOutputOffset(
    const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;  // 元素个数，gmOut_[] 的 index
}

// ============================================================
// VF 计算链 S1→S7
//   输入: src0 = input_gradients (g), src1 = input_features (x)  [FP32]
//   参数: beta (β), threshold (τ)
//   输出: dst = output_backprops  [FP32]
//   中间量 z/neg_z/exp_neg_z/denom/sigmoid_val/cmp_mask 全留寄存器，不占 UB buffer。
//   __simd_vf__ 内只能调 __simd_callee__ (AscendC::Reg::*) 和 constexpr 函数。
// ============================================================
template <typename T>
__simd_vf__ inline void SoftplusV2GradVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes,
    float beta, float threshold);

// ============================================================
// Kernel 类
//   TGrad = DTYPE_INPUT_GRADIENTS, TFeat = DTYPE_INPUT_FEATURES,
//   TOut  = DTYPE_OUTPUT_BACKPROPS. 三者可不同 (异 dtype 组合输出 FP32)。
//   UB 全程 FP32 (float): FP16/BF16 输入 Cast→FP32, VF FP32, 结果 Cast→输出 dtype。
//   Buffer 划分 (kPhysNodes=4):
//     B0 = staging (非 FP32 输入 CopyIn 源 / 非 FP32 输出 Cast 目的, 可复用)
//     B1 = input_features FP32 (Cast 后或直读)
//     B2 = input_gradients FP32 (Cast 后或直读)
//     B3 = result FP32 (VF 输出)
// ============================================================
template <typename TGrad, typename TFeat, typename TOut, int64_t RANK>
class SoftplusV2GradKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr uint32_t VL_F32 = AscendC::GetVecLen() / sizeof(float);

    static constexpr int B0 = 0;  // staging / output Cast
    static constexpr int B1 = 1;  // feat FP32
    static constexpr int B2 = 2;  // grad FP32
    static constexpr int B3 = 3;  // result FP32

    AscendC::TPipe pipe_;
    const SoftplusV2GradTilingData<RANK>* td_;
    AscendC::GlobalTensor<TGrad> gmGrad_;
    AscendC::GlobalTensor<TFeat> gmFeat_;
    AscendC::GlobalTensor<TOut>  gmOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    AscendC::MultiCopyParams<TGrad, static_cast<uint8_t>(ND)> nddmaGrad_;
    AscendC::MultiCopyParams<TFeat, static_cast<uint8_t>(ND)> nddmaFeat_;

public:
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR feat, GM_ADDR out,
                                const SoftplusV2GradTilingData<RANK>* td)
    {
        td_ = td;
        gmGrad_.SetGlobalBuffer((__gm__ TGrad*)grad);
        gmFeat_.SetGlobalBuffer((__gm__ TFeat*)feat);
        gmOut_.SetGlobalBuffer((__gm__ TOut*)out);
        for (int i = 0; i < kPhysNodes; i++)
            pipe_.InitBuffer(buf_[i], static_cast<uint32_t>(td->per_buf_bytes));

        // NDDMA 参数 (ND=min(RANK,5), RANK≤5 单次覆盖全维)
        const int64_t* dstShape = td->max_bro_shape;
        int64_t k = td->split.axis;
        int64_t nddma_dims = (RANK - k <= ND) ? (RANK - k) : ND;

        // input_gradients NDDMA params
        {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaGrad_.loopInfo.loopSize[nd]      = static_cast<uint32_t>((d == k) ? 0 : dstShape[d]);
                nddmaGrad_.loopInfo.loopSrcStride[nd] = static_cast<uint64_t>(td->input_strides[0][d]);
                nddmaGrad_.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
                nddmaGrad_.loopInfo.loopLpSize[nd]     = 0;
                nddmaGrad_.loopInfo.loopRpSize[nd]     = 0;
                inner *= (d == k) ? td->split.a_i : dstShape[d];
                nd++;
            }
            for (; nd < ND; nd++) {
                nddmaGrad_.loopInfo.loopSize[nd]      = 1;
                nddmaGrad_.loopInfo.loopSrcStride[nd] = 0;
                nddmaGrad_.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
                nddmaGrad_.loopInfo.loopLpSize[nd]     = 0;
                nddmaGrad_.loopInfo.loopRpSize[nd]     = 0;
            }
        }
        // input_features NDDMA params
        {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaFeat_.loopInfo.loopSize[nd]      = static_cast<uint32_t>((d == k) ? 0 : dstShape[d]);
                nddmaFeat_.loopInfo.loopSrcStride[nd] = static_cast<uint64_t>(td->input_strides[1][d]);
                nddmaFeat_.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
                nddmaFeat_.loopInfo.loopLpSize[nd]     = 0;
                nddmaFeat_.loopInfo.loopRpSize[nd]     = 0;
                inner *= (d == k) ? td->split.a_i : dstShape[d];
                nd++;
            }
            for (; nd < ND; nd++) {
                nddmaFeat_.loopInfo.loopSize[nd]      = 1;
                nddmaFeat_.loopInfo.loopSrcStride[nd] = 0;
                nddmaFeat_.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
                nddmaFeat_.loopInfo.loopLpSize[nd]     = 0;
                nddmaFeat_.loopInfo.loopRpSize[nd]     = 0;
            }
        }
        (void)nddma_dims;  // RANK=4 → always ≤ ND, no outer loop
    }

    __aicore__ inline void Process()
    {
        // Guard: cores beyond num_cores must return immediately (some platforms
        // launch all physical cores regardless of BlockDim).
        if (AscendC::GetBlockIdx() >= td_->multicore.num_cores) return;

        // 事件 ID
        int32_t evMTE2toV    = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE2    = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        int32_t evVtoMTE3    = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main,
                     td_->multicore.cores_tail, start, end);

        // inner_count = ∏ dims after split axis
        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg = GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o,
                                              td_->split.a_i, td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(flat, td_->max_bro_shape, RANK,
                                 td_->split.axis, td_->split.a_i, td_->split.a_o, coord);

            // 上轮 CopyOut(MTE3) 结束 → 本轮 CopyIn(MTE2) 可以开始
            if (flat != start) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // ── Phase 1: CopyIn input_gradients (g) → B2 (FP32) ──
            // B0 = staging for non-FP32; B2 = FP32 grad
            if constexpr (Spv2gIsFp32<TGrad>::value) {
                // FP32 直读 → B2
                CopyInOne<TGrad>(B2, a_i_seg, coord, 0, gmGrad_, nddmaGrad_);
                // MTE2 wrote B2; need MTE2_V before VF reads B2 → defer sync
            } else {
                // FP16/BF16: GM→B0(staging) → Cast→B2(FP32)
                CopyInOne<TGrad>(B0, a_i_seg, coord, 0, gmGrad_, nddmaGrad_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast(buf_[B2].Get<float>(), buf_[B0].Get<TGrad>(),
                              AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(count));
                // B0 free after Cast; need V_MTE2 before MTE2 reuses B0 (feat staging)
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            }

            // ── Phase 2: CopyIn input_features (x) → B1 (FP32) ──
            if constexpr (Spv2gIsFp32<TFeat>::value) {
                // FP32 直读 → B1
                CopyInOne<TFeat>(B1, a_i_seg, coord, 1, gmFeat_, nddmaFeat_);
            } else {
                // FP16/BF16: GM→B0(staging, reuse) → Cast→B1(FP32)
                CopyInOne<TFeat>(B0, a_i_seg, coord, 1, gmFeat_, nddmaFeat_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast(buf_[B1].Get<float>(), buf_[B0].Get<TFeat>(),
                              AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            }

            // ── Phase 3: VF S1→S7 (FP32 域) ──
            // B2 = grad FP32, B1 = feat FP32 → B3 = result FP32
            // MTE2_V sync: needed if any input was direct-FP32 (MTE2-written B2 or B1)
            if constexpr (Spv2gIsFp32<TGrad>::value || Spv2gIsFp32<TFeat>::value) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            }
            // else: both Cast paths already synced MTE2_V; B2/B1 are V-written (V→V implicit)
            {
                uint16_t rep = AscendC::CeilDivision(static_cast<uint32_t>(count), VL_F32);
                asc_vf_call<SoftplusV2GradVF<float>>(
                    (__ubuf__ float*)buf_[B3].Get<float>().GetPhyAddr(),   // dst
                    (__ubuf__ float*)buf_[B2].Get<float>().GetPhyAddr(),   // src0 = g
                    (__ubuf__ float*)buf_[B1].Get<float>().GetPhyAddr(),   // src1 = x
                    static_cast<uint32_t>(count), VL_F32, rep,
                    td_->beta, td_->threshold);
            }

            // ── Phase 4: CopyOut result → GM ──
            if constexpr (Spv2gIsFp32<TOut>::value) {
                // FP32 直写: B3 → GM
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                CopyOutOne(B3, a_i_seg, coord);
            } else {
                // FP16/BF16: Cast B3(FP32)→B0(TOut), then DataCopyPad B0→GM
                // B0 is free (staging done in Phase 1/2)
                AscendC::Cast(buf_[B0].Get<TOut>(), buf_[B3].Get<float>(),
                              AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(count));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                CopyOutOne(B0, a_i_seg, coord);
            }

            // 下轮 CopyIn 需等本轮 CopyOut 完成 (buffer 复用)
            if (flat != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    // NDDMA CopyIn: GM → buf_[slot], 随路 broadcast
    template <typename T>
    __aicore__ inline void CopyInOne(int slot, int64_t a_i_seg, const int64_t* coord,
        int inputIdx, AscendC::GlobalTensor<T>& gmIn,
        AscendC::MultiCopyParams<T, static_cast<uint8_t>(ND)>& baseParams)
    {
        int64_t off = CalcInputOffset(coord, td_->input_strides[inputIdx], RANK);
        auto params = baseParams;
        int64_t k = td_->split.axis;
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd) params.loopInfo.loopSize[nd] = static_cast<uint32_t>(a_i_seg);
            params.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
            inner *= params.loopInfo.loopSize[nd];
        }
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        AscendC::DataCopy<T, static_cast<uint8_t>(ND), cfg>(
            buf_[slot].Get<T>(), gmIn[off], params);
    }

    // CopyOut: buf_[slot] → GM (DataCopyPad, 无 32B 对齐要求)
    __aicore__ inline void CopyOutOne(int slot, int64_t a_i_seg, const int64_t* coord)
    {
        int64_t off = CalcOutputOffset(coord, td_->output_strides[0], RANK);
        // output shape = max_bro_shape (broadcast result), no broadcast on output side
        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];
        int64_t cnt = a_i_seg * inner_count;
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen   = static_cast<uint32_t>(cnt * sizeof(TOut));
        extParams.srcStride  = 0;
        extParams.dstStride  = 0;
        AscendC::DataCopyPad(gmOut_[off], buf_[slot].Get<TOut>(), extParams);
    }
};

// ============================================================
// VF 函数定义: S1→S7 计算链
//   __simd_vf__ 内只能调 __simd_callee__ (AscendC::Reg::*) 和 constexpr 函数。
//   Reg::Muls/Neg/Exp/Adds/Div/Compares/Select 全在寄存器内完成，中间量不占 UB。
// ============================================================
template <typename T>
__simd_vf__ inline void SoftplusV2GradVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes,
    float beta, float threshold)
{
    AscendC::Reg::RegTensor<T> gReg, xReg, zReg, negReg, expReg, denomReg, sigReg, dstReg;
    AscendC::Reg::MaskReg validMask, cmpMask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        validMask = AscendC::Reg::UpdateMask<T>(count);
        // Load g (input_gradients) and x (input_features)
        AscendC::Reg::LoadAlign(gReg, src0Addr, aReg);
        AscendC::Reg::LoadAlign(xReg, src1Addr, aReg);
        // S1: z = β * x                    (Muls, 标量乘)
        AscendC::Reg::Muls<T>(zReg, xReg, beta, validMask);
        // S2: neg_z = -z                   (Neg)
        AscendC::Reg::Neg<T>(negReg, zReg, validMask);
        // S3: exp_neg_z = exp(neg_z)       (Exp, FP32 域避免 FP16 上溢)
        AscendC::Reg::Exp<T>(expReg, negReg, validMask);
        // S4: denom = exp_neg_z + 1.0      (Adds, 标量加)
        AscendC::Reg::Adds<T>(denomReg, expReg, 1.0f, validMask);
        // S5: sigmoid_val = g / denom      (Div, 不可 in-place, buffer 需求 3)
        AscendC::Reg::Div<T>(sigReg, gReg, denomReg, validMask);
        // S6: cmp = (z > τ)               (Compares GT, FP32 域比较避免 z 已 Inf)
        AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(cmpMask, zReg, threshold, validMask);
        // S7: dst = Select(cmp, g, sig)   (z>τ → g, 否则 → g*sigmoid(z))
        AscendC::Reg::Select<T>(dstReg, gReg, sigReg, cmpMask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, validMask);
    }
}
```

## 验证信号

来源说明：本配方在该算子 RANK_4 分支的实测验证（修完 host 切分轴与测试 NaN 两个独立缺陷后的最终状态）。

- `./build.sh softplus_v2_grad_package` 退出码 0，按 9 种 dtype 组合生成独立 kernel binary（`[ascend950] Generating SoftplusV2Grad_<hash> ...`）（来源平台：950；证据等级：tool_observed）。
- ST 48/48 PASSED（含 NaN / +Inf / -Inf / 全零 / FP16 上溢逐点 bitwise 对齐），run.sh 退出码 0（来源平台：950；证据等级：tool_observed）。
- 公共 TilingUT 27/27 + RANK_4 TilingUT 12/12 无回归（来源平台：950；证据等级：tool_observed）。
- `grep -c '__simd_vf__'` = 5、`grep -c 'asc_vf_call'` = 2（均 ≥1 的验收项）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本骨架 `ND = min(RANK,5)`、单次 NDDMA 覆盖全维：RANK>5 且 split 轴落在外层维时需外层 Flat loop 多段搬运，该部分**未包含**在本配方中（机制与实现见[rank>5 广播缺外层 Flat loop 卡](../precision/rank_gt5_nddma_outer_loop_missing.md)）。
- 首次集成时 host tiling 曾存在 squeezed/padded 坐标系错位与 dtype_size 记账问题：本骨架要求 tiling 与 kernel 在同一 padded 坐标系、`dtype_size` 按 FP32 域=4 记账。
- 未验证：其他平台（a2 等）与更高 CANN 版本。

**未决项**：

- RANK_8 分支 kernel（ND=5 + 外层 Flat loop `nddmaOuterIters_=∏前3维`）未实现（本轨迹仅完成其 TilingFunc 与 ST 用例，TDD 红态；机制见关联卡）。
- 性能未测量（本轨迹无 profiling 信号），缓冲深度/双缓冲等优化方向未尝试。
## 坏实践

- 用 TQue 管理广播范式 UB（范式硬约束禁止）；把连续 Vector 链拆成 `AscendC::` UB API 分离调用（dav-3510 禁止）。
- 用 1D 拷贝循环替代 NDDMA 随路广播 CopyIn（含内层广播维时实测大面积错值）。
- kernel 侧使用 std::vector 等容器（`__aicore__` 环境不可用，辅助函数一律 int64_t* 版本）。

## 误诊

- 无（本卡为成功实现配方）。
