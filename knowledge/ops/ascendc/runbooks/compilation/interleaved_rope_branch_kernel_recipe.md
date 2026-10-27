---
type: "Runbook"
title: interleaved RoPE 分支 kernel 完整实现配方
description: dav-3510 上 interleaved 布局 RoPE（rotaryMode=interleaved ∧ BSND ∧ cos.ndim=2）分支 kernel 的完整实现配方：fused-pair 恒等式（outEven=even·c−odd·s / outOdd=odd·c+even·s）+ UB 级 DeInterleave/Interleave + asc_vf_call<MulAddVF> 计算链 + paddedH/paddedD 平面布局 + fp16/bf16 升 fp32 回转；ST 12/12 验证。
tags: [task.implementation, topic.compute.composite, topic.data_movement.layout_transform, topic.attention]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# interleaved RoPE 分支 kernel 完整实现配方

## 现象

在 Ascend950（dav-3510 向量核，regbase 单 binary 范式）上实现 apply_rotary_pos_emb 的 interleaved 布局分支（rotaryMode=interleaved ∧ layout=BSND ∧ cos.ndim=2，tilingKey=1；同一算子的 half 布局分支实现见库内 [RoPE half BSND kernel 平面分解实现](../compilation/rope_half_bsnd_planar_kernel_recipe.md)）：输入 query/key 形状 [B,S,N,D]（行 = 一个 token-head 的 D 个元素，偶位 x[2i] 与奇位 x[2i+1] 构成旋转对），cos/sin 形状 [S, D/2]（cosB=1 广播、Dh→D tile 即 c[2i]=c[2i+1]=cos_i），要求 out[2i]=x[2i]·cos_i − x[2i+1]·sin_i、out[2i+1]=x[2i+1]·cos_i + x[2i]·sin_i，覆盖 fp32/fp16/bf16 三 dtype、D=2..128、空张量（S=0）、NaN/Inf 特殊值。难点：奇偶交错的 element-stride=2 访问无法用 UB 侧 DataCopy 2D strided 表达（stride 以 32B 块为单位），且连续 Vector 计算链在 dav-3510 上必须走 `__simd_vf__` + `asc_vf_call`。本配方为最终落定的实现（独立 torch oracle 生成期望的 aclnn ST 12/12 全绿）。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（regbase），tiling 由 host 侧算好 totalRows/perCoreRows/D/Dh 等字段经 TilingData 下发，kernel 单入口按 `td->tilingKey` if/else 分发。
- API 路径：`AscendC::DeInterleave`/`AscendC::Interleave`（UB 级，`kernel_operator_vec_duplicate_intf.h`）；`AscendC::Reg::*`（`LoadAlign`/`StoreAlign`/`CreateAddrReg`/`UpdateMask`/`MulAddDst`）；`asc_vf_call`；`AscendC::DataCopy`（NDDMA）/`DataCopyPad`；`AscendC::Cast`（CAST_NONE / CAST_RINT）。
- 验证过的平台与 CANN 版本：Ascend950（Ascend950PR），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（配方绑定"成对旋转 + 交错布局 + 向量核"这一类算子，字段与形状参数化）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：half 布局（q[:, :H]/q[:, H:] 拼接式 rotate_half）走另一分支实现（见上述 half 分支卡）；cos.ndim=3、非 BSND 布局未验证；Cube/矩阵路径不适用；interleaved 链在白盒精度上相对 half 链的劣化现象与 D=88 分界证据见库内 [RoPE interleaved 链白盒精度劣于 half 链](../precision/rope_interleaved_whitebox_precision_regression.md)——精度敏感场景回落 half 路线。

## 根因/解决配方

核心是 **fused-pair 恒等式**，把"显式构造 rotate_half 向量再乘加"改写为"拆偶/奇两路 + 两组 MulAdd"，全程无 stride-2 访问：

```
rotate_half: rh[2i] = -x[2i+1], rh[2i+1] = x[2i]
out = x·cos_full + rh·sin_full，且 cos_full[2i]=cos_full[2i+1]=cos_i（Dh→D tile 性质）
⟹ 令 even[i]=x[2i], odd[i]=x[2i+1]（DeInterleave 拆分）：
    outEven[i] = even[i]·cos_i − odd[i]·sin_i = even[i]·cos_i + odd[i]·(−sin_i)   → out[2i]
    outOdd[i]  = odd[i]·cos_i  + even[i]·sin_i                                    → out[2i+1]
最后 Interleave(outEven, outOdd) 重组回交错全 D。
```

流水线（每 chunk，query 侧算完 CopyOut 后再算 key 侧，降 UB 峰值）：per-row NDDMA CopyIn（q/k 每行 D 元素、cos/sin 每行按 s 索引取 H 元素，N 轴广播）→ fp16/bf16 先 `Cast` 升 fp32（CAST_NONE 无损）→ ① `DeInterleave` per-row 拆 even/odd → ② `Muls` 取负 sin → ③ `Mul` + `asc_vf_call<MulAddVF>` 两组融合乘加（whole-chunk on cnt·paddedH）→ ④ Interleave 重组（flat 布局走 UB 级双输出整块；小 D padded 行走自写 `InterleaveVF` per-row）→ fp16/bf16 `Cast` 回原 dtype（CAST_RINT round-to-nearest-even）→ per-row `DataCopyPad` CopyOut。缓冲布局用 paddedH=alignUp(H,16)/paddedD=alignUp(D,16) 平面（每行基址恒 32B 对齐，规避 dav-3510 MTE 管线 507035），q 输入缓冲与 outFull 输出缓冲同槽复用（DeInterleave 读完 q 后 Interleave 才写）。

## 设计决策

- **fused-pair 恒等式 vs 显式 rotate_half**：显式做法需要构造 rh 向量（奇偶位取反交错摆放），在任何 UB API 下都要 stride-2 写或两次掩码写；恒等式改写后只有"拆（DeInterleave）"与"合（Interleave）"两个正交布局变换，乘加全部落在连续 even/odd 段上，可整块融合。
- **UB 级 Interleave/DeInterleave 的发现**：设计文档点名"DeInterleave/Interleave 是 regbase 白名单 Conversion-and-utility 族 API"，但按名字在 include 树里搜不到——它们实际声明在 `basic_api/kernel_operator_vec_duplicate_intf.h`（与 Duplicate 同头），寄存器级版本（`Reg::Interleave`/`Reg::DeInterleave`）在 `reg_compute/kernel_reg_compute_vec_duplicate_intf.h`；先 grep 头文件再决定手写与否，避免重造轮子。
- **重组步双路线**：flat（paddedH==H，H 为 16 倍数）用 UB 级 `AscendC::Interleave(dst0, dst1, src0, src1, count)` 双输出整块（源/目的各 count 元素、count 须偶、第二输出基址需 32B 对齐——PadUp(count,8) 保证）；非 flat 小 D 行用自写 `InterleaveVF` per-row（行长小于寄存器宽度、尾块 UpdateMask 掩码，实测正确）。整寄存器行上手写 `Reg::Interleave`+两次 `StoreAlign` 会两半互换（同族坑见库内 [DeInterleave 两路写反——按偶索引到 dst0 修正](../precision/deinterleave_dst0_is_even_lane_output.md)），这是保双路线的直接原因。
- **计算域**：fp16/bf16 一律升 fp32 计算（峰值 P+1 缓冲预算）再 CAST_RINT 回转；`Muls(ns, s, -1.0)` 预生成 −sin，使两组输出都是"Mul + MulAdd"同构形态，均可 `asc_vf_call` 融合。
- **whole-chunk Mul/MulAdd on cnt·paddedH**：padded 段的 even/odd/cos 是垃圾，但垃圾只污染 padding 段输出，有效 H 段索引不跨界；省掉 per-row 循环开销。
- **同步**：单 chunk 内 MTE2→V→MTE3 用 `SetFlag/WaitFlag`（MTE2_V / V_MTE2 / V_MTE3 / MTE3_MTE2）串行；跨 chunk 用 MTE3_MTE2 保证 CopyOut 完成后 CopyIn 才覆写同槽缓冲；query/key 两侧串行复用同一组事件 ID。
- **放弃的候选**：`Reg::Gather`/`Scatter` + `Arange`/`Xor` 构造 stride-2 索引实现 rotate_half——索引寄存器算术链长、可读性差，且 Interleave 语义已覆盖；NDDMA 2D strided 表达奇偶交错——UB 侧 stride 以 32B 块为单位，element-stride=2 无法表达。

## 系统信息

- 工具链：CANN 9.2.0（`set_env.sh` 环境），宿主 GCC 13.3.0，cmake 3.20.5；opc/opbuild 构建，`npu_op_kernel_sources` 单 kernel 源注册。
- 目标平台：ascend950（`ASCEND_COMPUTE_UNIT ascend950`，Ascend950PR），向量核 dav-3510，`GetVecLen()=256B`（fp32 64 元素/寄存器）。
- 目录布局：`operators/<Op>_package/{op_kernel,op_host,tests/{ut,aclnn,whitebox,blackbox}}`；ST 数据流 `gen_branch1_cases.py`（调独立 torch oracle golden.py 离线产 .bin）→ `branch1_cases.manifest` → `test_aclnn.cpp`（仅加载比对）→ `run.sh`（build + regenerate + run 双分支 manifest）。
- 相关文件：`op_kernel/<op>_kernel.cpp`（MulAddVF / InterleaveVF / Branch0Kernel / Branch1Kernel / 入口分发）、`op_kernel/…_tiling.h`（TilingData POD）、`op_host/…_tiling_host.{h,cpp}`（ChooseTilingKey/ComputeTiling/ComputeBranch1Tiling）、`op_host/…_host.cpp`（TilingFunc，SetTilingKey(0ULL) 单入口范式）、`tests/aclnn/{gen_branch1_cases.py,test_aclnn.cpp,run.sh}`、torch oracle golden 脚本。

## 完整代码

以下为终版关键实现，verbatim 取自事件流中的最终写入/编辑（fp32 与 fp16/bf16 双路径；缓冲声明、Init 布局预算见上文"根因/解决配方"叙述）。

融合乘加（branch-0/branch-1 共用，dav-3510 连续 Vector 计算链模板）：

```cpp
template <typename T>
__simd_vf__ inline void MulAddVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
```

小 D padded 行的 per-row 交错重组：

```cpp
// rotate_half Step C：Interleave src0/src1（各 H 元素/行）→ dst（D=2H 元素/行，交错）。
//   dst[2i]=src0[i], dst[2i+1]=src1[i]。每 repeat 处理 oneRepeatSize 个 src（src0/src1 各）→
//   2*oneRepeatSize 个 dst；dstReg0=交错低半、dstReg1=交错高半，分别 StoreAlign（掩码尾块）。
//   全程 __simd_vf__（dav-3510 连续 Vector 计算链须 __simd_vf__ + asc_vf_call）。
template <typename T>
__simd_vf__ inline void InterleaveVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t dCount, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg0, dstReg1;
    AscendC::Reg::MaskReg mask0, mask1;
    AscendC::Reg::AddrReg aSrc, aDst0, aDst1;
    uint32_t dstCount = dCount;  // UpdateMask 按引用递减
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aSrc = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aSrc);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aSrc);
        AscendC::Reg::Interleave(dstReg0, dstReg1, srcReg0, srcReg1);
        // dstReg0/dstReg1 各持交错结果的低/高半（各 oneRepeatSize 元素），分别 StoreAlign 到
        // dst 的 [2i*N .. +N) 与 [2i*N+N .. +2N)。用独立 AddrReg/MaskReg 变量避免复用 race。
        aDst0 = AscendC::Reg::CreateAddrReg<T>(static_cast<uint16_t>(2u * i), oneRepeatSize);
        mask0 = AscendC::Reg::UpdateMask<T>(dstCount);
        AscendC::Reg::StoreAlign(dstAddr, dstReg0, aDst0, mask0);
        aDst1 = AscendC::Reg::CreateAddrReg<T>(static_cast<uint16_t>(2u * i + 1u), oneRepeatSize);
        mask1 = AscendC::Reg::UpdateMask<T>(dstCount);
        AscendC::Reg::StoreAlign(dstAddr, dstReg1, aDst1, mask1);
    }
}
```

计算主体（终版 `ComputeRotary`，含 fp32 与 fp16/bf16 双路径）：

```cpp
    // V: rotate_half(interleaved DeInterleave+Muls+Interleave) + VF MulAdd 计算链。
    __aicore__ inline void ComputeRotary(uint32_t cnt)
    {
        const uint32_t D = td_->D;
        const uint32_t H = H_;
        const uint32_t nHpadded = cnt * paddedH_;  // whole-chunk padded H 元素数（含每行 padding 段）
        const uint32_t nDpadded = cnt * paddedD_;

        if constexpr (kIsFp32) {
            auto q   = bufQ_.Get<ComputeT>();
            auto even= bufEven_.Get<ComputeT>();
            auto odd = bufOdd_.Get<ComputeT>();
            auto c   = bufCos_.Get<ComputeT>();
            auto s   = bufSin_.Get<ComputeT>();
            auto ns  = bufNegSin_.Get<ComputeT>();
            auto oE  = bufOutEven_.Get<ComputeT>();
            auto oO  = bufOutOdd_.Get<ComputeT>();

            // ① DeInterleave per-row: even[row]=q[2::2], odd[row]=q[1::2]（srcCount=D 恒偶）
            for (uint32_t r = 0; r < cnt; ++r) {
                AscendC::DeInterleave(even[static_cast<int64_t>(r) * paddedH_],
                                      odd[static_cast<int64_t>(r) * paddedH_],
                                      q[static_cast<int64_t>(r) * paddedD_],
                                      static_cast<int32_t>(D));
            }
            // ② Muls: -sin（取负，folded 进 outEven 的 -odd·sin；等价 rotate_half 对 odd 取负）
            AscendC::Muls(ns, s, ComputeT(-1.0), static_cast<int32_t>(nHpadded));
            // outEven = even·c − odd·s = even·c + odd·(−s)  (VF MulAdd 融合，whole-chunk on paddedH)
            AscendC::Mul(oE, even, c, static_cast<int32_t>(nHpadded));
            uint16_t rep = static_cast<uint16_t>(CeilDivU32L(nHpadded, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)oE.GetPhyAddr(), (__ubuf__ ComputeT*)odd.GetPhyAddr(),
                (__ubuf__ ComputeT*)ns.GetPhyAddr(), nHpadded, kVecLen, rep);
            // outOdd = odd·c + even·s  (VF MulAdd 融合)
            AscendC::Mul(oO, odd, c, static_cast<int32_t>(nHpadded));
            rep = static_cast<uint16_t>(CeilDivU32L(nHpadded, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)oO.GetPhyAddr(), (__ubuf__ ComputeT*)even.GetPhyAddr(),
                (__ubuf__ ComputeT*)s.GetPhyAddr(), nHpadded, kVecLen, rep);
            // ③ Interleave: outFull[2i]=outEven[i], outFull[2i+1]=outOdd[i]（复用 bufQ_ 槽位）
            //    paddedH==H（H 为 16 倍数 → outEven/outOdd flat 无行间 padding）→ whole-chunk
            //    AscendC::Interleave 双输出；否则 paddedH>H → per-row InterleaveVF（每行 D 元素）。
            if (paddedH_ == H_) {
                uint32_t padCount = PadUp(static_cast<uint32_t>(cnt) * H_, 8u);
                AscendC::Interleave(q, q[padCount], oE, oO, static_cast<int32_t>(padCount));
            } else {
                uint16_t repI = static_cast<uint16_t>(CeilDivU32L(H_, kVecLen));
                for (uint32_t r = 0; r < cnt; ++r) {
                    asc_vf_call<InterleaveVF<ComputeT>>(
                        (__ubuf__ ComputeT*)q.GetPhyAddr() + static_cast<int64_t>(r) * paddedD_,
                        (__ubuf__ ComputeT*)oE.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                        (__ubuf__ ComputeT*)oO.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                        D, kVecLen, repI);
                }
            }
            // outFull 就位（= q / bufQ_），供 CopyOut。
        } else {
            auto qIn = bufQIn_.Get<T>();
            auto qFp = bufQFp_.Get<ComputeT>();
            auto cIn = bufCosIn_.Get<T>();
            auto sIn = bufSinIn_.Get<T>();
            auto cFp = bufCos_.Get<ComputeT>();
            auto sFp = bufSin_.Get<ComputeT>();
            auto even= bufEven_.Get<ComputeT>();
            auto odd = bufOdd_.Get<ComputeT>();
            auto ns  = bufNegSin_.Get<ComputeT>();
            auto oE  = bufOutEven_.Get<ComputeT>();
            auto oO  = bufOutOdd_.Get<ComputeT>();

            // 升精度 Cast（fp16/bf16 → fp32，CAST_NONE 无损；whole-chunk on padded）
            AscendC::Cast(qFp, qIn, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(nDpadded));
            AscendC::Cast(cFp, cIn, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(nHpadded));
            AscendC::Cast(sFp, sIn, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(nHpadded));
            // ① DeInterleave per-row ② Muls ③ Mul+MulAdd（同 fp32，在 fp32 域）
            for (uint32_t r = 0; r < cnt; ++r) {
                AscendC::DeInterleave(even[static_cast<int64_t>(r) * paddedH_],
                                      odd[static_cast<int64_t>(r) * paddedH_],
                                      qFp[static_cast<int64_t>(r) * paddedD_],
                                      static_cast<int32_t>(D));
            }
            AscendC::Muls(ns, sFp, ComputeT(-1.0), static_cast<int32_t>(nHpadded));
            AscendC::Mul(oE, even, cFp, static_cast<int32_t>(nHpadded));
            uint16_t rep = static_cast<uint16_t>(CeilDivU32L(nHpadded, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)oE.GetPhyAddr(), (__ubuf__ ComputeT*)odd.GetPhyAddr(),
                (__ubuf__ ComputeT*)ns.GetPhyAddr(), nHpadded, kVecLen, rep);
            AscendC::Mul(oO, odd, cFp, static_cast<int32_t>(nHpadded));
            rep = static_cast<uint16_t>(CeilDivU32L(nHpadded, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)oO.GetPhyAddr(), (__ubuf__ ComputeT*)even.GetPhyAddr(),
                (__ubuf__ ComputeT*)sFp.GetPhyAddr(), nHpadded, kVecLen, rep);
            // ③ Interleave → outFullFp（复用 bufQFp_ 槽位）；flat 用 whole-chunk 两输出，否则 per-row InterleaveVF
            if (paddedH_ == H_) {
                uint32_t padCount = PadUp(static_cast<uint32_t>(cnt) * H_, 8u);
                AscendC::Interleave(qFp, qFp[padCount], oE, oO, static_cast<int32_t>(padCount));
            } else {
                uint16_t repI = static_cast<uint16_t>(CeilDivU32L(H_, kVecLen));
                for (uint32_t r = 0; r < cnt; ++r) {
                    asc_vf_call<InterleaveVF<ComputeT>>(
                        (__ubuf__ ComputeT*)qFp.GetPhyAddr() + static_cast<int64_t>(r) * paddedD_,
                        (__ubuf__ ComputeT*)oE.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                        (__ubuf__ ComputeT*)oO.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                        D, kVecLen, repI);
                }
            }
            // 回转原 dtype（CAST_RINT round-to-nearest-even），复用 bufQIn_ 槽位作 outFullIn
            AscendC::Cast(qIn, qFp, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(nDpadded));
            // outFullIn 就位（= qIn / bufQIn_），供 CopyOut。
        }
    }
```

多核切分与 chunk 循环（Process，行 = B·S·N token-head，末核吃尾块）：

```cpp
    __aicore__ inline void Process()
    {
        const uint32_t coreId = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockDim = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t perCoreRows = td_->perCoreRows;
        const uint32_t totalRows = td_->totalRows;
        if (totalRows == 0) return;

        uint32_t start, end;
        if (coreId + 1 < blockDim) { start = coreId * perCoreRows; end = start + perCoreRows; }
        else if (coreId + 1 == blockDim) { start = (blockDim - 1) * perCoreRows; end = totalRows; }
        else return;
        if (start >= end) return;

        int32_t evMTE2_V    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evV_MTE2    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        int32_t evV_MTE3    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        bool isFirst = true;
        for (uint32_t r = start; r < end; r += myUbRows_) {
            uint32_t cnt = myUbRows_;
            if (r + cnt > end) cnt = end - r;
            bool isLast = (r + cnt >= end);
            ProcessChunk(r, cnt, evMTE2_V, evV_MTE2, evV_MTE3, evMTE3_MTE2, isFirst, isLast);
            isFirst = false;
        }
    }
```

（`ProcessSide`/`ProcessChunk` 的 MTE2→V→MTE3 `SetFlag/WaitFlag` 串行、per-row `CopyInQ`/`CopyInCosSin`/`CopyOut` 与 Init 的 paddedH/paddedD 预算代码均为终版 verbatim，机制见上文叙述。）

## 性能影响

- 本轨迹只做正确性验证（ST 精度全绿），未跑 profiling；无性能数据可引用。设计上的取舍（未实测验证）：query/key 两侧串行 + 单 slot 复用牺牲部分搬运并发换取 UB 峰值（P+1 预算内），双缓冲化是显然的下一步优化方向（未尝试）。

## 预防

- 移植本配方时保住三个不变量：padded 行 32B 对齐（kRowAlign=16 元素）、flat 判定用 `paddedH_ == H_`、`AscendC::Interleave` 的 count 偶数且第二输出基址 32B 对齐（PadUp 8）。
- 测试集必须同时覆盖：三 dtype、D=2（最小对）、D=6（非对齐 D）、D=128（整寄存器行）、大 shape 多核、空张量、NaN/Inf/全零——本配方的三个坑（对齐/两半互换/广播）各由其中一类用例暴露。

## 验证信号

以下信号来自该分支 kernel 的实现会话：迭代三轮（入口→对齐→重组）后全量 aclnn ST 与 UT 回归。

- ST 精度：branch-1 12 用例（三 dtype + B 广播 + D=2 + S=0 空张量 + 大 shape fp16 + D=6 非对齐 + NaN + 全零 + Inf + fp16 大值）numeric 11/11 PASSED + returns_empty 1/1 OK = 12/12，容差 fp32 1.22e-4 / fp16 9.77e-4 / bf16 7.81e-3（来源平台：950；证据等级：tool_observed）。
- 特殊值语义：NaN 输入产出 NaN（produces_nan）、全零产出全零（equals_zero）、Inf 按 oracle 传播，均通过（来源平台：950；证据等级：tool_observed）。
- 回归：TilingUT 57/57（公共 33 + branch-0 12 + branch-1 12）；既有 half 分支（tilingKey=0）用例结果不因本分支改动变化（来源平台：950；证据等级：tool_observed）。
- 计算链合规：`grep -cE '__simd_vf__|asc_vf_call' op_kernel/…_kernel.cpp` = 18 ≥ 1（MulAddVF/InterleaveVF 定义 + asc_vf_call 调用点均为真实代码）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 依赖 cos/sin 的 Dh→D tile 性质（c[2i]=c[2i+1]）与 cosB=1 广播；cos.ndim=3（[B,S,D/2]）或 cos 带独立 B 维的分支需改 CopyIn 索引，本卡代码不直接适用。
- fp16/bf16 升 fp32 计算使 UB 峰值比 native dtype 高约一倍（预算公式见对齐相关卡）；超小 UB 场景需重算 myUbRows。
- dav-3510 的 `Reg::Interleave` 双输出手写重组在整寄存器行上不可靠（同族坑见 DeInterleave 两路写反卡）；其他架构未验证，推断 `AscendC::Interleave` UB 级用法可迁移、`InterleaveVF` 需按目标架构重验。
- interleaved 链相对 half 链的白盒精度劣化现象（含 D=88 严格分界证据）见库内 RoPE interleaved 白盒精度回归卡；精度敏感场景应评估回落 half 路线。

**未决项**：

- 双缓冲/流水化（MTE2 与 V 重叠）未尝试；性能画像未做。
- `AscendC::DeInterleave` 对 srcCount 非偶、小于寄存器宽度等边角的行为未单独压测（本工程 srcCount=D 恒偶）。
## 坏实践

- 用 DataCopy 2D strided 表达奇偶交错（UB stride 以 32B 块为单位，element-stride=2 表达不了）。
- fp16 域直接乘加（精度损失）或 CAST 回转用非 RINT 模式（与 oracle 舍入不一致）。
- 把 `AscendC::Interleave` 当单输出用（它是 dst0/dst1 双输出、源各 count 元素，语义与直觉的"单流交错"不同）。

## 误诊

- 无（本卡为实现配方；迭代过程中的独立误诊——tilingKey 路由、UB 对齐等——均已各自闭环成库内对应卡）。
