---
type: "Runbook"
title: rank>5 广播缺外层 Flat loop 致大面积失配
description: 广播 rank 6–8 且 split 轴落在外层维时，单次 NDDMA 只覆盖最内 5 维，缺外层 Flat loop 多段搬运致输出大面积失配；补 seg_coord 分解与逐段 CopyIn/VF/CopyOut。
tags: [task.implementation, topic.data_movement.data_copy, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# rank>5 广播缺外层 Flat loop 致大面积失配

## 现象

在为 `softplus_v2_grad`（softplus 反向梯度算子，支持 0–8D 广播、fp32/fp16/bf16 三 dtype 及混合组合）实现高 rank 分支 kernel 后，运行白盒 aclnn ST（62 条 RANK_8 用例、16 类，覆盖主块/尾块/非对齐/广播/特殊值）时出现 20 条失败：

- `R8-C05_outer_loop_2d_r7 [2,3,4,5,6,7,8]`（rank=7，外循环 2 维）
- `R8-C06_outer_loop_3d_r8 [2,2,2,3,4,5,6,7]`（rank=8，外循环 3 维）
- `R8-C09_broadcast_6d [1,1,3,4,5,6]ot[2,2,3,4,5,6]` 9 条（rank=6，全部 dtype 组合）
- `R8-C10_broadcast_8d [1,1,1,3,4,5,6,7]ot[2,2,2,3,4,5,6,7]` 9 条（rank=8，全部 dtype 组合）

失败形态是**大面积 MISMATCH**而非小数值偏差，例如：

```
[MISMATCH] idx=2521 actual=-3.52692e-08 expected=-0.0269348
[MISMATCH] idx=2522 actual=-4.13359e-08 expected=0.0876916
[FAIL] R8-C10_broadcast_8d [1,1,1,3,4,5,6,7]ot[2,2,2,3,4,5,6,7] — 15120/20160 mismatches (first at idx 2520)
```

`actual` 呈 ~1e-8 量级的未初始化小垃圾值。同时 rank≤5 用例、rank=8 主块/尾块用例全部通过，低 rank 分支回归 48/48 不受影响——失败精确集中在"外层维需要多次 NDDMA 段搬运"的场景。当时 kernel 是 `SoftplusV2GradKernel<TGrad,TFeat,TOut,RANK>` 共享模板，`Process()` 内每个 tile 只发一次 CopyIn/CopyOut。

## 适用场景

- 工程形态：算子仓自定义算子 kernel 实现（`op_kernel/*.h/.cpp`）+ aclnn 白盒 ST 验证；本卡来自 kernel 实现阶段的精度失配修复。
- API 路径：`AscendC::DataCopy<T, ND, cfg>`（`MultiCopyParams` NDDMA 多维搬运，ND≤5）、`AscendC::DataCopyPad`、`__simd_vf__` + `asc_vf_call` 向量链、`SetFlag/WaitFlag<HardEvent::MTE3_MTE2>` 段间同步。
- 验证过的平台与 CANN 版本：Ascend 950（dav-3510）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——所有 rank>5 且用 NDDMA 做广播搬运的 Vector kernel。
- 路线优先级：`recommended_fastpath`（外层循环是 rank>5 的必经实现，无更简单替代）。
- 明确不适用/未验证：rank≤5 时单次 NDDMA 即覆盖全维（`split.axis >= RANK-ND` 恒成立），无需外层循环；Cube/SIMT 路径未验证。

## 根因

`AscendC::DataCopy<T, ND, cfg>` 的 `MultiCopyParams::loopInfo` 一次最多表达 **ND=5 个循环维**；当广播 `RANK>5` 且 `split.axis < RANK-ND` 时，一个 tile 的元素（`count = a_i_seg × inner_count`）横跨最内 5 维之外的**外层维**（`j ∈ [0, RANK-ND)`），而原实现沿袭低 rank 分支"ND=min(RANK,5) 单次覆盖全维、无外层循环"的假设，`Init()` 里甚至留着 `(void)nddma_dims;  // RANK=4 → always ≤ ND, no outer loop` 的注释——外层维对应的数据**从未被搬入 UB**，计算结果也**从未写回 GM**，输出区残留未初始化内存，故 MISMATCH 的 actual 是垃圾小值、且失配比例恰为"外层维元素占比"。

机制链（以 RANK=8、ND=5 为例）：NDDMA 单段覆盖 `max_bro_shape[3..7]`（最内 5 维，元素数 `nd_elems = ∏_{d=RANK-ND}^{RANK-1} max_bro_shape[d]`）；外层维 `max_bro_shape[0..2]` 需要额外 `outer_iters = count / nd_elems` 段搬运；缺这段循环时只有第 0 段被处理，其余 `outer_iters-1` 段输出位置全部漏写。

同平台 NDDMA 描述符 ≤5 维约束的另一独立 manifestation（高维逐元素算子 accum 输出错、修复后 var 仍错的分野）见[RANK大于5时NDDMA漏维只解释accum错误卡](rank_5_nddma_accum_var.md)——两卡机制同源（5 维描述符上限 + 外层循环缺失），失败形态不同。

## 解决配方

修改 `op_kernel/softplus_v2_grad_kernel.h` 三处（before/after 均为函数级完整代码，verbatim 取自事件流）。

### 1. Init()：NDDMA 段数推导（before → after）

before（低 rank 假设，`nddma_dims` 算完即丢弃）：

```cpp
        // NDDMA 参数 (ND=min(4,5)=4, 单次覆盖全维, nddmaOuterIters_=1)
        const int64_t* dstShape = td->max_bro_shape;
        int64_t k = td->split.axis;
        int64_t nddma_dims = (RANK - k <= ND) ? (RANK - k) : ND;
```

（同函数尾部）：

```cpp
        (void)nddma_dims;  // RANK=4 → always ≤ ND, no outer loop
    }
```

after（成员化 + 外层 Flat 维乘积）：

```cpp
        // NDDMA 参数 (ND=min(RANK,5), 外层 Flat loop 多段搬运)
        const int64_t* dstShape = td->max_bro_shape;
        int64_t k = td->split.axis;
        nddma_dims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        // nddmaOuterIters_ = ∏_{j=0}^{RANK-ND-1} max_bro_shape[j] (外层 Flat 维乘积)
        nddmaOuterIters_ = 1;
        for (int64_t j = 0; j < RANK - ND; j++) {
            nddmaOuterIters_ *= dstShape[j];
        }
```

（同函数尾部）：

```cpp
        (void)nddmaOuterIters_;  // 段数实际由 Process 据 a_i_seg 动态计算 (主块/尾块不同)
    }
```

成员声明同步增加：

```cpp
    AscendC::MultiCopyParams<TGrad, static_cast<uint8_t>(ND)> nddmaGrad_;
    AscendC::MultiCopyParams<TFeat, static_cast<uint8_t>(ND)> nddmaFeat_;
    int64_t nddmaOuterIters_;  // NDDMA 外循环次数 (∏_{j=0}^{RANK-ND-1} max_bro_shape[j])
    int64_t nddma_dims_;       // NDDMA 实际维度数 (RANK-split.axis 与 ND 取小)

public:
```

### 2. Process()：tile 内加外层 Flat loop（before → after）

before（单段处理，coord 只含 tile 坐标）：

```cpp
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
```

after（`nd_elems` 预计算 + `outer_iters` 分档 + `seg_coord` 逐段分解 + 段间同步改为 `need_wait` 标志；VF/Cast 的元素数改用 `seg_elems`）：

```cpp
        // 最内 ND 维元素数 (NDDMA 单次覆盖)
        int64_t nd_elems = 1;
        if constexpr (RANK > ND) {
            for (int64_t d = RANK - ND; d < RANK; d++)
                nd_elems *= td_->max_bro_shape[d];
        }

        int64_t coord[8] = {};
        int64_t seg_coord[8] = {};
        bool need_wait = false;  // 段间 MTE3_MTE2 同步标志 (段间串行)
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg = GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o,
                                              td_->split.a_i, td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(flat, td_->max_bro_shape, RANK,
                                 td_->split.axis, td_->split.a_i, td_->split.a_o, coord);

            // 外层 Flat loop 段数与单段元素数
            //   split.axis >= RANK-ND → NDDMA 单次覆盖整个 tile (seg=count, iters=1)
            //   split.axis <  RANK-ND → NDDMA 单次覆盖最内 ND 维, 外层 Flat loop 多段搬运
            int64_t seg_elems, outer_iters;
            if constexpr (RANK > ND) {
                if (td_->split.axis < RANK - ND) {
                    seg_elems = nd_elems;
                    outer_iters = count / seg_elems;
                } else {
                    seg_elems = count;
                    outer_iters = 1;
                }
            } else {
                seg_elems = count;
                outer_iters = 1;
            }

            for (int64_t outer = 0; outer < outer_iters; outer++) {
                // 计算 seg_coord: 在 coord 基础上分解 outer 到 [split.axis, RANK-ND) 范围
                //   split.axis 维: [coord[axis], coord[axis]+a_i_seg) (tile 内分块)
                //   (split.axis, RANK-ND) 维: [0, max_bro_shape[d]) (coord 未编码)
                for (int64_t d = 0; d < RANK; d++) seg_coord[d] = coord[d];
                if (outer_iters > 1) {
                    int64_t rem = outer;
                    seg_coord[td_->split.axis] += (rem % a_i_seg);
                    rem /= a_i_seg;
                    for (int64_t d = td_->split.axis + 1; d < RANK - ND; d++) {
                        seg_coord[d] = rem % td_->max_bro_shape[d];
                        rem /= td_->max_bro_shape[d];
                    }
                }

                // 上段 CopyOut(MTE3) 结束 → 本段 CopyIn(MTE2) 可以开始
                if (need_wait) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
                need_wait = true;

                // ── Phase 1: CopyIn input_gradients (g) → B2 (FP32) ──
                // B0 = staging for non-FP32; B2 = FP32 grad
                if constexpr (Spv2gIsFp32<TGrad>::value) {
                    // FP32 直读 → B2
                    CopyInOne<TGrad>(B2, a_i_seg, seg_coord, 0, gmGrad_, nddmaGrad_);
                    // MTE2 wrote B2; need MTE2_V before VF reads B2 → defer sync
                } else {
                    // FP16/BF16: GM→B0(staging) → Cast→B2(FP32)
                    CopyInOne<TGrad>(B0, a_i_seg, seg_coord, 0, gmGrad_, nddmaGrad_);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                    AscendC::Cast(buf_[B2].Get<float>(), buf_[B0].Get<TGrad>(),
                                  AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(seg_elems));
                    // B0 free after Cast; need V_MTE2 before MTE2 reuses B0 (feat staging)
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
                }

                // ── Phase 2: CopyIn input_features (x) → B1 (FP32) ──
                if constexpr (Spv2gIsFp32<TFeat>::value) {
                    // FP32 直读 → B1
                    CopyInOne<TFeat>(B1, a_i_seg, seg_coord, 1, gmFeat_, nddmaFeat_);
                } else {
                    // FP16/BF16: GM→B0(staging, reuse) → Cast→B1(FP32)
                    CopyInOne<TFeat>(B0, a_i_seg, seg_coord, 1, gmFeat_, nddmaFeat_);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                    AscendC::Cast(buf_[B1].Get<float>(), buf_[B0].Get<TFeat>(),
                                  AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(seg_elems));
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
                    uint16_t rep = AscendC::CeilDivision(static_cast<uint32_t>(seg_elems), VL_F32);
                    asc_vf_call<SoftplusV2GradVF<float>>(
                        (__ubuf__ float*)buf_[B3].Get<float>().GetPhyAddr(),   // dst
                        (__ubuf__ float*)buf_[B2].Get<float>().GetPhyAddr(),   // src0 = g
                        (__ubuf__ float*)buf_[B1].Get<float>().GetPhyAddr(),   // src1 = x
                        static_cast<uint32_t>(seg_elems), VL_F32, rep,
                        td_->beta, td_->threshold);
                }

                // ── Phase 4: CopyOut result → GM ──
                if constexpr (Spv2gIsFp32<TOut>::value) {
                    // FP32 直写: B3 → GM
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                    CopyOutOne(B3, seg_coord, seg_elems);
                } else {
                    // FP16/BF16: Cast B3(FP32)→B0(TOut), then DataCopyPad B0→GM
                    // B0 is free (staging done in Phase 1/2)
                    AscendC::Cast(buf_[B0].Get<TOut>(), buf_[B3].Get<float>(),
                                  AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(seg_elems));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                    CopyOutOne(B0, seg_coord, seg_elems);
                }

                // 下段 CopyIn 需等本段 CopyOut 完成 (buffer 复用, 段间串行)
                bool is_last_seg = (flat == end - 1) && (outer == outer_iters - 1);
                if (!is_last_seg)
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
            }
        }
    }
```

### 3. CopyOutOne()：按段搬出（before → after）

before（按整个 tile 连续块写出，会把未填充的缓冲一并写出去）：

```cpp
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
```

after（接收本段坐标与元素数，逐段写出）：

```cpp
    // CopyOut: buf_[slot] → GM (DataCopyPad, 无 32B 对齐要求)
    //   seg_coord: 本段外层 Flat loop 坐标 (逐段搬出)
    //   seg_elems: 本段元素数 (单段 NDDMA 切片元素数)
    __aicore__ inline void CopyOutOne(int slot, const int64_t* seg_coord, int64_t seg_elems)
    {
        int64_t off = CalcOutputOffset(seg_coord, td_->output_strides[0], RANK);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen   = static_cast<uint32_t>(seg_elems * sizeof(TOut));
        extParams.srcStride  = 0;
        extParams.dstStride  = 0;
        AscendC::DataCopyPad(gmOut_[off], buf_[slot].Get<TOut>(), extParams);
    }
```

`CopyInOne` **无需修改**：`split.axis < RANK-ND` 时其 NDDMA 维度索引 `k_nd = RANK-1-split.axis ≥ ND`，`loopSize` 保持 baseParams 的最内 5 维完整值，GM 偏移由传入的 `seg_coord` 决定——正好每段搬一个最内 5 维切片。

## 设计决策

- **为何按 `split.axis` 与 `RANK-ND` 的关系分档**：`split.axis ≥ RANK-ND` 时 NDDMA 覆盖的 5 维包含 split 轴，单段即覆盖整个 tile（`seg=count, iters=1`），走原路径零开销；只有 `split.axis < RANK-ND`（split 轴本身落在外层维）才进入多段。这保证低 rank 与主块场景行为逐位不变（回归 48/48 佐证）。
- **段数为何在 Process 动态算而非只用 Init 的 `nddmaOuterIters_`**：主块与尾块的 `a_i_seg` 不同（`GetUBSplitRange` 对尾块返回 `a_i_tail`），每 tile 的 `count` 随之变化，段数必须是 `count / seg_elems` 逐 tile 计算；Init 的乘积只作推导核对（`(void)` 保留）。
- **同步从 `flat != start` 改为 `need_wait` 标志**：段间（outer 迭代之间）也必须等上一段 CopyOut 完成才能复用 B0–B3 缓冲，原 tile 间判断不再覆盖段间；`is_last_seg` 保证最后一段不设多余 flag。
- **放弃的候选**：无（外层循环是该数据布局下唯一正确解；替代方案是把 ND 扩到 8，但 `MultiCopyParams` 模板维度上限为 5，不可行）。
- 当前方案路线档位：`recommended_fastpath`。

## 系统信息

- 工具链：CANN 9.1.0（CANN 安装根指向本机安装目录），`operators/build.sh <package>` 构建算子包。
- 目标平台：Ascend 950（dav-3510，构建产物 `ai_core/tbe/kernel/ascend950/`）。
- 相关文件：`op_kernel/softplus_v2_grad_kernel.h`（唯一改动文件）、`op_kernel/softplus_v2_grad_kernel.cpp`、`tests/aclnn/`（白盒 ST 驱动）。
- 验证命令：`cd operators/softplus_v2_grad_package/tests/aclnn && bash run.sh`。

## 验证信号

来源说明：本卡信号来自该算子 RANK_8 分支实现后的白盒 ST 修复闭环（修复前 20 失败 → 修复后全通过），全部为同一开发会话内的实测输出。

- 修复前：`RANK_8 summary: 42/62 PASSED, 20 FAILED`，`R8-C10_broadcast_8d — 15120/20160 mismatches`，`actual=-3.52692e-08 expected=-0.0269348`（来源平台：950；证据等级：tool_observed）。
- 修复后编译：`./build.sh softplus_v2_grad_package` 退出码 0（CPack 产出 custom_opp 包）（来源平台：950；证据等级：tool_observed）。
- 修复后运行：`==== RANK_8 summary: 62/62 PASSED, 0 FAILED ====`；之前失败的 R8-C05/R8-C06/R8-C09×9/R8-C10×9 逐一确认 PASS（来源平台：950；证据等级：tool_observed）。
- 跨分支回归：`==== RANK_4 summary: 48/48 PASSED, 0 FAILED ====`，低 rank 行为未受影响（来源平台：950；证据等级：tool_observed）。
- 修复已作为独立提交落库（仅改 `op_kernel/softplus_v2_grad_kernel.h` 与开发日志）（来源平台：950；证据等级：tool_observed）。

## 预防

- 写 NDDMA 搬运前先核对 `MultiCopyParams` 的维度上限（本平台 5 维）与张量 rank 的关系：`RANK > ND` 必须设计外层循环路径，并在 ST 用例里显式覆盖"split 轴落在外层维"的场景（本例 16 类中专门有 outer_loop_2d/3d 与 broadcast_6d/8d 类）。
- 失败形态速查：**大面积 MISMATCH 且 actual 为 1e-8 量级垃圾值 + 失败集中在特定 shape 类** → 先怀疑"某段数据从未被搬运"，而不是精度策略。

## 失效边界

- 本卡只验证了 `RANK ∈ {6,7,8}`、`ND=5`、Vector 通路（`__simd_vf__`）；`ND` 取其他值（其他平台 NDDMA 维度上限不同）时公式 `outer_iters = count / nd_elems` 需按新 ND 重推，未验证。
- `seg_coord` 分解假设外层维范围 `(split.axis, RANK-ND)` 的坐标未被 `FlatToEffectiveCoord` 编码（该函数只编码 `d < split.axis` 的维），若换用别的坐标编码函数需重推分解范围。
- 950 以外平台未验证，推断可迁移（NDDMA 维度上限机制相同），未实测。

**未决项**：

- `outer_iters = count / seg_elems` 依赖 `count` 恰为 `nd_elems` 整数倍（tile 沿外层维不跨段切割）；若未来 tiling 允许 tile 横跨外层维边界，需改为逐维余数分解，当前 tiling 策略下未触发。
- 段间串行（每段完整走 CopyIn→VF→CopyOut）未做段间双缓冲流水，大外层维数下有性能空间，未测量。
## 坏实践

- 沿袭低 rank 分支的"单次覆盖全维"假设而不为 `RANK > ND` 写专门路径——`(void)nddma_dims; // no outer loop` 这类"暂时不用"的注释就是信号。
- CopyOut 按整 tile 连续块写出（`cnt = a_i_seg * inner_count`），会把缓冲里从未填充的段一起写出去，掩盖"输入没搬全"与"输出写越界"两个问题的边界。

## 误诊

无——本闭环首轮定位即命中搬运缺失（失败用例全部是 outer-loop/broadcast 高 rank 类，且值形态为未初始化垃圾），未走过精度策略/容差方向的弯路。
