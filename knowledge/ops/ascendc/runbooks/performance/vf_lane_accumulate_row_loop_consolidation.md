---
type: "Runbook"
title: "VF收拢与lane累加消除scalar共bound"
description: "向量函数内每 chunk 一次 ReduceSum（单发射）+ 循环携带 Add 链与逐行 VF 调用使 scalar/发射成为共 bound（利用率 0.33、深度 1 流水）；改 lane 级 Add 累加 + 行末一次 ReduceSum、行循环收进单 VF（32→1 调/tile）、常量外提、full/tail 分离，在合并 DMA 快路之上同算子同协议再获约 1.48x（合计 9.16x），精度 18/18+239/239 strict 通过。"
tags: ["task.optimization", "topic.programming_model.execution_model.simd", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# VF收拢与lane累加消除scalar共bound

## 现象

在 Ascend950PR（CANN 9.2.0）上优化融合算子 FusedRmsNormSigmoidMul：tile 级合并 DMA 快路落地后（搬运通路已合并，见 [tile_merged_dma_eliminate_mte2_issue_bound](tile_merged_dma_eliminate_mte2_issue_bound.md)），profiling 指示剩余瓶颈是 scalar 共 bound（利用率 0.33，深度 1 流水）——计算通路的发射/标量开销成为第二瓶颈。当时的向量函数形态（`__simd_vf__` + `asc_vf_call` 调用）：

- `SquareAccVF`（行平方和累加）：每 256 通道 chunk 做一次 `ReduceSum`（单发射指令）得到部分和，再用 `Add(acc, acc, part)` 累加——ReduceSum 单发 + 循环携带单链，行内循环无法双发；
- `GateMulVF`（逐元素 rms·gamma→sigmoid→mul）：签名只收单行（`row` 标量），外层 host 循环逐行调用 `asc_vf_call<GateMulVF>(..., row, n)`——快路每 tile 32 行就是 32 次 VF 调用；`±1.0f` 的 `Duplicate(one, 1.0f, mask)`/`Muls(neg, rms, -1.0f, mask)` 在每个 chunk 循环内重复；`rstd` 的 `DIST_BRC_B32` 广播读也在每 chunk 重复（行内不变）；
- `while (left > 0)` 尾部循环里每个 chunk 重建 `UpdateMask`，全满 mask 无法提到循环外。

在合并 DMA 快路的候选之上实施 VF 收拢（设备侧四个 VF 重写 + host 快路门控微调），构建时先撞上 ccec HWLoop 归纳变量硬错误（嵌套内层循环须 uint16_t，见 [hwloop_induction_variable_uint16](../compilation/hwloop_induction_variable_uint16.md)），修正后一次通过。

## 适用场景

- 工程形态：AscendC device kernel（`__simd_vf__` 向量函数 + `asc_vf_call` 调用形态，Reg-SIMD 编程模型），行归约 + 逐元素后处理的融合算子。
- API 路径：`RegTensor`/`LoadAlign`/`StoreAlign`（LoadDist::DIST_UNPACK_B16 / DIST_BRC_B32 / DIST_FIRST_ELEMENT_B32）、`ReduceSum`、`Add`（MaskMergeMode::MERGING）、`Duplicate`、`UpdateMask`/`CreateMask<MaskPattern::VL1>`、`asc_vf_call`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510 / dav-c310-vec，VECTOR_REG_WIDTH 按 256B fp32 通道），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——行内 reduce + 行级逐元素的双段式行主序算子。
- 路线优先级：recommended_fastpath（本会话终选方案；与合并 DMA 快路叠加）。
- 明确不适用 / 未验证场景：group 路径（超长 R，本卡零修改保留）；行宽远小于向量寄存器宽度时 lane 累加的收益会退化（尾块占比大，未量化）；求和顺序敏感且不容许 lane 并行折叠误差的场景（本卡以精度验收为门禁，见"验证信号"）。

## 根因

计算通路存在三类发射/标量浪费：① 每 chunk 一次 `ReduceSum` 是单发射指令且其后 `Add` 形成循环携带依赖链，行内无法双发，深度 1 流水吃不满；② VF 调用按"行"为粒度（每行一次 `asc_vf_call`），32 行 tile 即 32 次调用发射；③ 循环不变量（±1.0f 常量、行级 rstd 广播、全满 mask、`td_` 热字段）在循环内重复构建。三者叠加使 scalar 利用率仅 0.33。

## 解决配方

**1. `SquareAccVF`：lane 级累加 + 行末一次归约 + full/tail 分离 + 尾块 MERGING 折叠**

修复前（每 chunk 一次 ReduceSum + 循环携带 Add；while 循环逐 chunk 重建 mask）：

```cpp
__simd_vf__ inline void SquareAccVF(__ubuf__ float* sums, __ubuf__ bfloat16_t* xs,
    uint32_t rows, uint32_t valid, uint32_t stride)
{
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> xb;
        RegTensor<float> x, sq, part, acc;
        MaskReg mask;
        MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
        constexpr uint32_t kVl = VECTOR_REG_WIDTH / sizeof(float);
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(acc, sums + row);
            uint32_t left = valid;
            uint32_t off = 0;
            while (left > 0) {
                uint32_t lanes = (left < kVl ? left : kVl);
                uint32_t maskCount = lanes;
                mask = UpdateMask<float>(maskCount);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xs + row * stride + off);
                Cast<float, bfloat16_t, kBf16ToFp32>(x, xb, mask);
                Mul(sq, x, x, mask);
                ReduceSum(part, sq, mask);
                Add(acc, acc, part, oneMask);
                left -= lanes;
                off += lanes;
            }
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sums + row, acc, oneMask);
        }
    }
}
```

修复后（lane 级 Add 累加可双发；行末一次 ReduceSum；全满 mask 循环外建一次；尾块用 `Add<float, MaskMergeMode::MERGING>` 折叠保留 mask 外已累加列）：

```cpp
// chunk 循环内 lane 级 Add 累加（可双发）+ 行末一次 ReduceSum，
// 替代每 chunk 一次 ReduceSum（单发）+ Add(acc, acc, part) 循环携带单链。
// 求和顺序变化（chunk 串行 -> lane 并行折叠）以全部 18 条 L0 精度验收为门禁。
// while(left>0) 改 full/tail 分离（uint16_t 归纳变量 + 全满 mask
// 循环外建一次）；尾块折叠用 MERGING 保留 mask 外已累加列。
__simd_vf__ inline void SquareAccVF(__ubuf__ float* sums, __ubuf__ bfloat16_t* xs,
    uint32_t rows, uint32_t valid, uint32_t stride)
{
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> xb;
        RegTensor<float> x, sq, part, acc, laneSum;
        MaskReg mask;
        MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
        constexpr uint32_t kVl = VECTOR_REG_WIDTH / sizeof(float);
        uint32_t fullLanes = kVl;
        MaskReg fullMask = UpdateMask<float>(fullLanes);
        const uint32_t full = valid / kVl;
        const uint32_t tail = valid % kVl;
        const uint16_t fullLoop = static_cast<uint16_t>(full);
        const uint16_t tailLoop = static_cast<uint16_t>((tail + kVl - 1) / kVl);
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(acc, sums + row);
            Duplicate(laneSum, 0.0f, fullMask);
            const uint32_t rowOff = static_cast<uint32_t>(row) * stride;
            for (uint16_t i = 0; i < fullLoop; ++i) {
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xs + rowOff + i * kVl);
                Cast<float, bfloat16_t, kBf16ToFp32>(x, xb, fullMask);
                Mul(sq, x, x, fullMask);
                Add(laneSum, laneSum, sq, fullMask);
            }
            for (uint16_t i = 0; i < tailLoop; ++i) {
                uint32_t tailLanes = tail;
                mask = UpdateMask<float>(tailLanes);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xs + rowOff + full * kVl);
                Cast<float, bfloat16_t, kBf16ToFp32>(x, xb, mask);
                Mul(sq, x, x, mask);
                Add<float, MaskMergeMode::MERGING>(laneSum, laneSum, sq, mask);
            }
            ReduceSum(part, laneSum, fullMask);
            Add(acc, acc, part, oneMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sums + row, acc, oneMask);
        }
    }
}
```

**2. `GateMulVF`：行循环收进单 VF（签名扩为 rows/count/stride）+ 常量外提 + rstd 广播读外提 + full/tail 分离**

修复前（单行签名；±1.0f 每循环 Duplicate/Muls；rstd 每 chunk 广播读；while 循环重建 mask）：

```cpp
__simd_vf__ inline void GateMulVF(__ubuf__ bfloat16_t* ys,
    __ubuf__ bfloat16_t* xs, __ubuf__ bfloat16_t* gammas,
    __ubuf__ bfloat16_t* others, __ubuf__ float* rstds,
    uint32_t row, uint32_t count)
{
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> xb, gb, ob, rmsb, gateb, yb;
        RegTensor<float> x, gamma, other, rstd, rms, neg, expv, denom;
        RegTensor<float> one, sigmoid, gate, product;
        MaskReg mask;
        constexpr uint32_t kVl = VECTOR_REG_WIDTH / sizeof(float);
        uint32_t left = count;
        uint32_t off = 0;
        while (left > 0) {
            uint32_t lanes = (left < kVl ? left : kVl);
            uint32_t maskCount = lanes;
            mask = UpdateMask<float>(maskCount);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xs + off);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gb, gammas + off);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(ob, others + off);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, rstds + row);
            Cast<float, bfloat16_t, kBf16ToFp32>(x, xb, mask);
            Cast<float, bfloat16_t, kBf16ToFp32>(gamma, gb, mask);
            Mul(rms, x, rstd, mask);
            Mul(rms, rms, gamma, mask);
            Cast<bfloat16_t, float, kFp32ToBf16Rint>(rmsb, rms, mask);
            Cast<float, bfloat16_t, kBf16ToFp32>(rms, rmsb, mask);
            Muls(neg, rms, -1.0f, mask);
            Exp(expv, neg, mask);
            Adds(denom, expv, 1.0f, mask);
            Duplicate(one, 1.0f, mask);
            Div(sigmoid, one, denom, mask);
            Cast<bfloat16_t, float, kFp32ToBf16Rint>(gateb, sigmoid, mask);
            Cast<float, bfloat16_t, kBf16ToFp32>(gate, gateb, mask);
            Cast<float, bfloat16_t, kBf16ToFp32>(other, ob, mask);
            Mul(product, gate, other, mask);
            Cast<bfloat16_t, float, kFp32ToBf16Rint>(yb, product, mask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(ys + off, yb, mask);
            left -= lanes;
            off += lanes;
        }
    }
}
```

修复后（行循环收进 VF，快路每 tile 调用 32→1，一般路径以 rows=1 复用同一函数体；±1.0f 循环外一次 Duplicate；rstd 的 DIST_BRC_B32 广播读外提到行级；rms/gate 的 fp32→bf16→fp32 量化往返顺序保留）：

```cpp
// 行循环收进单 VF（签名扩为 rows/count/stride，参照 SquareAccVF
// 既有行循环形态）；快路每 tile 后处理 VF 调用 32->1，一般路径以 rows=1 复用
// 同一函数体（rstds 由调用方偏移到当前行）。
// ±1.0f 循环外一次 Duplicate；rstd 的 DIST_BRC_B32 广播读外提
// 到行级（行内不变）。rms/gate 的 FP32->bf16->FP32 量化往返顺序保留（链语义）。
// full/tail 分离（全满 mask 循环外建一次）。
// 偏移口径全 uint32：row/stride/i*kVl 均为 uint32 元素偏移。
__simd_vf__ inline void GateMulVF(__ubuf__ bfloat16_t* ys,
    __ubuf__ bfloat16_t* xs, __ubuf__ bfloat16_t* gammas,
    __ubuf__ bfloat16_t* others, __ubuf__ float* rstds,
    uint32_t rows, uint32_t count, uint32_t stride)
{
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> xb, gb, ob, rmsb, gateb, yb;
        RegTensor<float> x, gamma, other, rstd, rms, neg, expv, denom;
        RegTensor<float> one, negOne, sigmoid, gate, product;
        MaskReg mask;
        constexpr uint32_t kVl = VECTOR_REG_WIDTH / sizeof(float);
        uint32_t fullLanes = kVl;
        MaskReg fullMask = UpdateMask<float>(fullLanes);
        const uint32_t full = count / kVl;
        const uint32_t tail = count % kVl;
        const uint16_t fullLoop = static_cast<uint16_t>(full);
        const uint16_t tailLoop = static_cast<uint16_t>((tail + kVl - 1) / kVl);
        Duplicate(one, 1.0f, fullMask);
        Duplicate(negOne, -1.0f, fullMask);
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            const uint32_t rowOff = static_cast<uint32_t>(row) * stride;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, rstds + row);
            for (uint16_t i = 0; i < fullLoop; ++i) {
                const uint32_t off = rowOff + i * kVl;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xs + off);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gb, gammas + i * kVl);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(ob, others + off);
                Cast<float, bfloat16_t, kBf16ToFp32>(x, xb, fullMask);
                Cast<float, bfloat16_t, kBf16ToFp32>(gamma, gb, fullMask);
                Mul(rms, x, rstd, fullMask);
                Mul(rms, rms, gamma, fullMask);
                Cast<bfloat16_t, float, kFp32ToBf16Rint>(rmsb, rms, fullMask);
                Cast<float, bfloat16_t, kBf16ToFp32>(rms, rmsb, fullMask);
                Mul(neg, rms, negOne, fullMask);
                Exp(expv, neg, fullMask);
                Add(denom, expv, one, fullMask);
                Div(sigmoid, one, denom, fullMask);
                Cast<bfloat16_t, float, kFp32ToBf16Rint>(gateb, sigmoid, fullMask);
                Cast<float, bfloat16_t, kBf16ToFp32>(gate, gateb, fullMask);
                Cast<float, bfloat16_t, kBf16ToFp32>(other, ob, fullMask);
                Mul(product, gate, other, fullMask);
                Cast<bfloat16_t, float, kFp32ToBf16Rint>(yb, product, fullMask);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(ys + off, yb, fullMask);
            }
            for (uint16_t i = 0; i < tailLoop; ++i) {
                uint32_t tailLanes = tail;
                mask = UpdateMask<float>(tailLanes);
                const uint32_t off = rowOff + full * kVl;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xs + off);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gb, gammas + full * kVl);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(ob, others + off);
                Cast<float, bfloat16_t, kBf16ToFp32>(x, xb, mask);
                Cast<float, bfloat16_t, kBf16ToFp32>(gamma, gb, mask);
                Mul(rms, x, rstd, mask);
                Mul(rms, rms, gamma, mask);
                Cast<bfloat16_t, float, kFp32ToBf16Rint>(rmsb, rms, mask);
                Cast<float, bfloat16_t, kBf16ToFp32>(rms, rmsb, mask);
                Mul(neg, rms, negOne, mask);
                Exp(expv, neg, mask);
                Add(denom, expv, one, mask);
                Div(sigmoid, one, denom, mask);
                Cast<bfloat16_t, float, kFp32ToBf16Rint>(gateb, sigmoid, mask);
                Cast<float, bfloat16_t, kBf16ToFp32>(gate, gateb, mask);
                Cast<float, bfloat16_t, kBf16ToFp32>(other, ob, mask);
                Mul(product, gate, other, mask);
                Cast<bfloat16_t, float, kFp32ToBf16Rint>(yb, product, mask);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(ys + off, yb, mask);
            }
        }
    }
}
```

（`RstdVF` 同法做 invR/eps 常量外提 + full/tail 分离；`FinalReduceVF` 做 full/tail 分离与 HWLoop uint16_t 归纳变量修正，此处不重复。）

**3. host 侧配套：快路字节精确 UB 方程（无魔数上限）+ 门控**

```cpp
// 快路 au 求解：字节精确 UB 方程 + coreNum 均衡扫描。真实清单（bf16 2B 口径）：
// x/other/y tile 各 au*ra*2B + gamma ra*2B + rstd au*4B（32B 对齐），即
// (3*ra*2+4)*au + ra*2 + 31 <= ubSize（深度 1；kernel 为深度 1 + 依赖解耦
// 事件流，深度 2 在余量内天然可行，不反向约束 au）。择优：优先 A%au==0 的
// 整除解，其内最小化关键路径 ceil(ao/usedC)*au，再偏好 ao%usedC==0（无尾
// tile），最后取更大 au（更大单条 DMA）；tile 单 tensor 字节下限约 2KiB
// （带宽拐点量级）。返回 0 = 无可行候选（回退一般路径）。
int64_t SolveFastAu(int64_t A, int64_t ra, int64_t coreNum, int64_t ubSize)
{
    const int64_t elemBytes = 2;  // bf16
    const int64_t tileMinBytes = 2048;
    if (A <= 0 || ra <= 0 || coreNum <= 0 || ubSize <= 0) return 0;
    const int64_t budget = ubSize - ra * elemBytes - 31;
    const int64_t linear = 3 * ra * elemBytes + 4;
    if (budget < linear) return 0;
    const int64_t auMax = budget / linear;
    const int64_t auMin = (tileMinBytes + ra * elemBytes - 1) / (ra * elemBytes);
    const int64_t lo = std::max<int64_t>(1, auMin);
    const int64_t hi = std::min(auMax, A);
    if (lo > hi) return 0;
    int64_t bestAu = 0;
    int64_t bestCp = 0;
    bool bestDivA = false;
    bool bestDivTiles = false;
    for (int64_t au = lo; au <= hi; ++au) {
        const int64_t ao = PubCeil(A, au);
        const int64_t used = std::min(ao, coreNum);
        const int64_t cp = PubCeil(ao, used) * au;
        const bool divA = (A % au == 0);
        const bool divTiles = (ao % used == 0);
        bool better;
        if (bestAu == 0) {
            better = true;
        } else if (divA != bestDivA) {
            better = divA;
        } else if (cp != bestCp) {
            better = cp < bestCp;
        } else if (divTiles != bestDivTiles) {
            better = divTiles;
        } else {
            better = au > bestAu;
        }
        if (better) {
            bestAu = au;
            bestCp = cp;
            bestDivA = divA;
            bestDivTiles = divTiles;
        }
    }
    return bestAu;
}
```

门控（group 决策先于 au 重解、触发域与基线一致；`pre/post` 保持原公式形态仅 au 数值变化，快路 kernel 不消费 preBufSize）：

```cpp
    // ---- 连续快路门控与 au 重解 ----
    // 条件：base 路径成立(group==false) && other 无广播维(所有 otherShape[i]
    // ==xShape[i]) && R 单 chunk(R<=ru)。本 ABI 下 x/y 恒连续（无描述符输入：
    // 行主序 stride + offset 0），other 无广播时整 tile 连续；非对齐 R 由
    // DataCopyPad 的 UB 32B 行距补齐。group 决策已用原始 au/pre/post 完成，
    // 触发域与基线一致（仅 base 路径内的 au 重解）。
    bool noBroadcast = true;
    for (int i = 0; i < 8; ++i) {
        if (t->otherShape[i] != t->xShape[i]) {
            noBroadcast = false;
            break;
        }
    }
    int64_t auFast = 0;
    if (!group && noBroadcast && R <= ru) {
        auFast = SolveFastAu(A, ra, static_cast<int64_t>(coreNum), static_cast<int64_t>(ubSize));
    }
    const bool arContigFast = auFast > 0;
    if (arContigFast) {
        au = auFast;
        post = PubUp(au * 4, blockSize);
        pre = PubUp(au * ra * 4, blockSize);
        ao = PubCeil(A, au);
        usedC = std::min<int64_t>(ao, coreNum);
    }
```

**4. Init 快路 buffer 清单与 Process 热字段局部化**

```cpp
        if (td_->arContigFast != 0) {
            // 快路真实清单（bf16 2B 字节口径）：
            //   x/other/y tile 各 au*ra*2B + gamma 常驻 ra*2B + rstd au*4B(32B 对齐)
            // 与 host SolveFastAu 的字节方程一致；深度 1 + 依赖解耦事件流。
            const int64_t tileBytes =
                td_->aUbFactor * td_->rUbFactorAlign * static_cast<int64_t>(sizeof(bfloat16_t));
            const int64_t gammaBytes =
                td_->rUbFactorAlign * static_cast<int64_t>(sizeof(bfloat16_t));
            const int64_t rstdBytes =
                (td_->aUbFactor * static_cast<int64_t>(sizeof(float)) + 31) & ~static_cast<int64_t>(31);
            pipe_->InitBuffer(buf0_, tileBytes);
            pipe_->InitBuffer(buf1_, gammaBytes);
            pipe_->InitBuffer(buf2_, tileBytes);
            pipe_->InitBuffer(buf3_, tileBytes);
            pipe_->InitBuffer(cache_, rstdBytes);
        } else {
            pipe_->InitBuffer(buf0_, td_->preBufSize);
            ...
        }
```

`Process`/`ProcessTile`/`ProcessPost` 内把 `td_->` 热字段（usedCoreNum/aBigCoreCnt/aUbFactor/rTotal/rUbFactorAlign 等）拷入局部标量长驻寄存器；一般路径 `GateMulVF` 调用点改传 `rstdBase + row, 1u, n, ...`（rows=1 复用同一函数体）；`CopyLogical/CopyOther` 做 rank 截断（shape 后的 1 维不再迭代）与指针局部化。

## 设计决策

- 为什么 lane 累加可行而必须以精度验收为门禁：求和顺序从"chunk 串行"变为"lane 并行折叠 + 一次跨 lane 归约"，数值分布改变；本算子 bf16 输入 fp32 累加域，18 条 L0（含超长 R 的 group 路径）+ 239 条全集 strict 验收全部通过才保留该改写。
- 为什么尾块用 `Add<float, MaskMergeMode::MERGING>`：MERGING 保留 mask 之外 lane 上已累加的列值，等价于"只把有效列折叠进来"，避免尾块整寄存器覆盖。
- 为什么 GateMulVF 保留 rms/gate 的 fp32→bf16→fp32 量化往返：这是融合前调用链的语义（Cast 算子真实存在），删掉会改变数值结果；性能上两次 Cast 在双发流水内可隐藏。
- 为什么 rstd 的 store→broadcast-load 拆在两个 `asc_vf_call`：跨 VF 调用边界保证 store 对后续 broadcast-load 的可见性（既有同步约定），合并进同一 VF 有可见性风险。
- 为什么快路 UB 方程去掉 auCap：深度 1 + 依赖解耦事件流下，深度 2 的缓冲在 UB 余量内天然可行，不需要用一般路径的 au0 反向约束快路 au；目标 case 两者同解（au=16），但边界 case 上快路可取更大 tile。
- 当前方案路线档位：recommended_fastpath（终选实现，相对合并 DMA 快路候选再获约 1.48x）。

## 系统信息

- 硬件/软件：Ascend950PR NPU 0（AIV=56、UB=253952B），CANN 9.2.0；设备 TU 以 dav-c310-vec 纯设备模式编译（split 构建 + AIVEC magic patch，见 [tpipe_eager_launch_507015_aivec_magic_patch](../compilation/tpipe_eager_launch_507015_aivec_magic_patch.md)）。
- 相关文件（候选隔离副本内）：`csrc/op_kernel/kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h`（4 个 VF 重写 + Init 快路清单 + Process 门控/热字段局部化 + ProcessFastLoop/ProcessTileFast）、`csrc/op_kernel/kernel/arch35/fused_rms_norm_sigmoid_mul_tiling_struct.h`（尾部 arContigFast）、`csrc/op_kernel/fused_rms_norm_sigmoid_mul_local_deps.h`（K_MAX_SHAPE_DIM=0 前置，见 [k_max_shape_dim_zero_before_first_include](../compilation/k_max_shape_dim_zero_before_first_include.md)）、`csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp`（SolveFastAu/门控/死常量清理）；GroupKernel 与基线逐字节一致。
- 同步检查：ascendc-sync-check strict 扫描（event/同步语义改动后必跑）。

## 性能影响

- 相对原始融合基线（同算子、同协议、同设备，参照路线=优化前逐行搬运+旧 VF 实现）：有效 case 几何加速比 9.16x（来源平台：950）。
- 相对合并 DMA 快路候选（参照路线=仅合并 DMA、VF 未收拢）：本卡（合并 DMA + VF 收拢）约 1.48x——成对量化 VF 收拢的边际贡献（来源平台：950）。
- 相对仅 host tiling 重平衡候选（参照路线=host-only）：约 5.12x（来源平台：950）。

## 预防

- `__simd_vf__` 内遇到"每 chunk 归约 + 循环携带累加"形态时优先改 lane 级累加 + 一次归约，并把全满 mask/常量/行级广播读外提；改完必须跑精度全集（求和顺序变了）。
- VF 调用粒度按"tile"而不是"行"设计签名（rows/count/stride 三参形态），一般路径以 rows=1 复用，避免两套函数体。
- 循环内 `Muls`/`Adds` 的立即数先 `Duplicate` 成寄存器再循环外提；`td_` 热字段在循环前局部化。
- 重写向量函数后注意 ccec HWLoop 嵌套循环归纳变量必须 uint16_t（本卡实施时先撞此错误）。

## 验证信号

以下信号来自该融合算子 VF 收拢候选（终选实现）的实施与验收会话（Ascend950PR / CANN 9.2.0）：在合并 DMA 快路之上的隔离副本内实施，构建后逐级验证（smoke→门控探针→同步扫描→边界形状→正式验收→fresh-process 身份），最后与基线同协议采集。scalar 共 bound 0.33 / 深度 1 流水的画像来自同会话前期 profiling 阶段。

- 构建：修正 HWLoop 归纳变量后 5/5 步骤成功、符号与基线一致；VF 预检信号活跃（asc_vf_call 9 / cast_trait 2，相对基线 6/2 增加来自行循环收拢）（来源平台：950；证据等级：tool_observed）。
- smoke 与验收：smoke 18/18 PASS；正式功能验收 18/18 strict PASS，逐 case blockDim/tilingKey 与基线 manifest 完全一致（来源平台：950；证据等级：tool_observed）。
- tiling 门控行为：目标 case blockDim=48（au=16、768=16×48 无尾 tile）tilingKey=0 快路命中；R=4096/A=1 与 A=32 快路、广播/tiny case 回退一般路径（来源平台：950；证据等级：tool_observed）。
- 边界形状等效正确性（numpy golden 逐元素精确比对）：(100,128) au=10 整除偏好（整除解优先于 crit 更小的 au=8）bad=0/12800；(7,130) tile<2KiB 回退一般路径 bad=0/910；(16,130) 非对齐 R=130（ra=144）真实快路 bad=0/2080；(5,17) bad=0/85（来源平台：950；证据等级：tool_observed）。
- 同步语义：ascendc-sync-check strict 扫描 0 findings（来源平台：950；证据等级：tool_observed）。
- 二进制身份：fresh-process 绝对路径直载 libfrnsm.so，kernel 符号与基线一致，目标 case 98304/98304 元素与 golden 精确一致（bad=0）（来源平台：950；证据等级：tool_observed）。
- 功能全集（精度不回归，终选实现上）：L0+L1 共 239 条 kernel 格式用例 strict 回归 239/239 PASS（来源平台：950；证据等级：tool_observed）。
- 性能：目标 case 10 组中位采集（首样本为频率爬升离群、中位数稳健）validation 30/30；有效 case 几何加速比相对原始融合基线 9.16x（同算子同协议成对，远超抖动阈值）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 求和顺序变化（lane 并行折叠）对极长行、强离群值的误差行为未做理论分析——本卡以验收门禁覆盖（18 条 L0 + 239 条全集），更换 dtype/更大行宽时须重跑精度全集。
- 行宽小于 kVl（256 fp32 通道）时 full 段为空、收益退化为尾块路径（未量化）。
- `GateMulVF` 的量化往返（fp32→bf16→fp32）是本算子链语义；其他算子若无该语义可直接省略两次 Cast（更快但数值不同，未在本算子验证）。
- 快路 UB 字节方程与 Init buffer 清单一一对应：改缓冲布局（深度、gamma 常驻与否）必须同步改两处。
- 集成回注册工程时门控条件需补显式连续性判断（注册 ABI 下输入可带描述符，见 [direct_optimized_integrate_back_registered](../compilation/direct_optimized_integrate_back_registered.md)），直调门控不可照抄。
- 同平台 `__simd_vf__` 内部 rep 循环截断与 UpdateMask 递减语义的另两个坑见 [simd_vf_multi_rep_truncation_large_d](../precision/simd_vf_multi_rep_truncation_large_d.md) 与 [updatemask_manual_decrement_required_despite_doc](../precision/updatemask_manual_decrement_required_despite_doc.md)——本卡 full/tail 分离形态（全满 mask 循环外建一次 + 尾块单独循环）同时避开了这两类 while 循环形态风险。

**未决项**：

- lane 折叠求和的数值误差上界未做理论推导（当前以验收门禁替代）；长行 fp16 场景的理论分析待补。
- `SolveFastAu` 的全区间扫描在大 A 下 O(auMax) 的开销上界未评估（host 侧一次性求解，实测无感，但 A 极大时建议改稀疏候选集）。
## 坏实践

- 在 VF 内用 `while (left > 0)` + 每 chunk `UpdateMask`——全满 mask 无法复用、归纳变量类型不受控（易撞 HWLoop uint16_t 规则）。
- 逐行 `asc_vf_call` 而不扩展 VF 签名收拢行循环——发射次数随行数线性增长。
- 循环内 `Muls(x, y, -1.0f, mask)` / `Adds(..., 1.0f, ...)` 立即数形态——每次都占用标量发射槽，应 Duplicate 外提。
- 改完 VF 不跑同步扫描与精度全集就上性能采集——求和顺序与 store→load 可见性都是静默风险。

## 误诊

- 曾认为 fresh-process 身份检查程序段错误是候选二进制问题：实际是测试脚本 ctypes 未设 `restype=c_void_p` 导致 64 位指针截断（测试侧缺陷，非 kernel 问题）；改用带完整 argtypes/restype 的包装后同一二进制通过。
