---
type: "Runbook"
title: "tile级合并DMA消除MTE2逐行发射bound"
description: "逐行小片段（约 256B/行）GM↔UB 搬运使 MTE2 发射/描述符成为瓶颈（利用率 0.78）；base 路径加 host 门控连续快路：tile 级合并 DataCopyPad（单 tensor 单描述符）+ gamma UB 常驻 + x tile 复用 + 合并写回 + 双 MTE2_V 事件依赖解耦，同算子同协议相对原始基线 6.18x，精度 18/18 strict 通过。 另记录本家族的预算常数：arch3510 事件池 QUE_MAX_EVENT=8、每方向独立位图（kernel_macros.h 按 __NPU_ARCH__ 分档，其余架构为 4），是「L1 buffer 数量上限 / event id 还能取几个」的事实锚点。"
tags: ["task.optimization", "topic.data_movement.data_copy", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["950"]
aliases:
  - "事件池池深 每方向独立位图 arch3510"
  - "event id 数量上限 每核每方向"
  - "event pool depth per core per direction"
---

# tile级合并DMA消除MTE2逐行发射bound

## 现象

在 Ascend950PR（AIV=56、UB=253952B，CANN 9.2.0）上做融合算子 FusedRmsNormSigmoidMul 的性能优化：目标 case 是 bf16 的 `(128,6,128)` 类形状（A=768 行、每行 R=128 个元素，行字节 256B），原始融合基线同协议（10 组中位，msprof application / task-based 采集）相对等价的非融合五算子链（RmsNorm→Cast→Sigmoid→Cast→Mul）约 2.4x 慢。前期 profiling 归因（见"验证信号"来源说明与 [mte2_issue_overhead_vs_bandwidth_bound](mte2_issue_overhead_vs_bandwidth_bound.md)）：MTE2 发射/描述符利用率 0.78，瓶颈形态是 base 路径逐行搬运——每行一条 256B 的 5D ND-DMA 小片段，单核 x/other 两路输入就要发射上百条 MTE2 描述符，发射开销吃满搬运通路。当时的代码状态：`ProcessPost` 按"行"为最小搬运单位逐行 `CopyInPost(aBegin + row, r, n)` / `CopyOut(...)`，`ProcessTile` 的 reduce 段也是逐行装载。

## 适用场景

- 工程形态：AscendC device kernel 直调工程（`<<<>>>` launch + split 构建）；优化手段同样适用于注册工程 op_kernel（本会话后续集成已验证）。
- API 路径：`DataCopyPad` + `DataCopyExtParams`/`DataCopyPadExtParams`（Normal 模式行批量搬运与 UB 32B 行距补齐）、`TPipe::FetchEventID(HardEvent::MTE2_V/V_MTE3/MTE3_MTE2)`、`asc_vf_call` 向量函数。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510 / dav-c310-vec），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——行主序连续、按行 reduce + 逐元素后处理的 bf16 融合算子族（行字节对齐或非对齐均可，非对齐走行批量分支）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：输入带广播维（other 形状≠x 形状）或 R 超过单 chunk 上限的 case（host 门控会自动回退原一般路径，本卡快路不接管）；超长 R 走 group 路径的 case（本卡零修改保留）；fp16/fp32 输入未验证（UB 方程与字节口径按 bf16 推导）；行字节数不能被 32 整除且行数极大时行批量分支的描述符数仍随行数线性增长（本卡只保证"每 tile 每 tensor 1 条"在 GM 侧连续成立时的上限）。

## 根因

base 路径以"行"为搬运最小单位：R=128、bf16 时每行仅 256B，一条 `DataCopyPad`/ND-DMA 只搬一行，单核一个 tile（au 行）就要 au×2 条 MTE2 描述符（x/other 两路）+ au 条 MTE3 写回描述符；MTE2 的发射/描述符带宽被小片段请求吃满（profiling 利用率 0.78），而单条搬运的数据量又远低于带宽拐点，导致搬运通路成为首要 bound。

## 解决配方

在 base 路径内新增 host 门控的"连续快路"（不新增 tilingKey，group 路径与一般路径零修改保留为回退）：

**1. host 门控（tiling 侧）**——满足三条件才启用：base 路径成立（`group==false`）、other 无广播维（所有 `otherShape[i]==xShape[i]`，此时整 tile 在 GM 侧连续）、R 单 chunk（`R<=ru`）。TilingData 尾部追加 `int32_t arContigFast` 门控字段（launch ABI 不变，host/device 共享头文件两 TU 同步重编译）：

```cpp
    int64_t xStorageOffset;                  // x 首逻辑元素的 storage element offset
    int64_t gammaStorageOffset;              // gamma 首逻辑元素的 storage element offset
    int64_t otherStorageOffset;              // other 首逻辑元素的 storage element offset
    // ---- 尾部追加（快路门控，host/device 共享同一头文件同步重编译）----
    int32_t arContigFast;                    // 1=base 路径连续快路；0=原一般路径/group 路径
    // 快路门控条件（host ComputeTilingImpl 判定）：
    //   base 路径成立(group==false) && other 无广播维(所有 otherShape[i]==xShape[i])
    //   && R 单 chunk(R<=ru) —— 本 ABI 下 x/y 恒连续(offset 0 行主序)，other 无广播
    //   时整 tile 连续；非对齐 R 由 DataCopyPad 的 UB 32B 行距补齐。
```

**2. tile 级合并搬入/写出（设备侧核心）**——行字节 32B 对齐时单条连续直拷（rows×rowBytes 一条描述符）；非对齐时 blockCount 行批量（GM 侧 srcStride=0 连续，UB 侧行距由 DataCopyPad 自动补齐到 `AlignUp(rowBytes,32)=rUbFactorAlign*2B`，与 VF 的行 stride 一致）。一次调用 = 单条 MTE2/MTE3 描述符，替代原逐行 256B 小片段：

```cpp
    // ===== base 路径连续快路 =====
    // tile 级合并搬运：rowBytes 32B 对齐时单条连续直拷（rows*rowElems*2B，
    // GM/UB 双连续）；非对齐时 blockCount 行批量（GM 侧 srcStride=0 连续，
    // UB 侧行距由 DataCopyPad 自动补齐到 AlignUp(rowBytes,32)=
    // rUbFactorAlign*2B，与 VF 的行 stride 一致）。一次调用 = 单条
    // MTE2 描述符（含 nburst），替代原逐行 256B NdDma。
    __aicore__ inline void CopyContigRows(LocalTensor<bfloat16_t> dst,
        GlobalTensor<bfloat16_t> src, uint32_t rows, uint32_t rowElems)
    {
        const uint32_t rowBytes = rowElems * static_cast<uint32_t>(sizeof(bfloat16_t));
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        if ((rowBytes & 31u) == 0u) {
            DataCopyExtParams params{1, rows * rowBytes, 0, 0, 0};
            DataCopyPad(dst, src, params, pad);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(rows), rowBytes, 0, 0, 0};
            DataCopyPad(dst, src, params, pad);
        }
    }

    // gamma 常驻预载（快路）：每核一次，单行 rTotal 元素
    __aicore__ inline void PreloadGammaFast()
    {
        CopyContigRows(buf1_.Get<bfloat16_t>(), gammaGm_, 1,
            static_cast<uint32_t>(td_->rTotal));
    }

    // y tile 合并写回：对齐时单条连续写（aLen*rTotal*2B），非对齐时
    // blockCount 行批量（UB 行距=rUbFactorAlign*2B，GM 行距=rowBytes 连续）
    __aicore__ inline void CopyOutContigTile(int64_t aBegin, uint32_t aLen)
    {
        const uint32_t rowElems = static_cast<uint32_t>(td_->rTotal);
        const uint32_t rowBytes = rowElems * static_cast<uint32_t>(sizeof(bfloat16_t));
        WaitFlag<HardEvent::V_MTE3>(outputReady_);
        if ((rowBytes & 31u) == 0u) {
            DataCopyExtParams params{1, aLen * rowBytes, 0, 0, 0};
            DataCopyPad(yGm_[aBegin * td_->rTotal], buf3_.Get<bfloat16_t>(), params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(aLen), rowBytes, 0, 0, 0};
            DataCopyPad(yGm_[aBegin * td_->rTotal], buf3_.Get<bfloat16_t>(), params);
        }
        SetFlag<HardEvent::MTE3_MTE2>(outputReuse_);
    }
```

**3. 快路单 tile 流程 + 双 MTE2_V 事件依赖解耦**——x 的 `SetFlag(MTE2_V)` 先于 other 发射：SquareAccVF 只消费 x，等待期间 other 的描述符继续传输（重叠）；x 的 `WaitFlag(MTE2_V)` 只置于首次消费前；reduce 段载入的 x tile 在 post 段直接复用（R 单 chunk，无二次 GM 读）；多 tile 时上一 tile 的 MTE3 读完 buf3 后才 V 重写 buf3（`[!firstTile] WaitFlag(MTE3_MTE2)`）。Init 里为 other 增取第 2 个 MTE2_V 事件 ID（arch3510 事件池 QUE_MAX_EVENT=8、每方向独立位图，同方向取第 2 个安全）：

```cpp
        copyReady_ = pipe_->FetchEventID(HardEvent::MTE2_V);
        copyOtherReady_ = pipe_->FetchEventID(HardEvent::MTE2_V);
        outputReady_ = pipe_->FetchEventID(HardEvent::V_MTE3);
        outputReuse_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
```

```cpp
    // 快路单 tile：MTE2 一次性发射（x SetFlag 先于 other 发射，SquareAccVF
    // 只消费 x，等待期间 other 描述符继续传输=重叠）；WaitFlag(MTE2_V) 仅
    // 置于首次消费前；reduce 段载入的 x tile 在 post 段直接复用（R 单 chunk，
    // 无二次 GM 读）；y 经 SetFlag(V_MTE3)→CopyOut→(多 tile 时)WaitFlag
    // (MTE3_MTE2) 后再 V 重写 buf3。VF 链与数值顺序同一般路径
    // （SquareAccVF/RstdVF/GateMulVF 仅以 buf+row*ra 基址参数化调用）。
    __aicore__ inline void ProcessTileFast(int64_t aBegin, uint32_t aLen, bool firstTile)
    {
        LocalTensor<bfloat16_t> x = buf0_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> other = buf2_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gamma = buf1_.Get<bfloat16_t>();
        LocalTensor<float> rstd = cache_.Get<float>();
        const uint32_t rLen = static_cast<uint32_t>(td_->rTotal);  // 门控保证单 chunk
        CopyContigRows(x, xGm_[aBegin * td_->rTotal], aLen, rLen);
        SetFlag<HardEvent::MTE2_V>(copyReady_);
        CopyContigRows(other, otherGm_[aBegin * td_->rTotal], aLen, rLen);
        SetFlag<HardEvent::MTE2_V>(copyOtherReady_);
        WaitFlag<HardEvent::MTE2_V>(copyReady_);
        Duplicate(rstd, 0.0f, aLen);
        asc_vf_call<SquareAccVF>((__ubuf__ float*)rstd.GetPhyAddr(),
            (__ubuf__ bfloat16_t*)x.GetPhyAddr(),
            aLen, rLen, static_cast<uint32_t>(td_->rUbFactorAlign));
        asc_vf_call<RstdVF>((__ubuf__ float*)rstd.GetPhyAddr(),
            td_->invRTotal, td_->epsilon, aLen);
        WaitFlag<HardEvent::MTE2_V>(copyOtherReady_);
        if (!firstTile) {
            // 多 tile：上一 tile 的 MTE3 读完 buf3 后才能 V 重写 buf3
            // （与上一 tile 的 MTE2 发射/V reduce 段重叠，依赖解耦）
            WaitFlag<HardEvent::MTE3_MTE2>(outputReuse_);
        }
        __ubuf__ bfloat16_t* xBase = (__ubuf__ bfloat16_t*)x.GetPhyAddr();
        __ubuf__ bfloat16_t* otherBase = (__ubuf__ bfloat16_t*)other.GetPhyAddr();
        __ubuf__ bfloat16_t* yBase = (__ubuf__ bfloat16_t*)buf3_.Get<bfloat16_t>().GetPhyAddr();
        __ubuf__ bfloat16_t* gammaBase = (__ubuf__ bfloat16_t*)gamma.GetPhyAddr();
        __ubuf__ float* rstdBase = (__ubuf__ float*)rstd.GetPhyAddr();
        const uint32_t stride = static_cast<uint32_t>(td_->rUbFactorAlign);
        for (uint32_t row = 0; row < aLen; ++row) {
            asc_vf_call<GateMulVF>(yBase + row * stride, xBase + row * stride,
                gammaBase, otherBase + row * stride, rstdBase, row, rLen);
        }
        SetFlag<HardEvent::V_MTE3>(outputReady_);
        CopyOutContigTile(aBegin, aLen);
    }
```

**4. host 侧 au 均衡求解**——快路 tile 尺寸不再受一般路径魔数限制，但在其内做 coreNum 均衡扫描（候选集=2 的幂∪16 的倍数∪上界；关键路径行数上界 `crit=Ceil(Ceil(A/au)/usedC)*au` 最小优先，其次 `A%au==0` 整除、更大 au、更大 usedCoreNum；tile 单 tensor 字节 ≥2KiB 软过滤；UB 方程按快路真实 buffer 清单与 bf16 2B 字节口径，深度 2 优先、深度 1 兜底）：

```cpp
int64_t SolveFastAu(int64_t aTotal, int64_t raAlign, int64_t coreNum,
                    int64_t ubSize, int64_t auCap)
{
    constexpr int64_t elemBytes = 2;  // bf16 字节口径
    constexpr int64_t minTileBytes = 2048;  // tile 单 tensor 字节软下限（约 2KiB）
    if (aTotal < 1 || raAlign < 1 || coreNum < 1 || ubSize < 1 || auCap < 1) return 0;
    const int64_t gammaBytes = PubUp(raAlign * elemBytes, 32);
    auto tileBytes = [raAlign](int64_t au) { return PubUp(au * raAlign * elemBytes, 32); };
    auto rstdBytes = [](int64_t au) { return PubUp(au * 4, 32); };
    int64_t upper2 = 0, upper1 = 0;
    for (int64_t au = 1; au <= auCap; ++au) {
        if (5 * tileBytes(au) + gammaBytes + rstdBytes(au) <= ubSize) upper2 = au;
        if (3 * tileBytes(au) + gammaBytes + rstdBytes(au) <= ubSize) upper1 = au;
    }
    const int64_t upper = upper2 >= 1 ? upper2 : upper1;
    if (upper < 1) return 0;
    // 候选集：2 的幂 ∪ 16 的倍数 ∪ upper（均截断到 upper，去重）
    int64_t cands[300];
    int32_t candNum = 0;
    for (int64_t p = 1; p <= upper && candNum < 300; p <<= 1) cands[candNum++] = p;
    for (int64_t m = 16; m <= upper && candNum < 300; m += 16) {
        bool dup = false;
        for (int32_t i = 0; i < candNum; ++i) {
            if (cands[i] == m) { dup = true; break; }
        }
        if (!dup) cands[candNum++] = m;
    }
    {
        bool dup = false;
        for (int32_t i = 0; i < candNum; ++i) {
            if (cands[i] == upper) { dup = true; break; }
        }
        if (!dup && candNum < 300) cands[candNum++] = upper;
    }
    bool hasMinTile = false;
    for (int32_t i = 0; i < candNum; ++i) {
        if (cands[i] <= aTotal && tileBytes(cands[i]) >= minTileBytes) hasMinTile = true;
    }
    int64_t bestAu = 0, bestCrit = 0, bestExact = 0, bestUsed = 0;
    for (int32_t i = 0; i < candNum; ++i) {
        const int64_t au = cands[i];
        if (au > aTotal) continue;  // 超过总行数的候选无意义（候选恒含 1）
        if (hasMinTile && tileBytes(au) < minTileBytes) continue;
        const int64_t tiles = PubCeil(aTotal, au);
        const int64_t used = std::min(tiles, coreNum);
        const int64_t bigLoop = PubCeil(tiles, used);
        const int64_t crit = bigLoop * au;  // 最慢核行数上界
        const int64_t exact = (aTotal % au == 0) ? 1 : 0;
        const bool better = bestAu == 0 || crit < bestCrit ||
            (crit == bestCrit && (exact > bestExact ||
                (exact == bestExact && (au > bestAu ||
                    (au == bestAu && used > bestUsed)))));
        if (better) {
            bestAu = au;
            bestCrit = crit;
            bestExact = exact;
            bestUsed = used;
        }
    }
    return bestAu;
}
```

门控与重解调用点（auCap 传一般路径解 au0，保证快路 au 不超过它、`ChooseGroupTiling` 触发域不因快路改变）：

```cpp
    bool fast = false;
    if (!group) {
        bool noBroadcast = true;
        for (int i = 0; i < 8; ++i) {
            if (t->otherShape[i] != t->xShape[i]) { noBroadcast = false; break; }
        }
        const bool singleChunk = (ro == 1);  // R <= ru
        if (noBroadcast && singleChunk) {
            const int64_t auFast = SolveFastAu(A, ra, static_cast<int64_t>(coreNum),
                static_cast<int64_t>(ubSize), au);
            if (auFast > 0) {
                fast = true;
                au = auFast;
                ao = PubCeil(A, au);
                usedC = std::min<int64_t>(ao, coreNum);
```

## 设计决策

- 为什么用 host 门控快路而不是改写一般路径：group 路径（超长 R）与一般路径（广播、多 chunk R）零修改保留为回退，快路命中条件不满足时语义与基线完全一致，回归风险隔离在门控条件内。
- 为什么不新增 tilingKey：新增 tilingKey 会改变路由与编译产物集合；在 base 路径（tilingKey=0）内做 au 重解 + TilingData 尾部字段门控，blockDim/tilingKey 对非目标 case 逐 case 保持与基线一致（18/18 验收核对）。
- 为什么 x 与 other 用两个 MTE2_V 事件 ID：x 先发射先消费（SquareAccVF），other 的完成等待推迟到 GateMulVF 前，搬入与计算重叠；arch3510 事件池 QUE_MAX_EVENT=8（kernel_macros.h 按 `__NPU_ARCH__` 分档），同方向取第 2 个 ID 安全（其他架构池深不同，移植需重查）。UB buffer 跨迭代复用的同步竞态反面案例见 [cross_iteration_ub_reuse_mte2_vf_race](../runtime/cross_iteration_ub_reuse_mte2_vf_race.md)。
- 为什么 au 扫描要"整除解优先于更小关键路径"：A=768 时 au=8 的 crit 与 au=16 持平，但 au=16 使 768=16×48 无尾 tile（48 核均匀），尾 tile 会让最后一核拖长尾延迟；实测目标 case au=16→48 核。
- 为什么 tile≥2KiB 软过滤：单条 DMA 的带宽拐点量级在 2KiB 附近，过小 tile 合并收益有限反而增加调度数。
- 当前方案路线档位：recommended_fastpath（本会话终选方案的前半部分，与 VF 收拢叠加后 9.16x）。

## 系统信息

- 硬件/软件：Ascend950PR NPU 0（AIV=56、UB=253952B），CANN 9.2.0；设备 TU 以 `dav-c310-vec` 纯设备模式编译（split 构建 + AIVEC magic patch 为既有既定流程，见 [tpipe_eager_launch_507015_aivec_magic_patch](../compilation/tpipe_eager_launch_507015_aivec_magic_patch.md)）。
- 相关文件（候选隔离副本内）：`csrc/op_kernel/kernel/arch35/fused_rms_norm_sigmoid_mul_tiling_struct.h`（尾部追加门控字段）、`csrc/op_kernel/kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h`（CopyContigRows/PreloadGammaFast/CopyOutContigTile/ProcessTileFast/Init 双事件）、`csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp`（SolveFastAu + 门控）；group 路径与一般路径文件零修改。
- SDK 事实锚点：`kernel_macros.h` 中 `__NPU_ARCH__==3510` 归入 `QUE_MAX_EVENT=8` 分支（其余架构为 4）。

## 性能影响

- 相对原始融合基线（同算子、同协议、同设备，参照路线=优化前逐行搬运实现）：有效 case 几何加速比 6.18x（来源平台：950）。
- 与同会话仅做 host tiling 重平衡的候选成对对比（参照路线=host-only tiling 重平衡）：合并 DMA 快路约 3.45x 于该候选（同协议成对，来源平台：950）——量化说明搬运通路合并是该算子的首要杠杆。
- 与叠加 VF 收拢的终选实现成对对比（参照路线=本卡路线）：本卡之上 VF 收拢再获约 1.48x（来源平台：950）。

## 预防

- 融合算子出现"整体慢于非融合链"时，先看 MTE2 发射/描述符利用率与单条搬运字节数：逐行小片段（行字节远小于 2KiB）搬运是首要嫌疑（判别方法见 [mte2_issue_overhead_vs_bandwidth_bound](mte2_issue_overhead_vs_bandwidth_bound.md)）。
- 在 GM 侧连续性成立时（无广播、行主序、offset 0），优先把"行循环 × 单行搬运"折叠为"tile × 合并搬运"；行字节非 32B 对齐时用 DataCopyPad Normal 模式行批量而不是退回逐行。
- 事件依赖按消费点拆分（每个独立 TBuf 的搬入各自 SetFlag，WaitFlag 只放在首次消费前），不要一把全等。

## 验证信号

以下信号来自该融合算子 Round 1 合并 DMA 快路候选的实施与采集会话（Ascend950PR / CANN 9.2.0）：候选在基线隔离副本内实施，构建后先做 tiling 仿真对账再上板验收，最后与基线同协议串行采集。基线瓶颈画像（MTE2 发射/描述符利用率 0.78、逐行 256B 小片段）来自同会话前期的 msprof 分析阶段。

- 构建通过：split 构建 5/5 步骤完成、AIVEC magic patch 5 sites、目标 kernel mangled 符号提取成功（来源平台：950；证据等级：tool_observed）。
- tiling 行为：18 条 L0 的 blockDim/tilingKey 与基线 manifest 逐 case 一致（广播/tiny case 正确回退一般路径）；目标 case au=16、48 核整除无尾 tile、fast=True、tile=4KiB（来源平台：950；证据等级：tool_observed）。
- 功能验收（精度不回归）：18 条 L0 strict 18/18 PASS，golden/cases/runner SHA 与冻结口径一致（来源平台：950；证据等级：tool_observed）。
- 同步语义：ascendc-sync-check strict 扫描 0 findings（来源平台：950；证据等级：tool_observed）。
- 性能：目标 case 10 组中位采集 validation 30/30 launch 映射通过；有效 case 几何加速比相对原始融合基线 6.18x（同算子同协议成对，差异远超抖动阈值判为有效提升）（来源平台：950；证据等级：tool_observed）。
- 集成后经注册接口复验：18/18 L0 PASS（aclnn 接口，TTK Rev 2474d06）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 门控条件外（other 广播、R 多 chunk、group 路径）自动回退原路径，本卡快路不接管——这些场景的优化需另行设计（未验证）。
- QUE_MAX_EVENT 池深按架构分档（arch3510 为 8，其余多为 4）：双 MTE2_V 事件 ID 用法在池深 4 的架构上仍安全（2≤4）但未实测；同时取更多同方向 ID 前必须重查目标架构常量。
- 非对齐行批量分支的描述符数为行数而非 1：行数极大且行字节非对齐时合并收益退化（本会话非对齐 case 行数小，未测极值）。
- UB 方程的字节清单按快路 buffer 布局（x/other/y tile + gamma + rstd）推导，改动 buffer 布局（如再加双缓冲深度）必须同步改方程，否则越界或浪费。
- 仅 bf16 实测；fp16/fp32 的元素字节与对齐口径不同，方程需按 `sizeof(T)` 重推（未验证，机制可迁移）。

**未决项**：

- 快路 au 与一般路径 au0 的解耦上限（auCap）在本候选保留（快路 au≤au0）；后续 VF 收拢候选已验证可安全去掉该上限并按快路字节方程全域扫描（见 [vf_lane_accumulate_row_loop_consolidation](vf_lane_accumulate_row_loop_consolidation.md)），两档取舍的量化边界未单独采集。
- 带广播维 other 的合并搬运（按广播语义展开描述符）未尝试。
## 坏实践

- 保持逐行 `DataCopyPad` 循环而指望靠增大 au 提升性能——描述符数随行数线性增长，发射 bound 不解除（本会话 host-only tiling 重平衡候选实测仅 1.79x，证明不解除搬运 bound 时核数翻倍收益有限，见 [host_tiling_magic_au_cap_ub_equation_rebalance](host_tiling_magic_au_cap_ub_equation_rebalance.md)）。
- 用 5D ND-DMA（`NdDmaDci`）逐行拼小片段——直调单 TU 下还无法编译（见 [nd_dma_dci_single_tu_target_feature_split_build](../compilation/nd_dma_dci_single_tu_target_feature_split_build.md)），且发射开销同样存在。
- 把 `WaitFlag(MTE2_V)` 放在所有搬入之后统一等待——丢掉 x 先行消费与 other 传输的重叠。

## 相关卡片

- [PB-45 `TPipe::Reset()` 释放全局 `g_tpipeImpl` 事件池](../compilation/pb_45_tpipe_reset_frees_the_global_g_tpipeimpl_event_poo.md)：本卡的 `QUE_MAX_EVENT=8` 是预算数字，该卡是**超出这个预算时的失败形态**——编译不报错、kernel 静默挂死，且 `TPipe::Reset()` 会释放整个全局池。只看本卡的数字而不知道越界后不报错，会把挂死误判成算法问题。
- [OL-223 Reset-safe FetchEventID L0 fences](../../optimizations/ol_223_reset_safe_fetcheventid_l0_fences.md)：在池深受限且存在 Reset 时，取/还 event id 的正确写法。
