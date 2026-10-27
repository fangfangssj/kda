---
type: "Runbook"
title: b16 DataCopyPad UB→GM 源非 32B 对齐致硬件 fault
description: fp16/bf16 下 DataCopyPad UB→GM blockCount=1 用非 32B 对齐的 UB 源地址（如 postBuf[104]=208B）直接硬件 fault（aclrtSynchronizeStream 失败、无 kernel 报错）；治本办法是选整除输出边界的切分因子让源恒对齐。
tags: [task.debugging, topic.alignment, topic.data_movement.memory_access, topic.tiling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# b16 DataCopyPad UB→GM 源非 32B 对齐致硬件 fault

## 现象

group 分支 kernel 的 3H 边界 clamp 修复后，11 条 ST 用例里只剩 G6 稳定崩溃：`[ST] G6 ERROR: aclrtSynchronizeStream 失败`（单独重跑同样崩，确定性复现），其余 10 条全 PASS。G6 的 shape：fp16、B=512、H=120、bias=1；tiling 实算 rLoopCntTotal=22 → usedCoreNum=22、rGroupCnt=22，Phase 2 的 A 轴 aTotal=6H=720、切分因子 aUbFactorP2=136（preBuf 均分后 FloorAlign(·,8)），共 6 个 chunk（[0,136)…[680,720)），其中 chunk [272,408) 跨 3H=360 边界、chunk [408,544)/[544,680)/[680,720) 整段在 hidden 侧。崩溃 kernel 没有任何 NPU 侧报错码/日志，host 侧只见流同步失败；同算子 base 路径（Branch-0，20/20）与 group 其余 dtype/shape 全部正常。

当时代码状态：Phase 2 每核一个 A chunk，链路为 CopyIn workspace（GM→UB）→ ReduceSum RA → BiasPostElewise（fp32 cacheBuf → D_T postBuf，dense [aLen]）→ `CopyOutBiasDual`（postBuf 按 3H 边界二分，两路 `DataCopyPad` UB→GM，blockCount=1）。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke，AscendC 模板 kernel，aclnn ST 上板联调）。
- API 路径：`AscendC::DataCopyPad`（UB→GM，`DataCopyExtParams`，blockCount=1）/ `AscendC::DataCopy`（UB→UB）/ `AscendC::Duplicate`——b16（fp16/bf16）路径；CANN 实现 `x86_64-linux/asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h`（b16 UB→GM 走 `copy_ubuf_to_gm_align_b16`）。
- 验证过的平台与 CANN 版本：Ascend 950（dav_c220 对应 arch）+ CANN 9.0.0-beta.2。与 frontmatter `platforms` 一致。
- 适用粒度：operator_family——凡是"归约结果落 UB 后按内部边界分段拷出 GM"的 b16 kernel 都可能踩；对齐元素数 = 32B/sizeof(D_T)（fp16/bf16:16，fp32:8）。
- 路线优先级：recommended_fastpath（切分因子整除输出边界，源恒对齐）；搬运侧 patch（scratch 中转/分段 padding）为 banned（见坏实践）。
- 明确不适用/未验证：fp32（elemSz=4，8 元素即 32B，FloorAlign(·,8) 已天然满足）本开发会话 中未复现该崩溃；GM→UB 方向无此约束（workspace 搬入全程正常）；多 burst（blockCount=3 + srcStride）形态在 base 内核对非对齐 blockLen 稳定工作，机制差异未验证。

## 根因

fp16/bf16 的 UB→GM `DataCopyPad`（blockCount=1）要求 UB 源地址 32B 对齐：跨界 chunk 的 hidden 段源 `postBuf[part1Clamped]` 在 3H 不是对齐元素数（16）倍数时得到非 32B 对齐地址，硬件 DMA 直接 fault；而 release 上板构建既无参数校验（`CheckDataCopyPadParams` 仅 `ASCENDC_CPU_DEBUG` 下编译）也无 OOM trap（`CheckGmMemOverflow→trap` 仅 `ASCENDC_OOM==1` 启用，`g_gm_overflow_check` 为 constexpr，构建系统未定义该宏），所以故障只表现为 host 侧 `aclrtSynchronizeStream 失败`，无任何 kernel 侧错误信息。

机制链与触发算式（G6 实算）：

- 对齐元素数 `alignElems = 32B / sizeof(D_T)`：fp16 → 16 个元素 = 32B。
- 切分因子按 8（BLOCK_F32）对齐是 fp32 口径：`aUbFactorP2 = FloorAlign(139, 8) = 136`。136×2B = 272B，272 mod 32 = 16 —— 对 fp16 既不保证 blockLen 对齐，也不保证分段后源偏移对齐。
- 3H = 3×120 = 360；360×2B = 720B，720 mod 32 = 16 —— 输出边界本身不是 32B 的倍数。只要 3H 不是 16 的倍数（360/16 = 22.5），任何 16 倍数的 aUbFactorP2 都会让跨界 chunk 的 `part1Clamped = 3H − aOff` 落在 16 的非倍数上。
- 触发点：chunk [256,384)（对齐到 16 的 aUbFactorP2=128 时）`part1 = 360−256 = 104` → 源偏移 104×2 = 208B，208 mod 32 = 16 ≠ 0 → 非对齐源 → fault。原始 aUbFactorP2=136 时纯 hidden chunk 的整段拷出（源 postBuf[0] 对齐、blockLen 272B）同样崩，而 part1 段（源对齐、blockLen 272B）不崩——blockLen 的确切作用存在轨迹内矛盾证据（见"误诊/遗留问题"），工程上按"源必须对齐、blockLen 尽量对齐"双约束处理。

## 判别方法

无源码级调试器下的定位链（全部为本开发会话 实测步骤，可复用为套路）：

1. **阶段裁除**：临时把 Phase 2 整体 no-op → G6 由 ERROR 转 FAIL（不崩）→ 崩在 Phase 2。
2. **替换可疑计算**：`ReduceSum isReuseSource=true→false` 仍崩；`ReduceSum` 换成 `Duplicate(cacheBuf[0], 0.0f, aLenUb)` 仍崩 → 与 ReduceSum 无关。
3. **逐段加回二分**：跳过 `BiasPostElewise + V_MTE3 sync + CopyOutBiasDual` → 不崩；只加回 BiasPostElewise → 不崩；加回 V_MTE3 sync + CopyOutBiasDual → 崩；只留 V_MTE3 sync 不留拷出 → 不崩 → **崩溃锁定在 CopyOutBiasDual 的 DataCopyPad UB→GM**。
4. **kernel 侧打印**：`AscendC::PRINTF`（配 `ASCENDC_DUMP=1 ./run.sh`）看到 blk=0/1/2 打印 after ReduceSum、blk=3 只有 before 无 after——注意 MTE3 是异步队列，PRINTF 只能把范围缩到"该核附近"，真正 fault 在异步拷出（本例 PRINTF 结论曾误导为 ReduceSum，后被第 3 步纠正）。
5. **对齐实验**：只把 part2 的 blockLen floor 对齐到 32B（272B→256B）→ 不崩；再只保留 part2（part1 注释）→ 仍崩，此时源 `postBuf[104]=208B` 非 32B 对齐 → 锁定源地址对齐。
6. **交叉验证**：`DataCopy`（UB→UB）从 postBuf[104] 搬到对齐 scratch 再拷出 → 仍崩 → UB→UB 搬运同样要求 32B 对齐源（排除法：后续 DataCopyPad 已是源对齐+blockLen 对齐的安全形态）。
7. **快速心算判别**：对每个拷出调用算 `srcByteOff = idx×sizeof(D_T)`，`srcByteOff mod 32` 与 `blockLen mod 32` 任一非 0 即高危；重点查"按输出内边界分段"产生的偏移（边界值 × elemSz 往往不是 32B 倍数）。

## 解决配方

治本思路：**选 aUbFactorP2 为 3H 的因数**，让 chunk 边界与 3H 输出边界重合——没有任何 chunk 跨界，每次拷出的源恒为 `postBuf[0]`（32B 对齐），且 6H=2×3H 自动整除（无尾 chunk），blockLen=aLen×elemSz 允许非 32B 对齐（源对齐时可过）。

before（切分因子按 fp32 口径 FloorAlign(·,8)，fp16 下 136×2B=272B 非对齐；`CopyOutBiasDual` 存在跨界二分 → 产生非对齐源）：

```cpp
    // ── Phase 2 切分（kernel 侧现算）──
    const int64_t preInElems = Base::td_->preBufSize / static_cast<int64_t>(sizeof(float));
    int64_t aUbFactorP2 = preInElems / rGroupCnt_;             // 按 R 分组均分 preBuf
    if (aUbFactorP2 >= BLOCK_F32) {
        aUbFactorP2 = (aUbFactorP2 / BLOCK_F32) * BLOCK_F32;   // FloorAlign(·, 8) 32B 对齐（fp32）
    }
    aUbFactorP2 = (aUbFactorP2 < Base::td_->postBufSize / static_cast<int64_t>(sizeof(D_T)))
                      ? aUbFactorP2
                      : Base::td_->postBufSize / static_cast<int64_t>(sizeof(D_T));
    aUbFactorP2 = (aUbFactorP2 < aTotal_) ? aUbFactorP2 : aTotal_;   // ≤ 6H 兜底
    if (aUbFactorP2 < 1) {
        aUbFactorP2 = 1;                // 安全下界（理论不可达：preBuf ≥ 8 fp32）
    }
```

after（aUbFactorP2 取 3H 的最大因数 ≤ cap；拷出退化为整 chunk 单路、源恒 postBuf[0]；`CopyOutBiasDual` 函数整体删除）：

```cpp
    // ── Phase 2 切分（kernel 侧现算，§2.3）──
    //   aUbFactorP2 取 3H 的最大因数 ≤ cap（preBuf/postBuf/6H 三约束 min），保证
    //   chunk 边界对齐 3H → 无 chunk 跨越 3H → CopyOut 源恒在 postBuf[0]（32B 对齐）。
    //   b16 DataCopyPad UB→GM blockCount=1 要求 src 32B 对齐；此切分规避非对齐源。
    const int64_t preInElems = Base::td_->preBufSize / static_cast<int64_t>(sizeof(float));
    int64_t cap = preInElems / rGroupCnt_;
    cap = (cap < Base::td_->postBufSize / static_cast<int64_t>(sizeof(D_T))) ? cap
              : Base::td_->postBufSize / static_cast<int64_t>(sizeof(D_T));
    cap = (cap < aTotal_) ? cap : aTotal_;
    const int64_t threeH = 3 * Base::H_;
    int64_t aUbFactorP2 = cap;
    while (aUbFactorP2 > 0 && threeH % aUbFactorP2 != 0) { --aUbFactorP2; }
    if (aUbFactorP2 < 1) { aUbFactorP2 = 1; }
```

```cpp
        // CopyOut bias：aUbFactorP2 整除 3H → chunk 不跨越 3H → 整 chunk 属 input 或 hidden。
        // 源恒在 postBuf[0]（32B 对齐），blockLen = aLen×elemSz（可能非 32B 对齐但源对齐可过）。
        const int64_t biasBnd = 3 * Base::H_;
        const int64_t elemSz2 = static_cast<int64_t>(sizeof(D_T));
        if (aOff + aLen <= biasBnd) {
            // 整 chunk → gmInBias_
            AscendC::DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(aLen * elemSz2);
            ext.blockCount = 1; ext.srcStride = 0; ext.dstStride = 0; ext.rsv = 0;
            AscendC::DataCopyPad(Base::gmInBias_[aOff],
                                 Base::postBuf_.template Get<D_T>(), ext);
        } else {
            // 整 chunk → gmHidBias_（aOff ≥ 3H，因 aUbFactorP2 整除 3H）
            AscendC::DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(aLen * elemSz2);
            ext.blockCount = 1; ext.srcStride = 0; ext.dstStride = 0; ext.rsv = 0;
            AscendC::DataCopyPad(Base::gmHidBias_[aOff - biasBnd],
                                 Base::postBuf_.template Get<D_T>(), ext);
        }
```

各用例实算的因子（同一公式）：G6 H=120 → 3H=360，cap=139 → 因子 120（360/120=3）；G5 H=128 → 384，因子 128；E07 H=16 → 48，因子 48；P4 H=256 → 768，cap=384 → 384；P6 H=768 → 2304，cap=512 → 384（2304/512=4.5 非整除，递减到 384）。

## 设计决策

- 为何选"切分因子整除输出边界"而不是搬运侧 patch：一个约束同时消除两类问题（跨界二分的非对齐源 + 尾 chunk 的非对齐 blockLen），且 6H=2×3H 自动保证无尾 chunk；代码反而变简单（`CopyOutBiasDual` 整个删除）。
- 为何用"自 cap 递减找因数"而不是 `3H/ceil(3H/cap)`：后者不保证整除（P6：2304/ceil(2304/512)=2304/5=460.8 非整数）；`gcd(3H, cap)` 会退化（G6：gcd(360,139)=1）；递减循环 O(cap)、cap≤约 1.5k，每核每次 Phase 2 现算一次可接受。
- 为何放弃分段 padding（CeilAlign blockLen 多写 GM）：padding 字节落入相邻 chunk 的 GM 区间，多核并发的 chunk 间形成写写竞态，且跨界 chunk 的 padding 会溢出到另一路 bias 输出（实测 G5 grad_hidden_bias mare=1.000e+00 残留）。
- 为何放弃 scratch 中转：`DataCopy` UB→UB 同样要求 32B 对齐源（实测仍崩），中转不解决对齐只移动问题；Reg::Load/LoadUnAlign 非对齐读路线调查后因复杂度放弃（未验证）。
- 与 base 内核的对照：base `CopyOutBias` 用 blockCount=3 + srcStride/dstStride 的 3-burst 切片式写法，非对齐 blockLen（如 H=17 fp16 的 34B）在 Branch-0 ST 稳定通过——多 burst 形态是另一条可行路线，本 kernel 因 Phase 2 postBuf 为 flat 布局（非 3 段 aUnit 间距）未采用；两条路线的对齐容差差异未做机制级验证。base 写法逐字如下（供对照迁移）：
  ```cpp
  template <typename D_T>
  __aicore__ inline void ThnnFusedGruCellGradBaseKernel<D_T>::CopyOutBias(int64_t aOff, uint32_t aLen)
  {
      const int64_t elemSz = static_cast<int64_t>(sizeof(D_T));
      const int64_t aLenBytes = static_cast<int64_t>(aLen) * elemSz;
      const int64_t copyPadBytes = (aLenBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
      const int64_t gapUB = (static_cast<int64_t>(aUnit_) * elemSz - copyPadBytes) / BLOCK_BYTES;

      AscendC::DataCopyExtParams cp{};
      cp.blockCount = 3;                                   // 3 段（dense 单根路径时 aLen=H、两 gap=0）
      cp.blockLen = static_cast<uint32_t>(aLenBytes);
      cp.srcStride = gapUB;
      cp.dstStride = (H_ - static_cast<int64_t>(aLen)) * elemSz;
      cp.rsv = 0;
      AscendC::DataCopyPad(gmInBias_[aOff], postBuf_.Get<D_T>(), cp);               // [in_r, in_z, in_n]
      AscendC::DataCopyPad(gmHidBias_[aOff], postBuf_.Get<D_T>()[3 * aUnit_], cp);  // [hid_r, hid_z, hid_n]
  }
  ```
- 当前方案属 recommended_fastpath。

## 系统信息

- 改动文件：`op_kernel/arch35/thnn_fused_gru_cell_grad_group.h`（`Phase2Process` 的切分与拷出段；`CopyOutBiasDual` 删除）。
- 工具链/实现参考：CANN 9.0.0-beta.2；`x86_64-linux/asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h`（`DataCopyPadUB2GMImpl`→`copy_ubuf_to_gm_align_b16`、`CheckDataCopyPadParams` 仅 CPU_DEBUG）、`.../basic_api/kernel_utils_base.h`（`GetGMLen`/`CheckGmMemOverflow`/trap 仅 `ASCENDC_OOM==1`）、`.../basic_api/utils/kernel_utils_macros.h`（`g_gm_overflow_check` constexpr）。
- 验证入口：`cd operators/ThnnFusedGruCellGrad_package/tests/aclnn && ./run.sh`（G6 = fp16 B=512 H=120 bias=1 用例）；构建 `cd operators && ./build.sh ThnnFusedGruCellGrad_package`。
- 相关文件：`op_kernel/arch35/thnn_fused_gru_cell_grad_base.h`（blockCount=3 参照写法）、`tests/aclnn/st_cases_branch1.json`。

## 完整代码

终版 `Phase2Process` 的两个机制单元（切分计算 + bias 拷出）逐字取自事件流（终版全文未整体重写，由初版全量写入 + 编辑链组装，以下为修复因果单元）：

```cpp
template <typename D_T>
__aicore__ inline void ThnnFusedGruCellGradGroupKernel<D_T>::Phase2Process()
{
    // ── Phase 2 切分（kernel 侧现算，§2.3）──
    //   aUbFactorP2 取 3H 的最大因数 ≤ cap（preBuf/postBuf/6H 三约束 min），保证
    //   chunk 边界对齐 3H → 无 chunk 跨越 3H → CopyOut 源恒在 postBuf[0]（32B 对齐）。
    //   b16 DataCopyPad UB→GM blockCount=1 要求 src 32B 对齐；此切分规避非对齐源。
    const int64_t preInElems = Base::td_->preBufSize / static_cast<int64_t>(sizeof(float));
    int64_t cap = preInElems / rGroupCnt_;
    cap = (cap < Base::td_->postBufSize / static_cast<int64_t>(sizeof(D_T))) ? cap
              : Base::td_->postBufSize / static_cast<int64_t>(sizeof(D_T));
    cap = (cap < aTotal_) ? cap : aTotal_;
    const int64_t threeH = 3 * Base::H_;
    int64_t aUbFactorP2 = cap;
    while (aUbFactorP2 > 0 && threeH % aUbFactorP2 != 0) { --aUbFactorP2; }
    if (aUbFactorP2 < 1) { aUbFactorP2 = 1; }
    const int64_t aSplitChunkCntP2 = (aTotal_ + aUbFactorP2 - 1) / aUbFactorP2;   // CeilDiv
    const int64_t usedCoreNum = static_cast<int64_t>(Base::td_->usedCoreNum);
    const int64_t aSmallCoreLoopCntP2 = aSplitChunkCntP2 / usedCoreNum;
    const int64_t aBigCoreCntP2 = aSplitChunkCntP2 % usedCoreNum;
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + ((aBigCoreCntP2 > 0) ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? usedCoreNum : aBigCoreCntP2;

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    if (blockIdx >= usedCoreNumP2) {
        return;                         // 多余核 Phase 2 早退（SyncAll 已通过）
    }
    int64_t aLoopStart;
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 +
                      (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
    }
    const int64_t aLoopEnd = aLoopStart + ((blockIdx < aBigCoreCntP2) ? aBigCoreLoopCntP2
                                                                       : aSmallCoreLoopCntP2);

    bool firstIter = true;
    for (int64_t aLoop = aLoopStart; aLoop < aLoopEnd; ++aLoop) {
        const int64_t aOff = aLoop * aUbFactorP2;
        const int64_t aLen = (aTotal_ - aOff < aUbFactorP2) ? (aTotal_ - aOff) : aUbFactorP2;

        if (!firstIter) {
            // 跨迭代 WAR：上轮 ReduceSum 读 preReduceResult_ → 本轮 CopyIn 覆写；
            // 上轮 CopyOut 读 postBuf → 本轮 PostElewise 覆写
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Base::evVtoMte2_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Base::evMte3ToV_);
        }

        // CopyIn workspace [rGroupCnt, aLen] fp32 → preReduceResult_（复用 Phase 1 槽，§5.2）
        CopyInWorkspace(aOff, aLen);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(Base::evMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(Base::evMte2ToV_);

        // ReduceSum RA：dst=cacheBuf[0] 单层（无二分树）；srcShape={rGroupCnt, aLenUb} padded
        const int64_t aLenUb = (aLen + BLOCK_F32 - 1) / BLOCK_F32 * BLOCK_F32;  // CeilAlign(·, 8)
        uint32_t srcShape[2] = {static_cast<uint32_t>(rGroupCnt_),
                                static_cast<uint32_t>(aLenUb)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
            Base::cacheBuf_.template Get<float>()[0],
            Base::preRegion_.template Get<float>()[static_cast<uint32_t>(Base::biasElemOff_)],
            Base::preRegion_.template Get<uint8_t>()[static_cast<uint32_t>(Base::tailByteOff_)],
            srcShape, /*srcInnerPad=*/true);

        // BiasPostElewise（§9.7 机制）：cacheBuf[0] fp32 → postBuf D_T 单次缩位。
        const uint16_t repTime = static_cast<uint16_t>(
            (static_cast<uint32_t>(aLen) + kRepF32 - 1) / kRepF32);
        asc_vf_call<BiasPostElewiseVfImpl<D_T>>(Base::cachePtr_, Base::postPtr_,
                                                static_cast<uint32_t>(aLen), repTime);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Base::evVToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Base::evVToMte3_);

        // CopyOut bias：aUbFactorP2 整除 3H → chunk 不跨越 3H → 整 chunk 属 input 或 hidden。
        // 源恒在 postBuf[0]（32B 对齐），blockLen = aLen×elemSz（可能非 32B 对齐但源对齐可过）。
        const int64_t biasBnd = 3 * Base::H_;
        const int64_t elemSz2 = static_cast<int64_t>(sizeof(D_T));
        if (aOff + aLen <= biasBnd) {
            // 整 chunk → gmInBias_
            AscendC::DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(aLen * elemSz2);
            ext.blockCount = 1; ext.srcStride = 0; ext.dstStride = 0; ext.rsv = 0;
            AscendC::DataCopyPad(Base::gmInBias_[aOff],
                                 Base::postBuf_.template Get<D_T>(), ext);
        } else {
            // 整 chunk → gmHidBias_（aOff ≥ 3H，因 aUbFactorP2 整除 3H）
            AscendC::DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(aLen * elemSz2);
            ext.blockCount = 1; ext.srcStride = 0; ext.dstStride = 0; ext.rsv = 0;
            AscendC::DataCopyPad(Base::gmHidBias_[aOff - biasBnd],
                                 Base::postBuf_.template Get<D_T>(), ext);
        }

        if (aLoop != aLoopEnd - 1) {
            // 放行下轮：V 读完 preReduceResult_（V_MTE2）+ MTE3 读完 postBuf（MTE3_V）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Base::evVtoMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Base::evMte3ToV_);
        }
        firstIter = false;
    }
}
```

（`CopyInWorkspace`、`Phase1Process`、`Phase1OutputToWorkspace` 与初版一致，未在本次修复中改动。）

## 预防

- b16 kernel 的每个 UB→GM / UB→UB 搬出点，写完先心算 `源字节偏移 mod 32` 与 `blockLen mod 32`；凡是"用输出内部边界（如 3H）切段"的偏移，默认它不是 32B 倍数（边界值 × 2B 常含 16B 余数）。
- 切分因子的对齐口径要跟 dtype 走：fp32 用 8（BLOCK_F32）即可，fp16/bf16 需 16；更稳的做法是让因子整除输出边界，从源头消灭非对齐分段。
- 上板崩溃"无报错 + aclrtSynchronizeStream 失败"时，优先怀疑 DMA 对齐/越界（release 无 CPU_DEBUG 校验、无 OOM trap），不要指望 kernel 侧日志。
- 遇到 `ERROR: aclrtSynchronizeStream 失败` 先跑一遍"阶段裁除 → 替换可疑算子 → 逐段加回"的二分（见判别方法），比反复加 PRINTF 高效且不受时序干扰。

## 验证信号

本卡验证来自 G6 崩溃的定位与最终修复轮（同一 ST 套件与回归面）：

- 定位实验：Phase 2 禁用后 G6 转 FAIL 不崩；Duplicate 替换 ReduceSum 仍崩；逐段二分锁定 CopyOutBiasDual 的 DataCopyPad；part2-only（源 postBuf[104]=208B 非 32B 对齐）仍崩；`DataCopy` UB→UB scratch 中转仍崩（来源平台：950；证据等级：tool_observed）。
- 最终修复后：`==== Branch-1 ST summary: PASSED=11 FAILED=0 ERROR=0 TOTAL=11 ====`——G6 由稳定 ERROR 转 PASS（fp16 B=512 H=120，各输出按 Validation.md mare 预算达标）；Branch-0 ST 20/20、TilingUT 73/73、4 个 ops-impl-review 脚本（scalar_mainchain/vf_reachability/workspace_consistency/non3510_scaffold）全部 exit=0、smoke 1/1；末次全量复跑再确认（来源平台：950；证据等级：tool_observed）。

## 性能影响

未测量：切分因子从 136 改为 3H 因数（如 G6 的 120）会改变 Phase 2 的 chunk 数与并行度，本开发会话 未做两条切分路线的成对性能对比，采用本方案的性能代价未验证。

## 失效边界

- 本修复依赖"输出边界 = 3H、总量 = 6H = 2×3H"的结构；若输出边界与总量没有倍数关系（如三路不等宽输出），"因子整除边界"不再适用，需回到多 burst（blockCount>1）或对齐分段方案。
- 3H 无 ≤cap 的合用因数时循环会退化到小因子（chunk 数变多、并行度下降），极端时因子=1；本开发会话 用例未触发（最小因子 48）。
- `blockLen 非 32B 对齐 + 源对齐`的安全性在最终版被全量 11/11 间接支持（aUbFactorP2=120 时 blockLen=240B 非对齐仍过），但轨迹内存在反例（见误诊第 2 条），未上升到机制结论；其他平台/后续 CANN 版本未验证，推断可迁移但未验证。
- fp32 未复现该崩溃（8 元素天然 32B），bf16 与 fp16 同为 b16 路径（P6 bf16 最终通过）。

## 坏实践

以下路线在本开发会话 中全部实测失败或引入新缺陷，不要重复：

- **scratch 中转**（banned）：
  ```cpp
  AscendC::DataCopy(Base::SlotTensorD_T(SLOT_GATES_IN),
                    Base::postBuf_.template Get<D_T>()[postOff],   // postOff=104 → 208B 非对齐
                    static_cast<uint32_t>(paddedP2));
  ```
  UB→UB 同样要求 32B 对齐源——中转不解决对齐，仍硬件 fault。
- **CeilAlign padding 多写 GM**（banned）：blockLen 取 `CeilAlign(part, alignE)` 会把 postBuf 里 aLen 之后的未初始化垃圾多写进 GM；跨界 chunk 的 padding 还会溢出到另一路 bias 输出，多核并发 chunk 间形成写写竞态。实测：G5 `grad_hidden_bias FAIL mare=1.000e+00`（单元素全错）、G6 段 `mere=5.188e+01 nan_inf_mismatch=1`。
- **零填充 padding 区后再 padded 拷出**（banned）：`Duplicate(postBuf[p1c], 0, ...)` 的目的地址本身非 32B 对齐（p1c=104 → 208B），随后 G6 仍 ERROR（Duplicate 非对齐 dst 崩溃为推断归因）。
- **FloorAlign 截短 part1**（banned）：把拷出长度 floor 到对齐会丢尾元素，且按 `3H−aOff`（而非 part1Clamped）计算时把非跨界 chunk 的长度也改错——实测用例从 10 过退化到 4 过。

## 误诊

- "ReduceSum isReuseSource=true 的 bug"：改为 false 仍崩，排除。
- "blockLen 非 32B 对齐是根因"：第一轮对齐实验（只对齐 part2 blockLen → 不崩）曾支持此论，但同参数下 part1 段（源对齐、blockLen 272B 非对齐）不崩而 hidden 段（同 blockLen）崩的矛盾无法解释；part2-only 实验（源非对齐、blockLen 已对齐仍崩）把根因修正为**源地址对齐**。blockLen 的确切作用未定论（见遗留问题）。
- "MTE2_MTE3 事件类型用错（用 evMte3ToMte2_ 配 MTE2_MTE3 模板）"：确实存在该混用并修正（Base 新增 evMte2ToMte3_），但修正后仍崩——非根因。
- "PRINTF 显示 blk=3 死在 ReduceSum"：MTE3 异步队列导致打印位置与真实 fault 点错位，真实崩溃在异步拷出；PRINTF 结论只作范围参考。
- "workspace 指针可能指向 sys 区导致越界"：推理后放弃（Phase 1 写同一区域正常），未验证。

## 未决事项

- **优先项：复现并追根因到机制层**——`copy_ubuf_to_gm_align_b16` 对源地址/blockLen 对齐的确切硬件契约未读到实现级说明；尤其是"同参数（blockCount=1、blockLen=272B、源对齐）下 gmInBias_ 路径未崩而 gmHidBias_ 路径崩"的矛盾证据未解释。建议做最小复现 kernel，扫 {源对齐 × blockLen 对齐 × dst 张量} 矩阵。
- `DataCopy` UB→UB 源对齐约束为排除法归因，未单独最小复现。
- `Duplicate` 到非 32B 对齐 dst 是否必然 fault 未单独验证（agent 推断）。
- Reg::Load / LoadUnAlign 非对齐读路线未实测（调查后放弃）；若未来必须从非对齐 UB 偏移读数据，这是待验证方向。

## 复现增量：errcode 80 显式签名 + blockLen 澄清 + padded 行距布局修复路线

另一独立轨迹并入两个不同分支（RoPE half-BSND 与 interleaved）的同根因实例，扩展三件事：

**1. errcode 80 显式 plog 签名（dav-3510/CANN 9.2.0）**：本卡原实例（dav_c220/9.0.0-beta.2）表现为"无 kernel 报错、仅流同步失败"的静默 fault；新实例在 dav-3510 上有明确设备侧错误码：

```
The extend info: errcode:(80) errorStr: The MTE non-atomic instruction address is not aligned. subErrType: 0x4.
```

同一条 507035 下，80（MTE 非原子指令地址未对齐）/340（VEC 访 UB 未对齐）/95（MTE DDR 越界）对应完全不同的根因，先读 plog `errcode/errorStr` 再动手。

**2. blockLen 不要求对齐的实测澄清（关闭本卡未决事项）**：本卡原轨迹内对 blockLen 的作用存在矛盾证据并按"源对齐、blockLen 尽量对齐"双约束处理。新实例给出直接证据：**约束仅在源地址，搬运长度 blockLen 任意字节可过**——D=2 fp32 用例 `blockLen=4B` 通过；`CopyUbufToGmAlignV2` 实现里 `dstStride310 = AlignUp(dstStride*32 + burstLength, 32)` 只约束块间步进，单块 blockCount=1 时只需源地址对齐。"blockLen 非 32B 对齐 + 源对齐"的安全性由全量用例间接支持，仍非机制级结论。

**3. 修复路线二：padded 行距布局（paddedH=alignUp(H,16)）**：本卡原修复是"切分因子整除输出边界"；对逐行 CopyOut 形态（每行从 `o1[row*H]` 取源、`H×sizeof(T)` 非 32B 倍数时 row=1 即非对齐），更直接的修复是平面缓冲行距不用 H 而用 `paddedH = alignUp(H, 16)`（16 元素同时覆盖 fp32 的 8 元素/32B 与 fp16/bf16 的 16 元素/32B 粒度），使每行源偏移 `row × paddedH × sizeof(T)` 恒为 32B 倍数：

```cpp
    // Init 内：paddedH 与 UB 预算反推
    const uint32_t H = td_->Dh;
    paddedH_ = AlignUpU32(H, kAlignElems);   // kAlignElems = 16（32B 覆盖 fp32/fp16）
    if (paddedH_ < 1) paddedH_ = 1;

    const uint64_t usableUB = 196608ULL;     // usableUB ≈ 192KB
    uint64_t bytesPerRow;
    if constexpr (kIsFp32) {
        bytesPerRow = 7ULL * paddedH_ * sizeof(ComputeT);          // 7 个 fp32 平面缓冲
    } else {
        bytesPerRow = 6ULL * paddedH_ * sizeof(T) + 7ULL * paddedH_ * sizeof(ComputeT);
    }
    myUbRows_ = static_cast<uint32_t>(usableUB / bytesPerRow);
    if (myUbRows_ < 1) myUbRows_ = 1;
```

```cpp
    // 逐行 CopyOut（padded 行距，src 恒 32B 对齐；blockLen 任意字节即可）
    for (uint32_t row = 0; row < cnt; ++row) {
        uint64_t gmOff = static_cast<uint64_t>(rowStart + row) * D;
        int64_t ubOff = static_cast<int64_t>(row) * paddedH_;  // 32B-aligned
        AscendC::DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = H * sizeof(T);    // blockLen 任意字节即可
        ext.srcStride = 0;
        ext.dstStride = 0;
        AscendC::DataCopyPad(gm[static_cast<int64_t>(gmOff)], o1[ubOff], ext);
        AscendC::DataCopyPad(gm[static_cast<int64_t>(gmOff + H)], o2[ubOff], ext);
    }
```

padding 区（每行 `paddedH-H` 个元素）为垃圾值但不被 CopyOut，也不影响有效元素。interleaved 分支实例（同 trace 第二分支）进一步把该布局推广为 `paddedH/paddedD = PadUp(行元素数, 16)` 的通用多行平面布局：flat 紧排（cos/sin 搬到 `cosBuf[r*H]`、CopyOut 从 `out[row*D]` 取源）在 fp16 D<16/fp32 D<8 时除第 0 行外所有行基址都非 32B 对齐，形成"大 D 用例过、小 D 用例全挂 507035"的分化指纹；改 padded 布局后 507035 全部消失。配套细节：缓冲分配额外加 `kTailMargin=128` 元素余量，覆盖 `LoadAlign` 尾块按整寄存器（fp32 一次 64 元素）越读；计算链（DeInterleave/Muls/Mul/asc_vf_call）随之改 per-row 或 whole-chunk on `cnt*paddedH`。

**独立 trace 新增验证信号**（两个分支、同 trace，均为 tool_observed）：

- half 分支：修复前 ST 10/11 数值用例 507035，plog `errcode:(80) The MTE non-atomic instruction address is not aligned. subErrType: 0x4`；D=128 fp16 用例（128B 行距对齐）通过；paddedH=16 padded planar 后不再出现 80（该轮剩余 6/11 失败为 2D NDDMA 加载错位的纯精度问题，另卡；修复后 12/12 PASS）；D=2 fp32（blockLen=4B）通过证实 blockLen 不受约束（来源平台：950；证据等级：tool_observed）。
- interleaved 分支：修复前 D=128 大 shape 用例通过、D=2/4/6 的 10 个数值用例全部 `ACL error 507035`（大 D 过小 D 挂分化指纹）；paddedH/paddedD 平面布局重写后 507035 全部消失，numeric 10/11 通过（唯一失败为交错重组两半互换的独立根因，另卡闭环）（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界**：本卡原"GM→UB 方向无此约束"的观测限定于整缓冲、基址对齐的搬入；interleaved 分支的 flat 布局里 per-row NDDMA 搬入目的基址同样落在非 32B 偏移上（确切先触发的故障指令未逐条定位，行为级证据为布局前后 507035 翻转）——多行 UB 平面缓冲的搬运两侧基址都应按 padded 行距保证对齐。MTE2（GM→UB）目的侧对齐要求未单独最小复现（未验证）。
