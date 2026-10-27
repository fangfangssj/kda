---
type: "Runbook"
title: "2D NDDMA 小内维落位错位致精度失配"
description: "用 NdDmaParams dim=2（loopSize={H,cnt}）加载小 H（<32B 搬运粒度）的行分片时数据落位与 addr=base+Σidx*stride 模型不符，kernel 正常运行但小 D 用例数值失配（符号级错误、无任何错误码）；改为逐行 1D NDDMA 每行显式定址即修复。"
tags: [task.debugging, topic.precision, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 2D NDDMA 小内维落位错位致精度失配

## 现象

在 Ascend950PR（CANN 9.2.0，dav-3510）上调试自定义 RoPE 算子（half-BSND-cos2d 分支）kernel 的第三轮迭代：前两轮的 VEC 对齐崩溃（errcode 340）与 MTE3 对齐崩溃（errcode 80）均已消除，kernel 已能正常运行出数。但 ST 12 条用例只有 5/11 数值用例通过，6 条失败且全是**数值失配**（不再有任何 507035/崩溃）：

```
[a1_fp32] FAILED: queryOut mismatch at idx=34 actual=0.309741 expected=-0.0385935 (dtype=fp32 ... rtol=0.000122 atol=0.000122)
[a2_fp16] FAILED: queryOut mismatch at idx=34 actual=-0.238159 expected=0.30957 (dtype=fp16 ...)
[a3_bf16] FAILED: queryOut mismatch at idx=36 actual=0.470703 expected=-0.347656 (dtype=bf16 ...)
[b1_b_broadcast] FAILED: queryOut mismatch at idx=98 ...
[b5_d6_unaligned] FAILED: queryOut mismatch at idx=57 ...
[c3_query_inf] FAILED: queryOut mismatch at idx=34 ...
```

失配值连符号都错（actual=0.31 vs expected=-0.039），不是舍入级差异。通过/失败的分界与 D 强相关：D=4（a1/a2/a3/b1/c3）、D=6（b5）失败；D=2（b2_d2_min）、D=128（b4_large_fp16）及平凡输入用例（全 0、NaN 传播、常数大值）通过。当时 kernel 的输入加载统一用 **2D NDDMA**：

```cpp
    __aicore__ inline void LoadPlanarHalf(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
        AscendC::GlobalTensor<T>& gm, uint32_t rowStart, uint32_t cnt, uint32_t offsetInRow)
    {
        const uint32_t D = td_->D;
        const uint32_t H = td_->Dh;
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        AscendC::NdDmaParams<T, 2> params;
        params.loopInfo.loopSrcStride[0] = 1;   // innermost contiguous
        params.loopInfo.loopSrcStride[1] = D;   // GM row stride
        params.loopInfo.loopDstStride[0] = 1;
        params.loopInfo.loopDstStride[1] = paddedH_;  // UB planar row stride (padded)
        params.loopInfo.loopSize[0] = H;
        params.loopInfo.loopSize[1] = cnt;
        params.loopInfo.loopLpSize[0] = 0; params.loopInfo.loopLpSize[1] = 0;
        params.loopInfo.loopRpSize[0] = 0; params.loopInfo.loopRpSize[1] = 0;
        params.constantValue = T(0);
        AscendC::DataCopy<T, 2, cfg>(
            buf.Get<T>()[0], gm[static_cast<int64_t>(rowStart) * D + offsetInRow], params);
    }
```

cos/sin 的广播加载（`srcStride[1]=0` 重读同 H 个元素）同样是 2D NDDMA。计算链（Muls/Mul/asc_vf_call<MulAddVF>）与 CopyOut 均已在前两轮验证为对齐正确。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（aclnn 工程），用 NDDMA（`DataCopy<T, dim, cfg>` + `NdDmaParams`）做 GM→UB 的多维/跨步加载。
- API 路径：`AscendC::NdDmaParams<T,2>` / `DataCopy<T,2,cfg>`（MTE2 管线）；`dav_3510/kernel_operator_data_copy_impl.h` 的 `CopyGmToUbufAlignV2` 路径。
- 验证过的平台与 CANN 版本：Ascend950PR（dav-3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——dav-3510 上所有"最内维 loopSize 小于 32B 粒度（fp32<8 / fp16<16 元素）+ 外维跨步"的 2D NDDMA 加载。
- 路线优先级：recommended_fastpath（逐行 1D NDDMA 语义无歧义；性能敏感场景可再验证 2D 语义后切换）。
- 明确不适用 / 未验证场景：最内维 ≥32B 粒度的 2D NDDMA（D=128 fp16 用例在 2D 加载下通过）未观察到问题；H=1（D=2 用例）在 2D 加载下意外通过，未解释；其他架构未验证（推断可迁移）。

## 根因

对最内维 `loopSize[0]=H` 小于 32B 搬运粒度（fp32 H<8 / fp16 H<16 元素）的 2D NDDMA，实际数据落位与文档模型 `addr = base + Σ(index_d × stride_d)` 不一致，行分片（q1/q2）与 cos/sin 广播加载把数据搬错位；kernel 计算链本身正确，错在输入就已错位，故表现为纯精度失配（无崩溃、无对齐异常）。

确切机制（最小突发粒度约束？stride 单位语义？loopLpSize/loopRpSize 缺省值含义？）**未逐层验证**——修复为经验性替换，见遗留问题。

判别信号：失配按 D 分界（D=4/6 错、D=2/128 对）、失配幅度达符号级（非舍入级）、无任何运行时错误码——与"计算链 bug"（通常整块错）和"对齐崩溃"（507035）都不同。

## 解决配方

原则：小 H 的行分片/广播加载不用 2D 跨步 NDDMA，改用**逐行 1D NDDMA**，每行显式计算源/目的地址，语义无歧义。

修复后（q1/q2 行分片加载）：

```cpp
    __aicore__ inline void LoadPlanarHalf(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
        AscendC::GlobalTensor<T>& gm, uint32_t rowStart, uint32_t cnt, uint32_t offsetInRow)
    {
        const uint32_t D = td_->D;
        const uint32_t H = td_->Dh;
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        // 1D NDDMA per row (avoids 2D stride ambiguity for small H)
        AscendC::NdDmaParams<T, 1> params;
        params.loopInfo.loopSrcStride[0] = 1;
        params.loopInfo.loopDstStride[0] = 1;
        params.loopInfo.loopSize[0] = H;
        params.loopInfo.loopLpSize[0] = 0;
        params.loopInfo.loopRpSize[0] = 0;
        params.constantValue = T(0);
        for (uint32_t r = 0; r < cnt; ++r) {
            AscendC::DataCopy<T, 1, cfg>(
                buf.Get<T>()[static_cast<int64_t>(r) * paddedH_],
                gm[static_cast<int64_t>(rowStart + r) * D + offsetInRow], params);
        }
    }
```

修复后（cos/sin 按 `s=(absRow/N)%S` 逐行广播加载）：

```cpp
    __aicore__ inline void LoadCosSinPlanar(uint32_t rowStart, uint32_t cnt)
    {
        const uint32_t H = td_->Dh;
        const uint32_t N = td_->N;
        const uint32_t S = td_->S;
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        AscendC::NdDmaParams<T, 1> params;
        params.loopInfo.loopSrcStride[0] = 1;
        params.loopInfo.loopDstStride[0] = 1;
        params.loopInfo.loopSize[0] = H;
        params.loopInfo.loopLpSize[0] = 0;
        params.loopInfo.loopRpSize[0] = 0;
        params.constantValue = T(0);

        // Per-row: load cos[s(r)] / sin[s(r)] (H elements, broadcast across N)
        for (uint32_t r = 0; r < cnt; ++r) {
            uint32_t absRow = rowStart + r;
            uint32_t s = (absRow / N) % S;
            AscendC::DataCopy<T, 1, cfg>(
                bufCosH_.Get<T>()[static_cast<int64_t>(r) * paddedH_],
                gmCos_[static_cast<int64_t>(s) * H], params);
            AscendC::DataCopy<T, 1, cfg>(
                bufSinH_.Get<T>()[static_cast<int64_t>(r) * paddedH_],
                gmSin_[static_cast<int64_t>(s) * H], params);
        }
    }
```

注意：本次修复**没有**增加 padding 清零（每行 padding 仍是垃圾值），单独替换 NDDMA 维度即全绿——反证此前失败不是垃圾 padding 污染，而是 2D 加载把有效数据搬错位。

## 设计决策

- 为什么选逐行 1D 而不是继续调 2D 参数：2D 语义在文档模型与实测行为间存在未定位的偏差，调参试错成本高于显式定址；逐行 1D 每次调用的源/目的地址都由代码显式计算，行为确定。
- 被放弃的备选：`DataCopyPad(DataCopyExtParams)` 跨步块加载（`blockCount=cnt, blockLen=H*sizeof(T), srcStride=(D-H)*sizeof(T)`）——其 UB 侧 `dstStride310=AlignUp(dstStride*32+burstLength,32)` 的粒度换算使 paddedH 行距难以精确表达，且未实测；先给平面缓冲整体清零再 2D NDDMA（当时的主假设"垃圾 padding 污染"）——被修复实验否定（未清零也全绿）。
- 当前方案路线档位：recommended_fastpath（正确性优先）；逐行 1D 的 MTE2 指令条数增多，性能影响未测量。

## 预防

- 按文档模型 `addr = base + Σ index_d × stride_d` 推 2D NDDMA 落位而不实测——小内维下模型与硬件行为可能不一致，错位直接表现为精度失配且无任何错误码提示。
- 精度失配时先怀疑计算链/舍入——本例计算链完全正确，先做"通过/失败按 shape 参数分界"的判别（D=4/6 错、D=2/128 对）能快速指向加载器。

## 验证信号

来源说明：以下信号来自该算子 Branch-0 kernel 第三轮迭代的精度闭环（ST 失配 → 排查加载器 → 换逐行 1D NDDMA → 重建复跑全绿）。

- 修复前（2D NDDMA）：ST 数值 5/11——a1_fp32 idx=34 actual=0.309741 vs expected=-0.0385935、a2_fp16 idx=34、a3_bf16 idx=36、b1 idx=98、b5 idx=57、c3 idx=34 失配；b2_d2_min（D=2）、b4_large_fp16（D=128）、c1/c2/c4（NaN/全0/常数）通过（来源平台：950；证据等级：tool_observed）。
- 修复后（逐行 1D NDDMA，未加清零）：同套件 ST 12/12 全绿（numeric 11/11 PASSED + returns_empty 1/1 OK）；TilingUT 45/45 回归通过；`asc_vf_call/__simd_vf__` grep 7 处命中（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 逐行 1D NDDMA 的指令条数 = 行数×缓冲数，超大行数场景的 MTE2 下发开销未测量；若成为瓶颈需回验 2D 语义（见遗留问题）。
- 观察范围限于 H<8（fp32）/H<16（fp16）量级的小内维；H=1 意外通过、H=2/3 失败、H=64 通过的完整边界曲线未测绘。
- 其他架构（非 dav-3510）的 NDDMA 语义未验证（推断可迁移）。

**未决项**：

- **优先项：复现并追根因到机制层**——构造最小 2D NDDMA 用例（loopSize={2,cnt}，srcStride={1,4}，fp32），dump UB 实际落位与期望逐元素对比，定位偏差来自最小突发粒度、stride 单位还是 loopLpSize/loopRpSize 缺省语义；并解释 H=1 为何幸存。
- 2D NDDMA 在 H≥8（fp32）/16（fp16）时是否严格正确未单独验证（本会话只有 H=64 一个通过样本）。
- 逐行 1D 相对 2D NDDMA 的性能代价未测量。
## 坏实践

- 小内维 2D NDDMA 只按文档模型推落位就上 ST——错位无错误码，先按 shape 分界判别再定位加载器。
- 把"精度失配 + 无崩溃"默认归因计算链/舍入，不加最小复现就改算法。

## 误诊

- 调试初期主假设为"平面缓冲 padding 区垃圾值污染有效结果"，为此设计了清零方案（Muls 清零 / DataCopyPad rightPadding=0 填充）；最终修复未做任何清零即全绿，反证 padding 无辜——失败全部来自 2D NDDMA 有效数据落位错误。
