---
type: "Runbook"
title: "AR 归约读未初始化 pad 行 NaN 污染"
description: "三趟大 R 路径 CopyIn 只写有效行，对齐补出的 pad 行是未初始化 UB，AR pattern ReduceMax/ReduceSum 按 padded srcShape 全量读取，NaN 进入归约致 max/sum 损坏、大 R 用例精度失败；全 tile 预填归约恒等元 -inf 修复。"
tags: [task.debugging, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# AR 归约读未初始化 pad 行 NaN 污染

## 现象

在 Ascend950PR（CANN 9.2.0）上开发自定义 softmax 算子（bench_harness_softmax，Branch-0 NormalTailR）的黑盒 ST：排除 oracle 误报后停在 **38/46 precision PASSED**，剩余 8 个失败全部走三趟（大 R）路径，覆盖四组 shape：

- attention fp16/bf16 `[2,8,64,512]`（R=512，rLoopCnt=2）
- classification fp16/bf16 `[32,1000]`（R=1000，rLoopCnt=2）
- large_reduce fp16/fp32/bf16 `[1,131072]`（rLoopCnt=71/142）
- max_reduce fp32 `[1,2097152]`（rLoopCnt≈1131/2263）

全部 `first fail @ elem 0`，编译与 aclnn 调用无任何报错——纯输出值错。

在 kernel 三趟路径逐层加内核 PRINTF（block 0、首个 A-loop）观测：

- Pass 1 结束后行 max `m0` 卡在第一个 R 块的 partial（如 `-0.003084`），不再随后续 R 块增长；行 sum `s0` 出现 `416794.6`、`26123.5` 这类远超 `Σexp(x-m) ≤ R` 上界的发散值——max 和 sum 都坏。
- 直接打印 ReduceMax 输出 tmpBuf0 的前几个 lane：`rm p0=-0.000531 p1=nan p2=nan p3=nan`——p0（有效行）看似正常，p1-p3（pad 行）是 NaN。

当时的 `CopyIn` 只对 `validARows` 行做逐行 DataCopyPad（`paddingValue=-inf` 只补**有效行内**的 R 尾到块对齐），`aUbFactorAlign` 对齐补出的 pad 行（validARows..aUbFactorAlign-1）从不写入，是未初始化 UB；而 `DoReduceMax/DoReduceSum` 以 padded srcShape `{aUbFactorAlign, rUbFactorAlign}` 全量读取做 AR pattern 归约。

## 适用场景

- 工程形态：aclnn 自定义算子 kernel 开发 + 黑盒 ST 精度修复（`./run.sh` 全量用例）。
- API 路径：AscendC `ReduceMax/ReduceSum<float, Pattern::Reduce::AR, isReuseSource>(dst, src, sharedTmpBuffer, srcShape, srcInnerPad)`、`DataCopyPad`（DataCopyPadExtParams + paddingValue）、`Duplicate`、`PipeBarrier<PIPE_ALL>`；A 行/R 列按 DATABLOCK 对齐（fp32 aUbFactorAlign 为 8 的倍数、fp16 为 16 的倍数）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter platforms 一致）。
- 适用粒度：operator_family——一切"按对齐尺寸分配 UB tile、按有效行列数搬运、再对 padded shape 做 pattern 归约"的 reduce 类 kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：快表路径（R 全载无 pad 行）；直接对原始输入求和且无 exp 变换的 sum（预填 -inf 会污染，见失效边界）；ReduceMin（应预填 +inf，未实测）。

## 根因

CopyIn 只写有效行，对齐补出的 pad 行是未初始化 UB（位型按 float 解释常为 NaN），AR pattern ReduceMax/ReduceSum 按 padded srcShape 全量读取，未初始化数据进入归约，max/sum 结果被污染。

机制链：

- A 行按 DATABLOCK 对齐：`aUbFactor=1`（如 `[1,131072]` fp32）时 `aUbFactorAlign=8`，补 7 个 pad 行；fp16 对齐到 16 的倍数。DataCopyPad 的 `paddingValue` 只覆盖有效行内的 R 尾，不覆盖 pad 行，也不覆盖有效行内"DataCopyPad 自身补齐长度到 rUbFactorAlign"之间的缝隙。
- AR 归约对 srcShape[0]=aUbFactorAlign 行全部求值：pad 行的未初始化位型解释为 NaN（观测 `p1..p3=nan`）。
- NaN 进入 max/累加语义后比较恒假（`(pVal > mVal) ? pVal : mVal` 对 NaN 恒取 mVal），运行 max 不再前进（`m0` 卡在首块 partial）；sum 侧 `exp(NaN-m)=NaN` 使累加发散（`s0=416794.6 > R`）。
- 预填 -inf 后：-inf 是 max 的恒等元；sum 在 Sub+Exp 之后，`exp(-inf-m)=0` 也是恒等元——pad 区参与归约不再改变结果。

## 解决配方

`CopyIn` 在逐行 DataCopyPad 之前，先用 `Duplicate` 把整个对齐 tile 预填 -inf，并加 `PipeBarrier` 保证写可见（kernel 源码不动其他部分）。

修复前（只写有效行，pad 区未初始化；行内搬运循环为最终形态，修复点在其之前）：

```cpp
    __aicore__ inline void CopyIn(int64_t aLoopIdx, int64_t rIdx, int64_t validARows)
    {
        int64_t validRCols = ValidRCols(rIdx);
        if (validARows <= 0 || validRCols <= 0) return;
        T padVal = NegInfDType();
        auto dstTensor = preInBuf_.Get<T>();
        for (int64_t row = 0; row < validARows; ++row) {
            int64_t gmOff = GmOffset(aLoopIdx, rIdx) + row * td_->axisStride[0];
            AscendC::DataCopyExtParams extParams;
            extParams.blockCount = 1;
            extParams.blockLen = static_cast<uint32_t>(validRCols * sizeof(T));
            extParams.srcStride = 0;
            extParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = true;
            padParams.paddingValue = padVal;
            AscendC::DataCopyPad(dstTensor[row * td_->rUbFactorAlign],
                                 gmX_[gmOff], extParams, padParams);
        }
    }
```

修复后（先全 tile 预填 -inf，再逐行搬运覆盖有效区）：

```cpp
    __aicore__ inline void CopyIn(int64_t aLoopIdx, int64_t rIdx, int64_t validARows)
    {
        int64_t validRCols = ValidRCols(rIdx);
        // Pre-fill entire tile with -inf to avoid NaN from uninitialized pad rows.
        {
            auto t = preInBuf_.Get<T>();
            AscendC::Duplicate(t, NegInfDType(),
                static_cast<int32_t>(td_->aUbFactorAlign * td_->rUbFactorAlign));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if (validARows <= 0 || validRCols <= 0) return;
        T padVal = NegInfDType();
        auto dstTensor = preInBuf_.Get<T>();
        for (int64_t row = 0; row < validARows; ++row) {
            // …逐行 DataCopyPad 同修复前…
        }
    }
```

`NegInfDType()` 按 dtype 构造 -inf（bf16 用位桥，不用标量 cast——编译器不支持）：

```cpp
        } else {  // bfloat16_t
            uint16_t bits = 0xFF80u;
            union { uint16_t u; bfloat16_t b; } cv; cv.u = bits; return cv.b;
        }
```

判别探针：打印归约输出 dst 的前 aUbFactorAlign 个 lane——pad 行 lane 出 NaN 即中招；行 sum 用 `s0 > R` 上界判据比"max 应该多大"可靠（用例输入范围不同时 partial 接近 0 未必异常）。

## 设计决策

- 为什么预填整个 tile 而不是只填 pad 行：`Duplicate` 整 tile 是一次向量操作，不引入有效行数分支；有效区随后被 DataCopyPad 覆盖，-inf 只残留在 pad 区。
- 为什么填 -inf 而不是 0：0 会伪造 max（输入可正可负，全负行 max 会被 0 顶掉）；-inf 是 max 恒等元，且经 Sub+Exp 后 `exp(-inf)=0` 对 sum 也是恒等元。若某算子的 sum 直接对原始输入求和（无 exp 变换），pad 区应改填 0。
- 为什么加 `PipeBarrier<PIPE_ALL>`：Duplicate（V 流水）写 UB 后需全流水屏障再进 DataCopyPad/归约，保证预填可见——与本 kernel 其他段间屏障用法一致。
- 防御性保留（非本根因、随本轮落地的相邻修改）：tiling UB 预算公式扣减 `VF_PAD_TOTAL = 4×256 + DATABLOCK_BYTES`（UB 分配确实含 4 个 tile 槽的 vfPad 开销）；cacheBuf 由 MS_STRIDE 跨步 Duplicate 改连续 SetValue 布局；group 触发强制走 key=0 单核三趟（group Phase1/Phase2 未实现）。三者对通过率均无影响（38/46 不动），保留为正确性/简化改进。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（56 AIV），CANN 9.2.0，bisheng；ST 入口 `./run.sh`（46 用例），UT 入口 tests/ut（51 用例）。
- 相关文件：`operators/BenchHarnessSoftmax_package/op_kernel/arch35/bench_harness_softmax_kernel.h`（CopyIn/DoReduceMax/DoReduceSum/AccumulateMax/AccumulateSum/NegInfDType）、`op_host/arch35/bench_harness_softmax_tiling_arch35.cpp` 与 `op_host/bench_harness_softmax_tiling_host.cpp`（VF_PAD_TOTAL 预算）、`tests/ut/test_branch0_tiling.cpp`（UT oracle 同步）。

## 预防

- 任何"按对齐尺寸分配 UB、按有效行列搬运"的 kernel，在 pattern 归约/全 tile 读取之前必须把 pad 区预填为归约恒等元（max→-inf、min→+inf、sum→0 或经变换后的恒等元）。
- review 检查点：`InitBuffer` 尺寸用的对齐因子 vs `DataCopyPad` 实际写入范围之差（= 未初始化区）；`srcShape` 传的是 align 值还是 valid 值、`srcInnerPad` 与之是否配套。

## 验证信号

以下均来自该轮黑盒 ST/UT 全量复验：

- 修复前 38/46（8 个三趟用例失败，first fail @ elem 0）（来源平台：950；证据等级：tool_observed）。
- 探针观测：ReduceMax 输出 pad 行 lane 为 NaN（`p1=nan p2=nan p3=nan`）、行 max 卡首块 partial、行 sum 发散（`s0=416794.6`，上界应为 R）（来源平台：950；证据等级：tool_observed）。
- 预填 -inf 后同探针显示 `p1=-inf p2=-inf p3=-inf`，ST 全量 **46/46 precision PASSED、46 aclnn-OK、0 aclnn-FAIL**（来源平台：950；证据等级：tool_observed）。
- 清理调试 PRINTF 后复验：build SUCCESS、46/46、UT 51/51、`grep -cE '__simd_vf__|asc_vf_call'`=12（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 预填值必须是该归约的恒等元：max→-inf；Sub+Exp 之后的 sum→-inf（经 exp 变 0）；**直接对原始输入求和的 sum 不能填 -inf**（会把和打成 -inf），应填 0；ReduceMin 应填 +inf（未实测）。
- 快表路径（R 全载、无 pad 行）不需要也不触发本修复。
- 未验证，推断可迁移：其他 pattern（SA 等）的 padded srcShape 读取同理存在未初始化区风险；其他架构未实测。

**未决项**：

- **优先项**：NaN 从 pad 行传播到有效行累加值（m 卡死首块 partial / s 发散）的确切实现路径——AR 归约内部 dst 布局、sharedTmpBuffer 复用还是跨行合并——未追到 CCE 实现层；可用最小 repro（1×N fp32、aUbFactorAlign=8、仅写 1 行）复现后读 `ReduceARImpl`/反汇编确认。
- 预填 -inf 的性能代价（每块多一次全 tile Duplicate + PipeBarrier）未测量；若成瓶颈可评估只填 pad 行或归约前按有效行掩码。
## 坏实践

以下负尝试均实测，供避坑（全部回退）：

- 改用 4 参 `ReduceMax`（不带显式 sharedTmpBuffer，内部 PopStackBuffer）：**36/46 且 6 个 aclnn-FAIL**（疑似共享缓冲不足导致崩溃），回退 5 参形式。
- 用有效 `rUbFactor` + `srcInnerPad=false`：**27/46**——API 不知行 stride，按连续内存读，把 pad 列当数据，快表用例也坏；必须 `rUbFactorAlign` + `srcInnerPad=true`。
- `isReuseSource=true`：无变化（38/46 不动）。
- 修改 tiling UB 预算（VF_PAD_TOTAL 扣减）试图修精度：无效果——UB 溢出不是本失败的原因（作为防御性修正保留，但别指望它修精度）。
- 数值判断前不确认用例输入分布：曾按"uniform [-5,5]、行 max 应接近 5"断言 partial 异常，实际 attention 等用例输入范围不同（观测到 x0=-2021、-34 等）；`s0 > R` 这类**上界判据**比"max 应该多大"可靠。

## 误诊

- 曾按"UB 预算溢出"假设修 tiling——通过率不动；真实根因是 pad 区未初始化。
- 曾怀疑 ReduceMax 输出带 stride/打印位置错——打印 p0..p3 排除（p0 数值合理、pad 行才是 NaN）。
