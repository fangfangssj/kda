---
type: "Runbook"
title: "Reg 尾块 masked Add 清零跨块累加器"
description: "Reg 默认 ZEROING 下用 tail mask 更新跨块累加器会清空未激活 lane；先掩码平方、再 full mask 累加可保留历史和。"
tags: ["task.debugging", "topic.compute.reduction", "topic.implementation.masking", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Reg 尾块 masked Add 清零跨块累加器

## 现象

在 Ascend950 `dav-3510` 上运行一个按最后一维分块平方求和的 RegBase Kernel 时，构建已经成功，但 NPU pytest 出现稳定的形状选择性错误：带非完整尾块的两组输入在 fp16、fp32、bf16 三种 dtype 下全部失败，共 6 个失败；没有尾块的 16 个测试通过。失败输出的幅值约为参考结果的 7–8 倍，工具日志还报告了明显的大差值。

当时代码对每个 chunk 都执行：

```text
Mul(sqReg, xReg, xReg, mask)
Add(accReg, accReg, sqReg, mask)
```

其中最后一个 chunk 的 `mask` 只激活尾部有效元素。故障只随“是否有尾块”变化，而不随 dtype 变化，定位到跨 chunk 累加与 mask 合并语义。

## 适用场景

- 工程形态：Ascend950 Flash Kernel 直调，RegBase SIMD 跨 chunk 归约；
- API 路径：`UpdateMask<float>` → masked `Mul` → masked `Add` → `Reduce<SUM>`；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于任何把 `RegTensor` lane 作为跨 chunk 累加器、且末 chunk 使用部分 mask 的平方和/求和归约；
- 路线优先级：`recommended_fastpath`，让产生临时值的指令处理 tail mask，再用 full mask 把零填充临时值累加到历史 accumulator；
- 明确不适用：指令显式配置 `MERGING` 且已验证未激活 lane 保留目标值，或每个 chunk 独立 Reduce 后只累加标量的实现。

## 根因

Reg masked compute 默认使用 `MaskMergeMode::ZEROING`，所以尾块的 `Add(accReg, accReg, sqReg, tailMask)` 不只是忽略无效 lane，而是把 `accReg` 的未激活 lane 清零，破坏此前完整 chunk 已累积的跨块状态。 以向量宽度为 64 lane、尾块仅 1 lane 有效为例： ```text 完整 chunk 累加后：acc = [a0, a1, ..., a63] 尾块 Mul + ZEROING：sq = [t0, 0, ..., 0] 尾块 Add + tailMask + ZEROING：acc = [a0+t0, 0, ..., 0] Reduce<SUM>(acc)：只剩约 1/64 的历史平方和 rstd = 1 / sqrt(mean)：缩放约 sqrt(64) = 8 倍 ``` 这与工具观察到的输出幅值约 7–8 倍一致。

## 解决配方

保留 `Mul(..., tailMask)`，利用 ZEROING 把临时 `sqReg` 的无效 lane 置零；随后 `Add` 改用 `fullMask`，无效 lane 实际执行 `acc + 0`，从而保留历史累加值。以下代码为 trace 恢复的函数级 verbatim before/after。

### before

revision：构建已通过、首次 NPU pytest 前的 trace 工作树（汇总提交 40 位 SHA 未由 trace 返回）；path：`operators/rms_norm/rms_norm.asc`；symbol/line range：`rmsNormTileVf<T>` 73–146；支撑结论：tail mask 同时用于临时平方与跨块 accumulator 更新。

```cpp
template <typename T>
__simd_vf__ inline void rmsNormTileVf(__ubuf__ T *yAddr, __ubuf__ T *xAddr,
                                      __ubuf__ T *gammaAddr, uint32_t mT,
                                      uint32_t D, uint32_t dAlign,
                                      float eps, float invD)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint32_t chunkCount = (D + vectorLength - 1) / vectorLength;

    // FAIL-002: UpdateMask takes a MUTABLE uint32_t& (not const, not literal).
    uint32_t fullElems = vectorLength;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(fullElems);
    uint32_t oneElem = 1;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);

    AscendC::Reg::RegTensor<float> accReg, xReg, gReg, sqReg, yReg, rstdB;
    AscendC::Reg::RegTensor<T> xRawReg, gRawReg, yRawReg;

    for (uint32_t r = 0; r < mT; ++r) {
        __ubuf__ T *xRow = xAddr + r * dAlign;
        __ubuf__ T *yRow = yAddr + r * dAlign;

        // ---- reduce: acc = sum of x^2 over D (linear chunk reduction) ----
        AscendC::Reg::Duplicate<float>(accReg, 0.0f); // whole reg = 0
        for (uint32_t c = 0; c < chunkCount; ++c) {
            uint32_t offset = c * vectorLength;
            uint32_t remaining = D - offset; // named lvalue (FAIL-002)
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xRow + offset);
            } else {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xRawReg, xRow + offset);
                AscendC::Reg::Cast<float, T, kCastB16ToB32>(xReg, xRawReg, mask);
            }
            AscendC::Reg::Mul(sqReg, xReg, xReg, mask);
            AscendC::Reg::Add(accReg, accReg, sqReg, mask);
        }

        // ---- rstd = 1 / sqrt(mean(x^2) + eps), in-register (no scalarBuf) ----
        // varReg holds the reduce scalar in lane 0; later ops use scalarMask.
        AscendC::Reg::RegTensor<float> varReg, sqrtReg, oneReg, rstdReg;
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(varReg, accReg, fullMask);
        AscendC::Reg::Muls(varReg, varReg, invD, scalarMask);   // var = sum * (1/D) = mean
        AscendC::Reg::Adds(varReg, varReg, eps, scalarMask);   // var = mean + eps
        AscendC::Reg::Sqrt(sqrtReg, varReg, scalarMask);        // sqrt(var)
        AscendC::Reg::Duplicate<float>(oneReg, 1.0f, scalarMask);
        AscendC::Reg::Div(rstdReg, oneReg, sqrtReg, scalarMask); // rstd = 1 / sqrt(var)
        // broadcast rstd (lane 0) to all 64 lanes for the per-chunk y multiply
        AscendC::Reg::Duplicate<float>(rstdB, rstdReg, fullMask);

        // ---- y = x * rstd * gamma, cast at store boundary ----
        for (uint32_t c = 0; c < chunkCount; ++c) {
            uint32_t offset = c * vectorLength;
            uint32_t remaining = D - offset; // named lvalue (FAIL-002)
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xRow + offset);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(gReg, gammaAddr + offset);
                AscendC::Reg::Mul(yReg, xReg, rstdB, mask);
                AscendC::Reg::Mul(yReg, yReg, gReg, mask);
                AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yRow + offset, yReg, mask);
            } else {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xRawReg, xRow + offset);
                AscendC::Reg::Cast<float, T, kCastB16ToB32>(xReg, xRawReg, mask);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gRawReg, gammaAddr + offset);
                AscendC::Reg::Cast<float, T, kCastB16ToB32>(gReg, gRawReg, mask);
                AscendC::Reg::Mul(yReg, xReg, rstdB, mask);
                AscendC::Reg::Mul(yReg, yReg, gReg, mask);
                AscendC::Reg::Cast<T, float, kCastB32ToB16>(yRawReg, yReg, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yRow + offset, yRawReg, mask);
            }
        }
    }
}
```

### after

revision：精度修复后的最终 trace 工作树（提交操作成功，但 40 位 SHA 未由 trace 返回）；path：`operators/rms_norm/rms_norm.asc`；symbol/line range：`rmsNormTileVf<T>` 73–150；支撑结论：tail mask 只清理临时平方值，跨块累加使用 full mask。

```cpp
template <typename T>
__simd_vf__ inline void rmsNormTileVf(__ubuf__ T *yAddr, __ubuf__ T *xAddr,
                                      __ubuf__ T *gammaAddr, uint32_t mT,
                                      uint32_t D, uint32_t dAlign,
                                      float eps, float invD)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint32_t chunkCount = (D + vectorLength - 1) / vectorLength;

    // FAIL-002: UpdateMask takes a MUTABLE uint32_t& (not const, not literal).
    uint32_t fullElems = vectorLength;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(fullElems);
    uint32_t oneElem = 1;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);

    AscendC::Reg::RegTensor<float> accReg, xReg, gReg, sqReg, yReg, rstdB;
    AscendC::Reg::RegTensor<T> xRawReg, gRawReg, yRawReg;

    for (uint32_t r = 0; r < mT; ++r) {
        __ubuf__ T *xRow = xAddr + r * dAlign;
        __ubuf__ T *yRow = yAddr + r * dAlign;

        // ---- reduce: acc = sum of x^2 over D (linear chunk reduction) ----
        AscendC::Reg::Duplicate<float>(accReg, 0.0f); // whole reg = 0
        for (uint32_t c = 0; c < chunkCount; ++c) {
            uint32_t offset = c * vectorLength;
            uint32_t remaining = D - offset; // named lvalue (FAIL-002)
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xRow + offset);
            } else {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xRawReg, xRow + offset);
                AscendC::Reg::Cast<float, T, kCastB16ToB32>(xReg, xRawReg, mask);
            }
            AscendC::Reg::Mul(sqReg, xReg, xReg, mask);
            // Add with FULL mask: the masked Mul above zeroed sqReg's inactive
            // lanes (ZEROING is the default merge mode), so adding them is a
            // no-op. Using the tail mask here would ZERO accReg's inactive
            // lanes and destroy the cross-chunk accumulator (breaks non-mul-64 D).
            AscendC::Reg::Add(accReg, accReg, sqReg, fullMask);
        }

        // ---- rstd = 1 / sqrt(mean(x^2) + eps), in-register (no scalarBuf) ----
        // varReg holds the reduce scalar in lane 0; later ops use scalarMask.
        AscendC::Reg::RegTensor<float> varReg, sqrtReg, oneReg, rstdReg;
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(varReg, accReg, fullMask);
        AscendC::Reg::Muls(varReg, varReg, invD, scalarMask);   // var = sum * (1/D) = mean
        AscendC::Reg::Adds(varReg, varReg, eps, scalarMask);   // var = mean + eps
        AscendC::Reg::Sqrt(sqrtReg, varReg, scalarMask);        // sqrt(var)
        AscendC::Reg::Duplicate<float>(oneReg, 1.0f, scalarMask);
        AscendC::Reg::Div(rstdReg, oneReg, sqrtReg, scalarMask); // rstd = 1 / sqrt(var)
        // broadcast rstd (lane 0) to all 64 lanes for the per-chunk y multiply
        AscendC::Reg::Duplicate<float>(rstdB, rstdReg, fullMask);

        // ---- y = x * rstd * gamma, cast at store boundary ----
        for (uint32_t c = 0; c < chunkCount; ++c) {
            uint32_t offset = c * vectorLength;
            uint32_t remaining = D - offset; // named lvalue (FAIL-002)
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xRow + offset);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(gReg, gammaAddr + offset);
                AscendC::Reg::Mul(yReg, xReg, rstdB, mask);
                AscendC::Reg::Mul(yReg, yReg, gReg, mask);
                AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yRow + offset, yReg, mask);
            } else {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xRawReg, xRow + offset);
                AscendC::Reg::Cast<float, T, kCastB16ToB32>(xReg, xRawReg, mask);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gRawReg, gammaAddr + offset);
                AscendC::Reg::Cast<float, T, kCastB16ToB32>(gReg, gRawReg, mask);
                AscendC::Reg::Mul(yReg, xReg, rstdB, mask);
                AscendC::Reg::Mul(yReg, yReg, gReg, mask);
                AscendC::Reg::Cast<T, float, kCastB32ToB16>(yRawReg, yReg, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yRow + offset, yRawReg, mask);
            }
        }
    }
}
```

## 设计决策

- 选择“tail mask 产临时值 + full mask 更新状态”，因为 `sqReg` 的未激活 lane 已被 ZEROING 变成 0，`acc + 0` 能保持历史 lane；
- 没有改成 `MERGING` trait：这条轨迹未展示所有二元指令对自定义 merge trait 的可用模板形态，而 full-mask Add 已在现有 API 形态下验证；
- 没有改为每 chunk 立即 Reduce 标量：那会改变归约拓扑和指令数；
- 当前方案是 `recommended_fastpath`，同时保持正确性与原有跨 chunk 向量累加拓扑。

## 系统信息

- 工具链：CANN 9.0.1，Bisheng ASC 编译器；
- 目标：Ascend950PR / `dav-3510`；
- 相关文件：`operators/rms_norm/rms_norm.asc`；
- 相关 API：`UpdateMask`、`Mul`、`Add`、`Reduce<SUM>`、`MaskMergeMode::ZEROING`。

## 预防

- 把 RegTensor 分成“临时值”和“跨迭代状态”两类：tail mask 可用于前者，更新后者前必须核对 merge 语义；
- 用尾块存在/不存在成对用例测试跨 chunk 归约，且覆盖所有 dtype；
- 看到“对齐维通过、非对齐维错误”时，先查尾块 mask 对目的寄存器未激活 lane 的处理。

## 验证信号

来源说明：该结论来自 Ascend950 上真实 NPU pytest 的失败—修复闭环；修复前只有带归约尾块的用例失败，精确替换 `Add` 的 mask 后在同一测试集重建并复测。

- 修复前：6 个尾块相关测试失败、16 个通过，失败覆盖三种 dtype（来源平台：950；证据等级: tool_observed）。
- 修复后：构建成功，完整 NPU pytest 为 22/22 通过（来源平台：950；证据等级: tool_observed）。
- 随后的完整 cann-bench 正确性为 20/20 通过（来源平台：950；证据等级: artifact_verified）。

- 独立复现 R1：尾块 mask 用于 Add(accReg, accReg, sqReg, mask) 时，默认 ZEROING 会清零 accumulator 的未激活 lane；改成 masked Mul 生成零填充 sqReg，再用 fullMask 做 Add 后，22 个 NPU pytest 全部通过。（950，tool_observed）。
- 独立复现 R2：Reg::Add 默认 MaskMergeMode::ZEROING；跨 chunk 累加时对 Add 使用部分尾掩码会清零 accReg 掩码外通道，导致非 VL 整除输入出错。保留 Mul 的 tail 掩码并把 Add 改用 maskFull 后，跨 dtype 对齐/非对齐诊断全部通过。（950，tool_observed）。
- 独立复现 R3：LAST 求和从跨 chunk lane-wise Add+末次 Reduce 改为每 chunk masked Reduce<SUM>+lane0 标量累加；重编译成功，随后 pytest 为 91 passed、18 skipped。（950，tool_observed）。

## 失效边界

- 若前一条临时值指令不是 ZEROING，`sqReg` 的无效 lane 可能残留旧值，此时 full-mask Add 会把垃圾值加进 accumulator；必须先显式清零或确认 merge 模式；
- 若 accumulator 只在 scalar lane 中维护，应按其实际 lane 布局重新分析；
- A2/A3 未在本轨迹验证，不能直接宣称 mask 默认值与模板形态相同。

## 坏实践

- 把同一个 tail mask 机械地传给整条计算链：对无状态临时结果正确，不代表对跨迭代状态正确；
- 看到输出放大就调 epsilon、容差或参考实现：这会掩盖累加器被清零的结构性错误；
- 只测试对齐维度，无法触发尾块 merge 语义。

## 误诊

初看像是平方和、均值或倒数平方根的数值精度问题；但故障只由尾块触发且约呈 `sqrt(vector_width)` 量级放大，更符合跨 lane 累加状态被清零，而非 Cast 或 epsilon 误差。