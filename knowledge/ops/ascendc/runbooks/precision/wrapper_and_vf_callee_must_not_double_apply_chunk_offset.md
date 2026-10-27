---
type: "Runbook"
title: "分块 wrapper 与 VF callee 不可重复叠加 offset"
description: "asc_vf_call 已传入按 chunk 前移的 UB 指针时，VF callee 再加同一 offset 会从第二块开始按双倍偏移读错数据。"
tags: ["task.debugging", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 分块 wrapper 与 VF callee 不可重复叠加 offset

## 现象

在 Ascend 950 / `dav-3510` 上把 Softmax 的 GM↔UB staging 改成正确的 `TQue` 后，部分小输入已经正确，但需要多个 Vector chunk 的输入仍然失败。`operators/softmax/softmax.asc` 的 `ComputeLoadX` 在循环中把 `fAddr + offset` 和 `xAddr + offset` 传给 `asc_vf_call`，同时又把数值 `offset` 作为参数传给 `softmax_loadx_vf`；callee 随后再次执行 `xAddr + offset`。

这个现象只在第二个及后续 chunk 暴露：第一个 chunk 的 `offset=0`，重复相加仍是零，容易让单 chunk 测试产生“实现正确”的假象。

## 适用场景

- 工程形态：Ascend C Reg kernel 直调，以 `__aicore__` wrapper 循环调用 `__simd_vf__` callee。
- API 路径：`LocalTensor::GetPhyAddr` → chunk pointer arithmetic → `asc_vf_call` → `LoadAlign`/`StoreAlign`。
- 验证过的平台与版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致。
- 适用粒度：`operator_family`，适用于 wrapper/callee 两层都可能负责分块寻址的 Reg 算子。
- 路线优先级：`recommended_fastpath`，每一层地址只前移一次；本实现选择 wrapper 传已前移指针、callee 按零基址访问。
- 明确不适用：callee 接收原始 base pointer 并独占 offset 计算的实现；这种契约也可正确，但不能与本卡的 wrapper 契约混用。

## 根因

设第 `k` 个 chunk 的元素偏移为 `o = k × vectorLength`。错误实现的 wrapper 先传入 `xAddr + o`，callee 又从收到的指针取 `+ o`，实际地址为： ```text (xAddr + o) + o = xAddr + 2o ``` 当 `k=0` 时 `o=0`，地址碰巧正确；当 `k≥1` 时读取位置越过目标 chunk。输出地址虽然已在 wrapper 前移，但 callee 的输入 Load 再前移，导致 load/store 对应关系断裂，并可能访问行内错误区域或越过有效行数据。

## 解决配方

固定单一寻址责任：wrapper 传已按 chunk 前移的 `fAddr`/`xAddr`；callee 删除 `offset` 参数，对收到的指针直接 `LoadAlign`。调用点也删除多余实参。

### before

revision: trace snapshot before the double-offset fix
path: `operators/softmax/softmax.asc`
symbol: `softmax_loadx_vf`
lines: 38-57

```cpp
template <typename T>
__simd_vf__ inline void softmax_loadx_vf(__ubuf__ float *fAddr, __ubuf__ T *xAddr,
                                          uint32_t offset, uint32_t remaining)
{
    AscendC::Reg::RegTensor<float> fReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(fReg, xAddr + offset);
    } else {
        AscendC::Reg::RegTensor<T> srcReg;
        static constexpr AscendC::Reg::CastTrait traitUnpack = {
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg, xAddr + offset);
        AscendC::Reg::Cast<float, T, traitUnpack>(fReg, srcReg, mask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(fAddr, fReg, mask);
}
```

revision: trace snapshot before the double-offset call-site fix
path: `operators/softmax/softmax.asc`
symbol: `ComputeLoadX`
lines: 171-185

```cpp
template <typename T>
__aicore__ inline void ComputeLoadX(const AscendC::LocalTensor<float> &fBuf,
                                    const AscendC::LocalTensor<T> &xBuf, int64_t rLen)
{
    __ubuf__ float *fAddr = (__ubuf__ float *)fBuf.GetPhyAddr();
    __ubuf__ T *xAddr = (__ubuf__ T *)xBuf.GetPhyAddr();
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t elementCount = static_cast<uint32_t>(rLen);
    uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = elementCount - offset;
        asc_vf_call<softmax_loadx_vf<T>>(fAddr + offset, xAddr + offset, offset, remaining);
    }
}
```

### after

revision: trace snapshot after the double-offset fix
path: `operators/softmax/softmax.asc`
symbol: `softmax_loadx_vf`
lines: 39-58

```cpp
template <typename T>
__simd_vf__ inline void softmax_loadx_vf(__ubuf__ float *fAddr, __ubuf__ T *xAddr,
                                          uint32_t remaining)
{
    AscendC::Reg::RegTensor<float> fReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(fReg, xAddr);
    } else {
        AscendC::Reg::RegTensor<T> srcReg;
        static constexpr AscendC::Reg::CastTrait traitUnpack = {
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg, xAddr);
        AscendC::Reg::Cast<float, T, traitUnpack>(fReg, srcReg, mask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(fAddr, fReg, mask);
}
```

revision: trace snapshot after the double-offset call-site fix
path: `operators/softmax/softmax.asc`
symbol: `ComputeLoadX`
lines: 171-185

```cpp
template <typename T>
__aicore__ inline void ComputeLoadX(const AscendC::LocalTensor<float> &fBuf,
                                    const AscendC::LocalTensor<T> &xBuf, int64_t rLen)
{
    __ubuf__ float *fAddr = (__ubuf__ float *)fBuf.GetPhyAddr();
    __ubuf__ T *xAddr = (__ubuf__ T *)xBuf.GetPhyAddr();
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t elementCount = static_cast<uint32_t>(rLen);
    uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = elementCount - offset;
        asc_vf_call<softmax_loadx_vf<T>>(fAddr + offset, xAddr + offset, remaining);
    }
}
```

## 设计决策

- 选择“wrapper 前移指针、callee 零基访问”，因为 wrapper 已负责计算 `chunk`、`offset` 和 `remaining`，地址责任集中在同一层更容易审计。
- 另一种可行但未采用的契约是把 base pointer 与 offset 都传给 callee；若采用它，wrapper 必须停止前移指针。
- 当前方案保留 tail mask 的 `remaining` 参数，只删除与已前移指针重复的 offset。

## 预防

- 为每个 `asc_vf_call` 写出有效地址代数；调用前后的 offset 总和必须恰好出现一次。
- 测试至少覆盖一个需要两个以上 chunk 的输入；单 chunk 永远无法暴露 `0+0` 的重复偏移。
- wrapper/callee 接口命名可使用 `baseAddr` 或 `chunkAddr` 明示契约，避免二者混淆。

## 验证信号

来源是同一次 Ascend 950 远端构建和诊断。修改后构建成功，固定输入与覆盖单 chunk、多 chunk、多行和非末轴的诊断全部报告 `ok=True`；完整测试由 18/49 通过改善到 43/49 通过，剩余失败只出现在另一个 host 布局物化根因上。

- 修复后多 chunk 诊断输出与参考逐项一致，所有诊断 case 为 `ok=True`（证据等级: tool_observed；来源平台: 950）。
- 保留队列同步实现并只修复双重 offset 后，完整测试从 18/49 改善到 43/49 通过（证据等级: tool_observed；来源平台: 950）。

## 失效边界

本卡只处理“同一 offset 在两层重复叠加”。若错误来自字节偏移与元素偏移单位混用、尾块 `remaining` 计算或跨行 stride，需单独定位。A2/A3 未验证，推断可迁移。

## 坏实践

只用单 chunk 输入验证 Reg wrapper；第一个 chunk 的 offset 为零，重复寻址不会暴露。

## 误诊

队列同步修复后仍有较大输入失败，继续叠加 barrier 是误诊；剩余失败严格跟 chunk 数相关，应该先检查地址代数。