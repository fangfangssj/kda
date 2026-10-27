---
type: "Runbook"
title: "LoadDist 错用 DIST_NORM_B32 导致编译失败"
description: "Reg LoadAlign 误用 StoreDist 风格枚举 DIST_NORM_B32 时编译失败；fp32 常规加载应使用 LoadDist::DIST_NORM。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# LoadDist 错用 DIST_NORM_B32 导致编译失败

## 现象

在 Ascend950 Flash kernel 的 `__simd_vf__` 归一化函数中，需要从 UB 读取先前以 fp32 正常布局写入的标量和权重。代码把 `StoreDist::DIST_NORM_B32` 的命名习惯照搬到 `LoadDist`，远端 CANN 9.0.1 构建报：

```text
error: no member named 'DIST_NORM_B32' in 'AscendC::Reg::LoadDist'; did you mean 'DIST_BRC_B32'?
```

同一编译输出同时还出现其他独立错误；本卡只处理 `LoadDist` 枚举错用这一根因。

## 适用场景

- 工程形态：Ascend950 kernel 直调 / Flash `.asc` 构建；
- API 路径：`AscendC::Reg::LoadAlign<float, LoadDist::...>` 从 UB 加载 fp32 正常布局；
- 验证过的平台与版本：950（Ascend950PR_957c）/ CANN 9.0.1；
- 适用粒度：`platform_wide`，适用于同版本 dav-3510 Reg 加载分布枚举；
- 路线优先级：`recommended_fastpath`；
- 明确不适用：`StoreAlign` 的分布枚举、需要广播加载的 `DIST_BRC_B32` 场景，以及未验证的其他 CANN 版本。

## 根因

`LoadDist` 与 `StoreDist` 是两套独立枚举：CANN 9.0.1 的 fp32 正常加载使用 `LoadDist::DIST_NORM`，而 `DIST_NORM_B32` 只存在于 `StoreDist`，不能用于 `LoadAlign`。

机制链为：正常 fp32 数据由 `StoreAlign<..., StoreDist::DIST_NORM_B32>` 写入 UB → 重新读取时元素类型已经是 `float` → `LoadAlign<float, LoadDist::DIST_NORM>` 按正常连续分布加载；编译器提示的 `DIST_BRC_B32` 表示广播分布，不是同义替换。

## 解决配方

逐个检查 `LoadAlign` 的模板枚举类型；凡是 `float` 正常连续加载，把错误的 `LoadDist::DIST_NORM_B32` 改成 `LoadDist::DIST_NORM`，保留 `StoreAlign` 侧的 `StoreDist::DIST_NORM_B32`。

### before

代码证据：path `operators/rms_norm/rms_norm.asc`；symbol `normalize_vf<T>`；支撑“LoadDist 错用了仅 StoreDist 拥有的枚举”。源码仓固定 Git revision/blob URL 未包含在轨迹中，curate 前须补证。

```cpp
template <typename T>
__simd_vf__ inline void normalize_vf(__ubuf__ T *yAddr, __ubuf__ T *xAddr, __ubuf__ float *gAddr,
                                      __ubuf__ float *rstdAddr, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> rstdReg, xFReg, gReg;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM_B32>(rstdReg, rstdAddr);
    AscendC::Reg::Duplicate(rstdReg, rstdReg, maskFull);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg tail = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xFReg, xAddr + offset);
            AscendC::Reg::Mul(xFReg, xFReg, rstdReg, tail);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM_B32>(gReg, gAddr + offset);
            AscendC::Reg::Mul(xFReg, xFReg, gReg, tail);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, xFReg, tail);
        } else {
            AscendC::Reg::RegTensor<T> xReg, yReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + offset);
            AscendC::Reg::Cast<float, T, traitB16ToB32>(xFReg, xReg, tail);
            AscendC::Reg::Mul(xFReg, xFReg, rstdReg, tail);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM_B32>(gReg, gAddr + offset);
            AscendC::Reg::Mul(xFReg, xFReg, gReg, tail);
            AscendC::Reg::Cast<T, float, traitB32ToB16>(yReg, xFReg, tail);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, yReg, tail);
        }
    }
}
```

### after

代码证据：path `operators/rms_norm/rms_norm.asc`；symbol `normalize_vf<T>`；支撑“普通 fp32 LoadAlign 改用 LoadDist::DIST_NORM，而 StoreDist 保持不变”。源码仓固定 Git revision/blob URL 未包含在轨迹中，curate 前须补证。

```cpp
template <typename T>
__simd_vf__ inline void normalize_vf(__ubuf__ T *yAddr, __ubuf__ T *xAddr, __ubuf__ float *gAddr,
                                      __ubuf__ float *rstdAddr, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> rstdReg, xFReg, gReg;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(rstdReg, rstdAddr);
    AscendC::Reg::Duplicate(rstdReg, rstdReg, maskFull);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg tail = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xFReg, xAddr + offset);
            AscendC::Reg::Mul(xFReg, xFReg, rstdReg, tail);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(gReg, gAddr + offset);
            AscendC::Reg::Mul(xFReg, xFReg, gReg, tail);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, xFReg, tail);
        } else {
            AscendC::Reg::RegTensor<T> xReg, yReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + offset);
            AscendC::Reg::Cast<float, T, traitB16ToB32>(xFReg, xReg, tail);
            AscendC::Reg::Mul(xFReg, xFReg, rstdReg, tail);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(gReg, gAddr + offset);
            AscendC::Reg::Mul(xFReg, xFReg, gReg, tail);
            AscendC::Reg::Cast<T, float, traitB32ToB16>(yReg, xFReg, tail);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, yReg, tail);
        }
    }
}
```

## 设计决策

- 选择 `LoadDist::DIST_NORM`，因为目标是普通连续 fp32 读取，且同一工具包的已验证 Reg 示例使用这一形态；
- 放弃编译器近似建议 `DIST_BRC_B32`，因为它会改变加载分布为广播，并非只修正名称；
- 当前方案属于 `recommended_fastpath`：它恢复 API 契约，不增加额外搬运或分支。

## 预防

- 写 Reg load/store 模板时先确认模板参数类型是 `LoadDist` 还是 `StoreDist`，不要按字符串相似度互抄；
- 编译器给出相近枚举建议时，先查枚举定义和同类型示例，确认分布语义而非只求编译通过；
- code review 同时核对 `LoadAlign<float, LoadDist::DIST_NORM>` 与 `StoreAlign<float, StoreDist::DIST_NORM_B32>` 的不对称命名。

## 验证信号

本知识来自一次 Ascend950 Reg kernel 构建：修正三个普通 fp32 LoadAlign 调用后，下一次构建不再报告 `DIST_NORM_B32` 属于 LoadDist 的错误，并继续暴露后续独立错误。

- CANN 9.0.1 编译器明确报告 `LoadDist` 不含 `DIST_NORM_B32`（950，证据等级: tool_observed）；
- 工具包枚举查询显示 `StoreDist` 含 `DIST_NORM_B32`，而安装示例使用 `LoadDist::DIST_NORM` 做普通 fp32 加载（950，证据等级: tool_observed）；
- 完成全部独立修复后目标 `libop_rms_norm.so` 构建成功（950，证据等级: tool_observed）。

## 失效边界

- 需要把单个 B32 元素广播到寄存器的场景应检查 `DIST_BRC_B32`，不能机械使用本配方；
- CANN 9.0.1 之外版本的枚举集合未验证；
- A2/A3 未验证，不能从本轨迹推断其支持相同 Reg API。

## 坏实践

- 看到编译器建议 `DIST_BRC_B32` 就直接替换：可以改变数据分布语义，即使编译通过也可能产生静默错误；
- 把 `LoadDist` 与 `StoreDist` 当成同一枚举空间。

## 误诊

无证据支持把该错误归因于 include 缺失；编译器已经定位为 `LoadDist` 枚举成员不存在。