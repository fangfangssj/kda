---
type: "Runbook"
title: "Reg::Reduce 跨 chunk 覆写导致规约失真"
description: "dav-3510 Reg::Reduce 每次只产当前 chunk 的标量结果；循环复用 dstReg 会丢失前序 chunk，须显式折叠 lane 0。"
tags: ["task.debugging", "topic.compute.reduction", "topic.programming_model.execution_unit.vector", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Reg::Reduce 跨 chunk 覆写导致规约失真

## 现象

在 Ascend950 Softmax kernel 已成功编译后运行完整 NPU pytest，单个 Vector chunk 能覆盖的末轴规约用例通过，而规约长度跨多个 chunk 的末轴用例在 float16、float32、bfloat16 上均失败。失败输出整体偏大，观测到的 `Max diff` 落在 `0.414062~0.927876`，测试汇总为 55 passed、6 failed。

当时代码在 `softmax_max_vf` 和 `softmax_expsum_vf` 中把同一个 `acc`/`sumAcc` 作为每次 `Reg::Reduce` 的目标，假定多次调用会在旧目标值上继续累加。

## 适用场景

- 工程形态：Ascend C kernel 直调，`__simd_vf__` 内对长向量分 chunk 做标量规约；
- API 路径：`AscendC::Reg::Reduce<MAX/SUM>` + `LoadAlign` + 单 lane `StoreAlign`；
- 验证过的平台与版本：Ascend950PR_957b / dav-3510，CANN 9.0.1；
- 适用粒度：`operator_family`，适用于任何用多次 Reg Reduce 拼接长向量标量规约的算子；
- 路线优先级：`recommended_fastpath`，每 chunk 独立 Reduce，再用 lane-0 mask 显式折叠；
- 未验证场景：单 chunk 规约不会暴露该错误；跨核规约、Cube 规约及其他 CANN 版本未验证。

## 根因

当前 dav-3510 实现把 `Reduce<MAX>`/`Reduce<SUM>` 下沉为 `vcmax(dstReg, srcReg, mask, ...)`/`vcadd(dstReg, srcReg, mask, ...)`。每次调用产生的是当前 chunk 的规约结果，并写入目标寄存器，而不是把目标旧值当作跨调用累加器。 若共有 `K` 个 chunk，错误写法得到： ```text m_wrong = reduce_max(chunk[K-1]) s_wrong = reduce_sum(exp(chunk[K-1] - m_wrong)) ``` 正确全行规约应为： ```text m = max(reduce_max(chunk[0]), ..., reduce_max(chunk[K-1])) s = reduce_sum(exp(chunk[0]-m)) + ... + reduce_sum(exp(chunk[K-1]-m)) ``` 因此错误实现只按最后一个 chunk 的最大值和指数和归一化整行；分母被系统性低估，输出整体偏大。单 chunk 时 `K=1`，错误式与正确式相同，所以会出现“短规约通过、长规约失败”的强判别信号。

## 解决配方

每个 chunk 先 Reduce 到临时寄存器 `tmp` 的 lane 0，再用仅 lane 0 有效的 `scalarMask` 对运行累加器执行 `Max` 或 `Add`。最终只 Store lane 0 到 32B 标量槽。

### before

证据定位：revision 为轨迹内上板失败前源码态（40 位 Git SHA 未由轨迹暴露）；path=`operators/softmax/softmax.asc`；symbols=`softmax_max_vf`,`softmax_expsum_vf`；lines=48-89。两段完整函数支撑“跨 chunk 复用 Reduce 目标寄存器”的结论。

```cpp
__simd_vf__ inline void softmax_max_vf(__ubuf__ float *maxSlot, __ubuf__ float *xF,
                                        uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> acc;
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::Duplicate<float>(acc, -std::numeric_limits<float>::infinity());
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t off = chunk * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xF + off);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float>(acc, xReg, mask);
    }
    uint32_t oneElem = 1u;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(maxSlot, acc, scalarMask);
}

__simd_vf__ inline void softmax_expsum_vf(__ubuf__ float *sumSlot, __ubuf__ float *xF,
                                           __ubuf__ float *maxSlot, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> mReg, mbroad, xReg, tReg, eReg, sumAcc;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(mReg, maxSlot);
    AscendC::Reg::Duplicate<float>(mbroad, mReg, fullMask);   // broadcast lane 0 (LOWEST)
    AscendC::Reg::Duplicate<float>(sumAcc, 0.0f);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t off = chunk * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xF + off);
        AscendC::Reg::Sub(tReg, xReg, mbroad, mask);
        AscendC::Reg::Exp(eReg, tReg, mask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float>(sumAcc, eReg, mask);
    }
    uint32_t oneElem = 1u;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sumSlot, sumAcc, scalarMask);
}
```

### after

证据定位：revision 为轨迹内上板通过源码态（40 位 Git SHA 未由轨迹暴露）；path=`operators/softmax/softmax.asc`；symbols=`softmax_max_vf`,`softmax_expsum_vf`；lines=53-95。两段完整函数支撑“每 chunk Reduce 后显式折叠 lane 0”的结论。

```cpp
__simd_vf__ inline void softmax_max_vf(__ubuf__ float *maxSlot, __ubuf__ float *xF,
                                        uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t oneElem = 1u;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::RegTensor<float> acc, tmp, xReg;
    AscendC::Reg::Duplicate<float>(acc, -std::numeric_limits<float>::infinity());
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t off = chunk * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xF + off);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float>(tmp, xReg, mask); // tmp lane0 = chunk max
        AscendC::Reg::Max(acc, acc, tmp, scalarMask);                                // acc lane0 = max(acc, tmp)
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(maxSlot, acc, scalarMask);
}

__simd_vf__ inline void softmax_expsum_vf(__ubuf__ float *sumSlot, __ubuf__ float *xF,
                                           __ubuf__ float *maxSlot, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t oneElem = 1u;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::RegTensor<float> mReg, mbroad, xReg, tReg, eReg, sumAcc, tmp;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(mReg, maxSlot);
    AscendC::Reg::Duplicate<float>(mbroad, mReg, fullMask);   // broadcast lane 0 (LOWEST)
    AscendC::Reg::Duplicate<float>(sumAcc, 0.0f);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t off = chunk * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xF + off);
        AscendC::Reg::Sub(tReg, xReg, mbroad, mask);
        AscendC::Reg::Exp(eReg, tReg, mask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float>(tmp, eReg, mask); // tmp lane0 = chunk sum
        AscendC::Reg::Add(sumAcc, sumAcc, tmp, scalarMask);                         // sumAcc lane0 += tmp
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sumSlot, sumAcc, scalarMask);
}
```

## 设计决策

- 采用“chunk 内 Reduce、chunk 间标量折叠”，保留硬件 Reduce 对单 chunk 的优势，同时明确跨 chunk 状态；
- `scalarMask` 只打开 lane 0，避免 `tmp` 其余未定义 lane 污染累加器；
- 最大值累加器初始化为负无穷，求和累加器初始化为 0，分别保持 `max` 与 `+` 的单位元；
- 没有改测试容差、输入分布或参考实现；当前方案属于 `recommended_fastpath` 的正确 Reg 路线，但轨迹未比较其他正确规约实现的性能。

## 系统信息

- toolchain：CANN 9.0.1，BiSheng/Clang 15.0.5；
- 目标：Ascend950PR_957b，NpuArch=dav-3510；
- 工程：kernel 直调，fp16/float32/bfloat16 均在 fp32 路径做 max、exp、sum、div；
- 相关文件：`operators/softmax/softmax.asc`、`operators/softmax/test_softmax.py`；
- 头文件证据：`kernel_reg_compute_vec_reduce_impl.h` 中 ReduceMax/ReduceSum 下沉到 `vcmax`/`vcadd`。

## 预防

- 只要规约长度可能超过一个 Vector chunk，就禁止把同一个 `dstReg` 连续传给 `Reg::Reduce` 并假设其累加；
- 测试矩阵必须同时包含单 chunk 与多 chunk 规约，三种 dtype 均覆盖；
- 看到“短规约全过、长规约整体偏大”时，先核查跨 chunk max/sum 是否显式折叠，不先调容差。

## 验证信号

这条知识来自一次 Ascend950 Softmax kernel 的真实上板修复。修复前同一测试矩阵为 55 passed、6 failed，多 chunk 失败的 `Max diff` 为 `0.414062~0.927876`；替换为临时 Reduce + lane-0 `Max/Add` 后，完整 NPU pytest 为 61 passed。（来源平台：950；证据等级：tool_observed）

随后执行独立的全量 cann-bench 精度评测，全部 20 个 case 通过，覆盖 float16、float32、bfloat16 与末轴/非末轴路径。（来源平台：950；证据等级：tool_observed）

## 失效边界

- 单 chunk 规约不会暴露该故障，但使用显式折叠仍保持正确；
- 跨核规约还需要额外的核间归并，本配方只解决单个 `__simd_vf__` 内的跨 chunk 规约；
- A2/A3 与其他 CANN 版本未验证，推断可迁移前必须确认对应 Reduce 实现语义；
- 轨迹没有提供与其他正确规约路线的性能对比，不能把本卡写成通用性能收益结论。

## 坏实践

- 把 `Reg::Reduce(dst, src, mask)` 当作“会读取 dst 旧值”的累加器 API；
- 只测试单 chunk 规约后就宣称长向量路径正确；
- 通过放宽 `atol/rtol` 掩盖数量级明显的归一化错误。

## 误诊

失败最初可能被归到 dtype Cast 或非末轴搬运路径；但“所有 dtype 同时失败、单 chunk 通过、多 chunk 失败”的组合信号更直接指向共享的跨 chunk 标量规约。（证据等级: inferred，不作为验证信号）