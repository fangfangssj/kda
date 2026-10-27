---
type: "Runbook"
title: "__aicore__ 禁止 void* 转 __ubuf__ 桥接"
description: "dav-3510 上为混合 dtype Reg wrapper 用 void* 形参 + 内部 (__ubuf__ T*) 强转桥接 LocalTensor<half> 与 LocalTensor<float>，编译报 casting 'void *' to type '__ubuf__' not allowed in aicore function（5 处）；void* 桥接路线 banned，须改 const LocalTensor<T>& 形参 + 内部 GetPhyAddr()。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.language_extension"]
status: stable
sources: []
created_at: "2026-08-18T08:38:44Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["dav_3510"]
---

# __aicore__ 禁止 void* 转 __ubuf__ 桥接

## 现象

masked_scale 的 b16 路线需要 dst 为 `LocalTensor<half>`、x 为 `LocalTensor<half>`、mask 为 `LocalTensor<float>`（mask 在 plugin 侧已转 float32）。为绕开「一个模板 T 同时绑定 half 与 float」的推导冲突，尝试把 Reg wrapper 改成 `void*` 形参、内部再 `(__ubuf__ T*)` 强转：

```cpp
__aicore__ inline void masked_scale_reg_compute_float(
    void *dstAddr, void *xAddr, void *maskAddr,
    int64_t count, float scale)
{
    __ubuf__ float *d = (__ubuf__ float *)dstAddr;
    __ubuf__ float *x = (__ubuf__ float *)xAddr;
    __ubuf__ float *m = (__ubuf__ float *)maskAddr;
    ...
}
```

调用处把 `LocalTensor::GetPhyAddr()` 的返回转 `void*` 传入。`bash build.sh` 编译期报：

```
masked_scale_kernel.cpp:65:43: error: casting 'void *' to type '__ubuf__ float *' is not allowed in aicore function
masked_scale_kernel.cpp:66:43: error: casting 'void *' to type '__ubuf__ float *' is not allowed in aicore function
masked_scale_kernel.cpp:67:43: error: casting 'void *' to type '__ubuf__ float *' is not allowed in aicore function
masked_scale_kernel.cpp:79:35: error: casting 'void *' to type '__ubuf__ T *' is not allowed in aicore function
masked_scale_kernel.cpp:80:35: error: casting 'void *' to type '__ubuf__ T *' is not allowed in aicore function
```

## 适用场景

- **工程形态**：ops-direct-invoke-flash Reg kernel，`__aicore__`/`__simd_vf__` wrapper 与 `asc_vf_call` 传参。
- **API 路径**：`AscendC::LocalTensor<T>::GetPhyAddr()` → `__ubuf__ T*`；`asc_vf_call<...>(...)`。
- **验证过的平台与版本**：Ascend950PR（dav-3510），CANN 9.1.0。仅此平台实测。
- **适用粒度**：`operator_family`——任何在 dav-3510 上为混合 dtype Reg wrapper 选参策略的 kernel。
- **路线优先级**：`banned`——void* 桥接路线在 `__aicore__` 函数内编译期即被拒；正确路线见预防。
- **不适用/未验证场景**：a2/a3 未验证（`__aicore__` 地址空间约束是 AscendC 通用语义，推断可迁移但未实测）。

## 失败路线

被放弃的 void* 桥接 wrapper（banned，编译失败）：

```cpp
__aicore__ inline void masked_scale_reg_compute_float(
    void *dstAddr, void *xAddr, void *maskAddr,
    int64_t count, float scale)
{
    __ubuf__ float *d = (__ubuf__ float *)dstAddr;
    __ubuf__ float *x = (__ubuf__ float *)xAddr;
    __ubuf__ float *m = (__ubuf__ float *)maskAddr;
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t elementCount = static_cast<uint32_t>(count);
    uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
    asc_vf_call<MaskedScaleComputeFloatVf>(d, x, m, scale, elementCount, chunkCount);
}

template <typename T>
__aicore__ inline void masked_scale_reg_compute_b16(
    void *dstAddr, void *xAddr, void *maskAddr,
    int64_t count, float scale)
{
    __ubuf__ T *d = (__ubuf__ T *)dstAddr;
    __ubuf__ T *x = (__ubuf__ T *)xAddr;
    __ubuf__ float *m = (__ubuf__ float *)maskAddr;
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t elementCount = static_cast<uint32_t>(count);
    uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
    asc_vf_call<MaskedScaleComputeB16Vf<T>>(d, x, m, scale, elementCount, chunkCount);
}
```

调用处（同样 banned）：把 `LocalTensor::GetPhyAddr()` 的返回先转 `void*` 再传入：

```cpp
        using LocalT = AscendC::LocalTensor<T>;
        LocalT *xLT = reinterpret_cast<LocalT *>(&xLocal);
        using LocalF = AscendC::LocalTensor<float>;
        LocalF *maskLT = reinterpret_cast<LocalF *>(&maskF);
        LocalT *zLT = reinterpret_cast<LocalT *>(&zLocal);

        if constexpr (std::is_same<T, float>::value) {
            masked_scale_reg_compute_float(
                (void *)zLT->GetPhyAddr(), (void *)xLT->GetPhyAddr(),
                (void *)maskLT->GetPhyAddr(), count, scale_);
        } else {
            masked_scale_reg_compute_b16<T>(
                (void *)zLT->GetPhyAddr(), (void *)xLT->GetPhyAddr(),
                (void *)maskLT->GetPhyAddr(), count, scale_);
        }
```

被 ban 的原因：`(__ubuf__ float *)dstAddr` 等 5 处强转全部触发 `casting 'void *' to type '__ubuf__ ...' is not allowed in aicore function`。`__aicore__` 函数对地址空间有强类型约束：`void*` 到 `__ubuf__ T*` 的强转被编译器（bisheng/aicore 后端）拒绝，目的是防止跨地址空间（GM/UB/SM）的非法 reinterpret。`void*` 作为 kernel 入口参数合法，但一旦进入 `__aicore__` 内部 wrapper，就不能再用 `void*` 中转后强转回 `__ubuf__`。

边界与回退：void* 桥接路线 banned，整体回退；改让 wrapper 直接以 `const AscendC::LocalTensor<T>&` 形参接收队列出队的 tensor（dst、x 共用 T；mask 用独立固定类型 `const AscendC::LocalTensor<float>&`），在 wrapper 内部 `GetPhyAddr()` 取 `__ubuf__` 指针——地址空间转换由 LocalTensor 封装负责，不暴露 `void*` 中转。该路线编译推进到下一处错误，证明 void* 根因被消除。

## 验证信号

- void* 路线 `bash build.sh` 报 `error: casting 'void *' to type '__ubuf__ float *' is not allowed in aicore function`（5 处）（来源平台：950；证据等级：tool_observed）
- 改回 `const AscendC::LocalTensor<T>&` 形参后该组错误消失，编译推进到下一处错误，证明 void* 根因被该替代消除（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅 dav-3510 / CANN 9.1.0 实测；`__aicore__` 地址空间强转约束是 AscendC 通用语义，a2/a3 推断可迁移但未验证。
- 本卡只证明 void*→`__ubuf__` 强转在 `__aicore__` 内被拒；`__simd_vf__` 函数内 `__ubuf__` 形参本身合法（`asc_vf_call` 把 `__ubuf__` 指针传进去是正确路径），不要把两者混为一谈。

## 预防

不要用 `void*` 作 `__aicore__` wrapper 的中转类型来桥接不同 dtype 的 `LocalTensor`；直接用 `const LocalTensor<T>&` 形参，在 wrapper 内 `GetPhyAddr()` 取 `__ubuf__` 指针。混合 dtype 时让模板 T 只约束同 dtype 的 tensor 对（dst+x），不同 dtype 的入参（如 float mask）用独立固定类型：

```cpp
__aicore__ inline void masked_scale_reg_compute_float(
    const AscendC::LocalTensor<float> &dst,
    const AscendC::LocalTensor<float> &x,
    const AscendC::LocalTensor<float> &maskF,
    int64_t count, float scale)
{
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *xAddr = (__ubuf__ float *)x.GetPhyAddr();
    __ubuf__ float *maskAddr = (__ubuf__ float *)maskF.GetPhyAddr();
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t elementCount = static_cast<uint32_t>(count);
    uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
    asc_vf_call<MaskedScaleComputeFloatVf>(dstAddr, xAddr, maskAddr, scale, elementCount, chunkCount);
}
```