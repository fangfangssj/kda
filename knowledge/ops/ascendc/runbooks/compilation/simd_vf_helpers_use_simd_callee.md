---
type: "Runbook"
title: "simd_vf 互调导致 helper 编译失败"
description: "__simd_vf__ 函数不能直接调用另一个 __simd_vf__ helper；把被调 helper 标为 __simd_callee__ 后调用约束诊断消失。"
tags: ["task.debugging", "topic.programming_model.execution_model.simd", "topic.implementation.api_layer.simt_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# simd_vf 互调导致 helper 编译失败

## 现象

在 Ascend950 Reg 直调 kernel 中，多个 `__simd_vf__` 顶层包装器复用一个加载并可选 Cast 的 helper。远端编译在每个调用点报 `no matching function for call to 'load_fp32_chunk'`，候选说明为：

```text
candidate function not viable: simd_vf function can only call simd_callee function
```

helper 的函数体可以正常解析，失败发生在调用层级属性检查。

## 适用场景

- 工程形态：Ascend950 direct-launch / Reg 三层封装；
- API 路径：`asc_vf_call<top_vf>` → `__simd_vf__` 顶层函数 → 可复用 helper；
- 验证过的平台与 CANN 版本：Ascend950PR_957b / CANN 9.0.1；
- 适用粒度：`platform_wide`，适用于同一编译模型下的 SIMD helper 组织；
- 路线优先级：`recommended_fastpath`；
- 明确不适用：普通 host C++ helper、`__aicore__` 调度层、不是从 `__simd_vf__` 内部调用的函数。

## 根因

`__simd_vf__` 是由 `asc_vf_call` 进入的顶层 VF 函数属性，被另一个 `__simd_vf__` 直接调用不满足编译器调用约束；可复用的 VF 内部 helper 必须声明为 `__simd_callee__`。

## 解决配方

函数签名与函数体保持不变，只把被调 helper 的属性从 `__simd_vf__` 改成 `__simd_callee__`。

### before

代码来源：修复前 `operators/softmax/softmax.asc` file-write 快照；symbol `load_fp32_chunk<T>`；函数全体。该段支撑“helper 被错误标为顶层 VF”的结论。

```cpp
template <typename T>
__simd_vf__ inline void load_fp32_chunk(AscendC::Reg::RegTensor<float> &dst,
                                         __ubuf__ T *src, uint32_t off)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(dst, (__ubuf__ float *)src + off);
    } else {
        AscendC::Reg::RegTensor<T> sReg;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(sReg, src + off);
        AscendC::Reg::Cast<float, T, CAST_UP>(dst, sReg, AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>());
    }
}
```

### after

代码来源：同一 helper 的 annotation 修复快照；symbol 与函数体不变。该段支撑“内部 helper 改为 callee 属性”的解决配方。

```cpp
template <typename T>
__simd_callee__ inline void load_fp32_chunk(AscendC::Reg::RegTensor<float> &dst,
                                         __ubuf__ T *src, uint32_t off)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(dst, (__ubuf__ float *)src + off);
    } else {
        AscendC::Reg::RegTensor<T> sReg;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(sReg, src + off);
        AscendC::Reg::Cast<float, T, CAST_UP>(dst, sReg, AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>());
    }
}
```

## 设计决策

1. 顶层函数继续保留 `__simd_vf__` 并由 `asc_vf_call` 进入；
2. 复用加载逻辑标为 `__simd_callee__`，避免复制到每个顶层包装器；
3. 当前方案属于 `recommended_fastpath`，没有增加额外指令。

## 预防

- 画出三层调用关系：`__aicore__` wrapper → `asc_vf_call<__simd_vf__>` → `__simd_callee__` helpers；
- 新增 VF helper 后先做一个顶层包装器的最小编译；
- 看到多个调用点同时报同一 helper `no matching function` 时，先读 candidate note，而不是逐个改模板实参。

## 验证信号

来源说明：远端构建先在多个顶层 VF 调用点复现同一 annotation 诊断；修改 helper 后立即重编。

- 下一次编译不再出现 `simd_vf function can only call simd_callee function`，编译器继续报告互不相关的 Mask/PIPE/std::min 错误（证据等级：tool_observed；来源平台：950）。

## 失效边界

- 验证范围是 CANN 9.0.1 / dav-3510 Reg 直调编译模型；
- 本修复只解决 VF 调用层级，不保证 helper 内部 API 参数已满足各自契约；
- 其他架构未验证，推断可迁移。

## 坏实践

- 把所有 SIMD 函数都标成 `__simd_vf__`，忽略顶层入口与内部 callee 的区别；
- 因多个调用点报错而复制 helper 函数体，造成逻辑分叉。

## 误诊

错误文本表面是 `no matching function`，但候选 note 已明确不是模板类型推导失败，而是函数属性调用约束。