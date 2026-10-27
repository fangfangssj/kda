---
type: "Runbook"
title: "InitOutput GM 专用，UB 清零用 Duplicate"
description: "AscendC InitOutput(GlobalTensor<T>, uint32_t, T) 是 GM 专用清零 API，把 LocalTensor 作为第一参数传入会触发 'no known conversion from LocalTensor to GlobalTensor'；UB LocalTensor 清零应改用 Duplicate(local, value, mask)。判别依据为编译器候选函数 note 显示 InitOutput 唯一签名第一参数为 GlobalTensor<T>。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# InitOutput GM 专用，UB 清零用 Duplicate

## 现象

在 `MhcPreSinkhornBackward` arch35 确定性 kernel 的 `PhaseB_KDim` 入口（`op_kernel/arch35/mhc_pre_sinkhorn_backward_deterministic.h`），需要把 `gradInvRmsBuf_`（UB 上的 `TBuf` 取出的 `LocalTensor<U>`）在每个核的 bs 范围内清零。一度写成：

```cpp
if (idxBs == 0) {
    auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();
    InitOutput<U>(gradInvRmsUb, bsPerCore_, U(0));
}
```

在 ascend950 上编译：

```
bash build.sh --pkg --ops=mhc_pre_sinkhorn_backward --soc=ascend950
```

报 1 处错误（编译器候选函数 note）：

```
/home/developer/Ascend/cann-9.0.0-beta.2/aarch64-linux/asc/impl/basic_api/kernel_operator_common_intf_impl.h:110:29: note: candidate function template not viable: no known conversion from 'LocalTensor<float>' to 'GlobalTensor<float>' for 1st argument
    __out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)
```

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke），arch35 regbase 路线，`bash build.sh --ops=<op> --pkg --soc=ascend950` 编译。
- **API 路径**：`AscendC::InitOutput<U>(GlobalTensor<T>, uint32_t, T)`（GM 工作区清零，MTE3 通道）vs `AscendC::Duplicate(LocalTensor<T>, T, MaskReg/uint32_t)`（UB Vector 通道清零）。
- **验证过的平台与 CANN 版本**：Ascend950（build `--soc=ascend950` 实测）；CANN `cann-9.0.0-beta.2/aarch64-linux`。
- **适用粒度**：`platform_wide`——任何 AscendC kernel 里 UB LocalTensor 清零场景。
- **路线优先级**：`recommended_fastpath`——UB LocalTensor 清零直接用 `Duplicate`。
- **不适用/未验证场景**：GM（GlobalTensor）工作区清零——`InitOutput` 正是为此设计，本卡结论不适用；a3/a2 平台未验证（推断同 CANN 头一致）。

## 根因

`InitOutput` 的签名被 CANN 头锁定为：

```cpp
__out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)
```

即第一参数**必须**是 `GlobalTensor<T>`（GM 侧，走 MTE3 通道）。`LocalTensor<T>`（UB 侧）与 `GlobalTensor<T>` 是不相关类型，没有隐式转换，于是把 `gradInvRmsBuf_.Get<U>()`（`LocalTensor<U>`）传入时编译器报 `no known conversion from 'LocalTensor<float>' to 'GlobalTensor<float>'`。

UB 侧 LocalTensor 清零的正确 API 是 `Duplicate`（Vector 通道，按 mask 清零整个/部分 VReg），不是 `InitOutput`。

## 判别方法

**触发**（tool_observed 失败）：kernel 里把 `TBuf::Get<U>()` 或 `Que::DeQue<U>()` 得到的 `LocalTensor<U>` 传入 `InitOutput<U>(localTensor, count, value)`，`bash build.sh --pkg --ops=<op> --soc=ascend950` 编译报（verbatim 编译错，取自会话内 build 输出）：

```
no known conversion from 'LocalTensor<...>' to 'GlobalTensor<...>' for 1st argument
__out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)
```

**探针**（可执行检查）：

1. 读编译器 note 指向的 `cann-9.0.0-beta.2/aarch64-linux/asc/impl/basic_api/kernel_operator_common_intf_impl.h` 第 110 行，确认 `InitOutput` 唯一候选签名的第一参数类型为 `GlobalTensor<T>`（GM 侧，MTE3 通道）。
2. 查同头或 `kernel_operator_vec_intf_impl.h` 中 `Duplicate` 的签名，确认其接受 `LocalTensor<T>`（UB 侧，Vector 通道）。
3. 在报错 kernel 行把 `InitOutput<U>(local, count, value)` 改为 `Duplicate(local, value, count)`，重跑 `bash build.sh --pkg --ops=<op> --soc=ascend950`，观察 `no known conversion` 错是否消失。

**观测**（tool_observed）：

- 编译错 `no known conversion from 'LocalTensor<float>' to 'GlobalTensor<float>' for 1st argument`，候选签名 `__out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)`（来自 `kernel_operator_common_intf_impl.h:110` 的编译器 note）。
- `InitOutput` 候选签名第一参数为 `GlobalTensor<T>`，与 `LocalTensor<T>` 不相关类型、无隐式转换。

**判读**（根因解读）：`InitOutput` 是 GM 工作区清零专用 API（MTE3 通道），签名锁定第一参数为 `GlobalTensor<T>`；`LocalTensor<T>`（UB 侧）与 `GlobalTensor<T>` 是不相关类型、无隐式转换。把 `TBuf::Get<U>()`/`Que::DeQue<U>()` 产出的 `LocalTensor` 传入 `InitOutput` 必触发 `no known conversion` 编译错。关键判据二联：编译器 note 显示唯一候选签名第一参为 `GlobalTensor<T>`；调用点第一参来源为 `Get<U>()`/`DeQue<U>()`（即 UB LocalTensor）。UB 侧清零应改用 `Duplicate(localTensor, value, count/mask)`（Vector 通道）。

## 验证信号

来源：本次 `MhcPreSinkhornBackward` arch35 确定性 kernel 开发会话（Ascend950，CANN 9.0.0-beta.2，kernel 引入 `InitOutput` 后的构建）。

- `bash build.sh --pkg --ops=mhc_pre_sinkhorn_backward --soc=ascend950` 报 `no known conversion from 'LocalTensor<float>' to 'GlobalTensor<float>' for 1st argument`，候选签名 `__out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)`（来源平台：950；证据等级：tool_observed）
- `InitOutput` 候选签名本身（编译器 note，`cann-9.0.0-beta.2/aarch64-linux/asc/impl/basic_api/kernel_operator_common_intf_impl.h:110`）（来源平台：950；证据等级：artifact_verified）

## 预防

- 清零 UB `LocalTensor`：`Duplicate(local, value, count)`（Vector 通道）。
- 清零 GM `GlobalTensor`：`InitOutput(global, size, value)`（MTE3 通道）。
- 看到 `InitOutput` 第一参数是 `xxxUb`/`local`/`DeQue`/`Get<U>()` 产物 → 立刻判错，改 `Duplicate`。

## 坏实践

- 为"统一清零入口"把 UB LocalTensor 强转/reinterpret 成 `GlobalTensor` 再喂给 `InitOutput`——会绕过类型系统、把 UB 地址当 GM 地址走 MTE3，行为未定义。

## 失效边界

- **未验证，推断可迁移**：a3 / a2 平台同 CANN asc/impl 头，`InitOutput` 与 `Duplicate` 签名预计一致；本 trace 仅 ascend950 build 实测。
- **不适用**：GM 工作区清零（`InitOutput` 的本职场景）。