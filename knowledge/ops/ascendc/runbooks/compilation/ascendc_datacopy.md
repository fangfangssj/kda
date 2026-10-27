---
type: "Runbook"
title: "AscendC DataCopy 标量参数模板推导失败"
description: "AscendC::DataCopy 只接受 LocalTensor 间块搬运；把标量 T 或 tensor[i] 元素代理当参数传入即触发 candidate template ignored 模板推导失败，UB 内逐元素读写须改用 LocalTensor::GetValue/SetValue。"
tags: ["task.debugging", "topic.compilation", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-18T13:32:19Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# AscendC DataCopy 标量参数模板推导失败

## 现象

在 Ascend950（dav-3510）上用 ops-direct-invoke-flash 从零实现 cummin 算子。首版 `csrc/ops/cummin/op_kernel/cummin_kernel.cpp` 的扫描循环中，为做顺序前缀扫描，用 `AscendC::DataCopy` 在 UB 内逐元素搬运——把标量 `T current` 与 `inL[i]` 元素代理当 `DataCopy` 的参数。`make -j` 编译直接在模板实参推导阶段失败。

## 适用场景

- **工程形态**：ops-direct-invoke-flash 核函数构建（kernel 直调，无 aclnn/op_host tiling 接管）。
- **API 路径**：`AscendC::DataCopy` / `LocalTensor::GetValue` / `LocalTensor::SetValue`，在 `__aicore__` 设备函数内对 UB 单元素读写。
- **验证过的平台与 CANN 版本**：Ascend950 / dav-3510；CANN 版本未在 trace 中定位。
- **适用粒度**：`operator_family`（任何需要 UB 内顺序/条件逐元素访问的 AscendC kernel，如 cummin/cummax/带索引跟踪的 scan）。
- **路线优先级**：`recommended_fastpath`——UB 内逐元素访问的规范做法就是 GetValue/SetValue；DataCopy 仅用于 GM↔UB 批量搬运。
- **不适用或未验证场景**：a2/a3 未验证；GM↔UB 批量搬运仍应使用 DataCopy/DataCopyPad，本卡只针对 UB 内标量读写。

## 判别方法

**触发条件**：`__aicore__` 设备函数内出现形如 `AscendC::DataCopy(scalar, tensor[i], 1)` 或 `AscendC::DataCopy(tensor[i], scalar, 1)` 的调用——即 `DataCopy` 的 `dst`/`src` 参数之一为标量 `T`、`LocalTensor::operator[]` 返回的元素代理，而非 `LocalTensor`。编译时即触发，不依赖运行时。

**可执行探针**：对目标 kernel 工程执行 `make -j 2>&1 | tail -80`，在编译输出中检索 `candidate template ignored: could not match 'LocalTensor<T>' against 'int'`（或 `against` 后跟标量/代理类型名）。命中即确证 `DataCopy` 被误用于标量参数。

**原始观测**（Ascend950 / dav-3510，cummin kernel 首次编译）：
- `make -j` 报 `candidate template ignored: could not match 'LocalTensor<T>' against 'int'`——`DataCopy` 模板实参推导失败，第一参数标量 `current`（推导为 `int`）与 `inL[i]` 元素代理都不是 `LocalTensor`。
- 把扫描循环改为 `LocalTensor::GetValue(int32_t index)` 读、`LocalTensor::SetValue(int32_t index, T value)` 写后重编，该模板推导失败消失，编译推进到下一组错误（`__float_to_half` 等，属另一独立根因），证明本根因已消除。

**机制解释**：`AscendC::DataCopy(dst, src, count)` 的签名要求 `dst`/`src` 均为 `LocalTensor`，用于 GM↔UB 或 UB↔UB 的整块搬运；它不是 UB 内标量元素读写 API。标量 `T` 与 `LocalTensor::operator[]` 返回的元素代理类型都不是 `LocalTensor`，模板实参推导直接失败。UB 内逐元素读写应走 `LocalTensor::GetValue(int32_t index)` / `LocalTensor::SetValue(int32_t index, T value)`；批量 GM↔UB 搬运仍用 `DataCopy`/`DataCopyPad`。

## 验证信号

本卡验证来源：cummin kernel 首次编译失败的报错文本与改用 GetValue/SetValue 后的编译推进（同一 trace 的 make 工具调用与编译诊断）。

- `make -j` 编译报 `candidate template ignored: could not match 'LocalTensor<T>' against 'int'`，模板实参推导失败（来源平台：950；证据等级：tool_observed）
- 改用 GetValue/SetValue 后该模板推导失败消失，编译进入下一错误阶段（来源平台：950；证据等级：tool_observed）

## 预防

在 `__aicore__` 内做顺序扫描/条件逐元素更新时，默认用 `LocalTensor::GetValue/SetValue`；`DataCopy`/`DataCopyPad` 只用于 GM↔UB 或 UB↔UB 的整块搬运，不要拿标量或 `tensor[i]` 元素代理当其参数。评审 kernel 实现时，对 `DataCopy` 参数非 `LocalTensor` 的写法直接打回。

## 失效边界

- dav-3510（950）实测；a2/a3 未验证，推断可迁移（GetValue/SetValue 是 AscendC 通用 LocalTensor 接口）。
- 若后续 CANN 版本为 `LocalTensor` 增加 `operator[]` 直接返回可读写标量引用，则本卡可作为历史背景。
