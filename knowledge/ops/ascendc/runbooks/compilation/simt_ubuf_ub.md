---
type: "Runbook"
title: "SIMT __ubuf__ 静态数组 UB 占用不随分支释放"
description: "SIMT kernel 的 __ubuf__ 静态数组是编译期分配、生命周期绑所在函数栈的 UB 资源，运行时分支只控制是否解引用、不控制是否占用；把“需要该数组”与“不需要该数组”两条路径塞进同一函数，会令不解引用的路径也白白占住 UB。给出按解引用关系拆函数的判别方法。"
tags: ["task.debugging", "topic.implementation.memory_management"]
status: draft
sources: []
created_at: "2026-08-18T07:23:22Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
aliases: ["partialSums"]
---

# SIMT __ubuf__ 静态数组 UB 占用不随分支释放

## 现象

ops-blas 仓 aclblasStbsv（arch35/Ascend950）的 SIMT kernel（`blas/stbsv/arch35/stbsv_kernel_simt.cpp`）入口 `StbsvSimt` 在 PR 检视中被指出：函数开头声明 `__ubuf__ float xUb[UB_X_FLOATS]`（`UB_X_FLOATS=4096`，占 16KB UB），但当 `n > UB_X_FLOATS` 走 TILED 分块路径时，`TbsvSimtProcess<..., true>` 的 TILED 分支只读写 `xGm[xOff]`、不解引用 `xUb`，16KB UB 仍被该静态数组占据。`__ubuf__` 静态数组无法在运行时按条件释放——其生命周期与所在函数绑定，编译期即分配。本卡记录如何判别这类“静态数组占住不解引用路径 UB”的问题，不内嵌修复 diff。

## 适用场景

- **工程形态**：kernel 直调（SIMT kernel，`__simt_vf__`/`__simt_callee__` + `__ubuf__` 静态数组编程模型）。
- **API 路径**：`__ubuf__` 静态数组声明 + `asc_syncthreads` + `__gm__ float*`；模板 `TbsvSimtProcess<...,TILED>` 按 TILED 选择走 UB 缓存或 GM 直读。
- **验证过的平台与 CANN 版本**：本 trace 内未编译/上板验证；目标平台 arch35/Ascend950 仅在 PR 描述与代码路径中出现，`platforms` 记为 `[]`（未定位）。
- **适用粒度**：`operator_family`（所有用 `__ubuf__` 静态数组缓存中间量、且存在“部分路径不需要该缓存”分发的 SIMT kernel）。
- **路线优先级**：`recommended_fastpath`（按是否需要静态数组拆函数是 `__ubuf__` 模型下条件释放的唯一可靠手段）。
- **不适用或未验证场景**：AIV Vector 侧用 `TBuf`/`LocalTensor` 可按需 `InitBuffer`，不适用本卡的 `__ubuf__` 静态约束。NPU 编译/资源占用未在本 trace 实测。

## 判别方法

触发：SIMT kernel 的同一函数内声明了 `__ubuf__` 静态数组（如 `__ubuf__ float xUb[UB_X_FLOATS]`），且存在“部分运行时路径不使用该数组”的分支分发；PR 检视或 UB 占用审计指出不解引用该数组的路径仍占住其 UB 配额（本例为 xUb 占 16KB，TILED 路径不解引用）。

探针：对每个 `__ubuf__` 静态数组，逐路径核对“该路径是否解引用它”。可执行检查——在 kernel 源文件上 grep 数组名与分支标记，确认每条运行时路径对数组的实际访问：

```
grep -n "xUb\|if constexpr\|TILED" blas/stbsv/arch35/stbsv_kernel_simt.cpp
```

再读 `TbsvSimtProcess` 模板体，定位 `xUb` 的写/读是否被 `if constexpr (!TILED)` 守卫。

原始观测（取自本 trace 对 kernel 源文件的读取）：

- `StbsvSimt` 函数入口处声明 `__ubuf__ float xUb[UB_X_FLOATS];`（`UB_X_FLOATS=4096` → 16KB UB），与运行时 `n <= UB_X_FLOATS` 的判断在同一函数体内。
- `TbsvSimtProcess` 的 TILED 分支只读写 `xGm[xOff]`，对 `xUb` 的写发生在 `if constexpr (!TILED)` 守卫下（`xUb[row] = ...`）；TILED 路径不解引用 `xUb`。
- 即：xUb 的 UB 占用由“函数栈声明”决定，不由“运行时是否走 TILED”决定——TILED 路径不解引用但仍占 16KB。

解释：`__ubuf__` 静态数组是 SIMT 编程模型下的编译期分配 UB 资源，生命周期与所在函数栈绑定，无法在运行时按 `n` 是否超 4096 来“条件释放”。原 `StbsvSimt` 把 UB 路径（`n ≤ UB_X_FLOATS`，需 xUb）与 TILED 路径（`n > UB_X_FLOATS`，不需 xUb）写在同一函数体内，`xUb` 在函数开头声明 → 两条路径都占用 16KB UB，TILED 路径纯浪费。机制层归因：`__ubuf__` 静态模型与“运行时分支条件资源”根本不匹配——运行时 `if` 只切换解引用，不切换占用。判别要点：凡同一函数内声明 `__ubuf__` 数组且存在“部分路径不解引用”的分支，即命中此问题；修复方向是按“是否需要该数组”拆独立函数（不解引用该数组的路径所在函数不声明它），使每条路径只占自己所需 UB。

## 验证信号

- PR 检视意见标注 `__ubuf__ float xUb[UB_X_FLOATS]`（UB_X_FLOATS=4096，16KB）静态数组在 TILED 路径不解引用仍占 UB（来源平台：未定位；证据等级: tool_observed）。
- 读 `TbsvSimtProcess` 模板体观察到 TILED 分支只读写 `xGm[xOff]`，`xUb[row]` 的写发生在 `if constexpr (!TILED)` 守卫下，TILED 不解引用 xUb（来源平台：未定位；证据等级: tool_observed）。

> 说明：验证信号段来源 = 本 trace 对 PR 检视评论与 kernel 源文件的工具读取；本 trace 未编译/上板，UB 实际占用变化未实测，故本卡只作为“识别该静态数组占住不解引用路径 UB 模式 + 拆函数判别方向”的诊断卡，不声称修复后实测通过。

## 预防

- `__ubuf__` 静态数组的生命周期等于所在函数栈；凡存在“部分运行时路径不需要该数组”的 SIMT kernel，禁止把多路径塞进同一函数，必须按“是否需要该 UB 缓存”拆独立函数。
- 写 SIMT kernel 分发前先列“每条路径实际解引用哪些 `__ubuf__` 数组”，未解引用的数组不得在该路径的函数内声明。
- UB 资源审计按“最大单函数占用”算，而非“所有路径并集”——拆函数让每条路径只占自己所需。

## 坏实践

- 用“运行时 if + 编译期分配的 `__ubuf__` 数组”假装条件释放——`__ubuf__` 静态数组在函数入口即分配，运行时分支只控制是否解引用，不控制是否占用 UB。
- 把“需要 xUb 的 UB 路径”与“不需要 xUb 的 TILED 路径”留在同一函数体内仅靠 `if` 分支区分，会令 TILED 路径白白占住 16KB UB。

## 失效边界

- 判别只针对 `__ubuf__` 静态数组模型；AIV Vector 侧 `TBuf`/`LocalTensor` 可按需 `InitBuffer`/释放，不属本卡约束。
- 拆函数判别成立的前提是“不解引用该数组的路径”确实不解引用——若未来 TILED 分支新增对 xUb 的访问，拆分前提失效，须重新审视。
- 推断可迁移到其他“按 n/shape 分 UB 缓存与 GM 直读两路径”的 SIMT kernel——未在本 trace 实测，推断可迁移。
