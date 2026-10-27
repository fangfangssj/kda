---
type: "Runbook"
title: "aicore 禁止 const char* cast 到 __gm__ T*"
description: "arch35 CCE 设备编译器禁止在 aicore 函数内把非 __gm__ 限定指针（如 const char*）reinterpret_cast 成 __gm__ T*；要访问 TilingData struct 之外的 GM 尾区，须用已 __gm__ 限定的 GM_ADDR 入参做指针算术。判别=看编译错误文本 + 检查 cast 源是否非 __gm__ 限定。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.language_extension"]
status: stable
sources: []
created_at: "2026-08-18T05:56:48Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# aicore 禁止 const char* cast 到 __gm__ T*

## 现象

在 ops-nn/activation/fused_bias_leaky_relu 算子 arch35 kernel 调试中，为绕开标量传输精度问题，试图从 TilingData struct 之后的 GM 尾区读取预填标量向量。写法是用 `reinterpret_cast<__gm__ T*>(...)` 把一个 `const char*` 强转成 `__gm__ T*` 并 `DataCopy` 搬运到 UB。CCE 设备编译器在同一行连续报错 5 次并终止编译：

```
op_kernel/arch35/fused_bias_leaky_relu.h:118:24: error: casting 'const char *' to type '__gm__ T *' is not allowed in aicore function
[ERROR] TBE ... Opc tool compile failed.
```

`opc.py` 设备编译失败，退出码 2，`1 error generated`。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓 arch35 kernel，由 CCE 设备编译器 `opc` 编译；kernel 入口签名 `__global__ __aicore__ void op(GM_ADDR x, ..., GM_ADDR tiling)`。
- **API 路径**：`GM_ADDR tiling` → kernel 内部想按 `__gm__ T*` 访问 TilingData struct 之外的 GM 尾区数据。
- **验证过的平台与 CANN 版本**：Ascend950PR_9579（arch35/dav_c310），CANN 9.0.0，`opc` 设备编译器。
- **适用粒度**：`platform_wide`——arch35 上所有 aicore kernel 通用约束。
- **路线优先级**：`recommended_fastpath`——用已 `__gm__` 限定的 GM_ADDR 入参做指针算术，不 reinterpret_cast 非 __gm__ 指针。
- **不适用或未验证场景**：host tiling 代码（x86 GCC，无 `__gm__` 限定概念）不受此约束；a2/a3 未验证（推断可迁移，同为 CCE，但不得据推断写入 `platforms`）。

## 根因

CCE 设备编译器对 aicore 函数（带 `__aicore__`/`__global__ __aicore__` 标记）有地址空间限定约束：`__gm__` 限定指针只能来自同样 `__gm__` 限定的来源（GM_ADDR 入参、`GlobalTensor<T>`、`GetPhyAddr()` 返回的 `__gm__` 指针等），不能用 `reinterpret_cast`/C 风格转换把一个**非 `__gm__` 限定**的指针（如 `const char*`、栈指针）"升级"成 `__gm__ T*`。这是为了防止把 UB/栈/常量地址误当 GM 地址访问。TilingData 的 `tilingData` 栈局部变量本身由 `GET_TILING_DATA_WITH_STRUCT` 从 GM `tiling` 拷贝而来，已脱离 `__gm__` 空间；若在 kernel 里另起一个 `const char*` 再 cast 成 `__gm__ T*`，即违反该约束。

## 判别方法

**触发条件**（tool_observed 失败症状）：arch35 kernel `bash build.sh --ops=<op>` 编译失败，`opc.py` 退出码 2，错误文本含 `casting '<type> *' to type '__gm__ T *' is not allowed in aicore function`，且错误定位行存在对**非 `__gm__` 限定指针**的 `reinterpret_cast<__gm__ ...*>` 或 C 风格强转。

**可执行探针**：

1. 读错误行：确认 cast 的**目标**是 `__gm__ T*`、**源**是非 `__gm__` 限定类型（`const char*`/`char*`/栈指针/普通 `T*`）。
2. 对照实验——把 kernel 入口已 `__gm__` 限定的 `GM_ADDR tiling` 一路传进 `Init`，在 `__gm__` 指针上做偏移算术访问 struct 尾区，不再 reinterpret_cast 升级地址空间：

   修复前（`op_kernel/fused_bias_leaky_relu_apt.cpp`，未传 tiling GM_ADDR，Init 内靠 reinterpret_cast 自造 `__gm__ T*`，编译失败）：

   ```cpp
   NsFusedBiasLeakyRelu::FusedBiasLeakyRelu<D_T_X, BUFFER_MODE> op;
   op.Init(x, bias, y, &tilingData);
   op.Process();
   ```

   ```cpp
   // op_kernel/arch35/fused_bias_leaky_relu.h  (修复前，第 118 行，编译失败)
   __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                               const FusedBiasLeakyReluTilingData* tilingData);
   // Init 内部用 reinterpret_cast<__gm__ T*>(<非 __gm__ 指针>) 访问尾区 → error
   ```

   修复后（把 `tiling` GM_ADDR 传进 Init，在其上做 `__gm__` 偏移算术）：

   ```cpp
   NsFusedBiasLeakyRelu::FusedBiasLeakyRelu<D_T_X, BUFFER_MODE> op;
   op.Init(x, bias, y, tiling, &tilingData);
   op.Process();
   ```

   ```cpp
   // op_kernel/arch35/fused_bias_leaky_relu.h  (修复后，编译通过)
   __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR tilingGm,
                               const FusedBiasLeakyReluTilingData* tilingData);
   // Init 内部：用 (__gm__ uint8_t*)tilingGm + sizeof(struct) 计算 struct 之后的尾区地址，
   //            该结果本身已是 __gm__ 限定，无需 reinterpret_cast 升级地址空间。
   ```

3. 重编 `bash build.sh --ops=fused_bias_leaky_relu`，预期 `[SUCCESS] Build package success!`、退出码 0、不再报 aicore cast 错误。

**原始观测**（Ascend950PR_9579，arch35，CANN 9.0.0，`opc` 设备编译器）：

- 修复前编译失败：`fused_bias_leaky_relu.h:118:24: error: casting 'const char *' to type '__gm__ T *' is not allowed in aicore function`，`Opc tool compile failed`，退出码 2。
- 修复后编译通过：`[SUCCESS] Build package success!`，退出码 0。

**机制解释**（根因分析，属 interpretation）：CCE 对 aicore 函数强制 `__gm__` 地址空间来源一致性——`__gm__ T*` 只能由 `__gm__` 限定来源派生，`reinterpret_cast` 不能把非 `__gm__` 指针"升级"进 GM 地址空间。错误行 cast 源是 `const char*`（非 `__gm__`），故被拒。修复不是改 cast 写法，而是换数据来源：用 kernel 入口已 `__gm__` 限定的 `GM_ADDR tiling` 做偏移，结果天然 `__gm__` 限定，无须升级。

## 预防

在 aicore kernel 中访问 GM 时，始终从 `__gm__` 限定的来源（`GM_ADDR` 入参、`GlobalTensor<T>::GetPhyAddr()`）派生指针，不要用 `reinterpret_cast` 把非 `__gm__` 指针升级为 `__gm__ T*`。若只需读 TilingData struct 内字段，`GET_TILING_DATA_WITH_STRUCT` 已把 struct 从 GM `memcpy` 到栈局部变量，直接用 `tilingData->field` 即可，无需手算 GM 偏移——本判别只针对"struct 之外的 GM 尾区数据"访问场景。

## 验证信号

- 修复前编译失败：`fused_bias_leaky_relu.h:118:24: error: casting 'const char *' to type '__gm__ T *' is not allowed in aicore function`，`Opc tool compile failed`，`opc.py` 退出码 2（来源平台：950；证据等级: tool_observed）。
- 修复后编译通过：`[SUCCESS] Build package success!`，退出码 0，aicore cast 错误消失（来源平台：950；证据等级: tool_observed）。

## 失效边界

- Ascend950/arch35 实测；a2/a3 未验证，推断可迁移（同为 CCE 设备编译器、同一 aicore 地址空间约束），标注"未验证，推断可迁移"——不得据此把 a2/a3 写入 `platforms`。
- host tiling 代码（x86 GCC 编译）无 `__gm__` 概念，不受此约束，host 侧相同写法能过——误诊项已记录见 `## 误诊`。
- 仅当访问 TilingData struct **之外**的 GM 尾区数据时触发；只读 struct 内字段用 `tilingData->field` 不涉及此约束。

## 坏实践

用 `const char*`/`char*` 作中转再 `reinterpret_cast` 成 `__gm__ T*`——即使逻辑上地址数值正确，CCE 也会因地址空间限定不匹配而拒绝编译。

## 误诊

最初以为是 `reinterpret_cast` 的常规 C++ 用法问题，但 host 侧 tiling 代码用相同写法能过——根因是 aicore 函数的地址空间限定约束，不是通用 C++ cast 规则。判别时若只看 "reinterpret_cast 失败" 而忽略 "源类型非 __gm__"，会误判为普通 C++ cast 错误。