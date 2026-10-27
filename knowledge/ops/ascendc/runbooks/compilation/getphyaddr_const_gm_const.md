---
type: "Runbook"
title: "GetPhyAddr const __gm__ 返回不可去 const"
description: "AscendC GlobalTensor<T>::GetPhyAddr() 返回 const __gm__ PrimType*，reinterpret_cast 到 __gm__ uint8_t* 等非 const 指针报 casts away qualifiers；给出构建期判别探针与根因解释，修复方向为从 GM_ADDR 形参直接捕获裸 __gm__ 指针。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api", "topic.implementation.memory_management"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# GetPhyAddr const __gm__ 返回不可去 const

## 现象

在 Ascend910B（a2）上开发 aclnn 注册算子 `QuantBatchMatMulV3` 的 kernel（`op_kernel/quant_batch_mat_mul_v3.h`）。为探测 aclnn workspace 的可用偏移，需要在 kernel 内拿到 workspace 的非 const `__gm__ uint8_t*` 裸指针，对偏移后再 `SetGlobalBuffer` 构造第二个 `GlobalTensor` 视图。最初用 `wsGm_.GetPhyAddr()` 取基址并 `reinterpret_cast` 到 `__gm__ uint8_t*`：

```cpp
__gm__ uint8_t* wsBase = reinterpret_cast<__gm__ uint8_t*>(wsGm_.GetPhyAddr());
GlobalTensor<int32_t> testGm;
testGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(wsBase + sysWsSize_));
```

`cmake --build` 报：

```
/mnt/workspace/.../op_kernel/quant_batch_mat_mul_v3.h:332:34: error: reinterpret_cast from
'const __gm__ AscendC::GlobalTensor<int>::PrimType *' (aka 'const __gm__ int *') to
'__gm__ uint8_t *' (aka '__gm__ unsigned char *') casts away qualifiers
```

环境：CANN 9.0.0，bisheng kernel 编译器，SOC=Ascend910B / ascend910_93。

## 适用场景

- **工程形态**：aclnn/直调 kernel 工程，kernel 内需要拿 workspace 或某 `GlobalTensor` 的非 const `__gm__` 裸指针（典型：偏移后再 `SetGlobalBuffer` 构造第二个 `GlobalTensor` 视图、或做按字节偏移的 DMA）。
- **API 路径**：`GlobalTensor<T>::GetPhyAddr()` → `reinterpret_cast<__gm__ U*>(...)`；`SetGlobalBuffer`。
- **验证过的平台与 CANN 版本**：Ascend910B（a2），CANN 9.0.0，bisheng 编译器。仅在此平台实测过。
- **适用粒度**：`platform_wide`——所有平台所有用到 `GlobalTensor::GetPhyAddr()` 且需非 const 裸指针的 kernel。
- **路线优先级**：`recommended_fastpath`——在 `SetGlobalBuffer` 前直接从 `GM_ADDR` 形参 `reinterpret_cast` 捕获裸 `__gm__` 指针并保存为成员。`banned`——对 `GetPhyAddr()` 的 const 返回做去 const 的 `reinterpret_cast`。
- **明确不适用 / 未验证场景**：host 侧代码（非 `__gm__` 指针）不受此 `__gm__` const 约束；a3 / 950 未独立上板验证（机制同源、推断可迁移，但不入 `platforms`）。

## 判别方法

**触发条件（trigger）**：kernel 内对 `GlobalTensor<T>` 调 `GetPhyAddr()` 取基址，再 `reinterpret_cast` 到 `__gm__ uint8_t*` / `__gm__ T*` 等非 const 指针后做字节偏移或二次 `SetGlobalBuffer`，`cmake --build` 在该行报 `error: reinterpret_cast ... casts away qualifiers`。

**判别探针（probe，逐项可复现）**：

1. **构建取铁证**：`cmake --build <build-dir> --target all binary package`，stderr grep `casts away qualifiers`。命中下列原文即定位到本判别：
   ```
   error: reinterpret_cast from 'const __gm__ AscendC::GlobalTensor<int>::PrimType *'
   (aka 'const __gm__ int *') to '__gm__ uint8_t *' (aka '__gm__ unsigned char *') casts away qualifiers
   ```
2. **返回类型签名核验**：在报错行确认左值来源是 `GetPhyAddr()`（而非裸 `GM_ADDR` 形参）。`GetPhyAddr()` 的返回类型是 `const __gm__ PrimType*`（`PrimType` = `GlobalTensor<int>::PrimType` = `int`），即 `const __gm__ int *`；目标类型 `__gm__ uint8_t *` 无 const，故 `reinterpret_cast` 去 const。
3. **来源替换验证**：把基址来源由 `wsGm_.GetPhyAddr()` 改为 `GM_ADDR workspace` 形参直接 `reinterpret_cast`（`GM_ADDR` 是非 const 裸地址，到 `__gm__ uint8_t*` 不去 const），重新构建；`casts away qualifiers` 不再出现、构建进入链接/运行阶段即定因。

**观测（observation）**：报错原文（探针 1，`tool_result` 逐字）+ 返回类型签名（探针 2，`const __gm__ PrimType*`）。这两条均为构建期工具可观测，非 agent 口述。

**解释（interpretation）**：`AscendC::GlobalTensor<T>::GetPhyAddr()` 的返回类型是 `const __gm__ PrimType*`。C++ 规则下 `reinterpret_cast` 不能 casts away qualifiers（去 const），bisheng 编译器据此拒绝把 `const __gm__ int *` 转成 `__gm__ uint8_t *`。`__gm__` 地址空间限定符叠加在 const 之上，使常见的「先取址再按字节偏移」写法直接编译失败。根因在 `GetPhyAddr()` 的 const 返回是 API 契约的一部分（提示不应通过该接口拿到可写指针），而非编译器 bug。

## 预防

- 任何需要在 kernel 内对 `GlobalTensor` 底层地址做按字节偏移 / 二次视图的场景，一律在 `SetGlobalBuffer` 时从 `GM_ADDR` 形参同步保存一个裸 `__gm__ uint8_t*` / `__gm__ T*` 成员，不要事后回 `GetPhyAddr()` 取址。
- 不要用 `const_cast` 强行去 const 再 `reinterpret_cast`——虽能消编译错误，但破坏 `GetPhyAddr()` 的 const 契约，若底层确为只读映射会运行期崩溃。

## 验证信号

问题与验证来源：在 Ascend910B / CANN 9.0.0 上对 `op_kernel/quant_batch_mat_mul_v3.h` 的 workspace 探测代码做构建期 before/after 对照。

- before：`wsGm_.GetPhyAddr()` 经 `reinterpret_cast<__gm__ uint8_t*>` 触发 `error: reinterpret_cast from 'const __gm__ int *' to '__gm__ uint8_t *' casts away qualifiers`（来源平台：a2；证据等级：tool_observed）。
- after：基址来源改从 `GM_ADDR workspace` 形参直接 `reinterpret_cast` 捕获裸 `__gm__ uint8_t*` 成员后，`cmake --build` 的 error grep 不再出现 `casts away qualifiers`，构建进入链接/运行阶段（来源平台：a2；证据等级：tool_observed）。

## 失效边界

- **未验证，推断可迁移**：a3 / 950 上的 `const __gm__` 限定符语义跨平台一致，推断可迁移，但未独立上板验证，标「未验证，推断可迁移」，不入 `platforms`。
- **失效前提**：若后续 CANN 版本将 `GetPhyAddr()` 返回类型调整为非 const，本卡即失效——使用前先按报错信息确认返回签名。
- **不触发**：host 侧代码（非 `__gm__` 指针）；kernel 内不需非 const 裸指针、仅用 `GlobalTensor<T>` 类型化访问的场景。

## 坏实践

- 对 `GetPhyAddr()` 的 const 返回用 `const_cast` 强行去 const 再 `reinterpret_cast`——绕过编译器但破坏 API 契约，且若底层确为只读映射会运行期崩溃。

## 误诊

无（本卡根因单一，编译器报错原文直接指向 const-cast-away，且 after 替换实验闭环）。