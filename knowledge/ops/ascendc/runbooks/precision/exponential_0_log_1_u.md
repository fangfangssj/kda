---
type: "Runbook"
title: "exponential_ 0% 匹配须改 -log(1-u) 路径"
description: "自定义 AscendC kernel 拦截 aten.exponential_ 同 seed 0% bit-exact 匹配时，判别 kernel exponential 分支是否存 u 直接 Log(u)（坏模式）：native aclnnInplaceUniform 生成 u∈[0,1) 后存 1-u∈(0,1] 再 Log(1-u)+Muls(negInvLambd)；log(u) 路径浮点比特不同且 u=0 时 log(0)→inf，须改 -log(1-u) 路径。"
tags: ["task.debugging", "topic.precision", "topic.compute.random_sampling", "topic.compute.elementwise.math_function"]
status: draft
sources: []
created_at: "2026-08-18T05:35:20Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# exponential_ 0% 匹配须改 -log(1-u) 路径

## 现象

用自定义 AscendC `StatelessRandomUniformV3` kernel + `.so` TORCH_LIBRARY dispatch 拦截 `aten.exponential_` 后，同 `seed=42` 下与原生 NPU `exponential_` 0% bit-exact 匹配（4 shape 全 0% exact match，`max_abs_diff` 显著非零）。除 uniform 生成路径本身（IEEE754 `Uint32ToFloat` vs 位移乘法）外，exponential 变换路径也偏离 native：自定义 kernel 在 `isExp` 分支直接存 uniform `u` 再 `Log(u)+Muls(negInvLambd)`，而 native `aclnnInplaceUniform→StatelessRandomUniformV2` 生成 `u∈[0,1)` 后存 `1-u∈(0,1]` 再做 `-log(1-u)/lambd`。两条路径数学等价但浮点比特不同，且 `log(u)` 在 `u=0` 时退化为 `log(0)→-inf`。

## 适用场景

- **工程形态**：ops-direct-invoke Kernel 直调 + `.so` dispatch 拦截 `aten.exponential_`，目标同 seed bit-exact 复现 native。
- **API 路径**：AscendC `Log<float>` + `Muls<float>(..., negInvLambd)`；对标 native `aclnnInplaceUniform` 生成 uniform 后的 host/kernel log 变换。
- **验证平台与 CANN 版本**：arch35 / CANN 9.1.0 / AIV vector core；trace 未明示芯片型号，无法映射到注册平台 token `950/a3/a2`，`platforms: []`（未定位），推断同 arch 可迁移但未验证。
- **适用粒度**：`operator_family`（任何在 kernel 内做 uniform→exponential 变换的随机采样算子）。
- **路线优先级**：`recommended_fastpath`（bit-exact 复现 native `exponential_` 的唯一安全路径）。
- **不适用或未验证场景**：不需要 bit-exact 对齐、只过分布检验的 exponential 实现；Python 侧直接调自定义算子再 Python `log` 变换的路线为 `banned`（见 `## 坏模式` B）。

## 判别方法

**触发**：自定义 AscendC kernel 拦截 `aten.exponential_` 后，同 `seed` 与原生 0% bit-exact 匹配（`max_abs_diff` 显著非零），自定义首元素与 native 量级即不同。次级触发：Python 侧手搓 uniform+`log` 变换触发 `507035 AICORE_KERNEL_ERROR` 设备错误——但后者仅在 trace 终末总结叙述，原始 device log 未在事件流复现，证据等级 `agent_reported`，不单独作为判别闭环，仅作坏模式 B 的旁证。

**可执行探针（按序）**：

1. **同 seed 逐元素比对**：固定 `seed=42`，对多个 shape 跑自定义 kernel 与原生 `aten.exponential_`，比 `exact match %` 与 `max_abs_diff`。0% 匹配即命中本判别。
2. **检查 kernel exponential 分支存的是 `u` 还是 `1-u`**：定位 `isExp`（或等价 exponential 分支）写入 output 的表达式。待识别的坏模式 A（结构示意，非 verbatim 证据）：

```cpp
// 坏模式 A：存 u 再 Log(u) —— 偏离 native 且 u=0 时 log(0)→inf
float u = (...生成的 uniform∈[0,1) ...);
outLocal.SetValue(j, u);          // 直接存 u
...
AscendC::Log<float>(outLocal, outLocal, count);
AscendC::Muls<float>(outLocal, outLocal, negInvLambd, count);  // = -log(u)/lambd
```

   正确模式应存 `1.0f - fval`（`fval∈[0,1)`）使落点 ∈`(0,1]`，再 `Log+Muls(negInvLambd)` 得 `-log(1-u)/lambd`，与 native 计算路径逐比特对齐并规避 `log(0)`。
3. **排除 PRNG 本身**：切换手动标量 Philox4x32-10 与 CANN `PhiloxRandom<10>` SIMD intrinsic，若输出 uint32 流相同（本 trace 实测相同），则 PRNG 实现非根因，根因在 float 转换/exponential 变换/tiling 对齐层面。
4. **隔离变量**：本卡的 1-u 路径修复未单独隔离测量，100% match 是与 IEEE754 `Uint32ToFloat` 位操作、counter 按 4 元素 group 递增、`numPerCore` 对齐到 4 一起验证的闭包信号——判别时须把 1-u 路径作为完整修复集的一项，勿单独归因闭包 100% match。

**原始观测（tool_observed）**：同 `seed=42` 下 4 shape 全 0% exact match，自定义与 native 首元素量级即不同（来源平台:未定位；证据等级:tool_observed）。手动标量 Philox 与 CANN `PhiloxRandom<10>` SIMD intrinsic 产生相同 uint32 流，排除 PRNG 根因（来源平台:未定位；证据等级:tool_observed）。

**解释（root-cause interpretation）**：`exponential_(lambd)` 的标准采样是 `-log(U)/lambd`，`U~(0,1]`。native `aclnnInplaceUniform→StatelessRandomUniformV2` 用 IEEE754 位操作生成 `u∈[0,1)`（`(h & 0x7fffff) | (127<<23)` reinterpret 后 `-1.0f`），再存 `1-u∈(0,1]` 做 `Log(1-u)+Muls(negInvLambd)`。两条"等价"写法浮点结果不同：(a) 直接生成 `u∈(0,1]` 与生成 `u∈[0,1)` 再 `1-u` 得到的比特不同（舍入/边界差异），故 `-log(u)` 与 `-log(1-u)` bit 不同；(b) `1-u` 严格大于 0，消除 `log(0)→-inf` 崩溃面，而 `log(u)` 在 `u=0` 时崩溃。因此 bit-exact 复现 native 必须走 `-log(1-u)` 路径（存 `1-u`），`log(u+eps)` 数学等价但浮点不同且不消除 log(0)。这是根因解释，明确为 interpretation，非单点实测。

## 坏模式

- **坏模式 A（kernel 内 log(u) 路径）**：`isExp` 分支存 `u` 再 `Log(u)+Muls(negInvLambd)` → bit 偏离 native 且 `u=0` 时 `log(0)→inf`。判别方法·探针 2 已示。
- **坏模式 B（Python 侧手搓 exponential，banned）**：在 Python 侧直接调自定义 `stateless_random_uniform_v3` 生成 uniform 再用 Python `log` 变换做 `exponential_`。当 uniform 产出 0 时 `log(0)→inf`，`probs.div_(NaN)` 触发 NPU `507035` 设备错误。正确做法是不改 Python 调用，靠 `.so` TORCH_LIBRARY dispatch 让底层 kernel 走 `1-u + Log` 安全路径。该路线标 `banned`。

## 验证信号

- 同 `seed=42` 下 4 shape 全 0% exact match，自定义与 native 首元素量级即不同（来源平台:未定位；证据等级:tool_observed）
- 手动标量 Philox4x32-10 与 CANN `PhiloxRandom<10>` SIMD intrinsic 产生相同 uint32 流，排除 PRNG 根因（来源平台:未定位；证据等级:tool_observed）

## 失效边界

- arch35 / CANN 9.1.0 / AIV 设备观测；芯片型号未明示，未映射到注册平台，推断可迁移到同 arch AscendC 设备但未验证。
- 若 native `exponential_` 切换为 `aclnnSimThreadExponential`（SIMT 路径）或其他 log 变换实现，本 `-log(1-u)` 路径不再适用。
- 1-u 路径的隔离效果未单独测量（与 `Uint32ToFloat`、counter 映射、`numPerCore` 对齐一起验证），勿单独归因闭包 100% match。