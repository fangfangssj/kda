---
type: "Runbook"
title: "qgmm_mx FP4 fatbin 缺失致 107000"
description: "bisheng 9.1.0 编译的 qgmm_mx fatbin 只含 FP8 kernel 特化，缺 FP4 (FLOAT4_E2M1，模板参数 hh=uint8_t) 特化；Ascend950 运行 mxfp4_e2m1 nz multi 用例时 runtime BinaryGetFunctionByName 按符号名查不到 FP4 kernel，返回 107000、输出全零。FP4 路线 banned，回退 FP8 可用。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.execution_unit.cube"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# qgmm_mx FP4 fatbin 缺失致 107000

## 现象

在 Ascend950（CANN 9.1.0，bisheng-compiler 9.1.0）上运行 `ops-tensor` 的 `examples/gmm/qgmm_mx` 示例，跑 `mxfp4_e2m1 nz multi`（即 `GroupedMatmulMixWeightPrologue` 的 FP4 NZ 多组场景）：

```bash
cd examples/gmm/qgmm_mx
./build/gmm/qgmm_mx/qgmm_mx mxfp4_e2m1 nz multi 2 32 64 128 false false length 2 <out_dir> <out_dir>
```

返回：

```text
[ERROR] [Bisheng] Get kernel function failure! ret 107000
[ERROR] RUNTIME(...,qgmm_mx) BinaryGetFunctionByName:Cannot find kernel by name _Z14qgmm_mx_kernelIhhN7AscendC2Te14NDExtLayoutPtnENS1_11NZLayoutPtnEEvPhS4_S4_S4_S4_S4_S4_jlllhhhh.
```

输出全零（`mismatches=4035/4096`，`actual=0.0`），kernel 未在设备上执行。

demangle 该 symbol 得 `qgmm_mx_kernel<uint8_t, uint8_t, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn>`——模板参数 `hh`（`unsigned char, unsigned char`）即 `FLOAT4_E2M1` 在模板里以 `uint8_t` 表示。bisheng 编译产出的 fatbin 只注册了 FP8 变体（`u6__fp8e4m3`、`u6__fp8e5m2`），没有 FP4（`hh`）变体。

## 适用场景

- **工程形态**：Kernel 直调示例（`qgmm_mx`，非 aclnn/GEIR），直接调用 `GroupedMatmulMixWeightPrologue` Blaze kernel。
- **API 路径**：`bisheng/cce` 编译 → fatbin 注册 kernel symbol → runtime `BinaryGetFunctionByName` 按符号名加载。
- **验证平台与版本**：Ascend950，CANN 9.1.0，bisheng-compiler 9.1.0；`platforms: [950]`。
- **适用粒度**：`operator_family`——任何以 `uint8_t` 表示 `FLOAT4_E2M1` 模板参数并经 bisheng 编译的 Blaze GMM kernel 都可能踩此坑。
- **路线优先级**：`banned`——FP4 (E2M1) 经 bisheng 9.1.0 编译的 qgmm_mx 路线在本平台不可用，不可作为通过路径。
- **不适用/未验证场景**：FP8 (e4m3/e5m2) 变体在 fatbin 中存在、可正常加载，不在本卡范围；a2/a3 与其他 CANN 版本未验证。

## 失败路线

- **尝试**：用 `ops-tensor/examples/gmm/qgmm_mx` 验证 `GroupedMatmulMixWeightPrologue` 的 `mxfp4_e2m1` NZ 多组用例，期望与 CPU golden 对齐。
- **失败/回退**：bisheng 编译产出的 fatbin 只含 FP8 模板特化，FP4（`hh`）特化未被编译进二进制；runtime 按符号名 `_Z14qgmm_mx_kernelIhh...` 查不到，返回 `107000`；输出全零。回退路径：改用 FP8 变体（`mxfp8_e4m3`/`mxfp8_e5m2`）可正常加载执行；FP4 需等 bisheng 编译器支持或改走预编译 binary 之外的路径，本 trace 未找到可用 FP4 路径，故 FP4 路线 banned。
- **边界**：`FLOAT4_E2M1` 在模板层用 `uint8_t` 承载（每字节打包两个 FP4），bisheng 9.1.0 似乎跳过了该特化的实例化/注册。源码 `qgmm_mx.cpp` 同时声明 FP4 与 FP8 路径，非源码缺失。确切编译器机制未追到。

## 根因

`qgmm_mx_kernel` 的 `FLOAT4_E2M1` 模板特化（模板参数 `uint8_t`/符号 `hh`）未被 bisheng 编译器编译进 fatbin，fatbin 只注册了 FP8（`u6__fp8e4m3`/`u6__fp8e5m2`）特化。runtime 按符号名 `BinaryGetFunctionByName` 查找 FP4 kernel 失败，返回 `107000 invalid value`，kernel 不执行、输出全零。这是 bisheng 9.1.0 对 FP4 模板特化的编译支持限制，不是迁移代码本身的逻辑错误。该根因的编译器层机制为推断，但 107000 + fatbin 缺 FP4 symbol 为 tool_observed。

## 验证信号

- `[ERROR] [Bisheng] Get kernel function failure! ret 107000` + `Cannot find kernel by name _Z14qgmm_mx_kernelIhhN7AscendC2Te14NDExtLayoutPtnENS1_11NZLayoutPtnEEv...`（来源平台：950；证据等级：tool_observed）
- 输出 `actual=0.0, mismatches=4035/4096`（kernel 未执行致全零，非精度逻辑问题）（来源平台：950；证据等级：tool_observed）
- fatbin 注册 kernel 均为 FP8 变体（`u6__fp8e4m3`/`u6__fp8e5m2`），FP4（`hh`）变体缺失（来源平台：950；证据等级：tool_observed）

## 失效边界

- FP8 变体（`mxfp8_e4m3`/`mxfp8_e5m2`）在 fatbin 中存在、可加载，不在本卡 banned 范围。
- 其他 CANN 版本或更高 bisheng 版本若已支持 FP4 特化，本结论可能不再成立——未验证，仅推断可迁移到同 bisheng 9.1.0 + Ascend950 组合。
- a2/a3 未接触，`platforms` 不含。

## 坏实践

- 把 `107000` + 输出全零误判为精度逻辑错误去调 kernel 算式——实际 kernel 根本没执行，应先从 runtime 日志确认 `BinaryGetFunctionByName` 是否找到 symbol。
- 为“通过”而改用 FP8 跑 FP4 用例并宣称 FP4 已验证——属于测试侧放松，FP4 路线仍 banned。
