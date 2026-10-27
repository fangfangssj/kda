---
type: "Runbook"
title: "REGISTER_NONE_TILING 致 UnboundLocalError"
description: "按 DESIGN 规范在模板化 TilingData 的 kernel 入口写 REGISTER_NONE_TILING 时，CANN 9.1.0 op-compile 的 dfx 生成器在 _generate_binary_for_tiling_without_register 抛 UnboundLocalError；判别后改用 REGISTER_TILING_DEFAULT(具体模板实例化) 规避。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T15:22:33Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# REGISTER_NONE_TILING 致 UnboundLocalError

## 现象

在为 AscendC 算子 `softplus_v2_grad` 开发 TilingData 结构体时，developer 子 agent 按设计规范的原文规定，把 kernel 入口 `op_kernel/softplus_v2_grad_kernel.cpp` 的 tiling 注册从 `REGISTER_TILING_DEFAULT(...)` 改为规范规定的 `REGISTER_NONE_TILING;`。TilingData 是按 `kRank` 模板化的结构体 `template<int64_t kRank> struct SoftplusV2GradTilingData`（同时承载 `<4>` 与 `<8>` 两档，由 TilingKey 在运行期分发）。改完后在算子工程根目录执行：

```
./operators/build.sh softplus_v2_grad_package 2>&1 | tail -50
```

构建在 op-compile 阶段抛出一段 Python Traceback（不是 C++ 编译错误），最终落在 dfx 段生成器：

```
UnboundLocalError: cannot access local variable 'tiling_struct_size' where it is not associated with a value
```

调用栈关键帧：`gen_tiling_struct_size_and_dfx_section_file`（compile_op.py:779）→ `gen_dfx_section_for_one_tiling_key_dynamic`（ascendc_compile_gen_code.py:524）→ `generate_dfx_section_without_tiling_register`（ascendc_compile_dfx.py:450）→ `_generate_binary_for_tiling_without_register`（ascendc_compile_dfx.py:239）。构建产物路径为 `operators/softplus_v2_grad_package/build/SoftplusV2Grad_ascend950/dynamic/SoftplusV2Grad.py`，即构建目标平台为 ascend950。

## 适用场景

- **工程形态**：ACLNN 算子仓开发（ops-registry-invoke 工作流），kernel 入口 tiling 注册。
- **API 路径**：`REGISTER_NONE_TILING` / `REGISTER_TILING_DEFAULT(struct)` / `GET_TILING_DATA_WITH_STRUCT` 宏；op-compile 经 `asc_op_compile_base.asc_op_compiler` 链路生成 dfx 段。
- **验证过的平台与 CANN 版本**：ascend950（build 目标，见 traceback 中 `SoftplusV2Grad_ascend950/dynamic`）；CANN 9.1.0（`/home/Developer/Ascend/cann-9.1.0` 的 asc_op_compile_base）+ opp built-in 引用 cann-9.0.0-beta.2。仅 build 阶段定位，未在上板/ST 运行信号中验证。
- **适用粒度**：`operator_family`（任何使用按 RANK/维度模板化的 TilingData 结构体 + 试图用 REGISTER_NONE_TILING 的 AscendC 算子，在 CANN 9.1.0 dfx 生成器下都会触发）。
- **路线优先级**：`recommended_fastpath`（用 REGISTER_TILING_DEFAULT 注册一个具体模板实例化以走 dfx 已验证路径）。
- **明确不适用或未验证场景**：非模板化（单一 concrete struct）的 TilingData 用 REGISTER_NONE_TILING 是否同样触发，未验证；CANN < 9.1.0 或 > 9.1.0 版本是否已修复该 UnboundLocalError 未验证。

## 根因

op-compile 的 dfx 段生成器在 `REGISTER_NONE_TILING` 路径下（`generate_dfx_section_without_tiling_register` → `_generate_binary_for_tiling_without_register`，ascendc_compile_dfx.py:239）无条件读取局部变量 `tiling_struct_size`，但该路径并未给它赋值，导致 Python 层 `UnboundLocalError`。即 `REGISTER_NONE_TILING` 在当前 CANN 9.1.0 dfx 生成器里是一条未补全的代码路径——它需要"已注册的 tiling struct"才能拿到 `tiling_struct_size`，而 `REGISTER_NONE_TILING` 恰恰不注册任何 struct。

## 判别方法

- **触发条件**：kernel 入口（`__global__ __aicore__ void <op>(...)`）所在 `.cpp` 使用 `REGISTER_NONE_TILING;`，且 TilingData 为按维度/RANK 模板化的结构体（如 `template<int64_t kRank> struct ...TilingData`，由 `TilingKey` 在运行期分发 `<4>`/`<8>` 等实例）。
- **可执行探针**：在算子工程根目录执行 `./operators/build.sh <package> 2>&1 | tail -50`。
- **原始观测**：op-compile 阶段抛 Python Traceback，终点为 `UnboundLocalError: cannot access local variable 'tiling_struct_size' where it is not associated with a value`，栈帧定位 `asc_op_compile_base/asc_op_compiler/ascendc_compile_dfx.py` 的 `_generate_binary_for_tiling_without_register`（line 239）与 `generate_dfx_section_without_tiling_register`（line 450）（来源平台：950；证据等级: tool_observed）。
- **解释**：dfx 生成器（`gen_tiling_struct_size_and_dfx_section_file`）对每个 TilingKey 都要生成 dfx 段，`REGISTER_NONE_TILING` 走 `generate_dfx_section_without_tiling_register` 分支，但该分支的 `_generate_binary_for_tiling_without_register` 引用了一个从未被赋值的 `tiling_struct_size` 局部变量——这是 dfx 生成器自身的未补全分支，与 kernel C++ 代码正确性无关。读取 `ascendc_compile_dfx.py:180-300`、`ascendc_compile_gen_code.py:490-560`、`kernel_info_infer.py:110-305` 后确认：当存在 `ASCENDC_TPL`（模板）时，构建器会把所有 TilingKey 映射到 `REGISTER_TILING_DEFAULT` 中的 `default_tiling_struct`，因此必须注册一个具体的（实例化的）struct。规避方式是把 kernel 入口注册改为 `REGISTER_TILING_DEFAULT(<最大档模板实例化>)`（如 `REGISTER_TILING_DEFAULT(SoftplusV2GradTilingData<8>)`），用 sizeof 最大的那档以覆盖所有 TilingKey 档位。

## 设计决策

- **为何注册较大的 `<8>` 而不是 `<4>`**：op-compile 只允许注册一个 default tiling struct，Host TilingFunc 在运行期按 `TilingKey` 把 `<4>` 或 `<8>` 填进同一段 tiling-data buffer；注册 `<8>`（较大）后，buffer 容量 `sizeof(SoftplusV2GradTilingData<8>)` 同时覆盖 `<4>` 与 `<8>` 两次填充，`GetTilingData<...<8>>()` 在 rank>4 时也能成功读取。若注册 `<4>`，rank>4 的填充会越界。
- **为何不修 dfx 生成器**：`REGISTER_NONE_TILING` 路径的 `UnboundLocalError` 是 CANN 9.1.0 工具链自身缺陷，不应在算子工程侧绕过；改用已验证的 `REGISTER_TILING_DEFAULT` 路径是算子侧最小修复。
- **被放弃的方案**：第一次尝试按设计规范字面写 `REGISTER_NONE_TILING`（规范本意是"无统一 tiling struct"），被 dfx 生成器否决；这是"规范与现实工具链不一致"的典型冲突，以工具链实际行为为准。

## 预防

- 当 TilingData 必须按维度/RANK 模板化、且由 TilingKey 分发多档实例时，kernel 入口一律用 `REGISTER_TILING_DEFAULT(<最大档模板实例化>)`，不要按字面规范写 `REGISTER_NONE_TILING`。
- 在设计文档中若规定 `REGISTER_NONE_TILING`，应在算子工程实现时改注为"按 dfx 生成器实际要求注册一个具体实例化"。
- 升级 CANN 版本后，应回归验证 `REGISTER_NONE_TILING` 路径是否已修复（若修复则本卡失效边界下移）。

## 验证信号

- op-compile 在 `REGISTER_NONE_TILING` 下抛 `UnboundLocalError: cannot access local variable 'tiling_struct_size'`（ascendc_compile_dfx.py:239 `_generate_binary_for_tiling_without_register`）（来源平台：950；证据等级: tool_observed）
- 修复后 kernel 入口文件 `op_kernel/softplus_v2_grad_kernel.cpp` 使用 `REGISTER_TILING_DEFAULT(SoftplusV2GradTilingData<8>);`（来源平台：950；证据等级: artifact_verified）

## 失效边界

- **未验证，推断可迁移**：a2 / a3 平台（同为 ACLNN 算子仓开发、同一 asc_op_compile_base dfx 生成器链路，推断同样触发；未构建这两平台，未实测）。
- **未验证场景**：非模板化（单一 concrete struct）的 TilingData + `REGISTER_NONE_TILING` 是否同样触发 UnboundLocalError，无此组合证据。
- **版本边界**：本卡现象针对 CANN 9.1.0 的 `asc_op_compile_base`；更高版本若修复 `_generate_binary_for_tiling_without_register` 的 `tiling_struct_size` 赋值，则 `REGISTER_NONE_TILING` 可能变为可用，本卡届时需复核。

## 坏实践

- 误以为 `REGISTER_NONE_TILING` 是"无 tiling struct"的合法声明而照搬设计文档字面值，不读 dfx 生成器报错栈即认定是 kernel C++ 代码问题。
- 在 dfx 报错后，不去读 `ascendc_compile_dfx.py` 的 `_generate_binary_for_tiling_without_register` 实现，而是反复改 kernel C++ 代码（徒劳）。