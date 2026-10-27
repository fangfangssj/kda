---
type: "Runbook"
title: "ACLNNTYPE aclnn autogen 与手写同名符号冲突"
description: "regbase 算子 CMakeLists 带 ACLNNTYPE aclnn 时构建系统在 build/autogen/ 按 Nnopbase API 自动生成 aclnn_<op>.cpp；若 op_api/ 下同时存在同名手写 aclnn_<op>.cpp（旧式 L0 API），两个翻译单元定义相同符号，链接阶段报 multiple definition。判别方法：检查 autogen 与手写文件是否同名同符号，确认 ACLNNTYPE 值。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ACLNNTYPE aclnn autogen 与手写同名符号冲突

## 现象

SoftplusGrad（regbase broadcast 算子）执行 `bash build.sh --pkg --ops=softplus_grad --soc=ascend950 --vendor_name=softplus_grad_nn -j8`，编译通过但链接阶段报重复定义：

```
multiple definition of 'aclnnSoftplusGradGetWorkspaceSize'
multiple definition of 'aclnnSoftplusGrad'
```

两个翻译单元都定义了同一符号：

1. `build/CMakeFiles/opbuild_gen_aclnn_all.dir/autogen/aclnn_softplus_grad.cpp.o` —— 构建系统自动生成（`ACLNNTYPE aclnn` 触发）。
2. `activation/softplus_grad/CMakeFiles/ophost_nn_opapi_obj.dir/op_api/aclnn_softplus_grad.cpp.o` —— 仓内手写。

顶层 `CMakeLists.txt` 用 `add_modules_sources(... OPTYPE softplus_grad ACLNNTYPE aclnn ...)`；`op_api/aclnn_softplus_grad.cpp` 是从非 regbase 模板拷贝来的旧式 L0 API 实现（含 `l0op::SoftplusGrad`、`CREATE_EXECUTOR`、手写 CheckNotNull/CheckDtypeValid/Contiguous/Cast）。

## 适用场景

- **工程形态**：算子仓构建（ops-registry-invoke / regbase broadcast DAG 算子）。
- **API 路径**：`add_modules_sources(... ACLNNTYPE aclnn ...)` 触发 `build/autogen/aclnn_<opname>.cpp` 自动生成；autogen 用 regbase `Nnopbase` API。
- **验证过的平台与 CANN 版本**：Ascend 950（dav-3510 / arch35），CANN 9.1.0。
- **适用粒度**：`operator_family`（所有 regbase 算子，CMakeLists 带 `ACLNNTYPE aclnn`）。
- **不适用或未验证场景**：`ACLNNTYPE aclnn_exclude` 不生成 aclnn 接口（仅 GEIR 通路），无此冲突；旧式非 regbase 算子（无 autogen）需手写 aclnn，不适用本卡。

## 判别方法

### 触发信号

regbase 算子（CMakeLists 带 `ACLNNTYPE aclnn`）构建链接阶段报 `multiple definition of 'aclnn<OpName>GetWorkspaceSize'` 与 `multiple definition of 'aclnn<OpName>'`，编译阶段无错。错误中两个翻译单元分别来自 `build/autogen/aclnn_<opname>.cpp.o` 与 `op_api/aclnn_<opname>.cpp.o`。

### 探针

执行以下检查确认冲突来源：

```bash
# 1. 检查 autogen 是否生成了同名 aclnn 文件
find build -name "aclnn_<opname>.cpp" -path "*/autogen/*"
# 2. 检查 op_api 下是否存在同名手写文件
ls op_api/aclnn_<opname>.cpp 2>/dev/null
# 3. 确认 CMakeLists 的 ACLNNTYPE 值（aclnn / aclnn_exclude）
grep ACLNNTYPE op_host/CMakeLists.txt
```

若步骤 1 与步骤 2 同时命中，且步骤 3 显示 `ACLNNTYPE aclnn`（非 `aclnn_exclude`），则确认为 autogen 与手写同名冲突。

### 观测

两个翻译单元各自定义 `aclnn<OpName>GetWorkspaceSize` 与 `aclnn<OpName>`：

- autogen 文件（`build/autogen/aclnn_<opname>.cpp`）：regbase `Nnopbase` 风格（基本空指针检查 + `NnopbaseGetExecutor` + run，无自定义校验/Cast）。
- 手写文件（`op_api/aclnn_<opname>.cpp`）：旧式 L0 API（`l0op::<OpName>` + `CREATE_EXECUTOR` + 手写 CheckNotNull/CheckDtypeValid/Contiguous/Cast）。

### 解释

`ACLNNTYPE aclnn` 让构建系统在 `build/autogen/` 下按算子名生成 `aclnn_<opname>.cpp`（regbase `Nnopbase` executor 风格）。若 `op_api/` 下同时存在同名手写 `aclnn_<opname>.cpp`（旧式 L0 API），两个翻译单元各自定义 `aclnn<OpName>GetWorkspaceSize` 与 `aclnn<OpName>`，链接器报 `multiple definition`。regbase 算子的 tiling/kernel 用 `BroadcastBaseTiling<DAG>` + `BroadcastSch`，应配套 autogen 的 Nnopbase 通路，旧式手写 L0 API 文件属历史残留。规避方法：若必须手写 aclnn，文件名须避开 `aclnn_<opname>.cpp`（参考 leaky_relu_grad 用 `aclnn_leaky_relu_backward.cpp`，与 autogen 的 `aclnn_leaky_relu_grad.cpp` 不同名，避免冲突）。

## 验证信号

- 链接错误 `multiple definition of 'aclnnSoftplusGradGetWorkspaceSize'` 与 `multiple definition of 'aclnnSoftplusGrad'` 在 `bash build.sh --pkg --ops=softplus_grad --soc=ascend950` 构建输出中观测到，两个翻译单元分别来自 `build/autogen/aclnn_softplus_grad.cpp.o` 与 `op_api/aclnn_softplus_grad.cpp.o`（来源平台：950；证据等级：tool_observed）
- `mv op_api/aclnn_softplus_grad.cpp op_api/aclnn_softplus_grad.cpp.bak` 后重建 `bash build.sh --pkg` 链接通过，产出安装包 `cann-ops-nn-softplus_grad_nn_linux-x86_64.run`（来源平台：950；证据等级：tool_observed）
