---
type: "Runbook"
title: "GMM A8W4 561000 缺 binary entry 无 JIT 回退"
description: "GroupedMatmulWeightNz 新增 MX A8W4 Blaze tensor API dispatch 分支后，tiling 生成新 key 91308033，但预编译 kernel binary 无此 entry；ascend950 未设 jitCompile.flag 致无 JIT 回退，ACLNN 返回 561000。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# GMM A8W4 561000 缺 binary entry 无 JIT 回退

## 现象

在 Ascend950 上为 GroupedMatmulWeightNz 算子新增 MX A8W4 伪量化的 Blaze tensor API 迁移路径后（commit `22bf20bb2 feat(gmm): migrate pseudo-quant MX A8W4 to tensor API`，分支 `gmmmxa8w4`，仓库 `ops-transformer`），构建并安装 ophost so 与 kernel 源码到 CANN 安装目录，运行 aclnn 示例 `test_aclnn_grouped_matmul_weight_nz_mxa8w4_multi`，返回：

```text
aclnnGroupedMatmulWeightNz failed. ERROR: 561000
aclnnGroupedMatmulWeightNz MxA8W4 multi test failed. ERROR: 561000
```

开 `ASCEND_GLOBAL_LOG_LEVEL=0 + ASCEND_SLOG_PRINT_TO_STDOUT=1` 后看到完整错误链：

```text
[ERROR] RUNTIME(...) BinaryGetFunctionByEntry:BinaryGetFunctionByEntry failed, funcEntry=91308033
[ERROR] RUNTIME(...) rtsFuncGetByEntry:ErrCode=107000, desc=[invalid value], InnerCode=0x7110001
[ERROR] ASCENDCL(...) aclrtBinaryGetFunctionByEntryImpl:Binary get function by entry Failed, runtime result = 107000
[ERROR] OP(...) errno[361001] OpName:[aclnnGroupedMatmulWeightNz_0_GroupedMatmul] aclrtBinaryGetFunction byEntry failed, kernel name: GroupedMatmul, tiling key: 91308033
[ERROR] OP(...) errno[361001] Init function handle failed.
[ERROR] OP(...) errno[361001] Kernel Run failed. opType: 4, GroupedMatmul
[ERROR] OP(...) errno[361001] launch failed for GroupedMatmul, errno:361001.
[ERROR] OP(...) errno[561000] This is an error in GMM launch aicore
```

即 runtime 按 tiling key `91308033` 在预编译 kernel binary 里查 kernel function entry，查不到（`BinaryGetFunctionByEntry failed, funcEntry=91308033` → `aclrtBinaryGetFunctionByEntry` 返回 107000 → op_executor 报 361001 → ACLNN 层最终抛 561000 "GMM launch aicore"）。

## 适用场景

- **工程形态**：GroupedMatmulWeightNz 新增 MX A8W4 Blaze tensor API dispatch 分支后。
- **API 路径**：`aclnnGroupedMatmulWeightNzGetWorkspaceSize` / `aclnnGroupedMatmulWeightNz` → op_executor → `aclrtBinaryGetFunctionByEntry`（按 tiling key 查预编译 kernel binary）。
- **验证平台与版本**：Ascend950（SoC ascend950），CANN 9.1.0，bisheng-compiler 9.1.0。
- **适用粒度**：`operator_family`——GroupedMatmul 系列在 ascend950 注册时新增 dispatch 分支均会踩此坑。
- **不适用/未验证场景**：ascend310P（已显式设 `jitCompile.flag = static_false,dynamic_false`，行为不同，未在本 trace 验证）；a2/a3 未接触。

## 判别方法

当 ACLNN 报 `561000` 且日志含 `aclrtBinaryGetFunctionByEntry failed ... tiling key: <N>` 时，按下列探针定位：

1. **触发**：ACLNN 调用返回 `561000`，运行时日志含 `BinaryGetFunctionByEntry failed, funcEntry=<N>` 与 `aclrtBinaryGetFunctionByEntry failed, kernel name: <Op>, tiling key: <N>`（本例 `N=91308033`）。
2. **可执行探针 A——解码 tiling key**：从日志 `tilingKeyConfigure` 行读各字段（本例 `socVersionType:20 quantizationScenario:0 algorithm:2 transposeSituation:1 antiquantType:4 quantType:0 optionInputSituation:0 weightFormat:1 templateCustom:15 isSingleMultiSingle:1`，组合得 `91308033`）。该 key 由 `grouped_matmul_apt.cpp` 的 tiling 代码按 dispatch 条件生成。
3. **可执行探针 B——查预编译 kernel binary 是否有该 entry**：`find /usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/grouped_matmul_apt/ -name "*.o"`，列出已注册 tiling key。
4. **原始观测**（`tool_observed`）：本例 binary 有 `37756929 / 38805505 / 36782081 / 39927809 / 36716545 / 36773888 / 36773920 / 53551104 / 53551136 / 53485568 / 53485600` 等，但**无 `91308033`**。`grep "jitCompile" grouped_matmul_def.cpp` 仅 310P 命中（`config310P.ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")`），ascend950 config950 **未设**该 flag。
5. **核 dispatch 源码**：`grouped_matmul_apt.cpp` 中新增的 Blaze 分支：

```cpp
if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
              WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4 &&
              ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
              ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
#if IS_BLAZE
        INVOKE_GMM_WEIGHT_QUANT_MXA8W4_TENSOR_API_IMPL();
#else
        ...
```

`IS_BLAZE` 为真时走新 macro `INVOKE_GMM_WEIGHT_QUANT_MXA8W4_TENSOR_API_IMPL()`（调用 `GmmWeightQuantTensorApiMxKernel`）。该分支产生的 tiling key（91308033）未被任何预编译 binary 覆盖。
6. **解释**：新增 kernel dispatch 分支产生了一个旧预编译 kernel binary 不含的 tiling key（`91308033`）。ascend950 的 `grouped_matmul_def.cpp` 没有像 310P 那样显式声明 `jitCompile.flag`，导致 CANN runtime 在 `BinaryGetFunctionByEntry` 查不到 entry 时不会触发 CCE 动态编译回退，而是直接返回 `rtsFuncGetByEntry` 失败（107000），层层上报为 ACLNN `561000`。同时单独重建 ophost so 不足以让 flag 生效——flag 实际被 op info JSON 承载，必须 `--pkg` 重建并同步 JSON。此为解释层根因分析；修复闭环。

## 根因

新增 kernel dispatch 分支（Blaze MX A8W4 tensor API）产生了一个旧预编译 kernel binary 不含的 tiling key（`91308033`）。ascend950 的 `grouped_matmul_def.cpp` 没有像 310P 那样显式声明 `jitCompile.flag`，导致 CANN runtime 在 `BinaryGetFunctionByEntry` 查不到 entry 时不会触发 CCE 动态编译回退，而是直接返回 `rtsFuncGetByEntry` 失败（107000），层层上报为 ACLNN `561000`。同时单独重建 ophost so 不足以让 flag 生效，flag 实际被 op info JSON 承载，必须 `--pkg` 重建并同步 JSON。

## 验证信号

- ACLNN 运行返回 `561000`，日志 `BinaryGetFunctionByEntry failed, funcEntry=91308033` + `aclrtBinaryGetFunctionByEntry failed, kernel name: GroupedMatmul, tiling key: 91308033`（来源平台：950；证据等级：tool_observed）。
- ascend950 config 未设 `jitCompile.flag`（310P 设 `static_false,dynamic_false`），`grep "jitCompile" grouped_matmul_def.cpp` 仅 310P 命中（来源平台：950；证据等级：tool_observed）。
- 预编译 binary `.o` 列表无 `91308033`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- ascend310P 已显式 `static_false,dynamic_false`，行为不同，本判别未在 310P 验证——需按平台逐个确认 flag 语义。
- a2/a3 平台未接触，`platforms` 不含。
- 若改用 GEIR 模式（GE 图编译触发 CCE JIT），Blaze 模板实例化复杂，CCE 编译单轮可超 40 分钟，需更高性能编译服务器或预编译 binary（`build.sh --pkg --opkernel`）替代 JIT。

## 坏实践

- 只跑 `build.sh --ophost`、只替换 `libophost_transformer.so` 就期望 jitCompile 生效——flag 实际在 op info JSON，so 替换不更新 JSON。
- 直接删 `/usr/local/Ascend/cann-9.1.0/opp/built-in/.../grouped_matmul_apt/` 下预编译 binary 期望 CANN 自动 JIT 回退——ascend950 未设 flag 时不会回退，反而直接 561000。