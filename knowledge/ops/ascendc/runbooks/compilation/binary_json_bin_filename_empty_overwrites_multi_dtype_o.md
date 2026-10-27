---
type: "Runbook"
title: "binary.json bin_filename 空致多 dtype kerne"
description: "ops-nn 算子 binary.json 的 bin_filename 字段为空字符串时，build 系统将其替换为 _high_performance，使多 dtype/多 tiling_key 的 kernel .o 全部同名互相覆盖，最终生成的 kernel json 把所有 binList 条目都标成 float32 且指向同一个 .o；修复为每个条目填 OpType_<hash> 格式的 bin_filename。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# binary.json bin_filename 空致多 dtype kerne

## 现象

smooth_l1_loss_grad 算子在 Ascend950 (arch35) 上编译部署后，运行 ACLNN eager 示例前检查 `build/binary/ascend950/bin/config/ascend950/smooth_l1_loss_grad.json`（运行时 kernel 映射表）发现严重错乱：

1. `binList` 的 **3 个条目 `dtype` 全部是 `"float32"`**——本应分别是 `float16` / `float32` / `bfloat16`；
2. 3 个条目 **全部指向同一个 `_high_performance.json`**（即同一个 `.o` kernel 文件）；
3. 部署目录里实际只有 1 个名为 `_high_performance.o` 的 kernel 文件（无算子名前缀、无 tiling_key hash），而旧的可正常工作的部署文件名是 `SmoothL1LossGrad_<hash>_high_performance.o`。

即 3 个 dtype 的 binary 编译任务虽然各自“成功”，但因输出文件名相同（`_high_performance.o`）互相覆盖，最终留下的 `.o` 只是最后一次编译的产物（本例恰好是 fp32）。

根因在算子的 binary 配置源文件 `config/ascend950/smooth_l1_loss_grad_binary.json`：3 个 `op_list` 条目的 `bin_filename` 字段都是**空字符串** `""`。build 系统在生成中间 binary config（`build/binary/.../<hash>_param.json`）时把空 `bin_filename` 替换为字面量 `_high_performance`，`asc_opc` 再用该值作为输出 `.o` 文件名——3 个 dtype 同名，互相覆盖。

## 适用场景

- **工程形态**：构建（ops-nn 算子 `binary.json` 配置 + `build.sh`/`make binary` 编译多 dtype/多 tiling_key kernel）。
- **API 路径**：`config/ascend950/<op>_binary.json` 的 `op_list[].bin_filename` 字段；build 系统 `generate_bin_scripts` → `asc_opc --input_param=<hash>_param.json` 用 `bin_filename` 作为输出 .o 名前缀。
- **验证过的平台与 CANN 版本**：Ascend950（arch35 / DAV_3510），CANN 9.0.0。
- **适用粒度**：`platform_wide`（任何在 `binary.json` 注册 ≥2 个 dtype/tiling_key 且 `bin_filename` 为空的 ops-nn 算子都适用）。
- **路线优先级**：`recommended_fastpath`（手动填 `bin_filename` 是最快修复；完全清理 build 目录重建也能恢复自动填充）。
- **不适用或未验证场景**：单 dtype 算子无覆盖问题；a2/a3 平台未在本 trace 验证（推断可迁移，见 `## 失效边界`）。

## 根因

`asc_opc` 用 binary config（`<hash>_param.json`）中的 `bin_filename` 作为输出 `.o`/`.json` 的文件名前缀，最终文件名为 `<bin_filename>_high_performance.o`。

- 当 `bin_filename` 为合法的 `OpType_<tiling_key_hash>` 时，每个 dtype 各自生成不同文件名（如 `SmoothL1LossGrad_5bc6..._high_performance.o`），互不覆盖。
- 当 `binary.json` 中 `bin_filename` 为空字符串 `""` 时，build 系统的 `generate_bin_scripts` 步骤将其替换为字面量 `_high_performance`（而非用 op_type 或 tiling_key hash 自动填充），导致 3 个 dtype 的 `.o` 文件名都退化为 `_high_performance.o`，最后一次编译覆盖前两次。

reference 中其他算子（如 `CrossEntropyLossGrad`）的 `binary.json` 都显式填了 `OpType_<hash>` 格式的 `bin_filename`，因此不会触发。本算子 `bin_filename` 在 git HEAD 中本就为空；早期迁移编译能生成正确文件名，是因为 build 目录全新时 `gen_bin_scripts` 会按 tiling_key 自动计算并填充 hash——一旦 build 目录有残留（中间 config 已生成），该自动填充步骤被跳过，`bin_filename` 维持空/替换值，覆盖发生。

## 解决配方

before（`binary.json` 的 3 个条目，`bin_filename` 为空 → 互相覆盖）：

```json
{
  "op_type": "SmoothL1LossGrad",
  "op_list": [
    { "dtype": "float16",   "bin_filename": "", ... },
    { "dtype": "float32",   "bin_filename": "", ... },
    { "dtype": "bfloat16", "bin_filename": "", ... }
  ]
}
```

build 系统生成的中间 config 与最终 kernel json（错配状态）：

```json
// build/binary/ascend950/bin/config/ascend950/smooth_l1_loss_grad.json
"binList": [
  { "dtype": "float32", "kernel_name": "_high_performance", ... },
  { "dtype": "float32", "kernel_name": "_high_performance", ... },
  { "dtype": "float32", "kernel_name": "_high_performance", ... }
]
```

after（为每个条目填 `OpType_<hash>`，hash 按 tiling_key，与 dtype 一一对应；`PrecisionReduceFlag` 回退不影响 tiling_key，故 hash 不变）：

```json
{
  "op_type": "SmoothL1LossGrad",
  "op_list": [
    { "dtype": "float16",   "bin_filename": "SmoothL1LossGrad_5bc679e351dee42a7f22bae11adb9ce5", ... },
    { "dtype": "float32",   "bin_filename": "SmoothL1LossGrad_ea046702745dc9e76f3eac43fc4ed7d5", ... },
    { "dtype": "bfloat16", "bin_filename": "SmoothL1LossGrad_9d172dcac118827bbf1b9af0f850f3e5", ... }
  ]
}
```

修改后清理 `build/binary/` 与 `build/tbe/config/` 残留并重新 `make binary -j1` + `make package`，部署后 3 个 dtype 各 1 个 `.o` + 1 个 `.json`，文件名正确带 `SmoothL1LossGrad_<hash>_high_performance` 前缀，kernel json 的 `binList` 3 条目 dtype 分别为 float16/float32/bfloat16 指向各自 `.o`。

等效替代：**完全删除 `build/` 目录从头 `cmake` 重建**——build 目录全新时 `gen_bin_scripts` 会自动填充 hash，即使 `bin_filename` 仍为空也能生成正确文件名（本 trace 最终即用此法一次性通过，3 个 kernel `.o` 均正确命名）。代价是重新配置 cmake + 编译 host + 重新下载/编译第三方库，耗时显著长于手动填字段。

## 设计决策

- **为何选手动填 `bin_filename` 而非每次重建 build 目录**：手动填字段是 reference 算子的标准做法（`CrossEntropyLossGrad` 等均如此），改一行即修；重建 build 目录耗时大（cmake 配置 + host + 第三方库），且本 trace 早期重建曾因第三方库阶段超时。两种修复等价，手动填字段是 fast path。
- **hash 从何而来**：旧的可正常工作的 `.o` 文件名中已含 hash（`SmoothL1LossGrad_<hash>_high_performance.o`），该 hash 按 tiling_key 计算，只随 dtype 变；回退 `PrecisionReduceFlag` 不改 tiling_key，故 hash 复用旧值安全。
- **被放弃的设计**：曾尝试只删 `CMakeFiles/exe_compile_*` 片段后 `cmake` 重新配置——但删 CMakeFiles 破坏了 `exe_compile` target 的实际编译规则（build.make 内容变空），make 认为 up-to-date 却不产出 `.o`；最终放弃局部修复，改用完整重建或手动填字段。

## 预防

- `binary.json` 注册多 dtype/多 tiling_key 时，**每个 `op_list` 条目必须显式填 `bin_filename: "OpType_<hash>"`**，不留空——这是 reference 算子的既定规范。
- 编译后、部署前检查 `build/binary/.../bin/config/ascend950/<op>.json` 的 `binList`：dtype 列是否覆盖目标 dtype 集、是否有重名 `kernel_name`——发现重名/全 float32 即说明 bin_filename 缺失。
- 部署后检查 `.o` 文件名是否带 `OpType_<hash>` 前缀；出现裸 `_high_performance.o` 即为覆盖信号。

## 验证信号

本问题与修复均在本 trace 的 P0/P1 编译部署步骤内实测（Ascend950 / CANN 9.0.0）：

- 修复前 kernel json：`binList` 3 条目全 `dtype=float32`、全指向 `_high_performance.json`（来源平台：Ascend950；证据等级: tool_observed）。
- 修复前部署目录：仅有 `_high_performance.o`（无前缀），旧版为 `SmoothL1LossGrad_<hash>_high_performance.o`（来源平台：Ascend950；证据等级: tool_observed）。
- 手动填 `bin_filename` 或完全重建后：3 个 dtype 各 1 个 `.o` + 1 个 `.json`，文件名带 `SmoothL1LossGrad_<hash>_high_performance`，`binList` dtype 为 float16/float32/bfloat16 各异（来源平台：Ascend950；证据等级: tool_observed）。
## 失效边界

- a2/a3 平台未在本 trace 实测（未验证，推断可迁移：`binary.json` 与 `bin_filename` 机制与平台无关）。
- 若未来 build 系统在 `bin_filename` 为空时强制用 `op_type` + `tiling_key` hash 自动填充并缓存失效，则空字段不再是坑；但在当前 CANN 9.0.0 build 系统下空字段是真实陷阱。

## 坏实践

- 发现 `binList` 全 float32 后只改 kernel json 而不改源 `binary.json`：下次 `build.sh` 会重新生成错配的 kernel json，问题复发。
- 删 `binary.json` 源文件后再 `make binary`：`asc_opc` 报 `The input_param: .../smooth_l1_loss_grad_binary_high_performance.json_2 is not exist`，`gen_bin_scripts` 误认 target 已 built 不重新生成，需清 `tbe/.impl_timestamp` + 重新 cmake 才能恢复——比直接填字段更折腾。
