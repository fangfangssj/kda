---
type: "Runbook"
title: "ACLNN 561102 单 dtype 原型注册与 opbuild 字节约束"
description: "多 dtype 反量化算子 ST 全失败，根因为 opbuild 要求同一算子原型内同一 input 各 dtype 字节数一致，int8/int32 不能共存于单一 REG_OP/OP_ADD，需按字节数拆分算子原型组。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ACLNN 561102 单 dtype 原型注册与 opbuild 字节约束

## 现象

在 Ascend950PR (DAV_3510) 上用 Regbase 模式开发反量化算子 `DequantizeCustom`，spec.yaml 要求覆盖 6 种 dtype 组合（input int8/int32 × scale float16/bfloat16/float32）。完成首轮实现后运行 C++ ST 用例（`test_aclnn_dequantize`，40 个 case，覆盖全部 dtype 组合与多 rank 广播）。

ST 测试二进制编译通过（`Built target test_aclnn_dequantize`），但上板运行时 40 个用例全部失败。plog 对每个用例记录一条错误级联：

```
[ERROR] [op_executor.h:178][NNOP][~UniqueExecutor] errno[561102] OpName:[aclnnDequantize_22] When aclnnDequantizeGetWorkspaceSize do success, ReleaseTo(executor) should be called before return.
[ERROR] [kernel_workspace.cpp:1501][NNOP][SelectBin] errno[-1060232821] OpName:[aclnnDequantize_22_DequantizeAiCore] Failed to execute operator ... Reason: The dtype or format of the actual input or output parameter of the operator is inconsistent with that defined in the operator prototype OpDef.
[ERROR] [kernel_workspace.cpp:108][NNOP][GetWorkspace] errno[561000] ... Kernel GetWorkspace ...
[ERROR] [dequantize.cpp:65][NNOP][DequantizeAiCore] errno[561103] ... DequantizeAiCore failed.
[ERROR] [op_dfx.cpp:227][NNOP][DoLaunch] errno[361001] ... Init function handle failed.
```

关键判别句是 `SelectBin` 的 `Reason: The dtype or format of the actual input or output parameter ... inconsistent with that defined in the operator prototype OpDef`——指向算子原型注册的 dtype 与运行时实际传入的 dtype 不匹配，而非 kernel 内部逻辑错误。同期的 issue 文件记载：Kernel 二进制包 (`dequantize_custom.json`) 仅注册了 `int32 + float32 → float32` 一种组合，因此 5 种含 int8/bf16/fp16 的组合全部 `Cannot find binary for op DequantizeCustom`。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（op_graph proto + op_host + op_kernel + ACLNN L0），多 dtype 算子注册。
- API 路径：`REG_OP` / `OP_ADD`（op_graph proto）→ `ascendc_all_ops` 编译 → `simplifiedKey`（编码 input/scale/output dtype）→ 运行时 `SelectBin` 按 dtype 选 kernel 二进制。
- 验证过的平台与 CANN 版本：Ascend950PR（`--soc=ascend950`），CANN 9.1.0。仅 950 有实测证据。
- 适用粒度：`operator_family`——任何"同一 input 槽位需要同时注册不同字节数 dtype（如 int8 1B + int32 4B）"的多 dtype 算子。
- 路线优先级：`recommended_fastpath`——按字节数拆分算子原型组是 opbuild 约束下的正解。
- 不适用或未验证场景：a2/a3 未实测（推断可迁移，见 `## 失效边界`）；单 dtype 算子不触发本约束。

## 判别方法

**触发信号**（tool_observed）：ST/联调上板运行出现 `errno[561102] ReleaseTo(executor) should be called before return` + `SelectBin ... inconsistent with that defined in the operator prototype OpDef` + `errno[561000] Cannot find binary for op`，且测试二进制本身编译通过。

**探针**（按序执行）：

1. 读 kernel 二进制配置 `build/autogen/.../<op>_custom.json`（或运行时 `SelectBin` 报告的 OpDef），确认实际注册的 input/output dtype 列表是否覆盖 spec 要求的全部组合：

   ```bash
   cat operators/<op>/build/autogen/ascend_all_ops/*/<op>_custom.json | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['inputs'],d['outputs'])"
   ```

2. 读 `op_graph/<op>_proto.h` 的 `REG_OP(...)` / `.OP_ADD` 序列，统计同一 input 槽位声明的 dtype 集合。
3. 对比 plog：若 `SelectBin` 的 reason 是 "dtype ... inconsistent with ... OpDef" 而非 "tiling key" 或 "UB alignment"，则定位到原型注册层（非 kernel 计算层）。

**原始观测**（本 trace 内 tool_observed）：

plog 对 `aclnnDequantize_22`..`aclnnDequantize_39` 逐 case 输出上述错误级联；issue 文件 `operators/dequantize/issues/issue_20260621_kernel_dtype_single_01.md` 记载二进制配置仅 `{"input":"int32","scale":"float32","zero_point":"int32","output":"float32"}`，期望 6 组（int8/int32 × float16/bfloat16/float32）。

**解释**（interpretation，agent 根因分析，非 API 规范主张）：opbuild 在编译 `REG_OP` + `OP_ADD` 算子原型时，要求**同一算子原型内同一 input 参数的所有候选 dtype 具有相同字节数**。反量化算子的 `input` 与 `zero_point` 既要支持 `int8`（1B）又要支持 `int32`（4B），二者字节数不同，不能共存于同一个 `REG_OP`/`OP_ADD` 原型——若强行放一起，opbuild 只能为其中一种生成 `simplifiedKey`（本例实际只生成了 int32+float32），其余 dtype 在运行时 `SelectBin` 阶段找不到匹配二进制，报 `inconsistent with OpDef`。

正解是把算子原型按 input 字节数拆成两组：

- 4B 组：`REG_OP(DequantizeCustomI32)` + `OP_ADD` input/scale/zp/output 声明 int32/float*/int32/float*；
- 1B 组：`REG_OP(DequantizeCustomI8)` + `OP_ADD` 声明 int8/float*/int8/float*；

并相应拆分 `apt.cpp`（两个 op_graph 源）、kernel struct 头、kernel 编译目标，ACLNN L0 接口按 input dtype 分派到对应算子名。`OP_ADD` 同样受字节约束，不能只拆 `REG_OP` 而把所有 dtype 塞进单个 `OP_ADD`。该机制属 agent 推断（基于 plog `dtype inconsistent with OpDef` 与 issue 文件根因一致），拆分后的完整 before/after 代码未在 trace 中逐字恢复。

## 验证信号

- ST 上板 40 用例全失败，plog 对 `aclnnDequantize_22`..`39` 输出 `errno[561102] ReleaseTo(executor) should be called before return` + `SelectBin ... inconsistent with that defined in the operator prototype OpDef`（来源平台：950；证据等级：tool_observed）。
- issue 文件记载 Kernel 二进制仅注册 `int32 + float32 → float32` 一种组合，plog 同时记 `GetWorkspace: Cannot find binary for op DequantizeCustom`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- a2 / a3 平台未在本 trace 内实测，opbuild 的字节约束属工具链行为而非芯片代际特征，推断可迁移（未验证，推断可迁移）。
- 若算子只支持单一字节数的 input（如全是 int8 或全是 int32），则不触发本约束，无需拆分算子原型。

## 坏实践

- 只拆 `REG_OP` 而不拆 `OP_ADD`：`OP_ADD` 同样有字节约束，仍会编译失败或只生成单 dtype 二进制。该路线标记为 `banned`。
- 把 int8 与 int32 放进同一 `OP_ADD` input 槽位：opbuild 静默只生成一种 `simplifiedKey`，运行时才以 `SelectBin inconsistent` 暴露。