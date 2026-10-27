---
type: "Runbook"
title: "extern 预编译 kernel 自定义包致 aclnn 161001"
description: "某 extern 预编译 kernel 算子经 --ops 自定义包安装后，aclnn C 测试的 NnopbaseGetExecutor 返回 null 并报 161001；用于核查安装包 metadata 与 kernel binary 的可定位性，不提供未经验证的打包替代路线。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["ASCEND_CUSTOM_OPP_PATH"]
---

# extern 预编译 kernel 自定义包致 aclnn 161001

## 现象

在 `allto_all_matmul_apace`（MX 量化 FP8 AllToAll+Matmul 融合算子）的 aclnn 通路验证阶段，按自定义算子包方式编译安装：`bash build.sh --pkg --soc=ascend950 --ops=allto_all_matmul_apace -j32`，`.run` 包 install 到 `cann/vendors/custom_transformer/`，`source .../set_env.bash`，再跑纯 C aclnn 测试 `test_aclnn_apace`（先 `aclInit` 再 `aclnnAlltoAllMatmulApaceGetWorkspaceSize`）。`aclInit` 正常，但 `aclnnAlltoAllMatmulApaceGetWorkspaceSize` 直接返回 `aclError:161001`。算子背景：本算子 kernel 由 apace 独立预编译，op_host 侧通过 extern 声明 + `<<<>>>` 直调，算子本身没有走 opc 编译的 kernel 源码。

## 适用场景

- **工程形态**：aclnn C 测试（纯 C 调 `aclInit` + `aclnnXxxGetWorkspaceSize` + `aclnnXxx`），自定义算子包安装到 `cann/vendors/<vendor>/`。
- **API 路径**：CANN 自定义算子 aclnn 运行时注册——`NnopbaseGetExecutor("<OpName>")` 查算子 executor，并读取安装包中的 kernel metadata。
- **验证过的平台与 CANN 版本**：Ascend950（DAV_3510），CANN 9.2.0。仅在本 trace 实测。
- **适用粒度**：`exact_task`。只验证了 `allto_all_matmul_apace` 的这一打包结果，不能泛化为所有 extern kernel 必然失败。
- **路线优先级**：`diagnostic_only`。本卡保留已观测的不可用包状态和注册层探针，不提供替代打包路线。
- **不适用或未验证场景**：kernel 本身走 opc 编译的常规自定义算子不触发（opc 会生成 binary_info_config.json）；Python `torch.ops.torch_npu` 调用路径是否绕过此限制未在本 trace 验证。

## 已观测的失败状态

**尝试**：用 `bash build.sh --pkg --soc=ascend950 --ops=allto_all_matmul_apace -j32` 只编译该算子，`.run` 包 install 到 `cann/vendors/custom_transformer/`，`source set_env.bash` 后跑 aclnn C 测试 `test_aclnn_apace`。

**客观失败**：`aclnnAlltoAllMatmulApaceGetWorkspaceSize` 返回 `aclError:161001`，`NnopbaseGetExecutor("AlltoAllMatmulApace")` 返回 null。多次重装、校验 `libop_host_aclnnInner.so` 内确有对应 C++ class 符号后仍失败。日志曾指向 `binary_info_config.json` 缺失或损坏，但原始日志未完整保留，因此只能把 metadata 完整性作为检查方向，不能把 opc 生成机制当作已证实根因。

**边界**：手工补入一份 JSON 后仍为 161001，说明“文件存在”不足以恢复 executor；该 trace 没有继续验证 metadata 内容、binary 路径和 runtime 注册之间的精确契约，也没有得到可工作的对照包。

## 验证信号

- aclnn C 测试 `aclnnAlltoAllMatmulApaceGetWorkspaceSize` 返回 `aclError:161001`（来源平台：950；证据等级：tool_observed）。
- `NnopbaseGetExecutor("AlltoAllMatmulApace")` 返回 null（来源平台：950；证据等级：tool_observed）。
- 多次重装自定义包、补齐 aclnn wrapper 签名、手工补 binary_info_config.json 后 161001 稳定复现（来源平台：950；证据等级：tool_observed）。

## 误诊

在定位到 binary_info_config 缺失之前，agent 先修了一个真实的签名 bug：手动写的 `aclnn_allto_all_matmul_apace.{h,cpp}` 比 auto-generated inner function 少 3 个参数，导致 inner function 的 null 检查失败返回 161001。补齐签名后所有指针非空，但仍 161001，证明签名 bug 只是表层、真正卡点是 executor 注册层。签名 diff 的 verbatim before/after 未从 trace 恢复（autogen inner 文件 read 结果与相关 edit 内容均被 redact 为 `_see below_`），上述 3 个缺失参数基于 agent 报告，标 agent_reported，不作为本卡验证信号。

## 预防

- extern 预编译 kernel 进入 aclnn 包验证前，检查安装包是否同时包含 executor 注册信息、有效 kernel metadata 和 metadata 引用的实际 binary。
- 手工创建空壳或猜测格式的 JSON 不能作为修复验证；必须以 `GetWorkspaceSize` 成功和 kernel 可执行为闭环。

## 失效边界

- 本边界针对"extern 预编译 kernel + 自定义包 aclnn 注册"组合；常规 opc 编译 kernel 的算子不在此列。
- 全包构建、预置 metadata 等替代打包方式均未验证，不属于本卡结论。
- Python `torch.ops.torch_npu` 调用路径是否绕过 `NnopbaseGetExecutor` 的 binary_info_config 依赖未验证。
