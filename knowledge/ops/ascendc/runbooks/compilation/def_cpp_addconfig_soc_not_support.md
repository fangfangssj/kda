---
type: "Runbook"
title: "def.cpp 缺 AddConfig 致 SOC_NOT_SUPPORT"
description: "ops-math def.cpp 的 AICore().AddConfig 列表漏 ascend950，build 不生成该 SOC ops-info（aic-ascend950-ops-info.ini 0 字节），TTK 运行时报 SOC_NOT_SUPPORT；判别：SOC_NOT_SUPPORT + 0 字节 ops-info.ini 即指向 def.cpp AddConfig 遗漏，先查 def.cpp 注册块而非 kernel。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# def.cpp 缺 AddConfig 致 SOC_NOT_SUPPORT

## 适用场景

- 工程形态：算子 vendor 构建安装（ops-math `conversion/<op>` 目录 `build.sh --soc=<soc> --opkernel --pkg`，产物 `.run` 安装到 `opp/vendors/custom_math`）。
- API 路径：op_host `<op>_def.cpp` 的 `this->AICore().AddConfig("<soc>", <aicore_config>)` 注册块。
- 验证过的平台与 CANN 版本：Ascend950，CANN 9.0.0。
- 适用粒度：operator_family（所有需在新 SOC 上注册的 ops-math 算子 def.cpp）。
- 路线优先级：recommended_fastpath（在 def.cpp AddConfig 列表补目标 SOC）。
- 不适用或未验证场景：skills-dev 仓 def.cpp 已含 ascend950 注册时不适用；非 ops-math 仓结构的算子不适用；a2/a3 对应 soc 名的同型注册未实测，标注未验证、推断可迁移。

## 判别方法

触发（tool_observed）：在 Ascend950 上构建安装 custom_math vendor 的 `fill_diagonal_v2` 后，TTK 单用例（`-d -c -b=release` 三模式）均报 `Operator [fill_diagonal_v2] is not supported on current SOC.` / `SOC_NOT_SUPPORT`；TTK 日志只从 `bessel_i1e_custom`、`custom_nn`、`custom_cv` 三个 vendor 加载 ascend950 ops-info，不从 `custom_math` 加载。检查 installed vendor：`custom_math/.../config/ascend950/aic-ascend950-ops-info.json` 不存在，`aic-ascend950-ops-info.ini` 为 0 字节；构建日志明示 `[INFO] On [ascend950], [fill_diagonal_v2] not supported.`。

探针（executable check）：读 / grep op_host `<op>_def.cpp` 的 `AICore().AddConfig` 注册块，核对目标 SOC（如 `ascend950`）是否在列表中；并检查 vendor 安装目录 `<vendor>/op_impl/ai_core/tbe/config/<soc>/aic-<soc>-ops-info.json` 是否存在且非 0 字节。本例对 `conversion/fill_diagonal_v2/op_host/fill_diagonal_v2_def.cpp` 执行 filesystem read（def.cpp 实际读取工具事件在 trace 中可解析）。

观测（tool_observed）：`fill_diagonal_v2_def.cpp` 的 `AICore().AddConfig` 列表只覆盖 `ascend910b`、`ascend910_93`、`ascend310p`、`kirinx90`、`kirin9030`，缺 `ascend950`；对照 skills-dev 仓同算子 def.cpp 含 `AddConfig("ascend950", aicore_config)`，证实是 ops-math 仓的注册遗漏，而非 kernel 不支持。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：build 按 def.cpp 的 `AddConfig` 列表逐 SOC 生成 `aic-<soc>-ops-info.json`/`.ini`。def.cpp 未对 ascend950 注册 → 该 SOC 的 ops-info 不生成（`.ini` 0 字节、`.json` 缺失）→ 安装包未带 ascend950 ops-info → TTK 运行时在 custom_math vendor 找不到该 SOC 的算子注册 → `SOC_NOT_SUPPORT`。结论：`SOC_NOT_SUPPORT` + 目标 SOC ops-info.ini 0 字节这一组合，直接指向 def.cpp `AddConfig` 遗漏目标 SOC，应先查 def.cpp 注册块而非先排查 kernel 实现或 binary 编译。

## 验证信号

来源：本 trace 在 Ascend950（CANN 9.0.0）上对 ops-math `conversion/fill_diagonal_v2` 做 `build.sh --soc=ascend950 --opkernel --pkg` + `.run` 安装后 TTK 重跑，构建日志与 vendor 目录文件检查工具事件。

- 修复前 TTK 三模式均报 `Operator [fill_diagonal_v2] is not supported on current SOC.` / `SOC_NOT_SUPPORT`；构建日志 `[INFO] On [ascend950], [fill_diagonal_v2] not supported.`。（来源平台：950；证据等级：tool_observed）
- 修复前 `custom_math` vendor 的 `aic-ascend950-ops-info.ini` 0 字节、`aic-ascend950-ops-info.json` 不存在，TTK 不从 custom_math 加载 ascend950 ops-info。（来源平台：950；证据等级：tool_observed）
- 探针读 `fill_diagonal_v2_def.cpp` 的 `AICore().AddConfig` 列表仅含 ascend910b/ascend910_93/ascend310p/kirinx90/kirin9030，缺 ascend950；skills-dev 同算子 def.cpp 含 `AddConfig("ascend950", aicore_config)`。（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅 Ascend950 / CANN 9.0.0 实测。其他 SOC（a2/a3 对应 soc 名）的 `AddConfig` 注册机制同理但未实测，推断可迁移，标注未验证、推断可迁移。
- `aicore_config` 复用同一 config 的前提是目标 SOC 支持该 config 内的全部 dtype；若目标 SOC 不支持某 dtype（如 BF16），需单独 config，本卡的判别方法只覆盖 AddConfig 列表遗漏，不覆盖 dtype 不匹配导致的注册失败。
- 本卡只诊断到 def.cpp `AddConfig` 遗漏这一根因；补注册后的 build/TTK 闭环（ops-info.json 生成、binary 内核产出、TTK 通过）未在本卡恢复。