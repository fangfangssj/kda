---
type: "Runbook"
title: "ASCEND350 非 SocVersion 枚举成员致编译失败"
description: "把 build.sh 的 SOC 别名 ascend350 当 C++ 枚举名 platform_ascendc::SocVersion::ASCEND350 写进 tiling 代码会编译报错；该枚举只有 ASCEND950，build.sh 中 ascend350 与 ascend950 同映射 arch 3510 但 C++ 侧无 ASCEND350 成员。判别=grep platform_ascendc.h 的 SocVersion 枚举确认合法成员名，build.sh SOC 别名不能直接当 C++ 枚举名用。"
tags: ["task.debugging", "topic.compilation", "topic.quality.compatibility.version_migration"]
status: stable
sources: []
created_at: "2026-08-18T07:12:12Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ASCEND350 非 SocVersion 枚举成员致编译失败

## 适用场景

- 工程形态：算子 op_host 适配新平台编译（`bash build.sh --ops=<op> --soc=ascend950 --pkg`，产物安装到 vendor）。
- API 路径：`platform_ascendc::SocVersion` 枚举（op_host tiling 代码用 `socVersion == platform_ascendc::SocVersion::ASCEND950` 做平台分支）；`build.sh` 的 `SOC_TO_ARCH` SOC→arch 映射。
- 验证过的平台与 CANN 版本：Ascend950，CANN 9.0.0（本地 `/usr/local/Ascend/cann-9.0.0`）。
- 适用粒度：`platform_wide`（所有从历史 ascend350 视角迁移到 ascend950 的算子，只要把 SOC 别名当成 C++ 枚举名写进 tiling/def 代码都会踩）。
- 路线优先级：`recommended_fastpath`（直接删 ASCEND350 分支，只判 ASCEND950）。
- 不适用或未验证场景：a2 (ascend910b) / a3 (ascend910_93) 不进入 ASCEND950 分支，行为不变，未实测，标注未验证、推断可迁移。

## 判别方法

触发（tool_observed）：在 Ascend950 上执行 `bash build.sh --ops=advance_step --soc=ascend950 --pkg` 编译 op_host，bisheng/C++ 前端在 `op_host/advance_step_tiling.cpp` 第 286-287 行报错 `ASCEND350 is not a member of platform_ascendc::SocVersion`。当时代码按 SOC 名同时判断 ASCEND950 与 ASCEND350 两个分支。

探针（executable check）：grep 头文件中 `platform_ascendc::SocVersion` 枚举的合法成员名（如 `grep -n "ASCEND" /usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/include/tiling/platform/platform_ascendc.h`），核对 `ASCEND950` 是否存在、`ASCEND350` 是否存在；并读 `build.sh` 的 `SOC_TO_ARCH` 表，核对 `ascend350` 与 `ascend950` 各自映射到的 arch 字符串。

观测（tool_observed）：`platform_ascendc::SocVersion` 枚举的合法成员含 `ASCEND950`，无 `ASCEND350`；`build.sh` 的 `SOC_TO_ARCH` 把 `ascend350` 与 `ascend950` 都映射到同一个 arch 字符串 `3510`（运行时落到同一 SoC），但 C++ 侧枚举没有 `ASCEND350` 这个成员。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：`build.sh` 的 `SOC_TO_ARCH` 让 `ascend350` 与 `ascend950` 同映射到 arch `3510`，容易让人误以为 C++ 侧也有对应的 `SocVersion::ASCEND350` 枚举成员。但 `platform_ascendc::SocVersion` 枚举只暴露 `ASCEND950`，根本没有 `ASCEND350`——`ascend350` 只是 build.sh 的 SOC 别名（字符串），不是 C++ 枚举名。把 SOC 别名误当成 C++ 枚举名写进 tiling 代码就会触发编译失败。判别核心是“SOC 别名（build.sh 字符串）≠ C++ 枚举成员名”，出现 `X is not a member of platform_ascendc::SocVersion` 时应先 grep 枚举头文件确认合法成员名，而不是去查 kernel 或 binary。

## 验证信号

来源：本 trace 在 Ascend950（CANN 9.0.0）上对 `ops-nn/optim/advance_step` 做 `bash build.sh --ops=advance_step --soc=ascend950 --pkg` 编译，构建工具事件与 grep 枚举头文件工具事件。

- 修复前构建报 `ASCEND350 is not a member of platform_ascendc::SocVersion`，定位 `op_host/advance_step_tiling.cpp` 第 286-287 行的 `socVersion == ASCEND950 || ASCEND350` 分支。（来源平台：950；证据等级：tool_observed）
- grep `platform_ascendc.h` 的 `SocVersion` 枚举：合法成员含 `ASCEND950`、`ASCEND910B` 等，无 `ASCEND350`。（来源平台：950；证据等级：tool_observed）
- 修复 edit（`advance_step_tiling.cpp`）：tool_call 的 oldString 含 `ASCEND950 || ASCEND350` 双分支、newString 仅判 `ASCEND950`，逐字可核。（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 仅 Ascend950 / CANN 9.0.0 实测。a2/a3 对应 soc 名的同型平台分支判断未实测，推断可迁移（`SocVersion` 枚举合法成员名跨平台一致），标注未验证、推断可迁移。
- 本卡只诊断到“SOC 别名误当 C++ 枚举成员名”这一根因；删除 ASCEND350 分支后的构建闭环（`Build binary success`）未作为非 redacted tool_result 落地，本卡不发布 repair_recipe 的 verbatim before/after 代码块。
- 若 CANN 后续头文件真的新增 `ASCEND350` 枚举成员，本判据需重新审视；当前以头文件实际合法成员名为准。