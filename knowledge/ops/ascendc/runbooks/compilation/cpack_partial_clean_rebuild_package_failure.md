---
type: "Runbook"
title: 部分清理重建后 CPack 打包失败须全清重建
description: "自定义算子部分清理重建后 CPack 报 \"CMake Error ... file INSTALL cannot copy ... binary/dynamic ... Success.\" 与 \"CPack Error: Error when generating package\"；rm -rf build 全量清理重建即可恢复打包。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# 部分清理重建后 CPack 打包失败须全清重建

## 现象

在 Ascend950PR（CANN 9.1.0）上调试自定义算子（RoiAlign 类）：先做过一轮"只清 kernel 构建产物"的部分清理（`rm -rf build/op_kernel/ascendc_kernels` + `~/.cache/ascend/kernel_cache`），修完 kernel 编译错误后再次 `bash build.sh --soc=ascend950 -j8`，kernel 编译与链接均显示成功（`[100%] Built target binary`、两个 TilingKey 变体 `Generating ... Done`），但 CPack 打包阶段失败，报错原文：

```
CMake Error at .../build/op_kernel/cmake_install.cmake:46 (file):
  file INSTALL cannot copy file
  ".../build/op_kernel/ascendc_kernels/binary/dynamic"
  to
  ".../build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/bench_no_harness_roi_align_custom/op_impl/ai_core/tbe/bench_no_harness_roi_align_custom_impl/dynamic/":
  Success.
Call Stack (most recent call first):
  .../build/cmake_install.cmake:64 (include)

CPack Error: Error when generating package: bench_no_harness_roi_align_op_prj
gmake: *** [Makefile:71: package] Error 1
```

特征很怪：错误信息末尾是 `Success.`（把复制失败报成 Success），失败的"文件"是 `binary/dynamic` 这个动态二进制**目录**而非单个文件。当时算子包无法生成（`.run` 不出现），无法继续安装验证。

## 适用场景

- 工程形态：CANN 自定义算子仓构建打包（`build.sh` → cmake → CPack `custom_opp_ubuntu_x86_64.run`）。
- API 路径：CPack External 打包器、`cmake_install.cmake` 的 `file(INSTALL ...)` 规则、`build/op_kernel/ascendc_kernels/binary/dynamic` 目录。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——打包链行为与具体算子无关。
- 路线优先级：safety_fallback（fallback 声明：本卡是规避配方，非根因修复；代价是全量重编的时间成本，相对部分清理重建多一轮完整编译，本会话单轮约分钟级）。
- 明确不适用 / 未验证场景：干净 checkout 首次构建（未见此错）；错误信息不同（如磁盘满、权限）的场景；其他 CANN 版本未验证。

## 根因

部分清理重建后 `build` 目录内新旧产物并存，`binary/dynamic` 目录状态与 CPack 的 install 规则不一致（install 规则按文件处理一个实为目录/状态异常的路径），复制失败但错误等级被误报为 `Success.`；确切成因未追查（见 ## 遗留问题）。

## 解决配方

处置（本会话验证）：

```bash
rm -rf operators/bench_no_harness_roi_align/build
cd operators/bench_no_harness_roi_align && bash build.sh --soc=ascend950 -j8
```

全量清理后重建，打包恢复正常，输出：

```
Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created.
[INFO] Build completed successfully!
[INFO] Kernel binary: .../BenchNoHarnessRoiAlign_c3990d1b16e9498379a812cdc211752d.o
[INFO] Package: .../build/custom_opp_ubuntu_x86_64.run
```

随后 `cd build && bash custom_opp_ubuntu_x86_64.run --force` 安装成功，继续验证。

预防：对 `build/` 只做"全清"或"不动"两种操作，避免只删其中一部分子目录（尤其混过 kernel 缓存清理/重编之后）；打包失败且报文指向 `binary/dynamic` 时直接全清重建，不必在旧目录里修。

## 设计决策

- 选全量清理而非修补 install 规则：错误在 CPack/cmake_install 生成物层面，修补成本高于重编；全清是可重复、确定性的恢复路径。
- 判别依据：`[100%] Built target binary` 已证明编译无问题，失败局限在 packaging 阶段——不必回头查 kernel 代码。
- 当前方案路线档位：safety_fallback（规避型；根因修复待构建链侧定位）。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0，CPack + gmake。
- 相关文件/目录：`build/cmake_install.cmake`、`build/op_kernel/cmake_install.cmake`（file INSTALL 规则）、`build/op_kernel/ascendc_kernels/binary/dynamic`、`build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/`（打包暂存区）、`build/custom_opp_ubuntu_x86_64.run`（产物）。

## 验证信号

来源说明：本结论来自该算子构建排障会话，同一环境前后对照。

- 修复前：`file INSTALL cannot copy ... binary/dynamic ... Success.` + `CPack Error: Error when generating package`，`gmake package Error 1`（来源平台：950；证据等级：tool_observed）。
- 修复后（rm -rf build 全量重建）：`Self-extractable archive ... successfully created`、`Build completed successfully!`、`.run` 包生成，安装后失败用例正常执行（随后全量 ST 36/36）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 根因未追到机制层：`binary/dynamic` 目录为何复制失败、错误为何显示 `Success.` 均未验证；本卡仅证明"全清重建"能确定性恢复。
- CANN 9.1.0 验证；其他版本未验证，推断可迁移。
- 若全清后仍失败，属其他根因（磁盘/权限/路径），本卡不适用。

**未决项**：

- 【优先】复现并追根因到机制层：部分清理后 `binary/dynamic` 的目录状态与 `file(INSTALL)` 规则为何冲突、错误信息为何为 `Success.`；确认最小清理集（是否只需删 `binary/dynamic` 或 `_CPack_Packages`）以降低规避成本。
## 坏实践

- 在经历过部分删除的 `build/` 目录上反复增量重试打包——失败状态会持续，浪费时间。
- 被 `Success.` 字样迷惑以为构建其实成功、跳过打包失败继续安装旧包——会拿旧二进制做验证。

## 误诊

- 无（本闭环内未走弯路；编译阶段日志已确认 `[100%] Built target binary` 才进入打包排查）。
