---
type: "Runbook"
title: "rebase 引入新 kernel 文件旧 tiling 类型致编译失败"
description: "Conv3DBackpropInputV2/Conv3DTransposeV2/ExtendConvTranspose arch35 tiling data 框架迁移分支 rebase 上游后，新增 kernel 文件 conv3d_dx_small_kernel.h 仍引用已删除的旧 tiling 类型 conv_bp_v2_kernel::TConv3DInputV2Tiling 和旧嵌套访问 conv3DDxTiling，入口宏传新类型引用而 SmallKernel::Init 收旧类型指针，类型不匹配致编译失败；grep 可定位未适配文件。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: draft
sources: []
created_at: "2026-08-17T18:09:48Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
aliases: ["Conv3dDxSmallKernel", "DEFINE_STUCT_FIELD"]
---

# rebase 引入新 kernel 文件旧 tiling 类型致编译失败

## 现象

在 `Conv3DBackpropInputV2` / `Conv3DTransposeV2` / `ExtendConvTranspose` arch35 把 tiling data 从手写嵌套 struct（`conv_bp_v2_kernel::TConv3DInputV2Tiling`，`tilingData->conv3DDxTiling.xxx` 嵌套访问）重构为 `TILING_DATA_DEF` 框架新类型（`Conv3DBackpropInputArch35TilingData`，`tilingData.xxx` 扁平访问、`Init` 入参由 `&tilingData` 指针改为 `tilingData` 引用）后，执行 `git pull --rebase` 拉入上游最新代码。上游提交 `980b9378b ExtendConvTranspose新增small kernel模板` 引入的新 kernel 文件 `conv3d_backprop_input_v2/op_kernel/arch35/conv3d_backprop_input_v2/conv3d_dx_small_kernel.h` 仍基于迁移前的旧 tiling 结构编写。`grep` 排查发现该文件第 38 行 `Init` 参数为 `const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData`（旧类型指针）、第 43 行 `tiling_ = &tilingData->conv3DDxTiling;`（嵌套访问）、第 72 行 `const conv_bp_v2_kernel::TConv3DInputV2Tiling* tiling_ = nullptr;`（旧类型成员）——旧类型在新框架下已不存在，入口宏 `CONV3D_DX_INPUT_RUN_OP` 传 `tilingData`（新类型引用），SmallKernel::Init 收旧类型指针，类型不匹配致编译失败。

## 适用场景

- **工程形态**：算子仓内 tiling data 框架迁移后，rebase 上游新增 kernel 文件的适配诊断（上游新文件基于迁移前的旧 API 编写）。
- **API 路径**：kernel 侧 `Init` 签名 / `tiling_` 成员类型 / `tiling_->` 字段访问；入口宏 `GET_TILING_DATA(tilingData, tiling)` + `CONV3D_DX_INPUT_RUN_OP` / `CONV3D_DX_TRANSPOSE_RUN_OP` / `EXTEND_CONV_TRANSPOSE_RUN_OP`。
- **验证过的平台与 CANN 版本**：未定位（trace 内无 npu-smi / AddConfig / 编译运行实测；arch35 为 kernel 目录命名）。仅经 grep + 字段存在性验证。
- **适用粒度**：`operator_family`（Conv3DBackpropInputV2 / Conv3DTransposeV2 / ExtendConvTranspose arch35 共用入口宏与 `Conv3DBackpropInputArch35TilingData`）。
- **路线优先级**：`recommended_fastpath`（把新文件 3 处旧类型/嵌套访问改到新类型扁平访问是唯一保编译路径）。
- **不适用或未验证场景**：arch32 入口（仍用 arch32 自己的 tiling data 结构，正常现象）；非该框架迁移的算子；实际编译/上板未验证。

## 判别方法

- **触发**：tiling data 框架迁移分支 rebase 上游后，arch35 kernel 目录中 grep 命中仍引用已删除旧 tiling 类型（`conv_bp_v2_kernel::TConv3DInputV2Tiling`）和旧嵌套访问（`conv3DDxTiling`）的文件。

- **可执行探针**：对 arch35 kernel 目录跑 grep，扫描旧类型名和旧嵌套访问模式：

  ```bash
  grep -rn "conv_bp_v2_kernel::\|conv3DDxTiling\|TConv3DInputV2\|GET_TILING_DATA_WITH_STRUCT\|REGISTER_TILING_DEFAULT" \
    <op_kernel>/arch35/
  ```

  命中的文件即为未适配到新框架的残留文件（arch32 命中属正常，应排除）。

- **原始观测**（`tool_observed`）：grep 命中 `conv3d_dx_small_kernel.h` 三处旧式代码：

  ```
  # line 38 — Init 参数旧类型指针
  __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                              const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData,
                              GM_ADDR bias = nullptr, GM_ADDR scale = nullptr)

  # line 43 — 旧嵌套访问
      tiling_ = &tilingData->conv3DDxTiling;

  # line 72 — 旧类型成员
      const conv_bp_v2_kernel::TConv3DInputV2Tiling* tiling_ = nullptr;
  ```

  入口宏 `CONV3D_DX_INPUT_RUN_OP` 传 `tilingData`（新类型 `Conv3DBackpropInputArch35TilingData` 引用），SmallKernel::Init 收旧类型指针 `conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData*`，类型不匹配；且旧类型 `conv_bp_v2_kernel::TConv3DInputV2Tiling` 在迁移后已被删除，名字解析将失败。

- **解释**：tiling data 框架迁移分支只改了迁移时刻已存在的 kernel 文件（`conv3d_bp_impl_base.h`、`conv3d_dx_rowc_block.h`、`conv3d_dx_kernel_split_block.h` 等已适配为新类型 `const Conv3DBackpropInputArch35TilingData&` 引用 + `InitTilingData(tilingData)` 取地址存指针 + `tiling_->` 扁平访问）。上游后续提交 `980b9378b` 新增的 `conv3d_dx_small_kernel.h` 基于迁移前的旧 API 编写（`Init` 收旧类型指针、`tiling_ = &tilingData->conv3DDxTiling;` 嵌套访问、`tiling_` 是旧类型指针）。rebase 并入后，旧类型已删除、入口宏传参方式从指针改为引用，故类型不匹配 + 旧类型不存在 → 编译失败。根因是框架迁移分支未覆盖 rebase 引入的上游新文件——迁移契约（新类型 + 扁平访问 + 引用传参）需同步到所有 arch35 kernel 文件，包括 rebase 后才进入分支的新文件。

## 根因

tiling data 框架迁移分支只改了迁移时刻已存在的 kernel 文件，上游后续提交 `980b9378b` 新增的 `conv3d_dx_small_kernel.h` 基于迁移前的旧 API 编写。rebase 并入后，它引用的旧类型 `conv_bp_v2_kernel::TConv3DInputV2Tiling` 已被删除、入口宏传参方式从指针改为引用，故编译失败。根因是框架迁移分支未覆盖 rebase 引入的上游新文件——迁移契约（新类型 + 扁平访问 + 引用传参）需同步到所有 arch35 kernel 文件。

## 验证信号

- grep 命中 `conv3d_dx_small_kernel.h:38/43/72` 旧类型 `conv_bp_v2_kernel::TConv3DInputV2Tiling` 与旧嵌套访问 `conv3DDxTiling`（来源平台:未定位；证据等级:tool_observed）
- 适配后 grep 确认 arch35 内无 `conv3DDxTiling` 嵌套访问、无 `GET_TILING_DATA_WITH_STRUCT`、无 `REGISTER_TILING_DEFAULT`、无 `TConv3DInputV2`（来源平台:未定位；证据等级:tool_observed）
- SmallKernel + `conv3d_bp_small_kernel_func.h` 中 `tiling_->` 访问的所有字段在新结构体 `Conv3DBackpropInputArch35TilingData` 中全部存在（来源平台:未定位；证据等级:tool_observed）

## 预防

- tiling data 框架迁移分支 rebase 上游后，全量扫描 arch35 kernel 目录中所有 `Init` 签名 / `tiling_` 类型 / `tilingData->` / `&tilingData->` 嵌套访问，确认无旧类型残留：`grep -rn "conv_bp_v2_kernel::\|conv3DDxTiling\|TConv3DInputV2\|GET_TILING_DATA_WITH_STRUCT\|REGISTER_TILING_DEFAULT"` 应只在 arch32（未迁移范围）命中。
- 上游新增的 kernel 文件（`git show <upstream_commit> --stat --diff-filter=A` 列出的新增 .h/.cpp）是迁移契约最易遗漏的点：rebase 后必须逐个核对其 `Init` 签名与 `tiling_` 类型。

## 失效边界

- 仅适用于 `TILING_DATA_DEF` 框架迁移后 rebase 引入的上游新文件适配诊断；非该框架迁移的算子不适用。
- arch32 kernel 入口仍用旧嵌套访问（`conv3d_backprop_input_v2.cpp`、`conv3d_transpose_v2.cpp`），属 arch32 自己的 tiling data 结构，不在本迁移范围，正常现象。
- 推断可迁移到其他 arch 上同类框架迁移分支（未验证，推断可迁移）。
- 实际编译/上板信号未在 trace 内采集；上述为 grep 级静态验证，非编译通过/失败的原始 build-error 输出。

## 坏实践

- 只改 `tiling_` 成员类型而不改 `Init` 签名 → 入口宏传引用、`Init` 收指针，类型不匹配，仍编译失败。
- 只改 `Init` 签名而不改 `tiling_ = &tilingData->conv3DDxTiling;` → 嵌套访问旧 struct 子字段，新类型无 `conv3DDxTiling` 成员，编译失败。

## 误诊

无（grep 一次定位到 `conv3d_dx_small_kernel.h` 是唯一未适配的新文件；`980b9378b` 引入的其他 3 个文件 `conv3d_backprop_input_v2_small_kernel_tiling.cpp/.h` 与 `conv3d_bp_small_kernel_func.h` 已用新 `set_` 方式或字段在新结构体中存在，无需改）。
