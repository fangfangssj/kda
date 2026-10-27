---
type: "Runbook"
title: "REGISTER_TILING_DEFAULT 多 struct 冲突"
description: "同一算子多条模板实例化分支用 REGISTER_TILING_DEFAULT 注册不同 TilingData struct 时 opbuild 报 Only one default tiling structure can be configured；判别=grep 各 default 注册点比对 struct，改用 REGISTER_TILING_FOR_TILINGKEY 按 key 注册规避。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-18T08:28:54Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# REGISTER_TILING_DEFAULT 多 struct 冲突

## 现象

为 `max_pool3d_grad`（Ascend950 / arch35）新增一条 separable 三-pass kernel 分支时，在 kernel 入口 `pooling/max_pool3d_grad/op_kernel/max_pool3d_grad.cpp` 的 `if constexpr (IS_SIMT == 0 && USE_INT64_INDEX == 1)` 分支内用 `REGISTER_TILING_DEFAULT(MaxPool3DGradSeparableTilingData)` 注册了一条与既有 SmallKernel 的 `Pool3DGradNCDHWTilingData` **不同**的 TilingData struct。构建 kernel binary（`bash build.sh --soc=ascend950 --ops=max_pool3d_grad --opkernel -j8`）时 opbuild 直接报错并停止生成 binary：

```
[message:Only one default tiling structure can be configured.]
```

## 适用场景

- **工程形态**：算子仓开发，在已有算子 `op_kernel/<op>.cpp` 入口为新的 template 实例化组合（如 `IS_SIMT==0 && USE_INT64_INDEX==1`）新增一条带独立 TilingData struct 的 kernel 分支。
- **API 路径**：`REGISTER_TILING_DEFAULT` / `REGISTER_TILING_FOR_TILINGKEY` / `GET_TILING_DATA_WITH_STRUCT` 三个 tiling 注册与取用宏。
- **验证平台**：Ascend950（arch35），构建命令 `bash build.sh --soc=ascend950 --ops=max_pool3d_grad --opkernel -j8`；仅构建验证，未上板运行。
- **适用粒度**：`operator_family`——任何在已有算子仓新增带独立 TilingData struct 的 kernel 分支都会遇到，不限于 pooling。
- **不适用或未验证场景**：新分支若复用既有 default struct（不引入新 struct），仍可继续用 `REGISTER_TILING_DEFAULT`，不触发本冲突。a2/a3 未构建，未验证。

## 判别方法

**触发条件**（tool_observed 失败症状）：算子 kernel binary 构建在 opbuild 阶段失败，报错信息为 `Only one default tiling structure can be configured.`，且该算子的 `op_kernel/<op>.cpp` 入口存在多条 `if constexpr` 模板实例化分支。

**可执行探针**：
1. 在 `op_kernel/<op>.cpp` 入口 grep `REGISTER_TILING_DEFAULT`，列出所有调用点及其传入的 struct 名。
2. 比对各调用点传入的 struct——若存在 ≥2 个**不同**的 struct 被注册为 default，即为根因。
3. 对照 `REGISTER_TILING_FOR_TILINGKEY` 调用点，确认既有按 key 注册的分支（如 SIMT 路径）使用的 struct 与 default 槽位的 struct 不同、且不占 default 槽位。

**原始观测**（Ascend950，max_pool3d_grad kernel binary 构建）：
- 修改前：opbuild 报 `Only one default tiling structure can be configured.`，binary 生成中断（来源平台：950；证据等级：tool_observed）。
- 将 separable 分支的 `REGISTER_TILING_DEFAULT(MaxPool3DGradSeparableTilingData)` 改为 `REGISTER_TILING_FOR_TILINGKEY(<key 字符串>, MaxPool3DGradSeparableTilingData)` 后重新构建，该报错消失、binary 生成恢复（来源平台：950；证据等级：tool_observed）。

**机制解释**（根因分析，属 interpretation）：`REGISTER_TILING_DEFAULT(T)` 把 struct `T` 注册为**整个算子的唯一 default tiling structure**。同一算子的多条 `if constexpr` 模板实例化分支在编译期都被实例化（即使运行时只走一条）；若两条分支用 `REGISTER_TILING_DEFAULT` 注册了**不同的** struct，opbuild 在收集 default 注册时看到两个候选 default，无法决定用哪一个，遂报 `Only one default tiling structure can be configured.`。既有 SmallKernel 多条分支全部注册**同一个** struct 故不冲突；SIMT 分支用 `REGISTER_TILING_FOR_TILINGKEY` 按 key 注册、不占 default 槽位。新增分支引入第二个不同 struct 却仍占 default 槽位，正是冲突来源——按 key 注册即可规避。

## 预防

- 在已有算子仓新增 kernel 分支前，先 grep `REGISTER_TILING_DEFAULT` 与 `REGISTER_TILING_FOR_TILINGKEY`，确认既有 default struct 是哪个；只要新分支要用自己的 TilingData struct，一律走 `REGISTER_TILING_FOR_TILINGKEY` + 独立 key 字符串，不要碰 `REGISTER_TILING_DEFAULT`。
- tiling key 字符串必须与既有分支（含 SIMT）的 key 严格不重叠；最稳的做法是把本分支的模板参数组合完整拼进 key。
- host 侧 `GetTilingKey()` 与 kernel 侧 key 字符串必须一字一致；构建前自检两端。

## 坏实践

- 多条分支共用 `REGISTER_TILING_DEFAULT` 却传不同 struct——必触发本冲突。
- 为了让某条分支"看起来是主路径"而把既有 SmallKernel 也改成 `FOR_TILINGKEY`——不必要且扩大改动面；只新增分支改 key 即可。

## 验证信号

- 修改前 kernel binary 构建：opbuild 报 `Only one default tiling structure can be configured.`，binary 生成中断（来源平台：950；证据等级：tool_observed）。
- 将 separable 分支改为 `REGISTER_TILING_FOR_TILINGKEY` 后重新构建，该报错消失、kernel binary 恢复生成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡仅构建验证于 Ascend950（arch35）；a2/a3 未构建。`REGISTER_TILING_DEFAULT`/`REGISTER_TILING_FOR_TILINGKEY` 是 AscendC tiling 框架宏，跨平台机制一致——未验证，推断可迁移到 a2/a3，但 tiling key 字符串的取值空间与既有分支是否重叠需在各平台算子仓里重新核对。
- tiling key 字符串的具体值与 max_pool3d_grad 既有分支的参数体系绑定；迁移到其它算子时须换成该算子的模板参数组合，不能照抄。