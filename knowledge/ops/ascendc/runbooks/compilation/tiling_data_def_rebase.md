---
type: "Runbook"
title: "TILING_DATA_DEF 迁移 rebase 字段失配判别"
description: "把 tiling data 从手写嵌套 struct 重构为 TILING_DATA_DEF 宏框架后 rebase 上游,上游在旧 struct 新增的 7 个字段未同步补到新类型的 TILING_DATA_FIELD_DEF,kernel 侧 tiling_-> 直读这些字段会编译失败。判别=git status 报 rebase in progress 且冲突落在 SetRunInfoTiling 区域时,diff HEAD 侧旧 struct 字段集与本分支新类型字段集,缺字段即为根因,不可只取一侧。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: draft
sources: []
created_at: "2026-08-18T08:08:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# TILING_DATA_DEF 迁移 rebase 字段失配判别

## 现象

在 `conv/conv3d_backprop_input_v2`(及 `conv3d_transpose_v2`、`extend_conv_transpose`)arch35 上执行 `git pull --rebase`,把本地分支(把 tiling data 从手写嵌套 struct `conv_bp_v2_kernel::TConv3DInputV2Tiling` + 直接成员赋值 `dxt.xxx=` 重构为标准 `TILING_DATA_DEF` 宏框架的新类型 `optiling::Conv3DBackpropInputArch35TilingData` + `dxt.set_xxx()`)replay 到上游 commit 之上时,`conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.cpp` 出现 rebase 冲突。`git status` 报 `rebase in progress` + 两个 unmerged path。

冲突本质:上游在旧 struct 上**新增了 7 个字段**(`khDilation`、`kwDilation`、`hoExpand`、`woExpand`、`dkHkWk`、`hkWk`、`fixedShiftVal`),HEAD 侧仍用旧类型 + 直接赋值,本分支侧已切到新类型 + `set_` 但新类型定义里**没有这 7 个字段**;同时 kernel 侧 `impl/` 文件大量通过 `self->ctx.tiling_->khDilation`、`tiling_->dkHkWk`、`tiling_->fixedShiftVal` 等直读这些字段——若按本分支侧解决冲突(保留 `set_` 但不补字段),kernel 编译会失败(字段不在新类型中)。

## 适用场景

- **工程形态**:算子仓内 tiling data 框架迁移(手写嵌套 struct → `TILING_DATA_DEF` 宏框架),rebase 上游时同文件有新增字段。
- **API 路径**:`BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF(<type>, <name>)` / `END_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS`;host 侧 `dxt.set_<field>(...)`,kernel 侧 `tiling_-><field>` 直读。
- **验证过的平台与 CANN 版本**:未定位(trace 内无 npu-smi / AddConfig / 编译运行实测;arch35 仅为 kernel 目录命名,非 SPEC-frontmatter §6 注册平台词)。仅经 git status + grep + rebase 状态验证。
- **适用粒度**:`operator_family`(Conv3DBackpropInputV2 / Conv3DTransposeV2 / ExtendConvTranspose arch35 共用 `Conv3DBackpropInputArch35TilingData`)。
- **路线优先级**:`diagnostic_only`——本卡只给判别方法与根因解释,修复实现的终版 before/after 未从 trace 逐字恢复。
- **不适用或未验证场景**:arch32 旧 tiling data 不在本次重构范围;非 arch35 算子未验证;实际编译/上板未验证(trace 内未编译)。

## 根因

`TILING_DATA_FIELD_DEF` 框架的字段定义是 host/kernel 双向契约:`TILING_DATA_FIELD_DEF(<type>, <name>)` 在 host 侧生成 `set_<name>()` 写入方法、在 kernel 侧生成可直接 `tiling_-><name>` 读取的字段。当本分支把 tiling data 切到新框架但**只迁移了迁移时刻已存在的字段**,而上游后续在旧 struct 新增了 7 个字段并在 kernel 侧(impl/ 文件)通过 `tiling_->` 直读它们时,新类型若不同步补这 7 个 `TILING_DATA_FIELD_DEF`,kernel 侧 `tiling_->khDilation` 等访问会因字段不存在而编译失败。rebase 冲突只是把这个"字段集不一致"暴露出来——真正的根因是新框架类型定义与 kernel 侧访问的字段集不同步,而非冲突标记本身。

机制链:框架迁移只搬已有字段 ↔ 上游在旧 struct 新增 7 字段并在 kernel 直读 ↔ 新类型缺 7 个 `TILING_DATA_FIELD_DEF` ↔ kernel `tiling_->` 访问字段不存在 ↔ 编译失败。

## 判别方法

**触发条件**(tool_observed 失败症状):`TILING_DATA_DEF` 框架迁移分支 `git pull --rebase` 后,`git status` 报 `rebase in progress; onto <upstream>` 且有 unmerged path,冲突标记 `<<<<<<<` / `=======` / `>>>>>>>` 落在 `op_host/op_tiling/*.cpp` 的 `SetRunInfoTiling` / `SetRunBaseShapeInfoTiling` 区域。

**可执行探针**:
1. `grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" <op_dir>/` 定位冲突文件与区域,确认冲突是否落在 tiling setter 函数内。
2. 读冲突 HEAD 侧:它仍用旧 struct + 直接赋值(`dxt.<field> = ...`),记录 HEAD 侧出现的全部字段名集合 `F_head`。
3. 读冲突本分支侧:它用新类型 + `set_<field>(...)`,记录本分支侧 `set_` 调用的字段名集合 `F_branch`。
4. 读新类型定义头(`*_tiling_data_arch35.h` 内 `BEGIN_TILING_DATA_DEF`...`END_TILING_DATA_DEF`),记录新类型已声明的 `TILING_DATA_FIELD_DEF` 字段集合 `F_type`。
5. 读 kernel 侧 `impl/` 文件所有 `tiling_-><field>` 直读访问,记录 kernel 依赖的字段集合 `F_kernel`。
6. 比对:`F_head − F_type` 即为上游新增但新类型缺失的字段;若 `F_kernel ∩ (F_head − F_type)` 非空,则这些字段在 kernel 侧被直读但新类型未定义 → 按本分支侧解决冲突(保留 `set_` 但不补字段)必然编译失败。

**原始观测**(arch35,trace 内 git/rebase 状态验证):
- `git status` 报 `rebase in progress; onto <upstream>` + 2 个 unmerged path(`conv3d_backprop_input_v2_inner_product_tiling.cpp` 两段冲突区域)。
- HEAD 侧旧 struct 含 7 个上游新增字段(khDilation/kwDilation/hoExpand/woExpand/dkHkWk/hkWk/fixedShiftVal)用直接赋值;本分支侧新类型 `Conv3DBackpropInputArch35TilingData` 缺这 7 个 `TILING_DATA_FIELD_DEF`。
- kernel 侧 `impl/` 文件通过 `tiling_->` 直读这 7 个字段——若不补字段定义则字段不存在、编译失败。

**机制解释**(根因分析,属 interpretation):rebase 冲突在此区域不是普通文本冲突,而是"字段集不同步"的暴露点——新框架类型定义(`F_type`)与 kernel 侧访问契约(`F_kernel`)必须一致,任何 kernel 直读的字段都必须在 `TILING_DATA_FIELD_DEF` 中显式声明。判别核心是 `F_head − F_type` 与 `F_kernel` 的交集,而非冲突标记本身;看到冲突就只取一侧(只保 `set_` 不补字段,或只保旧赋值回退重构)都会失败——必须两侧合并(保 `set_` + 新类型签名,补齐上游新增字段的 `TILING_DATA_FIELD_DEF` 与 `set_` 调用)。

## 预防

- tiling data 框架迁移分支 rebase 上游前,先 `git diff <upstream>...HEAD -- <旧 struct 头>` 查上游对旧 struct 的字段增删,把新增字段同步补到新 `TILING_DATA_FIELD_DEF` 类型 + 对应 `set_` 调用,再解决 .cpp 冲突;不要只取一侧。
- kernel 侧 `tiling_-><field>` 的字段集是新类型定义的契约下限:任何 kernel 直读的字段必须在 `TILING_DATA_FIELD_DEF` 中显式声明,否则编译失败。
- 在 CI 加一步"kernel `tiling_->` 访问字段 ⊆ 新类型 `TILING_DATA_FIELD_DEF` 字段"的静态比对,可在 rebase 后编译前拦截字段集失配。

## 验证信号

- `git status` 报 `rebase in progress` + 2 个 unmerged path,冲突在 `SetRunInfoTiling` / `SetRunBaseShapeInfoTiling` 区域(来源平台:未定位;证据等级:tool_observed)
- 解决冲突后 `grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" conv/conv3d_backprop_input_v2/` 无命中,冲突标记清零(来源平台:未定位;证据等级:tool_observed)
- `GIT_EDITOR=true git rebase --continue` 退出,`git status` 回到 clean working tree,被重放的 commit 重新应用为新 commit(来源平台:未定位;证据等级:tool_observed)
- arch35 kernel 通过 `tiling_->` 访问的字段全部在新类型 `Conv3DBackpropInputArch35TilingData` 中存在(来源平台:未定位;证据等级:tool_observed)

> 实际编译/上板信号未在 trace 内采集;上述为 git/rebase 状态 + 静态字段存在性验证,非编译通过证据。

## 失效边界

- 本卡针对 `TILING_DATA_DEF` 宏框架迁移 + rebase 上游新增字段的合并模式;非该框架(手写 struct、`GET_TILING_DATA_WITH_STRUCT` 旧路径)不适用。
- arch32 tiling data 不在本重构范围,仍用旧嵌套 struct(正常现象,非漏改)。
- 推断可迁移到其他用 `TILING_DATA_DEF` 框架的 arch 算子(未验证,推断可迁移)。
- 编译失败本身未被 trace 实测捕获(未跑编译);判别基于 git/rebase 状态 + 字段集静态比对。

## 坏实践

- 只取本分支侧(`set_` + 新类型)而忽略上游新增字段 → kernel `tiling_->khDilation` 编译失败。
- 只取 HEAD 侧(旧类型 + 直接赋值)→ 丢失整个框架迁移成果,回退重构。
- 把冲突区域的 rebase 冲突当普通文本冲突手工删标记了事,不做字段集比对 → 字段集失配被掩盖,编译期才暴露。

## 误诊

无(根因在 rebase 冲突阶段一次定位成功:字段集不同步,而非冲突标记本身或 kernel 计算链缺陷)。