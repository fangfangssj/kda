---
type: "Runbook"
title: IR 属性名误改判别：IR 名与 aclnn ABI 名两层对照
description: GEIR/V2 改造后用 git diff 圈改文件、WORKFLOW_PROVENANCE 的 attr_abi_name 约定仲裁、attr index 访问假阴性排查，判别 IR 属性名是否被误改为 aclnn ABI 名。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# IR 属性名误改判别：IR 名与 aclnn ABI 名两层对照


## 现象

在一个 AscendC 自定义算子仓（对标 `aten::_thnn_fused_gru_cell_backward` 的 GRU Cell 反向算子，aclnn 接入）完成 GEIR/V2 图模式改造（V2 InferShape/InferDataType 校验升级 + `IMPL_OP` V2 注册）之后，执行误改排查任务：怀疑注册代码（`REG_OP`/`OpDef`）的属性名被顺手改掉（特别是 `has_bias` 相关）。

`git diff HEAD --name-only` 列出 4 个被改文件；逐文件看 diff 后发现：`op_graph/<op>_proto.h` 的 `REG_OP` 块与 `op_host/<op>_def.cpp` 的 `OpDef` 构造函数中，IR 属性名 `has_bias` 被改成了 camelCase 的 `hasBias`——而这两个文件自己的头部注释仍写着 `attr has_bias(Bool, 默认 false)`，代码与自身声明直接矛盾。全工程 grep 又发现 `hasBias` 本身大量合法存在（aclnn ABI 签名参数、gtest 变量、CSV 键），不能见 camelCase 就定罪，必须分层判别。

## 适用场景

- **工程形态**：aclnn 算子仓开发（ops-registry-invoke 流程）注册层核对；GEIR/V2/接口翻新类任务完成后的误改排查（diagnosis 任务）。
- **API 路径**：`REG_OP(...).ATTR(...)`（`graph/operator_reg.h`）、`OpDef` 构造 `this->Attr(...)`（`register/op_def_registry.h`）、`IMPL_OP` V2 注册（`register/op_impl_registry.h`）、aclnn ABI 签名、gert attrs 按 index 访问（`GetAttrPointer<bool>(0)` / `GetBool(0)`）。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册平台 950）；CANN 版本本轨迹未记录。
- **适用粒度**：operator_family（自定义算子 IR 注册/属性契约层；机制跨算子成立，仅本算子实测）。
- **路线优先级**：recommended_fastpath（diff 圈范围 + 约定记录仲裁 + index 访问排查的判别链）。
- **明确不适用/未验证场景**：无属性算子；IR 属性名与 aclnn 参数名恰好同名的算子（无两层区分）；按名读取 IR 属性的框架路径未排查；CSV attributes 键与注册名的实际匹配行为未运行验证。

## 判别方法

**触发**：GEIR/V2/接口类任务后，注册代码 diff 中出现属性名变更；或 grep 发现 camelCase 属性名出现在 `.ATTR(`/`.Attr(` 注册行。

**探针（可执行命令，verbatim）**：

```bash
# 1. 圈定改动文件（只看注册相关目录）
git diff HEAD --name-only -- operators/<pkg>/op_host/ operators/<pkg>/op_graph/ operators/<pkg>/op_kernel/

## 2. 逐文件看变更（重点 .ATTR / .Attr / REG_OP / OP_ADD 行）
git diff HEAD -- operators/<pkg>/op_graph/<op>_proto.h
git diff HEAD -- operators/<pkg>/op_host/<op>_def.cpp

## 3. 双形态 grep 建立分布（合法名与误改名同形，必须看出现在哪一层）
grep -rn "has_bias" operators/<pkg>/
grep -rn "hasBias" operators/<pkg>/

## 4. 约定记录仲裁（两层命名的文档化事实源）
grep -n "attr_abi_name\|has_bias\|hasBias" operators/<pkg>/tests/blackbox/WORKFLOW_PROVENANCE.json
```

**原始观测（verbatim）**：

- proto.h diff（IR 注册行被改名）：

```diff
-    .ATTR(has_bias, Bool, false)
+    .ATTR(hasBias, Bool, false)
```

- def.cpp diff（OpDef 注册行被改名）：

```diff
-        this->Attr("has_bias").AttrType(OPTIONAL).Bool(false);
+        this->Attr("hasBias").AttrType(OPTIONAL).Bool(false);
```

- 约定记录（`tests/blackbox/WORKFLOW_PROVENANCE.json`）：

```json
"aclnn_abi": "aclnnThnnFusedGruCellGradGetWorkspaceSize(gradHy, storage, hasBias, gradInputGatesOut, gradHiddenGatesOut, gradHxOut, gradInputBiasOut, gradHiddenBiasOut, workspaceSize, executor)（REQUIREMENTS §8.2.1，与部署头文件一致）",
"attr_abi_name": "hasBias（IR 属性名 has_bias；CSV attributes 键使用 aclnn 头文件 ABI 参数名，值域 {true, false} 含 proto 默认值 false，不生成缺省态）"
```

- attr 按 index 访问的代码（改名不直接崩溃的原因）：

```cpp
// op_host/arch35/<op>_tiling_arch35.cpp:167-168
const bool* hasBiasPtr = (attrs != nullptr) ? attrs->GetAttrPointer<bool>(0) : nullptr;
ctx.isBias = ((hasBiasPtr != nullptr) && *hasBiasPtr) ? 1 : 0;
```

```cpp
// tests/tiling/test_tiling_gtest.cpp:545（按注册序 index 追加，属性名不参与）
builder.AppendAttr(c.hasBias);
```

**解释**：CANN 自定义算子存在两层独立命名——**IR 属性名**（`REG_OP` `.ATTR()` / `OpDef` `.Attr()` 注册，snake_case `has_bias`，spec/proto.md 契约）与 **aclnn ABI 参数名**（camelCase `hasBias`，部署头文件签名）。GEIR 任务把后者误用到了前者。判别要点：①约定记录 `attr_abi_name` 明确两层名字；②同文件头注释仍写 `attr has_bias`（改名未同步，内部自相矛盾）；③tiling/InferShape/gtest 走 attr **index**（`GetAttrPointer<bool>(0)`/`GetBool(0)`/`AppendAttr`）访问属性，不按名访问——改名不会让这些路径立即崩溃，属**假阴性陷阱**，不能以「没崩」判定改名无害；④GEIR 任务边界是 V2 路由注册与校验，改既有 `REG_OP`/`OpDef` 属性名属 ABI 级变更，越界。结论：proto.h:27 与 def.cpp:59 两处判为误改需回退；infershape.cpp（桩→V2 全量校验）与 graph_infer.cpp（`op_def_registry.h`→`op_impl_registry.h` + `IMPL_OP` V2 注册）属 GEIR V2 合法工作，保留。

## 设计决策

- **probe 顺序**：先 `--name-only` diff 圈范围（避免全工程大海捞针）→ 逐文件 diff 定位变更行 → 双形态 grep 建立两层分布 → 约定记录仲裁（`attr_abi_name` 是文档化事实源）→ index 访问排查（解释为何改名未立即崩溃）。
- **放弃的备选**：只 grep `hasBias` 就下结论——100+ 匹配中 aclnn ABI 层合法名（ABI 签名参数、gtest 变量 `c.hasBias`、部分 CSV 键）与 IR 层误改名同形，无法区分；必须按「出现在哪一层」判别。
- **当前方案档位**：recommended_fastpath。

## 系统信息

- 工具链：git diff / grep / 文件读取（无构建、无运行——纯静态诊断任务）。
- 目标平台：Ascend950PR（结果产物 CSV 记录 soc=Ascend950PR_9579）。
- 相关文件：`op_graph/<op>_proto.h`、`op_host/<op>_def.cpp`（误改点）；`op_host/<op>_infershape.cpp`、`op_graph/<op>_graph_infer.cpp`（V2 合法改动）；`op_host/arch35/<op>_tiling_arch35.cpp`（attr index 访问）；`tests/blackbox/WORKFLOW_PROVENANCE.json`（约定记录）；`tests/tiling/test_tiling_gtest.cpp`；`tests/perf/*.csv`。

## 完整代码

完整诊断命令序列见「判别方法·探针」四步（verbatim，可直接执行；`<pkg>`/`<op>` 替换为目标算子包名/算子名）。

## 预防

- GEIR/V2/接口类任务提交前，diff 检查 `.ATTR(`/`.Attr(` 行是否被重命名；属性名变更属 ABI 级变更，须有显式裁定记录。
- `WORKFLOW_PROVENANCE.json` 的 `attr_abi_name` 字段是两层命名的仲裁源；发现 camelCase 出现在 `REG_OP`/`OpDef` 注册行时先查约定记录再定性。
- 排查「改名是否有害」时先确认访问方式：index 访问路径不感知改名（假阴性），不能作为无害证据。

## 验证信号

本信号来自该算子仓 GEIR 改造后误改排查诊断任务的实测（静态诊断，逐条如下）：

- `git diff HEAD --name-only`（op_host/ + op_graph/ + op_kernel/ 范围）列出 4 个被改文件（来源平台：950；证据等级：tool_observed）
- proto.h diff 原文：`-    .ATTR(has_bias, Bool, false)` → `+    .ATTR(hasBias, Bool, false)`（git index 99f6c72f..f8e71714）（来源平台：950；证据等级：tool_observed）
- def.cpp diff 原文：`-        this->Attr("has_bias").AttrType(OPTIONAL).Bool(false);` → `+        this->Attr("hasBias").AttrType(OPTIONAL).Bool(false);`（git index 1f9beaae..e2e26819）（来源平台：950；证据等级：tool_observed）
- `WORKFLOW_PROVENANCE.json:33-34` 约定记录原文：`aclnn_abi` 签名参数名 `hasBias`；`attr_abi_name` 注明「IR 属性名 has_bias」（来源平台：950；证据等级：tool_observed）
- grep `hasBias` 命中：`tiling_arch35.cpp:167-168` `attrs->GetAttrPointer<bool>(0)`、`def.cpp:59`、`proto.h:27`、`test_tiling_gtest.cpp:545` `builder.AppendAttr(c.hasBias)`、infershape V2 `attrs->GetBool(0)`（来源平台：950；证据等级：tool_observed）
- grep `has_bias` 命中 100+ 处：`proto.h:7`/`def.cpp:7` 头注释、tiling/empty kernel 注释、`test_tiling_gtest.cpp:544` 注释、perf CSV 文件等（来源平台：950；证据等级：tool_observed）

## 失效边界

- 「index 访问不感知改名」的结论仅覆盖 gert attrs 按 index 访问路径（tiling `GetAttrPointer<bool>(0)`、InferShape `GetBool(0)`、gtest `AppendAttr`）；是否存在按名读取 IR 属性的框架路径未排查。
- CSV attributes 键形态混用（perf 类 CSV 命中 `has_bias`、result_smoke.csv 用 `{'hasBias': False}`）与约定记录「CSV 键使用 aclnn ABI 参数名」存在张力——实际键匹配行为（改名后哪些用例会失败）未运行验证，本卡只确认约定记录与 diff 事实。
- 两层命名约定由本工程的 `WORKFLOW_PROVENANCE.json` 文档化；其他工程若无此记录文件，判别需回退到 spec.yaml/proto.md 契约。
- 其他平台未验证（IR 注册契约机制与芯片平台无关，未验证，推断可迁移）。

## 坏实践

- 只 grep camelCase 属性名、看到大量匹配就判定「本来就有 hasBias」——aclnn ABI 层合法名与 IR 层误改名同形，必须分层判别。
- 以「tiling 没崩/测试没挂」判定改名无害——tiling 与 gtest 按 index 访问属性，不感知改名，属假阴性证据。
- 把 V2 InferShape/InferDataType 的合法改动（校验升级、`IMPL_OP` 注册）一并判为误改回退——GEIR 任务边界内的正当工作，回退会破坏 V2 路由。

## 误诊

- 本诊断未走弯路；登记三类常见误判供对照：见「坏实践」前三条（同形名误判、假阴性无害判定、V2 合法改动误伤）。

## 未决事项

- CSV attributes 键与注册名的实际匹配行为未运行验证（可在受控环境重演改名状态跑用例确认失败面）。
- 按名读取 IR 属性的路径（框架侧按名 GetAttr）是否存在未排查——存在则改名会直接崩溃，判别结论需补充该分支。
