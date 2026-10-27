---
type: "Runbook"
title: "EZ3003 V2d/V2D 与 _camel_to_snake 数字拆分"
description: "AscendC 算子 OpType 含数字+字母（如 V2D）时，_camel_to_snake 把数字当分隔符拆出 _d，与测试框架 str.title() 的反向转换产生 V2d↔V2D 大小写错配，导致 GE 图编译报 No supported Ops kernel (EZ3003)。修法：目录与 OpType 对齐 _camel_to_snake 输出（apply_adagrad_v2_d / ApplyAdagradV2D），让两个转换算法同时闭环。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.quality.compatibility.version_migration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["str.capitalize"]
---

# EZ3003 V2d/V2D 与 _camel_to_snake 数字拆分

## 现象

测试人员用 aclnn_fuzz 框架（`xrunfk.py`）在 Ascend NPU 上跑 `ApplyAdagradV2d` 算子用例，命令形如 `python3 xrunfk.py npu_off ApplyAdagradV2d ApplyAdagradV2d_ApplyAdagradV2d_L0_001.cs`，框架内部走 GE 图模式（`geir/main --op_name ApplyAdagradV2d --device_id 0`）。算子图执行阶段看似正常完成，进入结果比较阶段时 golden 基准文件不存在、kernel_meta 目录里没有该算子的 kernel 文件，最终状态 `npuError / NPUFail / Failed`，RunGraph 返回 `4294967295`（即 -1）。核心报错原文：

```
Unsupported_Operator(EZ3003): No supported Ops kernel and engine are found for [ApplyAdagradV2d], optype [ApplyAdagradV2D].
Failed to select engine for [ApplyAdagradV2d][ApplyAdagradV2D].[FUNC:operator()][FILE:engine_place.cc]
```

报错里两个名字大小写不同：节点 type 是 `ApplyAdagradV2d`（小写 d，与 `REG_OP` 注册一致），而 GE 查找算子信息库时用的 optype 是 `ApplyAdagradV2D`（大写 D），信息库里只注册了小写 d 的版本，于是查不到 → 选 engine 失败 → 无 NPU 输出。

当时算子命名状态（改名前）：`REG_OP(ApplyAdagradV2d)` / `OP_ADD(ApplyAdagradV2d)` / `.ExtendCfgInfo("opFile.value", "apply_adagrad_v2d")` / `IMPL_OP_INFERSHAPE(ApplyAdagradV2d)` / `IMPL_OP_OPTILING(ApplyAdagradV2d)`，目录 `optim/apply_adagrad_v2d/`。

## 适用场景

- **工程形态**：算子仓开发（ops-nn / AscendC 框架），GE 图模式测试（aclnn_fuzz `xrunfk.py`、`geir/main` 等基于 GE 图编译的测试框架）。
- **API 路径**：算子注册链路 `REG_OP`(proto.h) → `OP_ADD`/`OpDef`(def.cpp) → `opFile.value` → `IMPL_OP_INFERSHAPE`/`IMPL_OP_OPTILING`；GE 算子信息库 `aic-ascend950-ops-info.json` 的 `op_type` 字段。
- **验证过的平台与 CANN 版本**：Ascend950（arch35）；CANN 9.0.0（GEIR example PASS）与 CANN 9.2.0（TTK kernel 208/210 PASS）。frontmatter `platforms` 仅含 `950`（trace 内实测平台）。
- **适用粒度**：`operator_family` —— 任何 CamelCase OpType 中含「数字+字母」段（如 `V2D`、`V2d`、`L2`、`Bf16`）的 AscendC 算子都可能踩此坑。
- **路线优先级**：`recommended_fastpath` —— 在算子命名设计阶段就让目录名 = `_camel_to_snake(OpType)` 输出、OpType = `str.title(目录名)` 输出，两算法同时闭环（详见 `## 解决配方`）。
- **不适用或未验证场景**：CANNDEV TBE 框架算子不受影响（tbe 文件名/kernel_name 手动指定，不经过 `_camel_to_snake` 自动转换）；TTK 框架（`_snake_to_pascal` 用 `part.capitalize()`）不受影响；a2/a3 平台未验证（推断可迁移，见 `## 失效边界`）。

## 根因

CANN AscendC 构建系统的 `_camel_to_snake` 函数把 CamelCase OpType 转 snake_case 目录名/kernel 函数名时，**数字会打断大写字母的连续性**：`ApplyAdagradV2D` → `apply_adagrad_v2_d`（数字 `2` 打断，`D` 被拆成 `_d`），而不是期望的 `apply_adagrad_v2d`。两者不一致 → kernel binary 编译失败。这是历史上 OpType 被迫改成小写 `d`（`ApplyAdagradV2d`）的直接原因。

而测试框架（aclnn_fuzz `xrunfk.py`）从上库代码拿到 snake_case 名（目录名 / `opFile.value` = `apply_adagrad_v2d`）后，用 Python `str.title().replace("_","")` 反推 CamelCase OpType。`str.title()` **把数字当单词分隔符**，数字后的小写字母被当作新单词首字母大写：`apply_adagrad_v2d` → `ApplyAdagradV2D`（大写 D）。这个大写 D 的 optype 拿去 GE 图里查算子信息库，但信息库里注册的是小写 d 的 `ApplyAdagradV2d` → 查不到 → EZ3003。

两个转换算法对「数字+字母」的处理正好相反，形成死结：
- AscendC `_camel_to_snake`：`V2D` → `v2_d`（数字打断，D 拆出）
- Python `str.title()`：`V2d` → `V2D`（数字后字母当新单词首字母大写）

用小写 d（`ApplyAdagradV2d` / `apply_adagrad_v2d`）只能让其中一方闭环：`_camel_to_snake(ApplyAdagradV2d)=apply_adagrad_v2d` ✅（目录对得上，kernel binary 能编译），但 `str.title('apply_adagrad_v2d')=ApplyAdagradV2D` ❌（GE 查不到）。反过来用大写 D（`ApplyAdagradV2D`）则 `str.title` 闭环但 `_camel_to_snake` 拆出 `apply_adagrad_v2_d` ≠ 目录名 `apply_adagrad_v2d` → kernel binary 编译失败。CANNDEV 官方用大写 D 不出问题，是因为 TBE 框架手动指定 tbe 文件名 `apply_adagradv2_d.py` 与 kernel_name `"apply_adagradv2_d"`，不依赖 `_camel_to_snake` 自动转换；AscendC 框架强制 OpType 经 `_camel_to_snake` 转换后必须与目录名一致，于是卡死在中间。

## 误诊

最初把大写 D 的来源归咎于 aclnn API 函数名 `aclnnApplyAdagradV2D`（大写 D），并提出“方案 2：确保测试框架从 op_info 配置读取 OpType、不从 aclnn API 名提取”。这个归因被用户一句话推翻：“aclnn 不是没有上库吗，为什么能读到”。核查事实：op_api 配置为 `ACLNNTYPE=aclnn_exclude`，源码不上 git、不随上库代码外传，测试人员根本拿不到 aclnn 头文件和 so。大写 D 不可能来自 aclnn。真正来源是测试框架自己对 snake_case 目录名做的 `str.title()` 反推。这条误诊的教训：归因前必须确认“大写 D 的物理来源在测试人员环境里到底存不存在”，不能只看本地算子仓里哪个名字带大写 D。

## 解决配方

核心思路：**顺着 `_camel_to_snake` 的拆分结果走，而不是逆着它**。把目录名改成 `_camel_to_snake(ApplyAdagradV2D)` 的输出 `apply_adagrad_v2_d`，OpType 恢复为 CANNDEV 官方大写 D `ApplyAdagradV2D`。这样两个转换算法同时闭环：

```
_camel_to_snake("ApplyAdagradV2D") = apply_adagrad_v2_d   ← 与目录名一致 ✅
str.title("apply_adagrad_v2_d")    = ApplyAdagradV2D      ← 与 OpType 一致 ✅
```

改动范围（21 个上库文件 rename + 1 个 `docs/zh/op_list.md` 修改，纯改名零逻辑改动）：

- 目录：`optim/apply_adagrad_v2d/` → `optim/apply_adagrad_v2_d/`（`git mv`）
- 11 个源文件改名（`apply_adagrad_v2d_*` → `apply_adagrad_v2_d_*`，含 proto.h / def.cpp / infershape.cpp / tiling_arch35.cpp / kernel .cpp/.h / tiling_data.h / tiling_key.h / examples / 2 个 UT）
- 3 类文件内符号全局替换：
  - OpType/类名/命名空间：`ApplyAdagradV2d` → `ApplyAdagradV2D`（~68 处，含 `NsApplyAdagradV2d` → `NsApplyAdagradV2D`、`ApplyAdagradV2dKernel` → `ApplyAdagradV2DKernel` 等）
  - snake_case：`apply_adagrad_v2d` → `apply_adagrad_v2_d`（~24 处）
  - 宏名/include guard：`APPLY_ADAGRAD_V2D` → `APPLY_ADAGRAD_V2_D`（~12 处）

before（关键注册点，改名前小写 d，逐行取自 trace 对改名前文件的 grep/read 输出）：

```cpp
// op_graph/apply_adagrad_v2d_proto.h
REG_OP(ApplyAdagradV2d)
OP_END_FACTORY_REG(ApplyAdagradV2d)

// op_host/apply_adagrad_v2d_def.cpp
class ApplyAdagradV2d : public OpDef
.ExtendCfgInfo("opFile.value", "apply_adagrad_v2d")
OP_ADD(ApplyAdagradV2d);

// op_host/apply_adagrad_v2d_infershape.cpp
IMPL_OP_INFERSHAPE(ApplyAdagradV2d)

// op_host/arch35/apply_adagrad_v2d_tiling_arch35.cpp
IMPL_OP_OPTILING(ApplyAdagradV2d)
```

after（关键注册点，改名后大写 D + v2_d 目录，逐行取自子代理 code review 对终版文件的 verbatim 引用，行号标注）：

```cpp
// op_graph/apply_adagrad_v2_d_proto.h  (proto.h:70, :80)
REG_OP(ApplyAdagradV2D)
OP_END_FACTORY_REG(ApplyAdagradV2D)

// op_host/apply_adagrad_v2_d_def.cpp  (def.cpp:33, :94, :98)
class ApplyAdagradV2D : public OpDef
.ExtendCfgInfo("opFile.value", "apply_adagrad_v2_d")
OP_ADD(ApplyAdagradV2D);

// op_host/apply_adagrad_v2_d_infershape.cpp  (infershape.cpp:81)
IMPL_OP_INFERSHAPE(ApplyAdagradV2D)

// op_host/arch35/apply_adagrad_v2_d_tiling_arch35.cpp  (tiling_arch35.cpp:309)
IMPL_OP_OPTILING(ApplyAdagradV2D)
```

> 注：上方仅展示改名触及的关键注册行（逐字取自 trace 的 grep 输出与子代理 code review 对终版文件的逐行确认）。`OpDef` 类体成员、`REG_OP` 块内部的算子 IO 声明等完整函数体未在 trace 的 file_read 事件中逐字恢复（markdown_transcript 适配器 redactions=674/773，多数文件读取结果被脱敏截断），故不以内联 `// ...` 形式补全；完整函数体恢复情况。

诊断命令（验证 `str.title()` 是大写 D 的来源，verbatim 取自 trace 的 bash 命令）：

```bash
echo "=== Python str.title() 行为验证 ==="
python3 -c "
name = 'apply_adagrad_v2d'
print(f'snake_case: {name}')
print(f'str.title(): {name.title()}')
print(f'title+replace: {name.title().replace(\"_\", \"\")}')
print()
# 对比其他算子
for n in ['apply_adagrad_dad', 'apply_add_sign', 'apply_keras_momentum', 'inplace_apply_adagrad_da']:
    print(f'{n} -> {n.title().replace(\"_\", \"\")}')
"
# 期望输出：
#   str.title(): Applyadagrad V2D
#   title+replace: ApplyAdagradV2D     ← 大写 D（数字 2 后 d 被当新单词首字母大写）
#   apply_adagrad_dad -> ApplyAdagradDad        ← 不含数字+字母，不受影响
#   inplace_apply_adagrad_da -> InplaceApplyAdagradDa
```

对照 TTK 框架（`ttk/test_spec/loader.py:18` 的 `_snake_to_pascal` 用 `part.capitalize()`）：

```python
# TTK 的 _snake_to_pascal: part.capitalize()  →  apply_adagrad_v2d → ApplyAdagradV2d（小写 d，正确）
# Python str.title():                          →  apply_adagrad_v2d → ApplyAdagradV2D（大写 D，错误）
# 二者差异：capitalize() 只大写每段首字母；title() 额外把数字后的小写字母也大写
```

非上库文件需同步改名（虽不进 git，但本地编译/TTK aclnn 模式需要）：`op_api/apply_adagrad_v2d.cpp` 的 `OP_TYPE_REGISTER(ApplyAdagradV2d)` 与 `ADD_TO_LAUNCHER_LIST_AICORE(ApplyAdagradV2d, ...)` 同步为大写 D；`tests/ttk/apply_adagrad_v2d_ttk_low.csv` 的 `op_name` 列 `apply_adagrad_v2d` → `apply_adagrad_v2_d`。归档仓（`LongTailInfo_dad_v2d`）也要同步：CSV `op_name`、Golden 函数名 `compute_golden_apply_adagrad_v2d` → `compute_golden_apply_adagrad_v2_d`、spec.yaml 算子名。

## 验证信号

本卡验证来自一次 Ascend950 上的 V2D 算子改名 + 重验开发会话。改名提交 commit message verbatim 记录了两个转换算法的闭环验证（`_camel_to_snake(ApplyAdagradV2D) = apply_adagrad_v2_d` ← 与目录名一致；`str.title('apply_adagrad_v2_d') = ApplyAdagradV2D` ← 与 OpType 一致）。改名后实测：

- 编译 SUCCESS，kernel binary 文件名 `ApplyAdagradV2D_*.o`（大写 D），binary config simplifiedKey `ApplyAdagradV2D/...`，op_info 注册名 `ApplyAdagradV2D`，proto.h 文件名 `apply_adagrad_v2_d_proto.h`（v2_d）（来源平台：950；证据等级: tool_observed）。
- GEIR example：`Run graph success` + 2 输出 (DT_FLOAT) + `Finalize success`（CANN 9.0.0，来源平台：950；证据等级: tool_observed）。
- TTK kernel 测试（CANN 9.2.0）：208/210 PASS。2 FAIL 均为大 Tensor（1000×1000、100×100×100）accum 输出 float32 累加精度限制，与改名前一致、非回归（来源平台：950；证据等级: tool_observed）。
- git rename 检测正常（相似度 77%~100%），21 files rename + 1 个 `op_list.md` 修改，无非上库文件误提交（来源平台：950；证据等级: tool_observed）。

## 性能影响

纯改名，零逻辑改动，无性能影响。

## 设计决策

- **为何选「顺着 `_camel_to_snake` 走」而非「逆着它」**：方案 A（改 OpType 避开数字+小写字母，如 `ApplyAdagradV2`、`ApplyAdagradTwo`）能避开 `str.title()` 陷阱，但会偏离 CANNDEV 官方名 `ApplyAdagradV2D`，丢失对标关系。方案 B（让测试框架改用 `capitalize()`）不在算子侧控制范围。方案 C（向 ops-nn 仓反馈 `_camel_to_snake` 框架 bug，让它对「数字+大写字母」不拆分）是根治但跨仓、周期长。最终选「让目录名匹配 `_camel_to_snake` 输出 + OpType 恢复 CANNDEV 大写 D」，因为这是唯一一个在算子仓控制范围内、且让两个转换算法同时闭环的方案，同时保住与 CANNDEV 官方名的对标。
- **放弃的候选**：用小写 d（`ApplyAdagradV2d`）是历史妥协——当时只验证了 `_camel_to_snake` 闭环（kernel binary 能编译），漏检了 `str.title()` 反推闭环，导致 GE 图模式测试框架查不到算子。本次改名补上了被漏掉的第二条闭环验证。
- **路线归属**：`recommended_fastpath` —— 命名设计阶段就让两算法同时闭环，是算子仓侧能控制、零逻辑改动、与 CANNDEV 对标的最佳路径。

## 预防

- 任何 CamelCase OpType 含「数字+字母」段（`V2D`/`V2d`/`L2`/`Bf16`/`F32` 等）的 AscendC 算子，命名设计时必须同时验证两条闭环：`_camel_to_snake(OpType)` == 目录名/snake_case；`str.title(目录名).replace("_","")` == OpType。任一不闭环都会在某个测试框架下暴露 EZ3003。
- 不要用「数字后小写字母」（`V2d`）这种 `_camel_to_snake` 与 `str.title()` 行为相反的写法；要么用「数字后大写字母 + 目录名带 `_d`」（`V2D` / `v2_d`，本卡方案），要么彻底避开数字+字母段。
- 归因报错时先确认「大写 D 的物理来源在测试人员环境里到底存不存在」，避免把本地有但测试人员没有的 aclnn API 名误当根因。

## 坏实践

- 用小写 d（`ApplyAdagradV2d`）作为永久妥协——只让 `_camel_to_snake` 闭环，漏检 `str.title()` 反推闭环，把问题推迟到 GE 图模式测试框架才暴露。
- 归因前不核实测试人员环境实际能拿到哪些文件（误把 `aclnn_exclude` 不上库的 aclnn API 名当大写 D 来源）。

## 失效边界

- 仅在 Ascend950（arch35）实测验证；CANN 9.0.0（GEIR example PASS）与 CANN 9.2.0（TTK kernel 208/210 PASS）。a2/a3 平台未验证，推断可迁移（`_camel_to_snake` 与 `str.title()` 的行为是 Python/CANN 构建系统层面、与芯片型号无关），但不得据此处写入 `platforms`。
- `str.title()` 陷阱是测试框架实现相关的：本卡验证针对 aclnn_fuzz `xrunfk.py`（用 `str.title()`，受影响）与 TTK（用 `_snake_to_pascal`/`capitalize()`，不受影响）。其他测试框架若用 `capitalize()` 或直接读 op_info 配置则不受影响。若测试框架升级改用 `capitalize()`，本卡描述的 EZ3003 不会再发生，但「两算法同时闭环」的命名设计仍是推荐做法（防御未来框架变化）。
- GEIR example 在 CANN 9.2.0 下因 build.sh 链接缺 `-lge_common -lge_common_base -lacl_rt` + `--start-group/--end-group` 未能补跑（环境阻塞，与改名无关；CANN 9.0.0 下 GEIR 已 PASS）。
