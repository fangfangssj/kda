---
type: "Runbook"
title: op_def 属性链写 Required() 编译失败
description: "ops::OpAttrDef 没有 Required() 设置方法，链式 .Required().ListInt() 报 has no member named 'Required'；必选/可选由 AttrType(Option) 设置、IsRequired() 仅查询，不调用 AttrType 时属性默认即必选。"
tags: [task.implementation, topic.compilation, topic.implementation.api_layer]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# op_def 属性链写 Required() 编译失败

## 现象

为 BenchNoHarnessDepthwiseConv2d（Ascend950PR，CANN 9.2.0）编写算子定义 `op_host/bench_no_harness_depthwise_conv_2d_def.cpp`。按图模式 REG_OP 的直觉（`REQUIRED_ATTR(kernelSize, ListInt)`）给 op_def 的属性链式调用加了 `Required()`：

```cpp
this->Attr("kernelSize").Required().ListInt();
```

cmake configure 阶段同一批报出 5 条同构错误：

```
error: 'class ops::OpAttrDef' has no member named 'Required'; did you mean 'IsRequired'?
    64 |         this->Attr("kernelSize").Required().ListInt();
    65 |         this->Attr("stride").Required().ListInt();
    66 |         this->Attr("padding").Required().ListInt();
    67 |         this->Attr("dilation").Required().ListInt();
    68 |         this->Attr("groups").Required().Int();
```

## 适用场景

- 工程形态：registry-invoke 算子的 op_def 定义文件（`ops::OpDef` 子类，`OP_ADD` 注册）。
- API 路径：`register/asc/op_def.h`——`OpDef::Attr(const char*)`（492 行）返回 `OpAttrDef&`；`OpAttrDef::AttrType(Option)`（331 行）与 `IsRequired()`（351 行）；`enum Option { IGNORE=0, OPTIONAL=1, REQUIRED=2, DYNAMIC=3, VIRTUAL=4 }`（59 行）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（与 frontmatter `platforms` 一致；验证等级为编译级）。
- 适用粒度：platform_wide——op_def API 契约，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：图模式 proto.h 的 `REG_OP/REQUIRED_ATTR` 宏体系（那是另一套写法，本卡不涉及）；显式 `AttrType(REQUIRED)` 链式写法未实测（只实测了"移除即默认必选"）。

## 根因

`ops::OpAttrDef` 没有 `Required()` 设置方法——必选/可选由 `AttrType(Option)` 设置（Option 枚举含 REQUIRED=2），`IsRequired()` 只是查询；且不调用 `AttrType` 时属性默认即为必选，链式 `.Required()` 是不存在的 API。

## 解决配方

修复前（照 REG_OP 宏直觉写 `.Required()`）：

```cpp
// op_host/bench_no_harness_depthwise_conv_2d_def.cpp（修复前）
        // 属性（顺序：kernelSize, stride, padding, dilation, groups）
        this->Attr("kernelSize").Required().ListInt();
        this->Attr("stride").Required().ListInt();
        this->Attr("padding").Required().ListInt();
        this->Attr("dilation").Required().ListInt();
        this->Attr("groups").Required().Int();
```

修复后（去掉 `.Required()`，属性默认必选）：

```cpp
// op_host/bench_no_harness_depthwise_conv_2d_def.cpp（修复后）
        // 属性（顺序：kernelSize, stride, padding, dilation, groups）
        this->Attr("kernelSize").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();
        this->Attr("groups").Int();
```

判别方法：`grep -n "Required\|ListInt\|enum Option\|AttrType" $ASCEND_HOME_PATH/x86_64-linux/include/register/asc/op_def.h`——确认方法集只有 `AttrType(Option)`（设置）与 `IsRequired()`（查询）。op_def 属性需要显式改可选时用 `AttrType(OPTIONAL)`，需要必选时省略调用即可。

## 设计决策

- 为什么选择移除而不是 `AttrType(REQUIRED)` 显式链：移除是零风险最小改动（add_example 参考无属性可用）；`AttrType(REQUIRED)` 语义等价但本会话未实测，不写未验证的姿势进配方。

## 系统信息

- 硬件/软件：x86_64 主机，CANN 9.2.0，目标 ascend950。
- 相关文件：`op_host/bench_no_harness_depthwise_conv_2d_def.cpp`；`include/register/asc/op_def.h`。
- 复现命令：`cd build && rm -f CMakeCache.txt && cmake -DASCEND_COMPUTE_UNIT=ascend950 ..`。

## 预防

- op_def 属性定义不要照抄 REG_OP/REQUIRED_ATTR 宏体系的动词；两套 API 的必选表达不同。
- 对 `OpAttrDef` 链式调用拿不准时先 grep op_def.h 的方法清单。

## 验证信号

以下信号来自该算子 op_def 首轮编译联调：修复前 5 条同构错误稳定复现，修复后消失。

- 修复前：cmake configure 报 5 条 `error: 'class ops::OpAttrDef' has no member named 'Required'; did you mean 'IsRequired'?`（def.cpp:64-68），`compile ascend all ops failed`（来源平台：950；证据等级：tool_observed）。
- 修复后：cmake configure 通过（`-- Configuring done`），build 推进到 op_api/kernel 阶段的独立问题；全链路修复后全量编译产出安装包（来源平台：950；证据等级：tool_observed）。
- 头文件事实锚点：`op_def.h:59 enum Option { IGNORE=0, OPTIONAL=1, REQUIRED=2, DYNAMIC=3, VIRTUAL=4 }`、`:331 OpAttrDef &AttrType(Option attr_type)`、`:351 bool IsRequired(void)`、`:492 OpAttrDef &Attr(const char *name)`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 以 CANN 9.2.0 op_def.h 为准；若后续版本补上 `Required()` 便利方法，本卡失效（未验证）。
- "省略 AttrType 即默认必选"是本会话实测行为（编译+后续整包构建通过），未从源码逐行验证默认值初始化。
- 其他 CANN 版本的 op_def API 面未逐一核对（未验证，推断可迁移）。

**未决项**：

- 无。
## 坏实践

- 把图模式 `REQUIRED_ATTR` 宏的写法直觉带进 op_def 链式 API。
- 看到 "did you mean 'IsRequired'" 就真去调 `IsRequired()`——那是查询不是设置。

## 误诊

- 无（编译器建议直接指向查询方法，读头文件即定位真 API）。
