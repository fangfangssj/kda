---
type: "Runbook"
title: "专用 kernel 路径输出不变时先核对共享库是否真的包含目标符号"
description: "Host 源码已有专用 launch 分支但 targeted 输出仍保持输入值时，用 nm 同时核对共享库专用符号与通用实例，先判别目标 kernel 是否被产入二进制。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.implementation.debugging"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["a3"]
aliases: ["shared_library"]
---

# 专用 kernel 路径输出不变时先核对共享库是否真的包含目标符号

## 现象

scatter 的 fp16 accumulate 专用路径没有运行时错误，但多组参数化 dtype/layout/shape probe 均报告 unchanged=true、exact=false；另一个边界配置曾出现不同输出。此时 Host 源码文本里已经能看到 launch_acc 分支，单看源码无法证明共享库内真的存在目标专用 kernel。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`nm`、`scatter_fp16acc_kernel`、`scatter_kernel`、`launch_acc` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：探针观测与本卡判别条件不一致或仍含混时停止套用结论，转向新的定向探针。

## 判别方法

### trigger

targeted 测试输出显示多组参数化配置的 acc 结果与初始 data 完全相同，且与 golden 不精确匹配。

### probe

在实际加载的共享库上执行完整探针，同时保留 Host 分支的文本对照：

```bash
nm build/libop_scatter.so 2>/dev/null |
  grep -i "scatter_fp16acc\|scatter_kernel" |
  head
grep -n "fp16 reduce\|launch_acc\|acc.to\|dataC.to" scatter.asc |
  head
```

### observation

本次 nm 输出列出十个 scatter_kernel 模板实例，没有任何 scatter_fp16acc 符号；紧随其后的 source grep 则命中 launch_acc 定义和 fp16 reduce 分支。

### interpretation

当 source 分支存在而目标专用 symbol 不在实际共享库中时，这个二进制至少没有提供按该名称期望的 specialization。应先追查实例化、导出或实际加载的 artifact，再讨论 stream 同步；源码中有 launch 分支本身不能证明该专用 kernel 已进入当前 .so。

## 解释

这个 probe 建立的是 source-present / artifact-absent 的判别边界：源码分支存在只能证明编译输入含候选路径，实际加载制品缺目标符号则说明该名称对应的 specialization 不能由当前 `.so` 直接调用。它不能进一步区分“未实例化、未导出、链接裁剪、符号改名”或“检查了错误制品”；这些原因必须用编译产物和实际加载路径继续拆分，不能把缺符号直接写成唯一根因。

## 验证信号

以下信号来自本次开发会话的定向运行输出，以及对实际共享库执行的 `nm` 与源码分支对照。

- 多数参数化 probe 报告 `unchanged(=data)=True` 且 `exact=False`，另一个对照报告 `exact=True`（来源平台：a3；证据等级: tool_observed）。
- `nm` 输出只列出十个通用 `scatter_kernel` 模板实例，没有 `scatter_fp16acc`；同次源码 grep 命中 `launch_acc` 定义与 fp16 reduce 分支（来源平台：a3；证据等级: tool_observed）。

## 判定边界

- nm 可能显示 mangled 名称，probe 必须同时搜索稳定的专用 stem 和通用 kernel stem。
- 若目标采用匿名符号、LTO 内联或不同导出命名，需要换成目标工具链可观测的符号/反汇编探针。
- 此规则只负责判别二进制存在性，不声称缺符号的唯一原因，也不等价于最终修复。

## 失效边界

如果共享库并非运行时实际加载的 artifact，或符号被完全内部化，先锁定实际加载路径再解释 probe。