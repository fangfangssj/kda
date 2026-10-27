---
type: Guide
title: Softmax 模板ABI与独立头文件依赖
description: 提取可复用集成知识，保留示例缺失依赖与构建边界。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/usage_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_base.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_recompute.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Softmax 模板ABI与独立头文件依赖

六套POD须Host/Device同字段顺序、宽度和pack；CPU编译此源码得到SmallR64、AR48、ARRecompute64、ARA160、ARARecompute128、ARAOnline88字节。#pragma pack(1)可消除padding但不保证架构访问、端序和传输布局全部兼容；末尾#pragma pack()复位默认而非恢复调用方原pack，独立封装宜push/pop并验证sizeof/offsetof。

AR构造传TPipe*、Init传对应td；ARA构造传td、Init传TPipe*，且类在SoftmaxV2Ops命名空间。默认ARA构造不初始化td，调用Init前必须赋合法数据。公共头未自动提供全部TilingData，入口应显式include匹配声明；AR Recompute使用DOUBLE_BUFFER而其本文件和base未定义这一名字，独立包含仍需明确常量来源，不能凭指南说独立就假设无依赖。把另一模板include来偶然补名不是稳定模块合同。

目录dav310在这些文档实际指目标dav-3510及MicroAPI/VF支持，不自动代表历史310芯片。模板后缀和include路径要在构建中形成真实可解析文件；改名是材料使用流程，知识只保留依赖、ABI和编译单元约束。op_kernel/platform_util等还可能属于配套工程头，不能把缺头全部归因SDK版本。

来源介绍ASC同翻译单元与ascendc_library/kernellaunch两种方式，device编译/链接架构应一致，可重定位device代码与device link匹配。__ASC_NPU_HOST__守卫是该参考针对host pass Reg类型可见性的适配，不宜盲目全局开启或当全部SDK必须宏。MicroAPI找不到也可能SDK/arch/include不匹配，未定义符号也不止一个原因。

compile_commands需在CMake配置时设置CMAKE_EXPORT_COMPILE_COMMANDS，已有构建目录要重配；单纯后续make不会生成。生成器支持、编译器选项及VF fusion实际效果以工程版本验证。当前仅CPU Host头通过本次算术/sizeof检查，未完成这些Kernel的SDK编译或NPU运行。
