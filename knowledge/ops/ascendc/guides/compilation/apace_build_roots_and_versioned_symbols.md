---
type: Guide
title: APACE 构建根目录与版本化符号
description: APACE 构建根目录与版本化符号的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/host-and-testing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/operator-anatomy.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE 构建根目录与版本化符号

APACE、Blaze、tensor_api、AscendC基础/高阶头、tiling和HCCL库必须来自相容安装或锁定源码组合。来源独立工程示例使用OP_KERNEL_ROOT解析apace/...前缀，以及tensor_api/include与tensor_api两层include根；仓内ST另由ops-tensor FetchContent提供Blaze。它们是不同打包形态，不以单一路径是否存在判所有版本缺组件。

给include增加apace前缀只统一根，不能自动把apace/block/aiv_comm改为apace/core/aiv_comm，也不能把_udma_impl符号改成_urma_impl；后缀目录和真实符号仍须匹配版本。CMakeCache可保留旧bisheng，CPATH也可引入另一套头；诊断需看实际编译命令、解析头及加载libhccl路径，局部清除旧构建缓存或显式依赖，不修改全局环境。

来源dav-3510独立工程用-xasc、--npu-arch=dav-3510、ASC_DEVKIT_MAJOR=9及hccl_fwk的--no-as-needed链接处理。AG原ST未带该宏，不能将某份CMake全复制为所有环境强制配方。HcclChannelAcquire undefined可由库版本、缺库或链接保留/顺序导致；工具包找不到也可能配置路径错，不唯一归因安装损坏。

共享namespace中冲突的LayoutA/FragTensorA别名会污染模板选择，可用类作用域、唯一命名或独立namespace隔离；共享namespace本身不必然非法。普通编译器不能仅凭include处理设备关键字或<<<>>>。来源工程审批、零复制门禁与命名规范未迁为技术强制规则。
