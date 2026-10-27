---
type: Guide
title: Blaze-SHMEM 样例的构建依赖身份
description: Blaze-SHMEM 样例的构建依赖身份的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/CMakeLists.txt
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/cmake/ascend.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/cmake/shmem.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/cmake/tensor_api.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/review-checklist.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 样例的构建依赖身份

此工程由bisheng -xasc同时处理Host/Device，target硬编码--npu-arch=dav-3510；NPU_ARCH校验虽接受dav-2201并warning，实际编译标志仍3510，不能当2201支持证据。SHMEM子构建也固定SOC_TYPE=Ascend950。ascend.cmake在project前选工具链，已有CMAKE_C/CXX_COMPILER不强制覆盖，故配置值/compile_commands/真实编译器须一致。SYSTEM_PREFIX探测CPU，但主CMake额外include硬编码x86_64-linux，不能据前者宣称支持任意Host架构。

SHMEM URL/tag缓存变量默认cann/shmem v1.5.0，tensor_api固定asc-devkit commit ad3d3bf04dddfb94370c534c39bdd305d8e38d88。两者clone命令在custom target，实际于build依赖执行，不是文档声称首次configure立即clone。已有CMakeLists或tensor.h就跳过clone/checkout，文件存在不证明版本相符；SHMEM tag也不同于不可变commit校验。

Blaze从toolkit opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common取，tensor_api从独立clone取；APACE来自ops-tensor的组织不能直接套到这里。SHMEM interface target传include、libdir、shmem/shmem_utils及两个bootstrap动态库，rpath指向install/shmem/lib，构建目标成功仍需核对运行时实际加载和ABI匹配。

cmake文件存在、头可找到、模板可实例化、链接成功和多卡运行分别是证据。源码-w隐藏告警不证明兼容；未知API可能来自版本/header/namespace/编译模式，不能所有“未定义”都归因为少link。REUSE/禁止修改依赖等是Skill工作约定，知识库保留实际依赖关系及升级风险。
