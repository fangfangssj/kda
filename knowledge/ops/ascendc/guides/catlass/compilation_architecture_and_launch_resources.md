---
type: Guide
title: CATLASS 编译架构与启动资源合同
description: 区分编译宏、目标架构、链接组件和实际核资源。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/architecture/03-compilation.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/architecture/02-device-calling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/rules.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/troubleshooting.md
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# CATLASS 编译架构与启动资源合同

来源以CATLASS_ARCH=2201对应其910B/910_93候选、3510对应950候选；这是固定reference的配置映射，不能代替运行设备识别。--npu-arch=dav-2201是编译器目标，CATLASS_ARCH是头文件条件编译，两者含义不同且应与所用版本/设备保持一致。

来源工程使用CMake ASC语言、.asc或显式将.cpp标记LANGUAGE ASC以及C++17。include与runtime/ascendcl/profapi/mmpa/c_sec/error_manager/graph_base/tiling_api/register/platform/unified_dlog/dl/m等链接项需按实际使用符号及安装包确定，不是所有CATLASS算子都固定依赖全表。BUILD_CATLASS_MODULE=ON通常需辨别是CMake配置选项还是实际编译宏，不能把-DBUILD_CATLASS_MODULE=ON加入任意target_compile_options就视为启用相同功能。

共享POD tiling头与Host侧含框架注册宏的tiling头需要区分，Device不可因名字相同就包含所有Host依赖；并非任意自定义tiling.h都禁止。来源SKILL_KERNEL_DIR引用交付kernel与“禁止外部路径”存在工程语境差异，保留真实依赖和可重现构建关系，不迁移固定Skill目录规则。

aclGetDeviceCapability的设备信息枚举不能与aclCompileOpt枚举混用。Platform资源查询或runtime能力查询均需检查返回状态、0值、目标设备及AIC/AIV启动关系；数学任务数也不一定等于核资源数。来源某查询失败后改用PlatformManager是排障候选，不证明全版本同样报错。

aclrtGetHardwareSyncAddr获得地址后SetSyncBaseAddr只适用于依赖该协议的启动路径，地址宽度和有效生命周期需保留；不能要求所有CATLASS入口无条件安装硬同步地址。Kernel能编译不证明workspace、核型、事件或运行时初始化已正确。
