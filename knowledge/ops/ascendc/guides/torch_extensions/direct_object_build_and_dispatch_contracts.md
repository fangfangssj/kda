---
type: Guide
title: 直调对象库构建与 PyTorch 分发合同
description: 补充双编译链、ABI3 范围、Meta 分发与运行产物身份，限定骨架示例的可执行程度。
tags:
- task.implementation
- topic.compilation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-build-guide/references/build-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-build-guide/references/build-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-op-templates/references/project-skeleton.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-op-templates/references/operator-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-op-templates/references/project-skeleton.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-op-templates/references/operator-template.md
  role: supporting
created_at: '2026-09-11T03:46:00Z'
updated_at: '2026-09-11T03:46:00Z'
platforms: []
---

# 直调对象库构建与 PyTorch 分发合同

来源直调工程将bisheng处理的Kernel/launch与g++处理的Torch plugin/extension分离，再合并对象库到Python扩展。设备文件可以是.cpp并显式-xasc，也可用对应工具链支持的.asc；后缀不是判定语法可用的唯一条件。普通Host声明头通过extern C连接launch，参数个数、整数宽度、指针空间及stream保持一致，std::tuple等C++接口还需相同C++ ABI。

来源CMake在同一语言配置中临时切换CMAKE_CXX_COMPILER，片段本身不足以证明每个target最终用了预期编译器；应检查实际生成命令/compile_commands，或使用工程支持的独立编译机制。对象库进入共享扩展还要匹配PIC、ABI选项及依赖。register_direct_launch_op宏在文中只有追加列表注释，需要真实宏实现；file(GLOB)自动发现目录还依赖重新配置、目录合法CMake及项目实际发现机制。

来源映射ascend910b/ascend910_93到dav-2201、ascend950到dav-3510，是所给构建版本的目标表；SoC名称检测、安装SDK支持、硬件可用能力和评测镜像应逐一一致。不能从架构名推导所有API或dtype可用。build.sh忽略未知参数会掩盖拼写错误，pip与python也可能来自不同环境；安装文件应对应本次构建。

_C.abi3.so与cp38-abi3 wheel名称不独立证明兼容：Python limited/stable ABI约束和Torch/TorchNPU C++ ABI约束不同，后者不会因ABI3标签自动跨版本稳定。PyInit__C负责Python入口，加载共享库时的TORCH_LIBRARY静态注册还需目标确实链接，不能只复制空extension片段就期待算子存在。

TORCH_LIBRARY_FRAGMENT声明schema，TORCH_LIBRARY_IMPL分别提供Meta与PrivateUse1实现。参数类型、默认值、可选性、别名/inplace和输出结构要与真实合同一致；来源要求和proto文本逐字相同是评测约定。Meta推导shape/dtype/layout，不访问真实输入数据；empty_like只适用于同形同类型输出，归约、广播、Matmul及动态输出不能照搬。

NPU实现应验证输入device、dtype、shape/stride及属性，设置对应DeviceGuard并使用当前正确stream。来源dtype lambda只处理float/half后无条件return0，其他dtype会返回未计算的输出，必须有明确支持分派或拒绝。标签RunOpApi不等于已经成功执行Kernel，异步错误仍需在适当完成点观测。

本地包与site-packages可指向不同制品；算子符号存在只是注册证据，需结合实际扩展路径/哈希、源版本、编译参数和执行轨迹。源码时间戳或重装成功不能单独证明新进程加载正确对象。重试一次成功也不能直接把配置失败命名为无害抖动，应保留日志并核对产物；瞬态也可能是构建竞态。
