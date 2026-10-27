---
type: Guide
title: 直调工程的编译单元、目标名与实际加载产物
description: 区分 ASC 设备编译、C++ 注册 ABI、CMake 目标和运行路径，保留模板依赖条件。
tags:
- task.debugging
- topic.compilation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/CMakeLists.txt
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_extension/ops.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_host/add_custom.asc
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/scripts/test_torch.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kernel_launch_details.md
  role: supporting
created_at: '2026-09-11T02:49:27Z'
updated_at: '2026-09-11T02:49:27Z'
platforms: []
---

# 直调工程的编译单元、目标名与实际加载产物

ASC设备单元可以使用设备关键字与启动语法，普通C++负责框架注册，通过声明头连接编译器生成的Host启动桥。来源桥参数包含blockDim、l2Ctrl、stream和Kernel指针参数；它是特定工具链启动ABI，不能把Kernel函数原型直接当作普通C ABI。Host包含Kernel .asc只在相应ASC编译单元中成立。C++标准/ABI、Torch/TorchNPU编译选项、CANN头与链接库须一致。

来源CMake在project前发现ASC，构建add_custom可执行文件与add_custom_ops共享库，同时无条件查找Python开发组件、Torch和TorchNPU，因此仅想构建可执行文件也仍受这些配置阶段依赖影响。应区分配置依赖与目标实际链接依赖；不能因存在独立Host入口就声称可不装框架完成当前配置。硬编码aarch64头目录也不自动适用x86_64。

编译选项--npu-arch=dav-2201对应特定目标，文档列出其他arch并不能证明源码、API和实际安装包支持它。输出文件存在、动态库加载成功、dispatcher注册成功、目标设备Kernel执行成功是不同证据。

CMake逻辑目标名、生成文件名和脚本查找路径要分别核对。来源run的OP_NAME为add，而可执行目标为add_custom；脚本进入build后，test_torch又用build/<so>相对路径，会查找build/build层级。运行失败可能是产物身份或工作目录不一致。--torch分支直接使用已有构建也可能测试旧源码，不能把测试通过归给尚未重建的修改。

测试报告应绑定实际加载的库路径、构建配置与源版本。来源复制、重命名和执行命令保留于Skill；这里记录使不同工程布局均成立的编译、ABI与产物约束。本批未构建这些模板。
