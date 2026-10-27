---
type: Guide
title: 麒麟 CCE 架构选择与仿真运行库解析
description: 从实际 CMake 模块提取 Kirin arch 映射、编译检测缺口和运行库选择条件。
tags:
- task.debugging
- topic.compilation
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/CMakeLists.txt
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/cmake/Modules/CMakeCCECompiler.cmake.in
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/cmake/Modules/CMakeCCEInformation.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/cmake/Modules/CMakeDetermineCCECompiler.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/cmake/Modules/CMakeTestCCECompiler.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/cmake/npu/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_vector_guide.md
  role: supporting
created_at: '2026-09-11T02:49:27Z'
updated_at: '2026-09-11T02:49:27Z'
platforms: []
---

# 麒麟 CCE 架构选择与仿真运行库解析

固定模板使用自定义CCE语言模块发现ccec，并将参与构建的.cpp设置为CCE。编译规则含-xcce、设备arch与-fPIC，设备对象链接含--cce-fatobj-link，静态库规则含--cce-build-static-lib。是否适用于另一发行版，应以该发行版编译器、头/库和支持矩阵为准，不能将模板模块视为通用CMake原生功能。

| 模板SOC | AiCore arch | VectorCore arch |
|---|---|---|
| KirinX90 | dav-l300 | dav-l300-eff |
| Kirin9030 | dav-l311 | dav-l311-eff |

模板产品前置正则接受更宽的Kirin前缀，而真正arch分支只有表中两项；core类型分支也可能遗漏其他输入。应对SOC/core组合做封闭校验，不能在arch未设置时继续。CMakeTestCCECompiler直接把WORKS置1，没有实际try_compile，因此“配置通过”不证明编译器可用。自动同步、依赖过滤、16000栈参数及关闭栈溢出记录也不能独立证明Kernel安全。

sim链接/加载需要与SOC一致的runtime_camodel与仿真库，npu使用真实runtime；头、编译器和动态库搜索路径可能来自不同安装根。来源构建阶段有runtime/lib64、tools/simulator等搜索项，运行脚本使用x86_64-linux/simulator/<SOC>/lib，应核实最终命中的文件。CMake目标名、OUTPUT_NAME与run mode拼接的文件名须一致。

run.sh的-i参数写入ASCEND_INSTALL_PATH，但后面实际传给CMake的是按其他环境推导的_ASCEND_INSTALL_PATH，不能假设显式参数已生效。来源设置LD_PRELOAD指向libstdc++.so.6.0.34，是某版仿真依赖的处理方式；需匹配Host架构、GLIBCXX符号和其他运行库，不能只按文件名判断兼容。教程下载的是x86-64包，不能直接用于aarch64，也无需把下载流程和二进制复制进知识库。

Simulator可能无法自动探测芯片，PlatformAscendCManager::GetInstance应采用所支持的显式SOC接口并检查nullptr，再使用平台Tiling API。来源无参必空的说法限于所述环境；不能推广为所有SDK永远如此。模板主程序availableCoreNum=1是样例配置，不是硬件最大核数。

麒麟模板拒绝CPU模式，源码CPU分支还含未定义blockDim和缺少tiling实参的旧调用，不能当可执行调测证据。文档接受sim/npu参数不等于当前Host发行版具备真实麒麟上板能力，参见[麒麟发行环境边界](kirin_simulator_distribution_scope.md)。本批未运行ccec、仿真器或设备。
