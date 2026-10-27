---
type: Guide
title: 注册包的架构选择与运行库身份
description: 抽取 ASC 构建目标、依赖布局和运行加载限制，保留脚本中的具体失败边界。
tags:
- task.implementation
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/CMakeLists.txt
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/build.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/examples/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/examples/CMakeLists_geir.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/examples/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_graph/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_host/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/torch/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/build.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_graph/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/custom_compile_options.ini
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/include/blaze/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/include/tensor_api/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/tests/st/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/tests/st/run.sh
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# 注册包的架构选择与运行库身份

ASC包通常把Host定义/推导/Tiling、手写ACLNN、图proto与设备Kernel分成构建目标，再加入npu_op_package。来源强调all不默认生成binary，应确认Host库、目标SoC设备实例和package都实际生成；只找到一个.o或非空.run不能证明全部dtype/模式实例完整或包内制品与本次源匹配。

Add根CMake默认多SoC列表，但Host/Kernel用ASCEND_COMPUTE_UNIT STREQUAL ascend950单字符串判断，混合列表会走arch22分支，不能自动声称一次构建所有架构正确。变量ARCH32实际指向arch22亦可能混淆。OpDef注册ascend910b/ascend950，脚本允许ascend910_93，需核对真实配置映射。Matmul脚本列SUPPORT_COMPUTE_UNITS却未校验，不等于参数支持已生效。

Add的Host include和包名硬编码aarch64，不能直接用于x86_64；Matmul按CMAKE_SYSTEM_PROCESSOR选include，但ST仍硬编码某x86_64 CPack临时路径。应把Host ISA、NPU SoC与SDK安装布局分开。单看头文件存在不保证编译器/库ABI一致，CMake find缓存也可能保留旧包。

cust_opapi除自身符号还可依赖nnopbase、opapi中的Contiguous/ViewCopy等；GE路径另涉及graph/ge_runner及对应graph_base/ge_compiler。stub库可服务链接，不证明运行实现可用。图proto头存在检查和加入include目录不保证实际被编入，应核对代码生成依赖及链接结果。配置阶段GLOB尚未生成的op_proto.cc可能为空，GENERATED属性不能补回未列出的路径。

Blaze与tensor_api是此模板的外部头文件依赖，需匹配版本并使实际设备编译命令看见include、内部impl及共享TilingData路径。custom_compile_options.ini中的OP_KERNEL_BASE_PATH替换须由真实构建器完成；它不是Shell命令。占位README不是库本体，也不应把未固定提交的clone动作作为可复现构建证据。

调用端精确选定vendor头文件/库可减少多包误选，运行时LD_LIBRARY_PATH、RPATH与进程缓存仍可能命中另一个同名libcust_opapi。examples/run.sh取第一个vendor目录与CMake精确Add包可能不一致，且CMAKE_SKIP_RPATH与显式链接选项需检查最终动态段。路径追加空项有当前目录含义，不能把构建成功当加载身份确认。

Mock/Real分开build目录有助于避免USE_MOCK缓存，但默认cmake参数不重置已有缓存；同理跨SoC/模式重配需核对实际选项。ST可选重建分支传--pkg，而根build.sh拒绝该参数；--case在Shell称YAML、README称索引，C++主程序又未解析argv，所以不能声称指定case已执行。shell set-e失败时可能在后续RESULT赋值/自定义消息前退出；退出码和日志须分别保留。

临时替换同一CMakeLists用于GE构建存在并发和中断窗口，恢复trap设置时点亦影响恢复；隔离配置/构建目录更容易确认源码身份。测试运行还需同一个Python环境的torch/torch_npu/CMake包和C++ ABI，单独进程import torch_npu成功不保证test.py已初始化torch.npu。具体安装、清理和执行顺序保留在Skill。
