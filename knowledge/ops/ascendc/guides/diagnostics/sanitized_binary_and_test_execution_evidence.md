---
type: Guide
title: 插桩产物与内存检测执行证据
description: 在自动化包装中区分构建、安装、实际测试执行和检测结果，保留证据链。
tags:
- task.debugging
- topic.memory
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-crash-debug/references/memcheck/README.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-crash-debug/references/memcheck/automated_workflow.md
  role: supporting
created_at: '2026-09-11T02:40:45Z'
updated_at: '2026-09-11T02:40:45Z'
platforms: []
---

# 插桩产物与内存检测执行证据

msSanitizer memcheck的有效观测需要受支持环境、可执行的真实失败用例，以及匹配的检测产物。来源包装使用-sanitizer与-g构建选项，再安装算子包并执行mssanitizer --tool=memcheck <application>；具体选项/产品支持以所用工具版本为准。普通构建成功不证明已插桩，调试信息存在也不独立证明sanitizer启用。

跳过构建时，应确认已有包对应目标源码、SOC、dtype/tiling变体和插桩配置。构建目录中的.run、安装目录中的库以及运行时实际加载的库是不同对象；多包查找、vendor环境或库搜索路径可能使测试加载旧产物。CANN编译环境和自定义算子运行环境都影响执行，不能因脚本分别加载就认定互不干扰。

构建失败、安装失败、测试未发现/未执行、工具启动失败、超时和memcheck报告异常应分别记录。ERROR/WARNING数量和前五条摘要不能替代完整原始报告及退出状态；pytest收集零用例或进程被超时终止时，即使没有非法访问报告也不是内存检查通过。增大600/1200秒包装超时只能改变观测窗口，不能独立修复死锁。

保留输入、实际产物身份、检测参数、运行阶段、原始报告和可用调用栈，将Illegal read/write、Misaligned或多核重叠位置对应到该构建中的地址计算。检测能力和内存池子分配覆盖见[内存检测报告](memory_sanitizer_reports.md)；无告警不证明精度、所有路径或时序均正确。

脚本JSON字段、目录布局、自动清安装目录、复制/安装命令和Skill触发方式属于工作流，不进入知识正文。这里提炼的是产物与观测的对应关系；本批未执行安装、memcheck或ST。
