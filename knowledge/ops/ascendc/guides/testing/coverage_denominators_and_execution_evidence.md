---
type: Guide
title: UT 覆盖率的统计范围与执行证据
description: 区分文件范围、行分支计数和语义断言，纠正仅编译或汇总表查零替代测试的做法。
tags:
- task.testing
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/coverage-guide/coverage-extraction.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/coverage-guide/test-implementation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ut-guide/build_ut.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/workflow/step1.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/workflow/step4.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ops-transformer/ops-transformer.md
  role: supporting
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# UT 覆盖率的统计范围与执行证据

覆盖率首先限定同一构建配置、实际运行集合和源文件范围。全仓报告与单算子报告分母不同，api、host、kernel、aicpu、tiling 等路径应按实际目录筛选后统计；lcov --extract 可以按路径模式提取，但只看 lcov --list 的前 50 行不能证明其余全是目标算子。

lcov --list 是文件汇总，grep ':0' 不能可靠定位未覆盖源代码。行信息需读取实际记录中的 DA 行执行计数，分支需 BRDA 及对应分支配置/源位置；0 与未采集或未执行计数不能混为同一原因。源文件、编译选项及测试运行必须与数据一致。0% 可能是未链接、未执行、过滤错或配置错，不能只归因于 CMake 漏项。

用例要满足目标分支条件及抵达它的前置条件，检查具体输出或错误行为。行覆盖不证明条件组合覆盖，分支覆盖不证明数值正确；只有返回码或 Shape 断言也不能证明 Kernel 结果。来源 80%/90%、固定边界或异常用例数量属于该 Skill 的目标，不是充分性定理。

--noexec 在来源构建脚本中只编译；CSV 读取、参数实例化和运行断言错误可能到执行时才出现。scoped --ops、模块标志和 --soc 的实际支持由项目 build.sh 决定，不把 -u --ophost、--ophost_test 等不同脚本选项混为通用 CANN 命令。来源的 ascend310p→arch20、ascend910b/ascend910_93→arch32、ascend950→arch35 是该仓构建分类，不能仅据文件夹断言芯片能力。

ops-transformer 来源通过 BUILD_PATH 定位构建资源，并按目标 SoC 选取架构测试；遗漏编入或 gtest_filter 未匹配会造成想测的路线未运行。编译成功、退出码 0、用例数及生成报告文件均需结合实际列出的测试和断言理解。CSV 数据与测试代码分离后，未改变 schema/框架时修改数据通常无需重编，但新进程仍需确实读取新文件。

来源日志开关 ASCEND_SLOG_PRINT_TO_STDOUT 与 ASCEND_GLOBAL_LOG_LEVEL 只作为其环境的调试线索；DEBUG/INFO/WARNING/ERROR 的 0/1/2/3 映射需按运行版本确认。日志更多不扩大测试断言的范围。自动/交互流程、选择文件、逐批运行和报告模板不属于本知识卡。
