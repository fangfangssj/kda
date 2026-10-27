---
type: Guide
title: TTK 数据范围、容差与门禁解释
description: 区分环境可启动、单用例链路和完整数值验收。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/ttk-converter/00-execution-order.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/ttk-converter/03-kernel-mode.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/ttk-converter/01-csv-common.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/workflow.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# TTK 数据范围、容差与门禁解释

来源范围映射为normal固定seed随机，zero=(0,0)，extreme=(0.9*dtype_max,dtype_max)，negative负随机，tiny_pos=(1e-7,1e-5)，all_ones=(1,1)，near_zero=(-0.01,0.01)，with_inf=(1,inf)，with_nan=(nan,nan)，seed42用于复现。区间端点不等于实际分布：无穷端点的随机uniform计算可能产生NaN或失败，nan字面量也未必被literal parser接受；是否确实注入目标特殊值需检查消费者与生成数据。extreme只覆盖正端，未自动包含负最大或溢出交互。

来源按输出dtype给FP16/BF16(rtol,atol)=(0.001,0.001)、FP32=(0.0001,0.0001)、其他(0.001,0.001)，另有absolute_precision默认1e-8。这些是测试工具配置，不是所有算子通用精度合同；整数/布尔通常需要相应精确规则，BF16也不能仅因表中与FP16相等而获得硬件保证。逐输出及TensorList成员tolerance需与实际oracle/误差公式一致。

kernel_gate passed只表示ops-test-kit可用且TTK kernel --help能在当前或setenv后启动；它未验证case schema、注册函数、计算结果或目标shape。passed_after_source说明采用了哪个环境脚本，不是设备执行成功。skipped保留未执行原因，不能计入通过。

来源wrapper只串行执行low/high各1条，并使用--golden-mode Disable。此模式禁用参考计算，dyn_precision=SUPPRESSED只符合该链路门禁，precision_status也未参与；不能对外称数值精度验证通过、全量用例通过或全部路径覆盖。memory_oob_status为空允许过门禁，也不等同明确完成了越界检测且PASS。

有效链路证据包含进程返回码、目标testcase_name结果行、编译/启动前置状态与相关日志；INVALID_TILING等前置失败应保留。CSV行数与JSON用例数相等只能检查转换数量，未确认参数语义或禁用行执行。Kernel模块据来源信任上游Mapper，不重新校验schema，因此上游证据版本与转换输入必须一致。

共享profiling/profiler目录可导致并发结果或清理争用，Directory not empty不单独证明算子错误；隔离输出或串行重现能区分环境干扰。报告中的stdout/stderr tail可能截断根因，结果缺行不能用其他用例的成功补齐。阶段执行命令、自动wrapper和交互门禁仍留在Skill。
