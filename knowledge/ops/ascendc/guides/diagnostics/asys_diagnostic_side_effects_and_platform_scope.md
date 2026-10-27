---
type: Guide
title: asys 诊断的产品范围、超时和压测电压恢复
description: 说明诊断权限、并发与依赖、HBM/CPU 超时差异，以及压力检测异常退出后的电压观测和恢复。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/diagnose-and-config.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/constraints.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/info-and-health.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T07:58:31Z'
platforms: []
aliases:
- asys diagnose
- stress_detect
- asys config
- aicore_stl_detect
---

# asys 诊断的产品范围、超时和压测电压恢复

asys 诊断会实际执行检测，部分模式会改变设备状态，不能当作纯日志解析。本卡保存固定 reference 的范围和副作用，服务 Ascend C 故障区分；没有执行这些命令，也没有核实全部工具发行版的产品支持。

## 产品与部署范围

来源总体限定 Ascend EP，不支持 RC；一键收集不支持集群、容器、虚拟机和云场景。不同子命令仍有更窄的约束，不能把离线读文件与设备诊断混为一谈。diagnose 全部模式及 config 要求物理机和 root；Device 固件/系统日志、黑匣子和 stackcore/coretrace 收集也需要 root，非 root 材料可能不完整。

| 能力 | 固定来源支持范围 |
|---|---|
| collect、launch、stacktrace、info、health | 所列 A2/A3、950、Atlas 训练/推理系列、200I/500 A2；仍受部署/权限限制 |
| analyze trace/coredump/stackcore/ub/aicore_error | 上述所列产品；具体输入与依赖另核对 |
| analyze coretrace | 仅 950 |
| diagnose stress/hbm/cpu、config、profiling | A2/A3 和 950；不支持 Atlas 训练/推理系列及 200I/500 A2 |
| diagnose aicore_stl_detect | 仅 950 |
| diagnose component | 所列产品，但当前仅 AI Vector 组件 |

“支持”指该资料中的功能条目，不表示硬件所有配置均可用。组件检测不支持并行；总体 asys 大量采集会占内存，不建议多进程；同用户同时间段同机器多个作业可能材料交叉；同一卡住进程也不能并行导出堆栈。这些约束涉及不同对象，不能只限制进程数就认定材料隔离。

## 模式、时间与产物

diagnose 的 -r 选择 stress_detect、hbm_detect、cpu_detect、aicore_stl_detect 或 component；-d 省略时检测结果覆盖所有设备。--timeout 仅 HBM/CPU 有效，来源默认 600 秒：HBM 为 `[0,604800]`，0 表示只执行一轮；CPU 为 `[1,604800]`，不能把 HBM 的 0 约定套到 CPU 或其他模式。

--output 指定时保存 `diagnose_result_{time_stamp}.txt`，省略则仅终端输出。压力检测需要事先存在配套算子二进制包，源码称为 `Ascend-cann-*-ops-*.run`；二进制包存在仍需匹配本次环境。解析文件用到 gdb、readelf、addr2line 的依赖见[解析输入与符号](asys_analysis_inputs_symbols_and_error_scope.md)，缺少它们不能直接归因设备故障。

## 压力检测的电压状态

stress_detect 涉及调整 Device 部分电压。正常结束通常自行恢复，异常退出可能不恢复。`asys config -d=ID --get --stress_detect` 读取相关配置，`--restore --stress_detect` 执行恢复；config 缺省 Device 0，与 diagnose 的全设备默认不同。不能在多卡压测后只恢复默认 0 就宣称全部设备恢复。

来源输出可含 `AI Core Voltage (MV)`、`Bus Voltage (MV)`，应以同一设备在检测前后、同一工具字段的值比较，保留原始单位标签。恢复命令输出成功不等于已验证业务恢复；应读回配置，并结合业务运行状态。不要把来源示例 850/900 等电压值作为手动写入目标或所有芯片的正常阈值。

检测结果按[模式返回语义](asys_health_and_diagnostic_result_semantics.md)解释。Fail/Warn 的范围不同，压力测试通过只说明其测试覆盖，不能证明全部算子、dtype、shape 和负载都正确。异常检测应保留本次设备、原配置、实际模式、结束方式、结果及 `[ERROR] AML` 关联 plog，不能用恢复环境后的结果替代故障时的现场证据。
