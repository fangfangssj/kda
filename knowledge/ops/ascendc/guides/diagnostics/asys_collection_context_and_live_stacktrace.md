---
type: Guide
title: asys 故障采集上下文、业务子进程与实时堆栈
description: 解释环境变量覆盖、编译与 Dump 材料前提、信号 35、堆栈导出参数和超时观测，避免采集成功却缺失关键证据。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/collect-and-launch.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/constraints.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/faq.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/analyze-files.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T08:00:11Z'
platforms: []
aliases:
- asys collect
- asys launch
- ASCEND_COREDUMP_SIGNAL
- stacktrace
---

# asys 故障采集上下文、业务子进程与实时堆栈

本卡用于 Ascend C 算子故障取证，限定于固定 asys reference 的采集行为。来源没有固定 asys/CANN 发行号，命令和信号机制须结合本次环境核对，不能把以下资料解释为所有版本的保证。本卡未运行 asys、发送信号或复跑业务。

来源限定 EP，不能在 RC 使用；一键收集不支持集群、容器、虚拟机和云场景。同用户同时间段同机器多作业可能交叉采集，且大量收集会占内存，不建议多进程并行。详细产品和权限范围见[asys 诊断执行边界](asys_diagnostic_side_effects_and_platform_scope.md)；能读一个文件不证明完整自动采集受支持。

## 收集已有材料与复跑材料

`asys collect` 不复跑业务；`--task_dir` 指定编译 `.o/.json` 和 dump 材料目录，未指定或在该目录未收集到时会从环境自动收集。`asys launch --task="完整业务命令"` 则创建子进程复跑后收集。自动回退意味着指定目录不一定是唯一来源，判断材料身份需核对原始路径、设备、pid、构建和时间。

两者 `--tar T|True`（不区分大小写）压缩成 tar.gz 且不保留未压缩目录，F|False 不压缩，默认不压缩；`--output` 为目录前缀，最终创建 `asys_output_timestamp`，默认执行目录。无写权限、无效路径或目录创建失败会导致退出。不要用上次同前缀的材料确认本次成功。

launch 不支持业务脚本内部把真实任务直接后台化，如 `python test.py &`，因为无法可靠感知业务结束点。主动终止 launch 不保证其业务子进程退出；Ctrl+Z 通常是挂起，不能当作清理完成。来源 FAQ 将残留进程及重定向写文件与后续复跑冲突关联，可能出现 Segmentation fault/BrokenPipeError；这些是诊断线索，不是这些错误的唯一根因。处理前应确认目标进程身份，不能按宽泛进程名结束其他任务。

## 环境变量与可收集信息

自动收集依赖执行 asys 时的 `ASCEND_PROCESS_LOG_PATH`、`NPU_COLLECT_PATH`、`DUMP_GRAPH_PATH`、`ASCEND_WORK_PATH`、`ASCEND_CACHE_PATH`、`ASCEND_CUSTOM_OPP_PATH`，应与业务运行时一致。launch 会临时接管 NPU_COLLECT_PATH、ASCEND_PROCESS_LOG_PATH、ASCEND_WORK_PATH，启动前手动设置会被覆盖；任务脚本内部再设置则反过来覆盖 asys，可能使材料不全。来源描述命令结束时关闭临时变量，不据此推断调用方 shell 原值的所有恢复细节。

| 材料 | 收集前提或范围 |
|---|---|
| 软件/硬件/状态、Host 日志 | `software_info.txt`、`hardware_info.txt`、`status_info.txt` 与 Host log；内容随环境/权限变化 |
| 全部健康信息 | `health_result.txt`；不能用终端只显示前 5 组故障替代 |
| 编译与 dump | `dfx/ops` 的 `.o/.json`、`data-dump`、`graph`；L0 exception dump 不收集 graph 目录信息 |
| 编译过程日志 | 来源仅在业务复跑时收集；设置 NPU_COLLECT_PATH 后系统才生成 `extra-info/ops/op_compile_stats.log`，记录在线编译/缓存/二进制等身份 |
| vendors 自定义包 | ASCEND_OPP_PATH 正确时，按 `vendors/config.ini` 的 load_priority 收集 vendors 下配置 |
| 独立自定义包 | ASCEND_CUSTOM_OPP_PATH 下 `config/*.json`；不能只看系统 OPP 路径 |
| 调试版二进制 | 正确 ASCEND_OPP_PATH 下的 `debug_kernel`；未配置则默认不收集 |
| run 包安装日志 | 安装用户与应用执行用户需一致 |
| Device 固件、系统日志、黑匣子与 stackcore/coretrace | 来源要求 root，普通用户结果范围受限 |

launch 的 `log/host/screen.txt` 可因业务重定向而为空，不代表业务没有输出；`user_cmd` 记录实际命令。`dfx` 将 bbox、data-dump、graph、ops、stackcore、atrace 分开；log/device 的 firmware、slogd、application、system 与 log/host 的 message、install、cann、driver 表示不同生产者，不应只按文件后缀混合。

launch 默认开启编译文件、GE/TF Adapter dump 图收集。asys.ini 的 `[launch]` 中 graph/ops 默认 TRUE，graph=FALSE 会使 dump_ge_graph/dump_graph_level 无效；来源默认 dump_ge_graph=2（不含权重等数据的基础图）、dump_graph_level=3（最后阶段图），分别映射 DUMP_GE_GRAPH/DUMP_GRAPH_LEVEL。log_level=INFO、log_event_enable=TRUE、log_print_to_stdout=FALSE 分别对应 ASCEND_GLOBAL_LOG_LEVEL、ASCEND_GLOBAL_EVENT_ENABLE、ASCEND_SLOG_PRINT_TO_STDOUT。业务脚本内的实际设置可覆盖这些值，配置文件默认值不是本次采集已启用的证据。

第三方依赖版本采集可由工具的 dependent_package.csv 配置，每行 `依赖名,查询命令`，来源要求逗号后无空格，例如 `cmake,cmake --version`。这是查询配置格式，不是应安装的固定软件全集；执行的查询应对应本次业务环境。

## 实时堆栈的信号与参数

`asys collect -r=stacktrace --remote=PID --all` 用于进程卡住的场景，PID 必须存在且至少为 2；`--all` 表示全部线程。`--timeout` 范围 1–60 秒，默认 10。未指定 `-r=stacktrace` 时 collect 表示一般材料收集，不能混用 --remote、--all、--quiet。--quiet 只关闭确认交互，不改变信号或适用条件。

来源说明实时堆栈会向目标发送**信号 35**。若 `ASCEND_COREDUMP_SIGNAL=none` 关闭相应 trace 信号处理，发送会使进程终止且无法导出；非 none 或未设置是来源所述开启条件，但仍需确认当前版本与进程实际初始化状态。不能把 --quiet 当作安全检查，也不能把此工具行为当成所有 Linux 进程的普遍信号含义。

同一卡住进程不支持并发导出；业务尚未卡住时也可能发生发送失败、生成超时或解析失败。asys 搜索 trace 目录，历史文件很多会延长执行；默认目录为 `$HOME/ascend/atrace/`。评估或整理旧材料时应保留故障证据，不把清空日志设为无条件前置动作。成功输出的 `stackcore file path` 需属于本次 PID/时间，后续按[符号解析合同](asys_analysis_inputs_symbols_and_error_scope.md)处理。

## 超时需要区分的观测

来源错误 `Generating the stackcore bin file timeout` 需要结合目标进程本次 plog 判断：

| 观测 | 解释边界 |
|---|---|
| 尚未见本次 `attr init success` | 可能未完成初始化；该关键字本身是完成的正向信号，不是失败原因 |
| `close the signal capture function` | 信号处理关闭，核对 ASCEND_COREDUMP_SIGNAL |
| `unregister all signal handlers, can not capture signal` | 业务已结束、相关资源释放，旧 PID 不再具有导出条件 |
| 对应 ERROR 记录 | 工具或导出功能异常，需保留具体上下文，不能只按 timeout 猜根因 |

默认 plog 位于 `$HOME/ascend/log/run|debug/plog/plog-{pid}_*.log`，这里的 run|debug 表示二选一路径，不是字面目录。验证采集是否完整需对照运行环境、启用选项、预期材料、信号初始化、退出状态及文件身份，不能只看有无压缩包。
