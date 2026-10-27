---
type: Guide
title: asys 文件解析的输入配对、符号搜索与错误范围
description: 区分 trace、coredump、stackcore、coretrace、Unified Bus 和 AI Core Error 解析的依赖、参数、版本及首错边界。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/analyze-files.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/constraints.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/output-layout.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T08:00:11Z'
platforms: []
aliases:
- asys analyze
- symbol_path
- coretrace
- stackcore
---

# asys 文件解析的输入配对、符号搜索与错误范围

解析成功只说明某个输入经过了对应解析路径，不能证明它来自正确任务或使用了匹配符号。本卡保存固定 asys reference 的合同，用于 Ascend C 排障材料判读；未运行 asys，未固定实际工具发行号。

## 输入与解析模式

| `asys analyze -r=...` | 主要输入/依赖 | 来源边界 |
|---|---|---|
| trace | `--file` 单个 trace `.bin` | asys 环境版本与生成 trace 的环境版本一致 |
| coredump | `--exe_file` 与 `--core_file`，依赖 gdb | 可执行文件必须与 core 配对，否则结果可能错误 |
| stackcore | `--file` 或 `--path`，依赖 readelf/addr2line | Host/Device stackcore 或由 coredump 解析产生的文本，匹配采集环境/动态库 |
| coretrace | `--file` 或 `--path`，依赖 addr2line | 来源仅支持 950；不能只按相似文件名改用 stackcore |
| ub | `--path` 下的指定二进制 | Unified Bus 维测文件，不是 Ascend C Unified Buffer 张量 |
| aicore_error | 日志与 dump 目录，--path 可省 | 省略则自动收集；多个错误只分析日志时间上的第一个 |

通用 `--output` 是输出目录前缀，通常创建 `asys_output_timestamp`，不传使用执行目录。空/非法值、无权限、创建失败会报错退出。readelf、addr2line 不但要存在，还须对执行用户可执行。Device stackcore/coretrace/Unified Bus 材料可由 [msnpureport 导出](msnpureport_export_scope_and_log_identity.md)，导出时间和设备映射仍需保留。

trace 将单文件解析为 txt，事件记录带时间戳及字段。不同事件字段集合可不同，不能强制每条都有 tid/count；数组字段是多个 ID，不能按固定列宽或单个数值解析。

## coredump 的两类选择

`--reg` 默认 0：0 不添加寄存器数据，1 每个线程加一条，2 为线程所有栈加寄存器数据，后者占 Host 资源且耗时。`--symbol` 默认 0：0 将带地址的行解析为 stackcore 格式，其他行显示 Ignore；1 只解析 `in ?? ()` 行，其他行保留 gdb 原始堆栈。Ignore 不是“没有故障”，两种输出也不能逐行直接当成相同格式。

结果 `[process]` 记录崩溃原因/pid/tid，`[stack]` 按线程给出帧，`[maps]` 记录地址区间、大小、偏移和映射对象。可进一步用 stackcore 模式解析符号。地址缺失或栈溢出可能使解析失败；有 core 而没有匹配 exe/共享库时，不能用另一构建的二进制填补并宣称定位准确。

## 符号文件与搜索方向

stackcore/coretrace 的 `--file` 和 `--path` 二选一；--path 会递归读取输入目录及子目录。`--symbol_path` 是逗号分隔的动态库目录，**按传入顺序查找、只扫描每个目录当前层、不递归子目录**。输入递归不表示符号也递归。省略 --symbol_path 时从材料获取动态库路径，因此跨机器或移动目录后可能找不到，或错误命中同名的其他版本。

来源要求在获取材料的环境解析，核心是工具、运行库、可执行文件和符号身份一致；迁移材料时必须额外证明这些身份，不能只复制一个文本就假定相同。stackcore 线程名获取失败可显示 unknown；coretrace 首行给信号和进程，函数解析失败可能退化为十六进制地址。未知名不等于不存在该线程/函数。

出现 `?`、`??:??` 或行号偏差时，核对是否具备匹配的调试信息和动态库。来源提示编译 `-g`、链接 `-rdynamic` 与符号可见性有关，但它们作用不同，不能承诺补一个选项就恢复所有优化后帧。优化导致代码重排、内联等也可能改变源码行对应关系；应连同构建配置、映射地址、符号文件一起解释。

## Unified Bus 文件名与内容

ub 模式读取 --path 下相应文件并生成同名 txt：`ubnl_dfx_config_item.bin` 为网络层配置，`ubnl_dfx_statistic.bin` 为统计，`ubnl_dfx_ssu_schedule.bin` 为 SSU 队列及丢包统计，`ubmem_daw.bin` 为 memory 配置，`ubtpl_acl_src.bin` 为 Transport 配置，`sl_to_vl.bin` 为 QOS 配置。msnpureport 的布局示例也出现 `ubtpl_acl.bin`，文件名存在来源差异，应核对本版本解析器接受名，不能任意重命名二进制以伪装格式匹配。

## AI Core Error 的首错与自动采集

来源以 `there is an aivec error exception` 或 `there is an aicore error exception` 为触发线索。`-d` 默认 Device 0，`--path` 可指定材料目录；不传时依赖 ASCEND_PROCESS_LOG_PATH、NPU_COLLECT_PATH、DUMP_GRAPH_PATH、ASCEND_WORK_PATH、ASCEND_CACHE_PATH、ASCEND_CUSTOM_OPP_PATH 自动收集。全部不存在时从执行目录收集；执行 asys 时的变量应与业务运行一致，否则可能混读旧材料。

同一材料中多个错误只分析按日志时间首次出现者，输出 info.txt 不等于已解释所有故障。`--output` 不能等于 --path 或位于其子目录，以免循环复制。日志隔离应围绕本次运行身份建立，不把删除现场设为解析的必需条件。判读应核对首个错误、设备、任务和编译制品；[msaicerr](msaicerr_dump_dtype_and_analysis_scope.md)虽也有首错与材料约束，不能因此假定两个工具的全部模式、默认路径和副作用一致。
