---
type: "Runbook"
title: "507035排障：plog定性+跳层二分+对账"
description: 自定义算子 NPU 联调报 sync stream 507035（向量核异常）而 stdout 无定位信息时：开 plog 按 errcode/errorStr、fault kernel_name+tilingKey、GetArgsInfo args 指针表、AIV bitmap/PC 四类键拿指令级现场，再配合 kernel 跳层二分定位最小触发单元、DevMalloc 尺寸对账判定缓冲是否真的分配。复现增量（507015 MIX 形态）：核型判别（AIC vs AIV 故障）以 device log "Bit map is" 行 aicmap/aivmap 为唯一可靠依据——"aicore exception" 报文与 l1 error info 标签都不区分核型，l1 error 非零曾把 AIV 故障误导成 AIC L1 误诊。
tags: [task.debugging, topic.implementation.debugging, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# 507035排障：plog定性+跳层二分+对账

## 现象

自定义算子（Ascend950PR）跑 C++ ST / aclnn 直调（NPU 真跑）时，多个用例统一报：

```
ACL error 507035 at .../test_aclnn.cpp:...: aclrtSynchronizeStream(stream)
```

507035 是 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（向量核异常，`include/acl/error_codes/rt_error_codes.h`）——`aclrtSynchronizeStream` 只告诉你"kernel 崩了"，不给任何细节：stdout 只有错误码与调用点行号，没有 kernel 侧输出、异常地址或出错指令。同一时期还可能混有 361001（算子执行错误）、161001/161002（参数校验）等 host 侧错误，需要一套固定的日志定位流程把"哪条指令、什么错误、哪个 kernel 变体、入参是什么"挖出来。本卡沉淀该流程在两条独立 trace（CANN 9.1.0 与 9.2.0）的联调中反复使用并产出关键证据的版本。

## 适用场景

- 工程形态：自定义算子 NPU 联调（ST/直连 aclnn 程序），kernel 执行期崩溃（507035 向量核异常等异步执行错误，报错点与出错点分离）。
- API 路径：AscendCL 日志环境变量（`ASCEND_GLOBAL_LOG_LEVEL`/`ASCEND_SLOG_PRINT_TO_STDOUT`/`ASCEND_PROCESS_LOG_PATH`）；host 侧 plog（默认 `~/ascend/log/debug/plog/`，设备位图在 `debug/device-0/`）；runtime DFX 错误信息（`ProcessDavidStarsCoreErrorInfo`/`PrintErrorInfoForDavinciTask`/`GetArgsInfo`/`PrintTaskErrorMsg`/`PreCheckTaskErr`）；`DevMalloc` 记录；kernel config JSON。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510）+ CANN 9.1.0 与 9.2.0。
- 适用粒度：platform_wide——CANN 9.x 上所有自定义算子的向量化异常初筛。
- 路线优先级：recommended_fastpath（崩溃初筛第一站，不改代码、不改测试即可拿到指令级现场）。
- 明确不适用 / 未验证场景：host 侧段错误（aclnn 框架层崩溃，走 host 崩溃路径）；纯编译错误；hang/超时类问题（无错误码可 grep）；其他 CANN 版本的 plog 字段差异（未验证）。

## 判别方法（排障流程，含可执行命令）

**第 1 步：错误码语义速查**：

```bash
cd $ASCEND_HOME_PATH && grep -rn "507035" include/ x86_64-linux/include/ | head -5
# → include/acl/error_codes/rt_error_codes.h:#define  ACL_ERROR_RT_VECTOR_CORE_EXCEPTION 507035
```

**第 2 步：跑测时把 DFX 日志打到 stdout**（host 侧参数/框架层错误会用 `[ERROR] OP(...)` 行自曝文件:行号）：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1 && export ASCEND_SLOG_PRINT_TO_STDOUT=1
./build-real/test_aclnn_<op> 2>&1 | grep -iE "<op>|nullptr|161001|PARAM_NULL|not supported|GetWorkspaceSize" | head -30
```

**第 3 步：开 plog 重跑并找最新文件**（设备侧 errcode/fault kernel/args 只落 plog，stdout 覆盖不到）：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=0 ASCEND_PROCESS_LOG_PATH=~/plog
rm -rf ~/plog/* 2>/dev/null
cd <test_build_dir> && timeout 300 ./test_aclnn   # 或 ./run.sh
ls -lt ~/ascend/log/debug/plog/ | head -3   # 未重定向时用默认路径
```

注意：`/tmp` 下建目录可能 Permission denied，用家目录（`~/plog`）。

**第 4 步：按四类键 grep plog**（核心步骤，键选错会被海量 ATRACE 淹没）：

```bash
# 设备日志：AIV/AIC error bitmap（aivmap 非零 = 向量核异常，可数出几个核）
grep "Bit map is :" ~/plog/debug/device-0/*.log | tail -10
# 主 plog：错误码与错误串 + fault kernel + 任务失败详情 + 同步错误码
grep -iE "(errcode|errorStr|out of bounds|tilingKey|fault kernel_name)" ~/plog/debug/plog/plog-<pid>_*.log | head -10
grep -n "PrintTaskErrorMsg\|PreCheckTaskErr\|TaskFailCallBack\|rtStreamSynchronize" ~/plog/debug/plog/*.log | grep -v "result=0x0" | head -20
# kernel 实际收到的入参指针表（判别"某输出/workspace 是否被塞了占位/无效地址"）
grep -A2 "GetArgsInfo.*AIC_INFO.*args" ~/plog/debug/plog/plog-<pid>_*.log | head -20
```

**第 5 步：读四类证据并解释**：

- `The extend info: errcode:(N) errorStr: ...`——设备侧错误码，与下表对照；
- `PrintErrorInfoForDavinciTask ... fault kernel_name=<Op>_<hash>_<k>`——崩的是哪个 kernel 变体（`_<k>` 后缀对应 simplifiedKey/TilingKey 变体）；
- `GetArgsInfo:tilingKey = K, ... args(0 to N) after execute:0x...`——kernel 实际收到的入参指针表；
- `ProcessDavidStarsCoreErrorInfo: ... error code = N, pc start/current, mte error info, vec error info`——指令级错误现场（pc 偏移、MTE/VEC 错误位）；`PrintAicErrReportInfo` 的 aivmap bitmap 告诉你哪些核炸、`sqe_type=1(aiv)` 告诉你是向量类任务。

**errcode 解释表**（实测语义）：

| plog errcode | errorStr | 指向 |
|---|---|---|
| 95 | The DDR address of the MTE instruction is out of range | MTE 搬运目标/源地址越界——查输出/入参/workspace 指针有效性（占位输出、workspace 声明与分配、GM 偏移） |
| 264 | The address for scalar to access GM is invalid | 标量（SetValue/GetValue）访问 GM 无效——查 workspace 分配大小/指针（同批 trace 实测） |
| 271 | The address for scalar to access the internal buffer is out of bounds | 标量（GetValue/SetValue）访问 UB 越界——查 ubFactor/缓冲分配/索引上界 |

**第 6 步：kernel 内跳层二分——定位最小触发单元**。对可疑执行路径逐层短路（每轮 `build.sh` + 跑 ST），例如 Group 分核路径：Phase 2 换成已知正确的旧路径 → 仍失败 → 排除 Phase 2；再短路 Phase 1 → 全 PASS → 问题在 Phase 1；只短路其中的 workspace 写函数 → 全 PASS → 最小触发单元 = 写 workspace GM。二分裁剪同样适用于计算链：把计算逐段替换为最简拷贝（只做 Q→Y / 只写一个最小块到可疑 GM 区），锁定"访问哪块 GM 就崩"。

**第 7 步：指针有效性 + 分配对账**：

```bash
# 把可疑 GM 访问改成"读 32B 到 UB"重跑：读也炸 => 指针无效，而非写参数（对齐/blockLen）问题
grep -i "DevMalloc" ~/plog/debug/plog/*.log | grep -v "size=64" | head -20
```

期望尺寸（如 workspace 880B）没有对应 `DevMalloc` 记录 ⇒ 上层（aclnnGetWorkspaceSize）返回了 0、缓冲从未分配。

**第 8 步（可选）：对照 kernel 配置与变体**：

```bash
cat $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/<op>.json
```

看 `simplifiedKey`（dtype/bool 参数组合 → 变体映射）与 `kernelList`（变体名），把 fault kernel 的 `_N` 后缀对回参数组合，确认崩的是哪条 TilingKey 路径；带可选输出的算子还可核对 `optionalOutputMode` 等框架行为开关。

## 原始观测（两条 trace 的 plog 关键行）

```
# 设备日志（bitmap）
... PrintAicErrReportInfo ... Bit map is : aicmap[0..4] : 00000000,...,
aivmap[5] : 00111110, aivmap[6] : 11111111, aivmap[7] : 11111111,
aivmap[8] : 00111111, aivmap[9] : 11000000, ...
# 主 plog（任务失败 + 同步错误码）
[ERROR] ... PrintTaskErrorMsg:Task run failed, stream_id=61, pos=0, task_sn=17,
sqe_type=1(aiv), errType=0x1(task exception), sqSwStatus=0.
[ERROR] ... PreCheckTaskErr:An error occurred in the kernel task, retCode=0x31, [vector core exception].
[WARNING] ... TaskFailCallBack: ... retCode=0x715005e,[vector core exception]
[ERROR] ... rtStreamSynchronize:ErrCode=507035, desc=[vector core exception], InnerCode=0x715005e
# 指令级现场
The error from device(chipId:0, dieId:0), ... there is an aivec error exception, core id is 0,
error code = 95, dump info: pc start: 0x120041000000, current: 0x1200410000c8, ...
para base: 0x120000200000, mte error: 0x80000000, ...
The extend info: errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4.
# fault kernel + tilingKey + args 指针表
fault kernel_name=BenchNoHarnessUnique_5b5227062859d6097351dcb83f899f41_3
GetArgsInfo:tilingKey = 0, print 1 Times totalLen=(13*8), argsSize=104, schemMode=0, blockDim=1
args(0 to 12) after execute:0x120000017000, 0x120000018000, 0x120000019000, ...
# DevMalloc 对账（workspace 未分配的实锤）
[INFO] ... DevMalloc: size=64, type=1024, moduleId=33.        ← 只有 tiling data
```

## 根因（解释）

507035 = 向量核（AIV）任务异常在 `aclrtSynchronizeStream` 的暴露码（InnerCode=0x715005e，retCode=0x31）。bitmap 告诉你哪些核炸；`sqe_type=1(aiv)` 告诉你是向量类任务；errcode 95/264/271 分别把范围缩到 MTE GM 越界 / 标量 GM 无效 / 标量 UB 越界。但 plog **不告诉你哪条指令、哪个指针**——必须靠跳层二分收窄到最小代码单元，再用"读试验"区分"指针无效"与"访问方式非法"，最后用 DevMalloc 尺寸对账判定缓冲是否真的分配过。本流程闭环实例：写 workspace GM 的任何读/写都触发 → 指针无效 → DevMalloc 无 880B 记录 → workspaceSize=0、nullptr 传入（见同批 `aclnn_l2_tiling_workspace_gap_507035`）。

## 设计决策

- 为什么优先 plog 文件而不是只看 stdout：stdout 的 DFX 只覆盖 host 侧（op_api 检查、框架 op 注册），设备侧 errcode/fault kernel/args 只落 plog；两者互补，不可互相替代。
- 为什么 grep 键用 `errcode|errorStr|fault kernel_name|tilingKey`：plog 里 ATRACE/INFO 行占绝对多数，宽泛的 `ERROR|FATAL` 初筛可行但噪声大；错误详情四键直接命中崩溃记录块。
- 跳层二分要求"被短路层的输出不影响判据"——若短路本身改变语义（如去掉同步），结论只对被试路径有效。
- DevMalloc 对账只能证明"没分配"，不能证明"为什么没分配"（上层返回 0 的原因需另行追查）。

## 系统信息

- 硬件/软件：Ascend950PR（arch3510），CANN 9.1.0 与 9.2.0；plog 默认路径 `~/ascend/log/debug/plog/plog-<pid>_<ts>.log`，设备位图 `~/ascend/log/debug/device-0/*.log`；测试脚本重定向过日志目录时按实际重定向位置找。
- 相关文件：ST 测试二进制（`tests/st/build-real/test_aclnn_<op>` 等）；kernel config `opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/*.json`。

## 预防

- 联调脚本里固定导出两个日志环境变量，崩溃后 plog 已就位，不必复现。
- 自定义算子仓保持 issues/ 目录记录 plog 结论（本流程产出的定位证据应随 issue 归档）。

## 验证信号

来源说明：以下信号来自两条独立 trace（unique 算子迭代一联调、softmax 算子 Group 分支 507035 定位会话）用本配方实际拿到的证据，多次重复使用均有效：

- stdout DFX 定位：`[ERROR] OP(...): [aclnn_bench_no_harness_unique.cpp:49][NNOP][CheckNotNull] errno[161001] ...`（来源平台：950；证据等级：tool_observed）。
- plog errcode 定位：`errcode:(95) The DDR address of the MTE instruction is out of range. subErrType: 0x4`、`errcode:(271) The address for scalar to access the internal buffer is out of bounds`（来源平台：950；证据等级：tool_observed）。
- fault kernel + tilingKey 定位：`fault kernel_name=BenchNoHarnessUnique_5b5227062859d6097351dcb83f899f41_3`、`GetArgsInfo:tilingKey = 0, ... blockDim=1`（来源平台：950；证据等级：tool_observed）。
- args 指针表定位：`args(0 to 12) after execute:0x120000017000, 0x120000018000, 0x120000019000, ...`——据此发现可选输出被塞占位地址 `0x120000019000`（来源平台：950；证据等级：tool_observed）。
- 指令级现场：`ProcessDavidStarsCoreErrorInfo: ... error code = 95, pc start: 0x120041000000, current: 0x1200410000c8, mte error info: 0x20028, mte error: 0x80000000, vec error info: ...`（来源平台：950；证据等级：tool_observed）。
- bitmap/同步错误码定性（另一 trace）：aivmap[5..9] 多核非零、`PreCheckTaskErr retCode=0x31 [vector core exception]`、`rtStreamSynchronize:ErrCode=507035, InnerCode=0x715005e`（来源平台：950；证据等级：tool_observed）。
- 跳层二分三轮（失败/失败/全 PASS）逐轮可复现；DevMalloc 清单仅 size=64，修复（改绑有效缓冲）后同批用例全 PASS（来源平台：950；证据等级：tool_observed）。
- errcode 264 `The address for scalar to access GM is invalid` 在同批 TopK/unique trace 的 workspace 越界场景实测出现（来源平台：950；证据等级：tool_observed）。

## 失效边界

- plog 文本只能给"哪条指令类出错 + 入参表"，不能给变量级现场；需要更强定位时换指令 pc 偏移反查 cubin 或二分 kernel（空 kernel 二分作为补充手段）。
- errcode 解释表覆盖本批实测的 95/264/271；其他 errcode 需查 CANN 错误码头文件或 plog 上下文自行解释。
- `ASCEND_GLOBAL_LOG_LEVEL=1` 以上才有足够细节；bitmap 前几轮可能全零（用例还在初始化），要看失败用例执行时刻附近的 bitmap。
- 个别 507035 场景 plog 无错误标记（同批另有 trace 实测 rg 搜 `507035|aic error|vector core|exception` 无命中）——此时跳层二分与指针有效性试验是唯一手段，不要指望 plog 必有信号。

**未决项**：

- errcode→机制的完整映射表未整理（仅 95/264/271 实测）；后续轨迹遇到新 errcode 应回填本卡。
- 指令级定位手段（错误 PC 反查 cubin）在本批 trace 未启用；若跳层二分代价过高（构建慢），值得补充。
## 坏实践

- 只看 `sync stream: 507035` 就开始改 kernel 代码——507035 不含任何细节，直接改代码等于盲改（多条 trace 的误诊均由此而来）。
- 用宽泛 `cat` 全量读 plog——文件数百 KB 起，被 ATRACE 淹没。
- 只看 `result=0x0` 的同步记录就认为没问题——失败记录是 `ErrCode=507035` 行，要用非零过滤。
- 在没有 plog 定性的情况下按经验猜根因（对齐/流水线/同步）直接改代码——本流程存在的原因。

## 误诊

- `StreamSynchronize` 的常规 INFO 行很多，第一次 grep 容易淹没；应过滤 `result=0x` 非零或直接搜 `ErrCode=507035`。

## 增量（507015 MIX 形态：核型判别以 aicmap/aivmap 位图为唯一可靠依据）

另一独立 trace（conv3d 反传 filter 算子，arch35 MIX：AIC Cube + AIV Vector，Ascend950PR + CANN 9.1.0）补 507015（aicore exception）形态下的核型判读规则与一个高代价误诊：

**规则**：`ret=507015` 的 "aicore exception" 报文与 `l1 error info` 字段标签**都不区分核型**；稳定指示故障核型的是 device log 里的位图行：

```bash
# 重定向 plog 后，在 device log 找 "Bit map is" 行——核型判别的唯一可靠信号
grep -aE "Bit map is" /tmp/plog_iter1/debug/device-0/device-*.log | head -1
```

原始观测（崩溃轮次稳定复现）：

```text
[INFO] CCECPU(2011,aicpu_scheduler):... [aicpusd_event_manager.cpp:558][PrintAicErrReportInfo] Bit map is : aicmap[0] : 00000000,aicmap[1] : 00000000,aicmap[2] : 00000000,aicmap[3] : 00000000,aicmap[4] : 00000000,aivmap[5] : 01111101,aivmap[6] : 11011111,aivmap[7] : 11101100,aivmap[8] : 00011011,aivmap[9] : 00000000,
```

解读：`aicmap[0..4]` 全零 + `aivmap[5..9]` 非零 → 故障核是 AIV（向量核），尽管外层报文标签写 `Info is aic error msg` / `aicore exception`（"aic error msg" 是该上报通道的通用命名，不代表 AIC）；反之 aicmap 非零才是 AIC（Cube 核）故障。host plog 里 `there is an aivec error exception`（aivec 字样）与 `vec error info: 0x...` 字段同轮出现时可互证；但 **`l1 error info` 非零 + "aicore error exception" 的组合曾在 AIV 故障轮次出现**（`l1 error info: 0xdffd0017fefe`），不能作为核型依据。

**误诊实录（本增量的核心教训）**：初始诊断把 `l1 error info` 非零 + "aicore error exception" 读作"AIC L1 错误、Matmul 缓冲越界"，据此排查 `SetBufferSpace(-1,-1,-1)` 与 baseM/baseN/baseK 的 L0 占用，全部排除后回看位图才发现故障一直在 AIV——该误诊消耗了整轮排查。凡核型判断一律以位图为准，且位图判读放进排查清单第一位（先于读错误码描述、先于看 l1 error info）；报告/issue 里记录崩溃时附上位图行原文，避免下游读报告的人被标签误导。

**位图语义的行为实验佐证**：单核最小化实验（只让 1 个 AIV 执行写操作）时位图收缩为 `aivmap[5]:00000001`，与实验设计一致——位图是逐核 bit 级事实，其余文本是通道级汇总标签。

验证信号（独立 trace 增补，conv3d 反传 filter 算子 507015 多轮排查，Ascend950PR / CANN 9.1.0）：

- host plog：`rtStreamSynchronize:ErrCode=507015, desc=[aicore exception], InnerCode=0x7150026`；`ProcessDavidStarsCoreErrorInfo` 含 `there is an aivec error exception, core id is 48 ... vec error info: 0x410315840039c2e6, cube error info: 0, l1 error info: 0`（来源平台：950；证据等级：tool_observed）。
- device log：`Bit map is : aicmap[0..4]=00000000, aivmap[5..9] 非零` 多轮稳定复现；单核最小化实验收缩为 `aivmap[5]:00000001`（来源平台：950；证据等级：tool_observed）。
- 位图结论与二分一致：只保留 AIV 写 workspace 的最小 kernel 即可复现 507015/位图非零；空 AIV 工作时无位图无报错（来源平台：950；证据等级：tool_observed）。
- 反例观测：`l1 error info: 0xdffd0017fefe` 出现在 AIV 故障轮次（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：

- 位图语义（aicmap/aivmap 分段、bit 与核号映射）仅从行为实验归纳，未查到官方字段文档；跨版本可能变化。
- "l1 error info 非零必与 AIC 相关"的反例只在 CANN 9.1.0 本算子场景观测到一次，其来源机制未解释。
- 非 MIX（单核型）芯片上位图意义退化；plog 重定向到 `/tmp`（`ASCEND_PROCESS_LOG_PATH`）每次运行隔离，避免串轮次。