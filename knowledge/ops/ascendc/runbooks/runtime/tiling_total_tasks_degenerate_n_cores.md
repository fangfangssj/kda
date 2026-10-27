---
type: "Runbook"
title: 大C_out用例超时：tiling切分退化仅N核并行
description: Tiling 多核切分按 (N, cOutTileCount) 计算 totalTasks，C_out 为 16 倍数时 nTile==COut、totalTasks 退化为 N，仅 N 个核工作；而 kernel 按 (N, C_out) pair 均分任务，大 C_out 用例单核需执行上百亿次 GetValue 标量 GM 读取，表现为 aclrtSynchronizeStream 不返回、测试进程被超时 SIGKILL（137）。
tags: [task.debugging, topic.tiling, topic.parallelism]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T00:54:40Z'
platforms: ["950"]
---

# 大C_out用例超时：tiling切分退化仅N核并行

## 现象

在 Ascend950PR（CANN 9.1.0，arch35/DAV_3510）上跑自定义算子 BenchNoHarnessConv2d（朴素直接卷积：`GlobalTensor::GetValue` 逐元素标量读 + 浮点累加 + `DataCopyPad` 写回）的 ST Real 全量测试：小程序用例全部 PASS，但运行到大规模用例 L0-032（N=8, C_in=512, H=W=64, C_out=512, K=3×3, fp16，输出 16,777,216 元素）时测试进程被 SIGKILL 杀死，退出码 137，日志停在该用例 shape 打印之后；单独复跑 L0-032 同样被杀。在测试程序各执行阶段插入 stderr 调试打印后确认：`aclnnBenchNoHarnessConv2d ret=0`（两段式接口下发成功）打印之后没有 `aclrtSynchronizeStream ret=` 打印——进程阻塞在流同步上等待 kernel 完成，300 秒超时仍不返回。也就是说 kernel 没有崩，是在 NPU 上执行极慢（准 hang）。

当时 tiling（`op_host/arch35/bench_no_harness_conv_2d_tiling.cpp`）的多核切分与 kernel（`op_kernel/arch35/bench_no_harness_conv_2d.h` 的 `Process`）的任务划分口径不一致：

```cpp
// tiling：按 (N, c_out_tile) 声明核数
int64_t nTile = CeilAlign(COut, int64_t(16)); // N = ceil(C_out / 16) × 16
int64_t cOutTileCount = CeilDiv(COut, nTile);
int64_t totalTasks = N * cOutTileCount;
int64_t usedCoreNum = (coreNum < totalTasks) ? coreNum : totalTasks;
int64_t blockFactor = CeilDiv(totalTasks, usedCoreNum);
```

```cpp
// kernel Process：按 (N, C_out) pair 均分到 usedCoreNum 个核
int64_t totalPairs = N_ * COut_;
int64_t perCore = (totalPairs + usedCoreNum_ - 1) / usedCoreNum_;
int64_t myStart = blockIdx_ * perCore;
int64_t myEnd = (myStart + perCore > totalPairs) ? totalPairs : (myStart + perCore);
```

## 适用场景

- 工程形态：AscendC 自定义算子（算子仓工程，host TilingFunc 声明多核参数 + kernel 侧按 pair 均分任务），aclnn 两段式接口 ST Real 模式验证。
- API 路径：`gert::TilingContext` 的 tiling 字段（usedCoreNum/blockFactor）、kernel 侧 `AscendC::GetBlockIdx()` 任务划分、`GlobalTensor::GetValue` 标量 GM 读取路径。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何 host tiling 与 kernel 侧任务切分口径不一致的算子；对"kernel 按 (N,C_out) pair 切分、tiling 按 (N,c_out_tile) 声明核数"的直卷/逐 pair 类 kernel 直接适用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：Cube/Load3D 路线（切分由 MMAD tiling 决定）；host 与 kernel 切分本来就一致的算子无此问题；其他 NPU 架构未验证（机制推断可迁移）。

## 根因

tiling 的 `totalTasks` 按 (N × cOutTileCount) 计算，当 `nTile = CeilAlign(COut, 16)` 恰好等于 COut（C_out 本身是 16 的倍数，如 512）时 `cOutTileCount = 1`、`totalTasks` 退化为 N——L0-032 只有 8 个核拿到任务；而 kernel 按 `totalPairs = N × COut = 4096` 个 (n, co) pair 用 `usedCoreNum = 8` 均分，`perCore = 512` 个 pair/核。

机制链（数值由用例 shape 与两侧代码推导）：

- 每 pair 有 HOut×WOut = 64×64 = 4096 个输出元素；
- 每个输出元素需 CIn×KH×KW = 512×3×3 = 4608 次乘加，x 与 filter 各一次 `GetValue` 标量 GM 读；
- 单核总读取次数：512 pair × 4096 元素 × 4608 × 2 ≈ 19.3G 次；
- 标量 GM 读延迟高（每次数百拍量级），19.3G 次读取远超执行时限 → `aclrtSynchronizeStream` 长期不返回 → 外层执行工具超时 SIGKILL（137）。

该切分逻辑自项目早期即存在（git diff 证实前后两轮相同）；早期验收记录显示 L0-032 曾通过但无耗时记录，说明该用例本来就处在超时边缘。

## 解决配方

修复前（tiling，`totalTasks` 退化为 N）：

```cpp
    // 7. 多核切分：按 (N, C_out_tile) 二维任务网格
    int64_t cOutTileCount = CeilDiv(COut, nTile);
    int64_t totalTasks = N * cOutTileCount;
    int64_t usedCoreNum = (coreNum < totalTasks) ? coreNum : totalTasks;
    int64_t blockFactor = CeilDiv(totalTasks, usedCoreNum);
```

修复后（totalTasks 与 kernel 的 pair 切分口径一致）：

```cpp
    // 7. 多核切分：按 (N, C_out) pair 分配给各核
    // Kernel 中按 (N, COut) pair 切分，Tiling 中 totalTasks 需与之一致
    // 注意：不能按 (N, cOutTileCount) 切分，否则当 nTile == COut 时 totalTasks = N，
    //       大量核闲置，大 shape 用例执行时间极长
    int64_t totalTasks = N * COut;
    int64_t usedCoreNum = (coreNum < totalTasks) ? coreNum : totalTasks;
    int64_t blockFactor = CeilDiv(totalTasks, usedCoreNum);
```

kernel 侧不用改（pair 均分本身均匀合理）。修改后重编译安装算子包（`bash build.sh --soc=ascend950 -j8 --make_clean` 后需再执行一次不带 `--make_clean` 的编译，`--make_clean` 只清理即退出；再 `bash build/custom_opp_ubuntu_x86_64.run --force` 安装），重编译 ST 后复跑。

判别 / 定位方法（本会话走通的链路）：

1. 先排除 host OOM 与进程崩溃：137 = 128+9（SIGKILL），查 dmesg OOM、`/proc/meminfo`、cgroup 用量与容器总 RSS；本例全部正常。
2. 在测试程序各阶段插 `fprintf(stderr, ...) + fflush(stderr)`，覆盖数据生成/量化/tensor 创建/GetWorkspaceSize/aclrtMalloc/aclnn 调用/aclrtSynchronizeStream。
3. 看日志最后一条打印：停在 `aclnnXxx ret=0` 之后、无 sync 返回打印 ⇒ host 下发完成、kernel 未执行完 ⇒ 转向 kernel 执行时间排查。
4. 对照 tiling 与 kernel 的切分公式，用用例 shape 代入算每核工作量（pair 数 × 每 pair 元素 × 每元素乘加）。

## 设计决策

- 为什么改 tiling 而不是改 kernel：kernel 的 (N, COut) pair 均分粒度均匀、实现正确；瓶颈在 tiling 声明的 `usedCoreNum` 被错误地压到 N，改一行公式即可让 `usedCoreNum = min(coreNum, N×COut)`。
- `nTile` 参数保留（tiling 数据中仍有用途），只是不再用它推导 totalTasks；注释中显式写明两端口径必须一致，防止回退。
- GetValue 标量路径本身极慢是已知的正确性优先取舍（本算子为 no-harness 基准的朴素直卷）；本修复只解决"核数没用满"的并行度坍缩，不改变算法路径。
- 当前方案路线档位：recommended_fastpath（正确性必需）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1，Health=OK），CANN 9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装根）；算子包安装于 `opp/vendors/bench_no_harness_conv_2d_custom/`。
- 相关文件：`op_host/arch35/bench_no_harness_conv_2d_tiling.cpp`（修复点）、`op_kernel/arch35/bench_no_harness_conv_2d.h`（kernel pair 切分，未改）、`tests/st/test_aclnn_bench_no_harness_conv_2d.cpp`（ST 程序与阶段插桩）、`tests/st/testcases/L0_test_cases.csv`（用例）。
- 复现命令：`cd tests/st && rm -rf build-real && mkdir build-real && cd build-real && cmake .. && make -j8 && ./test_aclnn_bench_no_harness_conv_2d ../testcases/L0_test_cases.csv`（大用例需给足超时预算）。

## 预防

- tiling 声明 usedCoreNum/blockFactor 时，必须核对 kernel 侧实际的任务划分口径——两侧公式写在同一处注释里互指。
- tiling 切分公式变更后，同步更新 UT 的 expectUsedCoreNum/expectBlockFactor 期望（本会话后续 UT 17 个参数化用例因期望失配而失败：expectUsedCoreNum=1 vs blockNum=32）。
- ST 大 shape 用例默认按"每核工作量 = pair 数 × 每 pair 元素 × 每元素乘加 × 2 次标量读"预估执行时间，超预算的用例单独给超时或分批跑。

## 验证信号

以下信号来自该算子联调验证会话：先以阶段插桩稳定复现 L0-032 阻塞在流同步，修正 tiling 切分后逐级放大验证。

- 修复前：L0 全量运行至 L0-032 被杀（EXIT_CODE=137，多次复现）；单跑 L0-032 在 300 秒超时下被工具终止，日志最后一条为 `[DEBUG] RunRealTest: aclnnBenchNoHarnessConv2d ret=0`、无 `aclrtSynchronizeStream ret=` 打印（来源平台：950；证据等级：tool_observed）。
- 修复前工作量推导：totalTasks=8（仅 8 核）、kernel perCore=512 pair、单核 ≈19.3G 次 GetValue 标量 GM 读（来源平台：950；证据等级：tool_observed——由两侧代码与用例 shape 代入推导）。
- 修复后：单独运行 L0-032，`aclrtSynchronizeStream ret=0`，PASS（MERE=1.76e-04, MARE=4.88e-04, threshold=9.77e-04, 16,777,216 元素）（来源平台：950；证据等级：tool_observed）。
- 修复后：L0 全量 50/50 PASS（fp16 43 + fp32 7）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修复只解决并行度坍缩；标量 GetValue 直卷的固有吞吐上限仍在——修复后 L1 大 spatial 用例（如 N=1, C=128, H=W=112 的 VGG block2，约 41 GFLOPs）在默认超时下仍被杀，需更长超时/分批运行才能完成（标量直卷路线的规模上限见 [GetValue 标量直卷规模上限](../performance/getvalue_direct_conv_feasibility_ceiling.md)）。
- `nTile == COut` 的退化只在 C_out 为 16 的倍数时完全成立；C_out 非 16 倍数时 cOutTileCount ≥ 2，症状较轻但同样存在核数浪费。
- 早期验收记录显示同一实现下 L0-032 曾通过（无耗时记录）——该用例本来就在超时边缘，通过与否受运行环境影响，不能作为"无问题"的证据。
- 其他 NPU 架构未验证（机制推断可迁移）。

**未决项**：

- 修复后 L0-032 的精确执行时长未单独计时（只确认在超时预算内完成）；建议下轮给大用例加计时输出。
- 标量直卷路径的可行规模上限（数十 GFLOPs 量级）与最大 shape 的 Cube 路线方案未落地，见同域负结果类结论。
## 坏实践

- tiling 声明 `usedCoreNum` 时不核对 kernel 侧任务划分口径，两边各写各的公式。
- 看到退出码 137 就按 OOM 方向排查内存（本例 cgroup/内存/RSS 全部正常，浪费大量排查时间；判别流程见"误诊"）。
- 用"子集用例通过"证明全量没问题——本例 17 条子集全过，但子集不含真正的大用例。

## 误诊

- OOM 理论：cgroup v1 limit 32GB、`/proc/meminfo` 总 754GB/可用 696GB、容器内全部进程总 RSS 仅 4.8GB、Python 试分配 5GB 成功、dmesg 无 OOM——全部排除；真实原因是执行工具超时杀掉了阻塞在 `aclrtSynchronizeStream` 的进程。
- 曾怀疑特定用例（L0-016 对称填充）触发：该用例子集复跑通过后排除。
- 曾怀疑 NPU 残留进程资源冲突（npu-smi 见 python3 占用 2.5GB HBM）：确认健康状态后排除。
