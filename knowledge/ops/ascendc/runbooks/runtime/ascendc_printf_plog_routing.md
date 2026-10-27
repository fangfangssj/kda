---
type: "Runbook"
title: "AscendC PRINTF 路由：TTK 查 plog 且防流水线扰动"
description: "kernel 内 AscendC::PRINTF 的输出直连 aclnn 进程走 stdout、TTK 评测下只进 plog（需 ASCEND_GLOBAL_LOG_LEVEL=1）；且留在 MTE 紧同步循环里的 PRINTF 会破坏 RMW 正确性，定位完必须移除。"
tags: [task.debugging, topic.architecture]
status: stable
sources: []
created_at: '2026-09-14T02:36:53Z'
updated_at: '2026-09-14T02:36:53Z'
platforms: ["950"]
---

# AscendC PRINTF 路由：TTK 查 plog 且防流水线扰动
## 现象

在 Ascend950PR（CANN 9.2.0）上调试自定义算子 BenchHarnessScatter 的 fp32 workspace 路径时，为确认 kernel 内分支是否执行，在 kernel 里加了 `AscendC::PRINTF("FP32PATH useFp32Acc=%d dataNumel=%d", ...)`。两件事先后发生：

1. 直连 aclnn 的自建诊断程序里，PRINTF 输出直接出现在进程 stdout（形如 `[AIV Block 0/1] FP32PATH enter useFp32Acc=1 ...`），可用管道 grep；但同一 kernel 在 TTK（ops-test-kit，`python3 -m ttk aclnn ...`）下运行时 stdout 里完全看不到 PRINTF 输出——一度据此误判"新代码路径没有执行"，进而怀疑内核缓存过期，浪费了多轮清缓存、核对 vendor 预编译 .o 时间戳的排查。
2. 排查期间为缩小崩溃范围，在 Phase A 的 MTE2→S→MTE3 循环内每步插了 PRINTF（"PA after MTE2 read / PA after S cast / PA after MTE3 write"）。这些调试语句留在代码里时，bf16 add 2M 用例出现 mism=1100275/2000000、lostUpd=1089973 的大规模"丢失更新"；把 PRINTF 全部移除并重建后，同一用例 mism=0/2000000。

## 适用场景

- 工程形态：AscendC kernel（`__aicore__` 侧）运行路径/中间值调试，覆盖两种运行方式——直连 aclnn 的独立 C++ 程序、TTK aclnn 黑盒评测。
- API 路径：`AscendC::Printf` / `AscendC::PRINTF`（kernel 侧打印）；环境变量 `ASCEND_GLOBAL_LOG_LEVEL`；plog 目录 `~/ascend/log/debug/plog/`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该平台所有 AscendC kernel 的打印调试，与算子无关。
- 路线优先级：recommended_fastpath（调试手法本身）。
- 明确不适用 / 未验证场景：host 侧（C++）打印不受影响；msprof 等其他采集通道未验证；PRINTF 扰动的确切机制（流水线插入 vs 编译器重排）未验证，只建立了"存在→出错、移除→恢复"的相关性。

## 判别方法

**触发**：需要在 kernel 内确认"某分支是否执行 / 某中间值是多少"，且运行入口可能是 TTK 等 harness。

**探针**（完整可复用链路）：

1. kernel 内加打印（`AscendC::PRINTF` 宏，前缀锚点唯一可 grep；下面是本轨迹 kernel `bench_harness_scatter_kernel.h` 中产生 plog 证据的真实调用点逐字引用——后续为缩小崩溃范围还用过硬编码 `useFp32Acc=1` 的变体格式串，同为有效锚点）：

```cpp
    if (td_->useFp32Acc) {
        // fp16/bf16 add/multiply: fp32 workspace accumulation path.
        // Phase A (all cores): data → acc (fp32), bulk vector Cast.
        AscendC::PRINTF("FP32PATH enter useFp32Acc=%d dataNumel=%d numCores=%d totalIdx=%d cid=%d",
            (int)td_->useFp32Acc, (int)td_->dataNumel, (int)td_->multicore.numCores,
            (int)td_->totalIndices, (int)AscendC::GetBlockIdx());
        CopyDataToAccFp32();
```

2. 直连 aclnn 运行：输出进进程 stdout，直接 `./repro | grep FP32PATH`。

3. TTK 下运行：stdout 不会有 kernel PRINTF；先清旧日志再开全局日志级别跑，然后到 plog 目录 grep：

```bash
rm -f /home/<user>/ascend/log/debug/plog/*.log 2>/dev/null
source $ASCEND_HOME_PATH/set_env.sh && export ASCEND_GLOBAL_LOG_LEVEL=1 && \
cd $TTK && timeout 600 python3 -m ttk aclnn -i <cases.csv> --pc=1 --task-prof false --run 1 \
    --warmup false --device-whitelist 0 --proc-timeout 300 --plugin <golden.py> -o <out.csv>
echo "=== plog grep ==="
grep -l "FP32PATH" /home/<user>/ascend/log/debug/plog/*.log
grep -oE "FP32PATH useFp32Acc=[0-9]+" /home/<user>/ascend/log/debug/plog/*.log | head
```

4. 打印 64 位地址等大值时，`%lx` 单次输出不可靠，拆高低 32 位分别打印。

**观测**（本会话实测）：
- 直连运行：`[AIV Block 0/1] FP32PATH enter useFp32Acc=1 dataNumel=8 ...` 出现在 bash 工具回显（stdout）。
- TTK 运行（ASCEND_GLOBAL_LOG_LEVEL=1）：`plog-<pid>_<ts>.log` 中 `FP32PATH useFp32Acc=1` 出现 8 次（8 核各一次），stdout 无输出。
- 副作用：Phase A 循环内逐步 PRINTF 时 bf16 add 2M lostUpd=1089973/2000000；移除后同用例 mism=0/2000000。

**解释**：kernel 侧 PRINTF 的输出通道取决于启动上下文——直连进程可见于 stdout，TTK 封装下只落 plog（plog 由全局日志级别控制写入 `~/ascend/log/debug/plog/`）。"stdout 没有打印"不等于"分支没执行"。同时，PRINTF 编译为内核指令插进 MTE2→S→MTE3 紧同步循环后，会扰动该循环内逐元素 GM RMW 的正确性（机制未验证；观测为移除即恢复），因此定位完成后必须全部移除调试打印再回归。

## 设计决策

- 选择"plog grep 确认路径"而不是"改 tiling/加返回码"来验证分支执行：不改任何被测代码语义，一轮 TTK 运行即可回答；且 plog 文件名含 pid/时间戳，可与运行一一对应。
- 调试打印全部加显式前缀（如 `FP32PATH`、`PA `），grep 锚点唯一，避免与框架日志混淆。
- 当前方案路线档位：recommended_fastpath（调试期标准做法）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；TTK=/tmp/ops-test-kit（aclnn 模式）；plog 目录 `/home/<user>/ascend/log/debug/plog/`。
- 相关文件：`op_kernel/arch35/bench_harness_scatter_kernel.h`（临时调试打印的插入与移除）、`tests/aclnn/repro_min1k.cpp`（直连运行对照）。

## 预防

- TTK 下验证 kernel 分支前，先记住路由规则：直连 stdout / TTK plog（`ASCEND_GLOBAL_LOG_LEVEL=1` + `~/ascend/log/debug/plog/`）；不要用"stdout 没有"下"路径未执行"的结论。
- 调试打印用可 grep 的唯一前缀；64 位地址拆 32 位打印。
- **任何留在 MTE 紧同步循环（逐元素 RMW、SetFlag/WaitFlag 密集区）内的 PRINTF 都必须在回归前移除**，并以"移除后重建 + 全量用例复跑"作为收尾步骤；交付前 grep 确认 kernel 源码无 PRINTF 残留（本会话收尾即 `grep -c PRINTF` = 0 后才终验）。

## 验证信号

以下信号来自该算子 fp32 路径调试会话的直连运行与 TTK 运行对照：

- 直连 aclnn 运行：kernel PRINTF 输出（`[AIV Block 0/1] FP32PATH enter ...` 等）出现在进程 stdout（来源平台：950；证据等级：tool_observed）。
- TTK 运行：stdout 无 PRINTF；`ASCEND_GLOBAL_LOG_LEVEL=1` 时 `~/ascend/log/debug/plog/plog-<pid>_*.log` 中 grep 到 `FP32PATH useFp32Acc=1` 共 8 次（对应 8 核）（来源平台：950；证据等级：tool_observed）。
- 副作用观测：Phase A 循环内逐步 PRINTF 存在时 bf16 add 2M 用例 mism=1100275/2000000、lostUpd=1089973；全部移除并重建后同用例 mism=0/2000000、maxRel=0（来源平台：950；证据等级：tool_observed）。
- 收尾核验：交付前 kernel 源码 grep PRINTF 计数为 0，最终重建 SUCCESS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 路由行为在 950 + CANN 9.2.0 + 该版本 TTK 实测；TTK 版本升级后 stdout/plog 行为可能变化（未验证）。
- "PRINTF 扰动 RMW 正确性"是相关性结论（存在→出错、移除→恢复），未定位到指令级机制，不排除与具体同步模式（逐元素 MTE2_S/S_MTE3/MTE3_MTE2）绑定；松散循环内插 PRINTF 未见异常（未系统验证）。
- plog 写入依赖 `ASCEND_GLOBAL_LOG_LEVEL=1`；级别不足时 TTK 运行可能不落 plog（本会话观察到默认级别下 TTK 不生成新 plog）。

**未决项**：

- 优先项：PRINTF 扰动逐元素 RMW 的指令级机制（插入的打印指令破坏了哪条同步/可见性约束）未验证，复现时可对照反汇编/流水线事件确认。
- TTK 各日志级别下 plog 的写入条件未系统梳理。
## 坏实践

- 在 TTK 下 grep stdout 找 kernel PRINTF，找不到就断言"分支没执行/二进制过期"，转去清内核缓存——本会话实际走过这段弯路（清 `/tmp/ops-test-kit/kernel_meta`、核对 vendor .o 时间戳均无效，plog 一查路径早已在跑）。
- 定位完成后把调试 PRINTF 留在 MTE 紧同步循环里直接跑精度回归——扰动会让"好代码"报大面积丢写。

## 误诊

- "新 kernel 没生效（mare 与旧版逐位相同）⇒ 缓存过期"：plog 证实新路径在运行，指标不变另有根因（见 [FloatToHalf 次正规阈值漏 exp102 卡](../precision/floattohalf_subnormal_threshold_exp102.md)）。
