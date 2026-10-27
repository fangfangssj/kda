---
type: "Runbook"
title: MIX 循环内 SyncAll 挂死，去 Cube 化绕开
description: "按 batch 切分的 MIX 模式 kernel（taskRation 1:2）在时序循环里用 SyncAll 同步 Cube/Vector，ST 挂死超时；去 SyncAll 变 507015、配平计数仍挂、LSTM 式 VECCALC 输出仍崩——四条路线全部失败，该形态下 MIX 路线 banned，已验证出路为纯 AIV 重写。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# MIX 循环内 SyncAll 挂死，去 Cube 化绕开

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义 GRU 算子维护一个 MIX 模式 kernel（Matmul 做矩阵乘 + Vector 做门控计算，binary_info_config.json 显示 `coreType: MIX`、`intercoreSync: 1`，taskRation "1:2"——每 AIC 配 2 AIV；按 batch×方向切核，每核数据独立）。为让 AIV 等 AIC 的 Matmul 结果，在时间步循环里加了 `SyncAll()`：

```cpp
for (int64_t t = 0; t < seqLen_; t++) {
    mmGh_.SetTensorA(gmHPrev_);    // AIC
    mmGh_.IterateAll(gmGh_);       // AIC
    SyncAll();                      // Both
    // Vector 代码...               // AIV
    SyncAll();                      // Both
}
```

运行 C++ ST 测试（Real NPU 模式）后，输出停在"执行测试用例"后**再无任何输出**，直到 600 秒工具超时杀掉命令——kernel 挂死。换小用例（S=10,B=4）、加 `timeout 60/120 stdbuf -o0` 直跑测试二进制，同样在第一个用例内挂死。

## 适用场景

- 工程形态：AscendC 自定义算子 MIX 模式 kernel（Cube Matmul + Vector 混合，taskRation "1:2"），时序循环内需要 Cube↔Vector 同步。
- API 路径：`AscendC::SyncAll`；`matmul::Matmul`（IterateAll/Iterate+GetTensorC）；`PipeBarrier<PIPE_V>`；binary_info_config 的 `coreType`/`intercoreSync`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0。
- 适用粒度：operator_family——按 batch 切分、无跨核数据依赖的时序（RNN 类）kernel；其他切分形态（跨核归约等本就需要全核同步）不适用本卡结论。
- 路线优先级：banned（该场景下"MIX + 循环内 SyncAll"路线）；已验证替代路线为纯 AIV 重写（Mul+ReduceSum 替代 Matmul，见验证信号与相关卡）。
- 明确不适用 / 未验证场景：CANN 内置 GRU/LSTM 在 MIX+SyncAll 下可工作（其 core 数/tiling 配置与本例不同，差异未定位）；其他 taskRation、其他平台未验证；CrossCoreSetFlag/CrossCoreWaitFlag 等替代同步原语未尝试。

## 判别方法

**触发**：MIX kernel 加 SyncAll 后 ST 无输出直至超时；去掉 SyncAll 立即变成 507015（AIV 读到未就绪数据）。

**探针**：

1. 挂死确认：`timeout 120 stdbuf -o0 ./test_aclnn_bench_no_harness_gru /tmp/L0_small.csv 2>&1 | tail -15`——输出止于用例头信息，无 PASS/FAIL 行，超时被杀。
2. 对照实验：注释全部 SyncAll → 5/5 变为 `[FAIL] 流同步失败: 507015`（crash 而非挂死），证明挂死由 SyncAll 引起、且无同步时数据竞争崩溃。
3. 核对编译产物：`build/op_kernel/ascendc_kernels/binary/ascend950/<op>/<Kernel>_<hash>.json` 中 `coreType: MIX`、`intercoreSync: 1`（确认处于 MIX 通道）。

**观测（四条尝试路线全部失败）**：

1. 每时间步 2×SyncAll（if 分支）+ 空转 else 分支补齐计数 → 600s 超时挂死（来源平台：950；证据等级：tool_observed）。
2. 注释全部 SyncAll → `[FAIL] 流同步失败: 507015`（5/5），AIV 读到未就绪的 Matmul 结果（来源平台：950；证据等级：tool_observed）。
3. if/else 每步配平为 3×SyncAll → 仍在第一个用例挂死（来源平台：950；证据等级：tool_observed）。
4. 采纳 CANN 内置 LSTM（LstmMmMergeFP32Seq）模式：mmGh_ C 类型 `TPosition::GM→VECCALC`、`Iterate()+GetTensorC(ubLocal,false,true)+End()`、移除 SyncAll、`PipeBarrier<PIPE_V>()` 同步 → 仍 507015，plog `aivec error exception ... mte error info: 0x3a24` → 回退该改法（来源平台：950；证据等级：tool_observed）。

**解释**：在该 kernel 的切分形态（blockDim=usedCores，taskRation 1:2）下，MIX 通道的 SyncAll 参与方与 kernel 代码里的调用次数无法对齐（推测与 AIC/AIV 两侧代码裁剪或框架实际启动的核数有关，机制未验证），循环内 SyncAll 必然挂死；而没有任何跨核同步时 AIV 又必然抢跑。即：这一形态下 MIX 路线进退两难。

## 解决配方

本卡为 negative_result：**没有在 MIX 模式内找到可用修复**。已验证的出路是完全去 Cube 化——纯 AIV 重写（移除 Matmul API 与 REGIST_MATMUL_OBJ，Mul+ReduceSum 实现矩阵乘，`GetCoreNumAiv()` 切核），编译产物 `coreType: VectorCore / core_type: AIV / intercoreSync: 0`，配合系统 workspace 声明后 5/5 用例 PASS。完整配方见《RNN 门控算子纯 AIV 路线：Mul+ReduceSum 矩阵乘》卡。

## 设计决策

- 尝试顺序（SyncAll→去 SyncAll→配平计数→LSTM 式 VECCALC+PipeBarrier）从最小改动到参考官方实现逐级升级，均失败后才放弃 MIX；参考实现（CANN LSTM）的失败说明其模式不能直接平移到本切分形态。
- 最终选择纯 AIV：该算子 Matmul 规模小（M=1，N=3H，K=inputSize/H），Vector 实现代价可接受；且按 batch 切分天然无跨核依赖，MIX 的跨核同步是纯负担。
- 当前方案路线档位：banned（MIX+循环内 SyncAll，本场景）；替代路线 recommended_fastpath（纯 AIV）。

## 预防

- 在 taskRation 1:2、按 batch 独立切分的 kernel 里，不要引入循环内 SyncAll 做 Cube/Vector 同步——本开发会话 四条变体全部失败。
- 选 MIX 前先确认确实需要 Cube 算力且存在可行的跨核同步方案；小 M 的 RNN 型负载优先考虑纯 AIV。

## 验证信号

以下信号来自该算子 MIX 版与重写版的 ST 运行对照（Real NPU 模式）：

- MIX + 每步 2×SyncAll：测试无输出直至 600s 工具超时（挂死）（来源平台：950；证据等级：tool_observed）。
- MIX 无 SyncAll：5/5 `[FAIL] 流同步失败: 507015`（来源平台：950；证据等级：tool_observed）。
- MIX + 每步 3×SyncAll 配平：小用例仍在用例内挂死（来源平台：950；证据等级：tool_observed）。
- MIX + VECCALC/Iterate/GetTensorC/End（LSTM 式）：仍 507015，plog aivec error + mte error info 0x3a24（来源平台：950；证据等级：tool_observed）。
- 出路验证：纯 AIV 重写（coreType=VectorCore）+ 系统 workspace 修复后 5/5 PASS，y MERE 2.32e-07~6.01e-06（阈值 5e-02）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "MIX+循环内 SyncAll 必挂死"仅在：950 + CANN 9.2.0 + taskRation 1:2 + blockDim=usedCores 的按 batch 切分形态下验证；其他切分/核比/taskRation 未验证，不排除有可用配置。
- CANN 内置 LSTM 的 MIX+SyncAll 可工作，说明问题不在 API 本身而在配置/代码形态的匹配——具体差异（usedCoreNum 来源、核数、二进制通道）未定位。
- 未测试 CrossCoreSetFlag/CrossCoreWaitFlag 等替代同步原语（本开发会话 未尝试）。

**未决项**：

- 优先项：复现并追 MIX 通道 SyncAll 挂死的机制层根因（框架实际启动核数、AIC/AIV 二进制中 SyncAll 的裁剪与计数），给出可用/不可用的精确边界。
- CrossCoreSetFlag/CrossCoreWaitFlag、不同 taskRation 下的行为未尝试。
- 纯 AIV 版相对 MIX 版的性能代价未测量（本会话无 profiling 数据）。
## 坏实践

- 在 AIC/AIV 代码路径不对称（if/else 两分支、提前 return）的 MIX kernel 里数着 SyncAll 个数配平——本例配平后依旧挂死，浪费多轮构建。
- 挂死后不做去 SyncAll 对照实验就断言"数据竞争"或"框架 bug"，缺少二分证据。

## 误诊

- "else 分支 SyncAll 计数不齐导致挂死"：配平计数后仍挂死，排除单一计数假设（真实机制未定位）。
- "照搬 CANN LSTM 的 VECCALC+PipeBarrier 模式即可"：该模式在本切分形态下仍 507015——官方参考实现不能脱离其 tiling/核配置平移。
