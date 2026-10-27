---
type: "Runbook"
title: "950 mix(1,2) mode=2 漏 AIV1 flag 致全零"
description: "DAV_3510 (Ascend 950) 上 __mix__(1,2) 双 AIV kernel 若沿用 A2/A3 的 CrossCoreSetFlag mode=2，AIC 发出的 flag 只路由到 AIV0，AIV1 永不收到，AIC↔AIV1 同步链断裂致输出全零。判别要点是查 mode 模板参数与 AIV1 flagId+16 偏移。"
tags: ["task.debugging", "topic.precision", "topic.implementation.synchronization", "topic.programming_model.execution_unit.cube"]
status: stable
sources: []
created_at: "2026-08-18T08:55:35Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["FIA_SYNC_MODE4"]
---

# 950 mix(1,2) mode=2 漏 AIV1 flag 致全零

## 现象

在 Ascend 950DT_9582 (SocVersion ASCEND950, NpuArch DAV_3510, CANN 9.0.0, bisheng 编译器) 上开发 GQA (Grouped Query Attention) 算子，kernel 路线选 `__mix__(1, 2)`（1 个 AIC + 2 个 AIV，AIV0/AIV1 做 M-split）。AIC 负责 C1 (Q·K^T) 与 C2 (P·V) 两次 Mmad，AIV0/AIV1 负责 V1 (Scale+CausalMask+SoftmaxFlashV2) 与 V2 (Rescale+Accumulate)。AIC 与 AIV 是独立计算单元，必须通过 CrossCore flag 信号交替协调：

```
AIC:  C1 ──SetFlag──→  AIV: V1 ──SetFlag──→  AIC: C2 ──SetFlag──→  AIV: V2
       ↑ WaitFlag              ↑ WaitFlag
```

最初按 A2/A3 经验使用 `CrossCoreSetFlag<mode=2, ...>(flagId)` 做跨核同步。运行后输出全零：`[FAIL] MERE=1.000000e+00` / `[FAIL] MARE=1.000000e+00` / `[INFO] Max absolute error: 8.574219e-01`，失败点 `output=0.000000` vs `golden=0.857422`；generated_project 早期首跑同样全零，Host 侧 `aclrtSynchronizeStream returned: 507015`。该全零为多因混合（mode 选择 + 多核派发 + LoadData 等），其中 mode=2 在双 AIV 下的 flagId 路由缺陷是 `__mix__(1,2)` 路线必须先排除的一项。

## 判别方法

### 触发

DAV_3510 (950) 上 `__mix__(1,2)`（或更多 AIV）kernel 运行后输出全零（`MERE=1.0` / `MARE=1.0` / `max_abs_err=0.857`），或 Host 侧 `aclrtSynchronizeStream` 返回 `507015`。编译无报错、无运行时异常打印，但输出缺失/全零。触发信号集中出现于 950 + 双 AIV mix kernel 路线，单 AIV (`__mix__(1,1)`) 路线不触发本路由缺陷。

### 探针

1. 在 kernel 源码中检索跨核同步原语的 mode 模板参数：`CrossCoreSetFlag<...>` / `CrossCoreWaitFlag<...>` 的第一个模板参数是否为 `mode=2`（`FIA_SYNC_MODE2` / 字面量 `2`）。
2. 检查 AIC 侧访问 AIV1 的 SetFlag/WaitFlag 是否对 flagId 加了 `+16`（`AIV1_FLAG_OFFSET` / `AIV_SYNC_FLAG_DIST`）偏移——若 AIC 与 AIV0/AIV1 共用裸 flagId 且 mode=2，即命中本缺陷。
3. 确认目标 NpuArch 为 DAV_3510（950/Ascend 950PR/950DT）：mode=2 路由缺陷是 950 专属，A2/A3 (910B) 上 mode=2 行为不同（无 AIV1 独立计数器空间问题）。
4. 对照官方示例 `asc-devkit/examples/01_simd_cpp_api/05_compatibility_guide/matmul_s4_950/matmul_s4.asc`（950 上用 `CrossCoreSetFlag<4, ...>(8)` / `CrossCoreWaitFlag<4, ...>(8)`）——若本卡 kernel 与该示例在 mode 上不一致，优先疑 mode 选择。

### 观测

- 最终 generated_project 源码 `gqa_tiling.h` 的 flagId 常量与机制注释逐字记录了 mode=2 与 mode=4 的差异（artifact_verified）：`mode=2 (兼容 A2/A3) 在 A5 上有 AIV0/AIV1 flagId 偏移差异：AIC SetFlag<2>(8) → 只到 AIV0 (flagId 0-10)，AIV1 永不收到（需 16-26）；AIV1 SetFlag<2>(8) → AIC 在 flagId 16-26 上等待（实际 wait 8 永不触发）`，并引用 `CrossCoreSetFlag(ISASI).md` 表 2 与 `2201到3510架构变更.md:60`。
- 同源 `gqa_kernel.inc` Process 主循环同步段已切换为 mode=4 + AIC 侧双发/双收（`SetFlag(flagId)` 送 AIV0、`SetFlag(flagId + AIV1_FLAG_OFFSET)` 送 AIV1），AIV 侧用裸 flagId（artifact_verified）。
- MVP 阶段与 generated_project 早期首跑的 `[FAIL] MERE=1.0 / MARE=1.0 / max_abs_err=0.857` 全零、`aclrtSynchronizeStream 507015` 均为 tool_observed 的运行信号。

### 判别

DAV_3510 (950) 改变了 AIC↔AIV 的 flag 计数器寻址方式，与 A2/A3 不兼容。`mode=2` 下 AIC 的 `SetFlag` 只达 AIV0（flagId 0-10），AIV1 必须经 flagId+16 偏移才能寻址，但 mode=2 不提供该独立计数器空间，导致 AIV1 永不收到 AIC 的"C1 done / C2 done"信号，AIV1 的"V1 done"信号也永远唤不醒 AIC 的 wait → AIC↔AIV1 同步链断裂 → AIV1 不执行对应阶段或 AIC 阻塞 → 输出缺失/全零。`mode=4` 是 950/Ascend 950PR/950DT 专属：AIV0/AIV1 各有 16 个独立计数器（`AIV_SYNC_FLAG_DIST=16`），AIV0 的 flagId X 在 AIC 视角也是 X，AIV1 的 flagId X 在 AIC 视角是 X+16。因此判别结论：950 `__mix__(1,2)` 双 AIV CrossCore 同步必须用 mode=4，且 AIC 侧访问 AIV1 时对 flagId 加 `AIV1_FLAG_OFFSET=16`；AIV 侧用裸 flagId。需注意本 trace 的全零为多因混合，mode=2 双 AIV 路由缺陷是其中一项，判别时应同时排查多核派发与 LoadData 等并列根因。

## 适用场景

- 工程形态：kernel 直调（ops-direct-invoke），`extern "C" __global__ __mix__(AIC, AIV) void kernel(...)` 直调核函数。
- API 路径：`AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId)` / `AscendC::CrossCoreWaitFlag<MODE, PIPE>(flagId)`（ISASI 跨核同步原语）；配合 `__mix__(1,2)` kernel 类型与 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 编译期分支宏。
- 验证过的平台与 CANN 版本：Ascend 950DT_9582 / SocVersion ASCEND950 / NpuArch DAV_3510，CANN 9.0.0，bisheng 编译器（`/usr/local/Ascend/cann-9.0.0/bin/bisheng`），构建命令 `--soc=ascend950`。frontmatter `platforms` 严格只填本 trace 内实测的 `950`。
- 适用粒度：`operator_family` —— 适用于所有在 DAV_3510 (950) 上用 `__mix__(1,2)`（或更多 AIV）并依赖 CrossCore flag 做 AIC↔AIV 同步的 Cube+Vector 协同算子（FlashAttention 类、双 AIV M-split 类）。
- 路线优先级：`recommended_fastpath` —— mode=4 + AIV1 flagId+16 偏移是 950 上 `__mix__(1,2)` 双 AIV 同步的推荐路线。
- 不适用或未验证场景：
  - A2/A3 (910B) 平台应继续用 mode=2（mode=4 是 950 专属，A2/A3 无 AIV1 独立计数器空间）；本 trace 未在 A2/A3 实测，不得将 mode=4 迁移到 A2/A3。
  - `__mix__(1,1)` 单 AIV kernel 在 950 上用 mode=2 不触发本路由缺陷（只有一个 AIV，flagId 0-10 即可达），但本 trace 的 MVP 阶段 `__mix__(1,1)+mode=2` 仍因其他根因输出全零，mode 选择非其主因。
  - mode=4 多核高同步密度下的非确定性行为见 `## 失效边界`，未由本卡解决。

## 验证信号

- operators/gqa MVP 阶段运行输出全零：`[FAIL] MERE=1.000000e+00` / `[FAIL] MARE=1.000000e+00` / `[INFO] Max absolute error: 8.574219e-01`，失败点 `output=0.000000` vs `golden=0.857422`（来源平台：950；证据等级：tool_observed）。该全零为多因混合，mode=2 双 AIV 路由缺陷为 `__mix__(1,2)` 路线其中一项。
- generated_project 早期首跑同样全零（MERE=1.0/MARE=1.0/max_abs_err=0.857），Host 侧 `aclrtSynchronizeStream returned: 507015`（来源平台：950；证据等级：tool_observed）。
- mode=4 + `AIV1_FLAG_OFFSET` 同步结构与 flagId 常量落入最终 generated_project 源码 `gqa_kernel.inc` / `gqa_tiling.h`，机制注释逐字引用 `CrossCoreSetFlag(ISASI).md` 表 2、`2201到3510架构变更.md:60` 与官方示例 `matmul_s4_950/matmul_s4.asc`（来源平台：950；证据等级：artifact_verified）。
- 构建命令固定 `--soc=ascend950`、bisheng 编译器路径、环境检查 0 错误（来源平台：950；证据等级：artifact_verified）。

## 坏实践

- 在 950 `__mix__(1,2)` kernel 上沿用 A2/A3 的 mode=2 CrossCore 同步代码而不改 mode、不加 `AIV1_FLAG_OFFSET`——AIV1 静默丢失同步，输出全零但编译/无报错，极易误判为数据通路或 tiling 问题。
- 把"放宽数值容差/改 golden/删失败 case"当作 mode=2 路由缺陷的解法——本 trace 中全零为多因混合，需并列排查 mode 选择、多核派发与 LoadData，不得用测试侧放松掩盖实现侧缺陷。
- 把 mode=4 迁移到 A2/A3——mode=4 是 950 专属，A2/A3 无 AIV1 独立计数器空间，迁移会引入新的不兼容。

## 预防

- 在 950 (DAV_3510) 上写 `__mix__(1,2)` 或更多 AIV 的 kernel 时，CrossCore 同步默认用 mode=4，AIC 侧访问任一非首 AIV 一律加 `AIV_SYNC_FLAG_DIST=16` 偏移；AIV 侧用裸 flagId。
- 在 kernel 入口 `InitSocState()` 后立即 `PipeBarrier<PIPE_ALL>()`，确保所有 pipe 初始化完成再开始 wait。
- flagId 选择避开 `SyncAll`/`Matmul` 等高阶 API 占用的编号，mode=4 用独立计数器空间与之互不干扰。
- 出现 950 双 AIV mix kernel 全零时，优先 grep `CrossCoreSetFlag` 的 mode 模板参数，确认非 mode=2 后再查数据通路。

## 失效边界

- A2/A3 (910B) 不得使用 mode=4：mode=4 是 950/Ascend 950PR/950DT 专属；A2/A3 无 AIV1 独立计数器空间。A2/A3 上应继续用 mode=2。本卡知识未在 A2/A3 实测，未验证，推断不可迁移到 A2/A3。
- mode=4 多核高同步密度（`usedCoreNum=32` 且 `tasksPerCore × s2N` 较大）下的非确定性精度失败：单 case 单跑通过、全量批跑偶现 MARE 跳变；单核 (`usedCoreNum=1`) 稳定。该非确定性的根因未追到机制层——8-bit 计数器溢出假设被主会话自身撤回，AIV0/AIV1 UB 共享与 GM 竞争被排除，最终推测为 DAV_3510 硬件层 mode=4 多核高密度并发下的未文档化时序行为。本卡不解决该多核非确定性。
- `--npu-arch=dav-c310` 不被 bisheng 支持（报 "NPU arch is not supported"），CMakeLists 用 `dav-3510` 为正确编译参数。此为独立工具链 gotcha，非本卡根因。