---
type: "Runbook"
title: "共享 Tiling 仿真 caps 致公式 UT 大面积回归"
description: "kernel 联调阶段在共享 ComputeTiling 留下 6 个仿真 caps（coreNum≤8/ubFormer≤1024 等），与 TilingUT oracle 的纯公式不一致，21 条 UT 回归失败；删 caps 恢复 DESIGN 公式后 38/38 恢复，同时暴露 kernel 侧 UB 预算缺陷。"
tags: [task.debugging, topic.memory, topic.alignment]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 共享 Tiling 仿真 caps 致公式 UT 大面积回归

## 现象

bessel_i1e 算子走多阶段工作流开发：kernel 实现阶段为了让上板联调稳定，开发者曾在**共享** TilingFunc 的 `ComputeTiling`（`operators/BesselI1e_package/op_host/bessel_i1e_tiling_host.cpp`）里加了一组"仿真稳定 caps"（如 `coreNum≤8`、`ubFormer≤1024`、`blockFormer≤dim0`）。后续为 Branch-1（FP16）实现 tiling 并验收时，公式型 TilingUT 突然大面积回归：

```
[  FAILED  ] Sec52/TilingFormulaTest.ComputeTiling/1 ... /12   （12/13 FAILED）
[  FAILED  ] Branch0/Branch0TilingTest.ComputeBranch0Tiling/1 ... /9（9/10 FAILED）
...
21 FAILED TESTS
```

而同一可执行文件里 Branch-1 自己的 10 条用例（走独立实现的 `ComputeBranch1Tiling`）全部通过。Branch-1 tiling 任务的验收因此判 FAILED，并明确指出回归源自更早的 kernel 实现阶段引入、与 UT oracle（按 DESIGN §5.2/§2 公式手算的独立期望值）不一致的 caps。

## 适用场景

- 工程形态：算子仓工作流（host 侧 TilingFunc + gtest TilingUT，UT 在宿主机编译运行、参数按 Ascend950 典型值 ubSize=196608）。
- API 路径：`optiling::ComputeTiling`/`ComputeBranch0Tiling`/`ComputeBranch1Tiling`（op_host tiling 实现）+ `tests/ut/bessel_i1e_tiling_ut`。
- 验证过的平台与 CANN 版本：tiling 参数面向 Ascend950（AIV，ubSize=196608=192KB），CANN 9.0.0-beta.2；UT 本身在宿主机 x86 运行（算子包按 ascend950 构建安装，UB 容量常量为 950 典型值）。
- 适用粒度：operator_family——所有"多分支共享 TilingFunc + 公式型 UT"的算子工程。
- 路线优先级：recommended_fastpath（删 caps 恢复公式契约）。
- 明确不适用 / 未验证场景：UT oracle 本身与 DESIGN 公式不一致的情形（那是 oracle 错，不是实现错）；kernel 确实需要受限参数的场景——应改 kernel 侧消费方式或分支独立实现，而不是改共享公式。

## 根因

kernel 联调阶段为求上板稳定，在**共享** `ComputeTiling` 中残留了 6 个仿真 caps（`coreNum≤8`、`blockFormer≤dim0`、`safeUbSize≤256KB`、`ubFormer≤65535/elemBytes`、`ubFormer≤1024`、`ubFormer≤blockFormer`），使输出偏离 DESIGN §5.2 纯公式；TilingUT 的 oracle 按 DESIGN 公式独立手算，两边不一致导致 21 条用例回归。共享公式是 DESIGN 契约，联调便利不能以偏离契约为代价。

## 解决配方

修复 = 删除全部 6 个 caps、恢复纯公式实现（统一字节量纲）。修复前（带 caps）：

```cpp
    const int32_t elemBits = elemBytes * 8;

    // §5.2.1 多核切分
    int64_t rawCores = CeilDiv(static_cast<int64_t>(dim0) * static_cast<int64_t>(elemBits), MIN_TILING_BITS);
    int32_t coreNum = static_cast<int32_t>(std::min(rawCores, static_cast<int64_t>(availableCoreNum)));
    if (coreNum < 1) { coreNum = 1; }
    // Cap max cores for simulation stability
    coreNum = std::min(coreNum, 8);
    out.coreNum = coreNum;

    int64_t perCore     = CeilDiv(dim0, out.coreNum);
    out.blockFormer     = AlignUp(perCore, ELEM_ALIGN_FACTOR);
    // Ensure blockFormer does not exceed total dim0 (relevant for single-core)
    out.blockFormer     = std::min(out.blockFormer, dim0);
    out.blockNum        = CeilDiv(dim0, out.blockFormer);

    // Compute tail block size
    int64_t blockTail = dim0 - (out.blockNum - 1) * out.blockFormer;

    // §5.2.2 UB 切分
    // Use conservative ubSize cap
    const int64_t safeUbSize = std::min(ubSize, static_cast<int64_t>(256 * 1024));
    int64_t bufferDivisor = static_cast<int64_t>(bufferNum) * static_cast<int64_t>(elemBytes);
    int64_t maxElemNum    = safeUbSize / bufferDivisor;
    int64_t alignFactor   = UB_ALIGN_BYTES / static_cast<int64_t>(elemBytes);
    out.ubFormer = AlignDown(maxElemNum, alignFactor);
    if (out.ubFormer < 1) { out.ubFormer = alignFactor; }

    // DataCopy blockLen is uint16_t → count * elemBytes <= 65535
    const int64_t maxDcElements = 65535 / static_cast<int64_t>(elemBytes);
    out.ubFormer = std::min(out.ubFormer, maxDcElements);

    // Cap to UB headroom and ensure ubFormer ≤ blockFormer
    out.ubFormer = std::min(out.ubFormer, static_cast<int64_t>(1024));
    out.ubFormer = std::min(out.ubFormer, out.blockFormer);

    // 首 block 循环
    out.ubLoopOfFormerBlock = CeilDiv(out.blockFormer, out.ubFormer);
    out.ubTailOfFormerBlock = out.blockFormer - (out.ubLoopOfFormerBlock - 1) * out.ubFormer;

    // 尾 block (blockTail already computed above)
    out.ubLoopOfTailBlock = CeilDiv(blockTail, out.ubFormer);
    out.ubTailOfTailBlock = blockTail - (out.ubLoopOfTailBlock - 1) * out.ubFormer;
```

修复后（纯 §5.2 公式，无任何 caps）：

```cpp
    const int32_t elemBits = elemBytes * 8;

    // §5.2.1 多核切分
    int64_t rawCores = CeilDiv(dim0 * static_cast<int64_t>(elemBits), MIN_TILING_BITS);
    int32_t coreNum = static_cast<int32_t>(std::min(rawCores, static_cast<int64_t>(availableCoreNum)));
    if (coreNum < 1) { coreNum = 1; }
    out.coreNum = coreNum;

    int64_t perCore  = CeilDiv(dim0, out.coreNum);
    out.blockFormer  = AlignUp(perCore, ELEM_ALIGN_FACTOR);
    out.blockNum     = CeilDiv(dim0, out.blockFormer);

    // §5.2.2 UB 切分（统一字节量纲）
    int64_t bufferDivisor = static_cast<int64_t>(bufferNum) * static_cast<int64_t>(elemBytes);
    int64_t maxElemNum    = ubSize / bufferDivisor;
    int64_t alignFactor   = UB_ALIGN_BYTES / static_cast<int64_t>(elemBytes);
    out.ubFormer = AlignDown(maxElemNum, alignFactor);
    if (out.ubFormer < 1) { out.ubFormer = alignFactor; }

    // 首 block 循环
    out.ubLoopOfFormerBlock = CeilDiv(out.blockFormer, out.ubFormer);
    out.ubTailOfFormerBlock = out.blockFormer - (out.ubLoopOfFormerBlock - 1) * out.ubFormer;

    // 尾 block
    int64_t blockTail = dim0 - (out.blockNum - 1) * out.blockFormer;
    out.ubLoopOfTailBlock = CeilDiv(blockTail, out.ubFormer);
    out.ubTailOfTailBlock = blockTail - (out.ubLoopOfTailBlock - 1) * out.ubFormer;
```

配套验收门（7 条）：build 退出 0、无 `do not registe tiling struct` 告警、UT 可执行文件存在、UT 全量通过、oracle 独立（oracle 函数体内零 `optiling::` 调用）、回归不退化、字段单位一致。

## 设计决策

- **为什么删 caps 恢复公式而不是改 oracle**：TilingUT 的价值在 oracle 独立于被测实现（按 DESIGN 公式手算）；把 oracle 改成迁就 caps 等于销毁测试。共享公式是 DESIGN 契约，联调便利不能以偏离契约为代价。
- **kernel 确实需要受限参数时怎么办**：分支独立实现（Branch-1 的 `ComputeBranch1Tiling` 即为不共用带 caps 共享路径的独立纯公式实现，10/10 通过），或改 kernel 侧对 tiling 值的消费方式——总之不污染共享公式。
- **`maxDcElements=65535/elemBytes` 这个 cap 的特殊性**：它的动机是 `DataCopyParams.blockLen` 的 `uint16_t` 上限，但按"blockLen 以 32 字节块计数"的语义（见库内 DataCopy blockLen 语义卡），上限实为 65535 块 ≈ 2MB/次调用，该 cap 在正确语义下过度保守。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 上板联调需要受限 tiling 参数时，用"分支独立实现/本地变量"承载，绝不把仿真 caps 写进共享 TilingFunc。
- 联调结束后跑一遍全部前序 TilingUT 作为"遗留 caps"探测器：公式型 oracle 对任何偏离公式的 cap 都会报警。
- 工作流层面可在 kernel 任务与 tiling 任务间加"共享 host 代码 diff 审查"：kernel 开发者改 `op_host` 共享文件时验收必须核对与 DESIGN 公式一致。

## 验证信号

来源说明：以下信号来自 bessel_i1e Branch-1 tiling 的验收-修复-复验闭环（UT 运行于宿主机 gtest，参数按 Ascend950 典型值）。

- 修复前：`Sec52/TilingFormulaTest` 12/13 FAILED、`Branch0/Branch0TilingTest` 9/10 FAILED，合计 `21 FAILED TESTS`；Branch-1 子集 10/10 通过（独立实现不受影响）（来源平台：950；证据等级：tool_observed）。
- 修复后：`[  PASSED  ] 38 tests.`（Sec55/DtypeTilingKey 5 + Sec52/TilingFormula 13 + Branch0 10 + Branch1 10），连续两次独立运行一致（来源平台：950；证据等级：tool_observed）。
- 7 条验收门全过：build 退出 0、无 tiling 未注册告警、UT 可执行存在、38/38、oracle 独立（三个 oracle 函数体内 `optiling::` 调用计数均为 0）、前序回归不退化、字段单位一致（来源平台：950；证据等级：tool_observed）。
- 修复提交内容经 `git diff HEAD~1` 核对只含 caps 移除（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **删 caps 的直接后果**：FP32 下 `ubFormer` 从 1024 恢复到 `AlignDown(196608/(2*4), 64)=24576`。若 kernel 端按本 shard 的布局（2 个 TQue + 17 个 TBuf，每个 `ubFormer*sizeof(T)` 字节）申请 UB，合计 19×98304B≈1.87MB，远超 192KB——**去 caps 后上板 run.sh 立即出现 `ACL error 507035`（12/14 用例，仅空 tensor/零轴两条 PASS）**。该崩溃属另一根因（kernel 侧 UB 超额），其修复闭环在后续开发会话完成，不在本卡；本卡只覆盖"caps 回归 → 删 caps → UT 恢复"这一层。
- caps 与 kernel UB 预算是耦合的：删 caps 前必须核对 kernel 端 `InitBuffer` 总量 ≤ UB 容量（本开发会话 的教训即"公式正确性修复暴露了 kernel 侧预算缺陷"）。
- UT 在宿主机运行，不能发现上板才能暴露的 UB 超额类问题（未验证，属平台行为差异）。

**未决项**：

- 去 caps 后触发的 507035 向量核异常（kernel 侧 19 个 UB buffer 超 192KB）的修复方案与上板验证在后续开发会话，属跨会话闭环（优先项：按 `Σ(InitBuffer 尺寸) ≤ ubSize` 重排 kernel 侧 buffer 预算）。
- 工作流层面未建立"kernel 任务改 op_host 共享文件"的防回归门禁（仅在人工验收时发现）。
## 坏实践

- 为联调稳定在共享 TilingFunc 里硬编码 caps 且不恢复（`coreNum = std::min(coreNum, 8)` 一类"临时"改动留在共享路径）。
- 发现 UT 回归后优先怀疑测试/oracle 而不是共享实现被前序任务污染。
- 用 `ubFormer≤1024` 这类经验值替代对 DataCopy 参数上限的真实核算（32B 块语义下上限约为 2MB/次调用）。

## 误诊

- Branch-1 tiling 实现阶段曾把 21 条共享 UT 失败记录为"前序内核开发遗留、非本任务引入的退化"而放行——归因正确但没有闭环修复；随后验收判 FAILED 并派发专项修复才删掉 caps。这说明"已知遗留"记录不能替代修复。
