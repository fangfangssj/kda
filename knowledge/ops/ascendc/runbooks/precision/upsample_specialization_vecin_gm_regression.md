---
type: "Runbook"
title: upsample特化路径VECIN直拷GM致TilingKey=2全失败
description: 新增的 upsample 特化路径从 VECIN 队列单元素直拷 GM 且假设窗口 k=1，导致全部 TilingKey=2 用例输出 0/错值；修复为路由回已验证的通用窗口路径，特化函数保留待重写。
tags: [task.debugging, topic.precision, topic.compute.neighborhood.convolution, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:15:37Z'
updated_at: '2026-09-16T11:15:37Z'
platforms: ["950"]
---

# upsample特化路径VECIN直拷GM致TilingKey=2全失败

## 现象

BenchNoHarnessAdaptiveAvgPool3d（3D 自适应平均池化，fp32，Ascend950PR/arch35，CANN 9.1.0）在引入多 TilingKey 分发（0=downsample、1=global_avg、2=upsample）后做联调验证：`bash build.sh --soc=ascend950 -j8 --make_clean` 重编译、`custom_opp_ubuntu_x86_64.run --force` 重安装，执行 `bash tests/st/run.sh --real --case testcases/L0_test_cases.csv`（37 条用例），10 条失败且模式完全一致——**所有 TilingKey=2（upsample 及任一维上采样的混合）用例全部失败**，TilingKey=0/1 全部通过：

- 16³→128³ 大放大比上采样用例：`[FAIL] MERE=2.98e+00 (>= 1.22e-04), MARE=2.27e+03`，`不匹配 [0]: 期望=-0.864330, 实际=0.000000`、`不匹配 [8]: 期望=3.208914, 实际=-0.864330`（实际输出为 0 或邻位错值）；
- 失败集构成：6 条纯 upsample 用例 + 3 条含上采样的混合用例 + 1 条 16³→128³ 大放大比用例，共 10 条，与 TilingKey=2 完全对齐。
- 同批用例在引入分发前的单一通用路径版本全部通过——即新启用的 `ProcessUpsample` 特化路径破坏了原本正确的行为。

## 适用场景

- 工程形态：aclnn 自定义算子 kernel 多 TilingKey 特化路径开发 + C++ ST 实测回归。
- API 路径：AscendC TQue（VECIN/VECOUT）、DataCopyPad 单元素搬运（`blockLen = sizeof(D_T_X)`）、Process 内 TilingKey 分发。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何按 TilingKey/形状分支做"特化优化路径"的池化/重采样类 kernel。
- 路线优先级：safety_fallback（回退通用路径；特化路径重写并验证前不启用）。
- 明确不适用 / 未验证场景：TilingKey=1（global）特化路径本轨迹实测通过，不在本卡范围。

## 根因

新启用的 `ProcessUpsample` 特化路径实现错误：从 VECIN 队列分配的 `xLocal` 直接 `DataCopyPad` 到 GM（非 AscendC 标准 GM→VECIN→VECCALC→VECOUT→GM 数据流），且逐输出位置单元素搬运（`blockLen = sizeof(D_T_X) = 4B`），同时假设所有输出位置窗口 k=1（非整数倍 upsample 与混合场景实际 k>1）。

修复前 Process 分发（verbatim）：

```cpp
    for (uint32_t nc = startNc; nc < endNc; nc++) {
        if (tilingKey_ == 1U) {
            // TilingKey=1: 全局平均池化特化路径
            ProcessGlobalAvgPool(nc);
        } else if (tilingKey_ == 2U) {
            // TilingKey=2: upsample 路径
            ProcessUpsample(nc);
        } else {
            // TilingKey=0: 通用 downsample 路径
            ProcessChannel(nc);
        }
    }
```

出问题的特化路径（verbatim，保留在代码中但不再被调用）：

```cpp
template <typename D_T_X>
__aicore__ inline void KernelAdaptiveAvgPool3d<D_T_X>::ProcessUpsample(uint32_t ncIdx)
{
    uint64_t ncInOffset = static_cast<uint64_t>(ncIdx) * D_ * H_ * W_;
    uint64_t ncOutOffset = static_cast<uint64_t>(ncIdx) * outD_ * outH_ * outW_;

    DataCopyParams elemCopyParams;
    elemCopyParams.blockCount = 1;
    elemCopyParams.blockLen = static_cast<uint16_t>(sizeof(D_T_X));
    elemCopyParams.srcStride = 0;
    elemCopyParams.dstStride = 0;

    for (uint32_t oD = 0; oD < outD_; oD++) {
        // upsample 场景下窗口为 1：start = floor(oD * D / outD)
        uint32_t startD = (oD * D_) / outD_;
        for (uint32_t oH = 0; oH < outH_; oH++) {
            uint32_t startH = (oH * H_) / outH_;
            for (uint32_t oW = 0; oW < outW_; oW++) {
                uint32_t startW = (oW * W_) / outW_;

                // 输入 GM 偏移（窗口 k=1，读取单个元素）
                uint64_t inOffset = ncInOffset
                    + static_cast<uint64_t>(startD) * H_ * W_
                    + static_cast<uint64_t>(startH) * W_
                    + startW;
                // 输出 GM 偏移
                uint64_t outOffset = ncOutOffset
                    + static_cast<uint64_t>(oD) * outH_ * outW_
                    + static_cast<uint64_t>(oH) * outW_
                    + oW;

                // CopyIn 单元素: GM → UB
                LocalTensor<D_T_X> xLocal = inQueueX.template AllocTensor<D_T_X>();
                DataCopyPad(xLocal, xGm[inOffset], elemCopyParams, {false, 0, 0, 0});
                inQueueX.EnQue(xLocal);
                auto xIn = inQueueX.template DeQue<D_T_X>();

                // CopyOut 单元素: UB → GM（窗口 k=1，无需 ReduceSum）
                DataCopyPad(yGm[outOffset], xIn, elemCopyParams);
                inQueueX.FreeTensor(xIn);
            }
        }
    }
}
```

## 解决配方

把 TilingKey=2 路由回已验证的通用路径 `ProcessChannel`（其窗口计算对 k=1 退化为单元素复制、对 k>1 正确归约），保留 TilingKey 分类供后续优化。修复后 Process 分发（verbatim）：

```cpp
    for (uint32_t nc = startNc; nc < endNc; nc++) {
        if (tilingKey_ == 1U) {
            // TilingKey=1: 全局平均池化特化路径
            ProcessGlobalAvgPool(nc);
        } else {
            // TilingKey=0 (downsample/mixed) 和 TilingKey=2 (upsample) 均走通用路径
            // TilingKey=2 保留分类供后续 per-position 优化，当前 ProcessChannel 已正确
            // 处理 upsample（k=1 时 ReduceSum 退化为单元素复制）和 mixed 场景
            ProcessChannel(nc);
        }
    }
```

同时在 `ProcessUpsample` 头部注释记录未启用的原因（verbatim）：

```cpp
// TilingKey=2: upsample 路径（当前未启用，保留供后续优化）
// 触发条件: output_d > D || output_h > H || output_w > W
//
// 已知问题（联调发现）：
//   1. 数据流违规：直接从 VECIN(inQueueX) DataCopyPad 到 GM，非标准 GM→VECIN→VECOUT→GM 路径
//   2. 窗口假设错误：假设所有输出位置 k=1，但非整数倍 upsample 和 mixed 场景下 k>1
//   当前 TilingKey=2 路由到 ProcessChannel 通用路径（已验证正确），本函数保留供后续
//   per-position 优化（仅对 k=1 位置做直接复制，k>1 位置走 ReduceSum）
```

## 设计决策

- 为何回退而不是修特化路径：通用路径 `ProcessChannel` 在引入分发前已被 37 条用例验证（含全部 upsample 用例 MERE=0.00e+00）；特化路径的价值（性能）未测量，而正确性风险已实测。安全回退优先。
- 保留 TilingKey=2 分类与函数体：为后续"仅 k=1 位置直接复制、k>1 位置归约"的 per-position 优化留接续点，且注释固化了两个已知问题。
- 被放弃的设计：单元素 DataCopyPad 直拷（VECIN→GM）——数据流违规 + 4B blockLen 疑似低于最小搬运粒度 + k=1 假设错误，三重问题叠加，未逐一隔离验证即整体回退。
- 当前属 safety_fallback：特化路径重写（走 VECOUT 队列、按窗口 k 分派）并全量回归前，不建议启用。

## 预防

- 新增特化路径必须在真实 NPU 上跑全量 TilingKey 覆盖用例后才能合入（本例 UT 35/35 通过但 UT 只覆盖 host 侧 tiling 逻辑，不执行 kernel，未拦截该缺陷）。
- kernel 数据流 code review：写 GM 只能经 VECOUT 队列；`DataCopyPad(yGm, xIn, ...)` 出现在 VECIN 张量上是直接红旗。
- upsample 特化前先枚举窗口 k 的分布：非整数倍缩放与混合场景必然存在 k>1 的输出位置。

## 验证信号

本卡信号来自该算子联调验证（ST Real 模式）：

- 修复前：37 条中 10 条失败（全部 TilingKey=2），16³→128³ 上采样用例 `MERE=2.98e+00, MARE=2.27e+03`、实际值 `0.000000`/邻位错值（来源平台：950；证据等级：tool_observed）。
- 修复后（重编译+重安装）：L0 回归 37/37——6 条纯 upsample 用例全部 `[PASS] MERE=0.00e+00, MARE=0.00e+00`，16³→128³ 用例 `MERE=0.00e+00`（来源平台：950；证据等级：tool_observed）。
- 扩展用例集（58 条，TilingKey 覆盖 TK0=22/TK1=12/TK2=24）58/58 通过（来源平台：950；证据等级：tool_observed）。
- UT（host 侧 tiling/infershape，35 条）35/35 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡修复为行为回退（upsample 走通用逐窗口路径），未测量性能代价；对 upsample 占比高的负载可能有性能影响（未验证）。
- 两个根因（数据流违规、k=1 假设）未独立隔离验证——哪个是主因、单元素 4B blockLen 是否本身违规，均未做受控实验。
- 特化路径重写方案（per-position k 分派 + VECOUT 出队）未实施。

**未决项**：

1. 【优先项】按机制层隔离两个根因：分别单独修复数据流（改走 VECOUT）与 k 假设（按窗口分派）后重跑 TilingKey=2 用例，确认各自贡献。
2. 验证单元素 `blockLen=4B` 的 DataCopyPad 是否低于硬件最小搬运粒度（与 ComputeWindow CopyOut 路径同为 4B 但工作正常的现象矛盾，需解释）。
3. upsample 特化的性能收益测量（相对通用路径的成对对比，避免绝对分数）后再决定是否重写启用。
## 坏实践

- 特化路径"看起来等价"就直接替换已验证的通用路径，且替换后仅依赖 host 侧 UT（不执行 kernel）作为门禁。
- 从 VECIN 队列分配的张量直接 DataCopyPad 到 GM。

## 误诊

- 无明显误诊轮次；失败模式（TilingKey 全量对齐）一次定位到特化路径。
