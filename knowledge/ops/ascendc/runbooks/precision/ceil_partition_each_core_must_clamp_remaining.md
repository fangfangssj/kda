---
type: "Runbook"
title: "多核 ceil 尾块为负 — 每核按 remaining 截断"
description: "ceil 分块后预计算末核 tail 可为负并让中后核越界；改为每核按 total-start 截断。"
tags: ["task.debugging", "topic.precision", "topic.tiling", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 多核 ceil 尾块为负 — 每核按 remaining 截断

## 现象

在编写 Flash 独立 Ascend C kernel、按 `(batch, sequence)` 的扁平任务数给 AI Core 分工时，远端卡 6 构建成功，但完整 NPU 参数化测试只有 25 个通过、60 个失败。失败随总任务数增大集中出现，小任务可以通过；当时代码用
`blockFactor = ceil(total / blockNum)`，再只为最后一个核预计算 `tail = total - (blockNum - 1) * blockFactor`。

## 适用场景

- 工程形态：kernel 直调 / Flash 单文件 `.asc` 的多核任务切分；
- API 路径：`GetBlockIdx()`、`GetBlockNum()` 与 Host 侧 ceil 分块；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b）/ CANN 9.0.1；
- 适用粒度：`operator_family`，适用于把一维总任务连续分配给多个核的算子；
- 路线优先级：`recommended_fastpath`，逐核 clamp 是正确性基线；
- 明确不适用：循环步进式 round-robin 分配、Host 已显式给每核 start/count 的 tiling。

## 根因

唯一根因是 `factor = ceil(total / blockNum)` 时仍用 `tail = total - (blockNum - 1) * factor` 只修最后一核：一旦 `(blockNum - 1) * factor > total`，`tail < 0`，而倒数若干核也可能已经满足 `start >= total`，从而分别发生末核空退和中间核 GM 越界。

## 解决配方

Kernel 接收 `total` 与 `factor`，每个核自己计算 `start = blockId * factor`；先以 `start >= total` 过滤空核，再用 `min(factor, total - start)` 得到当前核的真实任务数。

### before

以下为轨迹完整写入中的原始 `Process` 函数：

```cpp
__aicore__ inline void Process()
{
    int64_t bid = AscendC::GetBlockIdx();
    int64_t bsStart = bid * blockFactorBS_;
    int64_t bsCount = (bid == AscendC::GetBlockNum() - 1) ? blockTailBS_ : blockFactorBS_;
    if (bsCount <= 0) return;

    for (int64_t i = 0; i < bsCount; ++i) {
        int64_t bs = bsStart + i;
        int64_t b = bs / S_;
        int64_t s = bs % S_;
        // load cos/sin for this (b,s), reused across N heads and q/k
        auto cosUb = cosBuf_.Get<T>();
        auto sinUb = sinBuf_.Get<T>();
        CopyInCosSin(cosUb, sinUb, s);
        ProcessQK(qGm_, qOutGm_, b, s);
        ProcessQK(kGm_, kOutGm_, b, s);
    }
}
```

### after

以下为同一函数在 tiling 修复后、同步问题修复前的原始完整形态；除任务计数外其余逻辑保持不变：

```cpp
__aicore__ inline void Process()
{
    int64_t bid = AscendC::GetBlockIdx();
    int64_t bsStart = bid * blockFactorBS_;
    if (bsStart >= numBS_) return;                 // empty core
    int64_t remaining = numBS_ - bsStart;
    int64_t bsCount = remaining < blockFactorBS_ ? remaining : blockFactorBS_;

    for (int64_t i = 0; i < bsCount; ++i) {
        int64_t bs = bsStart + i;
        int64_t b = bs / S_;
        int64_t s = bs % S_;
        // load cos/sin for this (b,s), reused across N heads and q/k
        auto cosUb = cosBuf_.Get<T>();
        auto sinUb = sinBuf_.Get<T>();
        CopyInCosSin(cosUb, sinUb, s);
        ProcessQK(qGm_, qOutGm_, b, s);
        ProcessQK(kGm_, kOutGm_, b, s);
    }
}
```

## 设计决策

- 选择“传 total + 每核 clamp”，因为它对 ceil 产生的多余核天然安全，也不需要 Host 生成每核数组；
- 放弃“只修最后一个核的 tail”，因为 ceil 分块可能让不止最后一核越过 total；
- 当前路线为 `recommended_fastpath`：只有两次整数比较，不引入额外同步或搬运。

## 预防

- 推导多核连续段时同时检查 `start`、`count` 和 `start + count <= total`；
- 对 `total < blockNum`、`total % blockNum != 0`、`total` 刚跨越 blockNum 倍数的边界做上板测试；
- 不把 `tail <= 0` 只当“最后一核没任务”，它也提示 ceil 分配可能让前面的核越界。

## 验证信号

来源为一次 Ascend950 Flash 算子开发会话中的多核切分修复与连续 NPU 测试：

- 修复前完整套件 25 passed、60 failed（950，证据等级: tool_observed）；
- 改为逐核 clamp 后 29 passed、56 failed，新增通过的 4 个用例不再被该 tiling 根因阻断；剩余失败随后定位为独立同步与 B16 interleave 根因（950，证据等级: tool_observed）；
- 所有独立根因修完并干净重建后 85/85 通过（950，证据等级: tool_observed）。

## 失效边界

本配方只解决连续段切分的边界正确性，不处理负载不均衡和动态调度。其他平台未验证，推断可迁移。

## 坏实践

使用 ceil 得到固定 factor 后，只给最后一核一个 tail，而默认所有中间核都有效；当 total 较小时，这会同时漏掉空核保护与中间核越界保护。

## 误诊

看到“大规模失败、小规模通过”时只检查向量计算或精度容差，会错过 Host/Kernel 共同决定的任务范围错误。
