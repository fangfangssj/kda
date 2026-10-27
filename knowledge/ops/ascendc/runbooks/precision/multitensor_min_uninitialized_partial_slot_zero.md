---
type: "Runbook"
title: "多张量min范数合并读未写workspace零值"
description: 多张量并行时未处理某张量的核不写 workspace partial result，-inf（min）范数合并把未初始化的 0 当最小值，输出全 0；核跳过张量时补写中性元素（min=1e30f/max=-1e30f/sum=0）。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T03:21:08Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# 多张量min范数合并读未写workspace零值

## 现象

自定义算子按"两阶段多核"实现：阶段一各核对分到的张量块算 partial result 写入 workspace（布局 `partialResults[k * usedCoreNum + blockIdx]`），阶段二 0 号核对每个张量 k 跨核合并输出。迭代二联调时，唯一失败用例是**多张量 p=-inf（最小绝对值）并行**：

```text
[Real] L1_028 - fp32, L=2, p=-inf, et=normal, 多张量scalar=-inf最小绝对值并行
  [FAIL] 标量[0]: golden=0.005183, actual=0.000000, rel_err=1.00e+00
  [FAIL] 标量[1]: golden=0.000192, actual=0.000000, rel_err=9.99e-01
  [FAIL] MERE=1.00e+00 (threshold=1.22e-04), MARE=1.00e+00 (threshold=1.22e-03)
```

两个张量的实际输出都是 0.000000，而 golden 是 0.005183/0.000192（正确的最小绝对值）。对照实验锁定形态：单张量 -inf 用例（1D 1024、小张量 100 元素）全部 PASS，多张量（L=2，{1024, 512}）必 FAIL；同为多张量的求和类范数（L2/L0）PASS。

## 适用场景

- 工程形态：AscendC 多核两阶段归约 kernel（阶段一 partial → workspace → 阶段二跨核合并），尤其中间结果是 min/max 这类**非交换中性元素敏感**的归约。
- API 路径：workspace `DataCopyPad` 写入/`GetValue` 读取、`if constexpr (NORM_TYPE == ...)` 模板分支。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR，arch35）+ CANN 9.1.0。
- 适用粒度：`operator_family`（多核 workspace 归约模式在 min/max 类算子家族通用）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证场景：sum 类归约（0 恰好是中性元素，未写也不出错——这正是它长期掩盖问题的原因）；单核实现不涉及。

## 根因

多张量并行切分下，不是每个核都处理每个张量：核对张量 k 的 `overlapStart >= overlapEnd` 时**不写**该张量的 partial result，workspace 对应槽位保持分配时的 0；阶段二合并以"读取全部 usedCoreNum 个核的 partial"为前提做 min——0 比任何正的真实最小值都小，min 结果被未初始化的 0 劫持，输出恒 0。推导：golden=min|x|=0.005183 > 0，任一未写槽位值 0 < 0.005183，故 mergedResult=0，rel_err=1.00。

## 解决配方

阶段一（`ComputePartialResults`）补 else 分支：核不处理某张量时，显式写**该归约的中性元素**。

修复前（`if (overlapStart < overlapEnd)` 块内写 partial，块外什么都不写）：

```cpp
            // 确保 DataCopyPad(MTE3) 完成
            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        }
        // ← 核不处理此张量时直接跳过，workspace 槽位未写
```

修复后（else 分支写中性值，逐 NORM_TYPE 区分）：

```cpp
            // 确保 DataCopyPad(MTE3) 完成
            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        } else {
            // 当前核不处理此张量: 写入中性值, 避免 MergeAndOutput 读取未初始化数据
            // NORM_TYPE=3(max): -1e30f, NORM_TYPE=4(min): 1e30f, 其他(sum): 0.0f
            ComputeT neutralResult = 0.0f;
            if constexpr (NORM_TYPE == 3) {
                neutralResult = -1e30f;
            } else if constexpr (NORM_TYPE == 4) {
                neutralResult = 1e30f;
            }

            int64_t partialIdx = k * usedCoreNum_ + static_cast<int64_t>(blockIdx_);
            LocalTensor<ComputeT> tmpResult = resultBuf_.Get<ComputeT>();
            tmpResult.SetValue(0, neutralResult);

            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(sizeof(ComputeT)), 0, 0, 0};
            DataCopyPad(partialResultsGM_[partialIdx], tmpResult, outCopyParams);

            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        }
```

阶段二合并端的目标值初始化（既有代码，保持与中性元素一致，供对照理解）：

```cpp
ComputeT mergedResult = 0.0f;
if constexpr (NORM_TYPE == 3) {
    mergedResult = -1e30f;   // max 归并初值
} else if constexpr (NORM_TYPE == 4) {
    mergedResult = 1e30f;    // min 归并初值
}
```

中性元素表（按归约类型）：

| 归约 | 中性元素 | 说明 |
|---|---|---|
| min（-inf 范数等） | `+1e30f`（FLT_MAX 近似） | 任何真实值都更小 |
| max（inf 范数等） | `-1e30f` | 任何真实值都更大 |
| sum（L0/L1/L2/通用 p） | `0.0f` | 未写槽位天然无害——所以求和类从不出错、问题被掩盖 |

## 设计决策

- 选"写入方补中性值"而不是"合并方跳过未写槽位"：合并方无法区分"未写"与"真实算出 0"（workspace 无有效标记位）；写入方每核每张量都留痕，布局契约简单（全部 usedCoreNum × tensorNum 槽位恒有效）。
- 中性元素用 `±1e30f` 而不是 `±FLT_MAX`：与阶段二既有归并初值一致（历史实现如此），避免两处不一致。

## 预防

- 设计多核 workspace 归约时先填一张"归约类型 × 中性元素"表，检查两处：写入方跳过路径是否补中性值、合并方初值是否同值。
- 测试矩阵必须包含"多张量 + min/max 类归约"组合——单张量与求和类都测不出该缺陷。
- workspace 不承诺零初始化，任何"依赖初始值"的假设都要显式写值覆盖。

## 验证信号

来自该算子迭代二联调会话的实测：

- 修复前：L1_028（多张量 p=-inf）actual=0.000000，rel_err=1.00；同套件单张量 -inf（L1_009/L1_047）PASS（来源平台：950；证据等级：tool_observed）。
- 修复后：同用例 PASS；全量 ST Real 模式 60/60（含 4 张量 2D 并行 L2、多张量各范数类型），UT 24/24（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 中性元素 `1e30f` 假设真实数据绝对值小于 1e30；若业务数据可能达到 FLT_MAX 量级，需换 `+inf` 语义并验证硬件 min 对 NaN/Inf 的行为。
- 输入含 NaN 时 min/max 归并行为未验证。
- 空 TensorList/空张量路径走独立分支（直接写 0 输出），与本卡机制无关。

**未决项**：

- 无（根因与修复均在机制层闭环）。
## 坏实践

- 依赖 workspace"反正会是 0"的隐式初始值——对 min 类归约，0 恰是最坏值。
- 合并端读到离谱结果先怀疑精度/tolerance，而不是检查"每个槽位是否都有人写"。

## 误诊

- 无（对照实验 + 代码走读直接定位）。
