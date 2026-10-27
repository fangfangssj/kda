---
type: "Runbook"
title: "SyncAll 修复尾元素丢写无效反致误差恶化"
description: "对多核标量直卷积的单元素丢写（仅 y[1,0,0,0]=actual 0），在 Process 末尾加 AscendC::SyncAll() 不仅无效，还使该用例 MERE 从 6.02e-04 恶化到 1.33e-02（约 22 倍）；移除后恢复——『末尾统一 SyncAll 修丢写』路线 banned。"
tags: [task.debugging, topic.parallelism, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# SyncAll 修复尾元素丢写无效反致误差恶化

## 现象

标量直接卷积 kernel（多核切分，`GlobalTensor::SetValue` 标量写 GM）达到 40/43 后，残留失败用例的失败面非常窄：43 个用例中仅此用例仅 1 个元素错——

```
[Real] 小通道奇数尺寸用例 (H_out=13,W_out=13)
  x[2,7,13,13] filter[7,7,3,3] y[2,7,13,13] dtype=float16
  [FAIL] MERE=6.02e-04 (< 9.77e-04), MARE=1.00e+00 (>= 9.77e-03)
    不匹配 [11..]: golden=7.67, actual=0.000000 ...
```

错元素 index=1183，换算即 y[1,0,0,0]——第二个 batch、第一个输出通道、第一个空间位置；按核内切分公式（totalPairs=N*COut=14，usedCoreNum=2，perCore=7）恰好是 **core 1 分到的第一个 pair 的第一个写点**。当时怀疑是多核 GM 标量写（SetValue）的可见性问题，于是在 Process 末尾加了全核屏障：

```cpp
        }
    }

    // 确保所有核的 GM 写入完成
    AscendC::SyncAll();
}
```

## 适用场景

- 工程形态：AscendC 多核 kernel（blockDim>1），用 `GlobalTensor::SetValue` 做标量 GM 写，ST 出现"极少数元素为 0"类丢写时的修复路线选择。
- API 路径：`AscendC::SyncAll()`（块间屏障）；`GlobalTensor::SetValue`；`SetBlockDim/usedCoreNum`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：same_operator（本算子的标量直卷积路径，基线见 [骨架占位输出全 0：标量直算先打通正确性卡](skeleton_placeholder_output_zero_scalar_conv.md)）；"末尾统一 SyncAll 不解决 GM 写可见性"的结论对同类多核标量写场景可参考（未在其他算子验证）。
- 路线优先级：banned（对"末尾统一 SyncAll 修复单元素丢写"这一做法，本场景实测无效且有害）。**fallback 声明**：SyncAll 作为常规多核协作屏障（核间分工依赖）的正当用法不受本卡影响；本卡 banned 的仅是"用末尾 SyncAll 修丢写"这一修复假设。
- 明确不适用 / 未验证场景：核间分工协作中的 SyncAll 正常用法；DataCopyPad 批量写替代 SetValue 的修复方案未试验。

## 根因

未追到机制层（这是本卡最重要的定性）：该单元素丢写的真因未定，候选假设为 (a) SetValue 标量 GM 写在多核场景的可见性/提交时序；(b) tiling 的 usedCoreNum 按 tile 任务数计算与 kernel 按 (N×COut) pair 切分的语义差异导致边界 pair 归属异常。可以客观断言的是：**该丢写不是"核间缺少结束屏障"所致**——加 `SyncAll()` 的对照实验证明屏障既不能修复该元素，还会引入新的错误元素。

## 判别方法

成对对照（本会话实测，可复用的判定流程）：

1. 基线跑全量：记录失败用例与指标（MERE=6.02e-04，MARE=1.00e+00，仅 1 元素 actual=0）。
2. 在 Process 末尾加 `AscendC::SyncAll();`，重建（`build.sh --soc=ascend950`）重装（`--force`）重跑全量。
3. 对比同用例指标：MERE 6.02e-04 → **1.33e-02**（错误元素从 1 个变多），MARE 仍 1.00e+00；其余用例与总数（40/43）不变。
4. 回退移除 SyncAll，重建重跑：恢复基线指标与总数——确认恶化由 SyncAll 引入且可逆。

结论：末尾统一 SyncAll 对该丢写无修复作用（MARE 不动），反而引入 ~22× 的 MERE 恶化，路线判 banned 并回退。

## 解决配方

本卡为负向结果，无正修复；正确处置是**回退**：

```cpp
        }
    }
}   // 移除 AscendC::SyncAll();，恢复无屏障版本
```

后续接续方向（未实施，写入遗留问题）：用 DataCopyPad 批量写替代 SetValue（批量 GM 写的可见性行为不同）；或在写后加显式内存屏障（区别于核间屏障）；或先核对 tiling usedCoreNum 与 kernel 切分粒度的一致性。

## 设计决策

- 为什么一开始尝试 SyncAll：丢写元素恰为某核第一个写点，"写入未提交/可见性"是最直观假设，SyncAll 是最小改动的屏障手段。
- 为什么判 banned 而不是"无效即弃"：实验给出了双重证据（MARE 不动=没修好；MERE 恶化 22×=引入新错误），说明盲目加屏障在该场景有实际危害，值得固化为负知识防止下个会话重试。
- 当前方案路线档位：banned（本场景）。

## 预防

- 多核 kernel 出现"极少数元素丢写"时，先做切分归属换算（失败 index → (n,co,ho,wo) → pair → 核号/写点），再决定修复方向；不要第一步就加全核屏障。
- 任何"加同步"的修复都必须做加/不加的成对对照跑（指标逐位对比），避免把"恰好没变差"误判为修复。
- 只看用例总数（40/43 不变）就下"无影响"结论——MERE 的逐指标对比才暴露恶化。

## 验证信号

以下信号来自该算子联调验证会话的成对对照实验（同构建/安装/运行流程，仅 SyncAll 一处差异；Ascend950PR + CANN 9.1.0）：

- 基线：该用例 `[FAIL] MERE=6.02e-04 (< 9.77e-04), MARE=1.00e+00 (>= 9.77e-03)`，仅 index=1183（y[1,0,0,0]）actual=0/golden=7.67；全量 40/43（来源平台：950；证据等级：tool_observed）。
- 加 SyncAll 后：同用例 `[FAIL] MERE=1.33e-02 (>= 9.77e-04), MARE=1.00e+00`——MERE 恶化约 22 倍（错误元素增多），MARE 不变；全量仍 40/43（来源平台：950；证据等级：tool_observed）。
- 回退后：恢复 MERE=6.02e-04/MARE=1.00e+00，全量恢复 40/43，三例失败信号与基线一致（成对回退信号）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定"标量 SetValue 多核写的末尾统一 SyncAll"这一修复路线；SyncAll 在核间分工协作中的正当用法不受影响。
- 仅在该算子该 kernel 实测；其他算子/其他写路径（DataCopyPad 批量写）的可见性行为未验证。
- 恶化机制（SyncAll 为何引入新错误元素）未追——不排除与该 kernel 的切分边界（部分核 pair 数不均等）相互作用。
- 其他多核 SyncAll 失败形态（reduce 合并 507035 等）另见 [507035：多核 reduce SyncAll 回退单核卡](../runtime/507035_syncall_multicore_reduce.md)，机制不同勿混同。

**未决项**：

- **优先项：复现该单元素丢写并追根因到机制层**（确切机制：SetValue 可见性 vs 切分归属），修复已另派任务接续。
- DataCopyPad 批量写替代 SetValue、写后显式内存屏障两个候选方案未试验。
## 坏实践

- 见到多核丢写就无对照地加全核屏障（本例实测有害）。
- 只看用例总数（40/43 不变）就下"无影响"结论——MERE 的逐指标对比才暴露恶化。

## 误诊

- "写入未提交/可见性"假设本身未被证实也未被证伪——SyncAll 实验只否定了"缺结束屏障"这一具体机制，真因仍开放（见遗留问题）。
