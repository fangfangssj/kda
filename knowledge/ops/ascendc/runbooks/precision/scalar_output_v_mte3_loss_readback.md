---
type: "Runbook"
title: 单标量输出向量链丢值：回读标量侧再写
description: 1 元素向量链（Sqrt/Cast）结果经 V→MTE3 直写 GM 不生效；Duplicate→Sqrt→GetValue 回读标量侧再 SetValue→S_MTE3→DataCopyPad 输出后逐位一致。
tags: [task.debugging, topic.implementation.pipeline, topic.implementation.synchronization, topic.compute.normalization]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# 单标量输出向量链丢值：回读标量侧再写

## 现象

在 Ascend950PR（CANN 9.1.0）上联调 BenchNoHarnessForeachNorm（L2 范数：多核算 `sum(x_i^2)` 写 workspace，core 0 合并后开方输出每张量一个 fp32 标量）。workspace 越界崩溃修好后，ST 第一个用例（1 个 fp32 张量 [1024]，N(0,1) 数据）kernel 正常执行但数值错误（原文）：

```text
[Real] L0_001 - L=1, scalar=2.0, 1D L2范数基础验证
  [FAIL] 标量[0]: golden=32.008255, actual=0.092471, rel_err=9.97e-01
  [FAIL] MERE=9.97e-01 (threshold=1.22e-04), MARE=9.97e-01 (threshold=1.22e-03)
```

golden=32.008255 即 sqrt(1024.53)（1024 个标准正态样本的 L2 范数）。当时 kernel 的收尾链是：core 0 把合并和 `mergedResult` 用 `resultBuf.SetValue(0, mergedResult)` 写 UB，`S_V` 同步后 `Sqrt(resultBuf, resultBuf, 1)`，再 `Cast(outLocal, ..., 1)`，`V_MTE3` 同步后 `DataCopyPad` 写 GM 输出张量。先后尝试：切单核模式、FP32 跳过 Cast 直接用输入缓冲、重排 S_V/V_S 同步位置——actual 均原样 0.092471；把结果强行改为 `totalElem` 输出仍是 0.092471（计算值根本没到输出）；常数 42.0 直写输出通路得到 42.000000（输出通路好）；跳过 Sqrt/Cast 直写 `mergedResult` 得到 1024.528320 = golden²（计算与合并都对）；恢复 Sqrt（独立缓冲 + Cast）后输出仍是 1024.528320——**开方结果在 V 管线到 MTE3 输出之间丢失**。

## 适用场景

- 工程形态：自定义 AscendC 算子 kernel 开发（单/少元素标量输出的归约类收尾段），ST 上板精度联调。
- API 路径：`Duplicate` / `Sqrt` / `Cast`（V 管线）、`LocalTensor::SetValue/GetValue`（S 标量侧）、`HardEvent::S_V / V_S / S_MTE3 / V_MTE3`、`PipeBarrier<PIPE_V>`、`DataCopyPad`（MTE3）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——归约/范数类"向量算出 1 个标量再写 GM"的收尾段模式。
- 路线优先级：recommended_fastpath（正确性优先；回读标量侧的额外同步开销未测）。
- 明确不适用 / 未验证场景：多核模式（usedCoreNum>1）在本会话恢复后段错误、未修复，本卡验证信号全部来自单核模式；大宽度向量主计算段（ubFactor 级）的同步模式未变、不受本卡影响。

## 根因

单元素（count=1）标量结果的向量链：`SetValue(S) → S_V → Sqrt(V) → Cast(V) → V_MTE3 → DataCopyPad(MTE3)` 中，Sqrt 的结果没有出现在最终 GM 输出里（输出停留在开方前的合并和 1024.528320；更早的 0.092471 是同步未修正阶段的脏值）。改为"向量结果经 `V_S` 同步后 `GetValue` 回读标量侧，再走已验证的 `SetValue → S_MTE3 → DataCopyPad` 通路"后结果逐位正确。

说明：本卡是**机制层定位 + 可复用修复模式**；"为什么 1 元素 V→MTE3 直写会丢值"（1 元素向量指令粒度、事件同步类型或 in-place 冒险）未追到指令级——该深挖列为遗留问题优先项。判别信号：跳过 Sqrt 直写合并和得到 golden²，恢复 Sqrt 输出不变，即锁定丢失发生在该向量段（旁路二分定位法见[内核垃圾值分层探针卡](../runtime/kernel_output_garbage_layered_probe_chain.md)）。

## 解决配方

修复前（失败链，`MergeAndOutput` 收尾段，verbatim——该版本输出 1024.528320，Sqrt 未生效）：

```cpp
        // 最终计算 + 输出
        {
            // 写入 mergedResult 到 resultBuf
            resultBuf.SetValue(0, mergedResult);
            event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV);
            WaitFlag<HardEvent::S_V>(eventIDSToV);

            // Sqrt: 使用 workBuf0 作为输出（避免 in-place）
            LocalTensor<ComputeT> workBuf0 = workBuf0_.Get<ComputeT>();
            if constexpr (NORM_TYPE == 2) {
                Sqrt(workBuf0, resultBuf, 1);
                PipeBarrier<PIPE_V>();
            } else if constexpr (NORM_TYPE == 5) {
                Log(workBuf0, resultBuf, 1);
                PipeBarrier<PIPE_V>();
                Muls(workBuf0, workBuf0, invP_, 1);
                PipeBarrier<PIPE_V>();
                Exp(workBuf0, workBuf0, 1);
                PipeBarrier<PIPE_V>();
            } else {
                // L0, L1, inf, -inf: 直接复制
                workBuf0.SetValue(0, mergedResult);
            }

            // Cast 回原 dtype
            LocalTensor<D_T_X> outLocal = inQueue_.template AllocTensor<D_T_X>();
            if constexpr (std::is_same_v<D_T_X, ComputeT>) {
                Cast(outLocal, workBuf0, RoundMode::CAST_NONE, 1);
            } else {
                Cast(outLocal, workBuf0, RoundMode::CAST_ROUND, 1);
            }
            PipeBarrier<PIPE_V>();

            // 写回 GM
            __gm__ D_T_X* outGM = GetTensorAddr(static_cast<uint16_t>(k), yPtr_);
            GlobalTensor<D_T_X> outTensorGM;
            outTensorGM.SetGlobalBuffer(outGM, 1);

            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(sizeof(D_T_X)), 0, 0, 0};
            DataCopyPad(outTensorGM[0], outLocal, outCopyParams);

            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

            inQueue_.FreeTensor(outLocal);
        }
```

修复后（验证通过链，verbatim）：

```cpp
        // 最终计算 + 输出
        {
            // 在标量侧计算最终结果
            ComputeT finalResult = mergedResult;
            if constexpr (NORM_TYPE == 2) {
                // L2: sqrt(sum) - 标量侧计算
                finalResult = mergedResult * mergedResult;  // 先平方
                // 使用近似平方根: 1/sqrt(x) * x
                // AscendC 标量侧没有 sqrtf, 用向量 API
            }

            // 使用 SetValue + S_MTE3 + DataCopyPad (已验证可行)
            LocalTensor<D_T_X> outLocal = inQueue_.template AllocTensor<D_T_X>();

            // 对于 L2: 使用向量 Sqrt
            if constexpr (NORM_TYPE == 2) {
                // 先用 Duplicate 写入 workBuf0
                LocalTensor<ComputeT> workBuf0 = workBuf0_.Get<ComputeT>();
                Duplicate<ComputeT>(workBuf0, mergedResult, 8);
                PipeBarrier<PIPE_V>();
                // Sqrt 到 workBuf1
                LocalTensor<ComputeT> workBuf1 = workBuf1_.Get<ComputeT>();
                Sqrt(workBuf1, workBuf0, 8);
                PipeBarrier<PIPE_V>();
                // 读取结果到标量
                event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventIDVToS);
                WaitFlag<HardEvent::V_S>(eventIDVToS);
                finalResult = workBuf1.GetValue(0);
                // S -> MTE3
                event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            } else {
                event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            }

            outLocal.SetValue(0, D_T_X(finalResult));

            event_t eventIDSToMTE3_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3_2);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3_2);

            __gm__ D_T_X* outGM = GetTensorAddr(static_cast<uint16_t>(k), yPtr_);
            GlobalTensor<D_T_X> outTensorGM;
            outTensorGM.SetGlobalBuffer(outGM, 1);

            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(sizeof(D_T_X)), 0, 0, 0};
            DataCopyPad(outTensorGM[0], outLocal, outCopyParams);

            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

            inQueue_.FreeTensor(outLocal);
        }
```

要点：① 标量入向量管线用 `Duplicate(buf, val, 8)`（V 操作，8 元素 = 32B 对齐），不用 S 侧 `SetValue` + `S_V` 依赖；② 向量结果经 `V_S` 同步后 `GetValue(0)` 回读标量；③ 输出统一走已验证的 `SetValue → S_MTE3 → DataCopyPad` 通路。

## 设计决策

- 为什么 `Duplicate` 填 8 个元素而不是 1 个：单元素向量操作在 32B 粒度的向量通路上可能不满足最小处理宽度，8×fp32=32B 对齐（该宽度选择与 GetValue 回读一起变更生效，单独贡献未拆分验证，见遗留问题）。
- 为什么回读标量侧而不是修 V→MTE3 通路：`SetValue→S_MTE3→DataCopyPad` 已被常数旁路实验验证可靠，把"最后一步写 GM"固定在已知好通路上，只把不确定性留在向量计算段。
- 备选（被放弃）：`reinterpret_cast` 直接把 float workBuf 当 D_T_X 输出、`DataCopy(workBuf1, workBuf0, 8)` 复制——测试后仍得不到开方结果，放弃。
- 当前方案路线档位：recommended_fastpath（正确性优先；回读标量侧的额外同步开销未测性能影响）。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；单核调试模式（tiling 临时 `usedCoreNum = 1; blockFactor = totalElem`）。
- 相关文件：`op_kernel/arch35/bench_no_harness_foreach_norm.h`（MergeAndOutput 收尾段、workBuf0/workBuf1/resultBuf TBuf）、`op_host/arch35/bench_no_harness_foreach_norm_tiling.cpp`（单核调试开关）、ST `test_aclnn_bench_no_harness_foreach_norm.cpp`（读回比对）。

## 预防

- 归约类算子收尾段（1 个标量写 GM）优先采用"向量算 → `V_S` 回读标量 → `SetValue` → `S_MTE3` → `DataCopyPad`"模板，不要让 1 元素向量结果直接走 V→MTE3。
- 出现"输出值 = 开方前的和 / 输入原值"这类特征时，先怀疑收尾向量链的可见性，再用旁路二分定位。

## 验证信号

来源说明：以下信号来自该算子迭代一 ST 联调会话：失败值复现 → 三次无效修复对照 → 旁路定位 → 最终链修复 → ST 重跑。

- 修复前：L0_001 `golden=32.008255, actual=0.092471, MERE=9.97e-01`；单核/跳过 Cast/同步重排后 actual 不变（来源平台：950；证据等级：tool_observed）。
- 修复前（隔离证据）：跳过 Sqrt/Cast 直写合并和 `actual=1024.528320`（= golden²，计算与合并正确）；恢复 Sqrt 后 `actual=1024.528320` 不变（开方结果丢失）（来源平台：950；证据等级：tool_observed）。
- 修复后：L0_001 `[PASS] MERE=0.00e+00, MARE=0.00e+00 (threshold=1.22e-04, 1 scalars)`，与 CPU golden 逐位一致（单核模式）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 验证信号均来自单核模式；多核模式（usedCoreNum>1）在本会话恢复后段错误，未在本卡范围内闭环。
- 修复链一次变更了三个变量（Duplicate 8 元素、独立 Sqrt 缓冲、GetValue 回读），"哪个变量是决定性因素"未拆分验证——更窄结论以整链模板为准。
- fp16/bf16 输入（D_T_X ≠ ComputeT）走 `Cast(outLocal, workBuf0, CAST_ROUND, 1)` 的老链未在新模板下重验（本会话仅 fp32 用例）。
- 其他 CANN 版本未验证（推断可迁移）。

**未决项**：

- 优先项：复现并追根因到机制层——1 元素向量链结果经 V→MTE3 丢失的确切原因（最小向量宽度？事件类型？in-place 冒险？），并拆分 Duplicate-8/独立缓冲/GetValue 三变量的贡献。
- 多核模式段错误根因（SyncAll/partials 合并路径）未诊断。
- 修复链的相对性能代价未测量。
## 坏实践

- 对 1 元素标量输出沿用大向量段的同步模板（SetValue→S_V→向量→V_MTE3→DataCopyPad），假定 1 元素与大宽度行为一致。
- 三个"盲修"（单核、跳 Cast、同步重排）在无观测变化后仍继续叠加改动而不先做旁路定位——浪费了多轮编译-安装-上板周期。

## 误诊

- 曾把 0.092471 归因为 FP32 输入仍走了 `Cast` 拷贝路径，跳过 Cast 直接用 inputData——值不变，排除。
- 曾怀疑 `resultBuf.SetValue` 与 `Sqrt` 之间同步顺序（S_V 提前/滞后），重排后值不变，排除。
