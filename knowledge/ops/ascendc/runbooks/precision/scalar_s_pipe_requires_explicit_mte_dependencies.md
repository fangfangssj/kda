---
type: "Runbook"
title: "标量 S pipe 误用 TQue 导致输出全常数"
description: "标量 C++ 直接访问 UB 时，TQue 的 V-pipe 事件不能覆盖 S pipe；改用 TBuf 与 PIPE_ALL 串行化 DMA、标量计算和写回。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-16T11:20:30Z"
platforms: ["950"]
---

# 标量 S pipe 误用 TQue 导致输出全常数

## 现象

在 Ascend C kernel 直调工程中，先用远端 helper 构建 `cummin.asc`，再执行完整 NPU pytest。源码已成功编译，接口测试也通过，但第一个 fp16 末轴前缀扫描用例报：

```text
AssertionError: cummin values mismatch ... max diff=11.84765625
```

设备结果的每个位置都是同一错误常量，而不是随前缀推进变化。失败时，`KernelCummin<T>::ProcessPathA` 用 `TQue::EnQue/DeQue` 包围 `DataCopyPad`，随后让标量 C++ 循环通过 `__ubuf__` 指针读输入并写输出。

## 适用场景

- 工程形态：Ascend C kernel 直调；DMA 把 GM 数据搬入 UB，标量 C++ 在 UB 上做顺序递推，再由 DMA 写回 GM；
- API 路径：`TPipe`、`TQue`/`TBuf`、`DataCopyPad`、`PipeBarrier<PIPE_ALL>()` 与 `__ubuf__` 指针；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致；
- 适用粒度：`operator_family`，适用于所有由标量 S pipe 直接消费 MTE2 结果或生产 MTE3 输入的 kernel；
- 路线优先级：`safety_fallback`；**这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 后续增量的 HardEvent 精细路线，注意其紧循环失步限制）。**
- 明确不适用：计算完全位于 Vector pipe 且由 TQue 正确管理的向量流水；未验证 Cube pipe、跨核同步和异步多 buffer 流水。

## 根因

`TQue::EnQue/DeQue` 建立的事件面向 Vector pipe，不能替代标量 S pipe 与 MTE2/MTE3 之间的跨管依赖，导致标量读取未完成的 UB load 或 MTE3 读取尚未完成的标量写入。 机制链为： 1. `DataCopyPad` 在 MTE2 发起 GM→UB load； 2. 标量 C++ 循环在 S pipe 读取同一 UB； 3. `TQue` 的 V-pipe 事件没有为 `MTE2 → S` 建立等待； 4. 标量循环在 S pipe 写输出 UB； 5. MTE3 随后从输出 UB 发起 store，但原实现没有 `S → MTE3` 依赖； 6. 因此 kernel 可正常编译和退出，却把陈旧 UB 内容复制为整段常量。

## 解决配方

将 DMA 输入、标量输出和累加器改为 `TBuf`，用三道 `PipeBarrier<PIPE_ALL>()` 显式串行化 `MTE2(load) → S(scalar) → MTE3(store) → buffer reuse`。以下代码均为事件流真实写入的完整函数，未做语法改写。

### before

path: `operators/cummin/cummin.asc`；symbol: `KernelCummin<T>::Init`；修复前完整写入行 115–143。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                               int64_t totalLines, int64_t L, int64_t innerCount,
                               int64_t numBlocks)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        valuesGm_.SetGlobalBuffer((__gm__ T *)values);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices);

        totalLines_ = totalLines;
        L_ = L;
        innerCount_ = innerCount;
        numBlocks_ = numBlocks;
        blockIdx_ = AscendC::GetBlockIdx();

        // Per-core contiguous scan-line range (flattened li = outer*innerCount + inner)
        lineStart_ = (blockIdx_ * totalLines) / numBlocks;
        lineEnd_ = ((blockIdx_ + 1) * totalLines) / numBlocks;

        // Buffers sized to MAX_TILE elements (all dtypes fit UB, see design doc §3).
        // Path A (innerCount==1): step/valOut/idxOut as tile buffers; acc single scalar.
        // Path B: step(valOut/idxOut) tile buffers via TQue; accVal/accIdx TBuf (MAX_TILE).
        pipe_.InitBuffer(inQ_, PIPELINE_DEPTH, MAX_TILE * sizeof(T));
        pipe_.InitBuffer(outVQ_, PIPELINE_DEPTH, MAX_TILE * sizeof(T));
        pipe_.InitBuffer(outIQ_, PIPELINE_DEPTH, MAX_TILE * sizeof(int64_t));
        if constexpr (true) {
            pipe_.InitBuffer(accValBuf_, MAX_TILE * sizeof(CT));
            pipe_.InitBuffer(accIdxBuf_, MAX_TILE * sizeof(int64_t));
        }
    }
```

path: `operators/cummin/cummin.asc`；symbol: `KernelCummin<T>::ProcessPathA`；修复前完整写入行 157–210。

```cpp
    __aicore__ inline void ProcessPathA()
    {
        CT accVal;
        int64_t accIdx = 0;
        for (int64_t li = lineStart_; li < lineEnd_; ++li) {
            int64_t base = li * L_;          // contiguous line start (element offset)
            accVal = CT(0);                  // set on first element below
            bool inited = false;
            int64_t remaining = L_;
            int64_t off = 0;
            while (remaining > 0) {
                uint32_t cnt = static_cast<uint32_t>(remaining < MAX_TILE ? remaining : MAX_TILE);
                AscendC::DataCopyExtParams cp{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};

                auto inLocal = inQ_.template AllocTensor<T>();
                AscendC::DataCopyPad(inLocal, xGm_[base + off], cp, pad);
                inQ_.EnQue(inLocal);
                auto step = inQ_.template DeQue<T>();
                __ubuf__ T *sp = (__ubuf__ T *)step.GetPhyAddr();

                auto vOut = outVQ_.template AllocTensor<T>();
                auto iOut = outIQ_.template AllocTensor<int64_t>();
                __ubuf__ T *vp = (__ubuf__ T *)vOut.GetPhyAddr();
                __ubuf__ int64_t *ip = (__ubuf__ int64_t *)iOut.GetPhyAddr();

                for (uint32_t i = 0; i < cnt; ++i) {
                    T raw = sp[i];
                    CT b = ToCompute<T>(raw);
                    if (!inited) {
                        accVal = b; accIdx = off + i; inited = true;
                    } else if (CumminTake<CT>(accVal, b)) {
                        accVal = b; accIdx = off + i;
                    }
                    vp[i] = FromCompute<T>(accVal);
                    ip[i] = accIdx;
                }
                inQ_.FreeTensor(step);

                outVQ_.EnQue(vOut);
                outIQ_.EnQue(iOut);
                auto vS = outVQ_.template DeQue<T>();
                auto iS = outIQ_.template DeQue<int64_t>();
                AscendC::DataCopyPad(valuesGm_[base + off], vS, cp);
                AscendC::DataCopyPad(indicesGm_[base + off], iS,
                                     AscendC::DataCopyExtParams{1, cnt * (uint32_t)sizeof(int64_t), 0, 0, 0});
                outVQ_.FreeTensor(vS);
                outIQ_.FreeTensor(iS);

                off += cnt;
                remaining -= cnt;
            }
        }
    }
```

### after

path: `operators/cummin/cummin.asc`；symbol: `KernelCummin<T>::Init`；修复后工作区完整函数。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                               int64_t totalLines, int64_t L, int64_t innerCount,
                               int64_t numBlocks)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        valuesGm_.SetGlobalBuffer((__gm__ T *)values);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices);

        totalLines_ = totalLines;
        L_ = L;
        innerCount_ = innerCount;
        numBlocks_ = numBlocks;
        blockIdx_ = AscendC::GetBlockIdx();

        // Per-core contiguous scan-line range (flattened li = outer*innerCount + inner)
        lineStart_ = (blockIdx_ * totalLines) / numBlocks;
        lineEnd_ = ((blockIdx_ + 1) * totalLines) / numBlocks;

        // Buffers sized to MAX_TILE elements (all dtypes fit UB, see design doc §3).
        // Single-buffered TBuf + DataCopyPad + PipeBarrier<PIPE_ALL> (scalar runs on the S
        // pipe, which TQue's V-only EnQue/DeQue does not sync; PIPE_ALL serializes
        // MTE2(load)->S(scalar)->MTE3(store) correctly — baseline correctness-first).
        pipe_.InitBuffer(stepBuf_, MAX_TILE * sizeof(T));
        pipe_.InitBuffer(valOutBuf_, MAX_TILE * sizeof(T));
        pipe_.InitBuffer(idxOutBuf_, MAX_TILE * sizeof(int64_t));
        pipe_.InitBuffer(accValBuf_, MAX_TILE * sizeof(CT));
        pipe_.InitBuffer(accIdxBuf_, MAX_TILE * sizeof(int64_t));
    }
```

path: `operators/cummin/cummin.asc`；symbols: `KernelCummin<T>::StoreOut`、`KernelCummin<T>::ProcessPathA`；修复后构建并上板验证的完整函数。

```cpp
    __aicore__ inline void StoreOut(int64_t off, uint32_t cnt)
    {
        AscendC::DataCopyExtParams cpv{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyExtParams cpi{1, cnt * (uint32_t)sizeof(int64_t), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_ALL>();                 // S(scalar) writes done -> MTE3
        AscendC::DataCopyPad(valuesGm_[off], valOutBuf_.Get<T>(), cpv);
        AscendC::DataCopyPad(indicesGm_[off], idxOutBuf_.Get<int64_t>(), cpi);
        AscendC::PipeBarrier<PIPE_ALL>();                 // MTE3(store) done -> reuse buffers
    }

    // ----- Path A: contiguous line (dim is last axis) -----
    __aicore__ inline void ProcessPathA()
    {
        CT accVal;
        int64_t accIdx = 0;
        for (int64_t li = lineStart_; li < lineEnd_; ++li) {
            int64_t base = li * L_;          // contiguous line start (element offset)
            bool inited = false;
            int64_t remaining = L_;
            int64_t off = 0;
            while (remaining > 0) {
                uint32_t cnt = static_cast<uint32_t>(remaining < MAX_TILE ? remaining : MAX_TILE);
                AscendC::DataCopyExtParams cp{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                AscendC::DataCopyPad(stepBuf_.Get<T>(), xGm_[base + off], cp, pad);
                AscendC::PipeBarrier<PIPE_ALL>();   // MTE2(load) done -> S(scalar)

                __ubuf__ T *sp = (__ubuf__ T *)stepBuf_.Get<T>().GetPhyAddr();
                __ubuf__ T *vp = (__ubuf__ T *)valOutBuf_.Get<T>().GetPhyAddr();
                __ubuf__ int64_t *ip = (__ubuf__ int64_t *)idxOutBuf_.Get<int64_t>().GetPhyAddr();
                for (uint32_t i = 0; i < cnt; ++i) {
                    T raw = sp[i];
                    CT b = ToCompute<T>(raw);
                    if (!inited) {
                        accVal = b; accIdx = off + i; inited = true;
                    } else if (CumminTake<CT>(accVal, b)) {
                        accVal = b; accIdx = off + i;
                    }
                    vp[i] = FromCompute<T>(accVal);
                    ip[i] = accIdx;
                }
                StoreOut(base + off, cnt);
                off += cnt;
                remaining -= cnt;
            }
        }
    }
```

## 设计决策

- 选择 `TBuf + PIPE_ALL`：它直接覆盖 MTE2、S、MTE3 的顺序关系，能用最小机制变更确认根因并恢复正确性；
- 放弃继续依赖 `TQue`：TQue 适合 Vector pipe 消费/生产，本例标量循环不在它覆盖的执行管；
- 未采用的候选：用 `HardEvent` 分别建立 `MTE2→S`、`S→MTE3` 与复用依赖，预计同步范围更窄，但本轨迹没有实现或实测；
- 当前方案属于 `safety_fallback`：以全管屏障换取正确性，不能视为性能最优路线。

## 系统信息

- 目标：Ascend950PR_957b / dav-3510；
- 工具链：CANN 9.0.1，bisheng；
- 工程文件：`operators/cummin/cummin.asc`、`operators/cummin/test_cummin.py`、`operators/cummin/CMakeLists.txt`；
- 验证入口：远端 helper build、NPU pytest、cann-bench eval。

## 性能影响

修复后的标量 + `PIPE_ALL` 路线在 950 上，相对 cann-bench 参考路线平均约为 `0.04x`（参照路线：cann-bench 内置 baseline；证据等级: tool_observed）。这说明该方案只适合作为正确性 fallback，后续应以精细事件同步或向量化路线替代。

## 预防

- 看到 `DataCopyPad → __ubuf__ 标量读` 时，必须检查 `MTE2→S` 依赖；
- 看到 `__ubuf__ 标量写 → DataCopyPad store` 时，必须检查 `S→MTE3` 依赖；
- 不要因为使用了 TQue 就默认所有执行管都已同步，先确认真正消费/生产 UB 的 pipe；
- 首次验证若出现“编译成功、无 runtime error、整段输出同一常量”，先查跨管同步和 buffer 复用，不先改数值公式。

## 验证信号

本知识来自一次 Ascend C 前缀扫描 kernel 的首次上板失败与同步修复：同一源码在替换 buffer/sync 路径前后连续构建和复验。

- 修复前首个 NPU 数据用例失败，结果全为同一错误常量且 `max diff=11.84765625`（来源平台：950；证据等级：tool_observed）；
- 修复后完整 NPU pytest `40/40` 通过（来源平台：950；证据等级：tool_observed）；
- 干净重建后再次得到 `40 passed`（来源平台：950；证据等级：tool_observed）；
- 最终 cann-bench `20/20` 通过，浮点 values 误差为 0，整数 values 与全部 indices 精确匹配（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：在标量读取前增加 MTE2→S 的 PIPE_ALL、标量写后增加 S→MTE3 的 PIPE_ALL 后，n=5/8/64/1024 四组 1D+d1 探针均 ok=True、det=True（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：补齐 Scalar pipe 与 MTE 的显式依赖后，修后 NPU pytest 40/40 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `PIPE_ALL` 会过度串行化，吞吐敏感或可流水的 kernel 不应直接把它当最终方案；
- 完全由 Vector pipe 完成的 TQue 流水不需要套用本卡结论；
- A2、A3 与其他 CANN 版本未验证，推断同类 pipe 依赖可迁移，但属于“未验证，推断可迁移”。

## 坏实践

- 把 `EnQue/DeQue` 当成覆盖所有执行管的通用 fence；
- 在输出全常数时先修改比较、Cast 或容差，而不检查 DMA 与标量 UB 访问的先后关系；
- 为了“先通过”而放宽测试精度阈值；该轨迹没有这样做，正确修复发生在 kernel 同步层。

## 误诊

整段同一常量容易被误认为“最终累加值被广播”或“数值公式错误”；本例的判别点是计算逻辑未改、只修复跨管同步后全部精度用例恢复。

## 后续增量（2026-09-16，并行策展批：HardEvent 精细路线实测、地址可见性与“标量写灌 TQue”错配两个新形态）

两条独立轨迹（bench_no_harness_gather 与 cross entropy loss 类算子，均为 Ascend950PR/arch35、CANN 9.1.0）把原卡“未采用的候选：用 HardEvent 分别建立依赖”补成实测路线，并新增两个错配形态。本卡证据自此覆盖三个独立 trace。

**1. HardEvent 精细同步路线实测（原卡候选转正）**：gather kernel 中 S pipe 的读写结果与它计算出的地址偏移对 MTE2/MTE3 默认不可见——表现为**输出首元素错值（2.0/7.0/11.0 等，超出输入值域）其余全 0**，且 `pipe_barrier<PIPE_ALL>` 包裹 GetValue/SetValue 后输出逐位不变（无效）。修复把数据搬运改 MTE2→32B 对齐 tempBuf→MTE3 双段，index 值仍由 S pipe `GetValue` 读取（批量 index 先 MTE2 搬入 UB、读前 `SetFlag/WaitFlag<HardEvent::MTE2_S>`），S pipe 算出的 srcOffset/outOffset 通过 `S_MTE2`/`S_MTE3` 事件对对 MTE 地址可见：

```cpp
            // S → MTE2 同步：确保 srcOffset（S pipe 计算）对 MTE2 可见
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyPad(tempLocal, xGm[srcOffset], elemParams, {true, 0, 0, 0});
            // MTE2 → MTE3 同步
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            // S → MTE3 同步：确保 outOffset（S pipe 计算）对 MTE3 可见
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(yGm[outOffset], tempLocal, outParams);
```

判别探针：用 `Duplicate`（V pipe）填充常量 42 替换整个逐元素循环，输出全 42.0 即隔离故障到 S pipe（CopyOut/buffer/MTE3 路径正常）。机制锚点：`kernel_reg.h` 的 `static_assert(pipe != PIPE_S, "PipeBarrier<PIPE_S> is not supported on current device!")` 与 `kernel_event.h` 的 HardEvent 枚举存在性。**限制**：该事件对方案在紧凑长循环（数百元素）中会失步（约 32B 流水深度后），须改每迭代 `pipe_barrier(PIPE_ALL)`——见同批「紧循环 SetFlag 失步需逐迭代屏障」卡。

**2. 新形态：标量写（SetValue）灌 VECIN TQue 的事件模型错配**：cross entropy loss kernel 的 ND gather 分支把 `AllocTensor` 取出的 VECIN 队列 tensor 用标量 `SetValue` 逐元素灌入收集结果，再 `EnQue/DeQue` 交给 V 计算——TQue 的 EnQue/DeQue 围绕“数据由 MTE2 搬运填入”设计（事件模型按 MTE2 生产者），标量写（头文件标注 `__inout_pipe__(S)`）不被覆盖，`DeQue` 返回时写入数据不保证对后续 V 计算（ReduceMax 等）可见。表现：`none` 模式输出恒等于 ln(C)（全 0 输入的解析值，即读到残留 0）；`mean`/`sum` 模式误差被聚合摊薄（如 MERE=2.55e-04 恰超阈值，差点放过）；2D 路径（MTE2 DataCopyPad 灌入）全部通过——失败按“数据来源分支”分桶是关键判别。修复：缓冲从 TQue 换 TBuf（去掉 EnQue/DeQue/FreeTensor），2D 路径 DataCopyPad 后手动补 `SetFlag/WaitFlag<HardEvent::MTE2_V>`，ND 路径标量写依赖同核流水自然有序（修后 22 用例全过证明足够；`V_V` 事件在该 arch 编译不可用）。

**3. 判别方法补充**：遇到“首元素错值+其余全 0”先做 V pipe（Duplicate 常量）对照；遇到“输出恒为全 0 输入的解析值”先按数据来源分支分桶（MTE2 灌入 vs 标量写灌入）——两者分别指向 S×MTE 可见性与 TQue 事件模型错配。

**4. 坏实践补充**：用 `mean`/`sum` 聚合用例当唯一验收口径——聚合会摊薄“读到未就绪数据”类错误，让回归漏网；给每个缓冲标注“生产者流水线”（MTE2 还是标量写）——MTE2 生产 → TQue 或 TBuf+MTE2→V 事件，标量写生产 → TBuf + 自然有序（或经实测可用的显式事件）。

**5. 验证信号**：

- gather 轨迹：修复前 8 用例数值错误（首元素错值、其余 0.0）+ 2 用例 507035，`pipe_barrier<PIPE_ALL>` 包裹无效（来源平台：950；证据等级：tool_observed）；Duplicate 对照输出全 42.0（来源平台：950；证据等级：tool_observed）；MTE 双段 + `MTE2_S` 同步（无 S→MTE 地址同步）3/10；补 `S_MTE2`/`S_MTE3` 后 9/10 bitwise=YES（剩余为紧循环深度问题，另卡）（来源平台：950；证据等级：tool_observed）。
- cross entropy 轨迹：修复前全量 105 用例 81 通过/24 失败（22 个 rank 3~8 精度失败 + 2 个空 tensor 段错误），L1_024 输出恒为 ln(10)（来源平台：950；证据等级：tool_observed）；换 TBuf + MTE2_V 事件后 105/105 通过、UT 74/74 无回归，L1_020/L1_024 MERE=3.55e-08/3.31e-08，回归用例 L1_027 MERE=3.36e-07（来源平台：950；证据等级：tool_observed）。