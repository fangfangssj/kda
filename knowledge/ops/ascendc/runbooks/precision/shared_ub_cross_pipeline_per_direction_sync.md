---
type: "Runbook"
title: "共享 UB 跨流水线复用须按方向显式同步"
description: "同一 UB 缓冲被 MTE2(搬入)/V(向量)/MTE3(搬出) 多条流水线交替读写而无显式跨流水线同步时，输出错值/NaN/垃圾索引且无报错；单流水线 PipeBarrier 与 TQue 单向事件都不构成跨流水线 happens-before，须按依赖方向 SetFlag/WaitFlag（每方向独立 eventId）或 PIPE_ALL 包夹。"
tags: [task.debugging, topic.implementation.synchronization, topic.implementation.pipeline, topic.precision]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# 共享 UB 跨流水线复用须按方向显式同步

## 现象

Ascend950（arch35/DAV_3510）上多个相互独立的算子开发会话出现同一族失败：kernel 编译通过、aclnn 调用成功、无任何报错，但输出大面积损坏，形态包括：

- 纯 AIV 多步 kernel（matmul→RMSNorm→matmul→RoPE 交替，中间结果经 GM workspace 传递）：c_kv 输出 MERE=7.29e+02（actual=73.0 vs golden=4.39e-04）、query_rope NaN——V（Cast/向量计算）、MTE2（读 GM）、MTE3（写 GM）之间只有 `PipeBarrier<PIPE_V>()`，且多个依赖方向复用同一个 eventId。
- GRU 算子 fp16/bf16 路径：约三分之一元素数据损坏（mismatch 38.3%/32.9%，MARE 数百）——`TBuf` 中转缓冲被 MTE2（DataCopyPad 写入）与 V（Cast 读取）交替访问，只做了 MTE2→V 单向同步，缺 Cast 之后的 V→MTE2 反向同步；fp32 路径（走 TQue 队列）完全正常。
- 多层 GRU 层间拷贝：多层+双向用例 y MERE=3.43~5.99、mismatch 84%~86%，单层全对——层间 GM→UB→GM 拷贝复用 VECOUT TQue 承载 MTE2→MTE3 数据流，缺同步。
- GQA 注意力 kernel：6 个用例输出与 golden 完全不符（MERE=60+），printf 探针实锤 Q tile 的值等于 K tile 的值（`q[0]=0.497314 k[0]=0.497314`，非确定出现）——搬运中转 buffer 被上一次的 V 消费者（Cast 读）与这一次的 MTE2 生产者（DataCopyPad 写）无屏障交替使用。
- TopK 算子首次上板：y MERE=inf、idx 为 int64 越界垃圾大数（量级 ~2^62）——`TBuf` 单缓冲数据流的 Duplicate(VEC)→DataCopyPad(MTE2)、CopyIn(MTE2)→Compute(VEC)、Compute(VEC)→CopyOut 三个边界全部无同步。

## 适用场景

- 工程形态：AscendC kernel（纯 AIV 向量路线）用 `TBuf`/裸 LocalTensor 手工管理 UB 缓冲、跨流水线复用同一缓冲的数据流；含"搬运→计算→搬运"循环、多步计算链经 GM workspace 传递、层间/阶段间中转拷贝的算子。
- API 路径：`SetFlag/WaitFlag<HardEvent::V_MTE3 / MTE3_MTE2 / V_MTE2 / MTE2_V / MTE2_MTE3>`（`kernel_operator_block_sync_intf.h` / `kernel_event.h`）；`PipeBarrier<PIPE_ALL>`；`TQue<TPosition::VECIN/VECOUT>`；`DataCopyPad`（MTE2/MTE3）；`Cast/Duplicate/Mul/ReduceSum`（V）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0 与 9.2.0（四个独立会话）。
- 适用粒度：platform_wide——该平台 AIV 流水线模型的架构性约束，与算子无关。
- 路线优先级：recommended_fastpath（正确性必需，无替代）。
- 明确不适用 / 未验证场景：使用 TQue（EnQue/DeQue）且队列方向恰好覆盖实际数据流方向时的自动同步路径；Cube/Mix 模式未验证；其他架构未验证（推断可迁移）。

## 根因

ascend950 的 MTE2（GM→UB 搬入）、V（向量计算）、MTE3（UB→GM 搬出）是三条独立流水线，`TBuf` 只是内存分配、不提供任何自动同步；`PipeBarrier<PIPE_V>` 只保证 V 流水线内部有序，`PipeBarrier<PIPE_MTE2>` 只等 MTE2 流水线——单流水线屏障不构成跨流水线 happens-before。TQue 的位置语义也只绑定单一方向对（VECIN≈MTE2→V，VECOUT≈V→MTE3），不覆盖其它方向。凡同一 UB 缓冲跨流水线复用，每个依赖方向都必须显式同步，且**不同依赖方向要使用不同的 eventId（同号复用会互相干扰产生 NaN）**。

依赖方向矩阵与各自现象：

1. **V→MTE3（生产者-消费者 RAW）**：V（Cast/Duplicate/标量写）写 UB 后直接发起 DataCopyPad 写 GM，MTE3 可能读到旧数据 → 输出整体错值。
2. **V→MTE2（缓冲复用 WAR）**：V 还在读缓冲时，下一轮 MTE2 DataCopyPad 覆写同一缓冲 → 部分元素被新旧混杂数据污染（典型指纹：约 1/3 元素错、fp32 路径不受影响、非确定出现、Q/K 串数据）。
3. **MTE2→V（搬运就位 RAW）**：DataCopyPad 异步返回不代表搬运完成，V 立即读取读到未就位数据。
4. **MTE2→MTE3（纯搬运流 RAW）**：GM→UB→GM 中转拷贝中 MTE3 在 MTE2 写完成前读 UB。
5. **MTE3→MTE2（GM 写读链/跨轮 WAR）**：本步 MTE3 写回未完成，下一步 MTE2 已开始读；或上一轮 MTE3 未读完 UB、下一轮 MTE2 已覆写。

## 解决配方（正解一：按方向 SetFlag/WaitFlag + 独立 eventId）

以纯 AIV 多步 kernel 的写回段为例。修复前（Cast 后仅 PipeBarrier，无任何跨流水线同步）：

```cpp
            // Cast fp32 → bf16
            LocalTensor<bfloat16_t> resultBf16 = xRaw;  // 复用 xRaw 缓冲
            Cast<bfloat16_t>(resultBf16, result, RoundMode::CAST_RINT, tileN32);
            PipeBarrier<PIPE_V>();

            DataCopyExtParams ocp;
            ocp.blockCount = 1;
            ocp.blockLen = static_cast<uint32_t>(tileN * sizeof(bfloat16_t));
            ocp.srcStride = 0;
            ocp.dstStride = 0;
            DataCopyPad(dstGm[dstOff + nProcessed], resultBf16, ocp);

            nProcessed += tileN;
```

修复后（写回段终版：V_MTE3 保证 Cast 产物对 MTE3 可见，MTE3_MTE2 保证 GM 写回对下一步读取可见）：

```cpp
            // 写回结果
            if constexpr (std::is_same_v<DST_T, bfloat16_t>) {
                // Cast fp32 → bf16, 写回 GM
                LocalTensor<bfloat16_t> resultBf16 = xRawBuf_.Get<bfloat16_t>();
                Cast<bfloat16_t>(resultBf16, result, RoundMode::CAST_RINT, tileN32);
                SetFlag<HardEvent::V_MTE3>(eventId3_);
                WaitFlag<HardEvent::V_MTE3>(eventId3_);
                DataCopyExtParams ocp;
                ocp.blockCount = 1;
                ocp.blockLen = static_cast<uint32_t>(tileN * sizeof(bfloat16_t));
                ocp.srcStride = 0;
                ocp.dstStride = 0;
                DataCopyPad(dstGm[dstOff + nProcessed], resultBf16, ocp);
            } else {
                // fp32 直接写回 GM (workspace)
                SetFlag<HardEvent::V_MTE3>(eventId3_);
                WaitFlag<HardEvent::V_MTE3>(eventId3_);
                DataCopyExtParams ocp;
                ocp.blockCount = 1;
                ocp.blockLen = static_cast<uint32_t>(tileN * sizeof(float));
                ocp.srcStride = 0;
                ocp.dstStride = 0;
                DataCopyPad(dstGm[dstOff + nProcessed], result, ocp);
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventId4_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId4_);
```

缓冲复用 WAR 同步（输入加载段：Cast 读完 xRaw 后才允许下一次 DataCopyPad 覆写同一 buffer）：

```cpp
        DataCopyPad(xRaw, srcGm[srcOff], xcp, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        Cast<float>(xFp32, xRaw, RoundMode::CAST_NONE, N32);
        // V→MTE2 同步: 确保 Cast 读完 xRaw 后 gamma 加载才能覆写
        SetFlag<HardEvent::V_MTE2>(eventId2_);
        WaitFlag<HardEvent::V_MTE2>(eventId2_);
        PipeBarrier<PIPE_V>();
```

事件号分配（每个依赖方向独立 eventId，终版声明）：

```cpp
    int8_t eventId_ = 0;   // MTE2→V
    int8_t eventId2_ = 1;  // V→MTE2
    int8_t eventId3_ = 2;  // V→MTE3
    int8_t eventId4_ = 3;  // MTE3→MTE2
```

要点：**每个方向一个专用事件号**；事件 ID 也可经 `TPipe::AllocEventID<HardEvent::...>()` 申请、`ReleaseEventID` 归还（`SetFlag/WaitFlag` 是自由函数而非 TPipe 成员，另卡记录）。

### 变体一：V→MTE2 反向同步（Cast 中转缓冲循环复用，GRU fp16/bf16 路径）

`TBuf` 中转缓冲被"MTE2 写（DataCopyPad 搬入权重行/bias/x/h0）↔ V 读（Cast 升 fp32）"循环交替使用时，只在 DataCopyPad 后做 MTE2→V 单向同步不够——第 n+1 轮 DataCopyPad 可以在第 n 轮 Cast 读完成之前覆盖缓冲。修复 = 在**每一处** Cast 之后插入 V→MTE2 事件对（独立事件号）：

```cpp
        } else {
            LocalTensor<D_T> wDataRaw = castFloatBuf_.Get<D_T>();
            DataCopyExtParams wcp_d; wcp_d.blockCount=1;
            wcp_d.blockLen=static_cast<uint32_t>(K*sizeof(D_T));
            wcp_d.srcStride=0; wcp_d.dstStride=0;
            DataCopyPad(wDataRaw, wGm[wOff + n * K], wcp_d, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(eventId_);
            WaitFlag<HardEvent::MTE2_V>(eventId_);
            wFloat = reduceTmpBuf_.Get<float>();
            Cast<float>(wFloat, wDataRaw, RoundMode::CAST_NONE, static_cast<int32_t>(K));
            // V→MTE2 同步: 确保 Cast 读完 castFloatBuf_ 后下一轮 DataCopyPad 才能写入
            SetFlag<HardEvent::V_MTE2>(eventId2_);
            WaitFlag<HardEvent::V_MTE2>(eventId2_);
            PipeBarrier<PIPE_V>();
            ...（后续 Mul/ReduceSum 消费 wFloat）
        }
```

配套成员声明（`eventId2_` 与既有 MTE2_V 事件 `eventId_=0` 区分）：

```cpp
    int8_t eventId_ = 0;   // MTE2→V 同步事件
    int8_t eventId2_ = 1;  // V→MTE2 同步事件 (保护 castFloatBuf_ 跨流水线复用)
```

### 变体二：GM→UB→GM 纯搬运流不得复用计算队列（多层 GRU 层间拷贝）

修复前（用 VECOUT TQue 做层间 GM→UB→GM 拷贝——VECOUT 位置只保证 V→MTE3 方向依赖，不建立 MTE2→MTE3 依赖）：

```cpp
        LocalTensor<float> data = outQueueHT_.template AllocTensor<float>();
        DataCopyPad(data, gmLayerOut_[copied], cp, {false, 0, 0, 0});
        outQueueHT_.EnQue(data);
        data = outQueueHT_.template DeQue<float>();
        DataCopyPad(gmLayerIn_[copied], data, cp);
        outQueueHT_.FreeTensor(data);
```

修复后（改用 `TBuf` 缓冲 + 显式双向事件同步，verbatim 函数级）：

```cpp
template <typename D_T, int HAS_BIAS>
__aicore__ inline void BenchNoHarnessGru<D_T, HAS_BIAS>::CopyLayerOutputToLayerInput()
{
    int64_t totalElements = seqLen_ * actualBatch_ * numDirections_ * hiddenSize_;
    // 使用 ghBuf_ (TBuf, VECCALC) 作为拷贝缓冲区
    // ghBuf_ 大小 = actualBatch * gateSize >= max(gateSize, ubElementCount)
    int64_t bufElements = (gateSize_ > ubElementCount_) ? gateSize_ : ubElementCount_;
    if (bufElements < 1) bufElements = 1;

    DataCopyExtParams cp; cp.blockCount=1;
    cp.srcStride=0; cp.dstStride=0;

    int64_t copied = 0;
    while (copied < totalElements) {
        int64_t thisChunk = bufElements;
        if (copied + thisChunk > totalElements) thisChunk = totalElements - copied;
        cp.blockLen = static_cast<uint32_t>(thisChunk * sizeof(float));

        // MTE2: GM(layerOut) → UB(ghBuf)
        LocalTensor<float> copyBuf = ghBuf_.Get<float>();
        DataCopyPad(copyBuf, gmLayerOut_[copied], cp, {false, 0, 0, 0});
        // 显式 MTE2→MTE3 同步 (TQue VECOUT 不适用于 MTE2→MTE3)
        SetFlag<HardEvent::MTE2_MTE3>(eventId_);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId_);
        // MTE3: UB(ghBuf) → GM(layerIn)
        DataCopyPad(gmLayerIn_[copied], copyBuf, cp);
        // MTE3→MTE2 同步 (确保 MTE3 写入完成后再下一轮 MTE2 读取)
        SetFlag<HardEvent::MTE3_MTE2>(eventId_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId_);

        copied += thisChunk;
    }
}
```

### 变体三：PIPE_ALL 包夹模式（TBuf + 手动 DataCopy 的通用防御形态）

官方 api-pipeline 文档方案二：跨流水线同步用 `PipeBarrier<PIPE_ALL>()`——每次 DataCopy 前后各一个，前屏障保证上一流水线对该缓冲的读写完成，后屏障保证本次搬运完成后再进入下一流水线（GQA 向量版 kernel 的 LoadHalfRow/StoreHalfRow 终版，verbatim）：

```cpp
template <typename D_T_Q, int IS_CAUSAL, int SCALE_MODE>
__aicore__ inline void BenchNoHarnessGqa<D_T_Q, IS_CAUSAL, SCALE_MODE>::LoadHalfRow(
    const GlobalTensor<D_T_Q>& srcGm, int64_t rowOffset,
    LocalTensor<D_T_Q>& dstUb, int64_t D)
{
    PipeBarrier<PIPE_ALL>();  // 确保前一个 V(Cast) 读 dstUb 完成，才能 MTE2 写
    DataCopy(dstUb, srcGm[rowOffset], static_cast<int32_t>(D));
    PipeBarrier<PIPE_ALL>();  // 确保 MTE2 搬运完成，才能 V(Cast) 读
}
```

```cpp
template <typename D_T_Q, int IS_CAUSAL, int SCALE_MODE>
__aicore__ inline void BenchNoHarnessGqa<D_T_Q, IS_CAUSAL, SCALE_MODE>::StoreHalfRow(
    GlobalTensor<D_T_Q>& dstGm, int64_t rowOffset,
    LocalTensor<D_T_Q>& srcUb, int64_t D)
{
    PipeBarrier<PIPE_ALL>();  // 确保前一个 V(Cast) 写 srcUb 完成，才能 MTE3 读
    DataCopy(dstGm[rowOffset], srcUb, static_cast<int32_t>(D));
    PipeBarrier<PIPE_ALL>();  // 确保 MTE3 搬运完成
}
```

同一模式在 TopK 算子三个边界上的应用（CopyIn 内 Duplicate(VEC)→DataCopyPad(MTE2) 与 DataCopyPad(MTE2)→Compute(VEC)，及 Process 循环内 Compute(VEC)→CopyOut）：

```cpp
    LocalTensor<T> srcValue = srcValueBuf.Get<T>();
    // 先用极值填充整个 srcValue（innerPad 个元素），确保 inner~innerPad-1 的 padding 不被选入 top-k
    const T padVal = static_cast<T>(GetPadValue(isLargest_));
    Duplicate<T>(srcValue, padVal, innerPad_);
    // Duplicate(VEC) → DataCopyPad(MTE2) 写同一 UB，需同步保证 Duplicate 先完成，否则 padVal 覆盖真实数据
    PipeBarrier<PIPE_ALL>();
    ...（DataCopyPad 搬入真实数据）
    // DataCopyPad(MTE2) 异步，Compute(VEC) 读 srcValue 前需同步，保证搬运数据就位
    PipeBarrier<PIPE_ALL>();
```

```cpp
    for (int64_t i = 0; i < numRows; i++) {
        int64_t rowIdx = startRow + i;
        if (rowIdx >= outter_) {
            break; // 越界保护
        }
        CopyIn(rowIdx);
        Compute();
        // Compute(VEC: TopK/Cast) → CopyOut(MTE2: DataCopyPad) 需同步，保证计算结果就位后再回写
        PipeBarrier<PIPE_ALL>();
        CopyOut(rowIdx);
    }
```

### 判别与探针方法

- 触发信号：多步骤/多段 AIV kernel 输出"整体错值/NaN/垃圾 idx 但无任何报错"；"单层/单段/对齐路径全对、跨段/低精度路径全错"的组合对照（单层全过 + 多层全错 = 层间传递路径问题；fp32 全过 + fp16/bf16 部分错 = 中转缓冲跨流水线复用）。
- 探针一（换实现不动结果）：把归约/计算 API 换成等价的其他实现（Pattern→Level2→手动求和），三轮输出与误差逐位不变 ⇒ 输入数据通路（搬运/竞争）有问题，计算逻辑不是因，别再改算法。
- 探针二（kernel 内 printf 守卫打印）：`ASCENDC_DEBUG=1`，用"0 号核 + 首头 + 首 tile"守卫打印两块应不同的 tile 的前几个值——`q[0]==k[0]` 即搬运层串数据实锤；输出端对账：打印写出缓冲首值与 host 读回值对比。交付前移除探针。
- 探针三（读 kernel 源查三件事）：buffer 声明是 `TBuf` 还是 `TQue`；段间有无 `PipeBarrier`/`EnQue`/`DeQue`；同一段 UB 是否先后被 VEC 指令与 DMA 读写。任一为否即高度命中。
- 验证性实验：在疑似边界各加 `PipeBarrier<PIPE_ALL>()` 或事件对重建上板——输出恢复精确即证实同步根因。

## 设计决策

- 为什么按最小依赖粒度成对 Set/Wait 而不是函数级统一屏障：三种方向各自独立，避免过度串行化（时延敏感算子）；PIPE_ALL 包夹是改动最小的防御形态，代价是全流水线停顿（性能未量化）。
- 为什么 eventId 必须按方向分开：实测把 `MTE3_MTE2` 挂到与 `MTE2_V` 相同的事件号后出现 NaN 回归（事件互相踩踏），拆成独立事件号后消除。
- 为什么 GM→UB→GM 纯搬运不用 TQue：TQue 位置（VECIN/VECOUT）绑定固定流水方向语义；纯搬运流没有 V 参与，EnQue/DeQue 提供的依赖与实际需要的依赖不匹配，编译器不报错但静默产生数据竞争。
- 为什么 MTE3_MTE2 放在写回段末尾：它保护的是"本函数的 GM 写 → 下一函数的 GM 读"这条链，放在最后一个 DataCopyPad 之后语义最准确。
- 被放弃方案：改用 `TQue` 自动同步（改动面大；且方向不匹配时不安全，见变体二）；GM workspace 缓存中间结果（另一轨迹实测未修复且引入 507035）。
- 当前方案路线档位：recommended_fastpath（正确性必需；事件对 vs PIPE_ALL vs TQue 的性能对比未量化，见遗留问题）。

## 系统信息

- 硬件/软件：Ascend950PR（arch35/DAV_3510），CANN 9.1.0 与 9.2.0；硬同步事件枚举见 `x86_64-linux/ascendc/include/basic_api/impl/kernel_event.h`（MTE3_V/V_MTE3/MTE3_MTE2/MTE2_MTE3/MTE2_V/V_MTE2 等）与 `kernel_operator_block_sync_intf.h`。
- 实例算子（四个独立会话）：BenchNoHarnessMlaProlog（纯 AIV 多步链，`op_kernel/arch35/mla_prolog.h`）、BenchNoHarnessGru（GRU 门计算/层间拷贝，`op_kernel/arch35/bench_no_harness_gru.h`）、BenchNoHarnessGqa（GQA 注意力，`op_kernel/arch35/bench_no_harness_gqa.h`）、BenchNoHarnessTopK（TopK，`op_kernel/arch35/bench_no_harness_top_k.h`）。

## 预防

- 写 AIV kernel 先画流水线依赖表：每个 UB buffer 的生产者与消费者各属哪条流水线（MTE2/V/MTE3/标量），跨界必配 SetFlag/WaitFlag 或 PIPE_ALL，每个方向独立 eventId。
- `PipeBarrier<PIPE_V/PIPE_MTE2/PIPE_MTE3>` 只当本流水线内部顺序屏障使用，不当作跨流水线同步。
- 任何 `TBuf` 被"流水线 A 写、流水线 B 读、循环复用"时，先画双向同步清单：A→B 一对、B→A 一对，缺一不可；用 TQue 的场景确认队列方向覆盖实际数据流。
- review 低精度支持/多段重构 patch 时，grep 所有 `Cast<`/`DataCopyPad` 调用点，逐点检查其源缓冲的前序写者与后继写者属于哪条流水线。
- 多层/多段结构的 ST 用例必须包含"单段 vs 多段"对照组合。

## 验证信号

来源说明：以下信号来自四个相互独立的算子开发会话（均为 Ascend950PR 上板实测），按会话分组。

多步 AIV kernel（CANN 9.2.0）：

- 同步前：c_kv MERE=7.29e+02（actual=73.0 vs golden=4.39e-04）、query MERE=2.40e+02、query_rope NaN（来源平台：950；证据等级：tool_observed）。
- 加 V_MTE3（单变量对照）：c_kv PASS MERE=1.34e-03、MARE=7.75e-03（来源平台：950；证据等级：tool_observed）。
- 加 V_MTE2：k_rope MERE 从 2.31e+02 降到 9.571e-03、query_rope 从 71.0 降到 1.88e-02（来源平台：950；证据等级：tool_observed）。
- eventId 复用回归观测：MTE3_MTE2 与 MTE2_V 共用同一事件号时 c_kv/k_rope 出现 NaN，拆分独立 eventId 后 NaN 消失（来源平台：950；证据等级：tool_observed）。
- 全部同步 + 其余根因修复后：四输出全 PASS（query 8.41e-06 / query_rope 7.18e-06 / c_kv 0.00 / k_rope 0.00），复跑一致（来源平台：950；证据等级：tool_observed）。

GRU fp16/bf16 Cast 中转缓冲（CANN 9.2.0）：

- 修复前 fp16 y：MERE=4.80e-01、MARE=3.94e+02、3918/10240 不匹配（38.3%）；bf16 y：MERE=4.57e-01、MARE=4.64e+02、3374/10240（32.9%）；fp32 用例全部 PASS（走 TQue 队列路径）（来源平台：950；证据等级：tool_observed）。
- 修复后 L0 5/5：fp16 y MERE=1.18e-06/MARE=1.08e-03；bf16 y MERE=3.40e-06/MARE=1.26e-02；fp32 与修复前逐位一致无回归；ST 总计 33/33、UT 51/51、PyTorch ST 全量复验 33/33（来源平台：950；证据等级：tool_observed）。

GRU 层间拷贝（CANN 9.2.0，同会话另一独立修复）：

- 修复前：2 层双向类用例 y MERE=3.43~5.99、mismatch 84%~86%；单层用例全部 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后：15 条 fp32 子集 14/15 PASS——2 层 MERE=5.35e-06、3 层 1.54e-05、2 层双向 1.27e-05（唯一残余失败为大 shape 精度的另一独立问题）（来源平台：950；证据等级：tool_observed）。

GQA 搬运中转 buffer（CANN 9.1.0）：

- 修复前：6/6 用例 MERE=6.08e+01~7.51e+01；printf 实锤 `q[0]=0.497314 q[1]=-0.540039 k[0]=0.497314 k[1]=-0.540039`（Q==K 串数据，非确定出现）、`score[0,1]=0.698357 score[1,0]=0.698357` 对称污染（来源平台：950；证据等级：tool_observed）。
- 消除性证据：Pattern/Level2/手动求和三种归约实现输出与 MERE 逐位不变；跳过 `l0op::Contiguous` 亦一致（来源平台：950；证据等级：tool_observed）。
- 修复后：printf 显示 `q[0]=-0.067566 ... k[0]=0.497314`（Q≠K，污染消失）；输出端 `copyUb[0]=-0.052765` 与 host 读回 `npu[0]=-0.052765` 一致；该屏障修复带来确定的精度改善（decode 用例 MERE 由 6.14e+01 降至 1.17e+01；整体仍有其余缺陷未闭环，见失效边界）（来源平台：950；证据等级：tool_observed）。
- PIPE_ALL 包夹模式随终版组合验证：全量 27/27 PASS（fp16 MERE 2.18e-06~3.38e-06，含 bf16 用例）；终版 grep 单流水线 `PIPE_MTE2|PIPE_MTE3` 零残留（25 处 PipeBarrier 分布核对）（来源平台：950；证据等级：tool_observed）。

TopK 三边界（CANN 9.1.0）：

- 修复前：y `MERE=inf, mismatch=2157/2560`；idx 越界垃圾值（如 -4621113994562504248、4583381379665942452，量级 ~2^62）（来源平台：950 实机 ST；证据等级：tool_observed）。
- 修复后（三个边界补 `PipeBarrier<PIPE_ALL>()` 重建重装）：全 PASS——`y 比对: MERE=0.00e+00, MARE=0.00e+00`、`idx gather 校验: x.gather(dim, idx) == y` PASS；UT 回归 op_host 27/27 + op_api 13/13（来源平台：950 实机 ST；证据等级：tool_observed）。

## 失效边界

- 本卡闭合"跨流水线同步缺失"根因族；加同步后仍错的失败需另查布局错位、padding 填充、tiling 等独立根因（各实例会话中均另有独立问题另行成卡）。
- GQA 实例的输出方向（Cast 后 DataCopyPad 出 GM）同类竞争在该轨迹未完成正式屏障加固（靠 printf 的隐式同步观测到 GM 输出一致即停），整体精度未完全闭环；采用输出端方案前需补显式屏障并实测。
- PIPE_ALL 包夹在 GQA/TopK 轨迹未做单变量隔离（是按官方文档修正并随终版验证的防御性实现，不宣称"缺 PIPE_ALL 必崩"）；逐行 3 次屏障的性能代价显著劣于 TQue 双缓冲流水（机制成立，成对量化对比未测）。
- 事件对同步在时延敏感算子上的量化影响未测；更细粒度（如仅 pad 行）方案未评估。
- 仅验证 Ascend950PR（arch35）/CANN 9.1.0、9.2.0；其他架构与版本的事件语义未验证（推断可迁移）。
- 竞争是时序依赖的：部分用例可能碰巧通过，不要用"部分用例过了"反推同步正确。

**未决项**：

- 优先项：GQA 输出方向（Cast→MTE3）的显式屏障加固待补并实测；各 SetFlag/WaitFlag 与 PIPE_ALL 对时延的量化影响未测（大 batch 已有超时压力，需 profiling 数据）。
- PIPE_ALL 包夹 vs 事件对 vs TQue 双缓冲三方案的成对性能对比未做。
- "缺跨流水线同步时的精确失败形态"（数据错乱 vs 崩溃 vs 偶发）未做最小复现的单变量隔离实验。
## 坏实践

- `Cast`/向量写后仅 `PipeBarrier<PIPE_V>` 就发起 `DataCopyPad` 写 GM（缺 V→MTE3）；在"V 读 → MTE 写"交替复用的 buffer 前只加 `PipeBarrier<PIPE_MTE2>`（方向错：护不住在读的 V 消费者）。
- 同一 UB buffer 跨步骤/跨轮复用（V 读 → MTE2 写）不加 V→MTE2（WAR 冒险）。
- 多个 HardEvent 方向复用同一 eventId（实测 NaN 回归）。
- 因为"队列也能 AllocTensor/EnQue"就把 TQue 当通用 UB 缓冲用——位置语义不匹配时静默产生数据竞争，编译器不报错。
- 用 `TBuf` 承载输入/输出却按同步语义写代码（CopyIn 返回即认为数据就位）；同一段 UB 先后交给 VEC 指令与 DMA 写且无顺序保证。
- 见到 MERE 巨大/输出全垃圾就先改归约 API 形态/跳过 Contiguous/换数据类型/怀疑 tiling 或 TopK 参数——三轮等价实现输出逐位不变的实测证明全是无效动作；先查同步边界，一行屏障的验证成本远低于逐参数排查。
- 靠 printf 探针"碰巧"同步输出端（GetValue 隐式等待 V 完成）当成修复——那是调试副产物，交付前必须换成显式屏障并移除探针。
- 失败用例只看汇总指标不做"单段 vs 多段"组合对照，会误判为精度问题（MERE 高达 5.99 表面像精度，实为同步缺陷）。

## 误诊

- 多步 kernel 输出错值最早被归因于 GetValue 标量读与 Rsqrt（各自修复后错误未完全消失），同步缺失是第三个叠加根因；该归因演进为分析叙述，最终由 V_MTE3 单变量对照证实。
- TopK 轨迹初期怀疑 idx 垃圾值（~4.5e18 超出 int32 符号扩展范围）意味着 Cast 或内存地址错位——按位模式分析后排除：更可能是 Cast 的源（TopK 输出）本身是垃圾，即上游数据未就位；加屏障后 idx 恢复合法区间证实。
- GRU 轨迹曾把 fp16/bf16 失败归因为"只有 3 个 .o 二进制（少于 TilingKey 组合数）"的编译缺失，两度改 `ASCENDC_TPL_SEL` 重编无果——实际产物正常，真因是同步；也曾把门输出全错归因于 V→MTE2 事件冲突以外的布局因素（门分段布局是另一独立根因，另卡记录）。
- GQA 轨迹曾怀疑 Pattern `ReduceSum`/`ReduceMax` 跨行接口对齐要求、`l0op::Contiguous` 视图拷贝污染输入——等价替换实验输出逐位不变，排除。
