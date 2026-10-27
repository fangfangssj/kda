---
type: "Runbook"
title: "TBuf 跨 MTE2 与 Vector 须队列交接"
description: "GM 数据写入裸 TBuf 后立刻由 V pipe 消费时，A3 与 Ascend950 的独立轨迹均出现数值错误、NaN 或非有限值位置不一致；改为 VECIN TQue 交接后定向及完整测试通过。R4-R6：V→MTE3 RAW、MTE→V→S 链、V↔MTE WAW 三种方向同族；跨管线边界 PipeBarrier<PIPE_ALL> 亦可修复（不必强行队列化）。R7：跨迭代复用的多读者 TBuf（cos/sin 类）用 MTE2_V + V_MTE2 定向 SetFlag/WaitFlag 双向同步闭合，多读者生命周期不适合 TQue。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.data_copy", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-09-17T00:58:06Z"
platforms: ["a3", "950"]
---

# TBuf 跨 MTE2 与 Vector 须队列交接

## 现象

多条 A3 与 Ascend950 轨迹都在“GM 经 MTE2 写入裸 TBuf，随后由 V pipe 立即消费”的路径上失败：cos/sin 隔离探针与参考不一致，gamma 路径的设备结果与 golden 非有限值位置不一致，950 Softmax 路径则出现 NaN、极值或不归一化输出。改为 VECIN TQue 的 EnQue/DeQue 交接后，各轨迹的定向或完整测试均闭合。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`DataCopyPad`、`TBuf`、`TQue`、`MTE2` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：a3（CANN 版本未记录）与 950（CANN 9.0.1）。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

`DataCopyPad(gammaRaw, gammaGm_)` 由 MTE2 异步写 UB，紧随其后的 `Cast(gammaFp32, gammaRaw)` 由 V pipe 读取。裸 TBuf 没有在这两个调用之间表达生产者到消费者的队列交接；本次会话中，改为 TQue 的 EnQue/DeQue 后原非有限值位置不一致消失。该单跑只支持这条数据路径的因果闭环，不外推为所有 TBuf 的版本规范。

## 解决配方

### before

```cpp
    __aicore__ inline void LoadGamma()
    {
        uint32_t bytesPerRow = static_cast<uint32_t>(numCol_ * sizeof(T));
        AscendC::DataCopyExtParams copyParams{1, bytesPerRow, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto gammaFp32 = gammaFp32Buf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyPad(gammaFp32, gammaGm_, copyParams, padParams);
        } else {
            auto gammaRaw = gammaRawBuf_.Get<T>();
            AscendC::DataCopyPad(gammaRaw, gammaGm_, copyParams, padParams);
            AscendC::Cast(gammaFp32, gammaRaw, AscendC::RoundMode::CAST_NONE, numCol_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
```

配套声明：before 的低精度分支在 `Init` 中 `InitBuffer(gammaRawBuf_, bytesPerRow)`，成员是 `TBuf<VECCALC> gammaRawBuf_`。

### after

```cpp
    __aicore__ inline void LoadGamma()
    {
        uint32_t bytesPerRow = static_cast<uint32_t>(numCol_ * sizeof(T));
        AscendC::DataCopyExtParams copyParams{1, bytesPerRow, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto gammaFp32 = gammaFp32Buf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyPad(gammaFp32, gammaGm_, copyParams, padParams);
        } else {
            // Load raw gamma through a TQue so EnQue/DeQue provides the
            // MTE2->V sync the Cast needs (a bare TBuf DataCopyPad here would
            // race the immediately-following vector Cast).
            auto gammaRaw = inQueueGamma_.AllocTensor<T>();
            AscendC::DataCopyPad(gammaRaw, gammaGm_, copyParams, padParams);
            inQueueGamma_.EnQue(gammaRaw);
            gammaRaw = inQueueGamma_.DeQue<T>();
            AscendC::Cast(gammaFp32, gammaRaw, AscendC::RoundMode::CAST_NONE, numCol_);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueGamma_.FreeTensor(gammaRaw);
        }
    }
```

配套声明：after 在相同 `if constexpr` 中改为 `InitBuffer(inQueueGamma_, 1, bytesPerRow)`，成员改为 `TQue<VECIN, PIPELINE_DEPTH> inQueueGamma_`。

## 设计决策

常驻 gamma 最终仍保存在 fp32 TBuf，只把“原始低精度 gamma 的 DMA→Cast 交接”放进 VECIN TQue。这样既获得事件同步，也避免让常驻缓冲占用队列生命周期。

## 性能影响

初始化每核只执行一次，增加一次 EnQue/DeQue 与队列槽；轨迹没有配对性能数据，不主张性能收益。

## 预防

- 逐条列出 MTE2 写入后首个消费者属于哪条流水；
- GM→UB 后若紧跟 V 操作，优先采用匹配位置的 TQue 交接；
- dtype 分支分别测试，不能用 fp32 通过推断低精度初始化也安全。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 复现一修复前：补齐逻辑偏移后的 cos/sin 隔离探针中，参考值前半段与 NPU 输出不一致、后半段一致，`maxdiff=9.788...`（来源平台：a3；证据等级: tool_observed）。
- 复现一修复后：TQue EnQue/DeQue handoff 后定向探针与参考逐元素一致，`maxdiff=0`（来源平台：a3；证据等级: tool_observed）。
- 复现一修复后：clean build 后完整测试 `18 passed`（来源平台：a3；证据等级: tool_observed）。
- 复现二修复前：NaN 输入路径的设备结果与 golden 非有限值位置不一致（来源平台：a3；证据等级: tool_observed）。
- 复现二修复后：改用 TQue handoff 后完整测试列表全部 `PASSED`（来源平台：a3；证据等级: tool_observed）。
- 950 Softmax 路径改用 TQue 后，定向回归从 1/49 提升到 18/49；修复其余独立根因且保持队列交接不变后 49/49 通过（来源平台：950；证据等级: tool_observed）。
- 950 的另外两条独立轨迹在相同改法后分别 40/40 与 58/58 通过（来源平台：950；证据等级: tool_observed）。

- 独立复现 R4（DepthwiseConv2d fp32 路径，CANN 9.2.0）：TBuf accBuf 上 Axpy(V) 写完直接逐行 DataCopyPad(MTE3) CopyOut，5x5 核与非对称 padding 用例 MERE=8.06e-01/MARE=2.02e+03，失配位置随重跑漂移（[0]→[4320]/[4849]/[5676]/[8176]/[8177]，其中两个不同期望值的实际值完全相同）；CopyOut 循环前加一行 `PipeBarrier<PIPE_ALL>()` 后转 PASS（MERE=3.12e-07/MARE=7.40e-04）；A/B 翻转两次（去屏障→FAIL、加屏障→PASS）归因唯一；同结构 fp16/bf16 路径经 `yQueue_.EnQue/DeQue` 隐式同步故无此问题，fp32 路径"无 yQueue、accBuf 直接 CopyOut"省掉队列的同时也省掉了同步（来源平台：950；证据等级：tool_observed）。
- 独立复现 R5（同一算子 bf16 路径，CANN 9.2.0）：TBuf 上 `DataCopyPad(MTE)→Cast(V)→GetValue(S)` 三段直链无同步，bias/wScalar 标量读到未完成数据，输出 1e21~1e40 量级垃圾大数（`MERE=nan, MARE=1.56e+40` 等）；四个跨流水线边界各加 `PipeBarrier<PIPE_ALL>` 后 bf16 用例全部转 PASS（`MERE=3.23e-06, MARE=7.81e-03` 等），总通过率 64.3%→85.7%；修复仅插屏障、无算法改动（来源平台：950；证据等级：tool_observed）。
- 独立复现 R6（同一算子 strided 分支，CANN 9.2.0）：整块 `Duplicate(V)` 清零 xLocal 后无同步直接逐行 `DataCopyPad(MTE)` 写有效数据，WAW 竞争致有效数据被重新清零，dilation 用例输出退化为纯 bias（所有失配元素实际值相同且恰为该通道 bias，如 L0_006 五个失配元素全为 0.074768）；Duplicate 后加 `PipeBarrier<PIPE_ALL>` 后 dilation 用例全转 PASS、同轮 stride>1 用例仍失败（两缺陷干净隔离）（来源平台：950；证据等级：tool_observed）。
- 独立复现 R7（RoPE 类算子 cos/sin 跨迭代复用，CANN 9.1.0，定向事件对形态）：kernel 按 (B,S) 循环，每个 (b,s) 先 `LoadCosSin` 用 MTE2 `DataCopyPad` 把 cos/sin 写入 `TBuf<VECCALC>`（跨 N 个 head 复用，query/key 共 2N 次消费），再由 Vector 的 Cast/Mul 读取，两个方向均无同步。指纹：`queryOut` 全对（MERE=0.00e+00，1048576 elems）而 `keyOut` 从第 31 行（index 1920）起错（MERE=2.63e-01）——query 在时序上先于覆盖点而幸免，key 是覆盖前最后一个消费者；错误起始点与 LoadCosSin 第二次覆盖 cos/sin buffer 的节奏吻合。修复仅在两处插入**定向** HardEvent 对（其余不变）：LoadCosSin 尾部 MTE2→V、循环头部（第二轮起）V→MTE2（verbatim）：（来源平台：950；证据等级：tool_observed）

```cpp
    } else {
        // FP32: 直接加载到 FP32 buffer
        DataCopyPad(cosFp32, cosGm_[cosOffset], copyParams, padParams);
        DataCopyPad(sinFp32, sinGm_[cosOffset], copyParams, padParams);
    }

    // MTE2→Vector 同步：确保 cos/sin 的 DataCopyPad(MTE2) 完成后才被 Vector 操作使用
    // cos/sin 使用 TBuf（非 TQue），需手动同步
    event_t evtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evtID);
    WaitFlag<HardEvent::MTE2_V>(evtID);
}
```

```cpp
    for (uint32_t bsIdx = bsStart; bsIdx < bsEnd; bsIdx++) {
        uint32_t b = bsIdx / S_;
        uint32_t s = bsIdx % S_;

        // Vector→MTE2 同步：确保前一轮 (b,s) 的 Vector 操作（RotateHalfRow）完成后，
        // 才允许 LoadCosSin 的 MTE2 覆盖 cos/sin buffer（cos/sin 使用 TBuf，需手动同步）
        if (bsIdx > bsStart) {
            event_t evtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(evtID);
            WaitFlag<HardEvent::V_MTE2>(evtID);
        }

        // 加载 cos/sin（每 (B,S) 一次，复用 N 个 head）
        LoadCosSin(b, s);
```

  修复后 L0_005、L0_013 queryOut+keyOut 全部 PASS（MERE=0.00e+00），keyOut 分裂模式消失。R7 的设计取舍：cos/sin 生命周期跨多行（多读者复用），TQue 的一次性 EnQue/DeQue 语义不匹配，TBuf + 定向 HardEvent 对是贴合语义的选择——本卡原"TQue 交接"路线适用于一次性消费的初始化数据；两形态按"消费者是否跨迭代复用同一份缓冲"分流。该同步模式在后续整体重写的最终版 kernel 中被原样保留，是稳定做法（来源平台：950；证据等级：tool_observed）。"按依赖方向逐方向独立 eventId 显式同步"的一般化形态（MTE2/V/MTE3 多流水线交替读写同一 UB）另见[共享 UB 跨流水线复用须按方向显式同步卡](shared_ub_cross_pipeline_per_direction_sync.md)。

## 失效边界

本卡只证明 A3 与 950 上已列出的初始化路径。不同 TPosition、显式 SetFlag/WaitFlag、其他架构或编译器的事件语义须独立验证。若 DMA 与首个 V 读之间已有可靠队列依赖，不能重复套用本修复。

- 方向扩展（R4-R6）：除本卡原 MTE2→V 外，TBuf 上的 V→MTE3（RAW，失配位置随重跑漂移指纹）、MTE→V→S 三段链（标量读到垃圾大数指纹）、V 与 MTE 对同一 buffer 双写（WAW，输出退化为纯 bias 指纹）同样缺同步；修复可在跨管线生产-消费边界加 `PipeBarrier<PIPE_ALL>`，不必强行改队列（小 weight/bias 块建队列开销与结构改动大，屏障是最小侵入修复；屏障粒度精化与性能代价未测量）。
- R7 增补：显式定向 `SetFlag/WaitFlag`（MTE2_V / V_MTE2）已在一条独立轨迹闭合（DataCopyPad(MTE2) ↔ 基础向量 API 消费者的双向、跨迭代 WAR），本卡原"显式 SetFlag/WaitFlag 须独立验证"的边界对该形态解除；但 R7 验证的是基础向量 API 消费者——`asc_vf_call` 异步 VF 流水线的跨迭代 WAR 能否被 V_MTE2 事件对覆盖未验证（该场景三条同步路线失败的记录见[循环复用 UB buffer 跨迭代 MTE2/V 竞态卡](../runtime/cross_iteration_ub_reuse_mte2_vf_race.md)）。HardEvent 方向名与 MTE2/MTE3 的选择取决于搬运指令类型，换搬运 API 时方向要对应调整。
- 竞态时序相关：小数据块（如 KH*KW=9 元素）竞态窗口小，可能间歇性通过——"一次重跑通过"不能证伪该隐患；kernel 内 `yGm_.GetValue` 回读探针本身也是异步读，不加屏障会误导定位。
- TQue 的 EnQue/DeQue 只同步 EnQue 之后到消费侧的传递，不管理 EnQue 之前 V 与 MTE 之间的写序（R6：先 Duplicate 再 DataCopyPad 的 WAW 不被队列保护）。

## 坏实践

- 把 `PipeBarrier<PIPE_V>` 放在 Cast 之后，误以为它能补上 Cast 之前缺失的 MTE2→V 依赖；
- 用 fp32 分支正常作为裸 TBuf 初始化安全的证据；
- 只检查 Cast 支持矩阵，不检查生产者与消费者流水同步。

- R4-R6 增补：把 V 流水线写的 TBuf 直接交给 MTE3 流水线读而不加任何同步——TBuf 不提供队列语义，编译器不报错、小 shape 可能碰巧通过；在 TBuf 上把 `DataCopyPad → Cast → GetValue` 串成直链；"先整块 Duplicate 清零、再选择性 DataCopyPad"默认两条流水线按代码顺序执行；见到输出错乱/垃圾大数/退化为 bias 先查公式/索引/Cast RoundMode——先用逐 tap printf 证明计算正确再查同步层。
- R7 增补：用 `PipeBarrier<PIPE_ALL>` 或 SyncAll 代替定向 HardEvent 同步——正确但过重（排空所有流水线，破坏 MTE/Vector 并行），定向事件对开销更小；以为"同一函数里顺序写的代码就会顺序执行"——MTE 与 Vector 是不同硬件流水线，指令发射顺序不等于完成顺序。

## 误诊

若修复后非有限值位置仍不一致，先把 gamma 搬运与规约标量路径分别做独立探针，不能继续把错误归到同一交接点。

- R4 增补：曾怀疑 aclnn L2 层 ViewCopy/AllocTensor 搬错数据（plog 里 opResult device_ptr=0）与行 stride 不足致 StoreAlign 越界、多核写同一 yGm 偏移——分别经 plog 语义核对、stride 统一对齐后失败信号不变（且 fp16 更小 stride 通过的反例）、tiling 切分核对排除；屏障 A/B 翻转直接翻转结果反证。
- R5 增补：曾沿"Cast<float,bfloat16_t> 在 arch35 的支持性/RoundMode 映射"方向读 cast 头文件——确认 bf16↔float 及 CAST_NONE 语义无问题后排除，转向同步层；见到 bf16 输出大数先 printf 打出 GetValue 的原始标量值再下结论。
- R6 增补：曾逐段核对 SW=1 分支的 colStart/colEnd/pad 边界算式（疑索引错误）与 xQueue AllocTensor 复用脏数据——手工推演各 (kh,kw) 列范围均正确、非 strided 分支同队列结构无问题，排除后转向 buffer 写序。
