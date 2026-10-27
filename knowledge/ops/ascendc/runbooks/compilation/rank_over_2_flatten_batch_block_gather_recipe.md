---
type: "Runbook"
title: rank>2 展平批块搬运 gather 配方
description: rank>2 输入映射到按行处理 kernel 的完整配方：host 展平为 (N_total, C)（N_total=非 C 维乘积、cStride=C 后维乘积）下发 TilingData，kernel 按"批块 DataCopyPad（blockCount=C、blockLen=cStride）+ GetValue(c*batchRowAlign_+i) 收集 + 批内缓存"，rank 3~8 全量验证通过。
tags: [task.implementation, topic.data_movement, topic.alignment]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# rank>2 展平批块搬运 gather 配方

## 现象（任务场景）

在 Ascend950PR（arch35 / DAV_3510，CANN 9.1.0）上实现 cross entropy loss 类算子，需求要求 input 支持 rank 2~8、C 固定在第 1 维（dim 1）。kernel 是"逐行（逐样本）处理"形态：每行取 C 个元素做 ReduceMax/Exp/ReduceSum。迭代二只支持 2D（把高维输入当 2D 硬算），8 个 rank 3~5 硬标签用例精度全错（none 模式 MERE 0.22~0.92，误差随维度升高放大——跨行读取偏移错误）。迭代三需要一套把任意 rank 输入正确映射进按行 kernel 的实现。

## 适用场景

- 工程形态：aclnn 自定义算子（AscendC kernel + host TilingFunc），kernel 按行/按样本处理，输入第 1 维（dim 1）为"类别维 C"、其余维为样本维。
- API 路径：TilingFunc 的 `inputShape.GetDimNum()/GetDim(i)`、`context->GetTilingData<T>()`；kernel 的 `DataCopyPad`（DataCopyExtParams：blockCount/blockLen/srcStride/dstStride）、`LocalTensor::GetValue/SetValue`、`TBuf`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——展平/批块/gather 是布局映射配方，与算子数值逻辑无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：C 不在第 1 维的布局（需先 transpose 或改下标推导，未验证）；cStride 极大导致 `cLength_ * batchRowAlign_ * sizeof(T)` 超 UB 预算时须分块（本配方 UB 占用 = C × 对齐后 cStride × dtype 大小，轨迹内最大 ~31KB，远小于 192KB UB；更大 cStride 未验证）；gather 用标量 GetValue，C 很大时性能受限（轨迹内 C≤16384 为 2D 路径不受影响，ND 路径 C≤256 实测正确）。

## 根因（原始缺陷）→ 配方目标

迭代二把高维输入当 (N_total, C) 连续布局直读：`inputGM[row * cLength_]`。但 rank>2 时第 dim1 维（C）的元素在内存中被其后维（d1×d2×…）隔开——样本 row 的 C 个元素地址为 `n*C*cStride + c*cStride + i`，不是连续的。配方要同时解决"host 怎么展平"与"kernel 怎么按新布局取数"。

## 解决配方

### 1. Host：展平参数计算与下发

TilingData 增加两个字段：

```cpp
// op_kernel/arch35/bench_no_harness_cross_entropy_loss_tiling_data.h
struct BenchNoHarnessCrossEntropyLossTilingData {
    uint64_t nTotal = 0;           // 展平后总行数 N_total = N * d1 * d2 * ...
    uint32_t cLength = 0;          // C 维度（类别数，有效元素个数）
    ...
    uint64_t cStride = 1;          // C 维步长（C 维后所有维度乘积，2D 时为 1）
    uint32_t syncCoreNum = 1;      // SyncAll 参与核数
};
```

TilingFunc 计算（`op_host/arch35/*_tiling.cpp`）：

```cpp
    // 4. Flatten high-dim shape to (N_total, C)
    // C is dim 1, N_total = product of all other dims
    // cStride = product of all dims after C (dims[2..rank-1])
    // For 2D (N, C): cStride = 1 (C elements are contiguous)
    // For ND (N, C, d1, ..., dk): cStride = d1 * d2 * ... * dk (C elements are strided)
    uint64_t C = 0;
    uint64_t nTotal = 0;
    uint64_t cStride = 1;
    size_t dimNum = inputShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "input rank must be >= 2, got %zu", dimNum),
                return ge::GRAPH_FAILED);
    C = inputShape.GetDim(1);
    nTotal = 1;
    for (size_t i = 0; i < dimNum; i++) {
        if (i != 1) {
            nTotal *= inputShape.GetDim(i);
        }
    }
    // cStride = product of all dims after C (index 2 and beyond)
    for (size_t i = 2; i < dimNum; i++) {
        cStride *= inputShape.GetDim(i);
    }
```

### 2. Kernel：样本号到 (batch, spatial) 的映射与批块缓存

样本（行）`row` 映射：`n = row / cStride`（批号），`i = row % cStride`（批内空间下标）。一个批的输入在 GM 上是 `C × cStride` 个连续元素（起点 `n * C * cStride`），一次 `DataCopyPad` 整批搬入 UB，行进到新批才重搬（批内缓存 `curBatchIdx_`）：

```cpp
// op_kernel/arch35/bench_no_harness_cross_entropy_loss.h（LoadInputRowToWorkBuf，ND 分支）
        // ND path: read batch block and gather C elements
        int64_t n = static_cast<int64_t>(row) / static_cast<int64_t>(cStride_);
        uint32_t i = static_cast<uint32_t>(static_cast<int64_t>(row) % static_cast<int64_t>(cStride_));

        // Cache: only re-read batch block when batch changes
        if (n != curBatchIdx_) {
            curBatchIdx_ = n;
            auto batchBlock = batchBlockBuf.Get<T>();
            DataCopyExtParams batchCopyParams;
            batchCopyParams.blockCount = static_cast<uint16_t>(cLength_);       // C 个块
            batchCopyParams.blockLen = static_cast<uint32_t>(cStride_ * sizeof(T)); // 每块 cStride 元素
            batchCopyParams.srcStride = 0;  // contiguous in GM
            batchCopyParams.dstStride = 0;  // UB auto-pads to 32B alignment
            uint64_t gmOffset = static_cast<uint64_t>(n) * cLength_ * cStride_;
            DataCopyPad(batchBlock, inputGM[gmOffset], batchCopyParams, {false, 0, 0, 0});

            event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        }

        // Gather C elements from batch block into contiguous buffer
        auto batchBlock = batchBlockBuf.Get<T>();
        for (uint32_t c = 0; c < cLength_; c++) {
            T val = batchBlock.GetValue(c * batchRowAlign_ + i);
            xLocal.SetValue(c, val);
        }
        // V pipeline operations are naturally ordered（TBuf 方案；标量写与 TQue 事件模型的错配见库内「标量 S pipe 误用 TQue 导致输出全常数」卡）
```

### 3. 对齐参数与 UB 分配

```cpp
// Init：按 dtype 计算 UB 内每 C-块的对齐行宽
    uint32_t alignElements = 32 / sizeof(T);  // FP32: 8, FP16/BF16: 16
    batchRowAlign_ = static_cast<uint32_t>(
        (cStride_ + alignElements - 1) / alignElements * alignElements);

    // ND batch block buffer (only for cStride > 1)
    if (cStride_ > 1) {
        uint32_t batchBlockUbSize = cLength_ * batchRowAlign_ * sizeof(T);
        pipe.InitBuffer(batchBlockBuf, batchBlockUbSize);
        if constexpr (IS_HARD_LABEL == 0) {
            pipe.InitBuffer(targetBatchBlockBuf, batchBlockUbSize);  // 软标签 target 同构
        }
    }
```

关键语义（DataCopyPad stride 单位）：GM 侧 stride 单位是字节、UB 侧单位是 32 字节 dataBlock；`dstStride=0` 时 UB 中每个 block 仍按 32 字节对齐排放。因此 gather 下标用 `c * batchRowAlign_ + i`（对齐行宽作 stride），而不是 `c * cStride_ + i`——这是本配方最容易写错的一处。

### 4. 2D 快路径保留

`cStride_ == 1` 时走原 2D 路径（单块 `blockLen = cLength_ * sizeof(T)` 直拷 + MTE2→V 同步），零额外开销。

## 设计决策

- 为什么"整批搬 UB 再标量 gather"而不是逐元素从 GM 取：逐元素 GM 访问无批量搬运通路；批块 DataCopyPad 一次 MTE2 指令搬 C×cStride 连续数据，批内所有行共享（每批只搬一次）。
- 为什么 gather 用 `GetValue/SetValue` 标量对而不是向量搬运：样本在批内 stride 不规则（跨 C 块取同一下标 i），向量 API 无此 gather 形态；C 较小（ND 路径 C≤256）时标量代价可接受。
- 曾怀疑 `dstStride=0` 不做 32B 对齐补齐、想改用 `batchRowAlign_` 作显式 dstStride：查证 API 语义（UB stride 单位=32B dataBlock，dstStride=0 仍按 32B 对齐排放块）后确认现状正确，未改——避免把"对齐补齐"算两遍。
- 备选方案（未采用）：host 侧先 `Contiguous`/transpose 成 2D（op_api 层已有非连续 tensor 的 Contiguous 调用），会把拷贝成本转嫁给框架且引入额外 GM 往返；kernel 内 gather 免整表重排。
- 当前方案路线档位：recommended_fastpath。

## 性能影响

- 批内缓存效果：同一批（cStride 个样本）只做一次 C×cStride 的 DataCopyPad；无缓存时每样本都要整批搬运，放大 cStride 倍。
- 未做绝对性能测量；标量 gather 的耗时随 C 线性，ND 路径实测 C≤256 正确性优先。

## 预防

- 写高维支持前先在 TilingData 里显式下发 `cStride`，不要在 kernel 里按 shape 反推。
- gather 下标 review 点：UB 侧 stride 必须用对齐后行宽（`batchRowAlign_`），GM 侧偏移用元素单位乘 `cStride_`——两侧单位不同（UB=32B 块语义 / GM=字节语义）是天然坑点。
- UB 预算检查：`cLength_ * batchRowAlign_ * sizeof(T)` 必须与 tiling 的 UB 预算联合校验，超限时先分块再进 kernel。

## 验证信号

来源说明：以下为该算子迭代三联调验证的实测（C++ ST，Real 模式 NPU，每用例独立进程）。

- rank 3~8 用例全过：L1_020（3D(64,256,7) FP32 none）MERE=3.55e-08、L1_024（4D(16,10,28,28)）MERE=3.31e-08（12544 elems）、L1_028（5D）3.36e-08、L1_030（6D）6.25e-08、L1_032（7D）4.63e-08、L1_033（8D）4.06e-07；L1_073~L1_081（3D/4D 补充、d1=1024 等）全部 PASS（来源平台：950；证据等级：tool_observed）。
- 迭代二失败对照：同批用例 none 模式 MERE 0.22~0.92（issue 记录，本轨迹工具读取）（来源平台：950；证据等级：tool_observed，为历史记录转述）。
- 软标签高维：L1_027（4D 软标签 mean）MERE=3.36e-07、L1_075/L1_079 PASS（来源平台：950；证据等级：tool_observed）。
- PyTorch 路径同批用例独立复验通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- UB 占用 = C × 对齐后 cStride × dtype 大小；cStride 或 C 增大到超 UB 时须分块（轨迹内最大 ~31KB，未测上界）。
- `blockCount` 为 uint16_t：C > 65535 时批块搬运参数溢出，需分批（轨迹内 C≤256 的 ND 路径未触及）。
- 批内缓存要求行序按批递增（tiling 按连续行切核天然满足）；跨批乱序访问会退化为每行重搬（正确性不受影响）。
- 其他架构未验证（对齐语义为 AscendC 通用，推断可迁移，未验证）。

**未决项**：

- 标量 gather 的向量化替代（如按 32B 对齐段做块内搬移 + 小尾 gather）未探索。
- cStride/C 上界（UB 预算与 uint16 blockCount）未做系统测定。
- 软标签 target 批块未做批内缓存，搬运冗余可优化。
## 坏实践

- 高维输入直接按 2D 连续布局寻址（`row * C`）——迭代二的原始缺陷。
- gather 下标用 `c * cStride_`（未对齐行宽）——cStride 非 32B 对齐时读错元素。
- 每样本重搬整批数据（不做批内缓存）——搬运量放大 cStride 倍。

## 误诊

- 无。
