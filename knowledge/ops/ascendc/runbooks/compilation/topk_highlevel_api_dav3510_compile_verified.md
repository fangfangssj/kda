---
type: "Runbook"
title: "TopK 高阶 API 在 DAV_3510 编译可用（产品表未列）"
description: "TopK/Sort32/MrgSort/GatherMask 产品支持表未列 DAV_3510（设计 HIGH 风险），经 ascend950/arch35 端到端编译验证可用（kernel binary 生成）；含 tiling 侧 TopKTilingFunc/GetTopKMaxMinTmpSize 与 kernel 侧 TopK 模板调用的完整接入配方。仅编译验证，未上板。复现增量：大 inner 分块路径的 tmpLocalSize/topkTiling 必须按 innerBlockPad（块内宽度）重算——按整行 innerPad 计算时 inner≥16384 tmpLocalSize 超 UB ~248KB，TilingFunc 返回 GRAPH_FAILED。"
tags: [task.implementation, topic.compute.ordering]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# TopK 高阶 API 在 DAV_3510 编译可用（产品表未列）

## 现象（任务场景）

为 cann-bench 评测把 torch.topk 语义生成为自定义 ACLNN 算子（`y, idx = topk(x, k, dim, largest)`，7 种 dtype、1-8 维、k≤2048），目标平台 Ascend950PR（NpuArch=DAV_3510，arch35），CANN cann-9.1.0。设计阶段的可行性调查发现：AscendC 原生 TopK 高阶 API 家族（TopK/Sort32/MrgSort/GatherMask）的官方文档产品支持表均未列出 DAV_3510（只明确 A2/A3/Kirin），被列为 HIGH 风险——若不可用需回退到 Sort32+MrgSort+GatherMask 手动组合自研排序。迭代一按"fp16 + small inner ≤4096 + k≤2048 + 任意 dim + largest 双向"的 TilingKey=0 路径做端到端编译验证：手写 AscendC kernel 直接调用原生 TopK API，宿主 tiling 调用 TopKTilingFunc/GetTopKMaxMinTmpSize。构建（`bash build.sh --soc=ascend950 -j8`）最终全绿，kernel 二进制与算子包生成，TopK 相关编译警告 0 条，风险关闭、无需回退。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子仓，手写 AscendC kernel（SIMD/MemBase 路线，非 RegBase），TopK/排序类算子。
- API 路径：`AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, TopKMode>`、`Arange`、`Cast`、`Duplicate`、`DataCopyPad`；tiling 侧 `AscendC::TopKTilingFunc` 与 `AscendC::GetTopKMaxMinTmpSize`（头 `topk/topk_tiling.h`，位于 `asc/include/tiling/topk`）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / arch35 / DAV_3510）+ CANN cann-9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（SortSelect/TopK 类自定义算子；对"DAV_3510 上高阶排序 API 可用性"的结论为平台级）。
- 路线优先级：recommended_fastpath（相对自研 Sort32+MrgSort+GatherMask 组合：复用成熟实现、代码量小、语义与 torch.topk 对齐——inf 极大、nan 排前、tie 稳定排序）。
- 明确不适用/未验证场景：仅编译验证，未上板运行——精度/性能结论不在本卡；inner>4096（需分块+MrgSort 归并）与非 fp16 dtype（需 Cast 路径）为迭代二/三范围，未实现。

## 系统信息

- 工具链：`bash build.sh --soc=ascend950 -j8`（CMake + npu_op_code_gen/npu_op_library，opc/TBE 设备侧编译）。
- CANN：cann-9.1.0；tiling 头依赖目录（`asc/include/tiling/topk`、`asc/include/adv_api/sort`、`ascendc/include/highlevel_api`）需加入 op_host 编译目标（include 配置方法见 [Matmul 高阶 API 头不可见卡](matmul_highlevel_api_include_paths_ini.md)）。
- 目录布局：

```
operators/bench_no_harness_top_k/
├── op_host/
│   ├── bench_no_harness_top_k_def.cpp          # 算子注册（dtype 集合与模板同步）
│   ├── bench_no_harness_top_k_infershape.cpp   # y.shape[dim]=k, idx.shape=y.shape
│   └── arch35/bench_no_harness_top_k_tiling.cpp  # tiling（本卡 host 侧配方）
├── op_kernel/
│   ├── bench_no_harness_top_k_arch35.cpp       # 入口（snake_case 函数名）
│   └── arch35/
│       ├── bench_no_harness_top_k.h            # kernel 类（本卡 device 侧配方）
│       ├── bench_no_harness_top_k_tiling_data.h
│       └── bench_no_harness_top_k_tiling_key.h
└── op_api/ ...                                  # aclnn 两段式
```

## 完整代码

**host 侧：tiling 中 TopK 参数计算（op_host/arch35/<op>_tiling.cpp 的 TilingFunc 机制单元，verbatim）**——单行处理模型（outter=1）、isInitIndex=true、TOPK_NORMAL、取最小 tmp：

```cpp
    // 8. 获取 TopK Tiling 参数（单行处理 outter=1）
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t tmpMaxSize = 0;
    uint32_t tmpMinSize = 0;
    // isInitIndex=true, TOPK_NORMAL, isReuse=false
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, static_cast<int32_t>(innerPad),
                                  static_cast<int32_t>(1), false, true,
                                  AscendC::TopKMode::TOPK_NORMAL, isLargest, dtypeSize,
                                  tmpMaxSize, tmpMinSize);
    uint32_t tmpLocalSize = tmpMinSize; // 使用最小临时空间

    AscendC::tiling::TopkTiling topkTiling;
    bool tilingRet = AscendC::TopKTilingFunc(ascendcPlatform, static_cast<int32_t>(innerPad),
                                             static_cast<int32_t>(1), static_cast<int32_t>(k),
                                             dtypeSize, true, AscendC::TopKMode::TOPK_NORMAL,
                                             isLargest, topkTiling);
    OP_CHECK_IF(!tilingRet,
                OP_LOGE(context, "BenchNoHarnessTopK: TopKTilingFunc failed."),
                return ge::GRAPH_FAILED);

    // 9. UB 容量校验（fp16 路径）
    // srcValue(innerPad*2) + srcIndex(innerPad*4) + dstValue(kPad*2) + dstIndex(kPad*4) +
    // dstIndex64(kPad*8) + tmpLocal(tmpLocalSize) + finish(32)
    uint64_t ubRequired = static_cast<uint64_t>(innerPad) * (dtypeSize + sizeof(int32_t)) +
                          static_cast<uint64_t>(kPad) * (dtypeSize + sizeof(int32_t) + sizeof(int64_t)) +
                          tmpLocalSize + 32;
    OP_CHECK_IF(ubRequired > ubSize,
                OP_LOGE(context,
                        "BenchNoHarnessTopK: UB overflow. required=%lu, ubSize=%lu, innerPad=%ld, kPad=%ld.",
                        ubRequired, ubSize, innerPad, kPad),
                return ge::GRAPH_FAILED);
```

配套的多核切分与对齐计算（同一 TilingFunc 前序步骤，机制链）：

```cpp
    // 5. 计算 innerPad（32 对齐，TopK API 要求 inner 为 32 的倍数）
    int64_t innerPad = CeilAlign(inner, static_cast<int64_t>(32));

    // 6. 计算 kPad（32 字节对齐：half=2B→16 倍数, float=4B→8 倍数, int32 索引=4B→8 倍数）
    int64_t kPad = 0;
    uint32_t dtypeSize = 2;  // fp16 = 2 字节
    if (dtypeSize == sizeof(float)) {  // 4 字节 → 8 倍数
        kPad = CeilAlign(k, static_cast<int64_t>(8));
    } else {  // 2 字节（half）→ 16 倍数
        kPad = CeilAlign(k, static_cast<int64_t>(16));
    }

    // 7. 多核切分（沿 outter 维切分，inner 维不切分保证排序正确性）
    int64_t blockDim = std::min(coreNum, outter);
    OP_CHECK_IF(blockDim <= 0,
                OP_LOGE(context, "BenchNoHarnessTopK: blockDim must be > 0, outter=%ld.", outter),
                return ge::GRAPH_FAILED);
    int64_t outterPerCore = outter / blockDim;
    int64_t outterTail = outter % blockDim;
```

**device 侧：kernel Compute（op_kernel/arch35/<op>.h 的机制单元，verbatim）**——Arange 初始化索引 → TopK 模板调用 → 索引 int32→int64：

```cpp
template <typename T, int INNER_MODE>
__aicore__ inline void BenchNoHarnessTopK<T, INNER_MODE>::Compute()
{
    LocalTensor<T> srcValue = srcValueBuf.Get<T>();
    LocalTensor<int32_t> srcIndex = srcIndexBuf.Get<int32_t>();
    LocalTensor<T> dstValue = dstValueBuf.Get<T>();
    LocalTensor<int32_t> dstIndex = dstIndexBuf.Get<int32_t>();
    LocalTensor<int64_t> dstIndex64 = dstIndex64Buf.Get<int64_t>();
    LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
    LocalTensor<bool> finishLocal = finishBuf.Get<bool>();

    // ① 索引初始化：Arange 生成 0~innerPad-1（int32_t）
    //    inner~innerPad-1 的索引为补齐值（递增），不影响排序
    Arange<int32_t>(srcIndex, static_cast<int32_t>(0), static_cast<int32_t>(1),
                    static_cast<int32_t>(innerPad_));

    // ② TopK 排序（Normal 模式，单行 outter=1，isInitIndex=true 传入索引）
    //    模板参数：T=half, isInitIndex=true, isHasfinish=false, isReuseSrc=false, TOPK_NORMAL
    TopKInfo topKInfo;
    topKInfo.outter = 1;           // 单行处理
    topKInfo.inner = static_cast<int32_t>(innerPad_); // 32 对齐的内轴长度
    topKInfo.n = static_cast<int32_t>(inner_);        // 实际元素个数

    TopK<T, true, false, false, TopKMode::TOPK_NORMAL>(
        dstValue, dstIndex, srcValue, srcIndex, finishLocal, tmpLocal,
        static_cast<int32_t>(k_), topkTiling_, topKInfo, isLargest_);

    // ③ 索引 int32 → int64 转换（只转换前 k 个有效结果）
    Cast<int64_t, int32_t>(dstIndex64, dstIndex, RoundMode::CAST_NONE,
                           static_cast<uint32_t>(k_));
}
```

数据搬入的 padding 机制（CopyIn，配合位模式构造的 half ±inf）：

```cpp
template <typename T, int INNER_MODE>
__aicore__ inline void BenchNoHarnessTopK<T, INNER_MODE>::CopyIn(int64_t rowIdx)
{
    LocalTensor<T> srcValue = srcValueBuf.Get<T>();

    // 先用极值填充整个 srcValue（innerPad 个元素），确保 inner~innerPad-1 的 padding 不被选入 top-k
    const T padVal = static_cast<T>(GetPadValue(isLargest_));
    Duplicate<T>(srcValue, padVal, innerPad_);

    if constexpr (INNER_MODE == BENCH_TOPK_INNER_MODE_SMALL) {
        if (isLastDim_) {
            // dim==last：数据连续，一次搬运 inner 个元素
            // GM 偏移：rowIdx * inner（每行 inner 个连续元素）
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(inner_ * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad<T>(srcValue, gmX[rowIdx * srcRowStride_], copyParams,
                           {false, 0, 0, static_cast<T>(0)});
        } else {
            // dim!=last：每行 inner 个元素间隔 srcDimStride，循环单元素搬运（结构预留，后续迭代优化为 stride 重排）
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            int64_t baseOffset = rowIdx * srcRowStride_; // 逻辑行起始偏移
            for (int64_t t = 0; t < inner_; t++) {
                DataCopyPad<T>(srcValue[t], gmX[baseOffset + t * srcDimStride_], copyParams,
                               {false, 0, 0, static_cast<T>(0)});
            }
        }
    }
    // INNER_MODE == LARGE 分支由后续迭代实现（if constexpr 编译期分发）
}
```

UB buffer 规划（Init，fp16 路径）：

```cpp
    pipe.InitBuffer(srcValueBuf, innerPad_ * sizeof(T));
    pipe.InitBuffer(srcIndexBuf, innerPad_ * sizeof(int32_t));
    pipe.InitBuffer(dstValueBuf, kPad_ * sizeof(T));
    pipe.InitBuffer(dstIndexBuf, kPad_ * sizeof(int32_t));
    pipe.InitBuffer(dstIndex64Buf, kPad_ * sizeof(int64_t));
    pipe.InitBuffer(tmpBuf, tmpLocalSize_);
    pipe.InitBuffer(finishBuf, 32); // finishLocal 占位，isHasfinish=false 不实际使用
```

## 设计决策

- **技术路线**：通用 SIMD/MemBase（非 RegBase）。理由：核心依赖的 TopK/Sort32/MrgSort 属 SIMD vector 指令封装；DAV_3510 向下兼容 A2/A3 vector API（设计期假设，本次编译验证）。
- **排序算法选原生 TopK API 而非自研**（候选对比）：原生 API（Sort32+MrgSort+GatherMask 封装，成熟、语义匹配：inf 极大、nan 排前、tie 稳定排序，与 torch.topk 一致）✅；Sort32+MrgSort+GatherMask 手动组合（灵活但复杂）⬜ 备选（API 不可用时回退，本次未触发）；RadixSort（整数适合但浮点需位重解释、无原生 API）❌；自研 HeapSelect（大 inner 小 k 高效但自研复杂）❌。
- **单行处理模型**：每核每次处理 1 行（inner 个元素），循环 outterPerCore 次；TopK 以 outter=1 调用。inner 维不切分（切分则各核只得局部 TopK，归并代价高且正确性难保证），outter 维多核切分，`blockDim = min(coreNum, outter)` 避免空核。
- **largest 双向**：直接把 isLargest 传给 TopK API（内部取负实现方向切换），不做外层取负。
- **对齐**：innerPad 向上 32 对齐（TopK API 对 inner 的倍数要求）+ padding 用 ∓inf（随 isLargest 翻转）保证补齐元素不入 top-k；kPad 按元素大小对齐到 32 字节。
- **tmpLocal 取 GetTopKMaxMinTmpSize 的 min 值**（small inner 路径内存预算从紧）。
- **索引链路**：Arange 生成 int32 索引 → TopK 输出 int32 索引 → Cast<int64_t,int32_t>（CAST_NONE）满足 idx 固定 int64 的接口要求。

## 验证信号

段首来源说明：风险关闭与配方验证来自该算子迭代一的端到端构建（未上板）。

- `bash build.sh --soc=ascend950 -j8` 最终成功：`[INFO] Build completed successfully!`；生成 kernel 二进制 `build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_top_k/BenchNoHarnessTopK_9921bbcc9384f9f94ab1a08309c1028e.o`（含 .json）与算子包 `custom_opp_ubuntu_x86_64.run`（含 op_tiling/op_proto/op_api 全套 .so）（来源平台：950 编译链；证据等级：tool_observed）。
- 构建日志中 TopK/Sort/MrgSort/GatherMask 相关编译警告/错误 0 条（`grep -i "topk|sort|mrgsort|gathermask" build.log | grep -i "warning|error"` 无输出；全量 warning 共 14 条均为无关 deprecated 头警告）（来源平台：950 编译链；证据等级：tool_observed）。
- 设计期 HIGH 风险"原生 TopK API 在 DAV_3510 支持未明确"经此编译验证关闭，未触发回退路径（来源平台：950 编译链；证据等级：tool_observed）。

## 失效边界

- **仅编译验证**：算子未上板运行，TopK API 在 DAV_3510 上的运行时行为（精度/性能/异常）未验证；上板结论待 Real 模式 ST（未验证，不推断）。
- 覆盖范围限于 TOPK_NORMAL + fp16 + inner≤4096 + k≤2048；inner>4096 的分块归并、非 fp16 的 Cast 路径未实现（迭代二/三）。
- dim 非末维路径当前为循环单元素搬运（结构预留，性能未优化）。
- UB 预算按 fp16 排序 dtype 估算（最大 int64 Cast 路径约 144KB ≤ arch35 UB 248KB，设计期估算，未上板核验）。

**未决项**：

- 上板运行验证（Real 模式 ST + 精度比对）是第一优先接续点。
- 大 inner 分块 + MrgSort 归并、其余 6 种 dtype 的 Cast 路径待迭代二/三实现后按同法验证。
## 坏实践

- 因产品支持表未列新架构就绕开高阶 API 直接自研底层排序（先编译验证再决定）。
- inner 不对齐 32 直接调 TopK（违反 API 对齐约束）。
- padding 用有限大数而非 ∓inf（输入含极值时 padding 会被误选入 top-k）。

## 误诊

无（本卡为正向验证的接入配方；构建期间的失败均已归因到独立根因卡）。

## 后续增量（2026-09-17，并行策展批：大 inner 分块路径的 tmp 尺寸口径——按块重算而非整行）

迭代二大 inner 分块归并路径（inner > 4096 时按 `LARGE_INNER_BLOCK_SIZE=4096` 分块，逐块 TopK 后归并）的 Tiling UT 中，`fp16` + `inner=16384` 组合 TilingFunc 直接失败：探针 `[PROBE] fp16_large_16k FAILED (GRAPH_FAILED)`；而 inner=8192/12288 能通过但 `tmpLocalSize` 异常偏大（131072/196608，随 innerPad 线性增长），探针同时显示 `innerBlockSize=4096, innerBlockPad=4096`——kernel 实际按 4096 一块处理，tmp 却按整行算。

**根因**：大 inner 分块路径的 `GetTopKMaxMinTmpSize`/`TopKTilingFunc` 在 `isLargeInner` 分支之前统一按 `innerPad`（整行 32 对齐值）计算，而 kernel 每块只用 `innerBlockPad`(4096) 调 TopK——inner=16384 时 tmpLocalSize=262144 字节一项就超 UB 可用上限（约 248KB），tiling 的 UB 检查失败返回 GRAPH_FAILED。**修复**：在 `isLargeInner` 分支内基于 `innerBlockPad` 重算 tmp 与 topkTiling（`outter` 参数按单块单行取 1，与 kernel 逐块单行调用一致）：

```cpp
    if (isLargeInner) {
        innerBlockSize = LARGE_INNER_BLOCK_SIZE;
        innerBlockPad = CeilAlign(innerBlockSize, static_cast<int64_t>(32));
        numBlocks = CeilDiv(inner, innerBlockSize);
        // 分块 TopK 基于 innerBlockPad(4096) 重新计算 tmpLocalSize 和 topkTiling
        // （原先基于 innerPad 计算，inner>12288 时 tmpLocalSize 高估致 UB 溢出误判）
        uint32_t blockTmpMax = 0, blockTmpMin = 0;
        AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, static_cast<int32_t>(innerBlockPad),
                                      static_cast<int32_t>(1), false, true,
                                      AscendC::TopKMode::TOPK_NORMAL, isLargest, sortDtypeSize,
                                      blockTmpMax, blockTmpMin);
        tmpLocalSize = blockTmpMin;
        bool blockTilingRet = AscendC::TopKTilingFunc(ascendcPlatform,
                                                       static_cast<int32_t>(innerBlockPad),
                                                       static_cast<int32_t>(1), static_cast<int32_t>(k),
                                                       sortDtypeSize, true, AscendC::TopKMode::TOPK_NORMAL,
                                                       isLargest, topkTiling);
        OP_CHECK_IF(!blockTilingRet,
                    OP_LOGE(context, "block TopKTilingFunc failed."),
                    return ge::GRAPH_FAILED);
        mergeInner = numBlocks * kPad;
```

判别方法：用 tiling 探针对比 `tmpLocalSize` 与 `innerBlockSize`——若 `innerBlockSize` 固定而 `tmpLocalSize` 随 inner 增长（131072 → 196608 → 262144），即为同一错误模式；失败信号为 inner 超过某阈值后 TilingFunc 返回 GRAPH_FAILED、小 inner 全部正常。阈值随算子 UB 账本平移，判别以"tmpLocalSize 是否按块计算"为准，不要硬编码 inner 阈值。通用原则：任何"host 统一计算 tmp/tiling、kernel 分块消费"的骨架，tmp 尺寸计算口径必须与 kernel 单次调用的规模一致（块大小 × 单块元素，而非整行）。UB 检查失败返回 GRAPH_FAILED 时，先 dump tiling 字段核对 tmpLocalSize 量级，再怀疑 UB 真不够。

复现增量验证信号：

- 修复前探针：`fp16_large_16k FAILED (GRAPH_FAILED)`；`fp16_large_8k tmpLocalSize=131072`、`fp16_large_12k tmpLocalSize=196608`（随 innerPad 线性增长）（来源平台：950；证据等级：tool_observed）。
- 修复后探针：`fp16_large_16k tilingKey=257 ... innerBlockSize=4096 numBlocks=4 tmpLocalSize=65536`，8k/12k/16k 全组合通过、tmpLocalSize 恒为 65536（来源平台：950；证据等级：tool_observed）。
- 回归：迭代二 UT 56/56（op_host 42 + op_api 14）通过，`bash build.sh --soc=ascend950 -j8` 编译通过（来源平台：950；证据等级：tool_observed）。

复现增量边界：分块大小非 4096 时按实际 `innerBlockPad` 重算；不分块（整行一次 TopK）路径 tmpLocalSize 本来就该按 innerPad 算，不适用本增量。相关：TopK 保留 UB 预算漏记是另一独立根因（见 [507035 TopK 保留 UB 预算卡](507035_topk_reserved_ub_budget.md)）。