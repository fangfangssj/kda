---
type: "Runbook"
title: Mish 数值稳定链：Log/Tanh 禁重叠与 tmpBuf 轮转
description: y=x·tanh(softplus(x)) 的数值稳定展开 max(x,0)+log1p(exp(-|x|)) 用九步向量 API 实现：tmpBuf1/2/3 交替复用避免 Log/Tanh 地址重叠，sharedTmp 按 ubFormer×每元素字节保守估算，BF16 路径 Cast 上/下精度，非对齐尾部 DataCopyPad。
tags: [task.implementation, topic.activation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# Mish 数值稳定链：Log/Tanh 禁重叠与 tmpBuf 轮转

## 现象（任务）

在 Ascend950PR 上实现 Mish 激活 `y = x * tanh(softplus(x))` 的 AscendC elementwise kernel，要求 FP16/FP32/BF16 三 dtype（三 TilingKey）在 NPU 上精度达标（FP16 MERE<2^-10、FP32<2^-13、BF16<2^-7），支持任意 shape（含非对齐尾部、空 tensor、0 维）与多核切分。核心难点：① 直接展开 `softplus(x)=log(1+exp(x))` 在大正 x 处 exp 溢出；② `Log`/`Tanh` 不支持源/目的地址重叠，需独立 dst buffer；③ BF16 无直接计算路径，需升精度；④ 非对齐尾部不能用整块 DataCopy。

实现后三 dtype 全量上板验证通过（FP16 20/20、FP32 15/15、BF16 17/17，边界与特殊值全过）。

## 适用场景

- 工程形态：AscendC elementwise 数学/激活算子（aclnn 自定义算子，host tiling + kernel 模板 + 多 TilingKey）。
- API 路径：`Abs/Muls/Exp/Adds/Log/Maxs/Add/Tanh/Mul` 标准向量 API；`DataCopyPad`；`Cast`（`RoundMode::CAST_NONE`/`CAST_RINT`）；`TPipe/TQue/TBuf`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——配方结构适用于 softplus/tanh/GELU 类含 log/exp 的激活族；常数（UB 大小、对齐粒度）按平台重取。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：大负 x 全值域与 log1p golden 一致（受 Adds+Log 舍入边界限制，见失效边界）；其他架构的 UB_SIZE/对齐常量需重查（未验证，机制推断可迁移）；RegBase/VF body 内（Tanh 非 VF-safe）。

## 根因/配方要点

1. **数值稳定展开**：`softplus(x) = max(x,0) + log1p(exp(-|x|))`，exp 的自变量恒为非正，消除大正 x 溢出；`log1p` 项用 `Adds(exp(-|x|), 1.0) → Log` 组合（AscendC 无原生 log1p，其大负 x 舍入边界见失效边界）。数值动机：直接形式 `ln(1+e^x)` 中 e^x 随 x 指数增长——fp16 上限 65504，x≥12 时 e^x≈1.6e5 已溢出为 inf；且 `1+e^x` 在 e^x 大时发生"大数吃小数"（fp16 尾数 10 位，相对步长 2^-11，加 1 可能被完全吞掉），ln 结果产生系统性偏差。稳定形式把指数项压到 (0,1] 区间（e^{-|x|} ≤ 1 恒不溢出），log1p 对小 ε 保住有效位。
2. **buffer 交替复用**：九步链用 tmp1/tmp2/tmp3 三个 VECCALC buffer 轮转，保证 Log/Tanh 这类"dst 不得与 src 重叠"的 API 始终有独立 dst。官方文档明确 Log/Tanh **不支持源操作数与目的操作数地址重叠**，通用地址重叠约束还规定"API 中没有描述地址重叠约束的，视为不支持地址重叠"——其余 API 也按不支持处理。
3. **sharedTmp 保守估算**：Tanh 需要 sharedTmpBuffer；host tiling 按 `tmpBufferSize = ubFormer × tmpBytesPerElem`（FP32=4B/元素）分配，不用 GetTanhMaxMinTmpSize 精确值（保守但免查表）。
4. **BF16 升精度路径**：`Cast(BF16→FP32, CAST_NONE)`（无损）→ FP32 九步链 → `Cast(FP32→BF16, CAST_RINT)`（四舍六入五成双）。
5. **非对齐尾部**：CopyIn/CopyOut 用 `DataCopyPad`（`DataCopyPadParams{false,0,0,0}`），blockLen 按字节数给，支持任意元素数尾块。
6. **空 tensor/0 维**：tiling totalLength=0 时 Init/Process 直接早退，不分配 buffer。

## 完整代码

kernel 主体（`op_kernel/arch35/bench_no_harness_mish.h`，verbatim；含 9 步链、CopyIn/Compute/CopyOut 与 Process 的完整 UB 块循环）：

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BenchNoHarnessMishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void MishCompute(const LocalTensor<ComputeT>& xBuf, LocalTensor<ComputeT>& dstBuf,
                                       LocalTensor<ComputeT>& tmp1, LocalTensor<ComputeT>& tmp2,
                                       LocalTensor<ComputeT>& tmp3, LocalTensor<uint8_t>& sharedTmp,
                                       int32_t count);
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);

private:
    TPipe pipe;
    // 输入/输出队列（double buffer）
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // 临时 buffer（Log/Tanh 不支持地址重叠，需独立 dst buffer）
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> tmpBuf3;
    // BF16 专用：Cast 后的 FP32 输入缓冲
    TBuf<TPosition::VECCALC> bufXF32;
    // Tanh 共享临时空间
    TBuf<TPosition::VECCALC> sharedTmpBuf;

    GlobalTensor<D_T_X> inputGMX;
    GlobalTensor<D_T_X> outputGMY;

    const BenchNoHarnessMishTilingData* tilingData_ = nullptr;
    int64_t ubFormer_ = 0;
    int64_t inputGmOffset_ = 0;
```

```cpp
template <typename D_T_X>
__aicore__ inline void BenchNoHarnessMish<D_T_X>::Init(GM_ADDR x, GM_ADDR y,
                                                        const BenchNoHarnessMishTilingData* tilingData)
{
    tilingData_ = tilingData;
    ubFormer_ = static_cast<int64_t>(tilingData->ubFormer);

    // 空_tensor 直接返回，不分配 buffer
    if (tilingData->totalLength == 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    // 每个核的 GM 起始偏移
    inputGmOffset_ = static_cast<int64_t>(tilingData->blockFormer) * blockIdx;

    // 计算当前核处理的元素数
    int64_t myLength;
    if (static_cast<uint64_t>(blockIdx) < tilingData->blockNum - 1) {
        myLength = static_cast<int64_t>(tilingData->blockFormer);
    } else {
        myLength = static_cast<int64_t>(tilingData->totalLength) -
                   static_cast<int64_t>(tilingData->blockFormer) * (tilingData->blockNum - 1);
    }

    inputGMX.SetGlobalBuffer((__gm__ D_T_X*)x + inputGmOffset_, myLength);
    outputGMY.SetGlobalBuffer((__gm__ D_T_X*)y + inputGmOffset_, myLength);

    // Buffer 初始化（按是否 Cast 分支）
    if constexpr (IS_CAST) {
        // BF16 路径：inQueue/outQueue 为 BF16，计算 buffer 为 FP32（bufXF32 + tmp1/2/3 按 4 字节初始化）
        pipe.InitBuffer(inQueue, BUFFER_NUM, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(bufXF32, ubFormer_ * 4);
        pipe.InitBuffer(tmpBuf1, ubFormer_ * 4);
        pipe.InitBuffer(tmpBuf2, ubFormer_ * 4);
        pipe.InitBuffer(tmpBuf3, ubFormer_ * 4);
        pipe.InitBuffer(sharedTmpBuf, tilingData->tmpBufferSize);
    } else {
        // FP16/FP32 路径：所有 buffer 同类型
        pipe.InitBuffer(inQueue, BUFFER_NUM, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(tmpBuf1, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(tmpBuf2, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(tmpBuf3, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(sharedTmpBuf, tilingData->tmpBufferSize);
    }
}
```

```cpp
template <typename D_T_X>
__aicore__ inline void BenchNoHarnessMish<D_T_X>::MishCompute(const LocalTensor<ComputeT>& xBuf,
                                                              LocalTensor<ComputeT>& dstBuf,
                                                              LocalTensor<ComputeT>& tmp1,
                                                              LocalTensor<ComputeT>& tmp2,
                                                              LocalTensor<ComputeT>& tmp3,
                                                              LocalTensor<uint8_t>& sharedTmp,
                                                              int32_t count)
{
    // 数值稳定 Mish 计算：y = x * tanh(max(x,0) + log1p(exp(-|x|)))
    // tmpBuf 交替复用，Log/Tanh 使用独立 dst buffer 避免地址重叠
    uint32_t countU32 = static_cast<uint32_t>(count);
    Abs<ComputeT>(tmp1, xBuf, count);                              // Step1: tmp1 = |x|
    Muls<ComputeT>(tmp2, tmp1, static_cast<ComputeT>(-1.0f), count); // Step2: tmp2 = -|x|
    Exp<ComputeT>(tmp1, tmp2, countU32);                           // Step3: tmp1 = exp(-|x|)
    Adds<ComputeT>(tmp2, tmp1, static_cast<ComputeT>(1.0f), count);  // Step4: tmp2 = 1 + exp(-|x|)
    Log<ComputeT>(tmp3, tmp2, countU32);                           // Step5: tmp3 = log(1+exp(-|x|))
    Maxs<ComputeT>(tmp2, xBuf, static_cast<ComputeT>(0.0f), count);  // Step6: tmp2 = max(x, 0)
    Add<ComputeT>(tmp1, tmp2, tmp3, count);                        // Step7: tmp1 = softplus
    Tanh<ComputeT>(tmp2, tmp1, sharedTmp, countU32);               // Step8: tmp2 = tanh(softplus)
    Mul<ComputeT>(dstBuf, xBuf, tmp2, count);                      // Step9: dst = x * tanh(softplus)
}
```

每步的 src/dst 重叠关系（全部合规）：

| 步 | dst | src | 重叠? |
|---|---|---|---|
| 1 Abs | tmp1 | xBuf | 否 |
| 2 Muls | tmp2 | tmp1 | 否 |
| 3 Exp | tmp1 | tmp2 | 否 |
| 4 Adds | tmp2 | tmp1 | 否 |
| 5 Log | tmp3 | tmp2 | 否（Log 禁重叠，dst 独立） |
| 6 Maxs | tmp2 | xBuf | 否 |
| 7 Add | tmp1 | tmp2,tmp3 | 否 |
| 8 Tanh | tmp2 | tmp1 | 否（Tanh 禁重叠，dst 独立） |
| 9 Mul | dstBuf | xBuf,tmp2 | 否 |

count 类型注意：Abs/Muls/Adds/Maxs/Add/Mul 的"前 n 个"接口 count 是 `const int32_t&`，而 Exp/Log/Tanh 的 calCount 是 `uint32_t`——代码里用 `count` 与 `countU32` 两个变量分别喂（本机头文件核对的签名）。

```cpp
template <typename D_T_X>
__aicore__ inline void BenchNoHarnessMish<D_T_X>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<D_T_X> xLocal = inQueue.template AllocTensor<D_T_X>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(D_T_X));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGMX[offset], copyParams, DataCopyPadParams{false, 0, 0, 0});
    inQueue.EnQue(xLocal);
}

template <typename D_T_X>
__aicore__ inline void BenchNoHarnessMish<D_T_X>::Compute(int64_t currentNum)
{
    int32_t count = static_cast<int32_t>(currentNum);
    LocalTensor<D_T_X> xLocal = inQueue.template DeQue<D_T_X>();
    LocalTensor<D_T_X> yLocal = outQueue.template AllocTensor<D_T_X>();
    LocalTensor<uint8_t> sharedTmp = sharedTmpBuf.Get<uint8_t>();

    if constexpr (IS_CAST) {
        // BF16 路径：Cast → FP32 计算 → Cast 回 BF16
        LocalTensor<float> xF32 = bufXF32.Get<float>();
        LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf2.Get<float>();
        LocalTensor<float> tmp3 = tmpBuf3.Get<float>();
        // BF16 → FP32（CAST_NONE 无精度损失）
        Cast<float, bfloat16_t>(xF32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        // FP32 上执行 9 步计算链，结果写入 tmp1（Step9 dst=tmp1，与 src0=xF32/src1=tmp2 不重叠）
        MishCompute(xF32, tmp1, tmp1, tmp2, tmp3, sharedTmp, count);
        // FP32 → BF16（CAST_RINT 四舍六入五成双）
        Cast<bfloat16_t, float>(yLocal, tmp1, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
    } else {
        // FP16/FP32 路径：直接计算，结果写入 yLocal
        LocalTensor<ComputeT> tmp1 = tmpBuf1.template Get<ComputeT>();
        LocalTensor<ComputeT> tmp2 = tmpBuf2.template Get<ComputeT>();
        LocalTensor<ComputeT> tmp3 = tmpBuf3.template Get<ComputeT>();
        MishCompute(xLocal, yLocal, tmp1, tmp2, tmp3, sharedTmp, count);
    }

    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xLocal);
}

template <typename D_T_X>
__aicore__ inline void BenchNoHarnessMish<D_T_X>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<D_T_X> yLocal = outQueue.template DeQue<D_T_X>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(D_T_X));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[offset], yLocal, copyParams);
    outQueue.FreeTensor(yLocal);
}

template <typename D_T_X>
__aicore__ inline void BenchNoHarnessMish<D_T_X>::Process()
{
    // 空_tensor 直接返回
    if (tilingData_->totalLength == 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    int64_t myUbLoop;
    int64_t myUbTail;
    if (static_cast<uint64_t>(blockIdx) < tilingData_->blockNum - 1) {
        // 前核：处理完整 blockFormer
        myUbLoop = static_cast<int64_t>(tilingData_->ubLoopOfFormerBlock);
        myUbTail = static_cast<int64_t>(tilingData_->ubTailOfFormerBlock);
    } else {
        // 尾核：处理剩余元素
        myUbLoop = static_cast<int64_t>(tilingData_->ubLoopOfTailBlock);
        myUbTail = static_cast<int64_t>(tilingData_->ubTailOfTailBlock);
    }

    int64_t offset = 0;
    // 完整 UB 块循环
    for (int64_t i = 0; i < myUbLoop; i++) {
        CopyIn(offset, ubFormer_);
        Compute(ubFormer_);
        CopyOut(offset, ubFormer_);
        offset += ubFormer_;
    }
    // 尾部块
    if (myUbTail > 0) {
        CopyIn(offset, myUbTail);
        Compute(myUbTail);
        CopyOut(offset, myUbTail);
    }
}
```

## 设计决策

- **为何选数值稳定展开而不是朴素 `log(1+exp(x))`**：朴素式在大正 x（FP16 x>11、FP32 x>88）直接 inf；稳定式 exp 自变量恒 ≤0，全值域无溢出。代价是大负 x 的 `1+ε` 舍入边界（见失效边界），两害相权取溢出免疫。
- **为何三个 tmpBuf 轮转而非两个**：九步链中 Step5(Log) 与 Step8(Tanh) 都要求独立 dst；Step7 的 Add 需要 tmp2、tmp3 同时作输入，此时 tmp1 空闲做 dst——若只备 2 个 buffer，Step5 的 Log 结果与 Step6 的 Maxs 结果会争同一个 dst。3 个是"Log/Tanh 禁重叠 + 双目 Add 需两路输入"约束下的最小数量；三 buffer 方案让每一步的 dst 都不与任一 src 重叠，且 Step6(Maxs) 可复用 tmp2 而不破坏 Step5 结果（tmp3 持有）。两 buffer 方案需要更精细的活性分析，收益只有 4B/元素 UB。
- **为何 xBuf 全程只读、Step9 的 Mul 用独立 dst**：Mul 未声明支持重叠，且 x 要一直活到 Step9 作乘数；FP16/FP32 路径 dst 用 yLocal，BF16 路径用 tmp1（Cast 回 bf16 前的中转）。
- **为何 Tanh 显式传 sharedTmpBuffer 而不用"接口框架申请"重载**：显式传入可由 host tiling 统一预算 tmpBufferSize（保守按每元素字节估算），避免框架申请与 UB 预算脱节；框架申请方式需按 GetTanhMaxMinTmpSize 预留，本例未采用（备选，未对比）。
- **为何 BF16 走 Cast 升精度而不是 BF16 直算**：BF16 只有 8 位尾数，九步链累积误差不可控；Cast(FP32) 计算 + CAST_RINT 回转实测与 CPU golden 完全一致（MERE=0.00e+00）。FP16 路径维持直算（实测值域内达标），是精度/开销的折中。
- **为何 CopyIn/CopyOut 用 DataCopyPad**：非对齐 shape（如 3333、1000×1000）尾部不足 32B 对齐块，Pad 版搬运按 blockLen 字节直达，免整块对齐约束。
- **与设计文档的一个偏差**：设计文档原以为 Log 也带 sharedTmpBuffer 参数，核对头文件后发现 Log 的 calCount 版本没有该参数（临时空间框架自管），实际 sharedTmpBuf 只喂给 Tanh——以头文件签名而非文档叙述为准。
- 被放弃的设计：把 `max(x,0)` 与 `log1p` 合并成复用 buffer 的写法——需要四 buffer 且不省步数，放弃。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0，`bash build.sh --soc=ascend950` 产出 3 个 kernel 二进制（对应三 TilingKey）。
- 相关文件：`op_kernel/arch35/bench_no_harness_mish.h`（kernel 主体）、`op_kernel/bench_no_harness_mish_arch35.cpp`（入口）、`op_kernel/arch35/bench_no_harness_mish_tiling_key.h`（模板声明）、`op_kernel/arch35/bench_no_harness_mish_tiling_data.h`、`op_host/arch35/bench_no_harness_mish_tiling.cpp/.h`（tiling）、`op_api/aclnn_bench_no_harness_mish.cpp`（aclnn 层）。
- tiling 关键常量（950/arch35）：`UB_SIZE_950=253952`、`UB_RESERVE=2048`、`BLOCK_FORMER_ALIGN=512`；FP32 每元素 `bufferBytesPerElem=28`（in/out 队列 2×4×2 + 3 个 tmp×4）、`tmpBytesPerElem=4`，合计 32B/元素 → `ubFormer = FloorAlign(FloorDiv(251904,32),64) = 7872`、`tmpBufferSize=31488`。
- API 签名核对源：`$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/math/{log,tanh}.h`、basic_api 的 unary/binary 头。

## 性能影响

本轨迹未测量该链路性能（无 profiling 数据）；3 tmpBuf + sharedTmpBuf 的 UB 占用会压低 ubFormer（fp16 路径合计 14 字节/元素），属用空间换"禁重叠"约束下的正确性，性能代价未量化。

## 预防

- 写多步向量 API 链前先列 src/dst 表逐行查重叠；Log/Tanh 一律给独立 dst。
- count/calCount 类型逐 API 核对（int32_t vs uint32_t），不要统一一个变量硬传。
- softplus/激活类公式先换成数值稳定形式再落 kernel。
- 按设计文档（而非头文件签名）确定 API 参数表——Log 的 sharedTmpBuffer 就是一例文档与实现不一致。

## 验证信号

以下信号来自该算子三 dtype 全功能上板验证会话（多轮 ST Real 模式，NPU 输出 vs CPU golden；Ascend950PR + CANN 9.1.0）。

- 约束证据：Log/Tanh 文档"不支持源操作数与目的操作数地址重叠"、通用约束"未描述重叠约束视为不支持"（来源平台：950；证据等级：tool_observed）。
- 编译：`bash build.sh --soc=ascend950` exit 0，3 个 TilingKey kernel 二进制生成（来源平台：950；证据等级：tool_observed）。
- FP16（TK1）20/20 用例通过，MERE_max=3.28e-04 < 2^-10（来源平台：950；证据等级：tool_observed）。
- FP32（TK2）15/15 通过，MERE_max=6.96e-06 < 2^-13（大 shape [2048,2048] 4M 元素 MERE=1.13e-07）（来源平台：950；证据等级：tool_observed）。
- BF16（TK3）17/17 通过，MERE=0.00e+00（NPU Cast 上/下精度路径与 CPU golden 同算法完全一致）（来源平台：950；证据等级：tool_observed）。
- 边界用例：空 tensor、0 维标量、±1e5 极值、NaN/Inf 传播、非连续输入全部通过；非对齐 shape（3333、1000×1000）通过（来源平台：950；证据等级：tool_observed）。
- CPU golden 语义自测（ST mock，非 NPU）：mish(0)=0、mish(1)≈0.8651、mish(-1)≈-0.3034、mish(10)≈10、mish(-10)≈-0.000454、mish(+inf)=+inf、mish(-inf)=NaN（IEEE -inf×0）、mish(NaN)=NaN，20/20 通过（来源平台：950；证据等级：tool_observed，注意这是宿主 CPU golden 实现的自测，不是 kernel 上板信号）。

## 失效边界

- 数值稳定形式的推导按 fp16 溢出上限（x≥12 溢出）与"大数吃小数"机制给出；fp32 路径上 x≥89 才溢出，但稳定形式同样适用。未做两形式的逐位精度对比（未验证精度差异数值）。
- 大负 x 区间输出舍到 0（log1p 项坍缩），与 golden 的偏差按小值域标准管理；`Adds+Log` 组合模拟 log1p 的舍入边界是已知限制（本卡不展开）。
- buffer 数量与该 9 步链的具体数据流绑定；换算式（如用 Sub 代替 Muls 取负）需重排 src/dst 表。
- mish(-inf)=NaN 是公式语义（-inf·tanh(0)=-inf·0=NaN），对齐该语义的 golden 才适用本链。
- UB/对齐常量（253952/2048/512/64）绑定 950/arch35；跨架构移植按平台常量重算 ubFormer（未验证，推断可迁移）。
- FP16 直算路径的精度承诺限于实测安全值域；宽值域场景需评估 Cast 升精度路径。

**未决项**：

- Process 循环体的流水排布（CopyIn/Compute/CopyOut 重叠度）未做性能验证；如需优化先 profiling 再调 BUFFER_NUM。
- tmpBufferSize 保守估算与精确 API（GetTanhMaxMinTmpSize）的差值未实测。
## 坏实践

- 朴素展开 `log(1+exp(x))`——大正 x 必然溢出。
- Log/Tanh 的 dst 直接复用 src buffer——API 不支持地址重叠，结果是未定义/错值。
- BF16 直接进九步链——8 位尾数撑不住九步累积。
- count 类型混传（int32_t 接口喂 uint32_t 变量或反之）。

## 误诊

- 设计评审阶段曾按设计文档给 Log 传 4 参（含 sharedTmpBuffer）：实际 CANN 9.1.0 的 Log 只有 (dst,src[,count]) 签名，4 参版本是 Log2 的——按已装头文件核对后维持 3 参调用。
