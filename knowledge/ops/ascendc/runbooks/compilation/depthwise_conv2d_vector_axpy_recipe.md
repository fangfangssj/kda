---
type: "Runbook"
title: Depthwise 卷积 Vector Axpy 直接累加配方
description: depthwise 卷积不用 im2col+Cube（N=1 时阵列利用率 1/16），走 Vector 直接累加：DataCopyPad 逐 (kh,kw) 搬入带左右 padding 的行段、Duplicate 初始化 bias、Axpy&lt;float,half&gt; 融合乘加、Cast CAST_ROUND 回写；按 N×C 多核切分 + hTile 行分块，行 stride 32B 对齐。编译级验证。
tags: [task.implementation, topic.implementation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# Depthwise 卷积 Vector Axpy 直接累加配方

## 现象（任务）

在 Ascend950PR（DAV_3510，32 Cube 核 / 64 Vector 核，UB=248KB）上从零实现自定义算子 BenchNoHarnessDepthwiseConv2d 的 kernel：二维深度卷积 `y[n,c,oh,ow] = bias[c] + Σ_{kh,kw} x[n,c,oh*SH+kh*DH-PH, ow*SW+kw*DW-PW] * weight[c,kh,kw]`（groups=C，fp16 输入输出，首版限 stride=1/dilation=1）。要求不引入 im2col 中间内存、fp32 域累加避免 fp16 溢出，产出可编译的 arch35 kernel（本卡验证止于编译级：3 个 `.o` + custom_opp 安装包生成；上板精度/性能未验证）。

## 适用场景

- 工程形态：registry-invoke 算子仓开发，AscendC 手写 kernel（TQue/TBuf 管理 UB），per-channel 独立卷积类算子（depthwise / grouped conv channel_multiplier=1）。
- API 路径：`DataCopyPad<half>`（GM↔UB 非对齐搬运 + padding 填充）、`Duplicate<float>`（标量填充）、`Axpy<float, half>`（dst=float, src=half, scalar=half 融合乘加，每次迭代处理 64 个源元素）、`Cast<half, float>`（CAST_ROUND）、`GlobalTensor::GetValue` / `LocalTensor::GetValue`（标量读取）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（验证等级为编译级，未上板）。
- 适用粒度：operator_family——per-channel 逐元素加权累加类算子（depthwise conv、per-channel scale/shift）。
- 路线优先级：recommended_fastpath（正确性优先的首版实现；性能未测量，RegBase 列为后续可选优化）。
- 明确不适用 / 未验证场景：跨通道归约的普通卷积（应走 Cube/MMAD）；大 K（如 16×16=256 tap）时 Vector 累加次数多的性能代价未测量；bf16/fp32 路径与 strided 路径为空分支占位未实现；上板精度/多核切分正确性未验证。

## 根因/解决配方

核心思路：depthwise 每个输出通道只由对应输入通道卷积，无跨通道归约——把卷积核每个 (kh,kw) 位置的标量权重 `weight[c,kh,kw]` 作用于"一整段输出行"，正好是 Axpy 语义（`acc += w_scalar * x_seg`）。对每个 (kh,kw)，输出行 ow∈[0,WOut) 对应输入列段 `[kw-PW, kw-PW+WOut)`，用 DataCopyPad 的 leftPadding/rightPadding 把越界部分填 0，使 UB 里的行段长度恰为 WOut，与累加器逐元素对齐。

内存布局（UB，fp16 路径）：

- `halfStride = CeilAlign(WOut, 16)`（half 的 32B 对齐元素数）、`floatStride = CeilAlign(WOut, 8)`（float 的 32B 对齐元素数）——行首地址保证 32B 对齐，满足 Axpy/Cast 对 dst 起始地址的对齐要求。
- xQueue/yQueue：`TQue<VECIN/VECOUT, 2>` 双缓冲，每块 `hTile * halfStride` 个 half。
- accBuf：`TBuf<VECCALC>`，`hTile * floatStride` 个 float（fp32 累加器，单缓冲）。
- weightBuf：`CeilAlign(KH*KW, 16)` 个 half。
- UB 预算（host 侧 tiling 计算 hTile）：`perRowCost = 2(half)*2(双缓冲)*2(xQueue+yQueue)*halfStride + 4(float)*floatStride = 8*halfStride + 4*floatStride` 字节/行，`hTile = (ubSize - weightBufSize) / perRowCost`，再 clamp 到 [1, HOut]。

多核切分：按 N×C 维度（`ncTotal = N*C`，`blockFactor = CeilDiv(ncTotal, coreNum)`，coreNum 用 `GetCoreNumAiv()` 动态获取），每核串行处理自己的 (n,c) 段；SetBlockDim(usedCoreNum)。

padding 处理细节（stride=1/dilation=1）：

- 列向：每个 (kh,kw) 计算 `colStart = kw - PW`，有效段 `[validColStart, validColEnd)`，`leftPad = max(0, -colStart)`、`rightPad = max(0, colEnd - W)`，`DataCopyPadExtParams<half>{true, leftPad, rightPad, 0}` 让 UB 行段自动补 0；整列都在 padding 区（validCols<=0）时直接跳过该 (kh,kw)。
- 行向：输出行对应的输入行 `inRow = outH + kh - PH` 越界时，用 `Duplicate(xLocal[row*halfStride], (half)0, WOut)` 把 UB 该行清 0（AllocTensor 的初始内容不可信）。
- 权重标量为 0 时跳过该 (kh,kw)（`static_cast<float>(wScalar) == 0.0f`，half 与 0 比较先转 float）。

## 设计决策

- **Vector 直接累加 vs im2col+MatMul(Cube)**：选 Vector。depthwise 的 im2col 矩阵形状是 `[HOut*WOut, KH*KW] × [KH*KW, 1]`，N=1 时 Cube 16×16×16 阵列利用率仅 1/16，还要付 im2col workspace 与数据重排开销；小 K（3×3/5×5）时 Vector 效率高。大 K 时 Vector 累加次数多是已知劣势（未测量边界）。
- **通用 SIMD/MemBase vs RegBase**：选通用 SIMD。本算子目标是正确性与覆盖度优先；RegBase 编程模型复杂度高，且 Axpy 在标准 SIMD 路径已有含 half→float 融合累加的高效实现；RegBase 留作后续性能优化候选。
- **fp32 域累加**：`Axpy<float, half>(accFloat, xIn, wScalar, WOut)` 一条指令完成 half 源 × half 标量 → float 目的的乘加，既避免 fp16 逐次舍入/溢出又省一条显式 Cast。备选 Mul+Add 需先 Cast 且 Mul 不支持 bfloat16_t（bf16 迭代只能走 Cast+Axpy）。
- **逐 (kh,kw) DataCopyPad 而非整块 wInTile 加载**：每个 (kh,kw) 需要的输入列段起点不同（kw-PW），带左右 padding 的按需搬运让 UB 行段与累加器严格逐元素对齐，逻辑最简；代价是同一输入行可能被不同 kw 重复搬入（性能取舍，首版接受）。
- **行 stride 32B 对齐**：Axpy（dst=float 起始 32B 对齐）与 Cast 的硬件约束，同时避免跨行非对齐访问；行末 padding 区不参与计算（count=WOut）。
- 当前方案路线档位：recommended_fastpath（首版正确性路线）。

## 系统信息

- 硬件/软件：x86_64 主机交叉编译，CANN 9.2.0，Ascend950PR（arch35/DAV_3510，UB=248KB，64 AIV）。
- 相关文件（与事件流读写一一对应）：`op_kernel/arch35/bench_no_harness_depthwise_conv_2d.h`（Kernel 类）、`op_kernel/bench_no_harness_depthwise_conv_2d_arch35.cpp`（模板入口）、`op_kernel/arch35/bench_no_harness_depthwise_conv_2d_tiling_data.h`（TilingData 18 字段）、`op_kernel/arch35/bench_no_harness_depthwise_conv_2d_tiling_key.h`（ASCENDC_TPL DECL/SEL）、`op_host/arch35/bench_no_harness_depthwise_conv_2d_tiling.cpp`（host tiling：平台信息/校验/hTile/BlockDim/ASCENDC_TPL_SEL_PARAM）。
- 构建验证：`bash build.sh` → 3 个 `.o`（ascend950）+ `custom_opp_ubuntu_x86_64.run`。

## 完整代码

Kernel 类核心实现（`op_kernel/arch35/bench_no_harness_depthwise_conv_2d.h`，verbatim）：

```cpp
template <int IS_STRIDED>
__aicore__ inline void BenchNoHarnessDepthwiseConv2dOp<IS_STRIDED>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    const BenchNoHarnessDepthwiseConv2dTilingData* tilingData)
{
    // 读取 TilingData
    N_ = tilingData->N;
    C_ = tilingData->C;
    H_ = tilingData->H;
    W_ = tilingData->W;
    HOut_ = tilingData->HOut;
    WOut_ = tilingData->WOut;
    KH_ = tilingData->KH;
    KW_ = tilingData->KW;
    SH_ = tilingData->SH;
    SW_ = tilingData->SW;
    PH_ = tilingData->PH;
    PW_ = tilingData->PW;
    DH_ = tilingData->DH;
    DW_ = tilingData->DW;
    ncTotal_ = tilingData->ncTotal;
    blockFactor_ = tilingData->blockFactor;
    hTile_ = tilingData->hTile;

    // 计算行对齐 stride
    halfStride_ = AlignUp(WOut_, HALF_ALIGN_NUM);
    floatStride_ = AlignUp(WOut_, FLOAT_ALIGN_NUM);

    // 设置 GM Buffer
    xGm_.SetGlobalBuffer((__gm__ half*)x);
    weightGm_.SetGlobalBuffer((__gm__ half*)weight);
    biasGm_.SetGlobalBuffer((__gm__ half*)bias);
    yGm_.SetGlobalBuffer((__gm__ half*)y);

    // 初始化 UB Buffers
    // xQueue: 双缓冲，每块 hTile_ 行 × halfStride_ 个 half
    pipe_.InitBuffer(xQueue_, 2, hTile_ * halfStride_ * sizeof(half));
    // yQueue: 双缓冲，每块 hTile_ 行 × halfStride_ 个 half
    pipe_.InitBuffer(yQueue_, 2, hTile_ * halfStride_ * sizeof(half));
    // weightBuf: KH_ * KW_ 个 half（向上对齐 32B）
    pipe_.InitBuffer(weightBuf_, AlignUp(KH_ * KW_, HALF_ALIGN_NUM) * sizeof(half));
    // accBuf: hTile_ 行 × floatStride_ 个 float（fp32 累加器）
    pipe_.InitBuffer(accBuf_, hTile_ * floatStride_ * sizeof(float));
}

template <int IS_STRIDED>
__aicore__ inline void BenchNoHarnessDepthwiseConv2dOp<IS_STRIDED>::Process()
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t ncStart = blockIdx * blockFactor_;
    int64_t ncEnd = ncStart + blockFactor_;
    if (ncEnd > ncTotal_) {
        ncEnd = ncTotal_;
    }

    for (int64_t ncIdx = ncStart; ncIdx < ncEnd; ncIdx++) {
        ProcessOneChannel(ncIdx);
    }
}

template <int IS_STRIDED>
__aicore__ inline void BenchNoHarnessDepthwiseConv2dOp<IS_STRIDED>::ProcessOneChannel(int64_t ncIdx)
{
    // ncIdx = n * C + c
    int64_t n = ncIdx / C_;
    int64_t c = ncIdx % C_;

    // GM 偏移
    int64_t xOffset = n * C_ * H_ * W_ + c * H_ * W_;
    int64_t yOffset = n * C_ * HOut_ * WOut_ + c * HOut_ * WOut_;
    int64_t weightOffset = c * KH_ * KW_;

    // 1. 加载 weight[c, :, :] 到 UB
    LocalTensor<half> weightLocal = weightBuf_.Get<half>();
    DataCopyExtParams weightCopyParams{1, static_cast<uint32_t>(KH_ * KW_ * sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<half> weightPadParams{false, 0, 0, 0};
    DataCopyPad(weightLocal, weightGm_[weightOffset], weightCopyParams, weightPadParams);

    // 2. 获取 bias[c] 标量（GlobalTensor.GetValue，每 (n,c) 一次）
    half biasHalf = biasGm_.GetValue(c);
    float biasFloat = static_cast<float>(biasHalf);

    // 3. 按 hTile 分块处理 HOut 行
    for (int64_t hTileStart = 0; hTileStart < HOut_; hTileStart += hTile_) {
        int64_t rows = hTile_;
        if (hTileStart + rows > HOut_) {
            rows = HOut_ - hTileStart;
        }

        // 3.1 初始化累加器 = bias（覆盖 rows * floatStride_ 个元素，含行末 padding）
        LocalTensor<float> accFloat = accBuf_.Get<float>();
        Duplicate(accFloat, biasFloat, rows * floatStride_);

        // 3.2 遍历卷积核 (kh, kw)，逐位置 Axpy 累加
        for (int64_t kh = 0; kh < KH_; kh++) {
            for (int64_t kw = 0; kw < KW_; kw++) {
                half wScalar = weightLocal.GetValue(static_cast<uint32_t>(kh * KW_ + kw));
                // 权重为 0 时跳过（优化）
                // 注意：half 与 0 比较，需要通过 float
                if (static_cast<float>(wScalar) == 0.0f) {
                    continue;
                }

                // 计算输入列范围（stride=1, dilation=1: 输出 w 对应输入 w + kw - PW）
                int64_t colStart = kw - PW_;  // 输出 w=0 对应的输入列
                int64_t validColStart = colStart < 0 ? 0 : colStart;
                int64_t colEnd = colStart + WOut_;
                int64_t validColEnd = colEnd > W_ ? W_ : colEnd;
                int64_t validCols = validColEnd - validColStart;
                int64_t leftPad = colStart < 0 ? (-colStart) : 0;
                int64_t rightPad = colEnd > W_ ? (colEnd - W_) : 0;

                if (validCols <= 0) {
                    // 整列在 padding 区，贡献 0，跳过
                    continue;
                }

                // 申请 xQueue 一块 buffer
                LocalTensor<half> xLocal = xQueue_.template AllocTensor<half>();

                // 逐行加载输入（处理行 padding）
                for (int64_t row = 0; row < rows; row++) {
                    int64_t outH = hTileStart + row;
                    int64_t inRow = outH + kh - PH_;  // stride=1, dilation=1

                    if (inRow < 0 || inRow >= H_) {
                        // 输入行在 padding 区，该行贡献 0
                        // xLocal 对应行不需要加载（保持 AllocTensor 的初始状态）
                        // 但需要确保 Axpy 时该行数据为 0，否则会累加垃圾数据
                        // 用 Duplicate 填 0
                        Duplicate(xLocal[row * halfStride_], static_cast<half>(0), WOut_);
                        continue;
                    }

                    // 加载该行的有效列段（含 leftPad/rightPad 填充 0）
                    int64_t inRowOffset = xOffset + inRow * W_ + validColStart;
                    DataCopyExtParams rowCopyParams{
                        1, static_cast<uint32_t>(validCols * sizeof(half)), 0, 0, 0};
                    DataCopyPadExtParams<half> rowPadParams{
                        true, static_cast<uint8_t>(leftPad), static_cast<uint8_t>(rightPad), 0};
                    DataCopyPad(xLocal[row * halfStride_], xGm_[inRowOffset], rowCopyParams, rowPadParams);
                }

                xQueue_.EnQue(xLocal);
                LocalTensor<half> xIn = xQueue_.template DeQue<half>();

                // 逐行 Axpy 累加（行 stride 对齐，保证 32B 对齐）
                for (int64_t row = 0; row < rows; row++) {
                    Axpy<float, half>(accFloat[row * floatStride_], xIn[row * halfStride_], wScalar,
                                      static_cast<int32_t>(WOut_));
                }

                xQueue_.FreeTensor(xIn);
            }
        }

        // 3.3 Cast: float → half（逐行，行 stride 对齐）
        LocalTensor<half> yLocal = yQueue_.template AllocTensor<half>();
        for (int64_t row = 0; row < rows; row++) {
            Cast(yLocal[row * halfStride_], accFloat[row * floatStride_], RoundMode::CAST_ROUND,
                 static_cast<int32_t>(WOut_));
        }
        yQueue_.EnQue(yLocal);

        // 3.4 CopyOut: UB → GM（逐行）
        LocalTensor<half> yOut = yQueue_.template DeQue<half>();
        for (int64_t row = 0; row < rows; row++) {
            int64_t outH = hTileStart + row;
            int64_t outOffset = yOffset + outH * WOut_;
            DataCopyExtParams outCopyParams{
                1, static_cast<uint32_t>(WOut_ * sizeof(half)), 0, 0, 0};
            DataCopyPad(yGm_[outOffset], yOut[row * halfStride_], outCopyParams);
        }
        yQueue_.FreeTensor(yOut);
    }
}
```

Host 侧 tiling 的 UB 预算与多核切分（`op_host/arch35/bench_no_harness_depthwise_conv_2d_tiling.cpp`，verbatim 节选）：

```cpp
    // 4. 多核切分：按 N×C 维度
    int64_t ncTotal = N * C;
    OP_CHECK_IF(ncTotal <= 0,
                OP_LOGE(context, "ncTotal must be positive: N=%ld, C=%ld", N, C),
                return ge::GRAPH_FAILED);
    int64_t blockFactor = CeilDiv(ncTotal, coreNum);

    // 5. UB 切分：计算 hTile
    //    UB 预算（fp16 路径，双缓冲）：
    //    - xQueue (half, 双缓冲): 2 * hTile * halfStride * sizeof(half)
    //    - yQueue (half, 双缓冲): 2 * hTile * halfStride * sizeof(half)
    //    - accBuf (float):        hTile * floatStride * sizeof(float)
    //    - weightBuf (half):      CeilAlign(KH*KW, 16) * sizeof(half)
    //    halfStride = CeilAlign(WOut, 16)  (half 32B 对齐)
    //    floatStride = CeilAlign(WOut, 8)  (float 32B 对齐)
    int64_t halfStride = CeilAlign(WOut, HALF_ALIGN_NUM);
    int64_t floatStride = CeilAlign(WOut, FLOAT_ALIGN_NUM);
    int64_t weightBufSize = CeilAlign(KH * KW, HALF_ALIGN_NUM) * HALF_SIZE;
    // perRowCost = 2*halfStride*2(half,双缓冲) + 2*halfStride*2(half,双缓冲) + floatStride*4(float)
    //            = 8 * halfStride + 4 * floatStride  (字节)
    int64_t perRowCost = BUFFER_NUM * halfStride * HALF_SIZE * 2 +  // xQueue + yQueue（均为 half 双缓冲）
                         floatStride * FLOAT_SIZE;                   // accBuf（float 单缓冲）
    int64_t availableUb = static_cast<int64_t>(ubSize) - weightBufSize;
    int64_t hTile = 1;
    if (perRowCost > 0 && availableUb > perRowCost) {
        hTile = availableUb / perRowCost;
    }
    // 限制 hTile 不超过 HOut，至少为 1
    hTile = std::max(int64_t(1), std::min(hTile, HOut));
```

```cpp
    // 8. 设置 BlockDim
    int64_t usedCoreNum = CeilDiv(ncTotal, blockFactor);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 9. 设置 TilingKey（ASCENDC_TPL_SEL_PARAM 按 dtype/stride 编码）
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint32_t isStridedVal = 0;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, isStridedVal);
```

核函数模板入口（`op_kernel/bench_no_harness_depthwise_conv_2d_arch35.cpp`，终版 verbatim）：

```cpp
#include "arch35/bench_no_harness_depthwise_conv_2d.h"

template <typename D_T_X, int IS_STRIDED>
__global__ __aicore__ void bench_no_harness_depthwise_conv2d(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessDepthwiseConv2dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessDepthwiseConv2dTilingData, tilingData, tiling);

    // 按 dtype 分发到具体实现类（if constexpr 编译期分发）
    // 当期仅 fp16 路径实现，fp32/bf16 路径留空（后续迭代填充）
    if constexpr (std::is_same_v<D_T_X, half>) {
        NsBenchNoHarnessDepthwiseConv2d::BenchNoHarnessDepthwiseConv2dOp<IS_STRIDED> op;
        op.Init(x, weight, bias, y, &tilingData);
        op.Process();
    } else {
        // 后续扩展：fp32 / bf16 路径（当前留空，避免实例化失败）
    }
}
```

## 性能影响

无实测性能数据（本会话未上板、无 profiling）；相对路线对比仅有设计推算：相对 im2col+Cube 路线省去 workspace 与数据重排，但大 K 时累加次数多（未验证，推断可迁移的定性结论）。

## 预防

- depthwise/逐通道加权类算子先想 Vector Axpy，不要条件反射上 im2col+MatMul。
- UB 行 stride 一律按 32B 对齐（half 16 元素 / float 8 元素）再算缓冲预算。
- AllocTensor 拿到的 buffer 内容不可信，padding 行必须显式 Duplicate 清 0。

## 验证信号

以下信号来自该算子首版骨架开发会话：kernel/tiling/入口全链路编写后经多次编译错误修复（各根因独立闭环），最终全量编译通过。注意：验证止于编译级，上板精度/性能未验证。

- `rm -rf build && cmake -DASCEND_COMPUTE_UNIT=ascend950 .. && cmake --build . --target all binary package -- -j8` 全绿（来源平台：950；证据等级：tool_observed）。
- `op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_depthwise_conv2d/` 生成 3 个 `.o` 与 3 个 `.json`；`custom_opp_ubuntu_x86_64.run`（约 460KB）生成（来源平台：950；证据等级：tool_observed）。
- `bash build.sh` 端到端成功，`[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。
- 编译通过的用法锚点：`Axpy<float, half>` dst=float/src=half/scalar=half 每次迭代处理 64 个源元素、dst 起始 32B 对齐、scalar 类型须与 src 一致；`Cast<half,float>` 高→低精度用 CAST_ROUND；Mul 不支持 bfloat16_t（bf16 路线必须 Cast+Axpy）（API 语义来自 asc-devkit 文档，可编译性为本会话实测）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 编译级验证：fp16 精度、多核切分正确性、DataCopyPad padding 行为均未上板验证（NPU 实测未在本闭环发生，属优先待办）。
- 首版限 stride=1/dilation=1：strided 路径的列段计算需重推（`w*SW + kw*DW - PW`），当前公式不适用。
- 大 K 场景（K≥16）性能未测量；Axpy 64 元素/迭代的吞吐约束下可能成为瓶颈（未验证）。
- DAV_3510 之外的架构：Axpy/DataCopyPad 语义相同但核数/UB 容量不同，hTile 公式需按平台参数重算（未验证，推断可迁移）。

**未决项**：

- 优先项：上板验证 fp16 精度（对照 CPU golden）与多核切分正确性——本会话 ST 仅 Mock/CPU 级，NPU 实测未发生。
- strided（IS_STRIDED=1）路径的列段/行映射公式与 padding 处理待后续迭代实现。
- 逐 (kh,kw) 重复搬入的带宽代价与 wInTile 整块加载方案的对比未测量。
## 坏实践

- depthwise 上 im2col+Cube：N=1 使 Cube 利用率 1/16，还引入 workspace 与重排开销。
- 累加器直接用 half 逐次累加：逐次舍回原生 dtype 的精度损失（应 fp32 域累加）。
- 依赖 AllocTensor 缓冲的"初始状态"：padding 行不清 0 会累加垃圾数据。

## 误诊

- 无（本卡为实现配方，无失败闭环；相关编译失败见各 build 卡）。
