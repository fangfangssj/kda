---
type: "Runbook"
title: 双线性插值 kernel 行缓冲复用：DataCopyPad 批量整行搬运
description: 双线性插值类 kernel 用 DataCopyPad 按 DataCopyExtParams 批量搬 cLen 通道整行到 UB（GM 字节 stride + UB 32 字节对齐行距），内层循环改读 LocalTensor，小行宽回退 GM GetValue；配套 tiling UB 预算公式。
tags: [task.implementation, topic.data_movement, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 双线性插值 kernel 行缓冲复用：DataCopyPad 批量整行搬运

## 现象

BenchNoHarnessRoiAlign（roi_align 双线性插值，arch35/Ascend950PR，CANN 9.1.0）早期版本的 kernel 对每个采样点的 4 个插值邻居都直接 `GlobalTensor::GetValue` 读 GM（每个输出元素 4 次采样 × 4 邻居 × 逐通道），GM 标量访问次数巨大；API 最佳实践明确"GetValue/SetValue 仅限调试，生产代码应使用 DataCopyPad 批量搬运"。更早版本曾因 `DataCopyPad blockLen < 32 + 大 stride 在 gather 场景不可靠`回退到全标量实现。最终重构在保持精度模型不变（fp32 中间累加、坐标计算顺序对齐 golden）的前提下改为行缓冲复用：同一 `(oh, gh)` 的所有 `ow/gw` 采样共享 `y_low/y_high` 两行数据，用一次 `DataCopyPad` 把 `cLen` 个通道的整行搬进 UB，内层循环改读 `LocalTensor::GetValue`。重构后编译通过、ST 36 条用例 34 条通过（2 条失败经排除实验证明是与本优化无关的存量缺陷，前一版 22 条无回归）。

## 适用场景

- 工程形态：自定义算子 AscendC kernel 开发（ops-registry-invoke 算子仓，`op_kernel/arch35` + `op_host/arch35` tiling）；对"二维邻域采样/插值/窗口聚合"类 kernel 的 GM→UB 数据搬运优化。
- API 路径：`DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams)`、`TBuf<TPosition::VECCALC>::InitBuffer/Get`、`Duplicate`、`Muls`、`PipeBarrier<PIPE_ALL>`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——一切"按行/按窗口重复读取同一批源数据"的插值与池化类算子（bilinear/ROI/窗口聚合），行宽与通道切分方式可变。
- 路线优先级：recommended_fastpath（正确性已回归验证；性能收益未实测，见性能影响）。
- 明确不适用 / 未验证场景：
  - 行字节数 `W*sizeof(T) < 32`（小 W + fp16）时不走行缓冲——本配方显式回退 GM GetValue（早期经验：blockLen < 32 且带大 stride 的 gather 型 DataCopyPad 不可靠）。
  - `blockCount`（= cLen，一次搬运的通道数）受 DataCopyExtParams 的 uint16 上限约束（最佳实践标注 max 4095）；cTile 超出时需分批（本实现由 tiling 保证 cTile ≤ 预算，未单独验证 4095 边界）。
  - 非行连续的随机 gather 模式不适用（stride 语义只表达"行间跳过 H*W-W"这类规则间隔）。
  - 性能收益未做 profiling 实测，不能作为性能承诺。

## 根因/解决配方

结构：`Init` 按 `W*sizeof(T) >= 32` 决定是否启用行缓冲；`ProcessOneRoi` 在每个 `(oh, gh)` 采样行上先批量搬 `cLen` 通道的 `y_low/y_high` 两整行到 UB，再让内层 `ow/gw` 循环从 UB 标量读取做双线性插值累加；accBuf 清零用 `Duplicate`、除以采样数用 `Muls` 向量化；fp16 路径在 GetValue/SetValue 边界用 `static_cast` 升降精度，中间全程 fp32。

关键机制点：

1. **GM 源地址与搬运参数**：第 c 通道 y_low 行的 GM 起点 = `batchOffset + cStart*H*W + yLow*W`；`DataCopyExtParams{blockCount=cLen, blockLen=W*sizeof(T)（字节）, srcStride=(H*W-W)*sizeof(T)（GM 侧字节）, dstStride=0（UB 侧 32 字节块）}`——一次调用完成 cLen 个通道同位行的搬运。
2. **UB 行距 32 字节对齐**：DataCopyPad 在 UB 侧每个 block 自动按 32 字节对齐存放，`rowStride = align_up(W*sizeof(T), 32) / sizeof(T)`（元素数）；第 c 通道行首 = `c * rowStride`。
3. **tiling UB 预算公式**（行缓冲路径）：`accBuf + rowBufLow + rowBufHigh + boxesBuf ≤ ubSize × 0.9`，其中 `accBuf = ohTile*ow*cTile*sizeof(float)`、`rowBuf = cTile × align_up(W*typeSize, 32)`，`cTile = FloorDiv(ubAvail - 32, ohTile*ow*4 + 2*align_up(W*typeSize,32))`；`W*typeSize < 32` 回退路径退化为 `ohTile*ow*4 + 4*4` 分母。
4. **同步**：每次 DataCopyPad 后 `PipeBarrier<PIPE_ALL>()` 再读 UB；`Duplicate` 清零后、`Muls` 后各一道 PipeBarrier 保证 Vector 写入对标量 GetValue 可见。
5. **y_low == y_high 时也始终预取 rowBufHigh**（冗余搬运换正确性，避免读到上一轮脏数据）；accBuf 字节数向上对齐到 32 字节再 InitBuffer（避免尾部元素丢失）。

## 设计决策

- 为什么按 `(oh, gh)` 行粒度做缓冲而不是逐输出点：双线性插值中 `y_low/y_high` 两行被同一采样行的全部 `ow × gridW` 个采样点共享，行粒度搬运把 GM 访问从"每采样点 4×cLen 次"降为"每采样行 2 次 DataCopyPad"。
- 为什么 blockLen 用 `W*sizeof(T)` 且只对 `>= 32` 启用：早期经验表明 `blockLen < 32 + 大 stride` 的 gather 型 DataCopyPad 不可靠；整行（blockLen ≥ 32 字节）搬运实测可靠。小行宽场景显式回退 `GlobalTensor::GetValue` 保正确性。
- 为什么 `dstStride = 0`：UB 侧相邻 block 自动 32 字节对齐紧凑存放，行距由 `rowStride`（元素步长）在读取侧计算，不需要额外 UB 间隔。
- 为什么 fp16 不用向量 `Cast`：调研过 `RoundMode::CAST_ROUND` 的 Cast 向量路径，但最终实现在标量插值边界用 `static_cast<float>/static_cast<T>` 升降精度——插值本身逐元素标量，边界转换随之标量化，且坐标/权重计算顺序与 golden 逐位对齐（binSize 先除后乘等）；向量 Cast 方案未在本算子验证。
- 为什么清零/除法用 `Duplicate`/`Muls`：accBuf 每轮需清零、累加完需整体除以 count，向量 API 一次完成；两者均以 `PipeBarrier` 收尾保证后续标量读可见。
- 放弃的候选：纯 `GlobalTensor::GetValue` 全标量版（正确但 GM 访问次数大）；`DataCopyPad blockLen=4` 逐邻居小搬运（已证明不可靠）。
- 当前方案路线档位：recommended_fastpath（正确性回归通过；性能档位未测）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0（`x86_64-linux/asc/include` 头文件树、`opp/vendors` 自定义包安装）。
- 构建/安装/测试循环：`bash build.sh --soc=ascend950 -j8` → `bash build/custom_opp_ubuntu_x86_64.run --force` → `export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_roi_align_custom/op_api/lib/:...` → `cd tests/st && bash run.sh --set all`（Real 模式）。
- 相关文件（与实际读写一一对应）：
  - `op_kernel/arch35/bench_no_harness_roi_align.h`——kernel 主体（Init 行缓冲分配 + ProcessOneRoi 插值）。
  - `op_host/arch35/bench_no_harness_roi_align_tiling.cpp`——TilingFunc UB 预算公式。
  - `op_kernel/arch35/bench_no_harness_roi_align_tiling_data.h` / `..._tiling_key.h`——tiling 数据与 dtype 分支键。
  - `tests/st/test_aclnn_bench_no_harness_roi_align.cpp` + `tests/st/run.sh`——ST 精度回归。
  - `tests/ut/op_host/test_bench_no_harness_roi_align_tiling.cpp`——tiling UT（cTile 期望值需随公式更新）。
- API 参考依据：`asc/include/basic_api/kernel_struct...`（DataCopyExtParams 定义）、`ascendc-api-best-practices` 的 api-datacopy/api-buffer（stride 单位、blockCount 上限、GetValue/SetValue 禁用于生产）。

## 完整代码

kernel 主体（`op_kernel/arch35/bench_no_harness_roi_align.h`，verbatim 取自本会话完整写入事件，含行缓冲复用全功能实现；类成员与 Init）：

```cpp
template <typename D_T_X>
class BenchNoHarnessRoiAlign {
public:
    using T = D_T_X;

    __aicore__ inline BenchNoHarnessRoiAlign() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
                                const BenchNoHarnessRoiAlignTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneRoi(int64_t roiIdx);

    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf_;      // 累加缓冲 [ohTile * ow * cTile] fp32
    TBuf<TPosition::VECCALC> rowBufLow_;   // y_low 行缓存 [cTile * rowStride] T
    TBuf<TPosition::VECCALC> rowBufHigh_;  // y_high 行缓存 [cTile * rowStride] T

    GlobalTensor<T> xGm_;       // 特征图 [B, C, H, W]
    GlobalTensor<T> boxesGm_;   // ROI 框 [numBoxes, 5]
    GlobalTensor<T> yGm_;       // 输出 [numBoxes, C, outputHeight, outputWidth]

    int64_t numBoxes_ = 0;
    int64_t boxesPerCore_ = 0;
    int64_t B_ = 0;
    int64_t C_ = 0;
    int64_t H_ = 0;
    int64_t W_ = 0;
    int64_t outputHeight_ = 0;
    int64_t outputWidth_ = 0;
    float spatialScale_ = 0.0f;
    int64_t samplingRatio_ = 0;
    int64_t aligned_ = 0;
    int64_t cTile_ = 0;
    int64_t ohTile_ = 0;
    int64_t rowStride_ = 0;     // UB 中每行的元素步长（含 32 字节对齐 padding）
    bool useRowBuf_ = false;    // 是否使用行缓冲（W*sizeof(T) >= 32 时启用）
};

template <typename D_T_X>
__aicore__ inline void BenchNoHarnessRoiAlign<D_T_X>::Init(
    GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
    const BenchNoHarnessRoiAlignTilingData* tilingData)
{
    numBoxes_ = tilingData->numBoxes;
    boxesPerCore_ = tilingData->boxesPerCore;
    B_ = tilingData->B;
    C_ = tilingData->C;
    H_ = tilingData->H;
    W_ = tilingData->W;
    outputHeight_ = tilingData->outputHeight;
    outputWidth_ = tilingData->outputWidth;
    // spatialScale 从 bit pattern 还原
    {
        union { int32_t i; float f; } u;
        u.i = static_cast<int32_t>(tilingData->spatialScaleBits);
        spatialScale_ = u.f;
    }
    samplingRatio_ = tilingData->samplingRatio;
    aligned_ = tilingData->aligned;
    cTile_ = tilingData->cTile;
    ohTile_ = tilingData->ohTile;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    boxesGm_.SetGlobalBuffer((__gm__ T*)boxes);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    // accBuf: ohTile * outputWidth * cTile 个 float
    // 对齐到 32 字节边界（避免 InitBuffer 截断导致尾部元素丢失）
    int64_t accBufBytes = ohTile_ * outputWidth_ * cTile_ * sizeof(float);
    accBufBytes = (accBufBytes + 31) / 32 * 32;
    pipe.InitBuffer(accBuf_, accBufBytes);

    // 行缓冲：判断 W*sizeof(T) 是否 >= 32 字节
    // 早期经验：DataCopyPad blockLen < 32 + 大 stride 在 gather 场景不可靠
    // 对 blockLen >= 32 的整行搬运，DataCopyPad 可靠
    int64_t blockLenBytes = W_ * sizeof(T);
    useRowBuf_ = (blockLenBytes >= 32);

    if (useRowBuf_) {
        // 计算 UB 中每行的元素步长（32 字节对齐）
        int64_t blockLenAligned = (blockLenBytes + 31) / 32 * 32;
        rowStride_ = blockLenAligned / sizeof(T);
        // 分配行缓冲（cTile 通道 × rowStride 元素 × sizeof(T)）
        pipe.InitBuffer(rowBufLow_, cTile_ * rowStride_ * sizeof(T));
        pipe.InitBuffer(rowBufHigh_, cTile_ * rowStride_ * sizeof(T));
    }
}
```

ProcessOneRoi 的插值主循环（行缓冲批量搬运 + 内层 UB 读取，verbatim；后续加固点"始终预取 rowBufHigh"已并入）：

```cpp
            // === 3. 双线性插值 + 累加 ===
            for (int64_t oh = 0; oh < ohLen; oh++) {
                for (int32_t gh = 0; gh < gridH; gh++) {
                    // 采样点 y (匹配 golden.py: y = roi_start_h + oh*bin_size_h + (gh+0.5)*(bin_size_h/grid_h))
                    float y = roiStartH + static_cast<float>(ohStart + oh) * binSizeH +
                              (static_cast<float>(gh) + 0.5f) * binSizeHPerGrid;
                    if (y < 0.0f) y = 0.0f;

                    int32_t yLow = static_cast<int32_t>(y);
                    int32_t yHigh;
                    float ly;
                    if (yLow >= Hm1) {
                        yLow = Hm1; yHigh = Hm1;
                        ly = y - static_cast<float>(yLow);
                    } else {
                        yHigh = yLow + 1;
                        ly = y - static_cast<float>(yLow);
                    }
                    float hy = 1.0f - ly;

                    int64_t rowOffLow = static_cast<int64_t>(yLow) * W_;
                    int64_t rowOffHigh = static_cast<int64_t>(yHigh) * W_;

                    // 行缓冲复用：同一 (oh, gh) 的所有 ow 共享 y_low/y_high 行数据
                    if (useRowBuf_) {
                        // 使用 DataCopyPad 批量搬运 cLen 通道的整行数据到 UB
                        auto rowLowLocal = rowBufLow_.Get<T>();
                        auto rowHighLocal = rowBufHigh_.Get<T>();

                        // GM 源地址：xGm[batchOffset + cStart * H * W + yLow * W]
                        int64_t gmOffsetLow = batchOffset + cStart * H_ * W_ + rowOffLow;
                        int64_t gmOffsetHigh = batchOffset + cStart * H_ * W_ + rowOffHigh;

                        // DataCopyExtParams: blockCount=cLen, blockLen=W*sizeof(T),
                        //   srcStride=(H*W-W)*sizeof(T) [GM 字节], dstStride=0 [UB 32字节块]
                        DataCopyExtParams copyParams;
                        copyParams.blockCount = static_cast<uint16_t>(cLen);
                        copyParams.blockLen = static_cast<uint32_t>(W_ * sizeof(T));
                        copyParams.srcStride = static_cast<int64_t>((H_ * W_ - W_) * sizeof(T));
                        copyParams.dstStride = 0;

                        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

                        DataCopyPad(rowLowLocal, xGm_[gmOffsetLow], copyParams, padParams);
                        // 始终预取 rowBufHigh：当 yLow==yHigh 时两行数据相同，
                        // 冗余搬运但保证 rowBufHigh 内容正确（避免读到上一轮脏数据）
                        DataCopyPad(rowHighLocal, xGm_[gmOffsetHigh], copyParams, padParams);
                        // 确保 MTE2 搬运完成后再读取 UB
                        PipeBarrier<PIPE_ALL>();
                    }

                    for (int64_t ow = 0; ow < outputWidth_; ow++) {
                        for (int32_t gw = 0; gw < gridW; gw++) {
                            // 采样点 x (匹配 golden.py)
                            float xPos = roiStartW + static_cast<float>(ow) * binSizeW +
                                         (static_cast<float>(gw) + 0.5f) * binSizeWPerGrid;
                            if (xPos < 0.0f) xPos = 0.0f;

                            int32_t xLow = static_cast<int32_t>(xPos);
                            int32_t xHigh;
                            float lx;
                            if (xLow >= Wm1) {
                                xLow = Wm1; xHigh = Wm1;
                                lx = xPos - static_cast<float>(xLow);
                            } else {
                                xHigh = xLow + 1;
                                lx = xPos - static_cast<float>(xLow);
                            }
                            float hx = 1.0f - lx;

                            // 预计算权重
                            float wht1 = hy * hx;
                            float wht2 = hy * lx;
                            float wht3 = ly * hx;
                            float wht4 = ly * lx;

                            // 逐通道标量插值 + 累加
                            // fp16 路径: static_cast<float> 将 half 提升为 fp32 (对应设计 Cast 升 fp32)
                            int64_t accBase = oh * outputWidth_ * cLen + ow * cLen;
                            if (useRowBuf_) {
                                // 从 UB 行缓冲读取（快速）
                                auto rowLowLocal = rowBufLow_.Get<T>();
                                auto rowHighLocal = rowBufHigh_.Get<T>();
                                for (int64_t c = 0; c < cLen; c++) {
                                    int64_t rowIdx = c * rowStride_;
                                    float v1 = static_cast<float>(rowLowLocal.GetValue(rowIdx + xLow));
                                    float v2 = static_cast<float>(rowLowLocal.GetValue(rowIdx + xHigh));
                                    float v3 = static_cast<float>(rowHighLocal.GetValue(rowIdx + xLow));
                                    float v4 = static_cast<float>(rowHighLocal.GetValue(rowIdx + xHigh));

                                    float val = wht1 * v1 + wht2 * v2 + wht3 * v3 + wht4 * v4;
                                    float accVal = accLocal.GetValue(accBase + c);
                                    accLocal.SetValue(accBase + c, accVal + val);
                                }
                            } else {
                                // 回退路径：直接从 GM 读取（小 W 场景）
                                for (int64_t c = 0; c < cLen; c++) {
                                    int64_t chOff = (cStart + c) * H_ * W_;
                                    float v1 = static_cast<float>(xGm_.GetValue(batchOffset + chOff + rowOffLow + xLow));
                                    float v2 = static_cast<float>(xGm_.GetValue(batchOffset + chOff + rowOffLow + xHigh));
                                    float v3 = static_cast<float>(xGm_.GetValue(batchOffset + chOff + rowOffHigh + xLow));
                                    float v4 = static_cast<float>(xGm_.GetValue(batchOffset + chOff + rowOffHigh + xHigh));

                                    float val = wht1 * v1 + wht2 * v2 + wht3 * v3 + wht4 * v4;
                                    float accVal = accLocal.GetValue(accBase + c);
                                    accLocal.SetValue(accBase + c, accVal + val);
                                }
                            }
                        }
                    }
                }
            }

            // === 4. 除以 count 并写出 ===
            // 确保 accBuf 的累加写入对后续操作可见
            PipeBarrier<PIPE_ALL>();

            // 向量化除以 count（Muls 替代标量逐元素乘法）
            Muls<float>(accLocal, accLocal, invCount, static_cast<int32_t>(accTotal));
            PipeBarrier<PIPE_ALL>();

            // 标量写出（fp16 路径: static_cast<T> 将 fp32 回写为 half）
            int64_t roiOffset = roiIdx * C_ * outputHeight_ * outputWidth_;
            int64_t channelBase = cStart * outputHeight_ * outputWidth_;

            for (int64_t oh = 0; oh < ohLen; oh++) {
                for (int64_t ow = 0; ow < outputWidth_; ow++) {
                    int64_t srcOff = oh * outputWidth_ * cLen + ow * cLen;
                    int64_t dstOff = roiOffset + channelBase + (ohStart + oh) * outputWidth_ + ow;
                    for (int64_t c = 0; c < cLen; c++) {
                        float val = accLocal.GetValue(srcOff + c);
                        yGm_.SetValue(dstOff + c * outputHeight_ * outputWidth_, static_cast<T>(val));
                    }
                }
            }
```

accBuf 清零（Duplicate 向量化，verbatim）：

```cpp
            // 获取 accBuf 并清零 (向量化: Duplicate 替代标量循环)
            auto accLocal = accBuf_.Get<float>();
            int64_t accTotal = ohLen * outputWidth_ * cLen;
            Duplicate<float>(accLocal, 0.0f, static_cast<int32_t>(accTotal));
            // 确保 accBuf 清零写入 (Vector) 对后续标量 GetValue 可见
            PipeBarrier<PIPE_ALL>();
```

tiling UB 预算公式（`op_host/arch35/bench_no_harness_roi_align_tiling.cpp`，verbatim）：

```cpp
    // 4. UB 切分：计算 ohTile 和 cTile
    int64_t ohTile = (outputHeight < OH_TILE_MAX) ? outputHeight : OH_TILE_MAX;

    // 行缓冲复用优化版 UB 预算公式:
    //   accBuf    = ohTile * ow * cTile * sizeof(float)   [累加缓冲, 中间计算全程 fp32]
    //   rowBufLow = cTile * wAlignedBytes                 [y_low 行缓存, DataCopyPad 批量搬运]
    //   rowBufHigh= cTile * wAlignedBytes                 [y_high 行缓存]
    //   boxesBuf  = 32B
    //   约束: accBuf + rowBufLow + rowBufHigh + boxesBuf <= ubSize * 0.9
    //
    //   wAlignedBytes = align_up(W * typeSize, 32)  [DataCopyPad UB 32 字节对齐]
    //   typeSize = sizeof(T) = (fp32:4, fp16:2)
    //
    //   对 W*typeSize < 32 的场景（小 W + fp16），Kernel 不分配行缓冲，
    //   回退到 GlobalTensor::GetValue，此时 UB 预算退化为仅 accBuf + boxesBuf。
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    int64_t ow = outputWidth;
    int64_t blockLenBytes = W * typeSize;
    bool useRowBuf = (blockLenBytes >= 32);

    int64_t ubAvail = static_cast<int64_t>(static_cast<float>(ubSize) * UB_USAGE_RATIO);

    int64_t denom;
    int64_t fixedOverhead;
    if (useRowBuf) {
        // 行缓冲路径：accBuf + 2*rowBuf 系数
        int64_t wAlignedBytes = (blockLenBytes + 31) / 32 * 32;
        denom = ohTile * ow * FLOAT_SIZE + 2 * wAlignedBytes;
        fixedOverhead = BOXES_BUF_SIZE;
    } else {
        // 回退路径：仅 accBuf + 少量 workBuf 预留
        denom = ohTile * ow * FLOAT_SIZE + 4 * FLOAT_SIZE;
        fixedOverhead = BOXES_BUF_SIZE;
    }
    int64_t cTile = FloorDiv(ubAvail - fixedOverhead, denom);
    if (cTile < 1) {
        cTile = 1;
    }
```

## 性能影响

本轨迹未采集 profiling 数据（无 msprof/STATE 实测），GM 访问次数从"每采样点 4×cLen 次标量 GM 读"降为"每采样行 2 次批量 DataCopyPad"是结构事实而非实测收益；采用前请自行测量。不构成可引用的性能结论。

## 预防

- 改 kernel 缓冲结构（新增/删除 UB buffer）时，同步修改 tiling 的 UB 预算公式，并更新 tiling UT 的 cTile 期望值——本会话因公式变更导致 3 个硬编码期望值（1056/512/1024）的 UT 失败，后改为按公式推导期望值（`CalcExpectedCTile` 辅助函数）。
- 使用 DataCopyPad 前核对两侧 stride 单位：GM 侧是字节，UB 侧是 32 字节块；UB 内行距要按 `align_up(blockLen, 32)` 计算。
- 行缓冲内容跨迭代复用时，注意"本轮未填充的 buffer 被读到脏数据"——`y_low == y_high` 跳过预取就是此类陷阱，要么始终填充要么读侧区分。

## 验证信号

以下信号来自该算子重构会话（Ascend950PR Real 模式）：kernel 重构为行缓冲复用后走完整"编译 → 安装 → ST Real 模式回归 → 排除实验"链路。

- 编译/打包：`bash build.sh --soc=ascend950 -j8` 输出 `[INFO] Build completed successfully!`，生成双 dtype kernel 二进制（fp32/fp16 两个 .o）与 `custom_opp_ubuntu_x86_64.run`，`--force` 安装 SUCCESS（来源平台：950；证据等级：tool_observed）。
- ST 回归（Real 模式，36 条 = 前版 22 条 + 新增 14 条）：34 通过 2 失败；失败的两条（L1_007/L1_010）均为新增用例、不在前版集合内，前版 22 条全部通过——行缓冲优化无回归（来源平台：950；证据等级：tool_observed）。
- 排除实验（证明 2 条失败与本优化无关）：`useRowBuf_ = false`（禁用行缓冲、回退 GM GetValue）后 L1_007 仍 `MERE=6.990e-02, MARE=9.960e-01`、L1_010 仍同量级失败；移除 `Muls`（改标量乘 invCount）、`Duplicate` 改标量清零循环后均仍失败；调试改动全部回退后 34/36 维持（来源平台：950；证据等级：tool_observed）。
- UT 回归：op_api UT 26/26 通过（重构后）；op_host tiling UT 在更新 cTile 期望值后 46/46 通过（含按新公式推导期望值的 3 个修复用例与 11 个新增用例，合计 80/80）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `W*sizeof(T) < 32` 时走 GM GetValue 回退路径，本配方的搬运优化不生效（正确性仍由回退路径保证）。
- cTile 由 UB 预算决定；`blockCount = cLen` 受 uint16 限制（最佳实践标注 max 4095），超大 cTile 需拆批（本会话未触发该边界）。
- "blockLen < 32 + 大 stride 不可靠"的结论来自本算子早期迭代的 gather 场景经验，其他访问形态未验证；不要据此推断所有小 blockLen 都不可用。
- 仅在 Ascend950PR + CANN 9.1.0（arch3510）实测；其他 arch 的 UB 大小、DataCopyPad 语义细节需按平台常量重算（未验证，推断可迁移）。
- L1_007/L1_010 两条失败用例的根因在本段轨迹内未定位（见遗留问题），采用本配方不能解决它们。

**未决项**：

- 行缓冲复用的性能收益未测量（无 profiling 数据）；下一测量点：对同 shape 对比 useRowBuf 开/关两条路线的耗时（未尝试）。
- `DataCopyPad blockLen < 32 + 大 stride 在 gather 场景不可靠`的确切机制（硬件/DMA 语义）未追根因，只有经验性回避；小行宽场景的可靠批量搬运方案未探索。
- L1_007/L1_010 存量精度缺陷（输出最后一行全 0 / 大图边界 MARE 超阈）的根因定位在后续调试会话继续，本卡不覆盖。
## 坏实践

- 生产 kernel 用 `GlobalTensor::GetValue/SetValue` 做全量数据访问（API 最佳实践明确仅限调试）——早期版本即此形态，正确但 GM 访问次数失控。
- 行缓冲只在外层条件填充、读侧无条件读取（`if (yLow != yHigh)` 跳过 rowBufHigh 预取但插值仍读它）——会读到上一轮脏数据；正确做法是始终预取或读侧区分。

## 误诊

- L1_007/L1_010 失败后第一反应怀疑是行缓冲优化（DataCopyPad）引入的回归；禁用行缓冲后失败依旧，证明是存量缺陷被新增用例暴露——新增用例失败时先切换新优化开关复测，再归因。

## 复现增量：h0==h1 未初始化行缓冲的崩溃形态（errcode 271）与别名复用修复路线

另一独立轨迹并入 resize_bilinear 算子（arch35/DAV_3510，CANN 9.1.0，另一 trace）的故障形态实例——本卡上文"y_low == y_high 始终预取"设计点的崩溃面证据，并给出第二条修复路线（别名复用）：

**故障形态（高度非确定）**：`ProcessRow()` 在 `h0==h1`（整数坐标行：2 倍上采样 ac=false 的偶数输出行、`H_in=1`、align_corners 命中整数坐标或 clamp 到边界）时，第二输入行的 `DataCopyPad` 与 `Cast` 被正确跳过，但后续加权求和阶段仍**无条件**对 `rowH1Fp32` 执行 Gather；且对 `T==float`，`rowH1Fp32Buf` 在 `Init` 中**从未 InitBuffer**（仅 `!is_same<T,float>` 分支才分配）：

```cpp
// 修复前（检视定位的关键路径）
LocalTensor<float> rowH0Fp32 = rowH0Fp32Buf.Get<float>();
LocalTensor<float> rowH1Fp32 = rowH1Fp32Buf.Get<float>();   // float 时该 Buf 未 InitBuffer
if constexpr (std::is_same_v<T, float>) {
    rowH0Fp32 = rowH0RawBuf.Get<float>();
    if (h0 != h1) { rowH1Fp32 = rowH1RawBuf.Get<float>(); }  // h0==h1 时 rowH1Fp32 指向未分配 Buf
}
...
Gather(gatherDst, rowH1Fp32, offset0, 0u, wOutU);   // 无条件 Gather：读取未分配/未初始化内存
```

- `T==float`：`Get<float>()` 返回地址未定义的 LocalTensor，Gather 读非法/随机 UB 区域（潜在硬件异常）；
- `T!=float`：Buffer 已分配但内容未写，Gather 读垃圾值。

由于 `alphaH=0`，垃圾值乘 0 后不影响数值结果——这解释了"数值一直正确"与"偶发硬件异常"并存：是否越界取决于当时 UB 的残值/布局（典型未定义行为的非确定性表现）。观测：2x2→4x4 fp32 ACLNN 示例 `aclrtSynchronizeStream failed. ERROR: 507035`，plog `error code = 271 ... The address for scalar to access the internal buffer is out of bounds`；同一 4x4→8x8 shape 一度通过随后再失败；期间 ST 全量 23 用例（更大 shape，含 H_in=1 边界）始终通过；内核 .o MD5 前后一致。plog args 快照（tilingKey/blockDim/TilingData 参数逐项正确）可排除参数错误。

**修复路线二：h0==h1 时别名复用（与"始终预取"等价可选）**——`rowH1Fp32 = rowH0Fp32`（bilinear 在 h0==h1 时退化为单行，第二行数据逻辑上就是第一行），既满足"读侧永远拿到合法数据"，又不浪费 UB（无条件分配 fp32 路径会多占一块永不使用的 buffer）：

```cpp
    if constexpr (std::is_same_v<T, float>) {
        rowH0Fp32 = rowH0RawBuf.Get<float>();
        if (h0 != h1) {
            rowH1Fp32 = rowH1RawBuf.Get<float>();
        } else {
            rowH1Fp32 = rowH0Fp32;   // 别名复用：h0==h1 时第二行即第一行
        }
    } else {
        Cast(rowH0Fp32, rowH0Raw, RoundMode::CAST_NONE, wInU);
        if (h0 != h1) {
            LocalTensor<T> rowH1Raw = rowH1RawBuf.Get<T>();
            Cast(rowH1Fp32, rowH1Raw, RoundMode::CAST_NONE, wInU);
        } else {
            rowH1Fp32 = rowH0Fp32;
        }
    }
```

数值语义不变（`alphaH=0` 本就把第二行贡献归零）；若未来改动加权公式使 alphaH 与 h0==h1 判定脱钩，复用会引入错误数值——修改插值公式时必须重新验证该分支。备选（未验证）：把后两个 Gather 包进 `if (h0!=h1)` 条件执行（引入分支，改动最小性不如别名）。

**独立 trace 新增验证信号**（均为 tool_observed）：

- 故障观测：2x2→4x4 fp32 ACLNN 示例 `aclrtSynchronizeStream failed. ERROR: 507035`，plog `error code = 271 ... The address for scalar to access the internal buffer is out of bounds`（fault kernel 为预编译 AIC binary，tilingKey=0、blockDim=4、TilingData 参数逐项正确）（来源平台：950；证据等级：tool_observed）。
- 非确定性观测：4x4→8x8 示例一次 PASS 后同 shape 再失败；期间 ST 全量 23/23 始终通过；内核二进制 build 输出与 vendor 目录 MD5 一致（来源平台：950；证据等级：tool_observed）。
- 修复后回归：别名复用修复 + 重编译安装后，ST 全量 23/23 通过（含 1/4/16 元素边界用例）、UT 通过，无回归（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界与坏实践**：修复后未用触发过 271 的 2x2→4x4 小 shape 直接复验（示例当时已改大 shape）——"271 即本 UB"的因果链由检视代码路径 + 非确定性症状 + 机制推断支撑，缺修复后同 shape 复测的最后一环（优先补做）；`T!=float` 分支"已分配未写入"的垃圾值读取未观测到独立故障，风险上限未验证。坏实践补充：靠换大 shape 让示例"跑绿"当问题解决（触发面被绕开，UB 仍在）；见到非确定性 271 就反复重跑碰运气或归因"设备坏了"，不去读 plog 的 args/tilingKey 快照和检视条件消费路径。预防补充：双线性/近邻取样类 kernel 的 ST 用例集必须包含 `H_in=1`、整数坐标行等 h0==h1 场景的小 shape 用例（本例 ST 恰好全部用大 shape，才让该 UB 长期潜伏）。