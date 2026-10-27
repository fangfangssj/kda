---
type: "Runbook"
title: 单 kernel 多 TilingKey：dtype×策略模板整合配方
description: ASCENDC_TPL 双模板参数（D_T_X dtype + 策略 UINT 枚举）整合配方：DECL/SEL 声明全组合、tiling 按阈值选策略并 ASCENDC_TPL_SEL_PARAM 传双参、kernel if constexpr 编译期分块与按模式定容 UB、每 dtype 一个二进制挂多个 simplifiedKey；三个算子独立验证。复现增量：既有 3-dtype 模板算子追加 IS_STRIDED 运行时二值维度（迭代扩维而非新建），LIST 两取值共享同 dtype 二进制（.o 数=dtype 数、实例数=dtype×LIST 取值数），键值=基键+256，解禁维度时须同步迁移"预期失败"用例。复现增量：双 DATATYPE 参数（data dtype × index dtype 各自 INPUT 绑定，5×2=10 组合）与"可选输入组合走 TilingData 标志而非 TilingKey 维度"两个扩展形态。
tags: [task.implementation, topic.tiling, topic.compute.neighborhood.convolution, topic.programming_model]
status: stable
sources: []
created_at: '2026-09-16T11:15:37Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["950"]
---

# 单 kernel 多 TilingKey：dtype×策略模板整合配方

## 现象（任务）

单 kernel 自定义算子需要在同一个模板入口下整合"多种 dtype × 多种切分/执行策略"的 TilingKey 组合，典型场景：直接卷积 BenchNoHarnessConv2d（Ascend950PR / CANN 9.1.0 / arch35，`GetValue` 标量读 + float 累加）初版只有单 TilingKey（仅按 dtype 选择），大 M（HOut×WOut 超 UB）或大 K（C1×KH×KW×C0 超 L0 阈值）的 shape 无法切分，需要整合 3 dtype × 4 切分模式 = 12 TilingKey（SINGLE_PASS / K_SPLIT / M_SPLIT / M_K_SPLIT），且原全量 fp16 用例零回归、fp32 路径可用、UT 覆盖 4 模式分支。另一算子 cummin（BenchNoHarnessCummin，同为 ascend950/CANN 9.1.0）以同一机制整合 4 dtype × Full/Split 两策略共 8 条 kernel 路径。本卡给出两个实例共同验证的完整整合配方。

## 适用场景

- 工程形态：AscendC 自定义算子多 TilingKey 开发（单 kernel 模板多实例化，按 tiling 期决策选择实例）；策略维度可由 shape 阈值（M/K 规模）或输入属性推导。
- API 路径：`ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_DATATYPE_DECL/ASCENDC_TPL_UINT_DECL/ASCENDC_TPL_SEL/ASCENDC_TPL_ARGS_SEL`（op_kernel）+ `ASCENDC_TPL_SEL_PARAM`（op_host）+ `REGISTER_TILING_DEFAULT/GET_TILING_DATA_WITH_STRUCT`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.1.0（与 frontmatter `platforms` 一致；两个算子独立会话验证）。
- 适用粒度：operator_family——"按规模/属性选择切分或执行策略"的单输出算子（卷积类直接可用，其他算子族可套框架换阈值语义）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不同 dtype 需要完全不同 kernel 入口（非模板差异）的场景；bf16 计算路径（卷积实例中 bf16 仍为 bias 占位 stub，标量 float 转换后端不支持，待 Cast 向量算子）；Cube 路径（Load3D+MMAD+Fixpipe）未实现，本配方是 Vector/直接卷积路径；其他 SoC 的二进制合并粒度未验证。

## 解决配方（完整实现，verbatim 取自两个算子的整文件写入事件）

**1) `op_kernel/arch35/bench_no_harness_conv_2d_tiling_key.h`——声明 12 组合并实例化**：

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

// 声明 D_T_X (3 dtype) + TILING_MODE (4 mode) = 12 种组合
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessConv2d,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(TILING_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
);

// 为每种 dtype × TILING_MODE 组合生成独立的 TilingKey 实例化
ASCENDC_TPL_SEL(
    // TilingKey 0: FP16 + SINGLE_PASS
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, 0)
    ),
    // TilingKey 1: FP16 + K_SPLIT
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, 1)
    ),
    // TilingKey 2: FP16 + M_SPLIT
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, 2)
    ),
    // TilingKey 3: FP16 + M_K_SPLIT
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, 3)
    ),
    // TilingKey 4~7: FP32 + MODE 0/1/2/3 —— ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT) 同上
    // TilingKey 8~11: BF16 + MODE 0/1/2/3 —— ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16) 同上
    ...（按同模式列出其余 8 项，完整见源文件）
);
```

（注意：注释里的"TilingKey 0/1/2…"只是书写顺序，运行期实际 TilingKey 编码为 `dtype枚举值 | (mode序号<<8)`，勿按书写顺序写 UT 期望值。）

**2) `op_kernel/bench_no_harness_conv_2d_arch35.cpp`——kernel 入口带双模板参数**：

```cpp
#include "arch35/bench_no_harness_conv_2d.h"

template <typename D_T_X, int TILING_MODE>
__global__ __aicore__ void bench_no_harness_conv2d(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv2dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv2dTilingData, tilingData, tiling);

    NsConv2d::Conv2d<D_T_X, TILING_MODE> op;
    op.Init(x, filter, bias, y, workspace, &tilingData);
    op.Process();
}
```

**3) `op_host/arch35/bench_no_harness_conv_2d_tiling.cpp`——模式判定 + 双参 TPL_SEL_PARAM**（阈值与切分字段计算）：

```cpp
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

// TILING_MODE 阈值
constexpr int64_t K_TILE_MAX = 8192;     // K 维度单次上限（C1×KH×KW×C0）
constexpr int64_t M_TILE_MAX_FP16 = 65536;  // M 维度单次上限（fp16/bf16，128KB UB）
constexpr int64_t M_TILE_MAX_FP32 = 32768;  // M 维度单次上限（fp32，128KB UB）
```

```cpp
    // ODE 选择
    int64_t mTileMax = (dataType == ge::DT_FLOAT) ? M_TILE_MAX_FP32 : M_TILE_MAX_FP16;
    // 对齐 mTileMax 到 32 字节边界（DataCopyPad 对齐要求）
    int64_t alignSize = (dataType == ge::DT_FLOAT) ? 8 : 16;  // 32 / sizeof(dtype)
    mTileMax = (mTileMax / alignSize) * alignSize;

    bool needKSplit = (kTotal > K_TILE_MAX);
    bool needMSplit = (mTotal > mTileMax);

    uint32_t tilingMode = 0;  // SINGLE_PASS
    if (needKSplit) tilingMode |= 1;  // K_SPLIT bit
    if (needMSplit) tilingMode |= 2;  // M_SPLIT bit

    // 6. 计算 mTile/mLoop/kLoop/c1PerKLoop
    int64_t mTile, mLoop, kLoop, c1PerKLoop;

    if (needMSplit) {
        mTile = mTileMax;
        mLoop = CeilDiv(mTotal, mTile);
    } else {
        mTile = mTotal;
        mLoop = 1;
    }

    if (needKSplit) {
        // K_SPLIT: 按 C1 分组
        // 每次 K 循环处理 c1PerKLoop 组 C1
        // 确保 kSingle = c1PerKLoop * KH * KW * c0 ≤ K_TILE_MAX
        int64_t kPerC1 = KH * KW * c0;
        c1PerKLoop = CeilDiv(K_TILE_MAX, kPerC1);
        if (c1PerKLoop < 1) c1PerKLoop = 1;
        if (c1PerKLoop > c1) c1PerKLoop = c1;
        kLoop = CeilDiv(c1, c1PerKLoop);
    } else {
        c1PerKLoop = c1;
        kLoop = 1;
    }

    // 7. 多核切分：按 (N, C_out_tile) 二维任务网格
    int64_t cOutTileCount = CeilDiv(COut, nTile);
    int64_t totalTasks = N * cOutTileCount;
    int64_t usedCoreNum = (coreNum < totalTasks) ? coreNum : totalTasks;
    int64_t blockFactor = CeilDiv(totalTasks, usedCoreNum);
```

```cpp
    // 11. 设置 TilingKey（D_T_X + TILING_MODE）
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint32_t tilingModeVal = static_cast<uint32_t>(tilingMode);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, tilingModeVal);
```

**4) `op_kernel/arch35/bench_no_harness_conv_2d.h`——kernel 按 TILING_MODE 编译期分块**（Init 的 UB 分配与 Process 的 M_SPLIT 循环）：

```cpp
    // 分配 UB buffer：M_SPLIT 模式下使用 mTile，否则使用完整输出
    int64_t outElems;
    if constexpr (TILING_MODE == 2 || TILING_MODE == 3) {
        // M_SPLIT: UB 只需容纳一个 mTile
        outElems = mTile_;
    } else {
        // SINGLE_PASS / K_SPLIT: UB 容纳完整 HOut*WOut
        outElems = HOut_ * WOut_;
    }
    int64_t outBytes = outElems * sizeof(D_T_X);
    // 向上对齐到 32 字节
    int64_t outBytesAligned = (outBytes + 31) & ~31;
    pipe_.InitBuffer(outBuf_, outBytesAligned);
```

```cpp
            if constexpr (TILING_MODE == 0 || TILING_MODE == 1) {
                // SINGLE_PASS / K_SPLIT: 单次处理全部 M
                // K_SPLIT 在直接卷积中无需特殊处理（累加在寄存器中完成）
                for (int64_t ho = 0; ho < HOut_; ho++) {
                    for (int64_t wo = 0; wo < WOut_; wo++) {
                        float sum = ComputeSinglePixel(n, co, ho, wo, biasVal);
                        yLocal.SetValue(static_cast<int64_t>(ho * WOut_ + wo),
                                        static_cast<D_T_X>(sum));
                    }
                }
                // 一次性写入 GM
                WriteTileToGM(gmOffset, 0, mTotal);
            } else {
                // M_SPLIT / M_K_SPLIT: 分块处理 M
                // K_SPLIT 部分在直接卷积中无需特殊处理
                for (int64_t mIdx = 0; mIdx < mLoop_; mIdx++) {
                    int64_t mStart = mIdx * mTile_;
                    int64_t mEnd = (mStart + mTile_ > mTotal) ? mTotal : (mStart + mTile_);
                    int64_t actualMTile = mEnd - mStart;

                    for (int64_t m = mStart; m < mEnd; m++) {
                        int64_t ho = m / WOut_;
                        int64_t wo = m % WOut_;
                        float sum = ComputeSinglePixel(n, co, ho, wo, biasVal);
                        yLocal.SetValue(m - mStart, static_cast<D_T_X>(sum));
                    }

                    // 写入当前 tile 到 GM 对应位置
                    WriteTileToGM(gmOffset, mStart, actualMTile);
                }
            }
```

```cpp
template <typename D_T_X, int TILING_MODE>
__aicore__ inline void Conv2d<D_T_X, TILING_MODE>::WriteTileToGM(
    int64_t gmOffset, int64_t mStart, int64_t mCount)
{
    LocalTensor<D_T_X> yLocal = outBuf_.Get<D_T_X>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.blockLen = static_cast<int32_t>(mCount * sizeof(D_T_X));
    DataCopyPad(yGM_[gmOffset + mStart], yLocal, copyParams);
}
```

**5) 第二实例（cummin）：4 dtype × Full/Split 的同机制整合**——声明与 tiling 侧（verbatim）：

```cpp
// op_kernel/arch35/bench_no_harness_cummin_tiling_key.h
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BenchNoHarnessCummin,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(IS_SPLIT, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
```

```cpp
    // op_host/arch35/bench_no_harness_cummin_tiling.cpp（空 tensor 早退处；常规路径在算出
    // isSplit = (B > tileBMax) ? 1 : 0 后同样调用，dTypeX = static_cast<uint32_t>(dtype)）
    // 8. 空 tensor 检查
    if (totalElements == 0) {
        tiling->aSize = 0;
        tiling->cSize = C;
        tiling->bSize = B;
        tiling->origBSize = B;
        tiling->blockDim = 1;
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, 0);
        return ge::GRAPH_SUCCESS;
    }
```

Full/Split 不写两套函数：统一为 B 分片循环（FullColumn 时 bLoopCount=1），模板参数 IS_SPLIT 只影响 tiling 的切分参数，kernel 主体共用：

```cpp
template <typename T, int IS_SPLIT>
__aicore__ inline void BenchNoHarnessCummin<T, IS_SPLIT>::Process()
{
    // 空 tensor 或无数据可处理
    if (aEnd_ <= aStart_ || cSize_ == 0) {
        return;
    }

    // 循环遍历 B 分片 (FullColumn: bLoopCount=1, bStart=0, bLen=bSize_)
    for (uint32_t bLoop = 0; bLoop < bLoopCount_; bLoop++) {
        uint32_t bStart = bLoop * tileB_;
        uint32_t bLen = (bSize_ - bStart < tileB_) ? static_cast<uint32_t>(bSize_ - bStart) : tileB_;

        if constexpr (std::is_same_v<T, int32_t>) {
            ProcessInt32Segment(bStart, bLen);
        } else {
            ProcessFloatSegment(bStart, bLen);
        }
    }
}
```

## 设计决策

- 为什么切分模式做成 TilingKey 模板参数（编译期分块）而不是运行期 if：M_SPLIT 的 UB 缓冲大小依赖 mTile（MODE 2/3 只需 mTile、MODE 0/1 需要完整 HOut×WOut），`InitBuffer` 尺寸必须在编译期定；`if constexpr (TILING_MODE == ...)` 让每个实例拿自己的 UB 布局，运行期分支开销为零。
- 为什么 K_SPLIT 在直接卷积里"无需特殊处理"：本实现 K 维度（CIn×KH×KW）在寄存器 float 累加器里完成，没有 K 维度的片上 buffer 需要分块回写，kLoop/c1PerKLoop 字段只是 tiling 期决策记录（供 Cube 路径/后续优化用）；K_SPLIT 仍然值得独立 TilingKey，避免与 M_SPLIT 共用实例导致 UB 布局假设混乱。
- 阈值定标：M_TILE_MAX 按 UB 容量（fp16/bf16 65536 元素=128KB、fp32 32768=128KB，再向下对齐 32B）；K_TILE_MAX=8192 按 L0A/L0B 64KB 预留；原全量基线用例 mTotal≤50176<65536、kTotal≤4608<8192，天然全部落在 MODE=0，保证零回归。
- 为什么 ASCENDC_TPL_SEL_PARAM 传 `(dTypeX, tilingModeVal)` 两个原始值：编码由宏内部完成（dtype 枚举值 + UINT 值表序号），host 侧不手工拼 key——手工 `context->SetTilingKey` 叠加复合键会与运行时按 key 找二进制的机制冲突（361001 类错误）。
- dtype 用 `ASCENDC_TPL_INPUT(0)` 绑定输入 0：dtype 由框架从输入张量自动提取，tiling 只需传 `static_cast<uint32_t>(dtype)`，避免手写 dtype→key 映射表；策略开关用 8 位 UINT 枚举，可扩展（后续可加值 2）。
- 二进制粒度：构建按 dtype 产出 per-dtype 二进制（Full/Split 等策略特化合并在内），每个二进制挂多个 simplifiedKey；相对"每条路径一个二进制"减少包体与加载开销（两个实例均未做对照测量）。
- 设计文档写简化编码（如 dtypeIndex*2+IS_SPLIT）而实现用宏推导编码时，UT 必须按实现的真实编码断言（fp16→1/257、fp32→0/256、int32→3/259、bf16→27/283），文档差异在注释中说明——以可运行产物为准。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（arch35/DAV_3510），CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` 构建。
- 卷积实例产物：3 个 kernel 二进制（每 dtype 一个 .o，binInfo 的 simplifiedKey 各含 4 条 mode 记录，如 fp16 条目 `BenchNoHarnessConv2d/d=0,p=1/1,2/1,2/1,2/1,2`、`d=1,p=0/...` 等 4 键）。
- cummin 实例产物：vendor `bench_no_harness_cummin.json` binList 为 `bin 0: dtype=float32, keys=4 / bin 1: dtype=float16, keys=4 / bin 2: dtype=bfloat16, keys=4 / bin 3: dtype=int32, keys=4`——4 dtype 全覆盖、每二进制 4 simplifiedKey（含 Full/Split 组合）。
- 相关文件（两实例同构）：`op_kernel/arch35/<op>_tiling_key.h`（组合 DECL/SEL）、`op_kernel/<op>_arch35.cpp`（双模板入口）、`op_kernel/arch35/<op>.h`（按模式编译期分块的 kernel 类）、`op_host/arch35/<op>_tiling.cpp`（模式判定 + 双参 TPL_SEL_PARAM）、`op_kernel/arch35/<op>_tiling_data.h`（mTile/mLoop/kLoop/c1PerKLoop 等字段）、`tests/ut/op_host/test_<op>_tiling*.cpp`（TilingKey 分支断言）。

## 验证信号

段首来源说明：以下信号来自两个算子的独立开发会话（构建 + UT + ST 实跑），均为 tool_observed。

- 卷积实例构建：3 个 kernel 二进制生成（每 dtype 一个，各覆盖 4 mode；simplifiedKey 共 12 条记录）（来源平台：950；证据等级：tool_observed）。
- 卷积实例 UT：30/30 PASSED（7 InferShape 回归 + 2 TilingKey + 4 TilingMode + 17 参数化，12 组合 TilingKey 互异）（来源平台：950；证据等级：tool_observed）。
- 卷积实例 ST 回归：原全量 fp16 基线用例 43/43 无回归；新增 7 条 fp32 用例后全量 50/50 PASS（MERE/MARE 全部低于阈）（来源平台：950；证据等级：tool_observed）。
- 卷积实例 Mock 模式：扩展用例集 104/104 PASS（fp16/fp32/bf16 × 多 shape），基线用例 41/43 通过 + 2 SKIP（FLOPs>4G 跳过）（来源平台：950；证据等级：tool_observed）。
- cummin 实例构建：`bash build.sh --soc=ascend950 -j8` 成功，`BenchNoHarnessCummin_*.o/.json` 共 4 组 per-dtype 二进制，binList 4 dtype × 4 keys 全覆盖（来源平台：950；证据等级：tool_observed）。
- cummin 实例 UT：8 TilingKey 分支断言（fp16 1/257、fp32 0/256、int32 3/259、bf16 27/283）全量 29/29 通过（来源平台：host UT（ascend950 tiling 源）；证据等级：tool_observed）。
- cummin 实例 Mock ST：L0 28 通过/0 失败/2 跳过，L1 600 条 553 PASS/0 FAIL/47 SKIP（golden 通路验证，未跑 kernel）（来源平台：950 环境 host；证据等级：tool_observed）。

## 失效边界

- bf16 走 stub（填 bias），bf16 精度未实现——Mock bf16 全对是 CPU golden 路径，real 模式 bf16 正确性未验证。
- M_SPLIT 的 mTile 上界由 UB 容量推出，仅覆盖 mTotal ≤ 2×mTileMax 的用例实测（mLoop=2）；mLoop 很大的 shape 性能未测。
- K_SPLIT 字段（c1PerKLoop/kLoop）当前只记录决策、kernel 未按它分块（寄存器累加），Cube 路径接入后语义会变；`c1PerKLoop=CeilDiv(K_TILE_MAX, kPerC1)` 使 kTile 可超 K_TILE_MAX 一个 kPerC1 单位（8208>8192），设计注释与实现不一致，未修。
- 直接卷积（标量 GetValue）在大 shape 上性能远劣于 Cube 路径，本配方不解决性能。
- cummin 实例的 Real 模式逐用例输出在相邻任务分片，本卡不将其计为证据；Mock ST 只验证 golden 通路，不覆盖 kernel 执行。
- 仅在 ascend950 出包链实测；其他 SoC 的 per-dtype 二进制合并粒度与 simplifiedKey 数量未验证，机制推断可迁移。

**未决项**：

- bf16 计算路径（Cast 向量算子）留给后续迭代。
- Real 模式全量联调逐用例证据未在本卡覆盖（卷积实例 Real 基线 50/50 已含；cummin 实例待后续迭代记录）。
- 4 二进制 × 4 key 与"8 独立二进制"方案的包体/加载差异未测量；单/双缓冲两档 TilingKey 的性能差异未测量（无 profiling 数据）。
## 坏实践

- 在 tiling 函数里手工改写 TilingKey（`context->SetTilingKey` 复合键）绕缓存——361001 类错误。
- 把 mTileMax 直接用常数不做 32 字节向下对齐——DataCopyPad 非对齐 blockLen 尾块处理依赖对齐假设（本实现对齐后才分配）。
- 设计文档先行定义简化 TilingKey 编码而实现用宏推导编码，两者不一致且不核对——UT 期望值必须从实现的 ASCENDC_TPL 编码推导，否则全量误判。
- 为 Full/Split 写两套几乎相同的 kernel 函数体——统一 B 分片循环后模板参数只影响 tiling 参数，维护面减半。

## 误诊

无（本卡为正向实现配方；相关失败闭环见 ASCENDC_TPL dtype 声明、SPLIT_CORE 守卫等同会话其他卡）。

## 增量（既有模板算子追加运行时二值维度）

第三个独立实例（BenchNoHarnessDepthwiseConv2d 迭代三，Ascend950PR + CANN 9.2.0）验证"给**既有** 3-dtype 模板算子追加一个运行时二值维度"的扩维形态：迭代一/二已用 ASCENDC_TPL 实现 fp16/fp32/bf16 三 dtype（每 dtype 一个二进制，tiling 对 strided 参数硬拒绝），迭代三要求支持 strided/dilation 路径——模板维度从"D_T_X 一个参数"扩成"D_T_X × IS_STRIDED 两个参数"。

**扩维配方（与前两实例的差异点）**：

1. 声明侧在既有 `ASCENDC_TPL_ARGS_DECL` 尾部追加 8 位 UINT 维度，每个 dtype 的 `ARGS_SEL` 都带同一名单的 UINT LIST：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessDepthwiseConv2d,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(IS_STRIDED, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
// SEL 侧每个 dtype 的 ARGS_SEL 都带：ASCENDC_TPL_UINT_SEL(IS_STRIDED, ASCENDC_TPL_UI_LIST, 0, 1)
```

2. tiling 侧移除 strided 硬拒绝、按 stride/dilation 检测动态选值传 `ASCENDC_TPL_SEL_PARAM`：

```cpp
    bool isStrided = (SH != 1) || (SW != 1) || (DH != 1) || (DW != 1);
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint32_t isStridedVal = isStrided ? 1 : 0;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, isStridedVal);
```

3. kernel 侧类模板加 `template <int IS_STRIDED>` + `if constexpr (IS_STRIDED)` 切 strided/非 strided 寻址；IS_STRIDED 判定放 tiling 侧而非 kernel 运行时（编译期分发、无运行时分支开销）。

**键值与产物规律（UT 实测）**：`strided_key = non_strided_key + 256`——fp16 1→257、fp32 0→256、bf16 27→283；256 即 `ASCENDC_TPL_UINT_DECL(IS_STRIDED, 8, ...)` 的 8 位（bit 8）偏移，与 `FastEncodeTilingKeyDirect` 按 DECL 顺序编码的公式一致（见库内 tilingkey_dtype_enum_plus_mode_256 卡）。LIST 为每个取值实例化模板，但同一 dtype 的两个 IS_STRIDED 实例**合并进同一个 `.o`**——3 dtype 产出 3 个 `.o`（每个 binary_info_config.json 条目含 4 条 simplifiedKey，如 `d=0,p=0 / d=0,p=1 / d=1,p=0 / d=1,p=1`），覆盖 3×2=6 模板组合。"二进制数 = dtype 数、实例数 = dtype×LIST 取值数"。

**解禁维度的用例迁移**：迭代二 UT 中"strided 不支持"的预期失败用例（如 F004_stride_not_1/F005_dilation）在 tiling 解禁后语义翻转，必须同步移入成功用例集并另设替代负向用例——"迭代 X 只支持…"的硬拒绝解除时不迁移用例归属，UT 门禁会误报或漏报。

验证信号（复现增量，BenchNoHarnessDepthwiseConv2d 迭代会话）：

- 编译产物：`build/op_kernel/ascendc_kernels/binary/` 下 3 个 `.o`（fp32 16192B / bf16 21480B / fp16 18192B），每个对应 json 的 simplifiedKey 含 `d=0,p=0 / d=0,p=1 / d=1,p=0 / d=1,p=1` 4 条目，3×2=6 模板组合全覆盖（来源平台：950；证据等级：tool_observed）。
- UT 键值断言：fp16 strided tilingKey=257（1+256）、fp32 strided=256（0+256）、bf16 strided=283（27+256）；UT 全量 57/57 通过（来源平台：950；证据等级：tool_observed）。
- 上板分发：strided ST 用例在 kernel 侧 strided 分支缺陷修复后全部 PASS——IS_STRIDED=1 实例被正确选中执行（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 键值 "+256" 依赖 `ASCENDC_TPL_UINT_DECL(IS_STRIDED, 8, ...)` 的 8 位位宽与 DECL 参数顺序——位宽或顺序变化会改变键空间布局；LIST 取值多于 2 个（如 {0,1,2}）时的实例数/键值外推未验证。
- simplifiedKey 中 `d=/p=` 的确切编码语义未究明（仅确认覆盖关系）；按 simplifiedKey 精确排障时需自行核对该版本编译系统源码。

坏实践（复现增量）：

- 用"二进制文件数"验收模板维度扩展——dtype 数决定文件数、LIST 实例数决定模板数，两者不同；应按 binary_info_config.json 的组合覆盖验收。
- 在 tiling 里保留"迭代 X 只支持…"的硬拒绝而不随模板解禁同步删除——预期失败用例语义随之翻转，必须同步迁移用例归属。

## 增量（双 DATATYPE 参数与可选输入组合不进 TilingKey 两个扩展形态）

独立复现 R5（MoE FinalizeRouting，CANN 9.1.0，构建+UT 8/8 验证）与 R6（scatter 算子，CANN 9.1.0，10 组合全编译+ST 回归验证）扩展本配方的两个新形态。

**形态 1：双 DATATYPE 参数（data dtype × index dtype）**（scatter）：两个维度都是 dtype 且各自绑定输入——`ASCENDC_TPL_DATATYPE_DECL(D_T_DATA, ..., ASCENDC_TPL_INPUT(0))` + `ASCENDC_TPL_DATATYPE_DECL(D_T_IDX, C_DT_INT32, C_DT_INT64, ASCENDC_TPL_INPUT(1))`，5 data × 2 idx = 10 组合全 SEL（verbatim）：

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

// 声明模板参数：D_T_DATA 绑定输入 0，D_T_IDX 绑定输入 1
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessScatter,
    ASCENDC_TPL_DATATYPE_DECL(D_T_DATA, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16,
                              C_DT_INT32, C_DT_INT64, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_DATATYPE_DECL(D_T_IDX, C_DT_INT32, C_DT_INT64, ASCENDC_TPL_INPUT(1))
);

// 实例化全 10 个模板组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_DATA, C_DT_FLOAT16),
        ASCENDC_TPL_DATATYPE_SEL(D_T_IDX, C_DT_INT32)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_DATA, C_DT_FLOAT16),
        ASCENDC_TPL_DATATYPE_SEL(D_T_IDX, C_DT_INT64)
    )
    // ……（fp32/bf16/int32/int64 data × int32/int64 idx 同构，共 10 项）
);
```

kernel 入口为双模板参数 `template <typename D_T_DATA, typename D_T_IDX>`，host 侧 `ASCENDC_TPL_SEL_PARAM(context, dTypeData, dTypeIdx)` 传两个 ge 枚举原值；实际键值按位编码（如 bf16+int32 = 27 | (3<<8) 相关布局，断言写法见 tilingkey_dtype_enum_plus_mode_256 卡）。配套的 `_def.cpp` DataType/Format 列表必须同步按同一组合顺序展开为等长条目（对齐约束详见 opbuild_dtype_lists_must_be_equal_length 卡）；`op_api` L2/L0 与 `op_graph` proto 的 dtype 支持列表也同步扩为全 dtype。dtype 常量名以 `graph/c_types.h` 为准（C_DT_BF16=27，无 C_DT_BFLOAT16 别名——设计文档笔误会直接编译失败）。

**形态 2：可选输入组合走 TilingData 标志而非 TilingKey 维度**（MoE FinalizeRouting）：算子有 skip1/skip2/bias/scales/efsr 五个可选输入共 32 种组合，若进 TilingKey 则 SEL 组合爆炸；改为 kernel 内 `if (hasSkip1_)` 分支 + tiling 侧 `GetInputShape(i) != nullptr` 判存在性，TilingKey 只保留 dtype × mode 两个必要维度。配套 kernel 入口（双模板参数 + REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT，verbatim）：

```cpp
template <typename D_T_MAIN, int DROP_PAD_MODE>
__global__ __aicore__ void bench_no_harness_moe_finalize_routing(
    GM_ADDR expanded_permuted_rows,
    GM_ADDR expanded_src_to_dst_row,
    GM_ADDR skip1,
    GM_ADDR skip2,
    GM_ADDR bias,
    GM_ADDR scales,
    GM_ADDR expert_for_source_row,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessMoeFinalizeRoutingTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessMoeFinalizeRoutingTilingData, tilingData, tiling);

    NsBenchNoHarnessMoeFinalizeRouting::BenchNoHarnessMoeFinalizeRouting<D_T_MAIN, DROP_PAD_MODE> op;
    op.Init(expanded_permuted_rows, expanded_src_to_dst_row, skip1, skip2, bias, scales,
            expert_for_source_row, out, &tilingData);
    op.Process();
}
```

可选输入存在性经 TilingData 的 `uint8_t hasSkip1/hasSkip2/hasBias/hasScales/hasEfsr/isMixedPrecision` 标志传递（标准 C++ struct，弃用 BEGIN_TILING_DATA_DEF 宏）；tiling 侧 `ASCENDC_TPL_SEL_PARAM(context, dTypeMain, dropPadMode)` 的实参顺序逐一对齐 ARGS_DECL。机制锚点：`ASCENDC_TPL_INPUT(0)`（值 100000000+n）把 dtype 维度绑定到第 n 个输入的实际 dtype，运行时框架按输入 dtype 选择实例；kernel 侧类型映射 `DT_FLOAT16→half`、`DT_BF16→bfloat16_t`、`DT_FLOAT→float`；`template_argument.h` 有 host/kernel/pre 三态编译模式（kernel 侧包含时 DECL/SEL 宏展开为空，实例化由 SEL 在 host 侧驱动的编译命令提供）。

验证信号（复现增量）：

- MoE FinalizeRouting（形态 2）：全量构建（清空 build 后 configure + all binary package）成功，生成 kernel binary 与算子包、复跑 grep error/warning 为空、退出码 0；op_host UT（faker 框架）8/8 通过：TilingKey 正确设置为 1（首个 SEL 模板）、TilingData 全字段校验通过（来源平台：950；证据等级：tool_observed）。
- scatter（形态 1）：tiling_key.h 写入 10 组合声明/枚举 + 注册表展开后，opbuild 阶段连续生成多个 TilingKey kernel（多个 `<OpType>_<hash> ... Done`）；终态全 10 个 TilingKey 编译通过、Kernel 二进制生成（10 个 .o）；ST 回归迭代一 5/5、迭代二 54/54 无回归（来源平台：950；证据等级：tool_observed）。
- 机制锚点：`graph/c_types.h:17-18`（C_DT_FLOAT=0、C_DT_FLOAT16=1）与 `graph/types.h`（DT_* = ::C_DT_* 映射）、`template_argument.h` 的 host/kernel/pre 三态宏机制与 @@ 标记预处理均为头文件 grep/read 实测（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 形态 2 的骨架在"3 dtype × 4 mode"DECL 下只实例化了首个 SEL 模板（迭代一范围）；12 组合全 SEL + def.cpp 同步扩回 3 dtype 的完整形态由形态 1 的 10 组合证据与主卡 12 组合证据共同支撑，跨版本未验证。
- bf16 kernel 分支在 arch35 上另有后端 bf16 标量运算限制（位级转换方案见 bf16_scalar_to_fp32_bit_extension 卡），形态 1 的 bf16 组合编译依赖该前置修复。
- `ASCENDC_TPL_UINT_DECL` 非 dtype 类模板参数（常量 POD）未在本增量展开。

坏实践（复现增量）：

- 把可选输入组合编进 TilingKey 维度导致 SEL 组合爆炸（应走 TilingData 标志 + kernel 分支）。
- 照抄设计文档里的 dtype 常量名而不核对 c_types.h（C_DT_BFLOAT16 不存在）。
- kernel 入口的 GM_ADDR 参数顺序与 def.cpp 的 input/output 注册顺序不一致（必须逐一对齐，尾接 workspace、tiling）。