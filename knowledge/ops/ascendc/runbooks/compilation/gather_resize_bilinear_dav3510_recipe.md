---
type: "Runbook"
title: Gather 双线性插值 kernel — DAV_3510 编译级配方
description: resize_bilinear 类算子的 AscendC 完整实现配方：行级多核切分 + Gather 按字节偏移做 W 方向插值 + fp32 中间精度 + align_corners 折算 scale/bias 进 TilingData；Gather 文档支持矩阵未标注 DAV_3510，ascend950 编译实测通过（3 dtype 二进制生成）。验证等级为编译级，上板精度/性能未验证。
tags: [task.implementation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: "2026-10-10T02:49:37Z"
platforms: ["950"]
---

# Gather 双线性插值 kernel — DAV_3510 编译级配方

## 2026-10-10 静态复核：历史代码不能直接用于精度交付

本卡保留的完整代码仅是原三 dtype **编译**记录，未经上板精度验证。后续逐函数复核发现以下具体反例，因此撤回将该代码按字面复制为精度配方的建议；Gather 编译事实本身不变。

1. `tmpCalcBuf.Get<float>()` 重复取得的是同一地址。代码先将 floor(w) 的 scratch 改成 floor(w)+1，再计算 beta，令 w=0.25 时 beta=-0.75。按[既有 TBuf alias 修复](../precision/tbuf_repeated_get_alias_reorder.md)先算 beta/补权重，再覆盖 scratch，或保留独立 floor 缓冲。该机制已有卡，不新增重复结论。
2. `h0 == h1` 分支跳过第二行的 load/Cast，却在 Gather 时读取第二行。两条 raw/converted 行必须都已初始化，或把两条读取一致地绑定到同一已初始化行。零权重不能消除非法/未初始化读，NaN*0 仍是 NaN。
3. Host 示例的 `H_in/H_out` 比例适用于 size/recompute=True 合同。冻结 Torch 原 forward 若显式传 scalar scale_factor 且 recompute 未设，必须保留原标量倒数，不能为了匹配示例改 golden。H=1399、scale_factor=1.5、Hout=2098 时 1399/2098 与 2/3 不同。见[scale 合同边界](../precision/golden_interpolate_recompute_scale_factor.md)。align_corners=True 另按端点合同推导。

验证边界：以上为对本卡完整代码的静态反例与固定 Torch v2.10.0 `UpSample.h` 的合同核对，不是对这段历史代码的新编译或设备复测。独立 WHITE 候选在原37 FP32/13 FP16、原50输入/CPU grader下全部通过，zero50拒绝；完整 Q 两段为0.7217120792620955/0.7836363486488493，均未达1.1。该结果仅证明独立候选的固定50范围；没有 isolated ablation、冷消费者或修复带来加速的证明。

## 现象

要在 Ascend950PR（DAV_3510，CANN 9.1.0）上实现一个 4D NCHW 双线性插值算子（resize_bilinear 语义：fp16/fp32/bf16，`align_corners` 真/假两种坐标映射，`output_size` 与 `scale_factor` 互斥二选一）。设计阶段查 API 文档时发现关键风险：`Gather`（按偏移地址收集元素的核心 API）的产品支持矩阵只列了 Atlas A3/A2 训练推理系列（对应 DAV_2201），**没有标注 Atlas A5 / DAV_3510（Ascend950PR）**；同时产品映射表确认 Ascend950PR = Atlas A5 推理 = DAV_3510，与 A3 的 DAV_2201 不是同一代。按"文档没写就不敢用"走会直接卡死方案。最终决策按"DAV_3510 指令集向后兼容 DAV_2201 Vector API"的假设走 Gather 路线，用 `bash build.sh --soc=ascend950` 编译实测验证——3 个 dtype 的内核二进制全部生成、算子包打包成功，证明 Gather 在 DAV_3510 上至少编译级可用。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），手写 AscendC 内核 + tiling，目标 DAV_3510（Ascend950PR）。
- API 路径：`AscendC::Gather`（W 方向按字节偏移取元素）、`DataCopyPad`（行级 GM↔UB 非对齐搬运）、`Cast`（dtype 转换 + `RoundMode::CAST_FLOOR` 向量取整）、`CreateVecIndex`（设备侧等差序列）、`Muls/Adds/Maxs/Mins/Mul/Add/Sub`（坐标与加权计算）。
- 验证过的平台与 CANN 版本：Ascend950PR（`--soc=ascend950` 编译通过，3 dtype 二进制生成）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。**注意：本卡验证等级为编译级；上板精度/性能未在本卡覆盖范围内验证。**
- 适用粒度：operator_family（所有"按浮点坐标 gather + 加权"类插值/重采样算子：bilinear resize、grid sample 的双线性分支等）。
- 路线优先级：recommended_fastpath（DAV_3510 上此类 gather 密集算子）。
- 明确不适用/未验证场景：W_out 超过 Gather 单次 count 能力的超大分辨率（本实现未做 W 方向分块）；其他 NPU 架构（DAV_2201 上文档已支持但本轨迹未编译）；上板语义与性能。

## 解决配方

整体数据流（每输出行）：

```text
GM → DataCopyPad → UB(原 dtype 行 h0/h1) → Cast(fp32)
  → Gather(offset0/offset1, 字节偏移) × 4 次 + 加权求和(fp32)
  → Cast(原 dtype, CAST_RINT) → DataCopyPad → GM
```

多核切分：输出行总数 `totalRows = N*C*H_out` 均分到 AIV 核（`blockFactor = CeilDiv(totalRows, coreNum)`，每核处理连续 `blockFactor` 行，行内 W 方向整行向量化）。align_corners 双模式在 Host 侧 tiling 折算成 `scale/bias` 四个标量进 TilingData，kernel 统一用 `coord = idx * scale + bias`，不引入 TilingKey 分支。

关键机制链：

1. **W 方向坐标预计算（每核一次，`InitWDirection`）**：`CreateVecIndex(wFloat, 0.0f, wOutU)` 生成 `[0..W_out-1]`，经 `Muls(scaleW_)+Adds(biasW_)` 得源坐标 `w[j]`，`Maxs/Mins` clamp 到 `[0, W_in-1]`；`Cast(CAST_FLOOR)` 得 `w0[j]`，`w1[j]=min(w0+1, W_in-1)`，`beta[j]=w[j]-w0[j]`。
2. **Gather 偏移单位是字节**：`Muls(w0Int, w0Int, (int32_t)sizeof(float), wOutU)` 把元素下标乘以 4 变成字节偏移（fp32 中间数据），再按 `uint32_t` 读给 `Gather` 的 `srcOffset`；W_in ≤ 8192 时最大偏移 32768，远小于 INT32_MAX，int32/uint32 重解释安全。
3. **每输出行（`ProcessRow`）**：H 方向标量算 `h0/h1/alphaH`（clamp 非负后截断取整），`DataCopyPad` 搬输入第 h0/h1 行，Cast 到 fp32，4 次 Gather（(h0,w0)、(h0,w1)、(h1,w0)、(h1,w1)）+ 逐级加权 `y = Σ (α/1-α)(β/1-β)·x`，`Cast(CAST_RINT)` 回原 dtype，`DataCopyPad` 写回。

## 设计决策

- **路线选择（为什么不是 RegBase）**：DAV_3510 + vector 类算子的默认建议是 RegBase（寄存器组/DAG 计算图）路线，但双线性插值本质是"按浮点坐标 gather + 搬运"的非规则数据流，核心瓶颈在 GM↔UB 搬运而非 VF 寄存器计算；RegBase 的 DAG 模型不适配，选通用 SIMD/MemBase 路线（手写 AscendC + Gather/DataCopyPad）。
- **Gather 字节偏移的生成链**：设计阶段草案是 `float→Cast(CAST_FLOOR)→int32→Cast 回 float→Muls(sizeof)→再 Cast 回 int32` 五步；实现时确认 `Muls` 原生支持 int32_t 标量乘，简化为 `Cast(CAST_FLOOR) 得 int32 → Muls<int32_t>(×4)` 两步，再按 uint32_t 重解释。
- **align_corners 不进 TilingKey**：若做成 TilingKey 维度会使内核实例数翻倍；改为 Host 侧把两种公式（`align=true: (H_in-1)/(H_out-1), bias=0`；`align=false: H_in/H_out, bias=0.5*scale-0.5`）折算成 scale/bias 标量，kernel 零分支。
- **fp32 中间精度**：fp16/bf16 输入先 Cast 到 fp32 做插值再 Cast 回（CAST_RINT），满足 2^-10/2^-7 级精度标准；`T==float` 时用 `if constexpr` 完全跳过转换 buffer。
- **buffer 复用**：Init 阶段的 `wFloatBuf` 在 Process 阶段复用为 Gather 目标缓冲（wFloat 用完即弃）；UB 上 9 个 buffer 按 W_in/W_out 布局，极端 fp32 W_in=8192/W_out=4096 约 176KB < 248KB（DAV_3510 UB）。
- **单缓冲先行**：首轮 Tiling 固定 `BUFFER_MODE=0`，TilingKey 定义了双缓冲档位但未启用 → 只编出 3 个二进制（3 dtype × 1 buf）；双缓冲留给后续做性能对比。
- **被放弃的设计**：Host 侧 L0 API 预写 workspace 索引序列（无公开 API，改为设备侧 CreateVecIndex 生成）；W 方向 tileWOut 分块（当前分辨率整行放得下，留待超大 shape 需要时加）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 环境：Ascend950PR（DAV_3510）、CANN 9.1.0、x86_64 主机、`bash build.sh --soc=ascend950 -j8`。
- 文件清单（与实现一一对应）：
  - `op_kernel/arch35/bench_no_harness_resize_bilinear_tiling_data.h` — TilingData 结构（12 字段）
  - `op_kernel/arch35/bench_no_harness_resize_bilinear_tiling_key.h` — TilingKey 模板参数（D_T_X × BUFFER_MODE）
  - `op_kernel/arch35/bench_no_harness_resize_bilinear.h` — 内核类（本卡完整代码）
  - `op_kernel/bench_no_harness_resize_bilinear_arch35.cpp` — 内核入口
  - `op_host/arch35/bench_no_harness_resize_bilinear_tiling.cpp` — tiling 实现
  - `op_host/bench_no_harness_resize_bilinear_def.cpp` / `..._infershape.cpp`、`op_api/`（L0/L2）、`op_graph/`、`CMakeLists.txt`×4、`build.sh`
- API 事实（设计阶段文档核对）：Gather 的 `srcOffset` 为 `LocalTensor<uint32_t>`，单位字节，要求 src 元素位宽对齐、不越 UB；Gather 文档支持矩阵未列 DAV_3510（本卡以编译实测补证）。

## 原编译轨迹代码（历史留存；含上述已识别缺陷）

内核入口 `op_kernel/bench_no_harness_resize_bilinear_arch35.cpp`（终版完整）：

```cpp
#include "arch35/bench_no_harness_resize_bilinear.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void bench_no_harness_resize_bilinear(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessResizeBilinearTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessResizeBilinearTilingData, tilingData, tiling);
    NsBenchNoHarnessResizeBilinear::BenchNoHarnessResizeBilinear<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, workspace, &tilingData);
    op.Process();
}
```

TilingData `op_kernel/arch35/bench_no_harness_resize_bilinear_tiling_data.h`（终版完整）：

```cpp
#ifndef _BENCH_NO_HARNESS_RESIZE_BILINEAR_TILING_DATA_H_
#define _BENCH_NO_HARNESS_RESIZE_BILINEAR_TILING_DATA_H_

struct BenchNoHarnessResizeBilinearTilingData {
    int64_t n = 0;            // batch size N
    int64_t c = 0;            // channels C
    int64_t hIn = 0;          // input height H_in
    int64_t wIn = 0;          // input width W_in
    int64_t hOut = 0;         // output height H_out
    int64_t wOut = 0;         // output width W_out
    int64_t totalRows = 0;    // N * C * H_out, total output rows
    int64_t blockFactor = 0;  // rows per core
    float scaleH = 0.0f;      // H direction scale
    float scaleW = 0.0f;      // W direction scale
    float biasH = 0.0f;       // H direction bias
    float biasW = 0.0f;       // W direction bias
};
#endif
```

TilingKey `op_kernel/arch35/bench_no_harness_resize_bilinear_tiling_key.h`（终版完整）：

```cpp
#ifndef __BENCH_NO_HARNESS_RESIZE_BILINEAR_TILING_KEY_H__
#define __BENCH_NO_HARNESS_RESIZE_BILINEAR_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BenchNoHarnessResizeBilinear,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);

#endif
```

内核类 `op_kernel/arch35/bench_no_harness_resize_bilinear.h`（终版完整；由一次完整写入加两处小修（floor→截断、移除 `<cmath>`）组装恢复）：

```cpp
#ifndef BENCH_NO_HARNESS_RESIZE_BILINEAR_H
#define BENCH_NO_HARNESS_RESIZE_BILINEAR_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bench_no_harness_resize_bilinear_tiling_data.h"
#include "bench_no_harness_resize_bilinear_tiling_key.h"

#include <type_traits>

namespace NsBenchNoHarnessResizeBilinear {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class BenchNoHarnessResizeBilinear {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr int32_t FLOAT_SIZE = 4;  // sizeof(float)

public:
    __aicore__ inline BenchNoHarnessResizeBilinear(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const BenchNoHarnessResizeBilinearTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitWDirection();
    __aicore__ inline void ProcessRow(int64_t globalRowIdx);

private:
    TPipe pipe;

    // Init-phase buffers (W-direction, computed once)
    TBuf<TPosition::VECCALC> wFloatBuf;       // float[W_out] - W coordinates (reused as gatherBuf in Process)
    TBuf<TPosition::VECCALC> w0Buf;            // int32_t[W_out] -> uint32_t offset0
    TBuf<TPosition::VECCALC> w1Buf;            // int32_t[W_out] -> uint32_t offset1
    TBuf<TPosition::VECCALC> betaBuf;          // float[W_out] - beta weights
    TBuf<TPosition::VECCALC> oneMinusBetaBuf;  // float[W_out] - 1-beta weights
    TBuf<TPosition::VECCALC> tmpCalcBuf;       // float[W_out] - temporary (Init: w0Float/w1Float, Process: tmp)

    // Process-phase buffers (per row)
    TBuf<TPosition::VECIN> rowH0RawBuf;        // T[W_in] - raw input row h0
    TBuf<TPosition::VECCALC> rowH0Fp32Buf;     // float[W_in] - fp32 input row h0 (unused when T==float)
    TBuf<TPosition::VECIN> rowH1RawBuf;        // T[W_in] - raw input row h1
    TBuf<TPosition::VECCALC> rowH1Fp32Buf;     // float[W_in] - fp32 input row h1 (unused when T==float)
    TBuf<TPosition::VECCALC> yFp32Buf;         // float[W_out] - accumulated result
    TBuf<TPosition::VECOUT> yRawBuf;           // T[W_out] - output row (unused when T==float)

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    // Tiling parameters
    int64_t n_ = 0;
    int64_t c_ = 0;
    int64_t hIn_ = 0;
    int64_t wIn_ = 0;
    int64_t hOut_ = 0;
    int64_t wOut_ = 0;
    int64_t totalRows_ = 0;
    int64_t blockFactor_ = 0;
    float scaleH_ = 0.0f;
    float scaleW_ = 0.0f;
    float biasH_ = 0.0f;
    float biasW_ = 0.0f;

    // Computed in Init
    int64_t myRowStart_ = 0;
    int64_t myRowCount_ = 0;
};

// ============================================================================

template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessResizeBilinear<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const BenchNoHarnessResizeBilinearTilingData* tilingData)
{
    n_ = tilingData->n;
    c_ = tilingData->c;
    hIn_ = tilingData->hIn;
    wIn_ = tilingData->wIn;
    hOut_ = tilingData->hOut;
    wOut_ = tilingData->wOut;
    totalRows_ = tilingData->totalRows;
    blockFactor_ = tilingData->blockFactor;
    scaleH_ = tilingData->scaleH;
    scaleW_ = tilingData->scaleW;
    biasH_ = tilingData->biasH;
    biasW_ = tilingData->biasW;

    // Compute this core's row range
    int64_t blockIdx = GetBlockIdx();
    myRowStart_ = blockFactor_ * blockIdx;
    int64_t remainder = totalRows_ - myRowStart_;
    myRowCount_ = (remainder > blockFactor_) ? blockFactor_ : remainder;
    if (myRowCount_ < 0) {
        myRowCount_ = 0;
    }

    // Set up GlobalTensor
    inputGM.SetGlobalBuffer((__gm__ T*)x);
    outputGM.SetGlobalBuffer((__gm__ T*)y);
    (void)workspace;  // not used (index sequence generated on-device)

    // Allocate UB buffers
    // Init-phase: W-direction parameters
    uint32_t wOutU = static_cast<uint32_t>(wOut_);
    uint32_t wInU = static_cast<uint32_t>(wIn_);

    pipe.InitBuffer(wFloatBuf, wOutU * sizeof(float));
    pipe.InitBuffer(w0Buf, wOutU * sizeof(int32_t));
    pipe.InitBuffer(w1Buf, wOutU * sizeof(int32_t));
    pipe.InitBuffer(betaBuf, wOutU * sizeof(float));
    pipe.InitBuffer(oneMinusBetaBuf, wOutU * sizeof(float));
    pipe.InitBuffer(tmpCalcBuf, wOutU * sizeof(float));

    // Process-phase: input/output rows
    pipe.InitBuffer(rowH0RawBuf, wInU * sizeof(T));
    pipe.InitBuffer(rowH1RawBuf, wInU * sizeof(T));
    pipe.InitBuffer(yFp32Buf, wOutU * sizeof(float));

    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(rowH0Fp32Buf, wInU * sizeof(float));
        pipe.InitBuffer(rowH1Fp32Buf, wInU * sizeof(float));
        pipe.InitBuffer(yRawBuf, wOutU * sizeof(T));
    }

    // Compute W-direction coordinates (once per core)
    if (myRowCount_ > 0) {
        InitWDirection();
    }
}

// ============================================================================

template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessResizeBilinear<T, BUFFER_MODE>::InitWDirection()
{
    uint32_t wOutU = static_cast<uint32_t>(wOut_);
    float wInMinus1 = static_cast<float>(wIn_ - 1);

    // 1. Generate index sequence [0, 1, ..., W_out-1] as float
    LocalTensor<float> wFloat = wFloatBuf.Get<float>();
    CreateVecIndex(wFloat, 0.0f, wOutU);

    // 2. w[j] = j * scaleW + biasW
    Muls(wFloat, wFloat, scaleW_, wOutU);
    Adds(wFloat, wFloat, biasW_, wOutU);

    // 3. Clamp to [0, W_in-1]
    Maxs(wFloat, wFloat, 0.0f, wOutU);
    Mins(wFloat, wFloat, wInMinus1, wOutU);

    // 4. w0[j] = floor(w[j]) as int32_t
    LocalTensor<int32_t> w0Int = w0Buf.Get<int32_t>();
    Cast(w0Int, wFloat, RoundMode::CAST_FLOOR, wOutU);

    // 5. w0Float = (float)w0 (in tmpCalcBuf, temporary)
    LocalTensor<float> w0Float = tmpCalcBuf.Get<float>();
    Cast(w0Float, w0Int, RoundMode::CAST_NONE, wOutU);

    // 6. w1Float = w0 + 1, clamp to W_in-1 (reuse tmpCalcBuf)
    LocalTensor<float> w1Float = tmpCalcBuf.Get<float>();
    Adds(w1Float, w0Float, 1.0f, wOutU);
    Mins(w1Float, w1Float, wInMinus1, wOutU);

    // 7. w1[j] = (int32_t)w1Float
    LocalTensor<int32_t> w1Int = w1Buf.Get<int32_t>();
    Cast(w1Int, w1Float, RoundMode::CAST_FLOOR, wOutU);

    // 8. beta[j] = w[j] - w0[j]
    LocalTensor<float> beta = betaBuf.Get<float>();
    Sub(beta, wFloat, w0Float, wOutU);

    // 9. oneMinusBeta[j] = 1.0 - beta[j]
    LocalTensor<float> oneMinusBeta = oneMinusBetaBuf.Get<float>();
    Muls(oneMinusBeta, beta, -1.0f, wOutU);
    Adds(oneMinusBeta, oneMinusBeta, 1.0f, wOutU);

    // 10. offset0 = w0 * sizeof(float), offset1 = w1 * sizeof(float)
    //    (byte offsets for Gather from fp32 data)
    Muls(w0Int, w0Int, static_cast<int32_t>(FLOAT_SIZE), wOutU);
    Muls(w1Int, w1Int, static_cast<int32_t>(FLOAT_SIZE), wOutU);

    // w0Int and w1Int now contain byte offsets as int32_t
    // They will be reinterpreted as uint32_t for Gather in ProcessRow

    PipeBarrier<PIPE_V>();
}

// ============================================================================

template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessResizeBilinear<T, BUFFER_MODE>::ProcessRow(int64_t globalRowIdx)
{
    // 1. Compute H-direction coordinates (scalar)
    int64_t nc = globalRowIdx / hOut_;
    int64_t hOutIdx = globalRowIdx % hOut_;
    int64_t nIdx = nc / c_;
    int64_t cIdx = nc % c_;

    float h = static_cast<float>(hOutIdx) * scaleH_ + biasH_;
    h = (h < 0.0f) ? 0.0f : h;
    float hInMinus1 = static_cast<float>(hIn_ - 1);
    h = (h > hInMinus1) ? hInMinus1 : h;

    int32_t h0 = static_cast<int32_t>(h);  // h >= 0 after clamp, truncation == floor
    int32_t h1 = h0 + 1;
    if (h1 > hIn_ - 1) {
        h1 = static_cast<int32_t>(hIn_ - 1);
    }
    float alphaH = h - static_cast<float>(h0);
    float oneMinusAlphaH = 1.0f - alphaH;

    // 2. Compute GM offsets
    int64_t ncOffset = nIdx * c_ + cIdx;
    int64_t gmOffsetH0 = ncOffset * hIn_ * wIn_ + static_cast<int64_t>(h0) * wIn_;
    int64_t gmOffsetH1 = ncOffset * hIn_ * wIn_ + static_cast<int64_t>(h1) * wIn_;
    int64_t gmOffsetOut = ncOffset * hOut_ * wOut_ + hOutIdx * wOut_;

    uint32_t wInU = static_cast<uint32_t>(wIn_);
    uint32_t wOutU = static_cast<uint32_t>(wOut_);

    // 3. CopyIn: load input rows from GM
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = wInU * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    LocalTensor<T> rowH0Raw = rowH0RawBuf.Get<T>();
    DataCopyPad(rowH0Raw, inputGM[gmOffsetH0], copyParams, {false, 0, 0, 0});

    if (h0 != h1) {
        LocalTensor<T> rowH1Raw = rowH1RawBuf.Get<T>();
        DataCopyPad(rowH1Raw, inputGM[gmOffsetH1], copyParams, {false, 0, 0, 0});
    }

    PipeBarrier<PIPE_MTE2>();

    // 4. Cast to fp32 (if needed)
    LocalTensor<float> rowH0Fp32 = rowH0Fp32Buf.Get<float>();
    LocalTensor<float> rowH1Fp32 = rowH1Fp32Buf.Get<float>();

    if constexpr (std::is_same_v<T, float>) {
        // No cast needed: raw buffer IS fp32
        rowH0Fp32 = rowH0RawBuf.Get<float>();
        if (h0 != h1) {
            rowH1Fp32 = rowH1RawBuf.Get<float>();
        }
    } else {
        Cast(rowH0Fp32, rowH0Raw, RoundMode::CAST_NONE, wInU);
        if (h0 != h1) {
            LocalTensor<T> rowH1Raw = rowH1RawBuf.Get<T>();
            Cast(rowH1Fp32, rowH1Raw, RoundMode::CAST_NONE, wInU);
        }
    }

    PipeBarrier<PIPE_V>();

    // 5. Gather + weighted sum
    // Get offset tensors (reinterpreted as uint32_t for Gather)
    LocalTensor<uint32_t> offset0 = w0Buf.Get<uint32_t>();
    LocalTensor<uint32_t> offset1 = w1Buf.Get<uint32_t>();
    LocalTensor<float> beta = betaBuf.Get<float>();
    LocalTensor<float> oneMinusBeta = oneMinusBetaBuf.Get<float>();

    // gatherBuf reuses wFloatBuf (wFloat no longer needed after Init)
    LocalTensor<float> gatherDst = wFloatBuf.Get<float>();
    LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
    LocalTensor<float> tmp = tmpCalcBuf.Get<float>();

    // x00 = rowH0[offset0]
    // y = (1-alphaH) * (1-beta) * x00
    Gather(gatherDst, rowH0Fp32, offset0, static_cast<uint32_t>(0), wOutU);
    Muls(yFp32, gatherDst, oneMinusAlphaH, wOutU);
    Mul(yFp32, yFp32, oneMinusBeta, wOutU);

    // x01 = rowH0[offset1]
    // y += (1-alphaH) * beta * x01
    Gather(gatherDst, rowH0Fp32, offset1, static_cast<uint32_t>(0), wOutU);
    Muls(tmp, gatherDst, oneMinusAlphaH, wOutU);
    Mul(tmp, tmp, beta, wOutU);
    Add(yFp32, yFp32, tmp, wOutU);

    // x10 = rowH1[offset0]
    // y += alphaH * (1-beta) * x10
    Gather(gatherDst, rowH1Fp32, offset0, static_cast<uint32_t>(0), wOutU);
    Muls(tmp, gatherDst, alphaH, wOutU);
    Mul(tmp, tmp, oneMinusBeta, wOutU);
    Add(yFp32, yFp32, tmp, wOutU);

    // x11 = rowH1[offset1]
    // y += alphaH * beta * x11
    Gather(gatherDst, rowH1Fp32, offset1, static_cast<uint32_t>(0), wOutU);
    Muls(tmp, gatherDst, alphaH, wOutU);
    Mul(tmp, tmp, beta, wOutU);
    Add(yFp32, yFp32, tmp, wOutU);

    PipeBarrier<PIPE_V>();

    // 6. Cast to original dtype (if needed)
    DataCopyParams outCopyParams;
    outCopyParams.blockCount = 1;
    outCopyParams.blockLen = wOutU * sizeof(T);
    outCopyParams.srcStride = 0;
    outCopyParams.dstStride = 0;

    if constexpr (std::is_same_v<T, float>) {
        // No cast needed: yFp32 IS the output
        DataCopyPad(outputGM[gmOffsetOut], yFp32, outCopyParams);
    } else {
        LocalTensor<T> yRaw = yRawBuf.Get<T>();
        Cast(yRaw, yFp32, RoundMode::CAST_RINT, wOutU);
        PipeBarrier<PIPE_V>();
        DataCopyPad(outputGM[gmOffsetOut], yRaw, outCopyParams);
    }

    PipeBarrier<PIPE_MTE3>();
}

// ============================================================================

template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessResizeBilinear<T, BUFFER_MODE>::Process()
{
    for (int64_t i = 0; i < myRowCount_; i++) {
        ProcessRow(myRowStart_ + i);
    }
}

} // namespace NsBenchNoHarnessResizeBilinear
#endif // BENCH_NO_HARNESS_RESIZE_BILINEAR_H
```

Tiling 实现 `op_host/arch35/bench_no_harness_resize_bilinear_tiling.cpp` 的核心函数（终版完整）：

```cpp
// Get shape and attributes, compute output H_out/W_out and scale/bias
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context,
                                         int64_t* n, int64_t* c, int64_t* hIn, int64_t* wIn,
                                         int64_t* hOut, int64_t* wOut,
                                         float* scaleH, float* biasH, float* scaleW, float* biasW,
                                         ge::DataType* dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = inputX->GetStorageShape();
    OP_CHECK_IF(inputShape.GetDimNum() != 4,
                OP_LOGE(context, "input must be 4D, got %zu dims", inputShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    *n = inputShape.GetDim(0);
    *c = inputShape.GetDim(1);
    *hIn = inputShape.GetDim(2);
    *wIn = inputShape.GetDim(3);

    // Get attributes
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto outputSizeAttr = attrs->GetListInt(0);
    auto alignCornersPtr = attrs->GetBool(1);
    auto scaleFactorAttr = attrs->GetListFloat(2);

    bool alignCorners = (alignCornersPtr != nullptr) ? *alignCornersPtr : false;
    bool hasOutputSize = (outputSizeAttr != nullptr && outputSizeAttr->GetSize() > 0);
    bool hasScaleFactor = (scaleFactorAttr != nullptr && scaleFactorAttr->GetSize() > 0);

    if (hasOutputSize && !hasScaleFactor) {
        OP_CHECK_IF(outputSizeAttr->GetSize() != 2,
                    OP_LOGE(context, "output_size must have 2 elements"), return ge::GRAPH_FAILED);
        *hOut = outputSizeAttr->GetData()[0];
        *wOut = outputSizeAttr->GetData()[1];
    } else if (hasScaleFactor && !hasOutputSize) {
        OP_CHECK_IF(scaleFactorAttr->GetSize() != 2,
                    OP_LOGE(context, "scale_factor must have 2 elements"), return ge::GRAPH_FAILED);
        float sh = scaleFactorAttr->GetData()[0];
        float sw = scaleFactorAttr->GetData()[1];
        *hOut = static_cast<int64_t>(floor(static_cast<float>(*hIn) * sh));
        *wOut = static_cast<int64_t>(floor(static_cast<float>(*wIn) * sw));
    } else {
        OP_LOGE(context, "exactly one of output_size or scale_factor must be provided");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(*hOut <= 0 || *wOut <= 0,
                OP_LOGE(context, "output H/W must be positive: hOut=%ld, wOut=%ld", *hOut, *wOut),
                return ge::GRAPH_FAILED);

    // Compute coordinate mapping parameters
    if (alignCorners) {
        *scaleH = (*hOut > 1) ? static_cast<float>(*hIn - 1) / static_cast<float>(*hOut - 1) : 0.0f;
        *biasH = 0.0f;
        *scaleW = (*wOut > 1) ? static_cast<float>(*wIn - 1) / static_cast<float>(*wOut - 1) : 0.0f;
        *biasW = 0.0f;
    } else {
        *scaleH = static_cast<float>(*hIn) / static_cast<float>(*hOut);
        *biasH = 0.5f * (*scaleH) - 0.5f;
        *scaleW = static_cast<float>(*wIn) / static_cast<float>(*wOut);
        *biasW = 0.5f * (*scaleW) - 0.5f;
    }

    // Get dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0,
                OP_LOGE(context, "unsupported dtype: %d", static_cast<int>(*dataType)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BenchNoHarnessResizeBilinearTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape, attrs, compute output size and scale/bias
    int64_t n, c, hIn, wIn, hOut, wOut;
    float scaleH, biasH, scaleW, biasW;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &n, &c, &hIn, &wIn, &hOut, &wOut,
                                  &scaleH, &biasH, &scaleW, &biasW, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Set workspace size (system workspace only; index sequence generated on-device via CreateVecIndex)
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = Ops::Base::GetWorkspaceSize(context);

    // 4. Fill TilingData
    auto tiling = context->GetTilingData<BenchNoHarnessResizeBilinearTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BenchNoHarnessResizeBilinearTilingData), 0,
                         sizeof(BenchNoHarnessResizeBilinearTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling data failed"), return ge::GRAPH_FAILED);

    int64_t totalRows = n * c * hOut;
    int64_t blockFactor = CeilDiv(totalRows, coreNum);
    if (blockFactor < 1) {
        blockFactor = 1;
    }

    tiling->n = n;
    tiling->c = c;
    tiling->hIn = hIn;
    tiling->wIn = wIn;
    tiling->hOut = hOut;
    tiling->wOut = wOut;
    tiling->totalRows = totalRows;
    tiling->blockFactor = blockFactor;
    tiling->scaleH = scaleH;
    tiling->scaleW = scaleW;
    tiling->biasH = biasH;
    tiling->biasW = biasW;

    // 5. Set BlockDim
    int64_t usedCoreNum = CeilDiv(totalRows, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    context->SetBlockDim(usedCoreNum);

    // 6. Set TilingKey: D_T_X(dtype) + BUFFER_MODE
    // Iter 1: always single buffer (BUFFER_MODE=0)
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint32_t bufferMode = 0;  // single buffer for iter 1
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, bufferMode);

    return ge::GRAPH_SUCCESS;
}
```

## 性能影响

本卡验证等级为编译级，无 profiling 数据；双缓冲（BUFFER_MODE=1）档位已预留在 TilingKey 中但未启用，相对性能结论待上板 profiling 后补充（未验证）。

## 预防

- 在 DAV_3510 上选 API 前，先按"产品映射表 → SocVersion/NpuArch"确认目标芯片代际（Ascend950PR=DAV_3510≠A3 的 DAV_2201），文档未标注的 API 用最小内核编译实验快速验证，不要仅凭文档矩阵否决方案。
- Gather 的偏移是字节单位：元素下标必须乘 `sizeof(元素类型)`，且按 uint32_t 提供给 `srcOffset`。

## 验证信号

本卡的验证证据来自该双线性插值算子骨架实现阶段的实际构建记录（Ascend950PR 目标、CANN 9.1.0 工具链）；上板精度与性能不在本卡验证范围内（见失效边界与遗留问题）。逐条实测信号：

- 含 Gather 的内核 `bash build.sh --soc=ascend950 -j8` 编译通过：`BenchNoHarnessResizeBilinear_f78909abc3449787cb1387e1ce835dbb`、`..._95173887ababec6af13ee2af2358c51e`、`..._d56edd83c03a210ea2bb60c2ee7713b7` 共 3 个二进制（fp16/fp32/bf16 × 单缓冲）生成（来源平台：950；证据等级：tool_observed）。
- 算子包 `custom_opp_ubuntu_x86_64.run` 打包成功，包内 `op_impl/ai_core/tbe/kernel/ascend950/` 下 3 组 `.o/.json`、`aic-ascend950-ops-info.json` 齐全（来源平台：950；证据等级：tool_observed）。
- 构建后核验：内核源中 Gather 调用 4 处、CreateVecIndex 2 处、多核切分相关代码 8 处均在产物配置中反映（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **Gather 在 DAV_3510 上的证据等级是编译级**：文档未标注该平台，上板运行语义/数值正确性未在本卡验证范围内证实；推断可迁移到其他未标注的 DAV_3510 场景，未实测。
- W 方向未分块：W_out 超过 Gather count 模式单次能力（约 255 repeat × 每 repeat 64/128 元素，随 dtype 变化）或 UB 放不下整行时，本实现失效，需引入 tileWOut 分块（TilingData 已预留思路，未实现）。
- UB 容量按 9 buffer 布局估算（DAV_3510 248KB），极端 fp32 W_in=8192/W_out=4096 约 176KB；更大分辨率未验证。
- bf16 的向量 `Floor` API 不支持（文档结论），本实现 bf16 路径全程在 fp32 域取整，规避了该限制；若直接对 bf16 坐标取整不适用。
- 本实现同步使用单侧 PipeBarrier（PIPE_MTE2/PIPE_V），在裸 TBuf 跨流水线依赖下不充分——该场景的正确同步形态见 [裸 TBuf 跨 pipe 依赖需 PipeBarrier<PIPE_ALL> 卡](../precision/tbuf_cross_pipe_dependency_requires_pipe_all.md)（同一算子后续 ST 联调实测确认）。

**未决项**：

- 优先项：上板运行验证 Gather 数值语义与精度（对齐 CPU golden，含 align_corners 两模式、output_size/scale_factor 两模式、3 dtype）。
- 双缓冲模式实现与单/双缓冲相对性能对比（profiling）。
- W 方向分块（超大分辨率）与 tileWOut tiling 字段落地。
## 坏实践

- 因为文档支持矩阵没列目标芯片就直接放弃可用 API——先用最小实验验证，再下结论。
- 把 align_corners 做成 TilingKey 维度导致内核实例数按模式翻倍——Host 侧折算 scale/bias 更省。

## 误诊

- 曾担心 `Gather` 文档"Atlas A3"即 DAV_3510：产品映射表证实 A3 = Ascend910_93 = DAV_2201，与 Ascend950PR（DAV_3510）不同代；按 A3 支持推断 3510 支持属于跨代假设，必须实验验证（本卡即验证记录）。
