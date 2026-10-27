---
type: "Runbook"
title: arch35 RegBase VF 逐元素算子完整实现模式
description: ascend950/arch35 RegBase VF 逐元素算子完整骨架：TPipe/TQue 三级流水 + asc_vf_call/__VEC_SCOPE__ 调 VF body，窄类型 LoadAlign(UNPACK_B16)+Cast 到 fp32 计算再 Cast 回，ASCENDC_TPL 模板参数 3 dtype × 双缓冲组合；从 FP32 直算扩展全 dtype 时 if constexpr 编译期分发 + tiling 按 dtype 选 TYPE_SIZE；UT/ST 与 PyTorch 张量级比对验证。
tags: [task.implementation, topic.compute.elementwise.activation]
status: stable
sources: []
created_at: '2026-09-16T10:22:16Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# arch35 RegBase VF 逐元素算子完整实现模式

## 现象

在 ascend950（arch35 / DAV_3510，RegBase 向量路线）上手写一个逐元素激活算子（sigmoid：`y = 1 / (1 + e^(-x))`，支持 fp32/fp16/bf16），需要一套可直接复用的工程骨架：TilingData/TilingKey 模板参数声明、kernel 类三级流水（CopyIn→Compute→CopyOut）、`__simd_vf__` VF body 寄存器链、host 侧 tiling 切分与模板参数下发。本卡给出该骨架的完整实现与设计依据（以 sigmoid 为载体，模式适用于同类逐元素算子）。

## 适用场景

- 工程形态：算子仓（registry-invoke）手写 arch35 RegBase kernel + op_host tiling + UT/ST。
- API 路径：`asc_vf_call<funcPtr>`（`basic_api/kernel_common.h`）、`__simd_vf__` VF body、`Reg::LoadAlign/StoreAlign/Muls/Exp/Adds/Duplicate/Div/Cast/UpdateMask`（`basic_api/reg_compute/kernel_reg_compute_*.h`）、`ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL/ASCENDC_TPL_SEL_PARAM`（`ascendc/host_api/tiling/template_argument.h`）。
- 验证过的平台与 CANN 版本：Ascend 950（ascend950，Ascend950PR：64 AIV、UB 248KB）+ CANN 9.1.0。
- 适用粒度：operator_family（arch35 RegBase 逐元素/一元数学算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：Cube/矩阵类、需要跨块归约的算子未验证；非 arch35 架构（Reg API 布局不同）；高层 VecUnary API 路线（另一套写法）。

## 系统信息

- 工具链：CANN 9.1.0（x86_64 主机安装，`x86_64-linux/asc/include`），g++ 13.3.0 / cmake 3.20.5，构建命令 `bash build.sh --soc=ascend950 -j8`。
- 目标平台：ascend950（arch35 / DAV_3510；Ascend950PR：coreNum=64 AIV，ubSize=248KB=253952B）。
- 目录布局：`operators/<op>/{op_kernel/arch35, op_host/arch35, op_api, op_graph, tests/ut, tests/st}`。
- 相关文件（本模式核心，均与事件流实际写入一一对应）：
  - `op_kernel/arch35/<op>_tiling_data.h`（TilingData 结构）
  - `op_kernel/arch35/<op>_tiling_key.h`（ASCENDC_TPL 模板参数声明）
  - `op_kernel/arch35/<op>.h`（VF body + kernel 类）
  - `op_kernel/<op>_arch35.cpp`（kernel 入口）
  - `op_host/arch35/<op>_tiling.cpp`（tiling 切分与模板参数下发）
  - 配套：`op_host/<op>_def.cpp`、`op_host/<op>_infershape.cpp`、`op_host/CMakeLists.txt`（x86_64 主机上的 include 路径坑见 [模板硬编码 aarch64 路径卡](template_aarch64_hardcode_x86_64_host.md)）、`op_api/aclnn_<op>.{h,cpp}`。

## 完整代码

**1) VF body：fp32 直算路径 + 窄类型 Cast 路径（`op_kernel/arch35/<op>.h`）：**

```cpp
namespace NsBenchNoHarnessSigmoid {

using namespace AscendC;

// B32 数据每 repeat 元素数: 256B / 4B = 64
constexpr uint32_t B32_DATA_NUM_PER_REPEAT = 64;

// ============================================================================
// VF Body: FP32 直接计算路径
// ============================================================================
// Sigmoid(x) = 1 / (1 + exp(-x))
// 寄存器链: LoadAlign → Muls(-1) → Exp → Adds(1) → Duplicate(1) → Div → StoreAlign
// 中间值全部在 RegTensor 中，不占用 UB
// ============================================================================
__simd_vf__ inline void SigmoidVfFp32(
    __ubuf__ float* inAddr, __ubuf__ float* outAddr, uint32_t count)
{
    using namespace AscendC::Reg;

    uint32_t sreg = count;
    RegTensor<float> xReg;
    RegTensor<float> negXReg;
    RegTensor<float> expReg;
    RegTensor<float> sumReg;
    RegTensor<float> oneReg;
    RegTensor<float> yReg;

    uint32_t vfLoopNum = (count + B32_DATA_NUM_PER_REPEAT - 1) / B32_DATA_NUM_PER_REPEAT;
    for (uint32_t i = 0; i < vfLoopNum; i++) {
        MaskReg mask = UpdateMask<float>(sreg);

        LoadAlign<float, LoadDist::DIST_NORM>(xReg, inAddr + i * B32_DATA_NUM_PER_REPEAT);
        Muls<float>(negXReg, xReg, -1.0f, mask);       // negX = -x
        Exp<float>(expReg, negXReg, mask);              // exp = e^(-x)
        Adds<float>(sumReg, expReg, 1.0f, mask);        // sum = 1 + e^(-x)
        Duplicate<float>(oneReg, 1.0f, mask);           // one = 1.0
        Div<float>(yReg, oneReg, sumReg, mask);         // y = 1 / (1 + e^(-x))
        StoreAlign<float, StoreDist::DIST_NORM>(outAddr + i * B32_DATA_NUM_PER_REPEAT, yReg, mask);
    }
}

// ============================================================================
// VF Body: 窄类型 Cast 计算路径 (fp16/bf16)
// ============================================================================
// 精度策略（尽早升精度，尽晚降精度）：
//   1. LoadAlign(UNPACK_B16): 加载 64 个 B16 元素并 unpack 到 64 个 B32 寄存器
//   2. Cast(B16→B32): half/bf16 → float
//   3. 在 fp32 下完成 Sigmoid 计算
//   4. Cast(B32→B16): float → half/bf16
//   5. StoreAlign(PACK_B32): pack 64 个 B32 寄存器到 64 个 B16 元素并 store
// ============================================================================
// CastTrait 定义（参考 SDK sigmoid_3510_impl.h 内部实现）：
//   castTraitB16ToB32: 升精度，RoundMode=UNKNOWN（无舍入）
//   castTraitB32ToB16: 降精度，RoundMode=CAST_RINT（最近偶舍入），SatMode=NO_SAT
// 注意：RoundMode 在 AscendC 命名空间下，不在 AscendC::Reg 下
constexpr AscendC::Reg::CastTrait castTraitB16ToB32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename NarrowT>
__simd_vf__ inline void SigmoidVfNarrow(
    __ubuf__ NarrowT* inAddr, __ubuf__ NarrowT* outAddr, uint32_t count)
{
    using namespace AscendC::Reg;

    uint32_t sreg = count;
    RegTensor<NarrowT> xNarrow;
    RegTensor<NarrowT> yNarrow;
    RegTensor<float> xReg;
    RegTensor<float> negXReg;
    RegTensor<float> expReg;
    RegTensor<float> sumReg;
    RegTensor<float> oneReg;
    RegTensor<float> yReg;

    uint32_t vfLoopNum = (count + B32_DATA_NUM_PER_REPEAT - 1) / B32_DATA_NUM_PER_REPEAT;
    for (uint32_t i = 0; i < vfLoopNum; i++) {
        MaskReg mask = UpdateMask<float>(sreg);

        LoadAlign<NarrowT, LoadDist::DIST_UNPACK_B16>(xNarrow, inAddr + i * B32_DATA_NUM_PER_REPEAT);
        Cast<float, NarrowT, castTraitB16ToB32>(xReg, xNarrow, mask);   // B16 → B32 (升精度)
        Muls<float>(negXReg, xReg, -1.0f, mask);                        // negX = -x
        Exp<float>(expReg, negXReg, mask);                              // exp = e^(-x)
        Adds<float>(sumReg, expReg, 1.0f, mask);                        // sum = 1 + e^(-x)
        Duplicate<float>(oneReg, 1.0f, mask);                           // one = 1.0
        Div<float>(yReg, oneReg, sumReg, mask);                         // y = 1 / (1 + e^(-x))
        Cast<NarrowT, float, castTraitB32ToB16>(yNarrow, yReg, mask);   // B32 → B16 (降精度)
        StoreAlign<NarrowT, StoreDist::DIST_PACK_B32>(outAddr + i * B32_DATA_NUM_PER_REPEAT, yNarrow, mask);
    }
}
```

**2) kernel 类：三级流水（同文件）：**

```cpp
template <typename T, int BUFFER_MODE>
class BenchNoHarnessSigmoid {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline BenchNoHarnessSigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const BenchNoHarnessSigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// --- Init ---
template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessSigmoid<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const BenchNoHarnessSigmoidTilingData* tilingData)
{
    // 计算当前核负责的数据范围
    uint64_t blockIdx = AscendC::GetBlockIdx();
    uint64_t coreStart = tilingData->blockFormer * blockIdx;

    // 防御性检查：如果当前核的起始位置已超出总元素数，直接返回
    if (coreStart >= tilingData->totalNum) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    uint64_t coreEnd = coreStart + tilingData->blockFormer;
    blockLength_ = static_cast<int64_t>(
        (coreEnd > tilingData->totalNum) ? (tilingData->totalNum - coreStart) : tilingData->blockFormer);
    ubLength_ = static_cast<int64_t>(tilingData->ubFormer);

    // 绑定 GM 缓冲区
    inputGMX.SetGlobalBuffer((__gm__ T*)x + coreStart, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + coreStart, blockLength_);

    // 初始化 UB 缓冲区（DoubleBuffer 时 BUFFER_NUM=2）
    pipe.InitBuffer(inQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// --- CopyIn: GM → UB ---
template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessSigmoid<T, BUFFER_MODE>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inQueueX.EnQue(xLocal);
}

// --- Compute: UB → VF body → UB ---
template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessSigmoid<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();

    // 获取 __ubuf__ 地址，传入 VF body
    __ubuf__ T* xAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    uint32_t count = static_cast<uint32_t>(currentNum);

    // 通过 asc_vf_call 调用 __simd_vf__ 函数
    if constexpr (std::is_same_v<T, float>) {
        asc_vf_call<SigmoidVfFp32>(xAddr, yAddr, count);
    } else {
        asc_vf_call<SigmoidVfNarrow<T>>(xAddr, yAddr, count);
    }

    outQueueY.template EnQue<T>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

// --- CopyOut: UB → GM ---
template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessSigmoid<T, BUFFER_MODE>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

// --- Process: 三级流水主循环 ---
template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessSigmoid<T, BUFFER_MODE>::Process()
{
    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1)
            ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsBenchNoHarnessSigmoid
```

**3) kernel 入口（`op_kernel/<op>_arch35.cpp`，模板参数与 tiling_key.h 声明一一对应）：**

```cpp
#include "arch35/bench_no_harness_sigmoid.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void bench_no_harness_sigmoid(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessSigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessSigmoidTilingData, tilingData, tiling);

    NsBenchNoHarnessSigmoid::BenchNoHarnessSigmoid<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
```

**4) TilingData 与 TilingKey 声明（`op_kernel/arch35/`）：**

```cpp
// <op>_tiling_data.h：标准 C++ struct（禁用废弃的 BEGIN_TILING_DATA_DEF 宏）
struct BenchNoHarnessSigmoidTilingData {
    uint64_t totalNum = 0;     // 总元素数量
    uint64_t blockFormer = 0;  // 每个核处理的元素数量（按 ubBlockSize 对齐）
    uint64_t ubFormer = 0;     // 每次 UB 循环处理的元素数量（按 256B 对齐）
};
```

```cpp
// <op>_tiling_key.h：ASCENDC_TPL 模板参数（禁用废弃的 TILING_KEY_IS 宏）
// 注意：bf16 的常量名是 C_DT_BF16（不是 C_DT_BFLOAT16）
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessSigmoid,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);
```

**5) host 侧 tiling 核心切分（`op_host/arch35/<op>_tiling.cpp`，节选数据切分与模板参数下发两段，注册样板略）：**

```cpp
    // 5、多核切分：按 ubBlockSize (32B / sizeof(T)) 对齐
    int64_t typeSize = GetTypeSize(dataType);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = static_cast<uint64_t>(totalIdx);
    tiling->blockFormer = static_cast<uint64_t>(CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize));
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, static_cast<int64_t>(tiling->blockFormer));

    // 6、UB 切分
    // 双缓冲时 bufferNum=4 (2 inQueue × 2 + 2 outQueue × 2 → 等效 4 个 tensor)
    // 单缓冲时 bufferNum=2 (1 inQueue + 1 outQueue)
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? 4 : 2;
    int64_t alignFactor = UB_ALIGN_BYTES / typeSize;  // 256B / sizeof(T)
    int64_t ubFormer = Ops::Base::FloorAlign(
        Ops::Base::FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferNum), alignFactor);
    tiling->ubFormer = static_cast<uint64_t>(ubFormer);

    context->SetBlockDim(usedCoreNum);

    // 7、设置 TilingKey
    // 参数顺序与 tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义一致
    // 参数会映射到 kernel 入口模板参数：bench_no_harness_sigmoid<D_T_X, BUFFER_MODE>
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
```

（配套常量：`MIN_SPLIT_THRESHOLD = 1024`（双缓冲阈值）、`UB_ALIGN_BYTES = 256`、`WS_SYS_SIZE = 0`；空 Tensor 早退路径 `SetBlockDim(1)` + `ASCENDC_TPL_SEL_PARAM(context, dtype, 0)`。）

### 增量（从 FP32 直算扩展全 dtype 的四件套，`__VEC_SCOPE__` 形态）

独立复现（ForeachAddcdivScalar，`y_i = x1_i + (x2_i/x3_i)*scalar`，TensorList 输入，外层经典 AscendC 框架 + `__VEC_SCOPE__` VF body，从仅 FP32 扩展 FP16/BF16 一次通过：真实 NPU ST 65/65、PyTorch NPU 237 次张量级比对全过）。四件套：VF body（if constexpr 分发）、TilingKey 六组合、tiling 按 dtype 选 TYPE_SIZE、六层 dtype 声明（OpDef/tiling 白名单/L0/L2/tiling_key/proto，枚举名用 `ge::DT_BF16`/`C_DT_BF16`）。

RoundMode 命名映射（设计文档 ↔ SDK 枚举）：B16→F32 升精度 `RoundMode::UNKNOWN`（升精度无舍入，对应设计语义 CAST_NONE）；F32→B16 降精度 `RoundMode::CAST_RINT`（舍入到最近偶数；设计文档写的 CAST_ROUND 名称在 Reg::RoundMode 枚举中对应 round-to-nearest-even 的实现是 CAST_RINT——以 SDK 参考 `load_store_utils.h`/`erf_3510_impl.h` 的 castTrait 为准）。VL 始终按 float 计算（256B/4B=64 元素/循环），因为计算在 FP32 寄存器内完成。

VF body 终版（`op_kernel/arch35/*_regbase.h`，verbatim）：

```cpp
namespace NsForeachAddcdivScalar {

using namespace AscendC;
using namespace AscendC::Reg;

// CastTrait 常量（与 SDK 参考实现 load_store_utils.h / erf_3510_impl.h 一致）
// B16→F32 升精度：RoundMode::UNKNOWN（无舍入）
constexpr CastTrait castTraitB162B32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// F32→B16 降精度：RoundMode::CAST_RINT（舍入到最近偶数）
constexpr CastTrait castTraitB322B16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void AddcdivScalarVf(
    __ubuf__ T* x1Addr, __ubuf__ T* x2Addr, __ubuf__ T* x3Addr,
    __ubuf__ T* yAddr, uint32_t count, float scalar)
{
    // 计算始终在 FP32 进行，VL 基于 float（256B / 4B = 64 元素/循环）
    using ComputeT = float;
    constexpr uint32_t VL = GetVecLen() / sizeof(ComputeT);
    uint16_t loopNum = static_cast<uint16_t>((count + VL - 1) / VL);

    __VEC_SCOPE__
    {
        MaskReg maskReg;

        if constexpr (IsSameType<T, float>::value) {
            // ===== FP32 直算路径 =====
            // LoadAlign → Div → Muls → Add → StoreAlign
            RegTensor<float> x1Reg, x2Reg, x3Reg;
            RegTensor<float> divReg, mulReg, yReg;

            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t remain = count - i * VL;
                uint32_t active = remain > VL ? VL : remain;
                maskReg = UpdateMask<float>(active);

                // 1. Load: UB → RegTensor
                LoadAlign<float>(x1Reg, x1Addr + i * VL);
                LoadAlign<float>(x2Reg, x2Addr + i * VL);
                LoadAlign<float>(x3Reg, x3Addr + i * VL);

                // 2. Div: div = x2 / x3
                Div<float>(divReg, x2Reg, x3Reg, maskReg);

                // 3. Muls: mul = div * scalar
                Muls<float>(mulReg, divReg, scalar, maskReg);

                // 4. Add: y = x1 + mul
                Add<float>(yReg, x1Reg, mulReg, maskReg);

                // 5. Store: RegTensor → UB
                StoreAlign<float>(yAddr + i * VL, yReg, maskReg);
            }
        } else {
            // ===== FP16/BF16 升精度路径 =====
            // LoadAlign(DIST_UNPACK_B16) → Cast(B16→F32) → Div → Muls → Add
            //   → Cast(F32→B16) → StoreAlign(DIST_PACK_B32)
            RegTensor<T> x1B16Reg, x2B16Reg, x3B16Reg;
            RegTensor<float> x1Reg, x2Reg, x3Reg;
            RegTensor<float> divReg, mulReg, resultReg;
            RegTensor<T> resultB16Reg;

            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t remain = count - i * VL;
                uint32_t active = remain > VL ? VL : remain;
                maskReg = UpdateMask<float>(active);

                // 1. Load B16 原始 dtype（DIST_UNPACK_B16: 2B→4B 通道展开）
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1B16Reg, x1Addr + i * VL);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x2B16Reg, x2Addr + i * VL);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x3B16Reg, x3Addr + i * VL);

                // 2. Cast 升精度 B16 → F32（RoundMode::UNKNOWN，升精度无舍入）
                Cast<float, T, castTraitB162B32>(x1Reg, x1B16Reg, maskReg);
                Cast<float, T, castTraitB162B32>(x2Reg, x2B16Reg, maskReg);
                Cast<float, T, castTraitB162B32>(x3Reg, x3B16Reg, maskReg);

                // 3. FP32 计算: y = x1 + (x2 / x3) * scalar
                Div<float>(divReg, x2Reg, x3Reg, maskReg);
                Muls<float>(mulReg, divReg, scalar, maskReg);
                Add<float>(resultReg, x1Reg, mulReg, maskReg);

                // 4. Cast 降精度 F32 → B16（RoundMode::CAST_RINT，舍入到最近偶数）
                Cast<T, float, castTraitB322B16>(resultB16Reg, resultReg, maskReg);

                // 5. Store B16 原始 dtype（DIST_PACK_B32: 4B→2B 通道打包）
                StoreAlign<T, StoreDist::DIST_PACK_B32>(yAddr + i * VL, resultB16Reg, maskReg);
            }
        }
    }
}

} // namespace NsForeachAddcdivScalar
```

tiling 按 dtype 选 TYPE_SIZE（`op_host/arch35/*_tiling.cpp`，ubFactor 计算处，verbatim）：

```cpp
    // 7. 双缓冲选择
    uint64_t useDoubleBuffer = (totalElements > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    // 8. UB 切分
    // TYPE_SIZE 根据 dtype 选择（FP16/BF16=2, FP32=4）
    int64_t typeSize = TYPE_SIZE_FP32;
    if (*dataType == ge::DT_FLOAT16) {
        typeSize = TYPE_SIZE_FP16;
    } else if (*dataType == ge::DT_BFLOAT16) {
        typeSize = TYPE_SIZE_BF16;
    }
    int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferNum), ubBlockSize);
```

TilingKey 六组合声明与本卡第 4 节同型（D_T_X 三 dtype × BUFFER_MODE 单/双缓冲 = 6 组合）；备选路线对照（轨迹内读取过）：CANN 内置 `foreach_add_scalar_regbase.h` 走 MemBase（LocalTensor Cast + TQue CopyIn/CopyOut），`foreach_addcmul_scalar_regbase.h` 走 RegBase Predicate + LoadOneTensorForDtypeT/StoreOneTensorForDtypeT 工具——已有 RegBase VF body 的算子选 RegBase 原生 Load/Store+Cast 路线保持架构一致。

## 设计决策

1. **路线选 RegBase (SIMD-Regbase VF) 而非 MemBase/高层 API**：①高层 `Sigmoid` API 只支持 half/float、**不支持 bfloat16**，而 RegBase 统一 "Cast→fp32→计算→Cast 回" 天然覆盖三种 dtype；②Bench 评测算子要最优性能，寄存器内计算 + OOO 双发减少 UB 访问带宽；③中间值全在 RegTensor、不占 UB，相比需要 sharedTmpBuffer 的 MemBase 高层 API UB 预算更充裕。
2. **VF body 直接对标 SDK 自带实现**：`asc/impl/adv_api/detail/activation/sigmoid/sigmoid_3510_impl.h` 的 `SigmoidImplVF`（`__simd_vf__`，寄存器链 LoadAlign→Muls→Exp→Adds→Duplicate→Div→StoreAlign，half 走 UNPACK_B16+Cast）是官方同构参考；CastTrait 常量取值照抄 SDK 内部 `castTraitB16ToB32/castTraitB32ToB16`，保证舍入行为与官方一致。
3. **窄类型精度策略**：尽早升精度（LoadAlign UNPACK_B16 + Cast 到 fp32）、尽晚降精度（计算完 Cast 回 + StoreAlign PACK_B32），全程 fp32 计算——这是 bf16（8 位尾数）也能拿到接近 fp32 精度的关键。
4. **UB 预算与切分**：`4 × ubFormer × sizeof(T) ≤ 248KB`（双缓冲 bufferNum=4：in/out 队列各 2 份）；`alignFactor = 256B / sizeof(T)` 保证 VF body LoadAlign 地址 32B 对齐；`blockFormer` 按 GetUbBlockSize 返回值对齐——注意该 API 返回的是 32 字节块大小，直接用作元素对齐因子。
5. **kernel 侧 dtype 分发用 `if constexpr (std::is_same_v<T, float>)` 选 fp32 直算或窄类型 Cast 路径**，VF body 通过 `asc_vf_call<FuncPtr>(args...)` 以函数指针模板参数调用（`__ubuf__` 地址由 `LocalTensor::GetPhyAddr()` 取得）。
6. **`UpdateMask<float>(sreg)` 沿用 SDK 写法**（sreg 按引用传入、由 API 内部递减），不做手动 active 计算。
7. **放弃的候选**：TILING_KEY_IS 宏（已废弃）与 BEGIN_TILING_DATA_DEF 宏（已废弃）——改用 ASCENDC_TPL 模板声明与标准 C++ struct；测试侧曾发现 Reg::Duplicate 白名单标注 VF=No 但 SDK 示例在 VF body 内使用，实现时以 SDK 头文件为准直接使用（未触发问题），备选方案 Muls(0)+Adds(1) 未启用。

## 验证信号

以下信号来自该算子从骨架搭建到联调验证的完整开发会话（Ascend950PR，CANN 9.1.0）。

- 编译：`bash build.sh --soc=ascend950 -j8` 完整通过，3 个 dtype 变体 kernel 二进制生成（`BenchNoHarnessSigmoid_*.o` × 3）与自定义算子包 `custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。
- UT：op_host UT 25/25 PASS（TilingData 字段 totalNum/blockFormer/ubFormer、TilingKey 六组合、空 Tensor、失败路径、多核切分、InferShape）（来源平台：950；证据等级：tool_observed）。
- ST：Mock 模式 20/20 PASS（CPU golden 自测 18/18 + L0 用例）；联调阶段 NPU Real 模式 20/20 PASS，最高 MARE 1.80e-07（远低于 FP32 阈值 2^-13≈1.22e-04）（来源平台：950；证据等级：tool_observed）。

增量信号（ForeachAddcdivScalar 全 dtype 扩展，Ascend950PR + CANN 9.1.0，一次开发会话的构建、UT 与真实 NPU ST/PyTorch 验收记录）：

- 构建：`--soc=ascend950` 成功，3 个 kernel .o（6 TilingKey 组合），`binary_info_config.json` 含 bf16(27)/fp16(1)/fp32(0) 模板（来源平台：950；证据等级：tool_observed）。
- UT：tiling 参数验证 21/21（含 FP16/BF16 各 dtype 的 TYPE_SIZE/ubFactor/TilingKey 断言），补充边界后 41/41（来源平台：950；证据等级：tool_observed）。
- 真实 NPU ST：65/65 = 100%（全 dtype × 多 shape × 边界 × 特殊值 × Tail 非对齐）（来源平台：950；证据等级：tool_observed）。
- PyTorch NPU 批量验收 237 次张量级比对全过，最差 MARE/MERE：float32 6.85e-04/4.85e-06（阈 1.22e-03/1.22e-04）、float16 9.63e-04/5.77e-09（阈 9.77e-03/9.77e-04）、bfloat16 0.00e+00/0.00e+00（阈 7.81e-02/7.81e-03）（来源平台：950；证据等级：tool_observed）。

## 性能影响

本轨迹未做 profiling 对比（无性能实测信号）；RegBase vs MemBase 的性能差异为设计推断，未验证。

## 预防

- 写 arch35 RegBase kernel 前先读 SDK 同类实现（`asc/impl/adv_api/detail/**`）对齐寄存器链与 CastTrait 取值。
- tiling_key.h 的 bf16 常量写 `C_DT_BF16`；模板参数顺序必须与 `ASCENDC_TPL_SEL_PARAM` 下发顺序一致。

## 失效边界

- API 语义（LoadAlign/StoreAlign/UpdateMask/CastTrait/asc_vf_call）以 CANN 9.1.0 `x86_64-linux/asc/include` 头文件为准；其他版本需复核。
- UB 容量算式按 DAV_3510 的 248KB 标定；其他 SoC 换 `ubSize` 重算。
- `UpdateMask` 引用递减语义未逐行验证（沿用 SDK 写法，UT/ST 结果间接验证正确性）。
- 双缓冲阈值 1024 是本算子的策略常量，非通用最优值。
- （增量）配方绑定 arch35 RegBase 路线（`AscendC::Reg` 命名空间 + `__VEC_SCOPE__` 或 `__simd_vf__`/`asc_vf_call` 两种调用形态均已实测）；MemBase 路线（LocalTensor Cast）接口不同。输入含 inf/nan 时按 IEEE 754 经 FP32 计算传播（ST 特殊值用例验证）；但极深复合链的 ULP 累积未量化，超 2 步运算的算子需重测精度。本配方不处理非连续 Tensor 输入。
- （增量）升精度路径相对 B16 直算的性能代价未测量（无性能实测信号）；需要性能数据时按同算子两条路线成对相对对比补测。`SatMode::NO_SAT`（降精度方向）与 `UNKNOWN`（升精度方向）的饱和语义差异未逐项展开，沿用 SDK 参考取值。
- 本卡验证的是"基础寄存器链 + 窄类型 Cast"的简单逐元素形态；同平台另有 [RegBase VF __simd_vf__ 路线致 bisheng 编译崩溃](regbase_vf_simd_vf_bisheng_exit139_banned.md) 的 banned 记录（SIGSEGV/Do not know how to scalarize 变体）——复杂构造（特殊表达式/嵌套循环）走该路线前先小样编译验证，勿把本卡外推为"任意 VF 代码都可编"。
- 其他平台未验证，推断可迁移。

**未决项**：

- `UpdateMask(sreg)` 的引用递减机制未读实现验证，建议下个迭代读头文件确认。
- Reg::Duplicate 的 VF 白名单标注矛盾（文档标 No、SDK 在用）建议向上游求证。
## 坏实践

- 用废弃的 `TILING_KEY_IS`/`BEGIN_TILING_DATA_DEF` 宏（模板编译会拒绝）。
- 窄类型直接在原生 dtype 上算 sigmoid（bf16 下精度崩）而不走 fp32 Cast 路径。
- CastTrait 四个字段全部写 `AscendC::Reg::` 前缀（RoundMode 编译错，见 [Reg CastTrait RoundMode 命名空间卡](casttrait_roundmode_is_in_ascendc.md)）。

## 增量（fp16/bf16 扩展轮：CastTrait 取值依据与尾块 mask 按迭代重算）

同一算子（BenchNoHarnessSigmoid）后续 fp16/bf16 扩展轮的两个机制级结论与全量验证，回答上文遗留问题 1。

**1) CastTrait 取值依据（`x86_64-linux/asc/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_vconv_impl.h` 头文件直读，不再"照抄 SDK"）**：

- B16→B32（升精度）走 `layoutMerge` 分派，**不消费 roundMode**，`RoundMode::UNKNOWN` 即可；
- B32→B16（降精度）必须 `CAST_RINT`：`float→bfloat16_t` 对 `CAST_NONE` 有 `static_assert` 编译期直接禁止；`float→half` 传 `CAST_NONE` 会被**静默转换**成 `CAST_RINT`（不报错但语义不显式）；设计文档若建议 `CAST_ROUND`（away from zero）不可用——其精度不如 `CAST_RINT`（最近偶舍入 RNE，IEEE 默认，与 numpy `astype`/golden 模型一致）；
- `SatMode::NO_SAT` 的依据是算子性质：sigmoid 输出 (0,1) 永不越界；输出可能超出 fp16/bf16 表示范围的算子（如 exp）须重新评估饱和模式。

**2) 尾块 mask 按迭代活跃数重算（替代上文 VF body 的 `sreg` 总量写法）**：上文 `uint32_t sreg = count;` + 循环内 `UpdateMask<float>(sreg)` 的写法依赖"sreg 被逐次消耗"的未验证语义；扩展轮将其重构为按迭代显式计算活跃数（尾块迭代活跃数 = `count - i*64` 与 64 的较小值），不再依赖 `UpdateMask` 的引用递减行为（与库内 [VF UpdateMask 手动递减移除致输出全零](../precision/updatemask_manual_decrement_required_despite_doc.md) 卡的口径一致——mask 语义不应悬于隐式递减）。fp32 路径此前未暴露问题不代表旧写法语义正确，窄类型路径同样适用新写法。重构后循环段（verbatim）：

```cpp
    uint32_t vfLoopNum = (count + B32_DATA_NUM_PER_REPEAT - 1) / B32_DATA_NUM_PER_REPEAT;
    for (uint32_t i = 0; i < vfLoopNum; i++) {
        uint32_t remain = count - i * B32_DATA_NUM_PER_REPEAT;
        uint32_t active = (remain > B32_DATA_NUM_PER_REPEAT) ? B32_DATA_NUM_PER_REPEAT : remain;
        MaskReg mask = UpdateMask<float>(active);

        LoadAlign<NarrowT, LoadDist::DIST_UNPACK_B16>(xNarrow, inAddr + i * B32_DATA_NUM_PER_REPEAT);
        Cast<float, NarrowT, castTraitB16ToB32>(xReg, xNarrow, mask);   // B16 → B32 (升精度)
        ... fp32 寄存器链计算 ...
        Cast<NarrowT, float, castTraitB32ToB16>(yNarrow, yReg, mask);   // B32 → B16 (降精度)
        StoreAlign<NarrowT, StoreDist::DIST_PACK_B32>(outAddr + i * B32_DATA_NUM_PER_REPEAT, yNarrow, mask);
    }
```

（fp32 直算路径 `SigmoidVfFp32` 同步改为按迭代 `active` 生成 mask。）

**验证信号（复现增量，fp16/bf16 扩展轮会话，clean rebuild + UT + ST Real 全链路）**：

- 构建：`bash build.sh --make_clean` + `--soc=ascend950 -j8` 成功，`binary/ascend950/bench_no_harness_sigmoid/` 下 3 个 dtype `.o` 各含单/双缓冲入口（6 TilingKey 全部编出）（来源平台：950；证据等级：tool_observed）。
- UT：扩展轮全量 52/52 通过（含 tilingKey 1/2/257/258/283 等 fp16/bf16 单双缓冲分支与 64M 大 shape、UB 248KB 上限用例）（来源平台：950；证据等级：tool_observed）。
- ST Real：fp16/bf16 全部用例 PASS（如 bf16 14000 元素 MERE=5.52e-08、fp16 7777 元素在测试侧两处口径修复后 mismatches=0），最终 75/75（fp16 阈 9.77e-04 / bf16 阈 7.81e-03 均未超）（来源平台：950；证据等级：tool_observed）。
- SDK 语义佐证：`dav_3510/kernel_reg_compute_vec_vconv_impl.h` 中 float→bfloat16 的 CAST_NONE `static_assert` 与 float→half 的 CAST_NONE→CAST_RINT 静默转换、widening 不消费 roundMode 的分派路径均为头文件直读（来源平台：950；证据等级：artifact_verified）。

**坏实践补充**：

- 降精度 Cast 传 `CAST_NONE`（float→bf16 直接编译期报错；float→half 被静默改写，语义不显式）或按直觉传 `CAST_ROUND`（away-from-zero，与参考实现 RNE 不一致）。
- mask 用总 count 一次生成、循环内不复算尾块活跃数。
- 中间计算留在窄类型（如 fp16 域逐步算 sigmoid）——精度模型与"fp32 内部计算 + 末尾一次舍入"的金标准不一致。

## 增量（softplus_v2_grad full-load 分支：稳定 sigmoid 两分支与 where 链）

同算子族另一独立 trace（softplus_v2_grad Branch-1 full-load，`dx = where(beta*x > threshold, dy, dy*sigmoid(beta*x))`，fp32/fp16 双 dtype）对本模式的三点扩展，全部经 ST 双跑逐位一致验证：

**1) 数值稳定 sigmoid 的两分支写法（VF 内寄存器链，verbatim 节选）**——两分支都先用 `Mins(·, 0)` 把 exp 参数压到 ≤0（exp 结果 ∈(0,1]），避免 exp 上溢；`z>=0` 用 `1/(1+exp(-z))`，`z<0` 用 `exp(z)/(1+exp(z))`，再按 `Compares<GE>` 的 mask 用 `Select` 合流：

```cpp
        // ---- stable sigmoid, z >= 0 branch: sig_pos = 1/(1+exp(Mins(-z,0))) ----
        AscendC::Reg::Muls(t1, z, -1.0f, mask);   // -z
        AscendC::Reg::Mins(t1, t1, 0.0f, mask);   // Mins(-z, 0) -> exp arg <= 0
        AscendC::Reg::Exp(t1, t1, mask);          // exp(-z) in (0,1]
        AscendC::Reg::Adds(t1, t1, 1.0f, mask);   // 1 + exp(-z)
        AscendC::Reg::Duplicate(sig, 1.0f, mask);
        AscendC::Reg::Div(sig, sig, t1, mask);    // sig_pos = 1/(1+exp(-z))

        // ---- stable sigmoid, z < 0 branch: sig_neg = exp(Mins(z,0))/(1+exp(Mins(z,0))) ----
        AscendC::Reg::Mins(t2, z, 0.0f, mask);    // Mins(z, 0) -> exp arg <= 0
        AscendC::Reg::Exp(t2, t2, mask);          // exp(z) in (0,1)
        AscendC::Reg::Adds(t3, t2, 1.0f, mask);   // 1 + exp(z)
        AscendC::Reg::Div(t2, t2, t3, mask);      // sig_neg = exp(z)/(1+exp(z))

        // sig = (z >= 0) ? sig_pos : sig_neg
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(mGE, z, 0.0f, mask);
        AscendC::Reg::Select(sig, sig, t2, mGE);  // dst=src0 (sig_pos) in-place
```

直接 `exp(z)` 不做 `Mins(·,0)` 稳定化——z 大正数时 exp 上溢得 Inf，sigmoid 变 NaN，属坏实践。

**2) where 语义进 VF 链**：`Compares<float, CMPMODE::GT>(mGT, z, threshold, mask)` 产 MaskReg，`Select(dstReg, src0, src1, mask)` 为 mask 真取 src0；阈值判据用严格 `>`（`z == threshold` 走光滑分支），比较在 FP32 域做（z 已 Inf 时也只比出 GT 分支，不产生 NaN）。

**3) full-load 单 tile 退化形态**：tilingKey=1 保证 blockFormer ≤ ubFormer，一个 block 的工作集一次装进 UB，Process 无内循环（尾块 blockTail 区分）；三队列（2×VECIN + 1×VECOUT）深度 1 即可，无 double-buffer 需求——与上文 sigmoid 骨架的 tile-loop 循环形态互补，深度选择与带宽代价见[TQue 深度 1 带宽减半卡](../performance/tque_depth1_bandwidth_double_buffer.md)。广播范式（禁 TQue、NDDMA 随路广播）的姊妹骨架另见[混合 dtype 广播算子多核 kernel 骨架卡](broadcast_mixed_dtype_multicore_skeleton.md)。

验证信号（复现增量，softplus_v2_grad Branch-1 实现落地会话）：

- 构建通过：`./build.sh` 退出 0，fp32/fp16 两个 dtype 变体的 kernel 二进制均生成（来源平台：950；证据等级：tool_observed）。
- 范式门禁：`__simd_vf__` 计数 2、`asc_vf_call` 计数 2（要求 ≥1）（来源平台：950；证据等级：tool_observed）。
- ST：两次完整运行均 `total=15 PASSED=15 FAILED=0 ERROR=0`、exit=0，且两次输出 `IDENTICAL bit-wise run output`；覆盖标量/对齐/空 tensor/fp16 对齐/非对齐 1000 元素/rank8/beta·threshold 变体/NaN·±Inf·零点·阈值边界·全零·fp16 溢出（self=60000 → dx=dy）特殊值（来源平台：950；证据等级：tool_observed）。
- 回归：公共 TilingUT 22/22、Branch-1 TilingUT 11/11，exit=0（来源平台：950；证据等级：tool_observed）。