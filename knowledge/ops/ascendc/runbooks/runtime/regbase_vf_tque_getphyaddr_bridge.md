---
type: "Runbook"
title: arch35 RegBase VF：TQue 经 GetPhyAddr 桥接
description: Ascend950PR arch35 RegBase VF 完整骨架配方：外层经典 AscendC（TPipe/TQue/DataCopyPad 搬运）通过 LocalTensor.GetPhyAddr() 取 __ubuf__ 原始地址传入 __VEC_SCOPE__ VF body，寄存器内完成 Div→Muls→Add 融合链；单 dtype 单变体先行，编译级验证。
tags: [task.implementation, topic.implementation, topic.architecture]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# arch35 RegBase VF：TQue 经 GetPhyAddr 桥接

## 现象（任务场景）

在 Ascend950PR（arch35 / DAV_3510，CANN 9.1.0）按 registry-invoke 工程布局开发 foreach 类算子 BenchNoHarnessForeachAddcdivScalar（`y_i = x1_i + (x2_i / x3_i) * scalar`，TensorList 输入输出）的骨架版本：目标是单 TilingKey、单 dtype（FP32）、单张量（listLen=1）、单缓冲路径先把"host 声明 → tiling → kernel → VF 寄存器计算"全链路编译通过。技术路线选 RegBase：融合计算链（除法/乘标量/加法）在 `RegTensor` 寄存器内完成，FP32 中间值不占 UB——这要求解决一个经典 AscendC 工程里没有的桥接问题：外层搬运用 `TPipe`/`TQue`/`LocalTensor`（拿到的 `LocalTensor` 不是 `__ubuf__` 指针），而 VF body 的 `LoadAlign/StoreAlign` 只接受 `__ubuf__ T*`。骨架经九轮编译修复后构建成功：`--soc=ascend950` 生成 kernel 二进制（10024 字节 .o）与 custom_opp 算子包（446127 字节 .run）。

## 适用场景

- 工程形态：手写 AscendC 自定义算子（registry-invoke：op_host/op_kernel/op_api/op_graph 分层），elementwise/foreach 类算子的 RegBase VF 实现骨架。
- API 路径：外层 `TPipe`/`TQue<QuePosition::VECIN|VECOUT>`/`DataCopyPad`/`GetBlockIdx`；桥接 `LocalTensor::GetPhyAddr()` → `(__ubuf__ T*)`；VF 层 `__VEC_SCOPE__` + `AscendC::Reg` 的 `RegTensor`/`MaskReg`/`UpdateMask`/`LoadAlign`/`StoreAlign`/`Div`/`Muls`/`Add`/`GetVecLen`；host 侧 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL_PARAM`/`gert::TilingContext`。
- 验证过的平台与 CANN 版本：Ascend950PR（`--soc=ascend950` 编译通过），CANN 9.1.0。仅编译级验证，未上板（与 frontmatter `platforms` 口径一致：编译可构建该平台）。
- 适用粒度：operator_family——elementwise/foreach 类 RegBase 算子的通用骨架形态。
- 路线优先级：recommended_fastpath（arch35 向量类算子的默认路线之一）。
- 明确不适用 / 未验证场景：运行时精度/性能未验证（后续迭代）；FP16/BF16 需在 VF body 增加 Cast 升精度链（本骨架未含）；多张量 listLen>1 需 ListTensorDesc 遍历（后续迭代）；Cube/矩阵类算子不适用。

## 配方结构（分层模型）

```
op_host:  OpDef(dtype/格式声明) + InferShape + TilingFunc(多核切分/UB 切分/ASCENDC_TPL_SEL_PARAM)
op_kernel/arch35:
  tiling_data.h   TilingData 标准 C++ struct（totalElements/blockFactor/ubFactor/coreNum/scalar）
  tiling_key.h    ASCENDC_TPL_ARGS_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)) 单 dtype
  regbase.h       VF body：__aicore__ 函数 + __VEC_SCOPE__ 块（寄存器计算链）
  *.h             外层 kernel 类：Init/Process/CopyIn/Compute/CopyOut（TPipe/TQue/DataCopyPad）
  *_arch35.cpp    kernel 入口：REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT
op_api:   L0（平台检查/取标量/ADD_TO_LAUNCHER_LIST_AICORE）+ L2 aclnn 两段式
op_graph: REG_OP + DYNAMIC_INPUT/DYNAMIC_OUTPUT + ATTR(scalar, Float)
```

关键桥接：`Compute()` 里对 `LocalTensor` 调 `GetPhyAddr()` 并强转 `(__ubuf__ T*)`，作为 VF body 入参——外层经典 AscendC 的队列搬运与内层 RegBase 的寄存器计算由此接通。

## 完整代码

VF body（终版，含 uint16_t 循环变量约束的修正）：

```cpp
// op_kernel/arch35/bench_no_harness_foreach_addcdiv_scalar_regbase.h
template <typename T>
__aicore__ inline void AddcdivScalarVf(
    __ubuf__ T* x1Addr, __ubuf__ T* x2Addr, __ubuf__ T* x3Addr,
    __ubuf__ T* yAddr, uint32_t count, float scalar)
{
    // 向量长度：256B / sizeof(T)，FP32 = 64 元素
    constexpr uint32_t VL = GetVecLen() / sizeof(T);
    uint16_t loopNum = static_cast<uint16_t>((count + VL - 1) / VL);

    __VEC_SCOPE__
    {
        MaskReg maskReg;
        RegTensor<T> x1Reg, x2Reg, x3Reg;
        RegTensor<T> divReg, mulReg, yReg;

        for (uint16_t i = 0; i < loopNum; i++) {
            uint32_t remain = count - i * VL;
            uint32_t active = remain > VL ? VL : remain;
            maskReg = UpdateMask<T>(active);

            // 1. Load: UB → RegTensor
            LoadAlign<T>(x1Reg, x1Addr + i * VL);
            LoadAlign<T>(x2Reg, x2Addr + i * VL);
            LoadAlign<T>(x3Reg, x3Addr + i * VL);

            // 2. Div: div = x2 / x3
            Div<T>(divReg, x2Reg, x3Reg, maskReg);

            // 3. Muls: mul = div * scalar
            Muls<T>(mulReg, divReg, scalar, maskReg);

            // 4. Add: y = x1 + mul
            Add<T>(yReg, x1Reg, mulReg, maskReg);

            // 5. Store: RegTensor → UB
            StoreAlign<T>(yAddr + i * VL, yReg, maskReg);
        }
    }
}
```

外层 kernel 类（终版：单缓冲、GetPhyAddr 桥接、线性化多核切分）：

```cpp
// op_kernel/arch35/bench_no_harness_foreach_addcdiv_scalar.h（节选关键成员，模板参数已收敛为 <T>）
template <typename T>
class KernelForeachAddcdivScalar {
    static constexpr int32_t BUFFER_NUM = 1;  // 骨架阶段：单缓冲

public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                 GM_ADDR workspace, const BenchNoHarnessForeachAddcdivScalarTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX3;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> x3Gm_;
    GlobalTensor<T> yGm_;

    int64_t totalElements_ = 0;
    int64_t blockFactor_   = 0;
    int64_t ubFactor_      = 0;
    int64_t blockLength_   = 0;  // 本核处理的元素数
    float   scalar_        = 0.0f;
};

template <typename T>
__aicore__ inline void KernelForeachAddcdivScalar<T>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
    GM_ADDR workspace, const BenchNoHarnessForeachAddcdivScalarTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    blockFactor_   = tilingData->blockFactor;
    ubFactor_      = tilingData->ubFactor;
    scalar_        = tilingData->scalar;

    // 计算本核数据范围（线性化多核切分）
    int64_t blockIdx = GetBlockIdx();
    int64_t startOffset = blockFactor_ * blockIdx;
    int64_t remainder = totalElements_ - startOffset;
    blockLength_ = (remainder > blockFactor_) ? blockFactor_ : remainder;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 骨架阶段 listLen=1，直接从 GM_ADDR 绑定第一个（也是唯一一个）张量
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1 + startOffset, blockLength_);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2 + startOffset, blockLength_);
    x3Gm_.SetGlobalBuffer((__gm__ T*)x3 + startOffset, blockLength_);
    yGm_.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // 分配 UB 缓冲区（4 queues × BUFFER_NUM）
    int64_t queBufSize = ubFactor_ * sizeof(T);
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, queBufSize);
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, queBufSize);
    pipe.InitBuffer(inQueueX3, BUFFER_NUM, queBufSize);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, queBufSize);
}

template <typename T>
__aicore__ inline void KernelForeachAddcdivScalar<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> x1Local = inQueueX1.template AllocTensor<T>();
    LocalTensor<T> x2Local = inQueueX2.template AllocTensor<T>();
    LocalTensor<T> x3Local = inQueueX3.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(x1Local, x1Gm_[progress * ubFactor_], copyParams, {false, 0, 0, 0});
    DataCopyPad(x2Local, x2Gm_[progress * ubFactor_], copyParams, {false, 0, 0, 0});
    DataCopyPad(x3Local, x3Gm_[progress * ubFactor_], copyParams, {false, 0, 0, 0});

    inQueueX1.EnQue(x1Local);
    inQueueX2.EnQue(x2Local);
    inQueueX3.EnQue(x3Local);
}

template <typename T>
__aicore__ inline void KernelForeachAddcdivScalar<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> x1Local = inQueueX1.template DeQue<T>();
    LocalTensor<T> x2Local = inQueueX2.template DeQue<T>();
    LocalTensor<T> x3Local = inQueueX3.template DeQue<T>();
    LocalTensor<T> yLocal  = outQueueY.template AllocTensor<T>();

    // 获取 UB 原始地址，传入 VF body（外层 TQue ↔ RegBase VF 的桥接点）
    __ubuf__ T* x1Addr = (__ubuf__ T*)x1Local.GetPhyAddr();
    __ubuf__ T* x2Addr = (__ubuf__ T*)x2Local.GetPhyAddr();
    __ubuf__ T* x3Addr = (__ubuf__ T*)x3Local.GetPhyAddr();
    __ubuf__ T* yAddr  = (__ubuf__ T*)yLocal.GetPhyAddr();

    // 进入 VF body：LoadAlign → Div → Muls → Add → StoreAlign
    AddcdivScalarVf<T>(x1Addr, x2Addr, x3Addr, yAddr,
                       static_cast<uint32_t>(currentNum), scalar_);

    outQueueY.template EnQue<T>(yLocal);
    inQueueX1.FreeTensor(x1Local);
    inQueueX2.FreeTensor(x2Local);
    inQueueX3.FreeTensor(x3Local);
}

template <typename T>
__aicore__ inline void KernelForeachAddcdivScalar<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(yGm_[progress * ubFactor_], yLocal, copyParams);

    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void KernelForeachAddcdivScalar<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    // UB 分块循环
    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1)
            ? (blockLength_ - ubFactor_ * i) : ubFactor_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}
```

kernel 入口与 TilingKey：

```cpp
// op_kernel/bench_no_harness_foreach_addcdiv_scalar_arch35.cpp
template <typename D_T_X>
__global__ __aicore__ void bench_no_harness_foreach_addcdiv_scalar(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessForeachAddcdivScalarTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessForeachAddcdivScalarTilingData, tilingData, tiling);

    NsForeachAddcdivScalar::KernelForeachAddcdivScalar<D_T_X> op;
    op.Init(x1, x2, x3, y, workspace, &tilingData);
    op.Process();
}
```

```cpp
// op_kernel/arch35/bench_no_harness_foreach_addcdiv_scalar_tiling_key.h（单 dtype）
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessForeachAddcdivScalar,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT,
                               ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    ),
);
```

tiling 切分核心（多核 + UB 切分，终版）：

```cpp
// op_host/arch35/bench_no_harness_foreach_addcdiv_scalar_tiling.cpp（TilingFunc 核心，终版形态）
constexpr int64_t TYPE_SIZE = 4;       // float
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 4;   // 3输入+1输出, 单缓冲

// 5. 多核切分（对齐到 32B）
int64_t ubBlockSize = GetUbBlockSize(context);
int64_t blockFactor = CeilAlign(CeilDiv(totalElements, coreNum), ubBlockSize);
int64_t usedCoreNum = CeilDiv(totalElements, blockFactor);

// 6. UB 切分（骨架阶段：单缓冲, bufferNum = 4 queues × 1 = 4）
int64_t bufferNum = SINGLE_BUF_TENSOR_COUNT;
int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE, bufferNum), ubBlockSize);

// 7. 填充 TilingData 字段
tiling->totalElements = totalElements;
tiling->blockFactor = blockFactor;
tiling->ubFactor = ubFactor;
tiling->coreNum = static_cast<int32_t>(usedCoreNum);
tiling->scalar = scalarValue;

context->SetBlockDim(usedCoreNum);

// 8. 设置 TilingKey（参数顺序与 tiling_key.h 的 ASCENDC_TPL_ARGS_DECL 一致）
uint32_t dTypeX = static_cast<uint32_t>(dataType);
ASCENDC_TPL_SEL_PARAM(context, dTypeX);
```

L0 侧 foreach 语义（框架遍历 TensorList，每张量一次 tiling+launch）：

```cpp
// op_api/bench_no_harness_foreach_addcdiv_scalar.cpp（L0 核心）
const aclTensorList* BenchNoHarnessForeachAddcdivScalar(
    const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
    const aclScalar* scalar, aclTensorList* y, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessForeachAddcdivScalar, x1, x2, x3, scalar, y);

    OP_CHECK(IsAiCoreSupport(x1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    OP_CHECK(scalar != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "scalar is null"), return nullptr);
    float scalarValue = scalar->ToFloat();

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessForeachAddcdivScalar,
        OP_INPUT(x1, x2, x3), OP_ATTR(scalarValue), OP_OUTPUT(y));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessForeachAddcdivScalar AiCore failed."),
        return nullptr);

    return y;
}
```

## 设计决策

- **`__VEC_SCOPE__` 块而不是 `__simd_vf__` 函数**：对齐 CANN 内置 foreach RegBase 实现（`foreach_addc_list_regbase.h` 的 ComputeFloatPath、`ceil_3510_impl.h`）的实际形态——VF 计算写在 `__aicore__ inline` 函数内的 `__VEC_SCOPE__` 块，由外层 Compute 直接调用。首版用 `__simd_vf__` 函数时 FP32 变体编译失败且错误被 FATBIN 掩盖，改 `__VEC_SCOPE__` 后错误显式化为向量循环归纳变量约束（uint16_t）；两个变量同时改动，`__simd_vf__` 形态的独立影响未隔离验证，选 `__VEC_SCOPE__` 的主要依据是与内置实现一致。
- **外层保留经典 TPipe/TQue/DataCopyPad，而不是纯 RegBase（LoadDataTensor/整段直算）**：骨架期先复用最成熟的 GM↔UB 搬运与队列同步路径，把新变量集中到 VF 计算一段；`GetPhyAddr()` 桥接让两层各用所长。
- **单 dtype/单缓冲/单变体先行**：OpDef/proto/tiling/TilingKey/SEL_PARAM 全链路只声明 FP32——声明未实现的 dtype 会让构建系统生成并编译对应变体而失败（另有专卡记录该 FATBIN 缺 .o 根因）。BUFFER_MODE 维度推迟到后续迭代。
- **线性化多核切分（blockFactor 对齐 32B）**：TensorList 展平为一维元素空间按核均分，为后续跨张量边界处理预留；尾核短块由 `blockLength_` 自然收缩。
- **尾块用 `UpdateMask(active)`**：允许最后一批不满 VL（GetVecLen/sizeof(T)=64），mask 屏蔽多余 lane；`uint16_t` 循环上界隐含批数 ≤65535，由 tiling 的 ubFactor（本平台约 15k 元素/批）保证。
- **scalar 经 TilingData 下发**（tiling 填 `tiling->scalar`，kernel 从 tilingData 读）：标量属性进 TilingData 而非 kernel 参数，与 ADD_TO_LAUNCHER_LIST_AICORE 的 OP_ATTR 通道衔接。
- 被放弃的设计：双缓冲（BUFFER_MODE=1 变体，推迟）；`__simd_vf__` + asc_vf_call 形态（见上）；kernel 内直接遍历 TensorList（骨架阶段 listLen=1，直接绑第一个张量）。

## 系统信息

- 工具链：CANN 9.1.0（`ASCEND_HOME_PATH` 指向 cann-9.1.0 安装根），构建 `bash build.sh --soc=ascend950 -j8`；构建前若已安装同名 vendor 包（`opp/vendors/<op>_custom`）先执行其 `scripts/uninstall.sh`。
- 目标平台：Ascend950PR（arch35 / DAV_3510，向量核）。
- 目录布局与文件清单：`operators/bench_no_harness_foreach_addcdiv_scalar/{CMakeLists.txt, build.sh, op_host/{*_def.cpp, *_infershape.cpp, CMakeLists.txt, arch35/*_tiling.cpp}, op_kernel/{*_arch35.cpp, CMakeLists.txt, arch35/{*_tiling_data.h, *_tiling_key.h, *_regbase.h, *.h}}, op_api/{aclnn_*.cpp/.h, *.cpp/.h}, op_graph/{*_proto.h, CMakeLists.txt}}`。
- 参考实现：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/foreach/`（foreach_addcdiv_scalar / foreach_add_scalar / foreach_addc_list_regbase.h）、`$ASCEND_HOME_PATH/x86_64-linux/asc/impl/adv_api/detail/math/ceil/ceil_3510_impl.h`。

## 预防

- 先确定 VF body 形态（`__VEC_SCOPE__` + `AscendC::Reg` API）再写外层，桥接点（GetPhyAddr → `__ubuf__`）一次写对。
- dtype 集合声明 = 已实现集合（OpDef/proto/tiling/TilingKey/op_api 五层同步），骨架期不要预留。
- `__VEC_SCOPE__` 内循环变量与上界直接写 `uint16_t`。

## 验证信号

问题与验证来源：一次 foreach TensorList 算子骨架开发会话，全链路 FP32 单变体收敛后的构建输出与产物核验。

- `bash build.sh --soc=ascend950 -j8` 输出 `[INFO] Build completed successfully!`，kernel 二进制 `BenchNoHarnessForeachAddcdivScalar_54b5c99ce4a41f39eb04ebd32d5cc193.o`（10024 字节）与 `.json`（2529 字节）生成（来源平台：950；证据等级：tool_observed）。
- 算子包 `custom_opp_ubuntu_x86_64.run`（446127 字节，SHA256 32eec939...）由 CPack 打包成功（来源平台：950；证据等级：tool_observed）。
- 源文件清单与构建产物一一对应（19 个源文件均在工程内）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅编译级验证：kernel 运行行为（tiling 数值、GetPhyAddr 桥接在真机上的正确性、多核切分）未上板验证。
- FP32 直算路径专用于 float；FP16/BF16 直接套用本骨架会因缺少 Cast 升精度链在变体编译/数值上出问题（未验证）。
- `GetPhyAddr()` 桥接基于 LocalTensor 的 UB 物理地址连续性假设（TQue 分配的整块 buffer）；若外层改为非连续/多段 UB 布局，桥接点需重新设计（未验证）。
- 其他 arch 代际（非 arch35/DAV_3510）的 VF API 与循环约束未验证，推断可迁移但需按目标架构核对。

**未决项**：

- 上板运行与精度验证（后续接续：L0 用例 Real 模式执行；UT 侧已在并行任务推进但代码证据在后续记录段）。
- 后续迭代：listLen>1 的 ListTensorDesc 遍历与跨张量多核切分；FP16/BF16 的 Cast 升精度链与 BUFFER_MODE 双缓冲维度恢复。
## 坏实践

- OpDef 声明多 dtype 而 VF body 只实现一个（构建必然在未实现变体上失败）。
- 在 `__VEC_SCOPE__` 内用 `uint32_t` 循环变量（编译器硬约束 uint16_t）。
- 混用 `AscendC::MicroAPI` 与 `AscendC::Reg` 命名空间引用而不确认别名关系——以 `using namespace AscendC::Reg;` 为准（两者为别名，混用易引起误解）。

## 误诊

- 首版 `__simd_vf__` 函数形态的失败曾被归因于函数形态本身；实际同一时段还存在向量循环归纳变量类型错误（uint32_t），两变量同时改动后编译通过，`__simd_vf__` 的独立影响未验证（不作为结论使用）。
