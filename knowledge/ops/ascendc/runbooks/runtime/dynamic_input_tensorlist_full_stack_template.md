---
type: "Runbook"
title: foreach TensorList 动态输入算子全栈实现模板
description: DYNAMIC_INPUT 算子从 IR 定义、tiling（GetIrInputInstanceInfo/GetDynamicInputShape + elemCounts/elemOffsets 前缀和线性化多核切分）、kernel（GetTensorAddr 二级指针 + 跨张量边界 ProcessTensorRange）到 op_api（aclTensorList）的完整实现模式；骨架形态验证到安装校验，多张量线性化切分形态经 UT 21/21 + 真实 NPU ST 19/19 上板验证。
tags: [task.implementation, topic.implementation.api_layer.c_api, topic.tiling, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# foreach TensorList 动态输入算子全栈实现模板

## 现象

（方案轨完整配方卡）开发一个 foreach 类算子：输入是 TensorList（每个元素为一个任意 shape 张量）加一个标量 p，输出是与输入等长的标量 TensorList（每个张量一个 p-范数）。工程模板（add_example）只覆盖普通固定输入算子，DYNAMIC 输入在 IR 定义、tiling 取 shape、kernel 取张量地址、op_api 传参四层都没有现成参考。通过研读 CANN 安装目录内置的 `foreach_norm` 实现（`opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/.../dynamic/`）与 `aclnn_foreach_norm.h` 头文件签名，组装出全栈实现并一次跑通"opbuild → 交叉编译 → 打包 → 安装校验"（ascend950，CANN 9.1.0，x86_64 主机）。本卡沉淀这套模板，供下一个 TensorList/foreach 类算子直接复用。

## 适用场景

- 工程形态：算子仓自定义算子（ops-registry-invoke 流程），输入或输出为 DYNAMIC（TensorList）。
- API 路径：IR 层 `OpDef::Input().ParamType(DYNAMIC)`；tiling 层 `GetIrInputInstanceInfo`/`GetDynamicInputDesc`/`GetDynamicInputShape`；kernel 层 `GM_ADDR` 二级指针解引用；op_api 层 `aclTensorList`/`AllocTensorList`/`OP_INPUT(x, scalar)`。
- 验证过的平台与 CANN 版本：ascend950（DAV_3510/arch35）+ CANN 9.1.0，x86_64 主机交叉编译；验证到安装校验通过（未上板跑算子）。
- 适用粒度：`operator_family`（所有 TensorList 动态输入的 vector 类算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：动态输入的多 dtype IR 声明（opbuild 报 dtype size 0，另有专卡记录该根因与等长列表修法；本模板骨架阶段收敛单 dtype）；上板数值正确性未验证。

## 设计决策

- **参照物选择**：没有从零猜测 API，而是直接读 CANN 内置 foreach 系列实现与 `aclnn_foreach_norm.h` 签名——`DYNAMIC_INPUT(x, ...)`/`DYNAMIC_OUTPUT(y, ...)` 宏、kernel 侧 `GetTensorAddr` 的 GM_ADDR 解引用方式都从内置实现逐字移植；这比查分散文档快且与运行时行为一致。
- **tiling 信息获取**：TensorList 长度用 `GetIrInputInstanceInfo(0)->GetInstanceNum()`（AnchorInstanceInfo），逐张量 shape 用 `GetDynamicInputShape(irIndex, k)`，dtype 用 `GetDynamicInputDesc(0,0)->GetDataType()`（取第一个张量的 dtype，L2 已保证列表内 dtype 一致）。
- **切分策略**：元素级多核切分——把 TensorList 展平成全局元素区间，`blockFactor = CeilAlign(CeilDiv(totalElem, coreNum), ubBlockSize)` 按核均分，各核用区间交集遍历各张量；比"一张量一核"对长短混合列表负载更均衡。每张量元素数通过 TilingData 的 `elemCounts[]` 数组带给 kernel（上限 64）。
- **workspace 布局**：系统保留段（2MB，`SyncAll` 需要，`SetSysWorkspace` + `GetUserWorkspace` 机制）之后放用户段 `partialResults[tensorIdx * usedCoreNum + coreIdx]`，核间归约结果按 (张量, 核) 二维索引存放，核 0 在 `SyncAll` 后串行合并。
- **Reduce API 选型**：Level2 count 模式接口（`ReduceSum<T>(dst, src, src, count)`）——API 文档产品列表明细未列 950，但 DAV_3510 为新一代兼容架构，且最终在 ascend950 交叉编译实测通过（编译级验证）；`WholeReduceSum`（免 sharedTmpBuffer）因产品支持情况不明未采用。
- **精度路径**：输入统一 `Cast` 到 FP32 计算（`CAST_NONE` 升精度），输出 `Cast` 回原 dtype（`CAST_ROUND` 降精度）；归约中间量全部 FP32。
- **放弃的候选**：`TILING_KEY_IS` 旧宏方式（内置 foreach_norm 用它，但新工程统一用 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 模板编程方式）；`BEGIN_TILING_DATA_DEF` 宏定义 TilingData（改用标准 C++ struct）。

## 系统信息

- toolchain：CANN 9.1.0（ASCEND_HOME_PATH 指向安装根），x86_64 主机交叉编译 aarch64 目标；`bash build.sh --soc=ascend950 -j8`。
- 目标平台：ascend950（arch35，AICore vector 核）。
- 相关文件（与实际读写一一对应）：
  - `op_host/<op_name>_def.cpp`（IR 定义）
  - `op_host/<op_name>_infershape.cpp`（输出标量 shape）
  - `op_host/arch35/<op_name>_tiling.cpp`（tiling）
  - `op_kernel/arch35/<op_name>_tiling_data.h` / `<op_name>_tiling_key.h` / `<op_name>.h`（kernel 侧）
  - `op_kernel/<op_name>_arch35.cpp`（kernel 入口）
  - `op_api/<op_name>.cpp`（L0）/ `op_api/aclnn_<op_name>.cpp`（L2）
  - `op_graph/<op_name>_proto.h`、`CMakeLists.txt`×4、`build.sh`

## 完整代码

各层关键函数（verbatim，取自终版写入与修复后状态）。

**1. IR 定义（`op_host/<op_name>_def.cpp`，DYNAMIC 输入/输出声明；多 dtype 会触发 opbuild dtype size 0，骨架阶段收敛单 dtype）：**

```cpp
namespace ops {
class BenchNoHarnessForeachNorm : public OpDef {
public:
    explicit BenchNoHarnessForeachNorm(const char* name) : OpDef(name)
    {
        // 输入 x: TensorList (DYNAMIC_INPUT)
        // 骨架阶段仅支持 FP32, 后续扩展 FP16/BF16
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 scalar: 范数阶数 p (REQUIRED_INPUT, 1元素张量)
        this->Input("scalar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 输出 y: TensorList (DYNAMIC_OUTPUT, 标量张量)
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // AICore 配置: ascend950 (arch35)
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(BenchNoHarnessForeachNorm);
} // namespace ops
```

**2. tiling（`op_host/arch35/<op_name>_tiling.cpp`，TilingFunc 完整函数——动态输入信息获取 + 元素级多核切分 + workspace 布局）：**

```cpp
static ge::graphStatus BenchNoHarnessForeachNormTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入信息 (DYNAMIC_INPUT x)
    auto insInfo = context->GetIrInputInstanceInfo(0);  // IR index 0 = x
    OP_CHECK_NULL_WITH_CONTEXT(context, insInfo);
    size_t tensorNum = insInfo->GetInstanceNum();
    OP_CHECK_IF(tensorNum == 0,
                OP_LOGE(context, "TensorList x is empty"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorNum > MAX_TENSOR_LIST_SIZE,
                OP_LOGE(context, "TensorList x length %zu exceeds max %u", tensorNum, MAX_TENSOR_LIST_SIZE),
                return ge::GRAPH_FAILED);

    // 获取数据类型 (从第一个张量)
    auto inputDesc = context->GetDynamicInputDesc(0, 0);  // IR index 0, relative index 0
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    // 3. 计算 totalElem 和 elemCounts
    int64_t totalElem = 0;
    auto tiling = context->GetTilingData<BenchNoHarnessForeachNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(BenchNoHarnessForeachNormTilingData), 0,
                 sizeof(BenchNoHarnessForeachNormTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    for (size_t k = 0; k < tensorNum; k++) {
        auto shape = context->GetDynamicInputShape(0, k);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t elemCount = shape->GetStorageShape().GetShapeSize();
        tiling->elemCounts[k] = elemCount;
        totalElem += elemCount;
    }

    // 4. 获取 scalar (从输入张量, 这里仅在 tiling 阶段不读值, 由 kernel 读取)
    // scalar 是 REQUIRED_INPUT (index 1), tiling 阶段不读值
    // NORM_TYPE 在骨架阶段固定为 2 (L2 范数)
    float scalarVal = 2.0f;  // 固定 L2
    uint32_t normType = GetNormType(scalarVal);

    // 5. 多核切分
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t blockFactor = CeilAlign(CeilDiv(totalElem, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalElem, blockFactor);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }

    // 6. UB 切分
    int64_t ubFactor = ComputeUbFactor(ubSize, dtype);

    // 7. 设置 TilingData
    tiling->tensorNum = static_cast<int64_t>(tensorNum);
    tiling->totalElem = totalElem;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->usedCoreNum = usedCoreNum;
    tiling->scalarVal = scalarVal;
    tiling->invP = (normType == 5) ? (1.0f / scalarVal) : 0.0f;

    // 8. 设置 BlockDim
    context->SetBlockDim(usedCoreNum);

    // 9. 设置 TilingKey
    // 参数顺序与 tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义一致
    // D_T_X: 数据类型, NORM_TYPE: 范数类型
    uint32_t dTypeX = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, normType);

    // 10. 设置 WorkspaceSize
    size_t wsSize = ComputeWorkspaceSize(static_cast<int64_t>(tensorNum), usedCoreNum);
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = wsSize;

    OP_LOGD(context, "Tiling: tensorNum=%zu, totalElem=%ld, blockFactor=%ld, ubFactor=%ld, "
            "usedCoreNum=%ld, dtype=%d, normType=%u, wsSize=%zu",
            tensorNum, totalElem, blockFactor, ubFactor, usedCoreNum,
            static_cast<int>(dtype), normType, wsSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BenchNoHarnessForeachNorm)
    .Tiling(BenchNoHarnessForeachNormTilingFunc)
    .TilingParse<BenchNoHarnessForeachNormCompileInfo>(TilingParseForBenchNoHarnessForeachNorm);
```

其中 workspace 大小计算（系统 2MB 保留 + 用户段）：

```cpp
constexpr size_t RESERVED_WORKSPACE_SIZE = 2 * 1024 * 1024;  // 2MB 系统workspace (SyncAll)
constexpr size_t WORKSPACE_ALIGN = 256;

// 计算 workspace 大小
static size_t ComputeWorkspaceSize(int64_t tensorNum, int64_t usedCoreNum)
{
    // 用户 workspace: partialResults
    // 布局: partialResults[tensorIdx * usedCoreNum + coreIdx]
    size_t partialResultsSize = static_cast<size_t>(tensorNum * usedCoreNum * TYPE_SIZE_FLOAT);
    // 对齐到 256B
    partialResultsSize = CeilAlign(static_cast<int64_t>(partialResultsSize),
                                    static_cast<int64_t>(WORKSPACE_ALIGN));
    // 总 workspace = 系统workspace + 用户workspace
    return RESERVED_WORKSPACE_SIZE + partialResultsSize;
}
```

**3. kernel 侧 TilingData / TilingKey（标准 struct + ASCENDC_TPL 模板编程，弃用旧宏）：**

```cpp
// op_kernel/arch35/<op_name>_tiling_data.h
constexpr uint32_t MAX_TENSOR_NUM = 64;  // TensorList 最大长度

struct BenchNoHarnessForeachNormTilingData {
    int64_t tensorNum = 0;      // TensorList 长度 L (1~64)
    int64_t totalElem = 0;      // 所有序列张量的元素总数
    int64_t blockFactor = 0;    // 每核负责的元素数（32B 对齐）
    int64_t ubFactor = 0;       // UB 单次处理的 FP32 元素数
    int64_t usedCoreNum = 0;    // 实际使用核数
    float   scalarVal = 0.0f;   // 范数阶数 p
    float   invP = 0.0f;        // 1/p（通用 p 范数用，NORM_TYPE=5 时有效）
    // 每个张量的元素数（Kernel 侧通过此数组确定各张量边界）
    int64_t elemCounts[MAX_TENSOR_NUM] = {0};
};

// op_kernel/arch35/<op_name>_tiling_key.h
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BenchNoHarnessForeachNorm,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(NORM_TYPE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5)
);

// 骨架阶段: 仅 FP32 + NORM_TYPE=2
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(NORM_TYPE, ASCENDC_TPL_UI_LIST, 2)
    )
);
```

**4. kernel 入口 + DYNAMIC 输入取址（`op_kernel/<op_name>_arch35.cpp` 与 `<op_name>.h`；GetTensorAddr 的 GM_ADDR 二级指针机制从内置 foreach 实现移植）：**

```cpp
// op_kernel/<op_name>_arch35.cpp
template <typename D_T_X, int NORM_TYPE>
__global__ __aicore__ void bench_no_harness_foreach_norm(
    GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessForeachNormTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessForeachNormTilingData, tilingData, tiling);

    NsBenchNoHarnessForeachNorm::KernelBenchNoHarnessForeachNorm<D_T_X, NORM_TYPE> op;
    op.Init(x, scalar, y, workspace, &tilingData);
    op.Process();
}

// op_kernel/arch35/<op_name>.h —— GetTensorAddr: 从 DYNAMIC_INPUT 的 GM_ADDR 获取第 index 个张量的地址
// 框架机制: GM_ADDR 的前 8 字节存储偏移量(字节),
//          偏移量处存放 uint64_t 数组(各张量地址)
template <typename D_T_X, int NORM_TYPE>
__aicore__ inline __gm__ D_T_X* KernelBenchNoHarnessForeachNorm<D_T_X, NORM_TYPE>::GetTensorAddr(
    uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);  // 偏移/8 = uint64_t 元素数
    return reinterpret_cast<__gm__ D_T_X*>(*(retPtr + index));
}
```

Init 中绑定 workspace 与各核元素区间（系统/用户 workspace 机制）：

```cpp
template <typename D_T_X, int NORM_TYPE>
__aicore__ inline void KernelBenchNoHarnessForeachNorm<D_T_X, NORM_TYPE>::Init(
    GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
    const BenchNoHarnessForeachNormTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();

    // 保存 Tiling 参数
    tensorNum_ = tilingData->tensorNum;
    totalElem_ = tilingData->totalElem;
    blockFactor_ = tilingData->blockFactor;
    ubFactor_ = tilingData->ubFactor;
    usedCoreNum_ = tilingData->usedCoreNum;
    scalarVal_ = tilingData->scalarVal;
    invP_ = tilingData->invP;
    elemCounts_ = tilingData->elemCounts;

    // 保存 GM 指针
    xPtr_ = x;
    yPtr_ = y;

    // 设置系统 workspace (SyncAll 需要)
    SetSysWorkspace(workspace);
    userWorkspace_ = GetUserWorkspace(workspace);

    // 绑定 partialResults workspace
    // 布局: partialResults[tensorIdx * usedCoreNum + coreIdx]
    partialResultsGM_.SetGlobalBuffer(
        reinterpret_cast<__gm__ ComputeT*>(userWorkspace_), tensorNum_ * usedCoreNum_);

    // 计算当前核负责的元素范围
    myStart_ = static_cast<int64_t>(blockIdx_) * blockFactor_;
    myEnd_ = (myStart_ + blockFactor_ > totalElem_) ? totalElem_ : (myStart_ + blockFactor_);

    // 初始化 Buffer
    pipe_.InitBuffer(inQueue_, BUFFER_NUM, ubFactor_ * sizeof(D_T_X));
    pipe_.InitBuffer(workBuf0_, ubFactor_ * sizeof(ComputeT));
    pipe_.InitBuffer(workBuf1_, ubFactor_ * sizeof(ComputeT));
    pipe_.InitBuffer(reduceTmpBuf_, REDUCE_TMP_MIN);
    pipe_.InitBuffer(resultBuf_, RESULT_BUF_SIZE);
}
```

两阶段 Process（各核独立归约 → SyncAll → 核 0 合并输出）：

```cpp
template <typename D_T_X, int NORM_TYPE>
__aicore__ inline void KernelBenchNoHarnessForeachNorm<D_T_X, NORM_TYPE>::Process()
{
    // Phase 1: 各核独立归约
    ComputePartialResults();

    // 跨核同步
    SyncAll();

    // Phase 2: 核 0 合并 + 输出
    if (GetBlockIdx() == 0) {
        MergeAndOutput();
    }
}
```

（`ComputePartialResults` 内按全局元素区间与各张量区间求交集后 `GetTensorAddr(k, xPtr_)` 取址、`DataCopyPad` 搬入、FP32 归约、部分结果写 `partialResultsGM_[k * usedCoreNum_ + blockIdx_]`；`MergeAndOutput` 核 0 逐张量合并各核部分结果、终算后 `GetTensorAddr(k, yPtr_)` 写回——完整实现见工程终版，机制要点如上。）

### 增量（多张量线性化切分 + 跨张量边界处理，上板验证形态）

逐元素 foreach 形态（`y_i = x1_i + (x2_i/x3_i)*scalar`，三 TensorList 输入 + 标量属性 + TensorList 输出，listLen≤64，**单次 kernel 启动**处理整个 TensorList）的细化实现，UT 21/21 + ST Real 19/19（含 listLen=2/4/8 与 16M 元素大 shape）上板验证：

**TilingData**（`op_kernel/arch35/<op>_tiling_data.h`，纯 C++ struct，弃用 BEGIN_TILING_DATA_DEF 宏）：

```cpp
// TensorList 最大长度（规格上限 64）
constexpr int32_t FOREACH_MAX_LIST_LEN = 64;

struct <Op>TilingData {
    int64_t totalElements;
    int64_t blockFactor;
    int64_t ubFactor;
    int32_t listLen;
    int32_t coreNum;
    float   scalar;
    int64_t elemCounts[FOREACH_MAX_LIST_LEN];    // 各张量元素数
    int64_t elemOffsets[FOREACH_MAX_LIST_LEN + 1]; // 前缀和，elemOffsets[listLen]=total
};
```

**tiling 侧三段核心函数**（`op_host/arch35/<op>_tiling.cpp`，verbatim）：

```cpp
// 获取 TensorList 长度
static int32_t GetTensorListLen(gert::TilingContext* context)
{
    const auto* insInfo = context->GetIrInputInstanceInfo(0);
    if (insInfo == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(insInfo->GetInstanceNum());
}

// 获取 shape 和属性信息（遍历 TensorList）
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context,
                                           int64_t* totalElements,
                                           ge::DataType* dataType, float* scalarValue,
                                           int32_t* listLen,
                                           <Op>TilingData* tiling)
{
    // 获取 TensorList 长度
    *listLen = GetTensorListLen(context);
    OP_CHECK_IF(*listLen <= 0,
                OP_LOGE(context, "TensorList length is 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*listLen > FOREACH_MAX_LIST_LEN,
                OP_LOGE(context, "TensorList length %d exceeds max %d", *listLen, FOREACH_MAX_LIST_LEN),
                return ge::GRAPH_FAILED);

    // 获取 dtype（从第一个张量）
    auto inputDesc = context->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT};
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0,
                OP_LOGE(context, "invalid dtype: %d", static_cast<int>(*dataType)),
                return ge::GRAPH_FAILED);

    // 获取 scalar 属性
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scalarPtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalarPtr);
    *scalarValue = *scalarPtr;

    // 遍历 TensorList，累计元素总数，填充 elemCounts/elemOffsets
    *totalElements = 0;
    tiling->elemOffsets[0] = 0;
    for (int32_t i = 0; i < *listLen; i++) {
        auto inputShape = context->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
        auto shape = EnsureNotScalar(inputShape->GetStorageShape());
        int64_t elemCount = shape.GetShapeSize();
        tiling->elemCounts[i] = elemCount;
        *totalElements += elemCount;
        tiling->elemOffsets[i + 1] = *totalElements;
    }

    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口（切分 + TilingKey）
static ge::graphStatus <Op>TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto tiling = context->GetTilingData<<Op>TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(<Op>TilingData), 0, sizeof(<Op>TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t totalElements;
    ge::DataType dataType;
    float scalarValue;
    int32_t listLen;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalElements, &dataType, &scalarValue, &listLen, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 空 TensorList / 0 元素处理
    if (totalElements == 0) {
        tiling->listLen = listLen;
        tiling->scalar = scalarValue;
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), 0U);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分（对齐到 32B）
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t blockFactor = CeilAlign(CeilDiv(totalElements, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalElements, blockFactor);

    // 双缓冲选择
    uint64_t useDoubleBuffer = (totalElements > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    // UB 切分
    int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE, bufferNum), ubBlockSize);

    tiling->totalElements = totalElements;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->listLen = listLen;
    tiling->coreNum = static_cast<int32_t>(usedCoreNum);
    tiling->scalar = scalarValue;
    // elemCounts/elemOffsets 已在 GetShapeAttrsInfo 中填充

    context->SetBlockDim(usedCoreNum);

    // 设置 TilingKey：D_T_X (DATATYPE), BUFFER_MODE (UINT)
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}
```

**kernel 侧 Process（跨张量边界处理，`op_kernel/arch35/<op>.h`，verbatim；GetTensorAddr 机制见上文第 4 节）**：

```cpp
template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelForeachAddcdivScalar<T, BUFFER_MODE>::ProcessTensorRange(
    __gm__ T* x1Ptr, __gm__ T* x2Ptr, __gm__ T* x3Ptr, __gm__ T* yPtr,
    int64_t startOffset, int64_t elemCount)
{
    // 从原始指针创建 GlobalTensor（DataCopyPad 需要 GlobalTensor 类型）
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> x3Gm;
    GlobalTensor<T> yGm;
    x1Gm.SetGlobalBuffer(x1Ptr + startOffset, elemCount);
    x2Gm.SetGlobalBuffer(x2Ptr + startOffset, elemCount);
    x3Gm.SetGlobalBuffer(x3Ptr + startOffset, elemCount);
    yGm.SetGlobalBuffer(yPtr + startOffset, elemCount);

    // 单张量内 UB 分块循环
    int64_t loopCount = (elemCount + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1)
            ? (elemCount - ubFactor_ * i) : ubFactor_;
        CopyIn(i, currentNum, x1Gm, x2Gm, x3Gm);
        Compute(currentNum);
        CopyOut(i, currentNum, yGm);
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelForeachAddcdivScalar<T, BUFFER_MODE>::Process()
{
    if (totalElements_ == 0 || blockFactor_ == 0) {
        return;
    }

    // 线性化多核切分
    int64_t blockIdx = GetBlockIdx();
    int64_t globalStart = blockFactor_ * blockIdx;
    if (globalStart >= totalElements_) {
        return;
    }
    int64_t globalEnd = globalStart + blockFactor_;
    if (globalEnd > totalElements_) {
        globalEnd = totalElements_;
    }

    int64_t currentIdx = globalStart;
    int32_t tensorIdx = 0;

    // 跳过不含当前核元素的张量
    while (tensorIdx < listLen_ && elemOffsets_[tensorIdx + 1] <= globalStart) {
        tensorIdx++;
    }

    // 逐张量处理当前核负责的元素范围（可能跨多个张量）
    while (currentIdx < globalEnd && tensorIdx < listLen_) {
        int64_t tensorStart = elemOffsets_[tensorIdx];
        int64_t tensorEnd = elemOffsets_[tensorIdx + 1];
        int64_t localStart = currentIdx - tensorStart;
        int64_t localEnd = globalEnd;
        if (tensorEnd < localEnd) {
            localEnd = tensorEnd;
        }
        localEnd -= tensorStart;

        // 获取当前张量的 GM 基地址
        __gm__ T* x1Gm = GetTensorAddr(static_cast<uint16_t>(tensorIdx), x1Ptr_);
        __gm__ T* x2Gm = GetTensorAddr(static_cast<uint16_t>(tensorIdx), x2Ptr_);
        __gm__ T* x3Gm = GetTensorAddr(static_cast<uint16_t>(tensorIdx), x3Ptr_);
        __gm__ T* yGm  = GetTensorAddr(static_cast<uint16_t>(tensorIdx), yPtr_);

        ProcessTensorRange(x1Gm, x2Gm, x3Gm, yGm, localStart, localEnd - localStart);

        currentIdx = localEnd + tensorStart;
        tensorIdx++;
    }
}
```

Init 关键段（保存 TensorList GM_ADDR、拷 elemOffsets 到局部数组避免循环内反复读 GM 的 TilingData、按 BUFFER_NUM 分 UB）：

```cpp
    totalElements_ = tilingData->totalElements;
    blockFactor_   = tilingData->blockFactor;
    ubFactor_      = tilingData->ubFactor;
    listLen_       = tilingData->listLen;
    scalar_        = tilingData->scalar;

    x1Ptr_ = x1; x2Ptr_ = x2; x3Ptr_ = x3; yPtr_ = y;

    // 复制 elemOffsets 到局部数组（Kernel 内频繁访问，避免反复读 GM TilingData）
    for (int32_t i = 0; i <= listLen_; i++) {
        elemOffsets_[i] = tilingData->elemOffsets[i];
    }

    int64_t queBufSize = ubFactor_ * sizeof(T);
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, queBufSize);
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, queBufSize);
    pipe.InitBuffer(inQueueX3, BUFFER_NUM, queBufSize);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, queBufSize);
```

设计要点（增量）：线性化切分 vs 按张量切分——把全 list 元素总数摊到核上（每核 blockFactor 连续区间），小张量场景核利用率远好于"一 tensor 一 launch"或"按 tensor 边界切核"，代价是 kernel 侧维护"全局区间 → (tensorIdx, 本地区间)"映射（elemOffsets 前缀和 + 跳过逻辑）；`GetInstanceNum()` 是 IR 输入的实例数，不是 GetInputNum（那是展开后的 kernel 输入总数）；双缓冲（totalElements > 1024 阈值）编码进 TilingKey（`ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer)`），UT 可按 key 断言；`blockFactor` 必须 CeilAlign 到 ubBlockSize（32B），否则跨核边界出现非整块搬运。

**5. op_api L0（`op_api/<op_name>.cpp`，TensorList 经 OP_INPUT 直接下发；注意 OP_CHECK_NULL 宏契约）：**

```cpp
const aclTensorList* BenchNoHarnessForeachNorm(
    const aclTensorList* x, const aclTensor* scalar, const aclTensorList* y, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessForeachNorm, x, scalar, y);

    OP_CHECK(x == nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x is null"), return nullptr);
    OP_CHECK(scalar == nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "scalar is null"), return nullptr);
    OP_CHECK(y == nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "y is null"), return nullptr);
    OP_CHECK(executor == nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is null"), return nullptr);

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    // 调用 Kernel: 传递 TensorList + scalar + output TensorList
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessForeachNorm,
        OP_INPUT(x, scalar), OP_OUTPUT(y));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessForeachNorm AiCore launch failed."),
             return nullptr);

    return y;
}
```

**6. op_api L2（`op_api/aclnn_<op_name>.cpp`，TensorList 逐张量 Contiguous + 重建 TensorList + scalar 转张量，两段式接口）：**

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessForeachNormGetWorkspaceSize(
    const aclTensorList* x,
    const aclScalar* scalar,
    aclTensorList* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBenchNoHarnessForeachNorm, DFX_IN(x, scalar), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, scalar, y, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Contiguous: 对输入 TensorList 中每个张量做连续化
    uint64_t xSize = x->Size();
    std::vector<const aclTensor*> xContiguousVec;
    xContiguousVec.reserve(xSize);
    for (uint64_t i = 0; i < xSize; i++) {
        auto xContiguous = l0op::Contiguous((*x)[i], uniqueExecutor.get());
        CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        xContiguousVec.push_back(xContiguous);
    }

    // 创建连续化的输入 TensorList
    aclTensorList* xContiguousList = uniqueExecutor->AllocTensorList(
        xContiguousVec.data(), xContiguousVec.size());
    CHECK_RET(xContiguousList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 scalar 转换为 1元素张量
    const aclTensor* scalarTensor = uniqueExecutor->ConvertToTensor(scalar, DataType::DT_FLOAT);
    CHECK_RET(scalarTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用 L0 算子
    const aclTensorList* opResult = l0op::BenchNoHarnessForeachNorm(
        xContiguousList, scalarTensor, y, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnBenchNoHarnessForeachNorm(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBenchNoHarnessForeachNorm);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
```

**7. op_graph 原型（`op_graph/<op_name>_proto.h`）与 infershape（输出为 0 维标量列表）：**

```cpp
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))

// op_host/<op_name>_infershape.cpp
static ge::graphStatus InferShape4BenchNoHarnessForeachNorm(gert::InferShapeContext* context)
{
    // 获取输入 x 的 shape (动态输入的第一个张量)
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    // 输出 y: 标量张量 (0维, shape为空)
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // 输出为标量: shape 为空 (0维)
    outputShape->SetDimNum(0);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BenchNoHarnessForeachNorm).InferShape(InferShape4BenchNoHarnessForeachNorm);
```

## 预防

- 先读内置 foreach 实现再动手：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/`（dynamic/ 子目录）与 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnnop/aclnn_foreach_*.h`，确认 DYNAMIC 声明与 TensorList 签名形态。
- DYNAMIC 输入的 IR 定义先收敛单 dtype 跑通 opbuild，再扩多 dtype（多 dtype 触发 opbuild `dtype size 0` 的根因与等长列表修法另有专卡）。
- op_api 用 `l0op::Contiguous` 时记得给 `cust_opapi` 显式链 `libopapi.so`（见 [l0op::Contiguous 未定义符号致安装校验失败](../compilation/libcust_opapi_l0op_contiguous_undefined_symbol.md)），否则安装校验失败。

## 验证信号

问题与验证来源：一次完整的 TensorList 算子骨架开发（从空工程到安装成功）实测。

- 交叉编译输出 `[INFO] Build completed successfully!`，生成 kernel 二进制 `build/op_kernel/ascendc_kernels/binary/ascend950/<op_name>/BenchNoHarnessForeachNorm_....o`（来源平台：950；证据等级：tool_observed）。
- 安装包执行输出 `SUCCESS`，`libcust_opapi.so` 通过 dlopen 校验并部署到 `opp/vendors/<op_name>_custom/`（来源平台：950；证据等级：tool_observed）。
- 安装后核验：kernel binary、`aic-ascend950-ops-info.json`、`libcust_optiling.so`、`libcust_opapi.so` 均存在（来源平台：950；证据等级：tool_observed）。
未验证：上板执行、输出精度、性能（本卡证据止于构建与安装）。

增量信号（多张量线性化切分形态，Ascend950PR + CANN 9.1.0，完整开发-联调会话：UT → 重装包 → ST Real/Mock）：

- UT：21/21 通过（单张量回归 + listLen=2/4/8、同/异 shape、双缓冲阈值上下、空输入边界、非法 dtype 失败校验）（来源平台：950；证据等级：tool_observed）。
- ST Real（重装算子包后）：19/19 通过；多张量大 shape 用例 MERE=1.76e-09/9.02e-09/1.09e-09 量级、MARE 1e-07~2.6e-04 量级（16M/4M 元素）（来源平台：950；证据等级：tool_observed）。
- ST Mock（CPU golden 自检）：19/19 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 多 dtype DYNAMIC 声明会触发 opbuild `dtype size 0`（本模板骨架阶段收敛单 dtype FP32；恢复多 dtype 未验证）。
- `GetTensorAddr` 的偏移解引用机制（首 8 字节偏移 + uint64_t 地址数组）来自 CANN 9.1.0 内置实现，属运行时布局契约——CANN 升级后需重新核对（未验证，推断可迁移）。
- TensorList 长度上限 64、单张量元素数上限等由 TilingData 数组与 L2 校验常量决定，按算子需求调整。
- 上板数值正确性：骨架形态未验证；多张量线性化切分形态（增量）已由真实 NPU ST 19/19 验证（仅 FP32 实测；FP16/BF16 的 TilingKey dtype 维度已预留但未在其上验证）。FP32 Cast 通路与元素级切分的精度/性能表现以增量信号为准。本模板 L2 的逐张量 Contiguous 写法在后续真机联调中暴露过大张量挂起问题（须 IsContiguous 守卫，另有专卡）。
- （增量）listLen≤64（TilingData 定长数组）；超过须改动态结构（未验证）。仅逐元素无依赖算子验证；输出与输入张量一一对应（y[t] 与 x[t] 同 shape）。
- （增量）标量属性进 TilingData 带来的缓存复用坑（同 shape 异 scalar 复用旧 tiling）另有专卡——采用本模式时属性值与 TilingKey 的关系必须显式处理。

**未决项**：

- 恢复 FP16/BF16 多 dtype（IR 定义 + TilingKey 选择集 + L2 校验同步放开）并复测 opbuild。
- 上板运行验证数值精度与多核 SyncAll 路径正确性（接续点：ST Real 模式）。
## 坏实践

- 不看内置实现、按固定输入模板猜 DYNAMIC 输入的取址方式（GM_ADDR 语义猜不出）。
- tiling 里忽略 `GetInstanceNum()` 直接按固定输入处理 TensorList。

## 复现增量：直接强转 GM_ADDR 的故障形态（507035 + errcode 95 + double free）

另一独立轨迹并入同 trace 的运行时故障实例，补上"不按 GetTensorAddr 模式取址"时会发生什么（上文坏实践第 1 条的实测形态）：

**现象**：TensorList 算子首轮上板（FP32 [1024,1024]，listLen=1），`aclrtSynchronizeStream` 返回 507035，随后 `free(): double free detected in tcache 2`、进程 abort（double free 是 507035 失败后错误清理路径的伴生现象，不是独立内存 bug）。plog 定位设备侧真因：

```text
[ERROR] RUNTIME(...): [device_error_proc_c.cc:749] ProcessDavidStarsCoreErrorInfo:
  The error from device(chipId:0, dieId:0), ... there is an aivec error exception,
  core id is 67, error code = 95, ... mte error info: 0x6adb460000020305, ...
The extend info: errcode:(95) errorStr: The DDR address of the MTE instruction is
  out of range. subErrType: 0x4.
[ERROR] ... PreCheckTaskErr:An error occurred in the kernel task, retCode=0x31, [vector core exception].
[ERROR] ... rtStreamSynchronize:ErrCode=507035, desc=[vector core exception], InnerCode=0x715005e
```

**修复前（直接强转，触发 507035）**——listLen=1 场景下的"直观"写法：

```cpp
    // 迭代一：listLen=1，直接从 GM_ADDR 绑定第一个（也是唯一一个）张量
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1 + startOffset, blockLength_);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2 + startOffset, blockLength_);
    x3Gm_.SetGlobalBuffer((__gm__ T*)x3 + startOffset, blockLength_);
    yGm_.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);
```

**修复后（GetTensorAddr 解引用，同上文第 4 节机制；Init 改用第 0 个张量地址）**：

```cpp
    // DYNAMIC_INPUT: 从 TensorList GM_ADDR 提取第 0 个张量的实际数据地址
    __gm__ T* x1Addr = GetTensorAddr(0, x1);
    __gm__ T* x2Addr = GetTensorAddr(0, x2);
    __gm__ T* x3Addr = GetTensorAddr(0, x3);
    __gm__ T* yAddr  = GetTensorAddr(0, y);

    x1Gm_.SetGlobalBuffer(x1Addr + startOffset, blockLength_);
    x2Gm_.SetGlobalBuffer(x2Addr + startOffset, blockLength_);
    x3Gm_.SetGlobalBuffer(x3Addr + startOffset, blockLength_);
    yGm_.SetGlobalBuffer(yAddr + startOffset, blockLength_);
```

机制：DYNAMIC_INPUT/DYNAMIC_OUTPUT 下框架传给 kernel 的 `GM_ADDR` **不是张量数据地址，而是 TensorList 描述符地址**；kernel 直接 `(__gm__ T*)x1` 并 `SetGlobalBuffer` 后，MTE 会把描述符内存当作数据区访问，块长度按数据 shape 计算，越出该小块内存即触发 "The DDR address of the MTE instruction is out of range"（errcode 95 → host 侧 507035 vector core exception）。同机参考算子（foreach_norm）的 kernel 里有现成的解引用实现并注释了该框架机制，是独立佐证。

**配套与误诊提示**：tiling 侧对 DYNAMIC 输入须用 `GetDynamicInputShape(0, 0)`/`GetDynamicInputDesc(0, 0)`（替换 `GetInputShape(0)`/`GetInputDesc(0)`）——但本实例第一轮先修了 tiling API，重跑仍 507035：tiling API 用错不是该崩溃的根因（listLen=1 时两写法可能等价），读 plog 才定位到 kernel 侧取址。507035 + plog errcode 95 出现时，优先怀疑"把非数据指针当数据地址"（描述符/workspace 基址误用同属此类），再看越界偏移。

**独立 trace 新增验证信号**（均为 tool_observed）：

- 修复前：首用例 `流同步失败: 507035`，plog `errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4`（aivec exception, core id 67，retCode=0x31 vector core exception）（来源平台：950；证据等级：tool_observed）。
- 修复后（仅改 kernel 取址 + tiling Dynamic API，重编译重装）：同一用例 `[PASS] MERE=1.76e-09, MARE=2.05e-07 (threshold=1.22e-04, 1048576 elems, nan/inf mismatch=0)`（来源平台：950；证据等级：tool_observed）。
- 参考佐证：同机 foreach_norm kernel 的 GetTensorAddr 实现含相同机制注释（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界**：描述符布局（8 字节偏移 + uint64 地址数组）是 CANN 9.1.0 实测行为，属框架内部布局，版本升级可能变化（迁移时以参考算子/头文件为准重新核对）；仅 listLen≤64、AIV vector kernel 场景验证，Cube/AIC 路径未验证。