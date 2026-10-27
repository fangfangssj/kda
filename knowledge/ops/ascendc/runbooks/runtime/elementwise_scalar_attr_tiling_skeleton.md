---
type: "Runbook"
title: 带标量属性 elementwise 算子 tiling 骨架配方
description: 标量属性（base/scale/shift/beta/threshold）经 context->GetAttrs()->GetFloat(idx) 读入 TilingData；多核切分（blockFormer 对齐 + 首尾核独立 loop/tail）+ UB 切分（bufferDivisor 字节口径 + 32B 对齐）+ ASCENDC_TPL_SEL_PARAM 按 dtype/branch 设 TilingKey；空 tensor 早退。多 dtype 分支持统一字节量纲参数化（elemBytes/bufferNum/alignFactor），纯计算层与 glue 拆分便于 UT 同源编译 + 独立 oracle；骨架可被同算子其他分支二次复用，aclnn 冒烟可用"真实 tiling + kernel 桩"验证接口链路。
tags: [task.implementation, topic.tiling, topic.compute.elementwise.activation]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# 带标量属性 elementwise 算子 tiling 骨架配方

## 现象（任务场景）

开发自定义 aclnn 算子（Elementwise 指数运算 y = base^((x*scale+shift))，输入 x 一个 tensor + base/scale/shift 三个 float 标量属性，FP16/FP32/BF16，目标 Ascend950PR/arch35，使用 ASCENDC_TPL_ARGS_DECL 新模板机制）时，需要实现 op_host 侧 tiling：读平台信息与标量属性、多核切分、UB 切分、按 dtype 与计算分支设置 TilingKey，并处理空 tensor。参考样例 add_example 没有标量属性，属性读取姿势在样例里找不到。最终实现的 tiling 编译通过并随算子包成功产出（3 个 dtype 变体）。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子 op_host tiling 实现（新式 TPL 模板 + 标准 C++ struct TilingData）。
- API 路径：`gert::TilingContext`（`GetPlatformInfo`/`GetInputShape`/`GetInputDesc`/`GetAttrs`/`GetTilingData`/`SetBlockDim`/`GetWorkspaceSizes`）、`RuntimeAttrs::GetFloat(idx)`、`platform_ascendc::PlatformAscendC`（`GetCoreNumAiv`/`GetCoreMemSize`）、`ASCENDC_TPL_SEL_PARAM`、`IMPL_OP_OPTILING` 注册。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510，UB 253952B，编译级验证：tiling 随 cust_optiling 目标编译通过、算子包产出）；CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——单输入 elementwise + 标量属性的 tiling 结构可直接迁移；多输入/规约类需重写切分逻辑。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：上板运行时 tiling 数值正确性（本卡验证止于编译与打包，UT/上板属后续）；带广播/多维切分的算子；属性为 int/string/list 的读取（GetFloat 之外的重载未验证）。

## 解决配方（完整实现）

### 1. TilingData 结构（标准 C++ struct，op_kernel/arch35/<op>_tiling_data.h）

```cpp
#include <cstdint>

struct BenchNoHarnessExpTilingData {
    uint64_t totalLength = 0;       // 元素总数
    uint32_t coreNum = 0;           // 实际使用核数
    uint64_t blockFormer = 0;       // 每核基础元素数（512对齐）
    uint64_t blockNum = 0;          // 虚拟block数
    uint64_t ubFormer = 0;          // UB单次处理元素数（256B对齐）
    uint64_t ubLoopOfFormer = 0;    // 首block的UB循环次数
    uint64_t ubTailOfFormer = 0;    // 首block的UB尾部大小
    uint64_t ubLoopOfTail = 0;      // 尾block的UB循环次数
    uint64_t ubTailOfTail = 0;      // 尾block的UB尾部大小
    float scaleValue = 1.0f;        // scale属性
    float shiftValue = 0.0f;        // shift属性
    float lnBaseValue = 0.0f;       // ln(base)，仅 base > 0 时有效
};
```

### 2. tiling 实现（op_host/arch35/<op>_tiling.cpp，完整函数 verbatim）

```cpp
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "bench_no_harness_exp_tiling.h"
#include "../../op_kernel/arch35/bench_no_harness_exp_tiling_data.h"
#include "../../op_kernel/arch35/bench_no_harness_exp_tiling_key.h"

#include <cmath>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape) {
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

// 获取平台信息
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, uint32_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取 shape 和属性信息
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t* totalLength,
    ge::DataType* dataType, float* base, float* scale, float* shift)
{
    // 获取输入 shape
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    *totalLength = static_cast<uint64_t>(inputShapeX.GetShapeSize());

    // 获取 dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();

    // dtype 校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0,
        OP_LOGE(context, "BenchNoHarnessExp: unsupported dtype %d", static_cast<int>(*dataType)),
        return ge::GRAPH_FAILED);

    // 获取属性 (base, scale, shift)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* basePtr = attrs->GetFloat(0);
    const float* scalePtr = attrs->GetFloat(1);
    const float* shiftPtr = attrs->GetFloat(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, basePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, shiftPtr);
    *base = *basePtr;
    *scale = *scalePtr;
    *shift = *shiftPtr;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 计算 UB 单次处理元素数
static uint64_t CalcUbFormer(uint64_t ubSize, ge::DataType dataType)
{
    // 根据 dtype 确定 bufferDivisor
    uint32_t bufferDivisor;
    if (dataType == ge::DT_FLOAT) {
        bufferDivisor = BUFFER_DIVISOR_FP32;
    } else {
        // FP16 / BF16 都是 2 字节
        bufferDivisor = BUFFER_DIVISOR_FP16;
    }

    // ubFormer = ubSize / bufferDivisor，然后按 UB 块大小（32B）对齐
    // 对齐粒度: 32B / elemBytes = 32/2=16 (FP16) 或 32/4=8 (FP32)
    uint32_t elemBytes = (dataType == ge::DT_FLOAT) ? 4U : 2U;
    uint32_t alignElements = UB_BLOCK_SIZE / elemBytes; // FP16: 16, FP32: 8

    uint64_t ubFormer = ubSize / bufferDivisor;
    ubFormer = FloorAlign(ubFormer, static_cast<uint64_t>(alignElements));
    return ubFormer;
}

// 主 Tiling 函数
static ge::graphStatus BenchNoHarnessExpTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取 shape 和属性
    uint64_t totalLength = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    float base = -1.0f;
    float scale = 1.0f;
    float shift = 0.0f;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalLength, &dataType, &base, &scale, &shift) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 获取 Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. 填充 TilingData
    BenchNoHarnessExpTilingData* tiling = context->GetTilingData<BenchNoHarnessExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BenchNoHarnessExpTilingData), 0, sizeof(BenchNoHarnessExpTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalLength = totalLength;
    tiling->coreNum = coreNum;
    tiling->scaleValue = scale;
    tiling->shiftValue = shift;

    // 计算 TilingKey 中的计算分支
    uint32_t computeBranch = 1; // 默认自然指数路径
    if (std::fabs(base - 1.0f) < 1e-6f) {
        computeBranch = 3; // base == 1
    } else if (base > 0.0f) {
        computeBranch = 2; // 自定义底数
        tiling->lnBaseValue = logf(base);
    }

    // 空 Tensor 检查
    if (totalLength == 0) {
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, computeBranch);
        return ge::GRAPH_SUCCESS;
    }

    // 5. 多核切分
    // blockFormer: 每核基础元素数，对齐到 ELEM_ALIGN_FACTOR (512)
    tiling->blockFormer = CeilAlign(CeilDiv(totalLength, static_cast<uint64_t>(coreNum)), ELEM_ALIGN_FACTOR);
    // blockNum: 实际使用的核数
    tiling->blockNum = CeilDiv(totalLength, tiling->blockFormer);
    uint32_t usedCoreNum = static_cast<uint32_t>(tiling->blockNum);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // 6. UB 切分
    tiling->ubFormer = CalcUbFormer(ubSize, dataType);

    // 7. 计算首核和尾核的 UB 循环参数
    // 首核（非尾核）: 处理 blockFormer 个元素
    tiling->ubLoopOfFormer = FloorDiv(tiling->blockFormer, tiling->ubFormer);
    tiling->ubTailOfFormer = tiling->blockFormer - tiling->ubLoopOfFormer * tiling->ubFormer;

    // 尾核: 处理 totalLength - blockFormer * (blockNum - 1) 个元素
    uint64_t tailLength = totalLength - tiling->blockFormer * (tiling->blockNum - 1);
    tiling->ubLoopOfTail = FloorDiv(tailLength, tiling->ubFormer);
    tiling->ubTailOfTail = tailLength - tiling->ubLoopOfTail * tiling->ubFormer;

    // 8. 设置 BlockDim
    context->SetBlockDim(usedCoreNum);

    // 9. 设置 TilingKey
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, computeBranch);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBenchNoHarnessExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BenchNoHarnessExpCompileInfo {}; // 必须定义，入图场景依赖

// Tiling 注册入口
IMPL_OP_OPTILING(BenchNoHarnessExp)
    .Tiling(BenchNoHarnessExpTilingFunc)
    .TilingParse<BenchNoHarnessExpCompileInfo>(TilingParseForBenchNoHarnessExp);

} // namespace optiling
```

（常量 `WORKSPACE_NUM`/`WS_SYS_SIZE`/`ELEM_ALIGN_FACTOR=512`/`UB_BLOCK_SIZE=32`/`BUFFER_DIVISOR_*` 定义在 `<op>_tiling.h`。）

### 增量一（多 dtype 分支持统一字节量纲参数化，BesselI1e 形态）

单算子多 dtype（fp32/fp16/bf16）按 tilingKey 分支共享 TilingData 结构时，各分支只有**元素宽度**不同；把所有切分计算统一到**字节量纲**再按 `elemBytes` 折回元素数，三个分支用同一套公式参数化，避免每分支各写一套魔法数：

- UB 单次可驻留元素数：`maxElemNum = ubSize / (bufferNum * elemBytes)`（输入+输出 buffer 平分 UB）；
- 对齐因子随 dtype 折算：`alignFactor = UB_ALIGN_BYTES / elemBytes`（UB 按 256 字节对齐时，BF16 元素级对齐 = 256/2 = 128）；
- UB 块元素数：`ubFormer = AlignDown(maxElemNum, alignFactor)`（宁少勿超；`ubFormer < 1` 时兜底为一个对齐因子防死循环）；
- 多核切分：`rawCores = CeilDiv(dim0 * elemBits, MIN_TILING_BITS)` 与可用核数取 min；`perCore = CeilDiv(dim0, coreNum)`，`blockFormer = AlignUp(perCore, ELEM_ALIGN_FACTOR)`；
- 尾块字段：首块 `ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer)`、尾数 = `blockFormer - (loop-1)*ubFormer`；尾 block 同法以 `blockTail` 为基；
- 空张量（dim0==0）早退：coreNum=1、其余字段保持调用方 memset 归零，kernel 侧零迭代。

STUB→真实实现的核心函数（`ComputeBranch2Tiling`，verbatim；BF16 分支 elemBytes=2、alignFactor=128）：

```cpp
void ComputeBranch2Tiling(const BesselI1eBranch2TilingInputs& inputs, BesselI1eTilingData& out)
{
    // Branch-2: BF16 路径.
    // elemBytes=2, bufferNum=2, alignFactor=128 (=256/2).
    // 纯 DESIGN 公式（无仿真 caps），ubFormer=49152 等.
    const int64_t dim0            = inputs.dim0;
    const int32_t availableCoreNum = inputs.availableCoreNum;
    const int64_t ubSize          = inputs.ubSize;
    const int32_t elemBytes       = 2;    // BF16
    const int32_t bufferNum       = 2;    // 1 输入 + 1 输出
    const int32_t elemBits        = 16;

    out.dim0 = dim0;

    // 空 Tensor 早返
    if (dim0 == 0) {
        out.coreNum = 1;
        // 其余字段保持调用方 memset 归零
        return;
    }

    // 多核切分
    int64_t rawCores = CeilDiv(dim0 * static_cast<int64_t>(elemBits), MIN_TILING_BITS);
    int32_t coreNum = static_cast<int32_t>(std::min(rawCores, static_cast<int64_t>(availableCoreNum)));
    if (coreNum < 1) { coreNum = 1; }
    out.coreNum = coreNum;

    int64_t perCore  = CeilDiv(dim0, out.coreNum);
    out.blockFormer  = AlignUp(perCore, ELEM_ALIGN_FACTOR);
    out.blockNum     = CeilDiv(dim0, out.blockFormer);

    // UB 切分（统一字节量纲）：maxElemNum = ubSize / (bufferNum * elemBytes)
    int64_t bufferDivisor = static_cast<int64_t>(bufferNum) * static_cast<int64_t>(elemBytes);
    int64_t maxElemNum    = ubSize / bufferDivisor;
    int64_t alignFactor   = static_cast<int64_t>(UB_ALIGN_BYTES) / static_cast<int64_t>(elemBytes); // = 128
    out.ubFormer = AlignDown(maxElemNum, alignFactor);  // 49152
    if (out.ubFormer < 1) { out.ubFormer = alignFactor; }

    // 首 block 循环
    out.ubLoopOfFormerBlock = CeilDiv(out.blockFormer, out.ubFormer);
    out.ubTailOfFormerBlock = out.blockFormer - (out.ubLoopOfFormerBlock - 1) * out.ubFormer;

    // 尾 block
    int64_t blockTail = dim0 - (out.blockNum - 1) * out.blockFormer;
    out.ubLoopOfTailBlock = CeilDiv(blockTail, out.ubFormer);
    out.ubTailOfTailBlock = blockTail - (out.ubLoopOfTailBlock - 1) * out.ubFormer;
}
```

host 侧接入公共分发（替换临时 fallback，verbatim）：

```cpp
    } else if (tilingKey == 2) {
        BesselI1eBranch2TilingInputs inputs{
            dim0,
            static_cast<int32_t>(coreNum),
            static_cast<int64_t>(ubSize)
        };
        ComputeBranch2Tiling(inputs, tilingData);
    } else {
        // Unsupported dtype (ChooseTilingKey 返回 -1)
        return ge::GRAPH_FAILED;
    }
```

反模式提示：host 侧对未实现分支长期挂 fallback 单核 tiling（coreNum=1 的朴素切分）——性能退化且掩盖分支缺失，只应作过渡态；多分支共用 TilingData 时，把"未支持 dtype 返回 `ge::GRAPH_FAILED`"作为兜底分支，避免静默走错误切分。

### 增量二（三层拆分 + TilingUT 三套件 + aclnn 冒烟，softplus_v2_grad 形态）

结构上把 tiling 拆成三层，UT 与 glue 同源编译同一份纯计算函数：

1. **纯计算层** `ComputeTiling(in, out)`（无 CANN 依赖，UT 直接编）：多核切分 → UB 切分 → 分支判定。
2. **Host 流程层** `ComputeHostTiling(...)`：空短路 → 属性校验 → shape 校验 → dtype 校验 → 参数化（elemBytes/minDtypeBits/K）→ 调 ComputeTiling。
3. **glue 层** `TilingFunc(gert::TilingContext*)`：平台查询 → shape/dtype/attr 提取 → 调 ComputeHostTiling → `GetTilingData<T>` 回填 → `SetBlockDim`/`GetWorkspaceSizes`。

纯计算层核心（量纲统一为元素数；fp16 升精度 K 份 fp32 中间缓冲按 `bufferDivisor = bufferNum*elemBytes + kBuf*4` 摊销，verbatim）：

```cpp
void ComputeTiling(const SoftplusV2GradTilingInputs& in, SoftplusV2GradTilingData& out)
{
    out.totalLength = in.dim0;
    if (in.dim0 == 0) {  // 空短路
        out.coreNum = 1;
        out.blockDim = 1;
        out.tilingKey = 1;  // 复用 full-load 路径 + Kernel 侧 totalLength==0 守卫
        return;
    }

    // 多核切分（elemBits = minDtypeBits）
    int64_t coreNum = CeilDiv(in.dim0 * static_cast<int64_t>(in.minDtypeBits), kMinTilingBits);
    coreNum = std::min(coreNum, static_cast<int64_t>(in.availableCoreNum));
    if (coreNum < 1) {
        coreNum = 1;
    }
    int64_t blockFormer = AlignUp(CeilDiv(in.dim0, coreNum), kElemAlignFactor);
    int64_t blockNum = CeilDiv(in.dim0, blockFormer);
    int64_t blockTail = in.dim0 - (blockNum - 1) * blockFormer;

    // UB 切分（extraSize = 0；bufferDivisor 字节口径）
    // fp32 直算 K=0：bufferDivisor = bufferNum * elemBytes
    // fp16 升精度 K=3：bufferDivisor = bufferNum * elemBytes + K * 4（K 份 fp32 中间）
    int64_t bufferDivisor = static_cast<int64_t>(in.bufferNum) * in.elemBytes
                            + static_cast<int64_t>(in.kBuf) * 4;
    int64_t maxElemNum = in.ubSize / bufferDivisor;
    int64_t alignFactor = kAlign256 / in.elemBytes;  // fp32→64 元素；fp16→128 元素
    int64_t ubFormer = AlignDown(maxElemNum, alignFactor);
    if (ubFormer < 1) {
        ubFormer = alignFactor;
    }
    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    // 分支判定：blockFormer == ubFormer 归 full-load（<=）
    int32_t tilingKey = (blockFormer <= ubFormer) ? 1 : 2;

    out.coreNum = static_cast<int32_t>(coreNum);
    out.blockDim = static_cast<int32_t>(blockNum);
    out.blockFormer = blockFormer;
    out.blockTail = blockTail;
    out.ubFormer = ubFormer;
    out.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    out.ubTailOfFormerBlock = ubTailOfFormerBlock;
    out.ubLoopOfTailBlock = ubLoopOfTailBlock;
    out.ubTailOfTailBlock = ubTailOfTailBlock;
    out.tilingKey = tilingKey;
}
```

TilingFunc glue（API 用法完整示例，verbatim）：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // —— 平台查询：availableCoreNum / ubSize（带 null/退化守卫）——
    int32_t availableCoreNum = 1;
    int64_t ubSize = 196608;  // DAV_3510 级默认 UB 预算（运行时实测覆盖）
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC plat(platformInfo);
        uint32_t aiv = plat.GetCoreNumAiv();
        if (aiv > 0) {
            availableCoreNum = static_cast<int32_t>(aiv);
        }
        uint64_t ub = 0;
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        if (ub > 0) {
            ubSize = static_cast<int64_t>(ub);
        }
    }

    // —— shape / dtype 提取（StorageShape 二次取 .GetStorageShape()）——
    const gert::StorageShape* gradShp = context->GetInputShape(0);
    const gert::StorageShape* selfShp = context->GetInputShape(1);
    if (gradShp == nullptr || selfShp == nullptr) {
        return ge::GRAPH_FAILED;  // null_input
    }
    int64_t gradDim0 = gradShp->GetStorageShape().GetShapeSize();
    int64_t selfDim0 = selfShp->GetStorageShape().GetShapeSize();
    int32_t rank = static_cast<int32_t>(gradShp->GetStorageShape().GetDimNum());

    const gert::Tensor* gradT = context->GetInputTensor(0);
    const gert::Tensor* selfT = context->GetInputTensor(1);
    if (gradT == nullptr || selfT == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType gradDt = gradT->GetDataType();
    ge::DataType selfDt = selfT->GetDataType();

    // —— attr 提取（beta=index0, threshold=index1；带默认值兜底）——
    float beta = 1.0f;
    float threshold = 20.0f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* pBeta = attrs->GetFloat(0);
        const float* pThr = attrs->GetFloat(1);
        if (pBeta != nullptr) {
            beta = *pBeta;
        }
        if (pThr != nullptr) {
            threshold = *pThr;
        }
    }

    // —— 完整流程：校验 + 公式 + 回填 TilingData ——
    SoftplusV2GradTilingData td{};
    int32_t status = ComputeHostTiling(gradDim0, selfDim0, rank, gradDt, selfDt,
                                       beta, threshold, availableCoreNum, ubSize, td);
    if (status != kTilingSuccess) {
        return ge::GRAPH_FAILED;
    }

    // 回填 TilingData
    auto* outTd = context->GetTilingData<SoftplusV2GradTilingData>();
    if (outTd != nullptr) {
        *outTd = td;
    }
    context->SetBlockDim(static_cast<uint32_t>(td.blockDim));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;  // 逐元素无归约，无独立 workspace
    }
    return ge::GRAPH_SUCCESS;
}
```

关键 API 事实（均经 CANN 9.1.0 头文件核对）：`GetTilingData<T>()` 要求 tiling 容量 ≥ sizeof(T)（容量由注册的 TilingData 决定）；`GetAttrs()` 返回 `const RuntimeAttrs*`，`GetFloat(index)` 返回 `const float*`（可为 null，需兜底）；`GetInputShape(i)` 返回 `const StorageShape*`，须再 `.GetStorageShape()` 取 `Shape` 后 `GetShapeSize()/GetDimNum()`；dtype 从 `GetInputTensor(i)->GetDataType()` 取；`PlatformAscendC` 构造为 `explicit PlatformAscendC(fe::PlatFormInfos*)`，用 `context->GetPlatformInfo()` 传入。

**TilingUT 三套件结构**：公式用例（逐字段核对切分参数）+ 分支边界用例（full-load/tile-loop 阈值两侧）+ HostFlow 用例（校验短路/空张量/attr 兜底），oracle 按设计公式独立手算、不调用被测实现（可用 `awk` 提取 oracle 函数体 grep `optiling::` 自检独立性）。实现后跑一次 `grep -qi 'do not registe tiling struct' <build.log>`——该告警说明 tiling 结构注册失败，会静默退化。**骨架复用与冒烟**：公共 TilingFunc 实现后，同算子其他分支（如 Branch-1 full-load 简化版）可按同骨架二次实现并回归公共 UT；aclnn 冒烟可用"真实 tiling + kernel 桩"走 GetWorkspaceSize→TilingFunc 全链路验证接口（OpDef stub smoke）。

## 设计决策

- **属性读取用 `GetAttrs()->GetFloat(索引)`**：`TilingContext` 上没有直接的属性接口，属性经 `ExtendedKernelContext::GetAttrs()` 拿 `RuntimeAttrs`，再按注册顺序用 `GetFloat(idx)` 取 `const float*`（空指针必须 OP_CHECK）。add_example 无属性，这条路径是翻 CANN 头文件（runtime_attrs.h / extended_kernel_context.h）确认的。
- **ln(base) 在 tiling 侧预计算**：base>0 走自定义底数分支时 kernel 只需一次乘法（temp×ln(base)），把 `logf` 从 kernel 挪到 host tiling，避免每元素重复算对数。
- **blockFormer 按 512 对齐（ELEM_ALIGN_FACTOR）**：均分后向上取整对齐，保证非尾核处理量是 512 的倍数；尾核独立用 `totalLength − blockFormer×(blockNum−1)` 计算自己的 loop/tail，首尾核参数分开存（ubLoopOfFormer/ubTailOfFormer 与 ubLoopOfTail/ubTailOfTail）。
- **ubFormer 按 32B 块对齐**：`ubSize/bufferDivisor` 后 `FloorAlign` 到 32B/elemBytes 个元素（FP16→16 元素、FP32→8 元素），保证 DataCopy 搬运不跨块。
- **空 tensor 早退**：totalLength==0 时 SetBlockDim(1) 并直接设 TilingKey 返回成功，kernel 侧无需处理 0 元素。
- **TilingKey 用 `ASCENDC_TPL_SEL_PARAM(context, dTypeX, computeBranch)`**：dtype 用 `static_cast<uint32_t>(ge::DataType)` 传入，与 TPL DECL 的参数序一致；弃用废弃的 TILING_KEY_IS 宏。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：目标 Ascend950PR（arch35/DAV_3510，UB 248KB=253952B）；CANN 9.1.0；x86_64 主机构建。
- 目录布局：`op_host/arch35/<op>_tiling.{h,cpp}`（tiling 实现）、`op_kernel/arch35/<op>_tiling_data.h`（TilingData struct，kernel 与 tiling 共享）、`op_kernel/arch35/<op>_tiling_key.h`（TPL DECL/SEL）。
- 相关文件清单（与实际读写一一对应）：`op_host/arch35/bench_no_harness_exp_tiling.cpp`、`op_host/arch35/bench_no_harness_exp_tiling.h`（常量）、`op_kernel/arch35/bench_no_harness_exp_tiling_data.h`、`op_kernel/arch35/bench_no_harness_exp_tiling_key.h`。

## 预防

- 属性索引按 op_def 注册顺序硬编码（GetFloat(0/1/2) 对应 base/scale/shift），注册顺序变更时同步改 tiling 读取索引。
- tiling 内浮点函数用 `logf/std::fabs`（C 名字空间），不要写 `std::logf`——本轮曾报 `'logf' is not a member of 'std'`。
- TPL SEL 注册组合必须覆盖 op_def 声明的全部 dtype 变体（否则部分变体链接失败，另有专卡记录该根因）。

## 验证信号

以下信号来自该算子骨架开发的构建会话（构建级验证，未上板）：

- `bash build.sh --soc=ascend950`：tiling 随 cust_optiling 目标编译通过（首轮 `std::logf` 报错修正为 `logf` 后）（来源平台：950；证据等级：tool_observed）。
- 3 个 dtype 变体 kernel 全部编译 Done，`custom_opp_ubuntu_x86_64.run`（443K）与 `libcust_opmaster_rt2.0.so`/`liboptiling.so` 产出（来源平台：950；证据等级：tool_observed）。

增量信号（两个独立算子的 TilingUT/ST/aclnn 链路验证）：

- BesselI1e 多 dtype 分支（STUB→真实实现，UT 于 x86 宿主 gtest、算子包构建目标 ascend950）：实现前基线 `--gtest_filter='Branch2*'` 10 条 0 通过（STUB 不填字段，实际值全 0）；实现后全量 `[100%] 48 tests passed`（含 Branch-2 10/10、前序分支回归不退化）；构建无 tiling 注册告警（grep 'do not registe tiling struct' 无命中）；oracle 独立性自检（awk 提取 oracle 函数体 grep `optiling::` 无命中）；端到端 BF16 白盒 ST 17/17 PASS（含尾块/尾核/非对齐用例）（来源平台：950；证据等级：tool_observed）。
- softplus_v2_grad 三层拆分（UT 三套件 + 骨架复用 + 冒烟，Ascend950 + CANN 9.1.0）：公共 Tiling 实现后 TilingUT `[  PASSED  ] 22 tests.`（公式 11 + 分支边界 2 + HostFlow 9），build 退出 0；Branch-1（full-load 简化版，同骨架二次实现）落地后 Branch-1 TilingUT 11/11 通过、公共 TilingUT 22/22 回归通过；Branch-1 tiling 落地后的 aclnn 冒烟（真实 tiling + kernel 桩，走 GetWorkspaceSize→TilingFunc 全链路）`OpDef stub smoke summary: 4/4 cases PASSED`、EXIT_CODE=0（fp16/fp32、不同 beta/threshold 组合）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 验证止于编译与打包：tiling 数值的上板正确性（切分参数、TilingKey 分发）未在本轮验证（未验证场景：NPU 执行、精度/性能）。增量形态（BesselI1e/softplus_v2_grad）已由 TilingUT 与 BesselI1e 的白盒 ST 17/17 补齐上板证据。
- 平台参数（UB 253952B、AIV 核数）为运行时经 `PlatformAscendC` 查询，非硬编码；但 512 对齐因子与 bufferDivisor 的性能合理性仅在 950 上按设计推导，其他 SoC 未验证（推断可迁移，需按硬件参数复核）。
- 标量属性仅验证 float 型读取路径；int/string/list 属性用对应 GetXxx 重载，未验证。
- （增量）公式中的常量（`ELEM_ALIGN_FACTOR`、`MIN_TILING_BITS`、`UB_ALIGN_BYTES`）与 `ubFormer=49152` 这类具体数值是算子在给定 UB 容量/dtype 下的推导结果，不是普适值；换 UB 大小或 bufferNum 须按公式重推。多输入/输出算子（bufferNum>2）需改 `bufferDivisor`，未验证。`rawCores = CeilDiv(dim0 * elemBits, MIN_TILING_BITS)` 的核数需求估计是设计近似，未做核数-带宽联合调优。
- （增量）`ubSize` 默认兜底 196608 字节是 DAV_3510 级经验值，仅在 `GetPlatformInfo()` 为 null 时启用；`GetPlatformInfo()` 返回 null 的真实触发条件未复现验证。`kMinTilingBits/kElemAlignFactor/kAlign256/kBufferNum/kFp16KBuf` 为算子 tiling.h 声明的切分常量，换算子需按其 UB 摊销重定（fp16 升精度的 K 份数随中间变量个数变化）；tilingKey=1/2 的分支语义只对该算子的 kernel 约定有效。

**未决项**：

- tiling 切分数值的上板/UT 验证（op_host UT 工程在本轮开工但未完成编译，属后续阶段）。
## 坏实践

- 在 kernel 里每元素计算 `log(base)`——对数与输入无关，应提升到 tiling 侧算一次随 TilingData 下发。
- 空 tensor 不做早退，让 kernel 侧处理 0 元素循环——多一层边界条件，易引入空跑错误。
- 用废弃的 `TILING_KEY_IS` 宏手拼 TilingKey 而不是 `ASCENDC_TPL_SEL_PARAM`。
- （增量）STUB 直接 `return` 不填字段、靠调用方 memset——UT 会"绿"在全部为 0 的期望之外（靠独立 oracle 抓住）。
- （增量）host 侧对未实现分支长期挂 fallback 单核 tiling——性能退化且掩盖分支缺失，只应作过渡态。
- （增量）把切分公式写死在 glue 里，导致 UT 只能测 glue 而无法用独立 oracle 交叉验证公式。
- （增量）coreNum 直接取 `GetCoreNumAiv()` 拉满物理核而不按数据量折算（小 shape 空转大量核）。
- （增量）在 TilingFunc 里直接 `context->GetTilingData<T>()` 解引用不判空、attr 不给默认值兜底（GetFloat 可返回 null）。

## 误诊

- 无。
