---
type: "Runbook"
title: "HardSwish 逐元素 kernel 的 TQue 双缓冲实现配方"
description: "ascend910b 上 HardSwish 算子的可复用实现：2 个 TQue in-place、单 tile 快速路径、5 指令序列、remainder 均衡分块；15 测试点全部 Pass。"
tags: ["task.implementation", "topic.compute.elementwise.activation", "topic.implementation.pipeline", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# HardSwish 逐元素 kernel 的 TQue 双缓冲实现配方

## 现象

在 cann-launch-camp HardSwish 竞赛模板（`HardSwish_problem_267_template/code`，目标平台 ascend910b / CANN 8.5.0）下，需要实现 `y = x * relu6(x+3)/6` 的逐元素 HardSwish 算子，并在 cannjudge 评测平台 15 个测试点上逼近第一名时延。本卡记录最终验证通过的可复用实现配方（不包含第一名专属常量、shape 列表或绝对单跑分数）。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发（`op_kernel` 核函数 + `op_host` tiling/注册 + `tiling_key_hard_swish.h` 模板参数头），经 cmake/make 编译为 `.run` 包提交评测。
- **API 路径**：`kernel_operator.h`；`AscendC::TPipe`/`TQue`/`TBuf`/`LocalTensor`/`GlobalTensor`/`DataCopyPad`/`DataCopyParams`；host 侧 `platform_ascendc::PlatformAscendC`、`gert::TilingContext`、`OpDef` + `AddConfig("ascend910b")`；模板参数 `ASCENDC_TPL_DATATYPE_DECL`/`ASCENDC_TPL_DATATYPE_SEL`。
- **验证过的平台与 CANN 版本**：ascend910b 竞赛评测平台（CANN 8.5.0，15/15 Pass）。本地 Ascend910B3（CANN 9.0.0）仅用于编译，无法直接运行 ascend910b kernel（见 `## 失效边界`）。`platforms: [a2]`。
- **适用粒度**（transfer_scope）：`operator_family`——配方对 `relu6`/`hard_swish`/`hard_sigmoid` 等逐元素激活（含 clamp/min/max + 乘加链）算子通用；`blockFactor`/`tileLength` 由 host 按实测 shape 与 UB 反推，不写死。
- **路线优先级**：`recommended_fastpath`——2 TQue in-place + 单 tile 快速路径为推荐主路线；`banned` 的 TBuf+PipeBarrier 小数据分支见 `## 失败路线`。
- **不适用或未验证场景**：非 ascend910b/A2 架构（910A/310P 等）未验证；含跨位置聚合（reduction/conv）的算子不适用本逐元素流水配方；CANN < 8.5.0 未验证。

## 完整实现

实现要点（逐条对应 `## 完整代码`）：

1. **2 个 TQue，in-place 计算，无 tmp buffer**：仅 `inQueueX (VECIN)` 与 `outQueueY (VECOUT)` 两个 `TQue<..., BUFFER_NUM=2>`；中间量 `relu6(x+3)/6` 直接写回 `yLocal`，输出再与 `xLocal` 相乘写入 `yLocal`，省去第三个 tmp TQue。UB 占用 = `2 × tile × sizeof(DT) × BUFFER_NUM`，tile 可取 `ub_size / (sizeof(DT) × BUFFER_NUM × 2)`。
2. **单 tile 快速路径**：当 `blockLength <= tileLength` 时跳过分块循环，单次 `CopyIn → Compute → CopyOut`，减少分支与循环开销（小 shape / 多核切分后尾核命中）。
3. **5 指令计算序列**：`Adds(y,x,3) → Maxs(y,y,0) → Mins(y,y,6) → Muls(y,y,1/6) → Mul(y,x,y)`。等价于 `y = x · min(max(x+3,0),6) / 6`。关键：`-inf` 输入时 `Mul(-inf, 0)=NaN`，符合题目 `-inf → NaN` 要求（避免官方参考实现里 `Maxs(x,-3)` 把 `-inf→-3→0` 的错误路径）。
4. **remainder 均衡分块**（host）：`blockFactor = CeilDiv(length, num_cores_aiv)` 再 `AlignDown(elementPerBlock)`，`blockDim = CeilDiv(length, blockFactor)`；kernel 用 `remainderLength = max(0, length - blockFactor*blockIdx)`、`blockLength = min(remainderLength, blockFactor)` 处理尾核，比裸 CeilDiv 负载更均衡。
5. **DataCopyPad + DataCopyParams 字段赋值**：`blockLen` 用 `static_cast<uint16_t>(curLength * sizeof(DT))`（`DataCopyParams.blockLen` 为 `uint16_t`，聚合初始化易触发 narrowing），`blockCount=1, srcStride=0, dstStride=0`；CopyIn 用 `{false,0,0,0}` 默认 padding，CopyOut 不传 pad 参数。

## 系统信息

- **工具链 / CANN 版本**：CANN 9.0.0（本地编译）；评测平台 CANN 8.5.0。
- **目标平台**：ascend910b（A2，竞赛评测平台，`AddConfig("ascend910b")`）。
- **目录布局**：
  - `code/op_kernel/hard_swish.cpp` — 核函数（`KernelHardSwish` 模板类 + `hard_swish` `__global__` 入口）
  - `code/op_kernel/hard_swish_tiling.h` — tiling 结构体
  - `code/op_kernel/tiling_key_hard_swish.h` — tiling_key 模板参数（仅 DATATYPE）
  - `code/op_host/hard_swish.cpp` — Tiling 函数 + InferShape/InferDataType + `OpDef` 注册
  - `code/CMakeLists.txt` — `set(ASCEND_COMPUTE_UNIT ascend910b)` + `npu_op_package`
- **相关文件清单**（与 trace 读写一一对应）：`op_kernel/hard_swish.cpp`、`op_kernel/hard_swish_tiling.h`、`op_kernel/tiling_key_hard_swish.h`、`op_host/hard_swish.cpp`、`op_host/CMakeLists.txt`、`op_kernel/CMakeLists.txt`、`CMakeLists.txt`。
- **构建/安装命令**：`cd build && rm -rf * && cmake -DASCEND_CANN_PACKAGE_PATH=<cann> .. && make HardSwish_ascend910b && make package` → 生成 `custom_opp_ubuntu_aarch64.run`，目标机 `./custom_opp_ubuntu_aarch64.run` 安装到 `opp/vendors/custom`。

## 完整代码

以下三文件 verbatim 取自 trace 中对该算子仓终版的文件读取事件（核函数、host、tiling 头）。

### op_kernel/hard_swish.cpp

```cpp
// Kernel侧核函数实现
#include "kernel_operator.h"

#include "hard_swish_tiling.h"
#include "tiling_key_hard_swish.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr float HS_VAL3 = 3.0f;
constexpr float HS_VAL6 = 6.0f;
constexpr float HS_VAL0 = 0.0f;
constexpr float HS_INV6 = 1.0f / 6.0f;

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}

__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return CeilDiv(x, align) * align;
}

__aicore__ inline uint32_t MinU32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t MaxU32(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

template <class DT_X>
class KernelHardSwish {
public:
    __aicore__ inline KernelHardSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t inputTileLength, uint32_t blockFactor) {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t elementPerBlock = 32 / sizeof(DT_X);

        uint32_t remainderLength = (length > blockFactor * blockIdx) ? (length - blockFactor * blockIdx) : 0;
        this->blockLength = (remainderLength > blockFactor) ? blockFactor : remainderLength;
        this->tileLength = MaxU32(1, inputTileLength);
        this->alignedTileLength = MaxU32(elementPerBlock, AlignUp(this->tileLength, elementPerBlock));

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockFactor * blockIdx, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockFactor * blockIdx, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->alignedTileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->alignedTileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->blockLength == 0) return;
        if (this->blockLength <= this->tileLength) {
            CopyIn(0, this->blockLength);
            Compute(this->blockLength);
            CopyOut(0, this->blockLength);
            return;
        }
        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            const uint32_t curLength = MinU32(this->tileLength, this->blockLength - offset);
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(curLength * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, {false, 0, 0, 0});
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t curLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template AllocTensor<DT_X>();

        AscendC::Adds(yLocal, xLocal, static_cast<DT_X>(HS_VAL3), curLength);
        AscendC::Maxs(yLocal, yLocal, static_cast<DT_X>(HS_VAL0), curLength);
        AscendC::Mins(yLocal, yLocal, static_cast<DT_X>(HS_VAL6), curLength);
        AscendC::Muls(yLocal, yLocal, static_cast<DT_X>(HS_INV6), curLength);
        AscendC::Mul(yLocal, xLocal, yLocal, curLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(curLength * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t blockLength = 0;
    uint32_t tileLength = 1;
    uint32_t alignedTileLength = 1;
};

template <typename DT_X>
 __global__ __aicore__ void hard_swish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(HardSwishTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSwishTilingData, tiling_data, tiling);
    KernelHardSwish<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.tileLength, tiling_data.blockFactor);
    op.Process();
}
```

### op_kernel/hard_swish_tiling.h

```cpp
// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct HardSwishTilingData {
    uint32_t length;
    uint32_t tileLength;
    uint32_t blockFactor;
};
```

### op_kernel/tiling_key_hard_swish.h

```cpp
// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(HardSwish,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT),
    ),
);
```

### op_host/hard_swish.cpp

```cpp
// Host侧Tiling实现
#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/hard_swish_tiling.h"
#include "../op_kernel/tiling_key_hard_swish.h"

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;

    static uint32_t AlignDown(uint32_t value, uint32_t align) {
        if (align == 0) {
            return value;
        }
        return value / align * align;
    }

    static uint32_t CeilDiv(uint32_t x, uint32_t y) {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        const uint32_t typeSize = static_cast<uint32_t>(dtype_size_x);
        const uint32_t elementPerBlock = std::max<uint32_t>(1, 32 / typeSize);
        const uint32_t bytesPerElement = typeSize * BUFFER_NUM * 2;
        uint32_t maxTileByUb = static_cast<uint32_t>(ub_size / bytesPerElement);
        maxTileByUb = AlignDown(maxTileByUb, elementPerBlock);
        maxTileByUb = std::max<uint32_t>(maxTileByUb, elementPerBlock);

        uint32_t blockFactor = CeilDiv(length_x, static_cast<uint32_t>(num_cores_aiv));
        blockFactor = AlignDown(blockFactor, elementPerBlock);
        blockFactor = std::max<uint32_t>(blockFactor, elementPerBlock);
        uint32_t blockDim = CeilDiv(length_x, blockFactor);

        uint32_t tileLength = std::min(maxTileByUb, blockFactor);
        tileLength = AlignDown(tileLength, elementPerBlock);
        tileLength = std::max<uint32_t>(tileLength, elementPerBlock);

        HardSwishTilingData *tiling = context->GetTilingData<HardSwishTilingData>();
        tiling->length = length_x;
        tiling->tileLength = tileLength;
        tiling->blockFactor = blockFactor;

        context->SetBlockDim(blockDim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class HardSwish : public OpDef {
    public:
        explicit HardSwish(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(HardSwish);
}  // namespace ops
```

## 设计决策

- **为何选 2 TQue in-place 而非 3 TQue（带 tmp）**：参考 ops-nn `arch35` 官方实现用 input+tmp+output 共 3 个 TQue × 2 buffer = 6 份 UB，tile 上限 `ub_size/6`；本实现把中间量 `relu6(x+3)/6` 与最终 `x·...` 都写入 `yLocal`，省去 tmp，UB 占用 `ub_size/4`，tile 增大约 50%，减少分块循环次数。代价是 `xLocal` 在整条 Compute 链中需保持有效（`FreeTensor` 在 `EnQue(yLocal)` 之后），但因双缓冲已重叠，未观察到回退。
- **为何选 5 指令序列 `Adds→Maxs→Mins→Muls→Mul`**：比官方 6 指令（含 `Maxs(x,-3)` 做 `-inf` 修复）少 1 条；且 `Mul(-inf, 0)=NaN` 天然满足题目 `-inf → NaN` 语义，而官方 `Maxs(x,-3)` 会把 `-inf→-3→0`（错误）。被放弃的备选 `Adds→Mins→Maxs→Mul→Muls`（先把 `Muls` 提前）在 trace 中讨论过，最终采用 `Muls` 在 `Mul` 之前的版本（见 verbatim 代码），因 `Muls(y,y,1/6)` 与 `Mul(y,x,y)` 共用 `yLocal` 作目的操作数，先后顺序对寄存器压力更友好。
- **为何选 DataCopyPad + DataCopyParams 字段赋值**：trace 早期用聚合初始化 `DataCopyParams{1, uint32_t(...), 0, 0}` 触发 narrowing（`blockLen` 为 `uint16_t`），改字段赋值并显式 `static_cast<uint16_t>` 规避。`isPad=false` 让硬件自动处理尾块 padding，CopyIn/CopyOut 行为对称。被放弃的 `DataCopyExtParams`（5 字段含 `rsv=0`）底层同一条 MTE 指令、性能等价，本实现选字段更少的 `DataCopyParams`。
- **为何选 remainder 均衡分块**：裸 `CeilDiv(length, num_cores)` 会让前 N-1 核满载、末核只跑尾数，尾核空闲但 blockDim 已定；本实现 `blockFactor = AlignDown(CeilDiv(length, cores), elementPerBlock)`，`blockDim = CeilDiv(length, blockFactor)`，使各核 `blockLength` 更接近（末核 `remainderLength` 与满核差距缩小），降低多核尾延迟。
- **单 tile 快速路径**：当分块后 `blockLength <= tileLength`（小 shape 或大核数下的尾核）直接单次搬运-计算-回写，跳过循环边界判断，对小数据点时延敏感场景关键。
- **当前方案属哪档路线**：`recommended_fastpath`（2 TQue 双缓冲 + 单 tile + 5 指令 + remainder 均衡）。

## 性能影响

最终 TQue 双缓冲路线相对被放弃的 TBuf+PipeBarrier 小数据分支路线：在最小数据测试点（test1）上，TQue 路线快约 12%（来源平台：a2/910b 竞赛平台，CANN 8.5.0；证据等级：tool_observed）。详见 `## 失败路线`。绝对时延只进 manifest provenance，不写入正文。

## 失败路线

- **`banned`：TBuf + PipeBarrier 单核小数据快速路径**。曾尝试为 `≤1024` 元素的数据点另开一条 `TBuf + PipeBarrier` 单核路径（无 `InitBuffer` 双 buffer、无 `GetBlockNum/GetBlockIdx`、无 remainder 计算，`DataCopyPad → PipeBarrier → 5 指令 → PipeBarrier → DataCopyPad`），意图消除 TQue EnQue/DeQue 与双 buffer 分配开销。实测在竞赛平台最小数据点上反而比纯 TQue 双缓冲路线慢约 12%（回退为纯 TQue 后该点时延改善）。判定 `PipeBarrier` 同步开销抵消了 `TBuf`/`EnQue` 节省，对小数据点净负向。该 banned 路线的 kernel 写入事件在 markdown 转写中被截断为占位符，完整逐字代码未从 trace 事件恢复，故此处仅给结构描述与相对性能信号，不内联代码块。
- **`banned`：`ASCENDC_TPL_BOOL_DECL` 编译期 BOOL 分支**：见 DRAFT-build-01，平台 CANN 8.5.0 不支持，100% Compile Error。

## 预防

- 引入任何 `ASCENDC_TPL_*` 较新模板宏前，先在目标平台 CANN 版本做最小可编译样例，勿以本地版本编译通过为准（见 DRAFT-build-01）。
- 逐元素激活算子优先 2 TQue in-place + 单 tile 快速路径；勿为小数据点引入 TBuf+PipeBarrier 分支（净负向）。
- `DataCopyParams` 字段为 `uint16_t`，`blockLen` 需显式 `static_cast<uint16_t>`，勿用聚合初始化隐式 narrowing。

## 验证信号

本问题与验证来自 HardSwish 算子仓在 ascend910b 竞赛评测平台的提交轮次；信号来自竞赛平台 harness 客观输出。

- 最终 TQue 双缓冲路线在竞赛 ascend910b 平台 15/15 测试点 Pass、输出错误占比 0.00%（来源平台：a2/910b 竞赛平台，CANN 8.5.0；证据等级：tool_observed）
- 最终路线相对 TBuf+PipeBarrier 小数据路线在最小数据点快约 12%（来源平台：a2/910b 竞赛平台，CANN 8.5.0；证据等级：tool_observed）
- 本地 CANN 9.0.0 编译 `make HardSwish_ascend910b` 产出 `[100%] Built target HardSwish_ascend910b` + `custom_opp_ubuntu_aarch64.run` 打包成功（来源平台：a2/910B3 本地机，CANN 9.0.0；证据等级：tool_observed）
## 失效边界

- 配方在 ascend910b / CANN 8.5.0（竞赛评测平台）实测 Pass；910A/310P/310B 等其他架构未验证（未验证，推断可迁移到同 A2 架构、UB/核数相近的 ascend910b 系列）。
- 本地 Ascend910B3（CANN 9.0.0）无法直接运行 ascend910b 编译的 kernel（出现 `sqSwStatus=0x23` 内核崩溃，疑为平台/SoC 不匹配，未追到机制层根因），故本地仅作编译验证，所有上板 Pass 信号来自竞赛平台。
- CANN < 8.5.0 是否支持 `DataCopyPad`/`ASCENDC_TPL_DATATYPE_DECL` 等本配方依赖的 API 未独立验证。