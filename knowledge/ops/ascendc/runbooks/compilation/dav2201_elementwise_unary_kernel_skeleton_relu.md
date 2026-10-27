---
type: "Runbook"
title: "DAV_2201 element-wise unary kernel 骨架 多核"
description: "在 DAV_2201（Ascend 910B3）上用 ops-direct-invoke 实现单入单出 element-wise unary 算子（以 ReLU fp16 为例）的完整 kernel 骨架：运行时取核数→尾核 tailing→TQue 双缓冲→DataCopyPad Ext 搬运→AscendC::Relu 矢量计算→GlobalTensor 显式偏移。"
tags: ["task.implementation", "topic.compute.elementwise.activation", "topic.implementation.pipeline", "topic.tiling", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# DAV_2201 element-wise unary kernel 骨架 多核

## 现象

在 Ascend 910B3 / CANN 9.0.0 上用 ops-direct-invoke 工作流开发单入单出 element-wise unary 算子（ReLU，`y=max(0,x)`，fp16 in/out）。需要一套可复用的 kernel 骨架：运行时动态取核数、尾核 tailing、双缓冲 TQue 流水、DataCopyPad Ext 搬运、GlobalTensor 显式偏移避免内部位置跟踪问题。本卡记录该骨架的终版实现（经 Reviewer Round 1 复审 PASS 93/100、16/16 用例 bit-exact）。

## 适用场景

- **工程形态**：ops-direct-invoke Kernel 直调（`op_kernel/<op>_kernel.asc` + `op_host/<op>.asc` + CMakeLists.txt 双 target）。
- **API 路径**：`AscendC::TPipe` / `TQue<TPosition::VECIN|VECOUT, DOUBLE_BUFFER>` / `InitBuffer` / `AllocTensor`/`DeQue`/`EnQue`/`FreeTensor` / `DataCopyPad`+`DataCopyExtParams`+`DataCopyPadExtParams<T>` / `GlobalTensor::SetGlobalBuffer` / `AscendC::Relu`（Memory 矢量版）/ `GetBlockIdx`；host 侧 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)`。
- **验证过的平台与 CANN 版本**：Ascend 910B3（SocVersion `ASCEND910B`，NpuArch `DAV_2201`，`--npu-arch=dav-2201`），CANN 9.0.0。对应注册平台 token `a2`。
- **适用粒度**：`operator_family`——单入单出 element-wise unary 算子族（ReLU/Gelu/Tanh/Sigmoid 等）在 DAV_2201 上的 kernel 骨架可直接套用，改 compute 那一行与 tiling 常量即可。
- **路线优先级**：`recommended_fastpath`——双缓冲 TQue + DataCopyPad Ext 是 DAV_2201 上 element-wise unary 的推荐写法。
- **明确不适用或未验证场景**：需要广播/规约/多输入的算子不适用本骨架（广播维 stride=0、规约轴语义均未在此实现）；FP32/BF16 未实现（本例仅 half，多 dtype 需模板化）；A3 未实测——A3 未验证，推断可迁移（共享 DAV_2201）。

## 完整实现

实现位于 `op_kernel/relu_kernel.asc` + `op_kernel/relu_tiling.h`。host 侧 tiling 计算（`op_host/relu.asc` 的 `main()`）见 `## 系统信息`。本骨架在 default shape (32768) 端到端 bit-exact。

## 系统信息

- **toolchain**：bisheng（CANN 9.0.0），CMakeLists.txt 双 target（`add_executable(relu op_host/relu.asc)` + `add_library(relu_ops SHARED ...)` TORCH_LIBRARY 扩展）；`--npu-arch=dav-2201`（小写带连字符，见相关 build 卡）。
- **目标平台**：Ascend 910B3 / NpuArch DAV_2201 / UB 192KB。
- **目录布局**：`operators/ReLU/{op_kernel/relu_kernel.asc, op_kernel/relu_tiling.h, op_host/relu.asc, op_host/data_utils.h, op_extension/{relu_torch.cpp,register.cpp}, scripts/{gen_data.py,golden.py,verify_result.py,test_torch.py}, CMakeLists.txt, run.sh}`。
- **相关文件清单**（与本 trace 实际读写一一对应）：
  - `op_kernel/relu_tiling.h`：`TILE_LENGTH=4096`、`DOUBLE_BUFFER=2`、`ReLUTilingData{blockNum,totalLength,numPerCore,tailNumLastCore}`。
  - `op_kernel/relu_kernel.asc`：`KernelReLU` 类 + `relu_kernel` 核函数入口（`## 完整代码`）。
  - `op_host/relu.asc`：`main()` 用 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 运行时取 `availableCoreNum`，按 `totalLength` 算 `totalTiles/tilesPerCore/blockNum/numPerCore/tailNumLastCore`，填 `ReLUTilingData` 后 `KernelCall`。（注：模板 `main()` 的 `totalLength` 写死，见相关 build 卡——bin 通路仅覆盖单一 shape。）
  - `scripts/verify_result.py`：`np.allclose(rtol=1e-3, atol=1e-3)`，FP16 精度标准。

## 完整代码

`op_kernel/relu_tiling.h`（verbatim，trace file_read 终版）：

```cpp
#pragma once

#include <cstdint>
#include <algorithm>

// ============================================================================
// ReLU Tiling 设计 (DESIGN.md §2.2)
// ============================================================================
// half 类型: sizeof=2, elemBits=16
// UB=192KB, bufferNum=2 → maxElemNum = (192*1024*8) / (2*16) = 49152
// alignFactor = 256/2 = 128
// ubFormer = (49152 / 128) * 128 = 49152 (但受 TILE_LENGTH 限制)
// ============================================================================

// 单次搬运/处理的数据量（元素数），对齐到 256B (128 half)
constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 2;

// Tiling 结构体 (与 DESIGN.md §2.2 对齐)
struct ReLUTilingData {
    uint32_t blockNum;          // block 数量
    uint64_t totalLength;       // 元素总数量
    uint64_t numPerCore;        // 每个核处理元素数 (对齐到 TILE_LENGTH)
    uint64_t tailNumLastCore;   // 最后一个核处理元素数
};
```

`op_kernel/relu_kernel.asc`（verbatim，trace file_read 终版，Round 1 复审 PASS 版）：

```cpp
// ============================================================================
// Ascend C Kernel 实现 - ReLU 算子（纯 kernel，不含 host 代码）
// ============================================================================
// 算子: y = max(0, x), dtype=half (float16)
// API:  AscendC::Relu (Memory 矢量计算版, DAV_2201 支持 half)
// ============================================================================

#include "kernel_operator.h"
#include "relu_tiling.h"

class KernelReLU {
public:
    __aicore__ inline KernelReLU(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ ReLUTilingData* tiling)
    {
        this->tiling = tiling;
        blockIdx_ = AscendC::GetBlockIdx();

        // Guard against empty input (blockNum == 0 causes uint32_t underflow)
        if (tiling->blockNum == 0) {
            tileNum = 0;
            return;
        }

        if (blockIdx_ < tiling->blockNum - 1) {
            total = tiling->numPerCore;
            tileNum = total / TILE_LENGTH;
            tailTileElementNum = TILE_LENGTH;
        } else {
            total = tiling->tailNumLastCore;
            tileNum = (total + (TILE_LENGTH - 1)) / TILE_LENGTH;
            tailTileElementNum = total - TILE_LENGTH * (tileNum - 1);
        }

        // Compute base offset for this block
        xBasePtr = (__gm__ half *)((__gm__ char *)x + blockIdx_ * tiling->numPerCore * sizeof(half));
        yBasePtr = (__gm__ half *)((__gm__ char *)y + blockIdx_ * tiling->numPerCore * sizeof(half));

        // UB Buffer: VECIN for input, VECOUT for output (双 Buffer)
        pipe_->InitBuffer(inQueue, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
        pipe_->InitBuffer(outQueue, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t count = (i == tileNum - 1) ? tailTileElementNum : TILE_LENGTH;
            CopyIn(i, count);
            Compute(count);
            CopyOut(i, count);
        }
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t count)
    {
        AscendC::LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
        // Create a GlobalTensor for this specific tile with explicit offset
        __gm__ half *tileSrc = xBasePtr + tileIdx * TILE_LENGTH;
        AscendC::GlobalTensor<half> srcGm;
        srcGm.SetGlobalBuffer(tileSrc, count);
        AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padInParams{false, 0, 0, static_cast<half>(0)};
        AscendC::DataCopyPad(xLocal, srcGm, copyInParams, padInParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<half> xLocal = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueue.AllocTensor<half>();

        // ReLU: y = max(0, x) — Memory 矢量版, DAV_2201 原生支持 half
        AscendC::Relu(yLocal, xLocal, count);

        outQueue.EnQue<half>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t count)
    {
        AscendC::LocalTensor<half> yLocal = outQueue.DeQue<half>();
        // Create a GlobalTensor for this specific tile with explicit offset
        __gm__ half *tileDst = yBasePtr + tileIdx * TILE_LENGTH;
        AscendC::GlobalTensor<half> dstGm;
        dstGm.SetGlobalBuffer(tileDst, count);
        AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(dstGm, yLocal, copyOutParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe_;
    const __gm__ ReLUTilingData* tiling;
    uint32_t blockIdx_;
    __gm__ half *xBasePtr;
    __gm__ half *yBasePtr;

    // UB Queues: VECIN for input, VECOUT for output
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> outQueue;

    uint32_t total;
    uint32_t tileNum;
    uint32_t tailTileElementNum;
};

// 核函数入口: relu_kernel (保持 _kernel 后缀)
extern "C" __global__ __vector__ void relu_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KernelReLU op(&pipe);
    op.Init(x, y, (__gm__ ReLUTilingData*)tiling);
    op.Process();
}
```

## 设计决策

- **TQue 模板 bufferNum 必须与 InitBuffer 的 DOUBLE_BUFFER 一致**：早期版本写 `TQue<VECIN, 1>` 配 `InitBuffer(inQueue, DOUBLE_BUFFER, ...)`——模板参数 `1` 把队列深度上限锁死成 1 个 pending tensor，`InitBuffer` 传 `DOUBLE_BUFFER=2` 也无法生效，双缓冲名存实亡，MTE2 搬运与 Vector 计算无法重叠。改为 `TQue<VECIN, DOUBLE_BUFFER>`（=2）后队列深度与 init 一致，双缓冲才真正生效。**这是本族算子最易踩的坑**：以为 InitBuffer 传 DOUBLE_BUFFER 就有双缓冲，忽略 TQue 模板参数才是深度上限。
- **DataCopyPad 用 Ext 版而非窄 uint16 版**：早期用 `DataCopyPad(xLocal, srcGm, {1, static_cast<uint16_t>(...), 0, 0}, ...)`（`DataCopyPadParams` 字段为 uint16）。改为 `DataCopyExtParams`（uint32）+ `DataCopyPadExtParams<half>`：参数范围更大、类型更安全，避免大 tile 时 uint16 截断。性能一致，纯类型安全升级。
- **GlobalTensor 用 SetGlobalBuffer 显式偏移而非依赖内部位置跟踪**：每 tile 新建 `GlobalTensor` 并 `SetGlobalBuffer(tileSrc, count)` 显式指定该 tile 的 GM 起址与长度，避免 `GlobalTensor` 内部位置跟踪的已知问题。代价是每 tile 构造一次 `GlobalTensor`，对 element-wise 轻计算算子可接受。
- **blockNum==0 early return 防空输入 uint32_t 下溢**：原 `if (blockIdx_ < tiling->blockNum - 1)` 在 `blockNum==0`（空输入）时 `uint32_t` 下溢成 `UINT32_MAX`，所有 block 都走尾核分支越界。加 `if (tiling->blockNum == 0) { tileNum = 0; return; }` 提前返回。空输入是设计需容忍的边界。
- **核数运行时取、不硬编码**：host `main()` 用 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 取 `availableCoreNum`，据此算 `tilesPerCore/blockNum`；kernel 用 `GetBlockIdx()`。不硬编码 blockDim。尾核处理 `tailNumLastCore`，`tileNum` 对尾核用 `(total + TILE_LENGTH-1)/TILE_LENGTH` 向上取整。
- **跨 tile 流水未做**：`Process` 中 `CopyIn→Compute→CopyOut` 严格串行，未把下一 tile 的 `CopyIn` 与当前 tile 的 `CopyOut` 重叠。对 ReLU（compute 极轻，DataCopy 主导）影响有限；若搬到大计算量算子，应补跨 tile 重叠。Reviewer Round 1 标注为残余限制。
- **仅 half / 不模板化**：DESIGN.md §0.1 明确仅 float16，非缺陷（ReLU 是 `max(0,x)` 比较，无数值变换，FP16 满足精度）。多 dtype 需 `template<typename T>` 化，本例未做。

## 预防

- 写 TQue 时，模板 bufferNum 参数与 `InitBuffer` 的 depth 参数必须同名同值（都用 `DOUBLE_BUFFER`），不要一个写 `1` 一个写 `DOUBLE_BUFFER`。
- 用 DataCopyPad 时优先 `DataCopyExtParams` + `DataCopyPadExtParams<T>` 的 Ext 组合，不用窄 uint16 的旧 params。
- element-wise 算子每 tile 用 `SetGlobalBuffer` 显式偏移，不依赖 GlobalTensor 内部位置跟踪。

## 验证信号

- 可执行端 default shape (32768) 端到端 bit-exact：`Verification PASSED! Shape: (32768,)` / `Max diff: 0.0`（来源平台：a2；证据等级: tool_observed）
- 编译通过：`Built target relu` / `Built target relu_ops`（来源平台：a2；证据等级: tool_observed）
- Reviewer Round 1 复审 PASS 93/100，确认 TQue 双缓冲/DataCopyPad Ext/blockNum 防护均已修复（来源平台：a2；证据等级: artifact_verified）
## 性能影响

本 trace 无 msprof/profiling 实测数据（性能轨无 tool_observed 信号），不报任何绝对或相对性能数字。Reviewer 关于"双缓冲修复后性能改善"的叙述为 agent_reported，不进验证信号。

## 失效边界

- 本卡验证平台为 910B3（a2）。DAV_2201 为 A2/A3 共享架构，骨架在 A3 上推断可迁移，但 A3 未实测——A3 未验证，推断可迁移。
- 广播/规约/多输入算子不适用本骨架（未实现广播维 stride=0、规约轴语义）。
- FP32/BF16 未实现（需模板化，本例未做，未验证）。
- host `main()` 的 `totalLength` 在模板里写死，bin 通路仅覆盖单一 shape（见相关 build 卡）；多 shape 须改 host 或走 PyTorch 通路。