---
type: "Runbook"
title: TilingFunc 未写 TilingData 致输出全 0
description: ST 全量 FAIL 且输出全 0（恒等直通/常量输出探针仍全 0）而编译通过时，先查 op_host TilingFunc 是否为 stub——只 SetBlockDim+workspace 不写 TilingData、不 SetTilingKey，kernel 收到全 0 tiling（dim0=0）直接空转；须按 gert API 正确接线 ComputeTiling 结果。复现增量（CANN 9.1.0）：更早的形态——纯桩连 TilingData 结构体都不引用时结构体未编入 op_host，构建 exit 0 且无 tiling 注册告警，只能靠验收门 grep 检出；附"结构体定义+kernel 注册+host 引用"三件套清单与两条构建自检门。
tags: [task.debugging, topic.tiling, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# TilingFunc 未写 TilingData 致输出全 0

## 现象

bessel_i1e 算子（逐元素数学函数）Branch-0（tilingKey=0，FP32）kernel 实现完成后，`./operators/build.sh BesselI1e_package` 编译通过（ascend950，CANN 9.0.0-beta.2），`cd tests/aclnn && ./run.sh` 跑 14 个白盒 ST 用例（oracle 为 scipy.special.i1e，rtol=atol=1e-5）：12/14 FAIL，且所有失败用例 `got 0`（如 `[0-A_f32_main_block_aligned] Mismatch at [0]: got 0, expected 0.119728379`），只有两个空张量用例 PASS；大 shape 用例 `01_f32_normal_128x256` 得 `got 6.92682648`（后判明是未初始化 GM 残留值）。

为区分"计算链错"还是"数据没流动"，把 VF 计算换成恒等直通（output = input）和常量输出（从 UB 常量表取 32.0f）两种探针，结果**仍然全 0**——说明数据通路或调度根本没执行，问题不在算术。进一步用 `grep -r "TilingFunc\|ComputeBranch0Tiling\|ComputeTiling\|ChooseTilingKey" op_host/` 检查 host 侧，发现 `op_host/bessel_i1e_host.cpp` 的 TilingFunc 是一个最小 stub：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(1);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

它只设置 BlockDim 与 workspace，从不调用已实现的 `ComputeBranch0Tiling`/`ComputeTiling`（tiling host 模块里都有），也不写 TilingData、不设 TilingKey。kernel 侧 `GET_TILING_DATA` 拿到全 0 的 tiling（dim0=0），Process 判空直接返回，输出保持测试初始化的 0。另查 `build/autogen/op_proto.cc` 为空（`namespace ge {}`），确认 autogen 不会替你补 tiling。

## 适用场景

- 工程形态：算子仓（operators/ 工程包）aclnn 自定义算子开发 + 白盒 ST（run.sh）。
- API 路径：op_host 侧 `gert::TilingContext`（GetInputShape/GetInputTensor/GetPlatformInfo/GetTilingData<T>/SetTilingKey/SetBlockDim）+ `platform_ascendc::PlatformAscendC`（GetCoreNumAiv/GetCoreMemSize）+ kernel 侧 `REGISTER_TILING_DEFAULT`/`GET_TILING_DATA`。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"kernel 编译过、ST 全 0 失败"的自定义算子（kernel 未执行/切分层问题，非精度逻辑）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：输出非全 0 但数值错的精度问题；多输出/动态 shape 的 TilingFunc 细节未在本例覆盖。

## 根因

op_host 的 TilingFunc 是 stub：不把 `ComputeTiling`/`ComputeBranch0Tiling` 的结果写入 `context->GetTilingData<T>()`，也不 `SetTilingKey`。TilingData 缓冲区保持全 0，kernel 侧 `GET_TILING_DATA` 读到 dim0=0 → Process 空转 → 输出全 0。编译与二进制生成都正常，故障只在运行时静默表现。

## 判别方法

1. 现象侧：ST 全 FAIL + `got 0` + 编译通过 + 空张量用例 PASS（空张量不依赖 tiling 数值）。
2. 探针侧：把 kernel 计算换成恒等直通/常量输出——仍全 0 即排除计算链，锁定"kernel 未执行或数据未流动"。
3. host 侧：grep TilingFunc 与 tiling 计算函数的调用关系；查 `build/autogen/op_proto.cc` 是否为空（为空说明没有 autogen 兜底）。
4. 确认 TilingFunc 是否写 `GetTilingData<T>()` 并 `SetTilingKey`。

## 解决配方

修复前（stub，verbatim）：

```cpp
#include "../op_kernel/bessel_i1e_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
// NOTE: TilingFunc is a minimal stub. The complete tiling implementation
// (ComputeBranch0Tiling etc.) is in bessel_i1e_tiling_host.cpp but is not
// yet wired here. End-to-end tests will fail until the tiling is integrated.
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(1);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

修复后（正确接线，verbatim 取自修复编辑）：

```cpp
#include <cstring>
#include "../op_kernel/bessel_i1e_tiling.h"
#include "bessel_i1e_tiling_host.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // Get input tensor (for dtype) and shape (for dim0)
    const auto* xShape = context->GetInputShape(0);
    const auto* xTensor = context->GetInputTensor(0);
    if (xShape == nullptr || xTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t dim0 = xShape->GetOriginShape().GetShapeSize();
    auto dtype = xTensor->GetDataType();

    // Get platform info (available core count, UB size)
    auto* platInfo = context->GetPlatformInfo();
    if (platInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platInfo);
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum < 1) { coreNum = 1; }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // Choose tiling key via dtype → tilingKey (DESIGN.md §5.5)
    int32_t tilingKey = ChooseTilingKey(dtype);

    // Compute TilingData per branch
    BesselI1eTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));

    if (tilingKey == 0) {
        BesselI1eBranch0TilingInputs inputs{
            dim0,
            static_cast<int32_t>(coreNum),
            static_cast<int64_t>(ubSize)
        };
        ComputeBranch0Tiling(inputs, tilingData);
    } else {
        // Branch-1 (FP16) / Branch-2 (BF16): not yet implemented; fallback to simple tiling
        tilingData.dim0 = dim0;
        tilingData.coreNum = 1;
        tilingData.blockFormer = (dim0 > 0) ? dim0 : 0;
        tilingData.blockNum = (dim0 > 0) ? 1 : 1;
        tilingData.ubFormer = (dim0 > 0) ? dim0 : 0;
        tilingData.ubLoopOfFormerBlock = (dim0 > 0) ? 1 : 0;
        tilingData.ubTailOfFormerBlock = 0;
        tilingData.ubLoopOfTailBlock = (dim0 > 0) ? 1 : 0;
        tilingData.ubTailOfTailBlock = 0;
    }

    // Write tiling data to context
    auto* td = context->GetTilingData<BesselI1eTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *td = tilingData;

    // Set block dim (kernel launch grid)
    context->SetBlockDim(static_cast<uint32_t>(tilingData.blockNum));
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));

    // Workspace: none needed for this element-wise operator
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
```

## 设计决策

- 为什么用 `GetInputShape(0)->GetOriginShape().GetShapeSize()` 取元素数：`GetInputShape` 返回 `const gert::StorageShape*`，元素数须走 OriginShape 的 GetShapeSize（本会话第一轮用不存在的 `GetInputDataType`/`GetCoreNum`/`shape->GetDim(i)` 直接形态编译失败后回退，见误诊）。
- 为什么 dtype 走 `GetInputTensor(0)->GetDataType()` 而不是猜 shape 来源：TilingKey 按 dtype 分支（FP32→0），tensor 接口是可编译且语义正确的来源。
- 为什么平台信息走 `PlatformAscendC` 包装：`GetCoreNumAiv()` 与 `GetCoreMemSize(CoreMemType::UB)` 提供核数与 UB 容量，供 tiling 公式使用；直接在 TilingContext 上找同名方法不存在。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：ascend950 目标，CANN 9.0.0-beta.2；构建 `./operators/build.sh BesselI1e_package`；测试 `tests/aclnn/run.sh`（14 用例）。
- 相关文件：`op_host/bessel_i1e_host.cpp`（TilingFunc 接线）、`op_host/bessel_i1e_tiling_host.{h,cpp}`（ComputeTiling/ComputeBranch0Tiling/ChooseTilingKey 已有实现）、`op_kernel/bessel_i1e_tiling.h`（TilingData 结构）、`build/autogen/op_proto.cc`（确认为空）。
- 头文件参考：`$ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/tiling_context.h`、`.../asc/include/tiling/platform/platform_ascendc.h`。

## 验证信号

以下信号来自 bessel_i1e Branch-0 的 ST 联调循环：

- 修复前：run.sh 12/14 FAIL 全 `got 0`（如 0-A: got 0, expected 0.119728379），恒等直通与常量输出探针仍全 0，`01_f32_normal_128x256` got 6.92682648（未初始化值）（来源平台：950；证据等级：tool_observed）。
- 根因确认：`grep` 显示 TilingFunc 与 ComputeBranch0Tiling 无调用关系；`build/autogen/op_proto.cc` 仅 `namespace ge {}`（来源平台：950；证据等级：tool_observed）。
- 修复后：`./operators/build.sh BesselI1e_package` SUCCESS；run.sh 行为质变——kernel 真正执行（随后暴露 ACL 507035 向量核异常，属下一层独立问题，见对应卡）；恒等直通验证时 `0-A got 0.327972412`（= 输入值）、`0-G got 3.5`（= 标量输入）、`0-K got 1000`（= 输入）等 10/14 用例正确回读输入（来源平台：950；证据等级：tool_observed）。
- 独立验收复核：verifier 重新构建 + grep 检查通过，确认修复前状态为 build PASS / 12-14 FAIL（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只闭合"tiling 未接线导致全 0"；接线后暴露的 507035（GM 越界/UB 超额）与 kernel 挂起是独立根因，见同分片对应卡。
- TilingFunc 内 `AscendC::UB_SIZE` 常量不可用（不存在），UB 大小必须从 `GetCoreMemSize` 取；本轮也确认平台真实 UB 值未打印（数值未知）。
- 多输入/属性驱动的 tiling 分支未覆盖。
- 与"Host 缺 SetDataSize 致 TilingData 截断"（库内 host_setdatasize_tilingdata 卡）现象相近但机制不同：本卡是 TilingFunc 从未写 TilingData（整块全 0），该卡是写了但框架按默认 data_size 只搬结构体头部——判别时先 grep TilingFunc 是否调用 tiling 计算函数，再查 SetDataSize。

**未决项**：

- 修复后 tiling 公式与 kernel 消费端之间仍有二层缺陷（ubFormer>blockFormer 的越界、ubFormer 过大），见同分片 507035 两卡——tiling 接线本身闭环。
## 坏实践

- host 侧留 stub TilingFunc 却期望 kernel 端到端可测（"kernel 自身编译无误"不等于可验证）。
- 构建 autogen 产物未确认前假设"系统会自动补 tiling"。

## 误诊

- 全 0 出现后先怀疑 VF 常量表加载/StoreAlign 写出与 UB 常量表布局（把常量改成按 VL 重复填充、换成 c32Ptr 常量输出探针多轮），又怀疑 TQue EnQue/DeQueue 保数性——恒等/常量探针仍全 0 才转向 host tiling。
- 第一轮实现 TilingFunc 用了臆造 API（`context->GetInputDataType(0)`、`context->GetCoreNum()`、`GetTilingData(0)` 的 SetRawData 形态），host 编译失败（`no matching function`）后一度回退 stub 并在开发日志记"tiling 需后续接线"；正确形态（GetOriginShape().GetShapeSize/GetInputTensor/PlatformAscendC/GetTilingData<T>）在后续修复会话才落地。

## 后续增量（2026-09-17，并行策展批：纯桩不引用结构体——未编入 host 的静默形态与三件套验收门）

另一独立轨迹并入独立 trace（softmax 基准算子 opbuild 工程，Ascend 950 + CANN 9.1.0）：同一"stub TilingFunc"起点在本形态下**连运行期症状都没有**——host `TilingFunc` 只做 `context->SetBlockDim(1)` + workspace 置 0，不获取也不填充任何 tiling 结构体时，**结构体根本未编入 op_host**：

- 基线探测：`./build.sh` 退出码 0、构建日志 `grep -ci 'do not registe tiling struct'` 计数为 0（opbuild 的 tiling 注册告警不触发），但 `grep -rq '<XxxTilingData>' op_host/` 返回 `GREP_NOT_FOUND`——缺项不可靠构建结果发现。

**TilingData 落地三件套**（缺一处即不闭环）：

1. **结构体定义**（`op_kernel/*_tiling.h`）：普通 C++ struct + 默认成员初始化即可，host 与 kernel 都能 include；
2. **kernel 侧注册**（`op_kernel/*_kernel.cpp` 入口）：`REGISTER_TILING_DEFAULT(XxxTilingData)` + `GET_TILING_DATA(tilingData, tiling)`（框架必备样板，勿当算法逻辑删除）；
3. **host 侧编入**（`op_host/*_host.cpp` 的 TilingFunc）：`context->GetTilingData<XxxTilingData>()` 取指针并填充——桩阶段 memset 清零也算引用（本形态的最小修复）：

```cpp
// after（桩 TilingFunc 仅占位获取 TilingData 并清零，把结构体编入 host；文件头补 #include <cstring>）
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BenchHarnessSoftmaxTilingData* td = context->GetTilingData<BenchHarnessSoftmaxTilingData>();
    if (td != nullptr) {
        (void)memset(td, 0, sizeof(BenchHarnessSoftmaxTilingData));
    }
    context->SetBlockDim(1);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

**构建自检门**（修后必须全过）：

```bash
# 门 1：build 退出码 0 且无 tiling 注册告警
( set -o pipefail; cd operators && ./build.sh <pkg> 2>&1 | tee /tmp/build.log ) >/dev/null 2>&1
rc=$?; test $rc -eq 0 && ! grep -qi 'do not registe tiling struct' /tmp/build.log
# 门 2：host 真实引用结构体
grep -rq '<XxxTilingData>' operators/<pkg>/op_host
```

复现增量验证信号：

- 基线：`./build.sh` 退出码 0、`do not registe tiling struct` 计数 0，但 `grep -rq 'BenchHarnessSoftmaxTilingData' op_host` 返回 `GREP_NOT_FOUND`（来源平台：950 构建；证据等级：tool_observed）。
- 修后构建：`./build.sh` 输出 `SUCCESS`、`EXIT=0`；验收门 1 PASS（exit=0、无 tiling-struct 告警）、门 2 PASS（grep 命中 `GetTilingData<BenchHarnessSoftmaxTilingData>()` 与 memset 两处）；kernel 注册保持 intact（来源平台：950；证据等级：tool_observed）。
- 真机冒烟：aclnn 两段式调用（fp16/fp32/bf16，n=16）3/3 PASSED，RUN_EXIT=0（来源平台：950 上板；证据等级：tool_observed）。

复现增量边界与坏实践：

- `do not registe tiling struct` 告警的触发条件未在本轨迹复现（基线与修后计数均为 0）——不能依赖该告警检出"未编入"；门 2 的 grep 是可靠判据。host 侧 struct 可见性/双定义约束见 [TilingData 须 host 与 kernel 各定义一份卡](../compilation/tilingdata_host_kernel.md)；注册核验探针见 [TilingData 注册核验卡](../compilation/register_tiling_default_kernel_entry_mocker_probe.md)。
- 坏实践：host `TilingFunc` 桩只 `SetBlockDim(1)` 而不引用 tiling 结构体——构建干净通过但 tiling 链路不闭环；把 kernel 入口的 `InitSocState/REGISTER_TILING_DEFAULT/GET_TILING_DATA/PipeBarrier` 四行当"算法逻辑"删掉。
- TilingData 各字段数值语义的执行验证属后续真实切分实现（本增量只闭合骨架与门）；纯 Kernel 直调（不经 opbuild/tiling 序列化）未验证。