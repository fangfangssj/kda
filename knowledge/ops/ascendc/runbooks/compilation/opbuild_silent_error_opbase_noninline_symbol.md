---
type: "Runbook"
title: opbuild 静默失败：infershape 引非内联库符号
description: "广播算子 infershape 直接调用 Ops::Base::InferShape4Broadcast（OPBASE_API 非内联库函数）时，autogen 的 libascend_all_ops.so 不链接 opbase，op_build dlopen 报 undefined symbol 而 cmake 只显 opbuild ops error:（错误体为空）；查 plog 拿真实符号，改自包含手写广播推导后编译通过。"
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# opbuild 静默失败：infershape 引非内联库符号

## 现象

在一台 x86_64 宿主机上（CANN 9.1.0 安装根下无 `aarch64-linux` 目录，目标芯片 Ascend950PR / ascend950 / arch35）首次构建一个按 ops-registry-invoke 模板新建的广播类双输入算子工程（输出 shape = broadcast(x1, x2)，int16/int32/int64）。`infershape` 按官方工具头文件 `op_common/op_host/infershape_broadcast_util.h` 的推荐写法直接调用现成的广播推导函数：

```cpp
#include "op_common/op_host/infershape_broadcast_util.h"
...
static ge::graphStatus InferShape4BenchNoHarnessGcd(gert::InferShapeContext* context)
{
    return Ops::Base::InferShape4Broadcast(context, 2);
}
```

执行 `cmake ..` 配置时构建即失败，输出只有：

```
-- Opbuild generating sources
opbuild ops info:
opbuild ops result: 1
CMake Error at <CANN>/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:783 (message):
  opbuild ops error:
-- Configuring incomplete, errors occurred!
```

`opbuild ops error:` 后面没有任何具体原因。直接手动执行 `tools/opbuild/op_build ./autogen/libascend_all_ops.so ./autogen "--compute_unit=ascend950"` 也返回 `EXIT=1` 且 stdout/stderr 完全无输出，一度无法定位。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓工程（`npu_op_package` + `npu_op_code_gen` + `npu_op_library`）的 host 侧首次 cmake 配置/autogen 阶段。
- API 路径：`op_common/op_host/infershape_broadcast_util.h`（`Ops::Base::InferShape4Broadcast` / `Ops::Base::BroadcastShape` 等 OPBASE_API 函数）；`tools/opbuild/op_build`；plog 目录 `~/ascend/log/debug/plog/`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950，编译目标），CANN 9.1.0，x86_64 宿主机（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——凡 autogen 生成 `libascend_all_ops.so` 并由 op_build dlopen 校验的 CANN 9.x 工程均适用，与具体算子无关。
- 路线优先级：recommended_fastpath（自包含 infershape 写法）。
- 明确不适用/未验证场景：`CeilDiv`/`CeilAlign` 等 `op_common/op_host/util/math_util.h` 内的 `auto` 模板函数不受影响（头文件内定义、隐式 inline，可继续在 tiling 中使用）；aarch64 宿主、CANN 9.1.0 之外的版本未验证。

## 判别方法

**触发**：新增算子工程（或多输入广播类算子）首次 `cmake ..` 报 `opbuild ops result: 1` + `CMake Error at .../fwk_modules/func.cmake:783 (message): opbuild ops error:`，且错误信息为空。

**探针**（opbuild 静默失败时拿真实报错）：

1. 直接跑 op_build 确认无输出但退出码 1：

```bash
source $ASCEND_HOME_PATH/set_env.sh
cd <算子工程>/build
$ASCEND_HOME_PATH/tools/opbuild/op_build ./autogen/libascend_all_ops.so ./autogen "--compute_unit=ascend950"; echo "EXIT_CODE=$?"
```

2. op_build 的报错走 plog 而不是 stdout。清掉旧 plog 后重跑一次，再 grep 最新 plog：

```bash
rm -f ~/ascend/log/debug/plog/*.log 2>/dev/null
# 重跑上面的 op_build 命令
ls -t ~/ascend/log/debug/plog/ | head -2
grep -i "error\|fail" ~/ascend/log/debug/plog/plog-<pid>_*.log | head
```

**观测**（本会话实测）：plog 中出现

```
[ERROR] ASCENDCKERNEL(<pid>,op_build):...[op_build.cpp:49][main]dlopen error : [./autogen/libascend_all_ops.so] : ./autogen/libascend_all_ops.so: undefined symbol: _ZN3Ops4Base20InferShape4BroadcastEPN4gert17InferShapeContextEm
```

即 `Ops::Base::InferShape4Broadcast(gert::InferShapeContext*, unsigned long)` 未定义。

**解释**：autogen 阶段（`compile_ascendc_all_ops.py`）只把算子自己的 def/infershape/tiling 源文件编成 `libascend_all_ops.so`，不链接 opbase 库；`Ops::Base::InferShape4Broadcast` 是声明在 `infershape_broadcast_util.h`、实现在 opbase 库的 `OPBASE_API` 函数（非 inline），于是该 .so 带着未解析符号，`op_build` 一 dlopen 就失败。同目录 `math_util.h` 的 `CeilDiv`/`CeilAlign` 是头文件内 `auto` 模板（隐式 inline），所以 tiling 里用它们不会出这个问题——这也是参考模板 add_example（infershape 只做 `*output_shape = *input_shape`，不调外部函数）能编过的原因。

## 根因

infershape 实现调用了 opbase 库的非内联 `OPBASE_API` 函数 `Ops::Base::InferShape4Broadcast`，而 autogen 产出的 `libascend_all_ops.so` 不链接 opbase，op_build dlopen 校验时符号无法解析。（单句归因：**infershape 引用了 autogen 阶段不可链接的非内联库符号**。）

## 解决配方

infershape 改为自包含实现：只用 `exe_graph/runtime/shape.h` 的 `GetDimNum/GetDim/SetDimNum/SetDim`（头文件内联接口）手写广播推导，不调任何外部库函数。

**修复前**（完整函数，`op_host/bench_no_harness_gcd_infershape.cpp`）：

```cpp
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "op_common/op_host/infershape_broadcast_util.h"

namespace ops {

// 广播算子形状推导：双输入 → 单输出，输出 shape = broadcast(x1, x2)
static ge::graphStatus InferShape4BenchNoHarnessGcd(gert::InferShapeContext* context)
{
    // InferShape4Broadcast(context, inputNum) 支持多输入单输出的广播推导
    return Ops::Base::InferShape4Broadcast(context, 2);
}

IMPL_OP_INFERSHAPE(BenchNoHarnessGcd).InferShape(InferShape4BenchNoHarnessGcd);

} // namespace ops
```

**修复后**（完整函数，同一文件重写）：

```cpp
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "op_common/log/log.h"

namespace ops {

// 广播算子形状推导：双输入 → 单输出，输出 shape = broadcast(x1, x2)
// 手动实现广播规则，自包含不依赖外部库函数
static ge::graphStatus InferShape4BenchNoHarnessGcd(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    size_t dimNum1 = x1Shape->GetDimNum();
    size_t dimNum2 = x2Shape->GetDimNum();
    size_t maxDim = (dimNum1 > dimNum2) ? dimNum1 : dimNum2;

    outShape->SetDimNum(maxDim);
    for (size_t i = 0; i < maxDim; i++) {
        // 从右向左对齐：左侧不足的维度补 1（标量 dimNum=0 时全部补 1）
        int64_t dim1 = (i + dimNum1 < maxDim) ? 1 : x1Shape->GetDim(i - (maxDim - dimNum1));
        int64_t dim2 = (i + dimNum2 < maxDim) ? 1 : x2Shape->GetDim(i - (maxDim - dimNum2));
        // 广播校验：对应维度必须相等或一方为 1
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            OP_LOGE(context, "BenchNoHarnessGcd: shape broadcast failed at dim %zu: %ld vs %ld",
                    i, dim1, dim2);
            return ge::GRAPH_FAILED;
        }
        outShape->SetDim(i, (dim1 == 1) ? dim2 : dim1);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BenchNoHarnessGcd).InferShape(InferShape4BenchNoHarnessGcd);

} // namespace ops
```

广播规则要点：从右向左对齐维度；`GetDimNum()==0`（标量）按 `{1}` 处理；对应维相等或一方为 1，否则 `GRAPH_FAILED`；输出维取非 1 的一侧。

## 设计决策

- 备选方案一是"让 `compile_ascendc_all_ops` 链接 opbase 库"：要改 CANN 工具链的生成脚本，不可控、不可移植，放弃。
- 备选方案二是引用 `infershape_broadcast_util.h` 里其他函数（如 `Ops::Base::BroadcastShape`）：同为 `OPBASE_API` 库函数，问题相同，放弃。
- 选自包含手写：与参考模板 add_example 的"infershape 不调外部函数"风格一致，且 `gert::Shape` 的 `GetDimNum/SetDimNum/GetDim/SetDim/AppendDim` 都是头文件内联接口（已对照 `$ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/shape.h` 确认），autogen 阶段无链接依赖。
- 用三元运算符代替 `std::max`，避免 autogen 编译环境对 `<algorithm>` 的不确定性。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 宿主：x86_64 Ubuntu（g++ 13.3.0，cmake 3.20.5）；CANN 9.1.0（`ASCEND_HOME_PATH` 指向安装根，该安装内只有 `x86_64-linux`、没有 `aarch64-linux`）。
- 目标芯片：ascend950（Ascend950PR，arch35/DAV_3510）。
- 相关文件：`op_host/bench_no_harness_gcd_infershape.cpp`（本次修复对象）、`op_host/arch35/bench_no_harness_gcd_tiling.cpp`（不受影响，仅用 inline 模板工具）、`build/autogen/libascend_all_ops.so`（op_build 的 dlopen 对象）、`~/ascend/log/debug/plog/`（真实报错位置）。

## 预防

- 写广播/多输入算子的 infershape 时，不要直接调用 `infershape_broadcast_util.h` 的 `Ops::Base::*` 现成函数；先确认它是头文件内联还是库符号（`OPBASE_API` 前缀即库符号），否则在 autogen/opbuild 阶段必炸。
- 遇到 `opbuild ops error:` 后面为空的情况，第一反应去 plog（`~/ascend/log/debug/plog/`）grep `ERROR`，不要在 cmake 输出里反复找原因。

## 验证信号

以下信号来自该算子工程首次构建失败的排查与修复会话，均为工具实测：

- 修复前：`cmake ..` 报 `opbuild ops result: 1` + `func.cmake:783 opbuild ops error:`（空错误体）；手动执行 op_build 返回 `EXIT=1` 无输出（来源平台：950；证据等级：tool_observed）。
- 定位探针：plog 中 grep 到 `dlopen error : ./autogen/libascend_all_ops.so: undefined symbol: _ZN3Ops4Base20InferShape4BroadcastEPN4gert17InferShapeContextEm`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一工程 `cmake ..` 配置通过（`-- Opbuild generating sources - done`，生成 cust_optiling/cust_opapi/cust_op_proto/ascendc_kernels 四目标）（来源平台：950；证据等级：tool_observed）。
- 修复后：`cmake --build . --target all binary package` 成功，产出 3 个 kernel 二进制（int16/int32/int64）与 `custom_opp_ubuntu_x86_64.run` 安装包（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该结论在 CANN 9.1.0 + x86_64 宿主 + ascend950 目标上实测；其他 CANN 版本中 `InferShape4Broadcast` 若变为内联或 opbuild 校验方式改变，则不适用（未验证）。
- 手写广播推导覆盖标准 numpy 式双向广播（右对齐、补 1、标量 `{1}`）；不支持自动升维语义之外的特例（如显式 reshape 语义），未遇到也不处理。
- aarch64 宿主上 plog 路径与库名可能不同（未验证，推断可迁移）。

**未决项**：

- opbase 库符号清单与 autogen 链接范围的完整对照（哪些 `Ops::Base::*` 可安全使用）未系统梳理，复现时可 `nm -D` 各库逐一确认。
## 坏实践

- 看到 cmake 里 `func.cmake:783` 报错就以为是 cmake 配置本身的问题，反复改 `CMakeLists.txt` 的 include 路径——真实错误在 op_build 的 plog 里，与 cmake 无关。
- 在 infershape 里"图省事"调用工具头文件里的现成 `OPBASE_API` 函数而不看它是声明还是实现。

## 误诊

- "op_build 没有输出说明它把日志吞了，需要 strace"：实际它写 plog，`ASCEND_GLOBAL_LOG_LEVEL=1` + 查 plog 目录即可拿到完整 dlopen 错误。
