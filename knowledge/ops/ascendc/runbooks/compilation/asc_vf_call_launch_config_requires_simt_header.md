---
type: "Runbook"
title: "dim3 未声明：显式引入 SIMT 公共头"
description: "混合 SIMD/SIMT 核函数调用 asc_vf_call 时，显式包含 common_functions.h 以暴露 dim3 别名和正确重载。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# dim3 未声明：显式引入 SIMT 公共头

## 现象

在 Ascend950 Flash 直调工程中编译一个“外层 `__global__ __aicore__ __vector__` 核函数调用 `asc_vf_call<__simt_vf__>`”的混合实现时，`gather.asc` 已经包含 `kernel_operator.h`，但在 `dim3(threadNum)` 处停止编译。远程 CANN 9.0.1 编译器原样报告：

```text
gather.asc:102:9: error: use of undeclared identifier 'dim3'; did you mean 'cce::dim3'?
        dim3(threadNum),
        ^
__clang_cce_simt.h:29:40: note: 'cce::dim3' declared here
typedef __device_builtin__ struct dim3 dim3;
```

当时代码已经有完整的 SIMT VF 与外层 block kernel，失败不是模板参数、gather 地址计算或 host schema 引起，而是调用点所需类型与重载没有进入当前翻译单元的可见范围。

## 适用场景

- 工程形态：Flash kernel 直调工程；Torch 扩展 host 与 Ascend C kernel 同一 `.asc` 翻译单元。
- API 路径：`__global__ __aicore__ __vector__` 外层核函数 → `asc_vf_call<__simt_vf__>(dim3, args...)`。
- 验证过的平台与版本：Ascend950PR_957c（平台注册值 `950`），CANN 9.0.1，`NpuArch=dav-3510`。
- 适用粒度：`operator_family`；适用于需要从 Vector 外层启动 SIMT VF 的混合编程算子，不依赖 gather 数学语义。
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。
- 明确不适用 / 未验证场景：纯 SIMT 独立样例、无需 `asc_vf_call` 的核函数，以及 CANN 9.0.1 之外版本均未验证。

## 根因

CANN 9.0.1 混合 ASC 翻译单元仅含 kernel_operator.h 时，global dim3 alias 与 launch-config asc_vf_call overload 不可见；调用前须包含 SDK SIMT 公共/聚合头。cce::dim3 提示与 __vector__ qualifier 不是等价修复。

## 解决配方

先显式包含声明别名与重载的公共头，再使用三维聚合初始化调用带线程维度的 `asc_vf_call`。不要仅凭 `kernel_operator.h` 已被包含就假设 `dim3` 全局别名可见。

### before

revision: pre-fix trace write（Git revision 未在轨迹中记录）
path: `operators/gather/gather.asc`
symbol: `gather_kernel<X_T, IDX_T>`
line span: 96–105

```cpp
template <typename X_T, typename IDX_T>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR params,
    int64_t outputTotal, int32_t ndim, int32_t dim, uint32_t threadNum)
{
    asc_vf_call<gather_simt<X_T, IDX_T>>(
        dim3(threadNum),
        (__gm__ const X_T*)x, (__gm__ const IDX_T*)index, (__gm__ X_T*)y,
        (__gm__ const int64_t*)params, outputTotal, ndim, dim);
}
```

pre-fix include region（revision: pre-fix trace write；path: `operators/gather/gather.asc`；symbol: translation-unit includes；line span: 31–38）：

```cpp
#include <tuple>
#include <algorithm>
#include <vector>
#include <cstring>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
```

### after

revision: `76b4eab`（轨迹记录的短 Git revision）
path: `operators/gather/gather.asc`
symbol: `gather_kernel<X_T, IDX_T>`
line span: 111–119

```cpp
template <typename X_T, typename IDX_T>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR params,
    int64_t outputTotal, int32_t ndim, int32_t dim, uint32_t threadNum)
{
    asc_vf_call<gather_simt<X_T, IDX_T>>(
        dim3{threadNum, 1, 1},
        (__gm__ const X_T*)x, (__gm__ const IDX_T*)index, (__gm__ X_T*)y,
        (__gm__ const int64_t*)params, outputTotal, ndim, dim);
}
```

fixed include（revision: `76b4eab`, path: `operators/gather/gather.asc`, line 46, symbol: translation-unit include）：

```cpp
#include "simt_api/common_functions.h"   // dim3 alias + asc_vf_call(dim3, ...) -> cce::async_invoke
```

## 设计决策

1. 选择显式包含 `simt_api/common_functions.h`，因为它同时解决类型别名与带线程维度的 `asc_vf_call` 重载可见性，机制闭环比只把调用点改成 `cce::dim3` 更完整。
2. 轨迹中曾尝试改成纯 SIMT 并增加 `--enable-simt`，但该路线随后触发 `kernel_tpipe.h` 缺失的新编译错误，因此被放弃。
3. 当前方案属于 `safety_fallback`：它保证混合 SIMT 路线正确编译和执行，但轨迹中的任务级评测显示其性能明显低于优化基线。

## 系统信息

- toolchain：CANN 9.0.1，BiSheng/ASC，`dav-3510`。
- 目标平台：Ascend950PR_957c。
- 相关文件：`operators/gather/gather.asc`、`CMakeLists.txt`；诊断时读取 CANN 安装中的 `simt_api/common_functions.h` 与 `__clang_cce_simt.h`。
- 代码提交：轨迹只给出实现提交短哈希 `76b4eab`，未给出完整 40 位 revision 或固定 blob URL。

## 性能影响

最终混合 SIMT fallback 在同一任务、同一 950 平台上相对评测基线平均约 `0.02x`（来源平台：950；证据等级：tool_observed）。这是成对的任务内路线代价，不是可迁移的通用性能预期。

## 预防

- 写 `asc_vf_call<vf>(dim3{...}, ...)` 前，确认翻译单元显式包含 `simt_api/common_functions.h`。
- 同时核对可见的 `asc_vf_call` 是否真有 `dim3` 首参重载，不要只确认函数名存在。
- 对 `cce::async_invoke` 的调用契约按“线程维度写 launch info，VF 只接收余下 `Args...`”理解，避免给 VF 错加 `dim3` 业务形参。
- 修复后至少同时验证编译链接、真机测试与完整任务评测，不能把仅消失的语法错误当成功闭环。

## 验证信号

来源说明：这条知识来自一次 Ascend950 Flash gather 直调工程的首次远程编译、CANN 头文件核对和修复后全链路验证。

- 首次远程构建稳定报告 `use of undeclared identifier 'dim3'`，并提示唯一可见声明是 `cce::dim3`（来源平台：950；证据等级：tool_observed）。
- 远程读取 CANN 9.0.1 `simt_api/common_functions.h`，观察到 `using dim3 = cce::dim3` 与 `asc_vf_call(dim3 threadNums, Args&&...)` 的完整定义（来源平台：950；证据等级：tool_observed）。
- 最终远程构建完成 `Linking ASC shared library libop_gather.so` 与 `Built target op_gather`（来源平台：950；证据等级：tool_observed）。
- 修复后真机 pytest 为 50/50 通过，完整 cann-bench 为 20/20 通过且精度误差为零（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 本结论验证于 CANN 9.0.1；其他 CANN 版本可能调整 include 聚合关系，未验证，推断可迁移。
- 纯 SIMT 独立工程的入口、编译选项和可用头文件与混合工程不同，不应直接套用。
- 该修复只解决声明与重载可见性，不证明 SIMT fallback 是性能最优路线。

## 坏实践

- 为绕过一个未限定类型名错误，直接把整个 Flash 翻译单元切成 `--enable-simt`；这会改变编译模型，并可能把局部头文件问题扩大为工具链头文件不可用。
- 只把 `dim3` 改成 `cce::dim3`，却不核对带 `dim3` 首参的 `asc_vf_call` 重载是否可见；类型名通过后仍可能匹配错误调用路径。

## 误诊

最初把报错理解成“应该改用纯 SIMT kernel 入口”，但工具输出只证明 `dim3` 未声明；远程头文件证据表明缺失的是公共头暴露的别名和重载，而不是 gather 算法必须切换执行模型。