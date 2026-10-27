---
type: "Runbook"
title: "ASC_VDEQ_SIZE 未声明 — 勿直引 vector_compute.h"
description: "CANN 9.2.0 的 vector_compute.h 是声明源头但不是自包含用户入口；直接 include 会缺内部前置定义，改用 asc_simd.h umbrella。"
tags: ["task.debugging", "topic.implementation.api_layer.c_api", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# ASC_VDEQ_SIZE 未声明 — 勿直引 vector_compute.h

## 现象

为修复 `asc_float2int64_rn` 未声明，direct-launch Ascend C 工程把 API 声明源头 `c_api/vector_compute/vector_compute.h` 直接加入 `cummin.asc`。第二次构建中原来的函数未声明错误消失，但 `make` 仍退出 2，并在该叶子头内部产生 7 个新错误：

```text
vector_compute.h:70:45: error: use of undeclared identifier 'ASC_VDEQ_SIZE'
vector_compute.h:70:80: error: use of undeclared identifier 'ASC_VDEQ_SIZE'
vector_compute.h:71:24: error: use of undeclared identifier 'ASC_VDEQ_SIZE'
vector_compute.h:1166:44: error: use of undeclared identifier 'ASC_C_API_MRGSORT_ELEMENT_LEN'
vector_compute.h:1171:44: error: use of undeclared identifier 'ASC_C_API_MRGSORT_ELEMENT_LEN'
vector_compute.h:1176:46: error: use of undeclared identifier 'ASC_C_API_MRGSORT_ELEMENT_LEN'
vector_compute.h:1181:46: error: use of undeclared identifier 'ASC_C_API_MRGSORT_ELEMENT_LEN'
```

这不是调用点的 count、指针类型或 kernel 数据流错误；编译器尚未走到这些语义检查，就已在被直接包含的头内部缺少前置定义。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`vector_compute.h`、`asc_simd.h`、`ASC_VDEQ_SIZE`、`ASC_C_API_MRGSORT_ELEMENT_LEN` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖同一工具链版本、头文件图和 C-API 调用组合。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

`vector_compute.h` 虽然承载 `asc_float2int64_rn` 的函数声明，却在 CANN 9.2.0/dav-2201 的 include 图中依赖 umbrella 先引入的内部前置定义；把“声明源头”误当成“自包含的用户入口”会让 `ASC_VDEQ_SIZE` 与 `ASC_C_API_MRGSORT_ELEMENT_LEN` 在解析叶子头时未定义。

## 解决配方

保留调用点与数据流不变，只把叶子声明头替换为 public C-API umbrella。不要手工定义缺失宏，也不要直接包含 `impl/` 内部头来拼凑顺序。

### before

```cpp
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "c_api/vector_compute/vector_compute.h"  // asc_float2int64_rn (KB K8 sources: header role)

using namespace AscendC;
```

### after

```cpp
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "c_api/asc_simd.h"  // C-API umbrella (asc_float2int64_rn; toolchain fallback receipt proven)

using namespace AscendC;
```

## 设计决策

1. 先用知识检索区分“API 声明源文件”与“支持的用户入口”；知识只确认了声明源头，没有证明叶子头可直接 include。
2. 因公共入口仍是 gap，才使用安装工具链做最小 `bisheng -E` probe；`asc_simd.h` 能展开完整依赖并看到目标函数。
3. 最终采用 `asc_simd.h`，因为它随后通过完整工程 compile/link；手工补宏或包含内部 `impl/` 头会把实现细节固化进用户源码，属于 `banned` 做法。

## 系统信息

- Toolchain：CANN 9.2.0，bisheng，`dav-2201`；
- 平台：A3，SocVersion `Ascend910_9362`；
- 涉及文件：`cummin.asc`、`artifacts/cummin/build/build_r2.log`、`artifacts/cummin/toolchain_fallback/asc_float2int64_include_probe/probe_asc_simd.asc`、`artifacts/cummin/build/build_r3.log`；
- 证据限制：只存在 trace snapshot/短 revision，没有 40 位完整 Git revision 或公开 blob URL。

## 完整代码

以下完整最小函数来自轨迹真实 probe。它只证明 umbrella 在预处理阶段能提供目标声明；完整 compile/link 由真实工程 Build-3 补齐。

```cpp
#include "c_api/asc_simd.h"
__global__ __aicore__ void test_kernel(__ubuf__ int64_t* dst, __ubuf__ float* src) {
    asc_float2int64_rn(dst, src, 256);
}
```

## 预防

- API 卡的 `sources: role=header` 只能说明声明/实现来源，不能自动推出该文件可作为用户翻译单元的直接 include。
- 叶子头内出现成组未定义宏时，先检查 public umbrella/include order；不要在调用点反复改 count 或 pointer cast。
- 用两级证据闭环：最小预处理/编译 probe 定位入口，真实工程 compile/link 排除 probe 范围过窄。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：编译器报 ASC_VDEQ_SIZE 与 ASC_C_API_MRGSORT_ELEMENT_LEN 未声明。（来源平台：a3；证据等级: tool_observed）。
- 修复后：cmake exit=0、make exit=0。（来源平台：a3；证据等级: tool_observed）。
- 修复后：运行输出 BASIC TEST PASSED。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 结论严格绑定 A3/dav-2201/CANN 9.2.0；a2、950 和其他 CANN 版本的公共入口与叶子头自包含性未验证。
- 该闭环不证明所有 C-API 都必须包含同一个 umbrella；每个版本应以公开入口和最小 compile probe 为准。
- 本次运行只覆盖一个 fp32 连续轴样例；其余 dtype、非连续路径与 formal evaluation 未执行。

## 坏实践

- 把文档或 source metadata 中的 header 路径直接复制为用户 include。
- 手工 `#define ASC_VDEQ_SIZE` 或 `ASC_C_API_MRGSORT_ELEMENT_LEN` 让编译继续；这会绕过工具链头图契约。
- 直接包含 `impl/` 内部头补齐前置定义；内部 include 顺序不是稳定用户 API。

## 误诊

Build-2 的错误出现在 `vector_compute.h` 内部，容易被误认为 CANN 安装损坏。实际是把叶子声明头当成了公共入口；同一安装下改用 umbrella 后完整工程可成功构建。