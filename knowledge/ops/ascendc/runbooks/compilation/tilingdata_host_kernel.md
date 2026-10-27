---
type: "Runbook"
title: "TilingData 须 host 与 kernel 各定义一份"
description: "新增 TilingData struct 只写进 kernel 侧 struct.h 时 host 编译报 does not name a type；须在 host 可见头内同 namespace 重新定义一份，字段与 kernel 侧逐字一致，仿 SIMT 双定义模式。该机制同样适用于常量：host tiling 引用仅 kernel 头定义的常量（如 BLOCK_BYTES）报 not declared，须在 host 侧 tiling 头自定义同语义常量。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-18T08:28:30Z"
updated_at: "2026-09-17T00:58:06Z"
platforms: ["950"]
---

# TilingData 须 host 与 kernel 各定义一份

## 现象

为 `max_pool3d_grad`（Ascend950 / arch35）新增 separable tiling 路径时，把新 TilingData struct `MaxPool3DGradSeparableTilingData` 只写进了 kernel 侧头文件 `pooling/max_pool3d_grad/op_kernel/arch35/max_pool3d_grad_struct.h`，并在 host 侧 tiling 类头 `pooling/max_pool3d_grad/op_host/arch35/max_pool3d_grad_separable_tiling.h` 里直接 `#include "../../op_kernel/arch35/max_pool3d_grad_struct.h"` 以图取到该 struct。

构建 host 库（`bash build.sh --soc=ascend950 --ophost --noexec -j8`）报：

```
.../op_host/arch35/max_pool3d_grad_separable_tiling.h:28:5: error: 'MaxPool3DGradSeparableTilingData' does not name a type; did you mean 'MaxPool3DGradSeparableTiling'?
.../op_host/arch35/max_pool3d_grad_separable_tiling.h:22:50: error: 'MaxPool3DGradSeparableTilingData' was not declared in this scope; did you mean 'MaxPool3DGradSeparableTiling'?
.../op_host/arch35/max_pool3d_grad_separable_tiling.h:22:84: error: no matching function for call to 'gert::TilingContext::GetTilingData<<expression error>>()'
```

## 适用场景

- **工程形态**：算子仓开发，在 `op_host/arch35/` 下新增 tiling 类，其构造函数用 `context_->GetTilingData<MyTilingData>()` 取一个**本算子新定义的** TilingData struct 指针。
- **API 路径**：`GetTilingData<T>` 模板实例化要求 `T` 在 host 翻译单元内是完整类型；相关头为 `pool3d_grad_struct_common.h`（host 侧 struct 汇总头）与 `max_pool3d_grad_struct.h`（kernel 侧 struct 汇总头）。
- **验证平台与 CANN 版本**：Ascend950（arch35），`bash build.sh --soc=ascend950 --ophost --noexec -j8`。
- **适用粒度**：`operator_family`——任何"新 tiling 路径引入新 struct、且 struct 被同时用于 host tiling 计算与 kernel 计算"的算子都适用，不限于 pooling。
- **路线优先级**：`recommended_fastpath`——host/kernel 双定义是既有 SIMT 路径的标准模式，新路径照做即可。
- **不适用或未验证场景**：若新 struct 只在 kernel 侧使用、host 不 `GetTilingData` 它，则无需在 host 头定义；若能改公共 host 头 `pool3d_grad_struct_common.h` 也可只在那里定义一次（本开发会话 该公共头跨算子共享、不能改）。a2/a3 未构建，未验证。

## 根因

AscendC 算子的 TilingData struct 需要同时被两个编译器看到：
1. **host 编译器**（编译 `op_host/*.cpp`，经 `pool3d_grad_struct_common.h` 取 struct）；
2. **kernel 编译器**（编译 `op_kernel/*.cpp`，经 `max_pool3d_grad_struct.h` 取 struct，该头还会 include kernel-only 头如 `kernel_operator.h`）。

kernel 侧 `max_pool3d_grad_struct.h` 含 kernel-only 构造，host 翻译单元不能 include 它。因此一个被 host 与 kernel 共用的 struct 必须**在两处分别定义**：一次在 host 可见的头（如 `pool3d_grad_struct_common.h` 或 host tiling 类头），一次在 kernel 可见的头（`max_pool3d_grad_struct.h`）。两边定义在同一 `namespace Pool3DGradNameSpace` 内、字段完全一致，靠**翻译单元隔离**避免重定义——host 翻译单元只看到 host 那份，kernel 翻译单元只看到 kernel 那份。

既有 SIMT 路径正是此模式：`MaxPool3DGradSimtTilingData` 同时存在于 `pool3d_grad_struct_common.h`（host）与 `max_pool3d_grad_struct.h`（kernel）。本开发会话 的新 struct 只写进了 kernel 那份，host 翻译单元看不到，`GetTilingData<MaxPool3DGradSeparableTilingData>` 模板实例化时 `T` 不是完整类型，遂报 `does not name a type`。

## 增量：常量的同机制复现（BLOCK_BYTES 未声明，独立 trace，GroupNorm 算子，Ascend950PR + CANN 9.1.0）

同一"host/kernel 翻译单元隔离"机制对**常量**同样成立：op_host/arch35 的 TilingFunc 中用 `BLOCK_BYTES`（32B 对齐块常量）计算对齐粒度，而该常量只在 kernel 侧实现头 `op_kernel/arch35/<op>.h` 里定义（`constexpr uint32_t BLOCK_BYTES = 32;`），op_host 的 tiling.cpp 不包含 kernel 实现头（也不应包含——kernel 头依赖 `kernel_operator.h` 等 kernel 侧编译环境）。构建报：

```
/workspace/.../op_host/arch35/bench_no_harness_group_norm_tiling.cpp:85:29:
  error: 'BLOCK_BYTES' was not declared in this scope
   85 |     uint32_t elemPerBlock = BLOCK_BYTES / calcTypeSize; // 32/4 = 8
```

修复：在 host 侧 tiling 头补同语义常量（tiling.cpp 使用处不变）：

```cpp
namespace optiling {

// 内存块大小 (字节)
constexpr uint32_t BLOCK_BYTES = 32;
```

host tiling.cpp 中的使用处（保持不变，函数级节选自 TilingFunc 对齐计算段）：

```cpp
    // 5. 对齐计算 (中间计算用 fp32)
    uint32_t typeSize = ge::GetSizeByDataType(dtype);
    // 中间计算统一 fp32, 对齐按 fp32 计算
    constexpr uint32_t calcTypeSize = 4; // sizeof(float)
    uint32_t elemPerBlock = BLOCK_BYTES / calcTypeSize; // 32/4 = 8
    uint64_t groupSizeAlign = AlignUp(static_cast<uint64_t>(groupSize), elemPerBlock);
    uint64_t channelsPerGroupAlign = AlignUp(static_cast<uint64_t>(channelsPerGroup), elemPerBlock);
```

设计取舍：在 host 头"再定义一份"而不是把 kernel 头 include 进 host——kernel 实现头依赖 kernel 侧环境，host 编译单元引入会带来更多不可用符号；两个编译单元各自持有同名常量、以注释互指语义，是最小改动。备选（未采用）：host 侧用 `Ops::Base::GetUbBlockSize` 从平台信息动态取块大小。语义要点：该架构一个对齐块 32 字节，fp32（4B）时 8 元素/块；kernel 侧的 `FP32_ELEM_PER_BLOCK = BLOCK_BYTES / FP32_SIZE` 与 host 侧 `elemPerBlock` 必须同口径，否则 tiling 对齐值与 kernel buffer 布局错位。

增量验证信号：

- 修复前：`bash build.sh --soc=ascend950 -j8` host 编译报 `error: 'BLOCK_BYTES' was not declared in this scope`（tiling.cpp:85）（来源平台：950；证据等级：tool_observed）。
- 修复后：同一命令重跑，host 侧编译通过（后续报错转移到 kernel 侧 `sqrtf`，属另一独立根因）；最终 kernel 3 个 `.o` 与 `custom_opp_ubuntu_x86_64.run` 包生成，`[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。

## 判别方法

本规则按"触发 → 探针 → 观测 → 判别"四段组织；触发与观测均为 trace 中 tool_observed 的编译输出。

1. **触发（tool_observed 编译错）**：host 库构建报 `'MaxPool3DGradSeparableTilingData' does not name a type` / `was not declared in this scope`，并伴随 `no matching function for call to 'gert::TilingContext::GetTilingData<<expression error>>()'`，出错行指向 host tiling 类头里 `GetTilingData<新 struct>` 调用点。即命中本规则。

2. **探针（可执行检查）**：
   - 检查 host tiling 类头的 include 列表是否出现 `#include "../../op_kernel/.../struct.h"`——kernel 头通常带 kernel-only 构造，host 翻译单元吃不下；
   - 检查新 TilingData struct 是否**只**定义在 kernel 侧 `*_struct.h`，而在 host 可见的 struct 汇总头（如 `pool3d_grad_struct_common.h`）或 host tiling 类头里没有同 namespace 下的定义；
   - 可用 `grep -rn "struct <TilingDataName>" op_host/ op_kernel/` 确认该 struct 在 host 与 kernel 两侧是否各有一份。

3. **观测（tool_observed）**：修改前 host 构建报上述 `does not name a type` 三连错；把 host 头里对 kernel 侧 struct 头的 include 去掉、改为在 host 头内 `namespace Pool3DGradNameSpace` 中直接定义该 struct（字段与 kernel 侧逐字一致）后，host 库构建成功并链接 `libophost_nn.so`：`[100%] Built target ophost_nn`。

4. **判别（interpretation）**：若触发报错命中 `does not name a type` / `was not declared in this scope` 且探针确认 struct 仅在 kernel 侧定义、host 头试图 include kernel struct 头，根因是 host 翻译单元看不到该 struct（kernel 头含 kernel-only 构造不可被 host include），**正确修复方向是在 host 可见头内同 namespace 重新定义一份**，而不是修 `GetTilingData` 模板或给 kernel 头加 `#ifdef` 让 host 吃下。两边字段顺序与类型必须逐字一致，否则 host 写入的 tiling data 与 kernel 读取的布局错位，会引发运行时数据错乱（非编译期错误，更难查）。

## 预防

- 新增 TilingData struct 时，同步在 host 可见头与 kernel 可见头**各定义一份**，字段顺序/类型逐字一致；优先放在算子既有的 host struct 汇总头（若该头非跨算子共享、可改）。若公共 host 头不可改（如本开发会话 的 `pool3d_grad_struct_common.h` 被多算子共用），则把 host 那份定义放进新 tiling 类自己的 host 头。
- host 头 include 列表里若出现 `#include "../../op_kernel/..."`，视为危险信号——kernel 头通常带 kernel-only 构造，host 翻译单元不能吃。
- 用 `GetTilingData<T>()` 前，确认 `T` 在 host 翻译单元是完整类型（不只是前向声明）。

## 坏实践

- 只在 kernel 侧 struct.h 定义新 struct，指望 host 头 include kernel 头来取——host 编译器吃不下 kernel-only 头，要么直接报错、要么把 struct 定义吞掉后报 `does not name a type`。
- host 与 kernel 两份定义字段顺序/类型不一致——编译能过，但 tiling data 布局错位，运行时拿到错值，是极难定位的运行错。

## 验证信号

- host 编译报 `'MaxPool3DGradSeparableTilingData' does not name a type` / `was not declared in this scope` / `no matching function for call to 'GetTilingData<<expression error>>()'`，指向 host tiling 类头的 `GetTilingData` 调用点（来源平台：950；证据等级：tool_observed）。
- 改为 host 头内同 namespace 直接定义 struct、去掉对 kernel 侧 struct 头的 include 后，host 库构建成功：`[100%] Built target ophost_nn`（来源平台：950；证据等级：tool_observed）。

未上板运行：本卡只证明"host 翻译单元可见 struct、host 库链接成功"，不证明运行正确性。

## 失效边界

- 本卡仅构建验证于 Ascend950（arch35）；a2/a3 未构建。host/kernel 双定义是 AscendC tiling 框架的通用约束——未验证，推断可迁移到 a2/a3。
- struct 的具体字段集（32 个 `int64_t`）是 max_pool3d_grad separable 路径的设计产物；迁移到其它算子时字段随该算子的 tiling 需求变化，但"host/kernel 各定义一份、字段逐字一致"的模式不变。