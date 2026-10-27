---
type: "Runbook"
title: "OP_ADD 占位 TilingParse 先注册挤掉自定义回调"
description: "OP_TILING_LIB 编译模式下 OP_ADD 宏自动注册占位 TilingParse（OpAddCompilerInfoPlaceholder），且合并策略先注册者胜；自定义 TilingParse 必须用独立 IMPL_OP 注册放到 OP_ADD 之前，否则被 has been registered 告警丢弃。"
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# OP_ADD 占位 TilingParse 先注册挤掉自定义回调

## 现象

在 Ascend950PR（CANN 9.2.0）上为 BenchHarnessScatter 做 GEIR 图模式验收。算子的 host tiling 文件 `op_host/arch35/bench_harness_scatter_tiling_arch35.cpp` 里此前为排查 561002 问题把 TilingParse 注册注释掉了，只剩：

```cpp
// ─── Host registration (DESIGN §9.8) ───
// TilingParse temporarily disabled to isolate 561002 issue.
IMPL_OP_OPTILING(BenchHarnessScatter)
    .Tiling(TilingFunc);
```

为让 GE 图路径拿到 CompileInfo（coreNum/ubSize）参与二进制匹配，在 tiling 文件里把自定义 TilingParse 重新挂回 `IMPL_OP_OPTILING`：

```cpp
// ─── Host registration (DESIGN §9.8) ───
IMPL_OP_OPTILING(BenchHarnessScatter)
    .TilingParse<BenchHarnessScatterCompileInfo>(TilingPrepareForBenchHarnessScatter)
    .Tiling(TilingFunc);
```

重新构建、stage、跑 route 后，日志显示 TilingParse "已注册"，但紧接着是两条重复注册告警：

```
[DEBUG] GE(...): MergeFunctions:op type BenchHarnessScatter tiling_parse func registered.
[WARNING] GE(...): MergeFunctions:op type BenchHarnessScatter tiling_parse func has been registered.
[WARNING] GE(...): MergeFunctions:op type BenchHarnessScatter tiling_parse func has been registered.
```

对 `libcust_opmaster_rt2.0.so` 做 `nm` 检查，能看到**两个** TilingParse 模板实例共存：

```
... T _ZN8optiling35TilingPrepareForBenchHarnessScatterEPN4gert18TilingParseContextE
... W _ZN4gert16OpImplRegisterV211TilingParseIN3ops47OpAddCompilerInfoPlaceholderBenchHarnessScatterEEERS0_PFjPNS_18TilingParseContextEE
... W _ZN4gert16OpImplRegisterV211TilingParseIN8optiling30BenchHarnessScatterCompileInfoEEERS0_PFjPNS_18TilingParseContextEE
... W _ZN4gert16OpImplRegisterV217CreateCompileInfoIN3ops47OpAddCompilerInfoPlaceholderBenchHarnessScatterELi0EEEPvv
... W _ZN4gert16OpImplRegisterV217CreateCompileInfoIN8optiling30BenchHarnessScatterCompileInfoELi0EEEPvv
```

即 `OP_ADD` 宏在 `OP_TILING_LIB` 编译模式下自动注册了一个占位 TilingParse（`OpAddCompilerInfoPlaceholder<Op>`），它先于 tiling 文件里的自定义注册完成静态初始化；CANN 的合并策略是**先注册者胜**，后到的自定义回调被 `has been registered` 丢弃——CompileInfo 实际由占位回调（不填任何字段）负责。

## 适用场景

- 工程形态：自定义算子 op_host 注册开发——在 def 文件用 `OP_ADD(<Op>)` 注册 OpDef、在 tiling 文件用 `IMPL_OP_OPTILING` 注册 Tiling/TilingParse（op_build 会把两者编进 `libcust_opmaster_rt2.0.so`）。
- API 路径：`register/op_def_registry.h`（OP_ADD 宏）、`register/op_impl_registry.h`（`IMPL_OP_OPTILING`/`IMPL_OP`/`OpImplRegisterV2::TilingParse<T>`）、`register/asc/op_def_registry.h`（占位 TilingPrepare 宏展开）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.2.0 头文件下所有用 OP_ADD + 自定义 TilingParse 的算子（机制在 host 注册层，与具体 SOC 无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：本卡只闭合"注册优先级"问题；注册成功不代表回调会被调用（本例修好后回调在运行时/op_build 期仍未被调用，见失效边界与遗留问题）。te_fusion 图编译子路径不调 TilingParse 的另一现象见[te_fusion 未调 TilingPrepare 致 GEIR FPE 卡](te_fusion_tilingprepare_geir_fpe.md)。

## 根因

`OP_ADD(<Op>)` 宏按编译模式展开：`OP_TILING_LIB`（编进 opmaster SO）时除了注册 OpDef，还会自动挂一个占位 `TilingParse<OpAddCompilerInfoPlaceholder<Op>>(TilingPrepare<Op>)`。def.cpp 与 tiling_arch35.cpp 是不同翻译单元，静态初始化顺序由链接顺序决定——本工程的链接顺序使 def.cpp 的 OP_ADD 占位先注册，tiling 文件里 `IMPL_OP_OPTILING(...).TilingParse<...>(...)` 的自定义回调后到，被 `MergeFunctions` 的"先注册者胜"策略丢弃（`has been registered` 告警即丢弃记录）。

## 解决配方

把自定义 TilingParse 改用**独立的 `IMPL_OP` 注册**并放进 def.cpp、置于 `OP_ADD` 之前，使自定义回调必然先于占位注册完成静态初始化。

修复前（tiling 文件里注册，被占位挤掉）：

```cpp
// ─── Host registration (DESIGN §9.8) ───
// (bench_harness_scatter_tiling_arch35.cpp)
IMPL_OP_OPTILING(BenchHarnessScatter)
    .TilingParse<BenchHarnessScatterCompileInfo>(TilingPrepareForBenchHarnessScatter)
    .Tiling(TilingFunc);
```

修复后（def.cpp，注册提前 + 头文件引入完整类型；tiling 文件里只留 TilingFunc）：

```cpp
// (bench_harness_scatter_def.cpp —— 文件头部 forward-declaration 区域)
// Forward declaration of the host tiling callback (defined in
// op_host/arch35/bench_harness_scatter_tiling_arch35.cpp).  Used to bind
// SetTiling on the OpDef so op_build records the tiling entry, matching
// proto.md §1 `this->AICore().SetTiling(optiling::TilingFunc)`.
// Also includes BenchHarnessScatterCompileInfo for TilingParse registration.
#include "arch35/bench_harness_scatter_tiling_arch35.h"
namespace optiling {
ge::graphStatus TilingFunc(gert::TilingContext *context);
}  // namespace optiling
```

```cpp
// (bench_harness_scatter_def.cpp —— OP_ADD 之前)
/**
 * OP_ADD(BenchHarnessScatter):
 *   Static registration macro.  Instantiates the BenchHarnessScatter OpDef and
 *   registers it with the global operator definition registry so op_build can
 *   discover the operator's metadata and generate aic-*-ops-info.ini.
 *
 * Note: The IMPL_OP TilingParse registration below MUST appear before OP_ADD.
 * OP_ADD (in OP_TILING_LIB mode) auto-registers a no-op placeholder TilingParse.
 * By registering the custom TilingPrepareForBenchHarnessScatter first, it takes
 * precedence ("registered first" wins in CANN's merge logic), ensuring the
 * CompileInfo is properly populated with coreNum=1 and ubSize at compile time.
 * coreNum=1 makes the TBE dynamic compiler produce blockDim=1, matching the
 * kernel's CopyDataToY multi-core split (numCores=1 for small cases).
 */
IMPL_OP(BenchHarnessScatter)
    .TilingParse<optiling::BenchHarnessScatterCompileInfo>(optiling::TilingPrepareForBenchHarnessScatter);
OP_ADD(BenchHarnessScatter);
```

```cpp
// (bench_harness_scatter_tiling_arch35.cpp —— 注册移走后只剩 Tiling)
// ─── Host registration (DESIGN §9.8) ───
// TilingParse registration moved to op_host/bench_harness_scatter_def.cpp
// (before OP_ADD) so the custom callback takes precedence over the
// auto-registered placeholder from the OP_ADD macro.
IMPL_OP_OPTILING(BenchHarnessScatter)
    .Tiling(TilingFunc);
```

两个实现细节：

1. `TilingParse<T>` 模板要实例化 `CreateCompileInfo<T>/DeleteCompileInfo<T>`，需要 `T` 的**完整类型**——def.cpp 里先用前向声明（`struct BenchHarnessScatterCompileInfo;`）会编不过，必须 `#include` tiling 头文件拿到完整定义。
2. `IMPL_OP` 用 `__COUNTER__` 生成独立静态对象，同一文件内先写的那条语句先完成静态初始化，因此"写在 OP_ADD 之前"即可保证先注册。

## 设计决策

- 为什么不移到 infershape.cpp（编进 opsproto SO、加载更早）：opsproto SO 以 `OP_PROTO_LIB` 模式编译，OP_ADD 不注册占位 TilingParse，理论上可行；但把 tiling 注册塞进推理文件破坏文件职责划分，且依赖"opsproto SO 先于 opmaster SO 加载"的隐含顺序，不如同文件内语句顺序稳定。
- 为什么用独立 `IMPL_OP` 而不是把自定义 TilingParse 塞回 `IMPL_OP_OPTILING` 再调链接顺序：链接顺序不受源码控制（CMake 目标源列表变更即失效）；语句顺序是显式可见的。
- 当前方案档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.2.0（`include/register/op_def_registry.h`、`include/register/op_impl_registry.h`、`include/register/asc/op_def_registry.h`）。
- 相关文件（会话内实际修改）：
  - `operators/BenchHarnessScatter_package/op_host/bench_harness_scatter_def.cpp`——新增 include、IMPL_OP TilingParse 注册（OP_ADD 之前）。
  - `operators/BenchHarnessScatter_package/op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`——TilingParse 注册移出，仅保留 `.Tiling(TilingFunc)`。
- 验证用命令：route 日志 grep `RegisterOpImplToRegistry|MergeFunctions`、`nm -C libcust_opmaster_rt2.0.so | grep TilingParse`。

## 完整代码

自定义 TilingParse 回调实现（tiling 文件，修复后终版，verbatim；其调用时机问题见失效边界）：

```cpp
ge::graphStatus TilingPrepareForBenchHarnessScatter(gert::TilingParseContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    auto* compileInfo = context->GetCompiledInfo<BenchHarnessScatterCompileInfo>();
    if (platformInfo == nullptr || compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ap(platformInfo);
    // Set coreNum=1 so the TBE dynamic compiler produces blockDim=1, matching
    // the kernel's CopyDataToY multi-core split (numCores=1 for small cases).
    // The precompiled binary (ACLNN path) already has blockDim=-1 (dynamic),
    // so SetBlockDim at runtime is unaffected.  This aligns the GE graph path's
    // compiled blockDim with the kernel's numCores expectation.
    compileInfo->coreNum = 1;
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}
```

## 预防

- 在用 `OP_ADD` 的算子里挂自定义 TilingParse/GenSimplifiedKey 等回调时，先 `nm` 一下 opmaster SO 看是否同时存在 `OpAddCompilerInfoPlaceholder<Op>` 实例；看到 `MergeFunctions ... has been registered` 告警即说明有注册被丢弃。
- 注册类语句的顺序（`IMPL_OP(...)` vs `OP_ADD(...)`）当作显式契约写进文件注释，防止后人"整理代码"时挪动顺序导致回归。

## 验证信号

以下信号来自该算子 GEIR 验收会话的 route 日志与 SO 符号表实测。

- 修复前：`MergeFunctions:op type BenchHarnessScatter tiling_parse func registered.` 后跟两条 `... has been registered.` 告警；`nm` 显示占位（`OpAddCompilerInfoPlaceholderBenchHarnessScatter`）与自定义（`BenchHarnessScatterCompileInfo`）两个 `OpImplRegisterV2::TilingParse<>` 实例共存于 `libcust_opmaster_rt2.0.so`（来源平台：950；证据等级：tool_observed）。
- 修复后：注册顺序变为自定义在前——`register OP_IMPL : [TilingParse]`（独立 IMPL_OP）先于 `register OP_IMPL : [Tiling][TilingParse]`（OP_ADD 占位组合）出现（来源平台：950；证据等级：tool_observed）。
- 诚实边界：回调注册优先后，route 运行与 staging/op_build 期间均未观测到回调被实际调用（fprintf 诊断无输出），kernel 仍以 coreDim=6 的 TBE 产物下发——本配方只证明"注册优先级"闭合，不证明下游匹配恢复（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "先注册者胜"来自日志行为（registered/has been registered 对照）与注册顺序 A/B 实验；CANN 官方文档口径未核对（未验证，推断在 9.2.0 世代成立）。
- 注册成功 ≠ 被调用：本例修复后回调仍未被任何路径调用，说明调用时机由别的条件决定（binary matching 未触发、te_fusion 子路径不调 TilingParse）；不要把本卡当作"CompileInfo 一定被填充"的保证。
- 若工程把 def.cpp 与 tiling 文件编进不同 SO 或调整链接结构，占位与自定义的相对顺序需重新验证。

**未决项**：

- **优先项：定位自定义 TilingParse 被调用的确切条件**。注册顺序修复后回调在 route 运行与 op_build/staging 期均未被调用（fprintf 无输出）；它与"预编译二进制不命中"的关系见同批 GE 图路径 TBE 动态编译判别知识，te_fusion 子路径不调 TilingPrepare 的机制见[te_fusion 未调 TilingPrepare 致 GEIR FPE 卡](te_fusion_tilingprepare_geir_fpe.md)。
- 占位 TilingParse 的预期用途（为何 OP_ADD 要自动注册它）未查证；若官方有更规范的覆盖 API（如宏参数）应优先采用。
## 坏实践

- 在 tiling 文件里加 `IMPL_OP_OPTILING(...).TilingParse<...>(...)` 后看到日志有 "tiling_parse func registered" 就认为自定义回调已生效——那行日志可能对应占位注册，须核对 `has been registered` 告警与注册顺序。
- 用前向声明代替 include tiling 头文件来避免编译依赖——`TilingParse<T>` 模板需要完整类型，编不过。

## 误诊

- 最初把"日志显示 tiling_parse func registered"当作 TilingParse 已生效的证据，未注意随后的 `has been registered` 告警；直到 `nm` 看到占位符号才定位到注册被挤掉（agent_reported 的中途结论，最终被符号表观测纠正）。
