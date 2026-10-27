---
type: "Runbook"
title: TilingKey 声明与 kernel 模板四处签名须一致
description: ASCENDC_TPL_ARGS_DECL 定义的 TilingKey 参数集是 kernel 派发的单一事实源，必须与 kernel 入口模板、kernel 类模板、tiling 的 ASCENDC_TPL_SEL_PARAM 传参四处严格一致；改声明漏改使用侧报 no matching function for call to '<op>_0_tilingkey'。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# TilingKey 声明与 kernel 模板四处签名须一致

## 现象

在 Ascend950PR（CANN 9.1.0）自定义算子 BenchNoHarnessForeachAddcdivScalar 的骨架调试过程中，为了收敛编译范围，把 `op_kernel/arch35/*_tiling_key.h` 的 TilingKey 从"D_T_X（dtype）× BUFFER_MODE（缓冲模式）双参数"简化成"仅 D_T_X 单参数"（删除了 `ASCENDC_TPL_UINT_DECL(BUFFER_MODE, ...)`）。当时 kernel 侧文件没同步改：kernel 入口模板仍是 `template <typename D_T_X, int BUFFER_MODE>`、kernel 类仍是 `template <typename T, int BUFFER_MODE>`、tiling 函数仍按 `ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer)` 传两个选择参数。下一轮构建在 kernel_0 生成代码里报错（原文，个人路径前缀已泛化）：

```
<工程路径>/build/op_kernel/BenchNoHarnessForeachAddcdivScalar_ascend950/kernel_0/kernel_meta_BenchNoHarnessForeachAddcdivScalar_b8ebf55df6f31295bc4c8cd0fa232dc4/kernel_meta/BenchNoHarnessForeachAddcdivScalar_b8ebf55df6f31295bc4c8cd0fa232dc4_2254187_kernel.cpp:27:5: error: no matching function for call to 'bench_no_harness_foreach_addcdiv_scalar_0_tilingkey'
```

即框架按 TilingKey 声明生成的派发函数 `bench_no_harness_foreach_addcdiv_scalar_0_tilingkey` 找不到与之匹配的手写 kernel 入口——生成侧按"单 dtype 参数"生成调用，手写侧还是"双模板参数"签名。

## 适用场景

- 工程形态：手写 AscendC 自定义算子（registry-invoke），TilingKey 用 `ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL` 模板编程方式声明（非废弃的 TILING_KEY_IS 宏），且在开发过程中调整过 TilingKey 参数集（增删 dtype 维度、缓冲模式维度等）。
- API 路径：`ascendc/host_api/tiling/template_argument.h` 的 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_DATATYPE_DECL` / `ASCENDC_TPL_UINT_DECL` / `ASCENDC_TPL_SEL`；tiling 侧 `ASCENDC_TPL_SEL_PARAM(context, ...)`；kernel 入口 `__global__ __aicore__ void <snake_op_name>(...)` 模板参数。
- 验证过的平台与 CANN 版本：Ascend950PR（`--soc=ascend950` 构建通过），CANN 9.1.0。
- 适用粒度：platform_wide——ASCENDC_TPL 生成机制的平台契约，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：使用废弃 `TILING_KEY_IS` 宏的旧式 TilingKey 写法（本轨迹未使用）；只改 SEL_PARAM 传参个数而不动 TilingKey 定义的场景（同属不一致，机制相同）。

## 根因

`ASCENDC_TPL_ARGS_DECL` 声明的模板参数集是 kernel 派发的"单一事实源"：框架据此生成 `<snake_op_name>_<N>_tilingkey` 派发函数，把 TilingKey 选中的组合实例化到 kernel 入口模板上。四处必须严格一致：

1. `tiling_key.h`：`ASCENDC_TPL_ARGS_DECL` 声明的参数（个数、顺序、类型）；
2. kernel 入口：`template <typename D_T_X, int BUFFER_MODE>` 的模板参数列表；
3. kernel 类：`template <typename T, int BUFFER_MODE>` 及入口内的实例化 `Kernel...<D_T_X, BUFFER_MODE>`；
4. tiling 函数：`ASCENDC_TPL_SEL_PARAM(context, ...)` 的实参个数与顺序。

任何一处与声明脱节（本例：声明删了 BUFFER_MODE，入口/类/SEL_PARAM 仍带），生成代码调用的派发签名与手写 kernel 模板不匹配，即报 `no matching function for call to '<op>_0_tilingkey'`。

## 解决配方

修复前（四处不一致：tiling_key.h 已删 BUFFER_MODE，其余三处仍带）：

```cpp
// op_kernel/arch35/*_tiling_key.h（已简化为单参数）
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessForeachAddcdivScalar,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT,
                               ASCENDC_TPL_INPUT(0))
);
```

```cpp
// op_kernel/*_arch35.cpp kernel 入口（仍双模板参数——不一致）
template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void bench_no_harness_foreach_addcdiv_scalar(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessForeachAddcdivScalarTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessForeachAddcdivScalarTilingData, tilingData, tiling);

    NsForeachAddcdivScalar::KernelForeachAddcdivScalar<D_T_X, BUFFER_MODE> op;
    op.Init(x1, x2, x3, y, workspace, &tilingData);
    op.Process();
}
```

```cpp
// op_kernel/arch35/*.h kernel 类（仍双模板参数——不一致）
template <typename T, int BUFFER_MODE>
class KernelForeachAddcdivScalar {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    ...
};
```

```cpp
// op_host/arch35/*_tiling.cpp（仍传两参——不一致）
uint32_t dTypeX = static_cast<uint32_t>(dataType);
ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
```

修复后（四处同步收敛为单 D_T_X 参数）：

```cpp
// op_kernel/arch35/*_tiling_key.h
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessForeachAddcdivScalar,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT,
                               ASCENDC_TPL_INPUT(0))
);
```

```cpp
// op_kernel/*_arch35.cpp kernel 入口（单模板参数）
template <typename D_T_X>
__global__ __aicore__ void bench_no_harness_foreach_addcdiv_scalar(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessForeachAddcdivScalarTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessForeachAddcdivScalarTilingData, tilingData, tiling);

    NsForeachAddcdivScalar::KernelForeachAddcdivScalar<D_T_X> op;
    op.Init(x1, x2, x3, y, workspace, &tilingData);
    op.Process();
}
```

```cpp
// op_kernel/arch35/*.h kernel 类（单模板参数，缓冲数固化为常量）
template <typename T>
class KernelForeachAddcdivScalar {
    static constexpr int32_t BUFFER_NUM = 1;  // 迭代一：单缓冲
    ...
};

// 全部成员定义同步去参数：
// template <typename T>
// __aicore__ inline void KernelForeachAddcdivScalar<T>::Init(...)   （CopyIn/Compute/CopyOut/Process 同型）
```

```cpp
// op_host/arch35/*_tiling.cpp（单实参；空 Tensor 分支同样只传 dtype）
uint32_t dTypeX = static_cast<uint32_t>(dataType);
ASCENDC_TPL_SEL_PARAM(context, dTypeX);
```

改动清单（本轨迹实际执行的 4 组编辑）：tiling_key.h 重写（删 `ASCENDC_TPL_UINT_DECL(BUFFER_MODE,...)` 与对应 SEL 分支）；kernel 入口重写（模板与实例化去 BUFFER_MODE）；kernel 类 6 处编辑（类声明 + 5 个成员函数定义的模板头去 BUFFER_MODE，`BUFFER_NUM` 固化为 1）；tiling 函数 3 处编辑（空 Tensor 分支与正常分支的 `ASCENDC_TPL_SEL_PARAM` 均只传 dTypeX，缓冲数改用常量 `SINGLE_BUF_TENSOR_COUNT`）。

## 设计决策

- 选择"删 BUFFER_MODE 维度、缓冲数固化为常量"而不是保留维度但只启用单缓冲：骨架阶段目标是先把单变体编译跑通；BUFFER_MODE 作为 TilingKey 维度会在每个 dtype 上再翻倍变体数，与"逐迭代扩展"计划冲突。后续迭代再以独立变更 reintroduce，届时四处一起加回。
- kernel 类的 `BUFFER_NUM` 由模板参数改为 `static constexpr int32_t BUFFER_NUM = 1;`：TQue 的缓冲深度是编译期常量，不依赖 TilingKey 维度也能表达"单缓冲"。

## 系统信息

- 工具链：CANN 9.1.0，构建 `bash build.sh --soc=ascend950 -j8`（生成代码位于 `build/op_kernel/.../kernel_meta_*/kernel_meta/*_kernel.cpp`）。
- 目标平台：Ascend950PR（arch35 / DAV_3510）。
- 相关文件：`op_kernel/arch35/*_tiling_key.h`、`op_kernel/*_arch35.cpp`（入口）、`op_kernel/arch35/*.h`（kernel 类）、`op_host/arch35/*_tiling.cpp`。

## 预防

- 任何 TilingKey 参数集变更（增删维度）当作一个原子变更集处理：tiling_key.h + kernel 入口 + kernel 类 + tiling SEL_PARAM 四处同轮改完再构建，构建前 `grep -rn "BUFFER_MODE\|TPL_SEL_PARAM"` 核对无残留。
- 遇到 `no matching function for call to '<op>_N_tilingkey'` 第一反应查"声明与使用不一致"，直接对照四处签名，不要先怀疑生成器。

## 验证信号

以下信号来自一次 RegBase 骨架开发会话的构建输出与一致性修复后的构建对照（Ascend950PR，CANN 9.1.0）。

- 构建报 `error: no matching function for call to 'bench_no_harness_foreach_addcdiv_scalar_0_tilingkey'`（kernel_0 生成代码 27:5）（来源平台：950；证据等级：tool_observed）。
- 四处签名同步收敛后，下一轮构建中该错误消失（剩余失败为独立的 dtype 变体问题，另有专卡），最终 `--soc=ascend950` 构建成功、kernel 二进制生成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡覆盖"声明与使用不一致"这一根因；同会话 dtype 变体维度的另一个失败（OpDef 声明 dtype 驱动变体生成、FATBIN 缺 .o）是独立根因，见 [OpDef dtype 须被 ASCENDC_TPL_SEL 全覆盖](opdef_dtype_tpl_sel_coverage.md)。
- `ASCENDC_TPL_UINT_DECL` 的其他用法（非缓冲模式维度）机制相同但未逐一验证。
- 仅 CANN 9.1.0 实测；其他版本未验证，推断可迁移。

**未决项**：

- 无（根因定位到 ASCENDC_TPL 派发机制层，构建验证通过）。
## 坏实践

- 只改 `tiling_key.h` 一个文件就构建——TilingKey 是生成侧单一事实源，kernel 入口/类/SEL_PARAM 不同步必然报 no matching function。
- 用注释掉 `ASCENDC_TPL_UINT_DECL` 的方式"暂时禁用"维度——SEL_PARAM 仍传旧参数个数时错误相同。

## 误诊

- 无（生成代码报错 + 四处签名对照一轮定位）。
