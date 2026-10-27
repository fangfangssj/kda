---
type: "Runbook"
title: TilingKey 扩展绕不过 aclnn 执行器缓存
description: 把影响 Tiling 的 attr 扩展进 TilingKey 模板参数（ASCENDC_TPL_UINT_DECL/SEL，24 个键、2 个 binary）后，aclnn 执行器缓存命中时 tiling 函数根本不执行、TilingKey 不参与 cache key，同 shape 异 attr 仍复用首例结果；实测 3/8 失败与改前一致，改动已回退。
tags: [task.debugging, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# TilingKey 扩展绕不过 aclnn 执行器缓存

## 现象

在诊断出"aclnn 执行器缓存不含 attr、同 shape 异 attr 复用首例 Tiling"后，为让不同模式走不同 kernel 实例从而绕开缓存误复用，尝试把 `padding_mode`/`align_corners` 从 TilingData 运行时字段提升为 TilingKey 模板参数：`tiling_key.h` 增加 `ASCENDC_TPL_UINT_DECL(PAD_MODE, 2, ASCENDC_TPL_UI_LIST, 0, 1, 2)` 与 `ASCENDC_TPL_UINT_DECL(ALIGN_CORNERS, 2, ASCENDC_TPL_UI_LIST, 0, 1)`，kernel 入口/类模板同步加参，host tiling 的 `ASCENDC_TPL_SEL_PARAM` 传 4 参。构建成功且产物符合预期（2 个 .o binary、每 binary 打包 12 个 TilingKey 共 24 个），但**缓存启用时测试结果与改动前完全一致**：iter2 L0 仍 3/8 失败，失败形态仍是"同 shape 后续用例错成首例 zeros/false 模式"（MERE=6.884e-01、MARE=1.000e+00 等）；host tiling 仍只被调用 3 次（3 个 shape），且 3 次都是首例的 `padMode=0, alignCorners=0`。该改动随后整体回退。

## 适用场景

- 工程形态：registry-invoke 自定义算子（ASCENDC_TPL 模板参数机制）+ aclnn ST Real 精度验证；本卡是"attr 影响 Tiling 且同 shape 多模式用例"场景下的修复路线选型结论。
- API 路径：`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_UINT_DECL`/`ASCENDC_TPL_UINT_SEL`（`ascendc/host_api/tiling/template_argument.h`）、host 侧 `ASCENDC_TPL_SEL_PARAM(context, ...)`、aclnn 执行器缓存（libnnopbase）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950/arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`——结论关于 TilingKey 设置时机与缓存查找顺序的相对关系，与算子无关。
- 路线优先级：`banned`（作为"绕过执行器缓存"的手段已实测无效；TilingKey 本身用于 kernel 特化仍是正当用途）。
- 明确不适用 / 未验证场景：不涉及执行器缓存的场景（如 Kernel 直调、TilingKey 仅用于编译期特化/性能分派）不受本负结果约束；其他 CANN 版本未验证。

## 根因（失败机制链）

TilingKey 的设置发生在 **host tiling 函数内部**（`ASCENDC_TPL_SEL_PARAM(context, dtypeVal, interpModeVal, ...)`），而 aclnn 执行流程是：

```text
GetWorkspaceSize → 建/取 executor → AddInput/AddAttr/AddOutput
  → 缓存查找（cache key = 算子类型 + tensor shape/dtype，不含 attr、不含 TilingKey）
    → 命中：直接复用缓存 executor（含旧 TilingData 与旧 TilingKey），tiling 不执行
    → 未命中：执行 tiling → 此时才设置 TilingKey/TilingData → 写入缓存
```

因此 TilingKey 在时间上晚于缓存查找，**结构上不可能进入 cache key**；缓存命中时 tiling 函数根本不被调用，新 TilingKey 从未被计算。扩展模板参数只增加了 kernel 实例数量（构建产物从 4 键变 24 键），不改变缓存行为。

尝试的实现（verbatim，供复盘；均已回退）：

```cpp
// tiling_key.h（尝试版）：在 dtype、INTERP_MODE 之外增加 PAD_MODE、ALIGN_CORNERS
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessGridSampler3d,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(INTERP_MODE, 2, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(PAD_MODE, 2, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_UINT_DECL(ALIGN_CORNERS, 2, ASCENDC_TPL_UI_LIST, 0, 1)
);
// host tiling（尝试版）：4 参 SEL_PARAM
ASCENDC_TPL_SEL_PARAM(context, dtypeVal, interpModeVal, padModeVal, alignCornersVal);
```

```cpp
// kernel 入口（尝试版）
template <typename D_T_X, int INTERP_MODE, int PAD_MODE, int ALIGN_CORNERS>
__global__ __aicore__ void bench_no_harness_grid_sampler3d(
    GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGridSampler3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGridSampler3dTilingData, tilingData, tiling);

    NsGridSampler3D::GridSampler3D<D_T_X, INTERP_MODE, PAD_MODE, ALIGN_CORNERS> op;
    op.Init(x, grid, y, &tilingData);
    op.Process();
}
```

观测到的构建产物（机制细节，供理解 ASCENDC_TPL 打包行为）：24 个 TilingKey 打包进 2 个 .o binary（fp32 一枚、fp16 一枚，各 12 键），键编码为 `dtype + interp×256 + pad×1024 + align×4096`（如 zeros+ac=true=4096、reflection+ac=true=6144、fp16 nearest reflection true=1+256+2048+4096=6401）：

```text
Binary 1 (fp32): keys 0, 256, 1024, 1280, 2048, 2304, 4096, 4352, 5120, 5376, 6144, 6400
Binary 2 (fp16): keys 1, 257, 1025, 1281, 2049, 2305, 4097, 4353, 5121, 5377, 6145, 6401
```

## 设计决策

- 当时为什么尝试 TilingKey 扩展：TilingKey 选择不同 kernel binary/executor 实例，直觉上"不同模式=不同 executor"应能打断缓存；且附带收益是 kernel 可用编译期常量分派（消除运行时 switch）。
- 为什么回退：实测无效（缓存行为与键数量无关），且 24 键相对 4 键增加构建产物与维护面；无效改动不留。
- 正确的短期路线：测试侧 `ACLNN_CACHE_LIMIT=0` 禁缓存；根治在框架侧把 attr 纳入 cache key 或提供算子级"attr 影响 tiling"声明。
- 当前方案路线档位：`banned`（以"绕缓存"为目标的 TilingKey 扩展）。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；ASCENDC_TPL 宏定义位于 `${ASCEND_HOME_PATH}/x86_64-linux/asc/include/utils/tiling/template_argument.h`（`ASCENDC_TPL_UI_RANGE=0 / ASCENDC_TPL_UI_LIST=1 / ASCENDC_TPL_UI_MIX=2`）。
- 相关文件：`op_kernel/arch35/*_tiling_key.h`（模板参数声明）、`op_kernel/*_arch35.cpp`（kernel 入口模板）、`op_kernel/arch35/*.h`（类模板与成员函数签名，11 处 `template <typename T, int INTERP_MODE>` 同步改参）、`op_host/arch35/*_tiling.cpp`（SEL_PARAM 调用）；构建产物 `build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_grid_sampler3d/*.json`（kernelList）。

## 预防

- 面对"attr 影响 Tiling + 执行器缓存误复用"问题时，不要投入 TilingKey 扩展；先用最小探针（host tiling 计数日志）确认缓存命中时 tiling 是否执行，再选路线。
- 评估任何"绕过缓存"方案前，先确认该方案的键/标志物在**缓存查找之前**是否可见——晚于查找的任何设置都无效。

## 验证信号

以下信号来自该算子（grid_sampler_3d）迭代二联调验证会话（Real 模式，Ascend950PR + CANN 9.1.0）：

- 尝试落地：TilingKey 扩展后构建成功，产物 2 个 .o、24 个 TilingKey（每 binary 12 键），键编码 `dtype + interp×256 + pad×1024 + align×4096`（来源平台：950；证据等级：tool_observed）。
- 失败观测：缓存启用时 iter2 L0 仍 3/8 失败，失败用例与改动前同集合同形态（border 用例 MERE=6.884e-01/MARE=1.000e+00 等 zeros 模式特征）（来源平台：950；证据等级：tool_observed）。
- 机制观测：缓存启用时 host tiling 仍仅调用 3 次（3 个 shape）且 attrs 均为首例 zeros/false 值——新 TilingKey 对应的 tiling 分支从未执行（来源平台：950；证据等级：tool_observed）。
- 对照：`ACLNN_CACHE_LIMIT=0` 时同一二进制 24 键版本逐用例正确（6/8，剩余失败为独立精度问题），证明 kernel 24 键实现本身正确、失败纯属缓存（来源平台：950；证据等级：tool_observed）。
- 回退后（4 键 + run.sh 禁缓存）终验 8/8、40/40、15/15 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本负结果限定于"以绕过 aclnn 执行器缓存为目标"的 TilingKey 扩展；TilingKey 用于编译期特化、减少运行时分支、按 dtype/插值模式分派 kernel 等正当用途不受影响。
- 结论在 CANN 9.1.0 实测；若未来框架版本把 TilingKey/attr 纳入 cache key，本结论失效（届时缓存问题本身也不复存在）。
- 附带观测（ASCENDC_TPL 打包行为）：多 UINT 模板参数的键值按声明顺序高位编码（后声明参数占高位），框架按 DATATYPE 维度拆分 .o binary——该编码细节对其他算子推断可迁移（未验证，推断可迁移）。

**未决项**：

- 框架侧是否存在"把 attr 声明为 tiling 输入从而纳入 cache key"的接口——未找到（libnnopbase 无公开符号/文档线索），留待框架确认。
## 坏实践

- 不做缓存行为探针就直接改 TilingKey/模板签名（本例 11 处成员函数签名 + 入口 + host 调用同步改动），改动面大且徒劳。
- 把"构建产物键数量变多"当成"缓存已被区分"的证据——键是 tiling 产物的一部分，缓存命中时根本不会重新计算。

## 误诊

- "不同 TilingKey 必然对应不同 executor/缓存项"：TilingKey 是 tiling 阶段的**输出**之一，不是缓存查找的**输入**；时间顺序决定它进不了 cache key。
