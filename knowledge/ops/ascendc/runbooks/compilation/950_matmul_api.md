---
type: "Runbook"
title: "950 Matmul高阶API选型先查内置源"
description: "在 Ascend950 上为反量化+Matmul 融合选 Matmul 高阶 API 的 dtype/TPosition/CubeFormat/执行模式时，先用固定路径读 CANN 内置 weight_quant_batch_matmul_v2 源码确认 950 取值（TSCM/NZ + CFG_MDL + 单核顺序），避免自选 GM/ND half + MIX 致 507015。R2（，两条独立 trace 负结果矩阵）：GM(ND) 输入 + MIX 形态实测两种失败面目——KERNEL_TYPE_MIX_AIC_1_0 反复挂起并锁死 NPU（S=1 也挂）、默认 MIX 服务模型稳定 AIV 507015（aivmap 非零、Mock 全过）；多方案（SetDim/AIC 守卫/CUBE_ONLY/SetBufferSpace/生命周期）均无效，已验证出路=纯 AIV 向量重写或 L1 常驻 TSCM+NZ 接入模式。"
tags: ["task.debugging", "topic.implementation.api_layer.high_level_api", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T03:26:59Z"
platforms: ["950"]
---

# 950 Matmul高阶API选型先查内置源

## 现象

在为自定义算子 BenchNoHarnessWeightQuantBatchMatmul（A16W8 反量化 + Matmul 融合）在 Ascend950PR（DAV_3510 / arch35，CANN 9.1.0，安装路径 cann-9.2.0）上选 Matmul 高阶 API 的 dtype / TPosition / CubeFormat / 执行模式时，自行选择了 MatmulType<TPosition::GM, CubeFormat::ND, half> + KERNEL_TYPE_MIX_AIC_1_1（AIV 反量化 / AIC Matmul / CrossCoreFlag 同步），结果 NPU ST 全部 507015 崩溃。问题不在于某一参数调错，而在于没有先查 CANN 内置同源算子在 950 上实际选了哪组配置，导致在错误的 GM/ND half + MIX 路线里反复调参。

## 适用场景

- 工程形态：Kernel 直调 + aclnn，自定义算子在 950 上需要选 Matmul 高阶 API 的 MatmulType（TPosition / CubeFormat / dtype / isTranspose）、MatmulImpl 配置（CFG_MDL 等）与 kernel 执行模式（KERNEL_TASK_TYPE_DEFAULT 与否 / MIX vs 单核顺序）。
- API 路径：MatmulImpl<...>、REGIST_MATMUL_OBJ、MatmulType<TPosition::*, CubeFormat::*, T, trans>、KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_*)、if ASCEND_IS_AIC/ASCEND_IS_AIV、op.Init()+op.Process()。
- 验证过的平台与 CANN 版本：Ascend950PR / DAV_3510 / arch35；CANN 9.1.0（cann-9.2.0 安装路径）。platforms: [950] 严格对应。
- 适用粒度：operator_family（反量化 + Matmul 融合类，以及任何在 950 上对 Matmul 高阶 API 位置/格式/模式选择拿不准的场景）。
- 路线优先级：recommended_fastpath——动笔写 kernel 前，先按本卡探针读内置源确认配置（TSCM/NZ 方向），而非先盲选 GM/ND half + MIX 再调试；GM 输入路线在本平台多 trace 实证失败（见 复现增量）。
- 不适用或未验证场景：非 950 架构（如 a2/a3）的内置取值不同（custom.h 用 GM/ND、arch35 custom_weight_nz.h 用 GM/ND + GM/NZ 混合且仍 MIX），未验证，不写入 platforms。

## 判别方法

**触发条件**：在 950 上为反量化+Matmul 融合（或任何 Matmul 高阶 API）选 dtype/位置/格式/模式而拿不准，或自选的 GM/ND + MIX 路线已出现 507015。

**可执行探针**（固定路径，CANN 安装目录下的内置算子源码）：

```
BUILTIN_ROOT=<cann-9.2.0 安装根>/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/weight_quant_batch_matmul_v2

# 1) 确认 950(arch35) 入口 cpp 是否设 KERNEL_TASK_TYPE_DEFAULT、调用哪个 kernel 类
$BUILTIN_ROOT/arch35/weight_quant_batch_matmul_v2.cpp
$BUILTIN_ROOT/weight_quant_batch_matmul_v2.cpp            # 根目录 cpp 的 #if/__NPU_ARCH__==3510 #else 分支

# 2) 确认 950 单核顺序 kernel 的 Matmul 位置/格式/配置
$BUILTIN_ROOT/weight_quant_batch_matmul_v2_weight_nz_performance_base.h
$BUILTIN_ROOT/weight_quant_batch_matmul_v2_weight_nz_performance.h
$BUILTIN_ROOT/weight_quant_batch_matmul_v2_custom_weight_nz.h
```

**原始观测**（逐字取自上述内置源码，CANN cann-9.2.0 安装路径固定 blob 证据）：

`weight_quant_batch_matmul_v2_weight_nz_performance_base.h` 第 97-101 行（950 单核顺序 kernel 的 Matmul 类型定义）：
```cpp
using inputXType = MatmulType<TPosition::TSCM, CubeFormat::NZ, xType, aTrans>;
using inputWType = MatmulType<TPosition::TSCM, CubeFormat::NZ, xType, bTrans>;
using outputYType = MatmulType<TPosition::VECIN, CubeFormat::NZ, yType>;
using inputBiasType = MatmulType<TPosition::VECOUT, CubeFormat::ND, float>;
MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj_;
```

`weight_quant_batch_matmul_v2.cpp` 根目录 cpp 的 3510 分支（第 374-390 行 `#else`）：
```cpp
#else
    if constexpr (BATCH == WQBMMV2_BATCH_ZERO && ...) {
        INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzPerformanceKernel, ...);
    } else if constexpr (BATCH == WQBMMV2_BATCH_ONE && ...) {
        INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(WeightQuantBatchMatmulV2WeightNzKernel, ...);
    }
#endif
```

`INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL` 宏（同文件第 107-114 行）：
```cpp
#define INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(templateClass, ...)                                                       \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2NzTilingData, tilingDataIn, tiling);                     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                   \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
                &tPipe);                                                                                             \
        op.Process();                                                                                                \
    } while (0)
```

对照：`weight_quant_batch_matmul_v2_custom.h` 第 70-74 行（**非 3510** 通用分支）：
```cpp
using inputXType = MatmulType<TPosition::GM, CubeFormat::ND, xType, aTrans>;
using inputWType = MatmulType<TPosition::GM, CubeFormat::ND, xType, bTrans>;
using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, yType>;
using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj;
```

**解释**：

1. 950（arch35）的 3510 `#else` 分支不设置 `KERNEL_TASK_TYPE_DEFAULT`，直接 `op.Init()`+`op.Process()`，kernel 内部（weight_nz_performance.h）无 `if ASCEND_IS_AIV/AIC` 分支——即 ascend950 默认模式下 AIC 核同时支持 Cube（Matmul）与 Vector（Cast/Mul/DataCopyPad）指令，单核顺序执行反量化+Matmul，不需要 MIX 模式、不需要 CrossCoreFlag 同步。
2. 950 内置的 Matmul 位置/格式是 A/B 用 `TPosition::TSCM`（L1）+ `CubeFormat::NZ`，C 用 `TPosition::VECIN` + NZ，配置 `CFG_MDL`；而非自选的 `TPosition::GM` + `CubeFormat::ND`（GM/ND 是非 3510 的 custom.h 取值）。
3. B 矩阵类型是 `xType`（half/bf16），不是 `int8`——即 weight 先反量化为 xType 再放入 L1，而非把 int8 直接喂给 Matmul。
4. arch35 目录下的 `weight_quant_batch_matmul_v2.cpp`（新版）第 154 行设 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`、调用 `custom_weight_nz.h`（A=GM/ND、B=GM/NZ、C=GM/ND，仍 `if ASCEND_IS_AIC` MIX），与根目录 cpp 的 3510 `#else` 分支（单核顺序 + TSCM/NZ）存在版本差异；判别时以与本算子同源的、能跑通的分支为准，不要把 arch35 目录 cpp 的 MIX 取值当作 950 唯一选择。

**结论**：在 950 上做反量化+Matmul 融合，先按内置单核顺序 + TSCM/L1 + NZ + CFG_MDL 选型，再写 kernel；不要从 GM/ND half + MIX 起步。

## 验证信号

- ST 运行自选 GM/ND half + MIX 路线在 950 上返回 ACL error 507015，两用例（M=4,K=16,N=8 与 M=128,K=4096,N=4096）均 FAIL，总计 0 通过 2 失败（来源平台：Ascend950；证据等级: tool_observed）。
- AIC-only 变体 ST 运行同样返回 507015，CPU golden 4/4 PASS（MERE=0, MARE=0），证明崩溃在 NPU kernel 执行阶段而非精度逻辑（来源平台：Ascend950；证据等级: tool_observed）。
- 内置 `weight_nz_performance_base.h` 第 97-101 行：A/B = `MatmulType<TPosition::TSCM, CubeFormat::NZ, xType>`、C = `MatmulType<TPosition::VECIN, CubeFormat::NZ, yType>`、`MatmulImpl<..., CFG_MDL>`（来源平台：Ascend950；证据等级: artifact_verified）。
- 内置根目录 cpp 3510 `#else` 分支（第 374-390 行）+ `INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL` 宏（第 107-114 行）：不设 `KERNEL_TASK_TYPE_DEFAULT`，`op.Init()`+`op.Process()` 单核顺序（来源平台：Ascend950；证据等级: artifact_verified）。
- 内置 arch35 cpp 第 154 行 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 与 `custom_weight_nz.h` 第 70-74/150 行（A=GM/ND、B=GM/NZ、C=GM/ND、`if ASCEND_IS_AIC`）显示版本差异（来源平台：Ascend950；证据等级: artifact_verified）。

> 说明：以上为对 CANN 内置源码的读取观测（artifact_verified），用于判别正确选型。本开发会话 内未对按此选型重新实现的 kernel 跑通 ST——即该选型在本算子上能跑通是未验证结论，不得当作 tool_observed 验证信号使用。

## 预防

- 动笔写 kernel 前，先用本卡探针读 CANN 内置同源算子源码，确认本架构下 Matmul 的 TPosition / CubeFormat / 配置 / 执行模式取值，再据此设置自己的 MatmulType 与 KERNEL_TASK_TYPE_DEFAULT。
- 950 上反量化+Matmul 融合，默认选 TSCM/L1 + NZ + CFG_MDL + 单核顺序（不设 `KERNEL_TASK_TYPE_DEFAULT`），把 GM/ND + MIX 当作需要额外证据证明可行的非常规选项。

## 失效边界

- platforms: [950] 只覆盖 Ascend950PR / DAV_3510 / arch35。
- 非 3510 架构（a2 / a3）的内置取值不同（custom.h 用 GM/ND、arch35 custom_weight_nz.h 用 GM/ND + GM/NZ 混合且仍 MIX）；a2/a3 未在本开发会话 内验证，未验证，推断可迁移（同 Matmul 高阶 API 语义），不写入 platforms。
- CANN 版本差异：arch35 目录 cpp（新版，MIX_AIC_1_2 + custom_weight_nz.h）与根目录 cpp 的 3510 `#else`（单核顺序 + weight_nz_performance.h）并存；判别时以与本算子同源且能跑通的分支为准，版本升级时需重跑探针。

## 增量（GM 输入 + MIX 的两条独立 trace 负结果矩阵）

两条独立 trace（GQA 注意力算子 / conv3d 反传 filter 算子，均为 Ascend950PR + arch35，CANN 9.1.0）把"GM(ND) 输入 + MIX"路线的失败面目补全为两种形态，均为 tool_observed 上板实测：

**形态一（GQA，KERNEL_TYPE_MIX_AIC_1_0）：反复挂起并锁死 NPU**。Matmul 高阶 API（REGIST_MATMUL_OBJ + SetTensorA/B + IterateAll，A/B/C 全 GM）四轮尝试全败：

| 尝试 | 结果 |
|---|---|
| 默认形态（入口未声明 KERNEL_TASK_TYPE_DEFAULT） | ST 6/6 `流同步失败: 507015`（快速失败） |
| 加 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0)` | ST 执行阶段挂起（300s 超时被杀）+ `npu-smi info` 本身挂起（设备级锁定） |
| 修复已知 bug 后复测（统一 `AscendC::tiling::TCubeTiling` + Ka 字段 + rowMax/rowSum 越界修复 + `DisableBias()`/`WaitIterateAll()`/`End()` 生命周期补齐） | 仍挂起（EXIT_CODE=124），设备再度锁定 |
| 最小用例（B=4, S=1, S_kv=128，计算量最小的 decode 场景） | 60/90s 超时仍无输出挂起——排除规模/负载因素 |

期间归因线索：host 侧 `optiling::TCubeTiling` 与 kernel 侧 `AscendC::tiling::TCubeTiling` 跨命名空间 reinterpret_cast 布局不匹配 → tiling 数据全零 → Matmul 内部循环失控（类型统一后行为改变佐证）。Mock 模式（CPU golden + mock aclnn）6/6 PASS，证明测试框架与 golden 无关、挂起只在真机执行。终态决策：放弃 Matmul 高阶 API（反复导致 NPU 卡死），回退向量 API 方案（联调阶段"可失败的坏"优于"不可恢复的挂"）。

关键尝试代码（attempt 2/3 核心形态，verbatim）：

```cpp
// kernel 入口（op_kernel/bench_no_harness_gqa_arch35.cpp）
template <typename D_T_Q = half, int IS_CAUSAL = 0, int SCALE_MODE = 0>
__global__ __aicore__ void bench_no_harness_gqa(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
    REGISTER_TILING_DEFAULT(BenchNoHarnessGqaTilingData);
```

```cpp
// ProcessQueryHead（Q@K^T 段；attn@V 段同构）
    mmQK.SetTensorA(qBufGm);
    mmQK.SetTensorB(kBufGm, true);  // transpose K
    mmQK.IterateAll(scoresGm);
    // Bug fix: IterateAll 后必须 WaitIterateAll 等待结果写入 GM，避免数据竞争
    mmQK.WaitIterateAll();
    // Bug fix: 切换到 mmAV 前调用 End 释放 Matmul 计算资源，防止资源冲突
    mmQK.End();
```

**形态二（conv3d 反传 filter，默认 MIX 服务模型）：稳定 AIV 507015**。A/B 用 `MatmulType<TPosition::GM, CubeFormat::ND, ...>` + 默认 `CFG_NORM`，被解析为 `MatmulServiceAux(AIC) + MatmulClient(AIV)` 服务模型，与本算子 GM 输入结构不兼容。负结果矩阵：

| 尝试 | 结果 |
|---|---|
| 空 Process（仅 `SyncAll<true>` 两次） | 正常完成 |
| 仅生成阶段 + SyncAll | 正常完成 |
| 生成 + Matmul（MIX 服务模型） | AIV error ret=507015 |
| 仅 Matmul（跳过生成/Cast） | AIV error / 挂起（EXIT=124 超时） |
| `SetBlockDim(1)`（单组合） | 仍挂起——与多核 blockIdx 分配无关 |
| `MatmulCompute` 加 `if ASCEND_IS_AIV { return; }`（仅 AIC 执行） | 仍 AIV error——AIC 侧 Matmul 服务触发对侧 AIV 异常 |
| `SetBufferSpace(-1, -1, 160KB)`（显式压缩 Matmul UB 预留） | 无效，仍 AIV error |
| `ASCENDC_CUBE_ONLY`（Matmul→MatmulImpl 纯 Cube）+ AIC 守卫 | 可运行但输出全 0：AIV 阶段（生成/Cast）被整体跳过——改变启动语义而非修复 |
| CPU Mock（Matmul 解析为 MatmulClient 模拟） | 5/5 全通过——CPU Mock 通过不能佐证 NPU Matmul 可用 |

**已验证出路（互补两条）**：① 纯 AIV 向量重写（Mul+ReduceSum 逐行点积替代 Matmul，见 RNN 门控算子纯 AIV 路线卡；GQA 向量版最终 27/27）；② L1 常驻 TSCM+NZ + `CFG_ENVECND2NZ` 接入模式（swin 参考，见 arch35 MIX Matmul L1 常驻接入模式卡；本增量 trace 的 conv3d 算子按此落地后 507015 消除）。

**边界与坏实践（复现增量）**：形态一的"banned"是该轨迹的工程决策——期间环境确有多次异常（早前 npu-smi reset 导致的驱动长时间无响应、其他算子 op_build D 状态进程占用设备），挂起与 Matmul API 自身用法的因果未完全隔离；"M 管线与 MTE2/MTE3 跨流水线 SetFlag/WaitFlag 同步"假设未验证，可能是未排除根因之一。形态二结论止于"该组合在本算子场景不可用 + 已排除项清单"，故障的确切指令/寄存器层机制未定位。坏实践补充：设备已锁定（npu-smi 挂起）时反复重试测试（先清理进程并等待/申请人工恢复）；host/kernel 两侧 tiling 结构体跨命名空间 reinterpret_cast（布局不一致时 tiling 全零，Matmul 内部循环失控）；把 CPU Mock 通过当作 NPU Matmul 可用的证据；用 `ASCENDC_CUBE_ONLY` "绕过"AIV 异常（AIV 阶段全部失效、输出全 0，属改变语义的掩盖）。

验证信号（复现增量，两条 trace 的上板实测，Ascend950PR / CANN 9.1.0）：

- 形态一：默认形态 6/6 507015；MIX 形态挂起 + npu-smi 挂起（设备锁定）；bug 修复后复测 EXIT_CODE=124；最小用例 L0-04（S=1）timeout 60/90 仍挂起；Mock 6/6 PASS（来源平台：950；证据等级：tool_observed）。
- 形态二：生成阶段正常、加/仅 Matmul 即 AIV error 507015（aivmap 非零）；SetDim=1 亦挂起；AIV 守卫仍 AIV error；CUBE_ONLY 运行但输出全 0；CPU Mock 5/5 MERE=0；会话终态 UT 10/10、ST Real 0/5（来源平台：950；证据等级：tool_observed）。

遗留问题（复现增量）：形态一在干净设备上用最小 Matmul kernel（单次 IterateAll、无 gather/scatter）复现/排除挂起、把因果从"本轨迹环境"隔离到"API 组合"（优先项）；跨流水线同步假设验证；`KERNEL_TYPE_MIX_AIC_1_1`/`MIX_AICORE` 等其他形态未试；形态二的指令级定位（coredump/单指令复现）。
