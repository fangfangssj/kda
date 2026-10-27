---
type: "Runbook"
title: "Cube 直调 retCode=0x26：AIV 误跑 Cube API 越界"
description: "Cube 核直调 kernel 入口 extern \"C\" __global__ __aicore__ 未标记核类型时 bisheng 按 MIX 模式编译，kernel 被派发到 AIV 核执行 Cube 专属 API（LoadData/Mmad/Fixpipe），AIV 核无 L0A/L0B/L0C 硬件单元，访问 AIC 内部 buffer 越界，触发 aicore exception retCode=0x26 / ErrCode=507015，plog 表现为 aivector error 而非 Cube 侧错误。判别要点：代码用 Cube API、报错却是 aivector error + fixp_error 非零 + scalar access internal buffer of AICore is out of bounds。复现增量（950/算子仓形态）：默认 AIV 形态 kernel 含 Cube 指令时 ST 成批 507035 且 plog errcode=259 scalar instruction abnormal、cube error info 0；g_coreType==AIV 早退守卫使全部核空跑、输出全 0（判别探针而非修复）；KERNEL_TYPE_MIX 启用路线在该开发会话 反复挂死。"
tags: ["task.debugging", "topic.programming_model.execution_unit.cube", "topic.implementation"]
status: stable
sources: []
created_at: '2026-08-15T00:00:00Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["a3", "950"]
---

# Cube 直调 retCode=0x26：AIV 误跑 Cube API 越界

## 现象

编写 rain_fusion_attention 算子 PV 相乘（BlockMmadPV）的 Cube 核直调验证原型：kernel 入口写为 `extern "C" __global__ __aicore__ void pv_matmul_verify_entry(...)`，函数体调用 Cube 专属 API（`AscendC::LoadData` / `AscendC::Mmad` / `AscendC::Fixpipe`），通过 `<<<1, nullptr, stream>>>` 在 A3 实体卡（ascend910_93 / dav-2201, CANN 9.1.0）上启动。Host 调 `aclrtSynchronizeStream` 同步流失败，plog（`/home/developer/ascend/log/debug/plog/plog-<pid>_<ts>.log`）原文报错：

```
[ERROR] RUNTIME ... [davinci_kernel_task.cc:1262] ... PreCheckTaskErr:An error occurred in the kernel task, retCode=0x26, [aicore exception].
[ERROR] RUNTIME ... [stream.cc:1713] ... GetError:Stream Synchronize failed, stream_id=47, retCode=0x26, [aicore exception].
[ERROR] RUNTIME ... [stream.cc:1716] ... GetError:AI Core kernel task execution failed, retCode=0x26.
[ERROR] RUNTIME ... [api_c_stream.cc:158] ... rtStreamSynchronize:ErrCode=507015, desc=[aicore exception], InnerCode=0x7150026
[ERROR] RUNTIME ... [device_error_core_proc.cc:1118] ... PrintCoreInfo:The error from device(chipId:2, dieId:0) ... there is an exception of fftsplus aivector error, core id is 43 ... aic error mask: 0x6500020bd00028c ...
The extend info: errcode:(0, 0x200000000000000, 0) errorStr: The address for the scalar to access the internal buffer of AICore is out of bounds. fixp_error0 info: 0x1084ae3, fixp_error1 info: 0x13 ...
[ERROR] RUNTIME ... [davinci_kernel_task.cc:1210] ... [DFX_INFO]AI Core kernel execution failed ... fault kernel_name=pv_matmul_verify_entry ...
```

关键信号：`retCode=0x26 [aicore exception]`、`ErrCode=507015`、`aivector error`、`fixp_error` 非零、`errorStr: The address for the scalar to access the internal buffer of AICore is out of bounds`、故障 kernel 名为 `pv_matmul_verify_entry`。表面矛盾点：代码用的是 Cube 核 API，报的却是 **aivector error**（Vector 核侧异常）。

## 适用场景

- **工程形态**：AscendC Cube 核直调原型（`__global__ __aicore__` 入口 + `<<<>>>` 启动，非算子工程 / 非 aclnn）。
- **API 路径**：Cube 核 API（`DataCopy` Nd2Nz、`LoadData`/`LoadData2DParams`、`Mmad`/`MmadParams`、`Fixpipe`/`FixpipeParamsV220`），操作 L1/L0A/L0B/L0C Buffer。
- **验证过的平台与 CANN 版本**：A3 实体卡（ascend910_93 / `--npu-arch=dav-2201`），CANN 9.1.0；另有 Ascend950PR（DAV_3510 / arch35）+ CANN 9.1.0 算子仓形态的并行增量证据（见下方后续增量）——与 frontmatter `platforms: [a3, 950]` 一致。
- **适用粒度**：`operator_family`（所有 Cube 核直调 kernel 入口均适用，不限 PV matmul）。
- **路线优先级**：主判别与修复路线 recommended_fastpath（kernel 入口显式标记核类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` 编译期标记；`g_coreType == AIV` 运行时守卫在直调 MIX 形态下可作修复、在算子仓默认 AIV 形态下仅作判别探针）；`KERNEL_TYPE_MIX` 混合形态启用路线 banned（需单独评估——该路线在增量会话反复导致 NPU 卡死，见「遗留问题」）。
- **不适用或未验证场景**：算子工程（msOpGen / aclnn）方式由 op_host 侧控制核类型，不在本卡范围；Ascend950（950）仿真器见独立卡（cannsim 不支持 Cube 核直调）；A2（dav-200）未验证。

## 判别方法

**触发条件**：Cube 核直调 kernel（入口 `__global__ __aicore__`、函数体调 `LoadData`/`Mmad`/`Fixpipe` 等 Cube 专属 API）经 `<<<>>>` 启动后，`aclrtSynchronizeStream` 同步失败，plog 出现 `retCode=0x26 [aicore exception]` / `ErrCode=507015`，且故障定位为 `aivector error`（Vector 核侧）而非 Cube 核侧、`fixp_error` 非零、`errorStr` 指向 "The address for the scalar to access the internal buffer of AICore is out of bounds"。

**可执行探针**：
1. 在 kernel 入口函数体首行加运行时守卫 `if (g_coreType == AIV) { return; }`（参照官方 `cube_matmul_loaddata_operator` 示例），重新编译运行；若 `retCode=0x26` / `aivector error` 消失且 kernel 正常执行，即证实根因为 MIX 派发到 AIV 核。
2. 或在入口首行改用编译期宏 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);`，bisheng 据此将 kernel 标为纯 Cube 核、不派发到 AIV；若异常消除则同样证实。
3. 检查 bisheng 编译输出是否含 "kernel 类型未标记" 类警告——该警告是 MIX 派发的前置信号，出现即高度吻合本判别。
4. 反向排除：`FixpipeParamsV220.srcStride`、`LoadData` 的 `srcStride` 取值下溢是独立的小问题，单独修这些不会消除 `retCode=0x26` 崩溃；若仅改 Fixpipe 参数崩溃依旧，应转向核类型派发排查。

**原始观测**（A3 实体卡，CANN 9.1.0，PV matmul 直调原型）：
- plog：`PreCheckTaskErr ... retCode=0x26 [aicore exception]`、`rtStreamSynchronize:ErrCode=507015`、`there is an exception of fftsplus aivector error, core id is 43`、`errorStr: The address for the scalar to access the internal buffer of AICore is out of bounds`、`fixp_error0 info: 0x1084ae3, fixp_error1 info: 0x13`、`fault kernel_name=pv_matmul_verify_entry`。
- 入口加 AIV 守卫（做法 A）后 `aivector error` 与 `retCode=0x26` 消失，kernel 完整执行。
- 改用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`（做法 B）后同一 A3 卡上同样不再崩溃。

**机制解释**（根因，明确为分析结论）：bisheng 编译器对 `extern "C" __global__ __aicore__` 入口在**未显式标记 kernel 类型**（既无 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`，也无运行时 `g_coreType` 守卫）时，按 **MIX 模式**编译——kernel 同时被调度到 AIV（Vector）核与 AIC（Cube）核执行。Cube 专属 API（`LoadData`/`Mmad`/`Fixpipe`）操作的是 AIC 内部 buffer（L0A/L0B/L0C），AIV 核没有这些硬件单元，AIV 核执行这些 API 时访问 AIC 内部 buffer 越界 → `aivector error` → `aicore exception (retCode=0x26)`。这一机制链解释了"代码写的是 Cube API、报的却是 aivector error"的表面矛盾：错误来自被误派发的 AIV 核，而非 Cube 核自身参数错。修复方向是在入口标记核类型，确保 Cube 代码仅在 AIC 核执行。

## 验证信号

本次观测来自 rain_fusion_attention PV matmul Cube 核直调原型在 A3 实体卡上的开发会话。

- 修复前：`aclrtSynchronizeStream` 失败，plog `retCode=0x26 [aicore exception]`、`rtStreamSynchronize:ErrCode=507015`、`aivector error`、`errorStr: The address for the scalar to access the internal buffer of AICore is out of bounds`、`fixp_error` 非零、`fault kernel_name=pv_matmul_verify_entry`（来源平台：a3；证据等级: tool_observed）。
- 入口加 `if (g_coreType == AIV) { return; }` 守卫（做法 A）后，verify_result.py 输出 `Shape: [128,128], MaxDiff=0.007812, MeanDiff=0.000002, PassRatio=1.000000`、`RESULT: PASS`，`aivector error` 消失（来源平台：a3；证据等级: tool_observed）。
- 入口改用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`（做法 B）后，同一 A3 卡上 `MaxDiff=0.007812, PassRatio=1.000000` 通过（来源平台：a3；证据等级: tool_observed）。

## 预防

- 写 Cube 核直调 kernel 入口时，**默认**在首行加 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);`（编译期，官方推荐）；若无法改宏则加运行时 `if (g_coreType == AIV) { return; }` 守卫。
- 看到 plog 报 `aivector error` + `fixp_error` 非零 + `scalar access internal buffer of AICore is out of bounds` 而 kernel 用的是 Cube API 时，第一怀疑是 kernel 被 MIX 派发到 AIV 核执行，而非 Cube API 参数错。
- bisheng 对未标记类型的 `__aicore__` 入口会发 "kernel 类型未标记" 警告，不要忽略——该警告即本根因的前置信号。

## 坏实践

- 误把 `aivector error` 当成 Cube 侧 Fixpipe 参数错去反复改 `FixpipeParamsV220.srcStride`，未先查核类型派发（本开发会话 子 agent 一度怀疑 Fixpipe srcStride，绕路后才定位到 MIX 派发）。
- 在入口不加任何核类型标记的情况下，仅靠"代码里只调 Cube API"期望编译器自动识别——bisheng 不会据此推断为纯 Cube，仍按 MIX 编译。

## 失效边界

- CANN 9.1.0 头文件未暴露 `__cube__` 函数级标记，`KERNEL_TASK_TYPE_DEFAULT` 宏是当前唯一编译期标记途径；后续 CANN 版本若暴露 `__cube__`，做法可能变化。
- 950（Ascend950）**仿真器**无法验证本判别是否在 950 真实硬件上也成立——cannsim Ascend950PR 不支持 Cube 核直调 kernel 执行（见相关独立卡）；950 真实芯片的算子仓形态见下方后续增量（已实测）。A2（dav-200）：未验证，推断可迁移（同样有 Cube/AIV 区分）。

**未决项**：

- 【优先】`KERNEL_TYPE_MIX` 启用路线的可用性：该路线在 950 增量会话内反复导致 NPU 卡死（多轮复现），启用前必须单独评估——当前按 banned/需单独评估处理。
- `KERNEL_TYPE_MIX_AIC_1_0` 形态下 Cube 流水线的正确同步姿势（M 管线与 MTE2/MTE3 的 SetFlag/WaitFlag）未验证。
- "所有被调度核均报告 AIV"是对"守卫后输出全 0"的最简解释，未做逐核打印验证；`KERNEL_TYPE_MIX` 形态下 `g_coreType` 行为未验证。
## 后续增量（2026-09-17，并行策展批：950 算子仓默认 AIV 形态含 Cube 指令 → 507035 errcode 259，守卫只可作判别不可作修复）

同一 nugget 在 Ascend950PR（DAV_3510 / arch35）+ CANN 9.1.0 的**算子仓 kernel**（非直调原型）上复现，错误签名与暴露形态不同：GQA 算子 kernel 由纯向量版重写为 Mmad 低阶 Cube API 版（含 LoadData/Mmad/Fixpipe，L1/L0A/L0B/CO1 缓冲），**未声明 `KERNEL_TASK_TYPE_DEFAULT`（默认形态）**，编译安装后跑 ST Real 模式 6 条用例全部失败：

```text
[FAIL] 流同步失败: 507035
```

plog（`ASCEND_SLOG_PRINT_TO_STDOUT=1` 重跑后 grep）给出定位（多条同构，逐核重复）：

```text
[ERROR] RUNTIME(...): ... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0),
... there is an aivec error exception, core id is 0, error code = 259, ...
vec error info: 0x41005c180031103a, cube error info: 0, l1 error info: 0, ...
The extend info: errcode:(259) errorStr: The scalar instruction is abnormal. Possible cause:
The parameter violates the instruction constraints, the binary version does not match,
or the instructinon is overwirtten. subErrType: 0x4.
```

即异常落在向量核（aivec）任务上、`cube error info: 0`——与本卡主体一致：默认形态下 kernel 任务被调度到向量核执行，Cube 指令在向量核上非法。

**关键差异——守卫不是修复**：在 kernel 入口加 `if (g_coreType == AscendC::AIV) { return; }` 早退守卫做隔离实验后，507035 消失，但 6/6 变成 `MERE=1.00e+00`、DEBUG 打印 NPU 输出全部 `0.000000`——默认形态下被调度的核**全部**是 AIV，守卫等于让所有核空跑、输出全 0（把故障静默化）。这区别于本卡主体（直调 MIX 形态下 AIC 核仍执行 Cube 代码，守卫后即可通过）：算子仓默认形态下守卫只能作**判别探针**（507035 消失 + 输出全 0 的成对结果证明异常来自 kernel 体执行本身，而非任务下发/tiling 解析），不能作为解法。

隔离实验的守卫代码（加/移除均在 kernel 入口）：

```cpp
template <typename D_T_Q = half, int IS_CAUSAL = 0, int SCALE_MODE = 0>
__global__ __aicore__ void bench_no_harness_gqa(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // Cube operations (Mmad, LoadData, Fixpipe) only work on AIC cores.
    // AIV cores must return early to avoid scalar instruction exceptions.
    if (g_coreType == AscendC::AIV) {
        return;
    }
    REGISTER_TILING_DEFAULT(BenchNoHarnessGqaTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGqaTilingData, tilingData, tiling);
    NsBenchNoHarnessGqa::BenchNoHarnessGqa<D_T_Q, IS_CAUSAL, SCALE_MODE> op;
    op.Init(query, key, value, y, workspace, &tilingData);
    op.Process();
}
```

启用 Cube 的正确方向是在 kernel 入口声明混合核形态，例如 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0)`（枚举见编译器头 `__clang_cce_aicore_functions.h`：`KERNEL_TYPE_AIV_ONLY=0, KERNEL_TYPE_AIC_ONLY=1, KERNEL_TYPE_MIX_AIV_1_0=2, KERNEL_TYPE_MIX_AIC_1_0=3, KERNEL_TYPE_MIX_AIC_1_1=4, KERNEL_TYPE_MIX_AIC_1_2=5, KERNEL_TYPE_MIX_AICORE=8, KERNEL_TYPE_MIX_VECTOR_CORE=9`）——但 **KERNEL_TYPE_MIX 路线在该开发会话 后续反复导致 NPU 卡死**，采用前必须单独评估（见本增量边界与「遗留问题」）。

### 复现增量验证信号

来源说明：以下信号来自该 GQA 算子 Mmad 低阶 API 重写版在 950 真机的运行时联调与隔离实验（同一 kernel 加/移除入口守卫的成对观测）：

- Mmad 版（默认形态）ST 6/6 `流同步失败: 507035`（来源平台：950/Ascend950PR；证据等级：tool_observed）。
- plog：`error code = 259 ... The scalar instruction is abnormal`，aivec exception、`cube error info: 0`，多核同 PC 位置失败（来源平台：950；证据等级：tool_observed）。
- 加 `g_coreType == AscendC::AIV` 早退守卫后：507035 消失，6/6 `MERE=1.00e+00`，DEBUG 前 10 值 NPU 全 `0.000000`（来源平台：950；证据等级：tool_observed）。
- 移除守卫恢复后：507035 复现（3 条用例），另一条 `MERE=8.27e+01`（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践

- "所有被调度核均报告 AIV"是对"守卫后输出全 0"的最简解释（默认 AIV 形态下调度核即向量核），未做逐核打印验证；`KERNEL_TYPE_MIX` 形态下 `g_coreType` 行为未验证。
- 判别规则只覆盖"默认形态 + Cube 指令 → 507035/errcode 259"这一形态；507035 的其他成因（UB 越界、同步缺失等）不在判别范围内。
- `KERNEL_TYPE_MIX_AIC_1_0` 形态下 Cube 流水线的正确同步姿势（M 管线与 MTE2/MTE3 的 SetFlag/WaitFlag）未验证，且该路线在本开发会话 反复导致 NPU 卡死——启用前必须单独评估。
- 坏实践：在 kernel 入口用 `g_coreType == AIV` 早退来"规避"507035——异常消失的表象下是全部核空跑、输出全 0，属于把故障静默化（算子仓默认形态下）。
- 静态旁证（未上板验证）：`asc/impl/basic_api/dav_3510/kernel_operator_vec_transpose_impl.h` 中向量指令实现多带 `ASCEND_IS_AIV` 守卫（AIC 上为 no-op）——与"Cube 指令不能在 AIV 跑"同属核型约束的另一面。
