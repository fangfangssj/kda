---
type: Operator
title: "Polar 950 ACLNN 算子知识"
description: "说明 aclnnPolarGetWorkspaceSize/aclnnPolar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/docs/aclnnPolar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_api/aclnn_polar.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_api/aclnn_polar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_api/polar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_host/polar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_host/arch35/polar_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_host/arch35/polar_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_kernel/polar_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_kernel/arch35/polar_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_kernel/arch35/polar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_host/config/ascend950/polar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/CMakeLists.txt
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/include/op_common/op_host/tiling_base_class.h
    role: implementation
created_at: "2026-08-25T11:36:45Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Polar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnPolarGetWorkspaceSize` / `aclnnPolar`，公开两段式 ACLNN |
| 计算语义 | 对广播后的每个坐标生成 `complex(abs*cos(angle), abs*sin(angle))` |
| 输入输出 | `input/angle` 为 FLOAT、ND、rank ≤ 8，可非连续且可广播；`out` 为广播 shape 的 COMPLEX64 |
| 执行类型 | R-001 单核心语义 Kernel：外层按 AIV core 分核，节点内由 `asc_vf_call` 启动 1024 个 SIMT 线程；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001：950 FP32 通用广播 SIMT，唯一 key `0` |
| 核心资源 | 两路 FP32 GM 输入、交错 FP32 实部/虚部 GM 输出、296B Tiling POD、每线程标量/索引寄存器；无 AIV `TQue/TBuf`、dynamic UB、共享内存和算子自有 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→950 `IsRegBase` 原生 L0→Tiling/POD→`polar` entry→`PolarSimt<float>`→950 OpDef/config/CMake 已闭合；Contiguous、ViewCopy、executor 和 SIMT primitive 内部是框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `B = broadcast(input.shape, angle.shape)`，对 `B` 中线性坐标 `v` 对应的多维坐标 `c_d`，Kernel 先按广播 stride 找到两个输入位置，再写：

```text
absBase   = sum_d(c_d * absStride[d])
angleBase = sum_d(c_d * angleStride[d])
yBase     = sum_d(c_d * yStride[d])
y.real    = input[absBase] * cos(angle[angleBase])
y.imag    = input[absBase] * sin(angle[angleBase])
```

`absStride[d]` 或 `angleStride[d]` 在该输入维为 1 时被 Host 置 0，因此同一输入元素可被多个输出坐标读取。输出 `COMPLEX64` 在 Device 中按 `float*` 绑定，`yGm[2*yBase]` 是实部，`yGm[2*yBase+1]` 是虚部。实现不检查 `input >= 0`，负值、NaN、Inf 的传播取决于 FP32 乘法以及 `Simt::Sin/Cos` primitive。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `input` | 公开格式 ND，rank ≤ 8；允许非连续；与 `angle` 标准广播 | FLOAT | 公式中的极径 `abs`；非空时先过 `Contiguous`；无非负校验 |
| `angle` | 公开格式 ND，rank ≤ 8；允许非连续；与 `input` 标准广播 | FLOAT，必须与 `input` 相同 | 弧度角；非空时先过 `Contiguous` |
| `out` | view shape 必须精确等于广播 shape；公开格式 ND；允许非连续 | COMPLEX64 | 每个逻辑元素由相邻两个 FP32 分量组成；最终由 `ViewCopy` 提交 |
| `workspaceSize` | Host 标量输出 | `uint64_t*` | executor 全流程所需 Device workspace；空输入明确返回 0 |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有两路 Contiguous、Polar 临时输出、ViewCopy 与任务描述的生命周期 |
| `workspace/workspaceSize/stream` | Device 地址、字节数、ACL stream | 两段式运行参数 | 第二段必须使用第一段返回的 executor 与大小，并在给定 stream 上执行 |

实现级 `CheckShape` 允许两个输入 rank/shape 不同，只要能广播且 `out.viewShape` 等于广播结果；公开文档对“维度保持一致”的表述比实现更窄。950 的 `CheckFormat` 还要求三者 storage format 相同且不是私有格式；正式公开契约仍以文档和 OpDef 的 ND 为准。

### 入口与版本

CANN 9.1 的公开入口是 `aclnnPolarGetWorkspaceSize` 和 `aclnnPolar`，没有更高版本号入口。第一段创建 executor、完成检查和任务编排，第二段只调用 `CommonOpExecutorRun`。

平台分支发生在第一段：`IsRegBase()` 为真时调用原生 `l0op::Polar`；其 OpDef 只向 `ascend950` 注册 `polar_apt`，binary config、模块 CMake 的 `SUPPORT_COMPUTE_UNIT` 也都只含 `ascend950`。因此本卡覆盖 950 原生路径。公开 ACLNN 文档把 A2/A3 列为支持，但对应路径由 `Sin→Cos→Mul×2→Complex` 外部算子图组成；固定 Commit 未给出这些被调算子在 A2/A3 上的 Host/Tiling/Kernel 接线，且 Polar 顶层 README 与 ACLNN 文档的平台表相互冲突，所以未生成无后缀 A2/A3 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnPolarGetWorkspaceSize` 的执行顺序是：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize` 与 `executor` 出参；`CheckNotNull` 检查 `input/angle/out`，空指针返回 `ACLNN_ERR_PARAM_NULLPTR`。
2. `CheckDtypeValid` 要求两输入都是 FLOAT、彼此同 dtype，输出为 COMPLEX64；失败返回 `ACLNN_ERR_PARAM_INVALID`。
3. `CheckShape` 限制三个 Tensor 的 rank 都不超过 8，计算输入广播 shape，并要求它等于 `out` 的 view shape。
4. 950/RegBase 额外执行 `CheckFormat`：三者 storage format 相同且输入不是私有格式。
5. 创建 executor。任一输入为空时直接令 `workspaceSize=0`、移交 executor 并成功返回，不添加 Device 节点。
6. 非空时依次取得 `inputContiguous` 与 `angleContiguous`，950 调用 `l0op::Polar` 分配广播 shape 的连续 COMPLEX64 临时输出并添加 AICore launcher，随后 `ViewCopy(output,out)`。
7. 读取 `uniqueExecutor->GetWorkspaceSize()`，把 executor 所有权移交给调用方。第二段在调用者 stream 上运行已编排任务。

`l0op::Polar` 再做一次广播推形，分配输出，并以 `IsAiCoreSupport(abs,angle)` 检查两输入同 dtype且为 FLOAT；不满足则返回空指针，不存在 AICPU fallback。

### 参数、Workspace 与生命周期

Polar 核心节点的 Tiling 明确设置 `workspaces[0]=0`，entry 的 `workspace` 参数没有被 body 读取。公开 API 的总体 workspace 则是 executor 对两路 `Contiguous`、Polar 临时输出、`ViewCopy` 和框架任务空间的汇总：

```text
W_polar_core = 0
workspaceSize = executor.GetWorkspaceSize()
              = W_contiguous(input) + W_contiguous(angle)
                + W_viewCopy + W_framework
```

框架可能对已经连续的 Tensor 消除物化；固定仓没有公开这些框架节点的实际 slice offset、对齐或消除条件，因此不能把总体 `workspaceSize` 推断为 0。第一段持有所有临时 Tensor 和任务描述，`ReleaseTo(executor)` 后转交调用方；第二段完成前，workspace、executor、输入输出及 stream 都必须保持有效。

### 启动图与失败传播

非空 950 路径的逻辑启动图为：

```text
F0 Contiguous(input) ─┐
                      ├─> K0 polar / PolarSimt<float> ─> F2 ViewCopy(temp,out)
F1 Contiguous(angle) ─┘
```

| 节点 | Host 添加函数 | entry/body | launch 参数 | 输入→输出 | workspace/交接 | 失败传播 |
|---|---|---|---|---|---|---|
| F0 | `l0op::Contiguous(input,executor)` | 框架边界 | executor/stream | 用户 input view→连续 FP32 Tensor | executor 管理，slice 不可见 | 空返回立即终止第一段 |
| F1 | `l0op::Contiguous(angle,executor)` | 框架边界 | executor/stream | 用户 angle view→连续 FP32 Tensor | executor 管理，slice 不可见 | 空返回阻断 K0 |
| K0 | `ADD_TO_LAUNCHER_LIST_AICORE(Polar,...)` | `polar`→`PolarSimt<float>::Process` | `blockDim=min(N,C)`、key 0、296B POD；每个外层 block 内 `asc_vf_call(dim3(1024),...)` | 两路连续 FP32 GM→连续 COMPLEX64 临时 Tensor | 节点自有 workspace 0；GM 临时 Tensor 由 executor 持有 | launcher/Tiling 失败阻断 F2；无 fallback |
| F2 | `l0op::ViewCopy(output,out,executor)` | 框架边界 | executor/stream | 连续临时结果→用户 out view | executor 管理，slice 不可见 | 空返回使第一段失败 |

F0/F1 的结果都是 K0 的前驱，K0 结果是 F2 的前驱；同一 executor 在调用者给定 stream 上维护任务顺序与可见性。固定仓没有 Polar 自有 event、跨 stream、workspace 边或 cleanup Kernel。临时输出由 executor 持有到 F2 消费完成后才可回收。

## Route 与参数设计

### 分核与资源计算

设 `D=max(rank(abs),rank(angle))≤8`，`M[d]=max(absDim[d],angleDim[d])`，`N=product(M[d])`，`C=GetCoreNumAiv()`：

```text
elementsPerCore = floor(N / C)
formerCore      = N mod C
blockDim         = min(N, C)

block b start = b < formerCore
              ? b * (elementsPerCore + 1)
              : formerCore + b * elementsPerCore
block b count = b < formerCore ? elementsPerCore + 1 : elementsPerCore
```

Host 从最末维向前计算连续 stride。对某输入，`inputDim[d]==1` 时其 stride 写 0，否则写当前行主序累积 stride；`mergedStride` 与 `yStride` 都是广播输出的行主序 stride。低 rank 输入在前导维补 1。未使用的 `d∈[D,8)` 槽位写 `mergedStride=1`、其它三个 stride=0，Device 只循环到 `dimNum=D`。

每个外层 AIV block 只负责连续输出区间 `[start,start+count)`；其内部固定启动 1024 个 SIMT 线程，线程 `t` 处理 `i=t,t+1024,...<count`。不存在 tile、UB 容量或 workspace 上界选择。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 950 通用广播 SIMT | `GetTilingKey()==0`；`PolarSimt<float>` | 950 `IsRegBase()`；非空；两输入 FLOAT 且同 dtype；可广播；输出 COMPLEX64 且 shape 等于广播结果；公开 ND/rank≤8；Tiling `IsCapable()==true` | API/shape/dtype/format/Tiling/launcher 任一失败即报错；无第二 Device Route |

```text
if null/dtype/rank/broadcast/out-shape invalid: fail
if A5 format invalid: fail
if input.empty or angle.empty: success with workspaceSize=0 and no launch
Contiguous(input); Contiguous(angle)
if IsRegBase():
    l0op::Polar -> key 0 -> R-001 PolarSimt<float>
else:
    external Sin/Cos/Mul/Complex graph  # 非本 A5 Route
ViewCopy(temp,out)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 950 通用广播 SIMT | key 0；无 router 分支 | 950；FP32×FP32→COMPLEX64；广播 rank≤8；非空 | 单 K0；外围 F0/F1/F2 是框架边界 | `PolarSimt<float>::Init/Process/SimtPolarCompute<float>` | 全程 GM；每线程标量寄存器；无 UB/shared/atomic/barrier；executor stream 排序 | former core 多 1 元素；线程 while 处理任意 count；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 原生分支 | `IsRegBase()` | 只有 RegBase/950 走单 Polar Kernel | R-001 | 非 RegBase 走外部五节点组合图，不属于本卡 | `aclnnPolarGetWorkspaceSize` |
| 平台接线 | `ascend950` | OpDef、binary config 与 CMake 只发布 950 binary | R-001 | 非 950 无本 Kernel 接线 | `ops::Polar::Polar`、`polar_binary.json`、CMake |
| dtype | `abs.dtype==angle.dtype==FLOAT`，`out=COMPLEX64` | entry 固定实例化 `PolarSimt<float>`，输出按两个 float 分量写 | R-001 | ACLNN/L0/Tiling 失败，无 fallback | `CheckDtypeValid`、`IsAiCoreSupport`、`PolarTiling::CheckDtype` |
| rank | `D<=POLAR_MAX_DIM(8)` | POD 每类 stride 只有 8 项 | R-001 | API 或 Tiling 失败 | `CheckShape`、`CheckBroadcastAndMergeShape` |
| 广播合法性 | 每维 `a==b \|\| a==1 \|\| b==1` | stride-0 广播仅能表达标准逐维广播 | R-001 | API/Tiling 失败 | `CheckBroadcastAndMergeShape` |
| 非空 | API 的 `input->IsEmpty() \|\| angle->IsEmpty()` | 避免 `N=0` 时 blockDim 0 launch | R-001 | 成功返回、workspace 0、无 Kernel | `aclnnPolarGetWorkspaceSize` |
| AIV core 数 | `GetCoreNumAiv()>0` | 分核公式除以 `coreNum` | R-001 | TilingPrepare 失败 | `TilingPrepare4PolarAscendc` |
| SIMT 线程数 | 编译常量 `POLAR_THREAD_DIM=1024` | `LAUNCH_BOUND(1024)` 与 `asc_vf_call(dim3(1024))` 一致 | R-001 | 固定仓无动态线程 fallback | `PolarSimt::Process/SimtPolarCompute` |

### POD/ABI

`PolarTilingData` 在 `#pragma pack(push,8)` 下由 37 个 `int64_t` 槽位组成，故 `sizeof=296`、`alignof=8`；以下 offset 由声明顺序与 8B 对齐直接确定。Host 以整结构 `memcpy_s` 写入 raw Tiling buffer，entry 用同一个结构名解析。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `PolarTilingData` | `totalElements` | `int64_t`/64 | size296 align8 offset0 | 输出逻辑元素 | `DoOpTiling: totalElements_=product(mergedShape)` | `Init` 保存到 `totalElements_`，后续不读 | 乘法溢出检查固定仓不可见；非空应 >0 |
| 同上 | `elementsPerCore` | `int64_t`/64 | offset8 | 元素/AIV block | `totalElements_/coreNum` | `Process` 计算 start/count | floor 除法；要求 coreNum>0 |
| 同上 | `coreNum` | `int64_t`/64 | offset16 | AIV core | TilingParse 的 `GetCoreNumAiv()` | `Init` 保存到 `coreNum_`，后续不读 | launch 实际用 `blockDim`；Device inactive 字段 |
| 同上 | `formerCore` | `int64_t`/64 | offset24 | AIV block | `totalElements_%coreNum` | `Process` 决定前段 block 多 1 元素 | `[0,coreNum)` |
| 同上 | `dimNum` | `int64_t`/64 | offset32 | 维 | `max(absRank,angleRank)` | SIMT 多维解码循环上界 | `0..8`；标量时可为0 |
| 同上 | `mergedStride[8]` | `int64_t[8]`/512 | offset40，数组 extent64B | 元素 | `CalcStride: strideMerged`，未用槽写1 | `para.ms[d]`；`vv/ms[d]` 解码坐标 | 活动维 stride>0；只读 `[0,dimNum)` |
| 同上 | `absStride[8]` | `int64_t[8]`/512 | offset104，数组 extent64B | 输入元素 | `absDim[d]==1 ? 0 : strideAbs`，未用槽写0 | `absBase += c*para.absS[d]` | 0 表示广播；只读活动维 |
| 同上 | `angleStride[8]` | `int64_t[8]`/512 | offset168，数组 extent64B | 输入元素 | `angleDim[d]==1 ? 0 : strideAngle`，未用槽写0 | `angleBase += c*para.angleS[d]` | 0 表示广播；只读活动维 |
| 同上 | `yStride[8]` | `int64_t[8]`/512 | offset232，数组 extent64B | complex 元素 | `CalcStride: strideY`，未用槽写0 | `yBase += c*para.yS[d]` | 实际 GM float offset 再乘2 |

`blockDim` 不在 POD 中：`PostTiling` 直接调用 `context_->SetBlockDim(uint32_t(blockDim_))`。key 也不在 POD：基类最终以 `GetTilingKey()==0` 写入 context。`PolarStridePara` 是 body 内部按值传给 SIMT 函数的 256B 参数聚合，不是 Host ABI；其具体寄存器/参数区放置由编译器决定，固定仓不可见。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 950 通用广播 SIMT | 950 `IsRegBase`；key 0 | `aclnn_polar.cpp::aclnnPolarGetWorkspaceSize`；`polar.cpp::Polar/PolarAiCore`；`polar_tiling.cpp::PolarTiling` | `PolarTilingData`；`blockDim=min(N,C)`；inner threads=1024 | `polar_apt.cpp::polar`，无 key 分支，固定 `PolarSimt<float>` | `polar_simt.h::Init/Process/SimtPolarCompute`；`Simt::Sin/Cos` primitive | `polar_def.cpp` + `polar_binary.json` + `CMakeLists.txt`，均为 ascend950 |

## Kernel 实现

### 公共执行流程

只有 R-001。entry `polar(x1,x2,y,workspace,tiling)` 解析默认 `PolarTilingData`，忽略 workspace，固定构造 `PolarSimt<float>`。`Init` 拷贝 POD 标量与四组 stride，绑定三段 GM；`Process` 用外层 `GetBlockIdx()` 取得连续区间，然后以 `asc_vf_call` 在该 AIV block 内调用 1024-thread 的 SIMT 函数。SIMT 线程逐输出解码广播坐标，直接从 GM 读两输入并直接向 GM 写复数的两个 FP32 分量。

### Route R-001：950 通用广播 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/polar/op_api/aclnn_polar.cpp` | `CheckParams/aclnnPolarGetWorkspaceSize` | dtype、rank、广播/out shape、950 format 合法；非空；`IsRegBase()` | 两路 Contiguous 后调用 `l0op::Polar` |
| L0 | `math/polar/op_api/polar.cpp` | `Polar/IsAiCoreSupport/PolarAiCore` | 广播推形成功；两输入同为 FLOAT | 分配 COMPLEX64 输出并添加 `Polar` AICore launcher |
| Tiling base | `opbase/.../tiling_base_class.h` | `TilingBaseClass::DoTiling` | Shape→Platform→`IsCapable(true)`→OpTiling→LibApi→Workspace→Post→key | 任一步失败立即返回；成功 key 0 |
| Tiling | `math/polar/op_host/arch35/polar_tiling.cpp` | `CheckBroadcastAndMergeShape/CalcStride/DoOpTiling/PostTiling` | rank≤8、逐维可广播、AIV core>0 | 296B POD、workspace0、`blockDim=min(N,C)` |
| Router | `math/polar/op_kernel/polar_apt.cpp` | `polar` | 默认 POD；无 key/dtype 分支 | 固定 `PolarSimt<float>` |

#### Device 数据流

##### 函数级调用链

```text
aclnnPolarGetWorkspaceSize
  -> Contiguous(input), Contiguous(angle)
  -> l0op::Polar -> PolarAiCore -> ADD_TO_LAUNCHER_LIST_AICORE(Polar,...)
    -> Tiling4Polar -> PolarTiling::DoTiling -> key 0 / blockDim / PolarTilingData
    -> polar(x1,x2,y,workspace,tiling)
      -> PolarSimt<float>::Init
      -> PolarSimt<float>::Process
        -> asc_vf_call<SimtPolarCompute<float>>(dim3(1024), GM pointers, start, count, dimNum, strides)
          -> SimtPolarCompute<float>: coordinate decode -> GM loads -> Simt::Cos/Sin -> two GM stores
  -> ViewCopy(temp,out)
aclnnPolar -> CommonOpExecutorRun(workspace,workspaceSize,executor,stream)
```

##### 源码等价伪码

```text
# 这是保持源码循环、地址和条件的等价伪码
b = GetBlockIdx()
start = (b < formerCore) ? (elementsPerCore + 1) * b
                         : formerCore + elementsPerCore * b
count = (b < formerCore) ? elementsPerCore + 1 : elementsPerCore
if count <= 0: return

SIMT launch threads=1024:
  i = threadIdx.x
  while i < count:
    v = start + i
    absBase = angleBase = yBase = 0
    vv = v
    for d in [0, dimNum):
      coord = vv / mergedStride[d]
      vv -= coord * mergedStride[d]
      absBase   += coord * absStride[d]
      angleBase += coord * angleStride[d]
      yBase     += coord * yStride[d]
    r = absGM[absBase]
    a = angleGM[angleBase]
    yGM[2*yBase]     = r * cos(a)
    yGM[2*yBase + 1] = r * sin(a)
    i += 1024
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `absGm_` | `absBase=sum(coord[d]*absStride[d])` | 连续 input storage 元素数 | FP32/ND | F0 Contiguous 或原 input | 每个 SIMT 线程标量 load | 用户/executor；K0 全程只读 | stride 0 跨输出复用同一值；无显式 DMA alignment 要求 |
| GM `angleGm_` | `angleBase=sum(coord[d]*angleStride[d])` | 连续 angle storage 元素数 | FP32/ND | F1 Contiguous 或原 angle | `Simt::Cos` 与 `Simt::Sin` 的标量 load | 用户/executor；K0 全程只读 | 广播 stride 0；源码对同一角度表达式读取两次 |
| GM `yGm_` | `2*yBase` 与 `2*yBase+1` | `2*N` 个 FP32，即 `8*N` B | COMPLEX64，实虚交错 | SIMT 线程两次 store | F2 ViewCopy | executor 临时 Tensor；F2 后可回收 | 不同 `v` 映射不同 `yBase`，无写冲突 |
| GM Tiling | entry 参数 `tiling` | 296B | `PolarTilingData`/pack8 | Host `memcpy_s` | entry + `Init` | 框架 launch 参数，K0 初始化期有效 | 8B 字段；Host/Device 同结构 |
| `PolarStridePara` | 四个 8×int64 数组 | 256B 逻辑参数 | SIMT 调用参数 | `Init` 从 POD 拷贝 | 每个 SIMT 线程坐标解码 | `PolarSimt` 成员并按值传入；物理参数区固定仓不可见 | 非 AIV UB，不计入 UB_total |
| 每线程标量 | `idx/step/i/v/vv/bases/coord/r/cos/sin` | 编译器决定 | int64/FP32 registers | 当前线程 | 当前线程 | 单次 while 迭代或线程调用 | 寄存器数与 spill 固定仓不可见 |
| workspace | N/A | K0 为 0B | N/A | `PostTiling` | body 不读 | N/A | API 总 workspace 仍含框架节点 |

##### 分层内存布局

R-001 不创建 AIV `TPipe/TQue/TBuf/LocalTensor`，也没有 SIMT dynamic UB、shared memory 或通信 window；实际路径是 `GM scalar load → per-thread registers → GM scalar store`。因此不伪造 AIV `block-beta` 布局。

##### 字节账与复用顺序

```text
AIV_UB_total = 0 B                 # 无 InitBuffer/TQue/TBuf
SIMT_dynamic_ub = N/A              # asc_vf_call 未传 dynamic UB size
SIMT_shared_memory = 0 B           # body 未声明 shared storage
PolarStridePara logical payload = 4 * 8 * 8 = 256 B
PolarTilingData = (5 + 4*8) * 8 = 296 B
K0_workspace_total = 0 B
API_workspace_total = executor.GetWorkspaceSize()  # 框架切片固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 广播输入 GM 元素 | F0/F1 或用户 | 多个只读 SIMT 线程 | K0 不覆盖 | 输入只读；executor stream 前驱完成后启动 K0 | stride 0 只造成共享读取，不产生写冲突 |
| 输出 `yGm_[2*yBase:2*yBase+2)` | 唯一输出线程 | F2 ViewCopy | F2 消费后由 executor 回收临时量 | 同一 stream 的 K0→F2 依赖 | `v→yBase` 对广播输出是一一映射 |
| 每线程局部标量 | 当前线程当前迭代 | 同线程表达式 | 下一次 while 迭代 | 单线程程序顺序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| F0/F1→K0 | executor 同 stream 依赖 | Contiguous | K0 GM load | 两路连续输入 | 框架内部同步细节不可见 |
| block 分域 | 不相交 `[start,start+count)` | Host POD + `GetBlockIdx` | SIMT VF call | 输出逻辑索引 | 前 `formerCore` 个 block 多 1 元素；`count<=0` 返回 |
| 线程分域 | `i=threadIdx.x; i+=blockDim.x` | 1024 SIMT threads | 每线程 while guard | 当前 block 的 count 个输出 | `threadIdx.x>=count` 不执行；任意非 1024 倍 tail 被 while guard 覆盖 |
| 广播读取 | stride-0 只读共享 | GM input | SIMT scalar loads | 同一广播元素 | 无 barrier/atomic；只读无冲突 |
| 三角计算 | `Simt::Cos`、`Simt::Sin` | FP32 angle | FP32 multiplication | 实部/虚部标量 | 无升/降精度或 round 参数；primitive 的特殊值/近似细节固定仓不可见 |
| 结果写入 | 两个独占 GM scalar stores | 当前线程 | F2 ViewCopy | 一个 COMPLEX64 元素 | 实部先写、虚部后写；线程间地址不重叠，无 atomic/barrier |
| K0→F2 | executor 同 stream 依赖 | K0 | ViewCopy | 连续临时输出 | F2 只在 K0 完成对应依赖后消费 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| API producer | `math/polar/op_api/aclnn_polar.cpp` | `CheckParams/aclnnPolarGetWorkspaceSize/aclnnPolar` | 校验、空域、Contiguous、950 selector、ViewCopy、workspace/executor 生命周期 |
| L0 launcher | `math/polar/op_api/polar.cpp` | `Polar/PolarAiCore/IsAiCoreSupport` | 广播输出分配、FP32 AICore 选择、launcher 添加 |
| Host/Tiling producer | `math/polar/op_host/arch35/polar_tiling.cpp` | `PolarTiling::* / Tiling4Polar / TilingPrepare4PolarAscendc` | AIV core 数、广播 stride、均衡分核、workspace0、POD 序列化 |
| POD | `math/polar/op_kernel/arch35/polar_struct.h` | `PolarTilingData` | 296B、8 维上限与 Host/Device 共享字段顺序 |
| Kernel entry | `math/polar/op_kernel/polar_apt.cpp` | `polar` | 默认 POD、固定 `PolarSimt<float>`、workspace 未使用 |
| Device body | `math/polar/op_kernel/arch35/polar_simt.h` | `PolarSimt<float>::Init/Process/SimtPolarCompute<float>` | 1024-thread SIMT、坐标解码、广播地址、Sin/Cos/乘法、交错输出与 tail |

## 算子特定优化

### Trick T-001：stride-0 广播与输出坐标在 SIMT 线程内融合

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：任意合法的 rank≤8 标准广播；Host 对输入 size-1 维写 stride 0。
- Golden 来源：[`polar_simt.h:97-108`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_kernel/arch35/polar_simt.h#L97-L108)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int64_t d = 0; d < dimNum; d++) {
    int64_t c = vv / para.ms[d];
    vv -= c * para.ms[d];
    absBase += c * para.absS[d];
    angleBase += c * para.angleS[d];
    yBase += c * para.yS[d];
}
U absVal = absGm[absBase];
```

#### 机制与设计原因

Host 已把广播维编码成 stride 0，Device 用同一次输出坐标解码同时生成 abs、angle 与 y 三个地址。这样不需要先把任一输入广播物化为 `N` 元素 Tensor，也不需要额外 layout/broadcast Kernel 或 workspace。`mergedStride` 决定坐标商和余数；两个输入 stride 决定读取地址；`yStride` 决定复数逻辑元素，再由 `2*yBase(+1)` 映射到交错 FP32 分量。

#### 迁移边界

- 适用：输出元素能独立计算、输入遵循标准逐维广播、rank 有静态 POD 上限的 elementwise 复合算子。
- 前提：Host 与 Device 对右对齐维度、stride 0、行主序线性化和最大 rank 的约定完全一致。
- 失效：非标准广播、输出地址非一一映射、rank>8 或需要整 tile/vector 复用时，必须更换地址生成/POD 与实现；本 Route 没有 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：两级均衡分域加 1024-thread 跨步覆盖任意尾段

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：任意非空输出；`C=GetCoreNumAiv()>0`，外层 blockDim 为 `min(N,C)`。
- Golden 来源：[`polar_simt.h:65-78`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/polar/op_kernel/arch35/polar_simt.h#L65-L78)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t startIdx = (blockIdx < formerCore_) ? (elementsPerCore_ + 1) * blockIdx
                                             : formerCore_ + elementsPerCore_ * blockIdx;
int64_t count = (blockIdx < formerCore_) ? (elementsPerCore_ + 1) : elementsPerCore_;
if (count <= 0) return;
asc_vf_call<SimtPolarCompute<T>>(dim3(POLAR_THREAD_DIM),
```

#### 机制与设计原因

Host 用商和余数使任意两个外层 block 的工作量最多差 1，并在 `N<C` 时只启动 `N` 个 block；Device 再让 1024 个 SIMT 线程以 `i+=blockDim.x` 遍历各自 block 区间。两级分域均不要求 `N` 对齐到 core 数或 1024，避免 padding、尾 Kernel 和原子合并，且所有线程写地址互斥。

#### 迁移边界

- 适用：每个输出独立、每元素成本近似一致、可用连续线性索引表示的数据并行算子。
- 前提：外层 `blockDim`、POD 中 `elementsPerCore/formerCore` 必须来自同一 `C`；内层 `LAUNCH_BOUND` 与 `asc_vf_call` 的线程数必须一致。
- 失效：元素成本显著不均、需要跨线程归约/共享内存或单元素写入可能冲突时，应引入其他调度与同步；固定 Route 无 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 950 通用广播 SIMT | 设计选择依据 |
|---|---|---|
| 命中条件/优先级 | 950 RegBase、非空、FP32 输入、COMPLEX64 输出、合法广播；唯一 key 0 | 无第二模板或 fallback，所有合法 950 非空输入同路 |
| 模板与 dtype | `PolarSimt<float>` | ACLNN、L0、Tiling、binary config 都固定 FP32→COMPLEX64 |
| 循环与分核 | `min(N,C)` 外层 blocks；前余数 blocks 多 1 元素；内层 1024 threads 跨步 | 覆盖任意 N 且不需要尾 Kernel |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB/window；每线程使用标量与索引寄存器 | 直接 GM 标量计算，编译器寄存器数不可见 |
| 内存总占用与复用 | K0 workspace/UB/shared 都为0；广播以 stride0 复用只读 GM | 避免广播中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM scalar load/store + `asc_vf_call` + `Simt::Sin/Cos` | 无 Vector DMA、NDDMA、通信或 Cube API |
| 同步、精度与 tail | 无节点内 barrier/atomic；全 FP32；while guard 处理 thread tail | 输出独占写，外围由 executor stream 排序 |
| fallback/失败边界 | 非法参数/Tiling/launcher 失败；空域提前成功；无 Device fallback | 未知 key 分支不存在，entry 固定 body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | ACLNN 文档写支持，但顶层 README 写不支持；固定仓实现闭包缺失，未发布卡 | ACLNN 文档写支持，但顶层 README 写不支持；固定仓实现闭包缺失，未发布卡 | 公开支持，OpDef/config/CMake/Host/Kernel 闭合 |
| Host/Launcher | 非 RegBase 分支为 Sin、Cos、Mul×2、Complex 组合；被调节点的平台接线不可闭合 | 同 A2 | `IsRegBase`→单 `Polar` launcher，外围 Contiguous/ViewCopy |
| Kernel/资源 | 固定 Commit 无 Polar A2 原生 Kernel；组合节点实现不可闭合 | 固定 Commit 无 Polar A3 原生 Kernel；组合节点实现不可闭合 | 单 `PolarSimt<float>`；GM+寄存器；无自有 UB/workspace |

## 限制与失败边界

- 公开 dtype 仅 FLOAT×FLOAT→COMPLEX64，公开格式 ND，rank 最大 8；实现不校验极径非负。
- 输出 view shape 必须等于输入广播结果；逐维不满足 `a==b || a==1 || b==1` 时 API 或 Tiling 失败。
- 950 还要求三者 storage format 相同且输入不是私有格式；L0 不提供 AICPU 或其他 Device fallback。
- `totalElements_` 与连续 stride 使用 `int64_t` 连乘，固定仓没有显式乘法溢出检查；`blockDim_` 最终窄化为 `uint32_t`，但其上界来自平台 AIV core 数或 `N`。
- `Simt::Sin/Cos` 的近似算法、特殊值细节、寄存器分配与可能 spill 属 AscendC primitive/compiler 边界，固定仓不可见。
- Contiguous、ViewCopy 和 executor 总 workspace 的具体实现、slice offset、对齐与消除条件是框架边界，固定仓不可见；不能据 K0 workspace=0 推断公开 API 总 workspace 恒为0。
- A2/A3 的 ACLNN 文档支持表与 Polar 顶层 README 冲突，且非 RegBase 组合图的被调 Kernel/Host 平台接线无法从固定 Commit 闭合，因此没有创建 `polar.md`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/polar/docs/aclnnPolar.md` | 产品支持表、函数原型、参数约束 | 950 公开性、两段式 API、dtype/format/rank/错误语义 |
| 入口声明/约束 | `math/polar/op_api/aclnn_polar.h` | `aclnnPolarGetWorkspaceSize/aclnnPolar` | C ABI 与参数顺序 |
| 入口实现 | `math/polar/op_api/aclnn_polar.cpp` | `CheckParams/aclnnPolarGetWorkspaceSize/aclnnPolar` | 参数检查、空域、平台分支、任务图、workspace/executor |
| L0 launcher | `math/polar/op_api/polar.cpp` | `Polar/PolarAiCore/IsAiCoreSupport` | 广播临时输出、dtype selector、AICore launch |
| Host 注册 | `math/polar/op_host/polar_def.cpp` | `ops::Polar::Polar` | FLOAT/COMPLEX64 ND 与 ascend950 `polar_apt` 接线 |
| Host/Tiling | `math/polar/op_host/arch35/polar_tiling.h` | `PolarCompileInfo/PolarTiling` | 单模板、key0、类字段和基类覆盖 |
| Host/Tiling | `math/polar/op_host/arch35/polar_tiling.cpp` | `PolarTiling::*`、注册函数 | 平台 AIV 核数、广播 stride、分核、workspace0、POD 写入 |
| Host/Tiling 直接依赖 | `opbase/include/op_common/op_host/tiling_base_class.h` | `TilingBaseClass::DoTiling` | 固定 Tiling 阶段顺序、key 写入与失败传播 |
| POD | `math/polar/op_kernel/arch35/polar_struct.h` | `PolarTilingData/POLAR_MAX_DIM` | Host/Device ABI、字段顺序、8 维上限 |
| Kernel entry | `math/polar/op_kernel/polar_apt.cpp` | `polar` | 参数顺序、POD 解析、固定 FP32 body |
| Kernel body | `math/polar/op_kernel/arch35/polar_simt.h` | `PolarSimt/SimtPolarCompute` | SIMT 线程、地址、广播、数学运算、输出与 tail |
| 构建接线 | `math/polar/op_host/config/ascend950/polar_binary.json` | `Polar_5a8c...` | 950 FP32/COMPLEX64 ND binary |
| 构建接线 | `math/polar/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | 仅 ascend950、arch35 Tiling |

每行文件都已列入 Frontmatter `sources`；`TilingBaseClass` 是 Polar 直接 include 的公共基类，来源为 opbase 固定 Commit `d574f5571391390f8b619c6a134b65c819842318`，其余均为 ops-math 固定 Commit `165d41f02a43a62324b434cb6bda7c533b14da3e`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 5 个标量加 4×8 stride 的 296B pack8 结构 | 字段顺序、int64 位宽、数组长度、元素单位和未用槽默认值必须一致 | `PolarTilingData`、`PostTiling`、`PolarSimt::Init` |
| 调用顺序 | 两路 Contiguous→Polar→ViewCopy；Tiling Shape→Platform→OpTiling→Workspace→Post→key | K0 必须等两路连续输入；F2 必须等 K0；任一失败阻断后续 | `aclnnPolarGetWorkspaceSize`、`TilingBaseClass::DoTiling` |
| workspace/内存 | K0 直接 GM 读写且自有 workspace 0；广播 stride0 | 不得把 K0 的 0B 外推为 API 总 workspace；输出必须按 `2*yBase(+1)` 交错写 | `PostTiling`、`SimtPolarCompute` |
| 同步 | 节点内无 barrier/atomic；线程写区互斥；节点间由 executor stream 排序 | 不得让 ViewCopy 提前消费，也不得让两个输出线程写同一 `yBase` | `Process/SimtPolarCompute`、`CommonOpExecutorRun` |
| 精度与尾块 | FP32 Sin/Cos 与乘法；former core 均衡分域；1024-thread while guard | 不得改变实部/虚部顺序、线程步长或 `i<count` 保护 | `DoOpTiling`、`Process/SimtPolarCompute` |
| 失败边界 | 空域成功无 launch；非法 dtype/shape/format、coreNum≤0、launcher 失败均终止 | 不得把非 RegBase 组合图或无证据的 A2/A3 原生实现当作本 Route fallback | `CheckParams`、`TilingPrepare4PolarAscendc`、`PolarAiCore` |
