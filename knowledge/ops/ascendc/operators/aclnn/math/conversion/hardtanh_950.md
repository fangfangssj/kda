---
type: Operator
title: "Hardtanh 950 ACLNN 算子知识"
description: "说明 aclnnHardtanhGetWorkspaceSize/aclnnHardtanh 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/docs/aclnnHardtanh&aclnnInplaceHardtanh.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/aclnn_hardtanh.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/aclnn_hardtanh.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/clip_by_value.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/config/ascend950/clip_by_value_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-24T23:51:11Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Hardtanh 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnHardtanhGetWorkspaceSize` / `aclnnHardtanh`，公开两段式 ACLNN；不含 `aclnnInplaceHardtanh` |
| 计算语义 | `y_i = min(max(x_i, clipValueMin), clipValueMax)`，API 先保证 `clipValueMin <= clipValueMax` |
| 输入输出 | `self/out` 同 shape、同 dtype；公开 ND、rank 1～8、允许非连续；上下界是 Host `aclScalar`，入图前转成 `self` dtype 的单元素 Tensor |
| 执行类型 | R-001 单 Kernel `AIV_VECTOR`；R-002 单 AICPU CPU Kernel；AIV entry 显式拒绝 AIC 并声明 `KERNEL_TYPE_AIV_ONLY`，无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 950 AIV 标量上下界 ASS；R-002 AICPU 标量上下界串行/并行 |
| 核心资源 | R-001 GM + 两个 depth-2 `TQue`，UB 为 `4 * ubFormer * sizeof(T)`；R-002 直接读写 AICPU 可见 Tensor 内存，无 AscendC UB/POD |
| 来源闭包 | `source_closed`：公开 ACLNN→Contiguous/ScalarToTensor→`ClipByValueV2` selector→AIV Tiling/POD/entry/body 或 AICPU body→ViewCopy→executor 均已定位；Contiguous/ViewCopy/executor 内部是许可的框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `self` 的每个有效元素执行：

```text
lower_i = max(self_i, clipValueMin)
out_i   = min(lower_i, clipValueMax)
```

`clipValueMin` 与 `clipValueMax` 是标量。第一段接口先按标量自身 dtype 比较并拒绝 `min > max`，随后把二者转换成 `self` dtype 的单元素 Tensor。空 `self` 直接返回 `workspaceSize=0`，不构造 ClipByValueV2 Device 节点。非原地接口最终经 `ViewCopy` 写入独立 `out`；本卡不覆盖 `aclnnInplaceHardtanh`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；rank 1～8；允许非连续 | 950 API：FLOAT16、FLOAT、BFLOAT16、DOUBLE、INT8、UINT8、INT16、INT32、INT64 | 非空时先 `Contiguous`；是公式中的 `x` |
| `clipValueMin` | Host scalar，转为单元素 Tensor | 必须在平台支持表内并可转成 `self` dtype | 下界；实际指针必填，虽然头文件说明文字提到 `-1.0` 默认值，但实现对空指针直接报错，未实现省略参数 |
| `clipValueMax` | Host scalar，转为单元素 Tensor | 同上 | 上界；实际指针必填；要求 `clipValueMin <= clipValueMax` |
| `out` | shape 必须等于 `self`；允许非连续 | 必须与 `self` 相同 | 非原地输出；由 `ViewCopy` 提交 |
| `workspaceSize` | 标量输出 | `uint64_t*` | executor 全流程 workspace；空输入为 0 |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有 Contiguous、ScalarToTensor、ClipByValueV2、ViewCopy 的任务与临时量生命周期 |
| `workspace/workspaceSize/stream` | Device 地址、字节数、ACL stream | 两段式运行参数 | 必须使用第一段返回的大小和 executor，在指定 stream 上运行 |

### 入口与版本

公开入口是 CANN 9.1 的 `aclnnHardtanhGetWorkspaceSize` 与 `aclnnHardtanh`。第一段调用共享的 `GetWorkspaceSizeCommon`，第二段调用 `CommonOpExecutorRun`。固定仓同文件还含原地版本，但本逻辑算子只取非原地入口，不能把 `selfRef` 别名语义并入本卡。

950 由公开产品表、`ClipByValueV2` 的 `ascend950` OpDef、模块 `SUPPORT_COMPUTE_UNIT "ascend950"`、950 binary config 以及全局 AscendC binary config 共同闭合。无后缀 A2/A3 卡没有生成：公开表声称 A2/A3 支持，但固定仓中 `ClipByValueV2` AICore OpDef、模块构建和 binary config 都只接到 `ascend950`；而 L0 对五种 AICore dtype 只按 dtype 选择 AICore、没有非 950 平台 guard。该平台实现缺口不足以发布 `source_closed` A2/A3 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`GetWorkspaceSizeCommon` 的严格顺序为：

1. `CHECK_NOT_NULL(self, clipValueMin, clipValueMax, out)`；executor 创建失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. `CheckDtypeValid` 依据 `GetCurrentPlatformInfo().GetCurNpuArch()` 选择 dtype 表；950/RegBase 使用含 BF16 的九种 dtype 表，并要求 `self/out` dtype 相同。
3. `CheckShape` 限制 `self` rank 不超过 8 且 `self/out` shape 相同；`FRACTAL_NZ` 只记录潜在精度告警，不在这里失败。
4. `CheckMinMaxValue` 以标量原 dtype 的 `ToInt64` 或 `ToDouble` 检查 `min <= max`。
5. 空 Tensor：写 `workspaceSize=0`、释放 executor，结束；没有 Device launch。
6. 非空 Tensor：`l0op::Contiguous(self)`；把两个 scalar 分别 `ConvertToTensor(..., self->GetDataType())`；调用 `l0op::ClipByValueV2`；最后 `l0op::ViewCopy(hardtanhResult, out)`。
7. 读取 executor 汇总的 workspace，转移 executor 所有权。任一步返回空 Tensor/失败码都会立即阻断后续任务。

### 参数、Workspace 与生命周期

公开第一段返回：

```text
workspaceSize = executor.GetWorkspaceSize()
              = W_contiguous + W_ClipByValueV2 + W_viewCopy + W_framework
```

该总和由框架 executor 计算，固定仓未公开各框架节点的切片 offset。R-001 的 `ClipByValueV2` Tiling 明确为其节点申报 `16 * 1024 * 1024` 字节；AIV body 接收 `workspace` 但不读。R-002 AICPU body没有自定义 workspace 或 POD，任务空间由 `AicpuTaskSpace("ClipByValueV2")` 与 executor 管理，其精确字节布局固定仓不可见。

第一段拥有临时连续 Tensor、两个标量 Tensor、中间 `hardtanhResult` 与 task descriptors；`ReleaseTo(executor)` 后由调用方保存 executor。第二段把调用方 workspace、大小、executor 与 stream 交给 `CommonOpExecutorRun`；完成前不得释放或覆盖它们。

### 启动图与失败传播

非空输入的逻辑启动图为：

```text
F0 optional Contiguous(self)          # self 非连续时物化，框架边界
  -> K0 ClipByValueV2                 # R-001 AIV 或 R-002 AICPU
    -> F1 ViewCopy(result, out)        # 连续结果提交到 out，框架边界
```

| 节点 | Host 添加函数 | entry/body | launch 参数 | 输入→输出 | workspace | 失败传播 |
|---|---|---|---|---|---|---|
| F0 | `l0op::Contiguous` | 固定仓外框架实现 | executor/stream | `self`→连续临时 Tensor | executor 切片不可见 | 返回空指针则不添加 K0 |
| K0/R-001 | `ADD_TO_LAUNCHER_LIST_AICORE(ClipByValueV2,...)` | `clip_by_value_v2`→`ClipByValueV2Ass<T>` | `blockDim=blockNum`、key `2001100`、72B POD | 连续 `x`+两个单元素 bounds→连续 `y` | Tiling 申报 16MiB，body 不读 | infer/launcher/Tiling失败阻断 F1 |
| K0/R-002 | `ADD_TO_LAUNCHER_LIST_AICPU(ClipByValueV2,...)` | `ClipByValueV2CpuKernel::Compute` | AICPU context，无 TilingKey/POD | 同上 | body 无显式 slice | 参数或 `ParallelFor` 失败返回并阻断 F1 |
| F1 | `l0op::ViewCopy` | 固定仓外框架实现 | executor/stream | 连续 `hardtanhResult`→用户 `out` view | executor 切片不可见 | 返回空指针使第一段失败 |

F0、K0、F1 都由同一个 executor 在调用者 stream 上提交，任务顺序建立 producer→consumer 可见性；固定仓没有算子自有 event、跨 stream 或第二个核心语义 Kernel。中间 Tensor 由 executor 持有，F1 消费结束后才能回收。

## Route 与参数设计

### 分核与资源计算

R-001 的 `TilingPrepareForClipByValueV2` 生产 AIV core 数与 UB 字节数。由于 Hardtanh 两个 bounds 都是单元素 Tensor，`DimensionCollapse` 后四个输入输出都是单段；priority 1 的 `ClipByValueTilingSingleDim` 设置 `isSigDim=true`、`isAss=true`，命中 ASS key。

设 `C=GetCoreNumAiv()`、`U=UB bytes`、`S=sizeof(T)`、`N=self` 元素数：

```text
aliveNum     = 2                         # input queue + output queue
ubFormerByte = floor((U / 2 / aliveNum) / 128) * 128
ubFormer     = ubFormerByte / S
ubOuter      = ceil_div(N, ubFormer)
ubTail       = N % ubFormer; 0 -> ubFormer
blockFormer  = ceil_div(ubOuter, C)
blockTail    = ubOuter % blockFormer; 0 -> blockFormer
blockNum     = ceil_div(ubOuter, blockFormer)
```

若初算 `blockNum < C/2` 且 `ubFormer*S*aliveNum > 8KiB`，Tiling 将每核目标改成约 `2*N/C` 个元素、向上对齐到 128B，并以每 alive node 至少 `8KiB/aliveNum` 为下界，然后重算分核。`ubFormer==0`、`C==0` 或 `N==0` 会使 Tiling 失败；API 空 Tensor 已更早返回。

R-002 不使用 AIV 分核。AICPU 令 `N=NumElements(x)`、`B=N*sizeof(T)`；`N==1` 直接算一个元素，`B<200KiB` 串行，否则保留 2 个 CPU 并用 `ParallelFor` 分片。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 AIV ASS | L0 AICore；Tiling priority 1；key `2001100`；`ClipByValueV2Ass<DTYPE_X>` | 非空且 `self.dtype ∈ {FLOAT16,FLOAT,BFLOAT16,INT32,INT64}`；Hardtanh bounds 均为 scalar，合轴后 `isSigDim && minDim==1 && maxDim==1` | dtype 不在 AICore 表→R-002；Tiling 不成立→失败，无 Device fallback |
| 2 | R-002 AICPU scalar | `ClipByValueV2CpuKernel::Compute`→`DoCompute<T>` | 非空且 API 已接受、但 dtype 不在 AICore 表；950 即 `{DOUBLE,INT8,UINT8,INT16}` | 不支持 dtype 已被 ACLNN 校验拒绝 |

```text
if self.empty:
    workspaceSize=0; no Device route
else if self.dtype in {f16,f32,bf16,s32,s64}:
    ClipByValueV2AiCore
    -> priority 1 SingleDim template
    -> scalar bounds make isAss=true
    -> key 2001100 -> R-001
else:
    ClipByValueV2AiCpu -> R-002
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 AIV ASS | L0 AICore + priority1 + `2001100` | 950；五种 AICore dtype；`self/out` 同 shape；bounds scalar；非空 | 单 K0 AIV；外围可有 F0/F1 | `ClipByValueV2Ass<DTYPE_X>::Init/Process` | 两个 depth-2 Queue；Queue producer/consumer；workspace 16MiB 不读 | 最后一核取 `blockTail` loops，最后 loop 取 `ubTail`；失败无 fallback |
| R-002 AICPU scalar | L0 `else` | 950；`DOUBLE/INT8/UINT8/INT16`；同 shape；bounds scalar；非空 | 单 K0 AICPU；外围可有 F0/F1 | `DoCompute<double/int8_t/uint8_t/int16_t>`→`KernelScalarBound` | 直接内存；小数据串行，大数据 `ParallelFor`；无 UB/POD | `N==1` 特判；分片为 `[beg,end)`，无越界 tail；ParallelFor错误上抛 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台二进制 | `ascend950` | 只有 950 OpDef、CMake 与 binary config | R-001 | 非 950 不发布本实现 | `ClipByValueV2::ClipByValueV2`、模块 CMake |
| AIV-only | entry 中 `g_coreType == AIC` 立即返回；`KERNEL_TYPE_AIV_ONLY` | ASS 只使用 Vector/UB | R-001 | AIC 无 body；Host 必须按 AIV 核数 launch | `clip_by_value_v2` |
| dtype selector | `CheckType(dtype,AICORE_DTYPE_SUPPORT_LIST)` | AIV binary 仅五种 dtype | 两 Route | false→R-002 AICPU | `IsAiCoreSupport` |
| scalar专用 | `isSigDim && minDim==1 && maxDim==1` | bounds 可各读一次并用 `Maxs/Mins` | R-001 | priority1不命中时才尝试其它模板；Hardtanh固定命中 | `DetermineSigDimMode/GetTilingKey` |
| UB正值 | `ubFormer != 0` | depth-2 input/output Queue 必须各有 tile | R-001 | Tiling失败 | `CalcInitialUbParams` |
| 低利用率重切 | `blockNum<C/2 && ubFormer*S*2>8KiB` | 增加可用 AIV 核数，同时保持每核最低 UB 工作量 | R-001 | 条件 false 保留初算 | `AdjustForLowUtilization` |
| AICPU并行阈值 | `N*S >= 200KiB` | 小任务避免 ParallelFor 调度 | R-002 | false 串行执行同 body | `DispatchParallel` |
| 空域 | `self->IsEmpty()` | 防止零步长 Tiling/无效 launch | 两 Route | 第一段成功且 workspace 0 | `GetWorkspaceSizeCommon` |

### POD/ABI

R-001 的 `ClipByValueSigDimTilingData` 由 9 个连续 `int64_t` 字段组成，按声明顺序可得 `sizeof=72`、`alignof=8`、offset 每次递增 8B；R-002 无 Tiling POD。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ClipByValueSigDimTilingData` | `blockNum` | `int64_t`/64 | size72 align8 offset0 | AIV core | `set_blockNum(blockNum)` | `Init`: 越界核 guard、尾核判断 | 必须 >0 |
| 同上 | `ubFormer` | `int64_t`/64 | offset8 | 元素/tile | `set_ubFormer(ubFormer)` | Queue容量、整 tile 长度、core offset | 由 128B 对齐字节数除 `S` |
| 同上 | `ubTail` | `int64_t`/64 | offset16 | 元素 | `set_ubTail(ubTail)` | 最后一核最后 loop | 余数 0 时写 `ubFormer` |
| 同上 | `blockFormer` | `int64_t`/64 | offset24 | tile/core | `set_blockFormer(blockFormer)` | 普通核 loop 数、`gmBlockOffset` | ceil 分配 |
| 同上 | `blockTail` | `int64_t`/64 | offset32 | tile | `set_blockTail(blockTail)` | 最后一核 loop 数 | 余数 0 时写 `blockFormer` |
| 同上 | `xDim` | `int64_t`/64 | offset40 | 合轴后元素 | `set_xDim(xDim)` | ASS `Init` 保存但 body 不分支读取 | Hardtanh为 `N`，在此 Route inactive |
| 同上 | `minDim` | `int64_t`/64 | offset48 | 合轴后元素 | `set_minDim(1)` | ASS body不读取；key 已编码 scalar | 固定为1，inactive字段 |
| 同上 | `maxDim` | `int64_t`/64 | offset56 | 合轴后元素 | `set_maxDim(1)` | ASS body不读取 | 固定为1，inactive字段 |
| 同上 | `yDim` | `int64_t`/64 | offset64 | 合轴后元素 | `set_yDim(N)` | ASS body不读取；Host用于分核 | 必须 >0 |
| R-002 | N/A | N/A | 无 POD | N/A | AICPU context 直接提供 Tensor | `Compute` 读取 context | 不得把 AIV POD 传给 AICPU |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 AIV ASS | AICore dtype + priority1 + `2001100` | `op_api/clip_by_value.cpp::IsAiCoreSupport`；`clip_by_value_single_dim_tiling.cpp` | `ClipByValueSigDimTilingData` | `clip_by_value_v2_apt.cpp::clip_by_value_v2` | `clip_by_value_v2_ass.h::ClipByValueV2Ass` | v2 OpDef/CMake/950 binary/ascendc config |
| R-002 AICPU scalar | L0 dtype `else` | `ClipByValueV2AiCpu`；AICPU context | Tensor descriptors/data，无 POD | `REGISTER_CPU_KERNEL`→`ClipByValueV2CpuKernel::Compute` | `DoCompute/DispatchParallel/KernelScalarBound/ClampScalar` | `clip_by_value_v2_aicpu_def.cpp::ApplyMathAicpuDefaultCfg` |

## Kernel 实现

### 公共执行流程

两条 Route 都由同一 ACLNN 前处理生成连续 `x`、两个转换到 `self` dtype 的单元素 bounds，并生成连续中间输出；L0 只按 `self.dtype` 选择 AICore 或 AICPU。两条路线都只有一个核心语义 K0，直接生成每个 `y_i`，随后共享框架 `ViewCopy` 边。

R-001 的 entry 在 AIC 上立即返回，AIV 上解析 key/POD并构造 ASS body；未知 key直接返回，无 fallback。R-002 没有 Tiling router：AICPU registry 直接调用 `Compute`，其 dtype switch 只会看到 ACLNN 已放行且 L0 未交给 AICore 的四种类型。

### Route R-001：AIV 标量上下界 ASS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/clip_by_value_v2/op_api/aclnn_hardtanh.cpp` | `GetWorkspaceSizeCommon` | 非空；scalar 均 `ConvertToTensor(..., self dtype)` | 构造两个单元素 bounds |
| L0 | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `IsAiCoreSupport` | dtype 为 FP16/FP32/BF16/INT32/INT64 | `ClipByValueV2AiCore` |
| Tiling | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `DetermineSigDimMode` | 四个 collapse dims 均单段，`minDim=maxDim=1` | `isSigDim=true,isAss=true` |
| Tiling | 同上 | `IsCapable/GetTilingKey` | priority 1 capable 且 `isAss` | key `2001100` |
| Router | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | AIV + `TILING_KEY_IS(2001100)` | `ClipByValueV2Ass<DTYPE_X>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(ClipByValueV2, x, minTensor, maxTensor, y)
  -> clip_by_value_v2(x,min,max,y,workspace,tiling)
    -> GET_TILING_DATA_WITH_STRUCT(ClipByValueSigDimTilingData)
    -> ClipByValueV2Ass<T>::Init
    -> ClipByValueV2Ass<T>::Process
      -> CopyIn -> Compute(Maxs then in-place Mins) -> CopyOut
  -> executor stream edge
ViewCopy(y, out)
```

##### 源码等价伪码

```text
# 伪码；单位除注明外均为元素
if coreType == AIC: return
if key != 2001100: return

b = GetBlockIdx()
if b >= blockNum: return
loops = (b == blockNum-1) ? blockTail : blockFormer
lastLen = (b == blockNum-1) ? ubTail : ubFormer
base = b * blockFormer * ubFormer
minValue = minGM[0]
maxValue = maxGM[0]
InitBuffer(inQueueX, depth=2, ubFormer*S bytes each)
InitBuffer(outQueue, depth=2, ubFormer*S bytes each)

for i in [0, loops-1):
    len = (i == loops-1) ? lastLen : ubFormer
    off = i * ubFormer
    xLocal = Alloc(inQueueX)
    DataCopyPad(xLocal, xGM[base+off], blockCount=1,
                blockLen=len*S bytes, srcStride=0, dstStride=0, no pad)
    EnQue(inQueueX, xLocal)
    xLocal = DeQue(inQueueX)
    yLocal = Alloc(outQueue)
    Maxs(yLocal, xLocal, minValue, len)
    Free(inQueueX, xLocal)
    Mins(yLocal, yLocal, maxValue, len)      # 同一输出buffer原地封上界
    EnQue(outQueue, yLocal)
    yLocal = DeQue(outQueue)
    DataCopyPad(yGM[base+off], yLocal, blockLen=len*S bytes)
    Free(outQueue, yLocal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `base+i*ubFormer` | 有效 `len` | T/flattened ND | F0/self | `CopyIn` | executor/调用方；K0只读 | 整 tile 128B 对齐，tail 用 DataCopyPad |
| GM `min/max` | 0 | 各1元素 | T/scalar | ConvertToTensor | `Init::GetValue(0)` | executor；Init后值驻留标量成员 | 每核各读一次 |
| `inQueueX` | Queue私有 | depth2 × `ubFormer*S` | T/连续 | `CopyIn` | `Compute` | TPipe；每 tile Alloc→EnQue→DeQue→Free | 双缓冲；Free后 slot 可复用 |
| `outQueue` | Queue私有 | depth2 × `ubFormer*S` | T/连续 | `Compute` | `CopyOut` | TPipe；每 tile Alloc→EnQue→DeQue→Free | `Maxs`结果被`Mins`原地覆盖 |
| GM `y` | `base+i*ubFormer` | 有效 `len` | T/flattened ND | `CopyOut` | F1 ViewCopy | executor 中间 Tensor | 每核区间不重叠 |
| node workspace | executor slice | 16MiB | bytes | Tiling申报 | body不读取 | executor | 固定仓不显示其系统用途 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ0["inQueueX slot0 · ubFormer*S"]:1
  inQ1["inQueueX slot1 · ubFormer*S"]:1
  outQ0["outQueue slot0 · ubFormer*S"]:1
  outQ1["outQueue slot1 · ubFormer*S"]:1
```

##### 字节账与复用顺序

```text
inQueueX = 2 * ubFormer * S
outQueue = 2 * ubFormer * S
UB_total = 4 * ubFormer * S
         <= U                         # ubFormer=floor((U/4)/128)*128/S
dynamic_ub = N/A
workspace_total(node) = 16 * 1024 * 1024 bytes
=> ubFormer>0，整tile容量按128B向下对齐
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueueX` slot | `CopyIn` | `Compute::DeQue` | `FreeTensor` 后下一 tile | EnQue/DeQue/FreeTensor | 不得在 Compute 完成前复用 |
| `outQueue` slot | `Maxs`，随后 `Mins` 原地更新 | `CopyOut::DeQue` | `FreeTensor` 后下一 tile | EnQue/DeQue/FreeTensor | `Mins` 必须在 EnQue 前完成 |
| scalar成员 | `Init::GetValue(0)` | 每次 `Maxs/Mins` | Kernel结束 | 只读成员 | bounds 在 K0 全生命周期有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + `EnQue` | MTE/CopyIn | `DeQue` in Compute | input slot | `blockLen=len*S` 字节；无显式 padding |
| 下界→上界 | 顺序 Vector 指令 | `Maxs` | 同 buffer `Mins` | `yLocal` | 全程 T dtype，无升精度/cast；固定仓未说明 Vector NaN比较细则 |
| UB→GM | `EnQue/DeQue` + `DataCopyPad` | Compute | CopyOut | output slot | 仅写 `len` 个有效元素 |
| 分核 | 独占连续区间 | Host block split | 每个 AIV | GM output | 最后一核使用 `blockTail`，其最后 tile 使用 `ubTail` |
| K0→F1 | executor 同 stream 顺序 | K0 | ViewCopy | 连续结果 Tensor | 无算子自有 event/atomic/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `DetermineSigDimMode/CalcInitialUbParams/CalcBlockParams/PostTiling` | scalar ASS key、UB/core公式、72B POD与16MiB workspace |
| Kernel entry/router | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` key `2001100` | AIC返回、AIV-only、POD解析与模板实例 |
| Device body/helper | `conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h` | `ClipByValueV2Ass<T>::Init/Process/CopyIn/Compute/CopyOut` | core/tile offset、scalar缓存、Queue、Maxs/Mins与tail |

### Route R-002：AICPU 标量上下界

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/clip_by_value_v2/op_api/aclnn_hardtanh.cpp` | `CheckDtypeValid` | 950允许 DOUBLE/INT8/UINT8/INT16 | 继续构图 |
| L0 | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `IsAiCoreSupport/ClipByValueV2` | dtype 不在五种 AICore表 | `ClipByValueV2AiCpu` |
| Launcher | 同上 | `ClipByValueV2AiCpu` | `AicpuTaskSpace("ClipByValueV2")` | 添加 AICPU task |
| Router | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `Compute/DispatchByDtype` | 四种 dtype switch | `DoCompute<T>` |
| Body | 同上 | `DoCompute` | bounds `NumElements()==1` | `KernelScalarBound` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICPU(ClipByValueV2, x, minTensor, maxTensor, y)
  -> ClipByValueV2CpuKernel::Compute(ctx)
    -> NormalCheck -> GetInputAndCheck
    -> DispatchByDtype
      -> DoCompute<T>
        -> N==1 ? ClampScalar
        -> else DispatchParallel("real-scalar", threshold=200KiB)
          -> KernelScalarBound<T>
            -> ClampScalar<T>
  -> executor stream edge
ViewCopy(y, out)
```

##### 源码等价伪码

```text
# 伪码
validate 3 inputs, 1 output, equal dtype, x/y element count
validate both bounds scalar or both full-shape        # Hardtanh固定scalar
if N == 0: return OK                                 # API已更早拦截
switch dtype: double/int8/uint8/int16 -> DoCompute<T>

mn = minPtr[0]
mx = maxPtr[0]
if N == 1:
    y[0] = ClampScalar(x[0], mn, mx)
elif N*S < 200KiB:
    for i in [0,N): y[i] = ClampScalar(x[i],mn,mx)
else:
    cpu = GetCPUNum(ctx)
    cores = max(1, max(cpu,2)-2)
    perUnit = CeilMultiple(N, cores)
    grainBytes = perUnit*S
    ParallelFor(ctx,N,grainBytes,[beg,end):
        for i in [beg,end): y[i] = ClampScalar(x[i],mn,mx))

ClampScalar(float/double):
    if isnan(x) || isnan(mn) || isnan(mx): return quiet_NaN
    return min(max(x,mn),mx)
ClampScalar(integer): return min(max(x,mn),mx)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input `x` | `x_ptr+i` | N元素 | T/连续 | F0/self | worker `[beg,end)` | executor/调用方，只读 | 分片互不重叠 |
| scalar bounds | index0 | 各1元素 | T | ConvertToTensor | `mn_v/mx_v` | executor；进入 worker 前按值捕获 | 每 task 预读一次 |
| output `y` | `y_ptr+i` | N元素 | T/连续 | worker | F1 | executor中间 Tensor | 每索引唯一 writer |
| AICPU局部状态 | C++标量/闭包 | `mn_v,mx_v,beg,end` | registers/stack，精确字节固定仓不可见 | `DoCompute/ParallelFor` | `KernelScalarBound` | 每 task/shard | 无共享写冲突 |
| workspace | N/A | body 无显式申请 | N/A | N/A | N/A | AICPU task/executor内部固定仓不可见 | 不臆造 UB/TBuf |

##### 分层内存布局

N/A：R-002 不使用 AscendC AIV UB、`TQue`、`TBuf` 或 dynamic UB；实际路径是 AICPU 直接访问输入、单元素 bounds 和输出 Tensor 内存，线程只保留标量与区间。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
explicit_device_workspace_in_body = 0
executor/AICPU task metadata bytes = 固定仓不可见
parallel threshold = N * sizeof(T) >= 200 * 1024 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `mn_v/mx_v` 值捕获 | `DoCompute` | 每个 shard | task结束 | C++闭包生命周期 | bounds 必须在 worker创建前读出 |
| 输出分片 `[beg,end)` | 当前 worker | F1 | K0完成后 | `ParallelFor` 返回 | ParallelFor 必须给出不重叠区间 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 小任务 | 直接函数调用 | `DoCompute` | 返回 | 全输出 | `N==1`特判；否则 `[0,N)` |
| 大任务 | `CpuKernelUtils::ParallelFor` | shard workers | `DispatchParallel` 返回 | 各自输出分片 | grain由 `perUnit*S` 产生；最后 shard由 `[beg,end)`自然截断 |
| 浮点比较 | `isnan` + `std::min/max` | `ClampScalar` | output store | 当前元素 | DOUBLE 显式传播任一操作数 NaN；无额外 cast |
| 整数比较 | `std::min/max` | `ClampScalar` | output store | 当前元素 | INT8/UINT8/INT16 原 dtype比较 |
| K0→F1 | executor同stream | AICPU task | ViewCopy | output Tensor | ParallelFor错误码上抛，不执行成功后继 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `ClipByValueV2AiCpu` | AICPU task space、输入输出与失败检查；无 Tiling |
| Kernel entry/router | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `Compute/DispatchByDtype` | 参数、dtype与空域分发 |
| Device body/helper | 同上 | `DoCompute/DispatchParallel/KernelScalarBound/ClampScalar` | scalar预载、200KiB阈值、CPU保留、分片、NaN与tail |

## 算子特定优化

### Trick T-001：Host 将双标量 bounds 专门化为 ASS key

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：合轴后的四个 Tensor 都是一维段，且 `minDim==1 && maxDim==1`
- Golden 来源：[`conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp:195-209`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp#L195-L209)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
void ClipByValueTilingSingleDim::DetermineSigDimMode(const std::vector<std::vector<int64_t>>& dims)
{
    isSigDim = false;
    if (dims[ClipByValueSigDim::X_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::MIN_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::MAX_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::Y_INDEX].size() == 1) {
        isSigDim = true;
        xDim = dims[ClipByValueSigDim::X_INDEX].front();
        minDim = dims[ClipByValueSigDim::MIN_INDEX].front();
        maxDim = dims[ClipByValueSigDim::MAX_INDEX].front();
        yDim = dims[ClipByValueSigDim::Y_INDEX].front();

        isAss = (minDim == 1 && maxDim == 1);
```

#### 机制与设计原因

Hardtanh 固定把两个 Host scalar 转为单元素 Tensor，因此 Host 能在 Tiling 阶段选择 key `2001100`，跳过通用广播的维度/stride POD、NDDMA 或 UB broadcast body。ASS body每核只在 `Init` 读取两个标量，然后以 `Maxs/Mins` 处理连续 tile；这减少 bounds 的 GM 搬运和通用广播索引计算。该结论是源码结构收益，不代表已测得性能比例。

#### 迁移边界

- 适用：一个大 Tensor 被两个 scalar bounds 截断的逐元素算子。
- 前提：bounds 经 Host 明确转成单元素、dtype与 `x/y` 相同，平台存在对应 scalar Vector 指令。
- 失效：任一 bound 变成逐元素或一般广播 Tensor 时必须进入 single-dim/general broadcast 的其它实现，不能复用 ASS POD/body。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 `minDim/maxDim` 的 scalar 判定省略、却仍使用 `GetValue(0)` 会错误地忽略非标量 bounds；固定仓通过 `isAss` 与独立 key 阻止这种误用。

### Trick T-002：输出 Queue 内原地封上界

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：key `2001100` 的 `ClipByValueV2Ass<T>::Compute`
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h:115-123`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h#L115-L123)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ClipByValueV2Ass<T>::Compute(const int64_t& len)
{
    auto xLocal = inQueueX.template DeQue<T>();
    auto yLocal = outQueue.template AllocTensor<T>();
    Maxs(yLocal, xLocal, minValue, len);
    inQueueX.FreeTensor(xLocal);
    Mins(yLocal, yLocal, maxValue, len);
    outQueue.EnQue(yLocal);
}
```

#### 机制与设计原因

`Maxs` 直接把下界截断结果写进最终 output Queue slot，随即以同一 `yLocal` 作为 `Mins` 的 dst/src 原地封上界；输入 slot 在 `Maxs` 后立即释放。这样 UB 只需 input/output 两类 Queue，不物化第三个中间 Tensor。`len` 来自当前整 tile 或 `ubTail`，两条指令只处理有效元素。

#### 迁移边界

- 适用：第二个逐元素阶段允许 dst/src 别名，且第一阶段结果之后不再需要。
- 前提：Vector primitive 支持该原地形式，Queue slot 在 `Mins` 完成前不 EnQue/覆盖。
- 失效：第二阶段不支持别名或后续仍需第一阶段结果时必须增加独立 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在 `Mins` 前 EnQue 或复用 `yLocal` 会破坏 producer→consumer 顺序；固定实现只在两次 Vector 指令都完成后 EnQue。

### Trick T-003：AICPU 标量预载并按字节阈值避开小任务并行

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：bounds 为 scalar；`N>1`，由 `DispatchParallel` 再按 `N*sizeof(T)` 与 200KiB 比较
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp:163-173`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp#L163-L173)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr int64_t elem_bytes = static_cast<int64_t>(sizeof(T));

if (is_scalar) {
    // Specialized branch: min/max broadcast; preload the scalars into local registers.
    const T mn_v = mn_ptr[0];
    const T mx_v = mx_ptr[0];
    auto body = [x_ptr, y_ptr, mn_v, mx_v](int64_t b, int64_t e) {
        KernelScalarBound<T>(x_ptr, y_ptr, b, e, mn_v, mx_v);
    };
    return DispatchParallel(ctx, "real-scalar", data_num, elem_bytes, kParallelBytesThresh, body);
}
```

#### 机制与设计原因

上下界先按值读入并捕获，各 worker 不再逐元素读两个 bounds。分发器以真实元素字节数控制串行/并行：小于 200KiB 直接调用相同 body；大任务才查询 CPU 数、保留两个 CPU 并执行 `ParallelFor`。这同时减少标量内存读取和小任务调度开销；收益未做真机量化。

#### 迁移边界

- 适用：AICPU 上 bounds/系数为只读 scalar 的纯逐元素循环。
- 前提：每个输出索引唯一写入，scalar 在 worker 生命周期不变。
- 失效：bounds 为逐元素 Tensor时必须走 `KernelElemBound`；阈值是本实现常量，迁移平台不能无证据照搬。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证的不同阈值反例；不得把 200KiB 推广成其它算子的通用最佳值。

## 模板与实现对比

| 对比维度 | R-001 AIV ASS | R-002 AICPU scalar | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 五种 AICore dtype；Tiling priority1 scalar ASS | API剩余四种 dtype | L0 dtype能力表先分设备族 |
| 模板与 dtype | `ClipByValueV2Ass<DTYPE_X>`；FP16/FP32/BF16/S32/S64 | `DoCompute<T>`；double/int8/uint8/int16 | binary/primitive支持集合不同 |
| 循环与分核 | AIV连续 tile，former/tail core | AICPU串行或 `[beg,end)` shards | UB容量 vs CPU调度阈值 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 两个 depth-2 Queue；scalar成员；无TBuf/dynamic UB/通信 | C++标量/闭包；无AscendC UB | 执行族真实资源 |
| 内存总占用与复用 | `4*ubFormer*S`；output slot原地Maxs→Mins | body显式额外buffer 0；输出分片独占 | scalar specialization避免bounds物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、Maxs、Mins | 直接load/store、std::min/max、ParallelFor | 无SIMT/NDDMA/通信/Cube |
| 同步、精度与 tail | Queue同步；原dtype；最后核/最后tile tail | ParallelFor join；DOUBLE显式NaN；区间end tail | 各执行族保护方式 |
| fallback/失败边界 | unknown key返回；Tiling失败无AICPU fallback | ParallelFor/参数错误上抛 | L0只在launch前按dtype选一次 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定仓无对应 `ClipByValueV2` AICore注册/二进制闭包；不发布 | 同A2；不发布 | 公开支持且源码闭合 |
| Host/Launcher | API有平台dtype表，L0却只按dtype选AICore；非950 AICore终点缺失 | 同左 | 五dtype AIV、四dtype AICPU |
| Kernel/资源 | 固定仓只有 AICPU body可见，无法覆盖全部公开dtype | 同左 | ASS AIV + AICPU scalar；无Cube/AIC/MIX_CV |

## 限制与失败边界

- 四个 ACLNN对象任一为空即失败；头文件文字中的 min/max默认值没有由该原型实现，不能传空指针表示默认。
- `self/out` 必须同 dtype、同 shape；rank上限8。`FRACTAL_NZ` 只告警潜在精度失败，文档公开 layout 为 ND。
- API 先以 scalar 原 dtype 比较 `min/max`，再转换到 `self` dtype；转换窄化后是否仍保持严格数值次序由框架 scalar转换语义决定，固定仓不可见。
- R-001 全程以 T dtype执行 `Maxs/Mins`，没有升精度；Vector primitive 对 NaN 的精确比较/传播规则固定仓不可见。R-002 DOUBLE 显式传播 NaN。
- R-001 申报16MiB workspace但 body不读；executor总 workspace 的框架节点 offset、Contiguous/ViewCopy实现和系统用途固定仓不可见。
- R-001 unknown key直接返回且 L0不会在 Tiling/launch失败后改走 AICPU；不得把 AICPU描述成运行时 fallback。
- A2/A3公开支持与固定仓 AICore接线存在实现缺口，因此未生成 `hardtanh.md`；Atlas 200I/500 A2 在公开表中明确不支持。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/clip_by_value_v2/docs/aclnnHardtanh&aclnnInplaceHardtanh.md` | 产品表、非原地原型、参数表 | 950公开性、语义、dtype/shape/错误 |
| 入口声明/约束 | `conversion/clip_by_value_v2/op_api/aclnn_hardtanh.h` | `aclnnHardtanh*` | 两段式 ABI 与计算图 |
| 入口实现 | `conversion/clip_by_value_v2/op_api/aclnn_hardtanh.cpp` | `CheckParams/GetWorkspaceSizeCommon/aclnnHardtanh` | 校验、空域、Contiguous、scalar转换、ViewCopy、executor |
| Host/Launcher | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `IsAiCoreSupport/ClipByValueV2AiCore/ClipByValueV2AiCpu` | dtype设备分发与launcher |
| Host/平台注册 | `conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp` | `ClipByValueV2` OpDef | 950 AICore dtype与注册 |
| Host/Tiling | `conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_tiling.cpp` | `TilingFor/PrepareForClipByValueV2` | registry入口、AIV core/UB producer |
| Host/Tiling/POD | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.h` | `ClipByValueSigDimTilingData` | POD字段顺序与key注册 |
| Host/Tiling | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `ClipByValueTilingSingleDim` | ASS selector、分核/UB/workspace |
| Kernel entry/body | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | AIV-only、key router |
| Kernel entry/body | `conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h` | `ClipByValueV2Ass` | ASS AIV数据流、Queue、tail |
| Kernel entry/body | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `ClipByValueV2CpuKernel/DoCompute` | AICPU selector、计算、并行、精度 |
| 构建接线 | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu_def.cpp` | `ApplyMathAicpuDefaultCfg/OP_ADD` | AICPU dtype与开放注册 |
| 构建接线 | `conversion/clip_by_value_v2/op_host/config/ascend950/clip_by_value_v2_binary.json` | `ClipByValueV2` op_list | 950五种 AICore binary dtype |
| 构建接线 | `conversion/clip_by_value_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/add_all_modules_sources` | 仅 950、arch35、依赖公共 tiling |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `ClipByValueV2` | ascend950 AscendC binary与auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | ASS可复用9×int64 POD与key `2001100` | 字段顺序、72B布局、元素/核/tile单位不能变；AICPU无POD | `ClipByValueSigDimTilingData` |
| 调用顺序 | Contiguous→scalar转换→ClipByValueV2→ViewCopy | 空域短路、错误立即阻断、非原地out提交顺序 | `GetWorkspaceSizeCommon` |
| workspace/内存 | AIV两Queue双缓冲、16MiB节点workspace | depth必须计入；output slot只能在Mins后提交；body不应假定workspace内容 | `CalcInitialUbParams/ClipByValueV2Ass::Init` |
| 同步 | Queue EnQue/DeQue/Free与executor同stream | 不得提前覆盖 Queue slot；AICPU ParallelFor必须完成后返回 | `Process/DispatchParallel` |
| 精度与尾块 | AIV原dtype Maxs→Mins；AICPU DOUBLE显式NaN；独占tail | 不能擅自交换比较顺序或假设 Vector NaN语义；只写有效len/end | `Compute/ClampScalar` |
| 失败边界 | L0按dtype一次选择，Tiling/launch/ParallelFor错误上抛 | unknown key不是fallback；A2/A3核心缺失不能推测补齐 | `ClipByValueV2/clip_by_value_v2/DispatchParallel` |