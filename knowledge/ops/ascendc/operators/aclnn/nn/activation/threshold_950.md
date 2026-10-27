---
type: Operator
title: "Threshold 950 ACLNN 算子知识"
description: "说明 aclnnThreshold 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold/docs/aclnnThreshold&aclnnInplaceThreshold.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold/op_host/op_api/aclnn_threshold.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold/op_host/op_api/threshold.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_host/threshold_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_kernel/arch35/threshold_tiling_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_kernel/arch35/threshold_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_kernel/arch35/threshold_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_host/config/ascend950/threshold_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
created_at: "2026-08-25T19:05:51Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Threshold 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnThresholdGetWorkspaceSize` + `aclnnThreshold`，公开两段式 ACLNN；核心 L0 为 `l0op::Threshold`，实际算子类型为 `ThresholdV2`。本卡不覆盖语义独立的 `aclnnInplaceThreshold`。 |
| 计算语义 | 对每个元素执行 `out[i] = self[i] > threshold ? self[i] : value`；等于阈值时选择 `value`。 |
| 输入输出 | `self/out` 为 ND、0～8 维并允许非连续；公开 dtype 为 BF16/FP16/FP32/INT8/UINT8/INT16/INT32/INT64；`threshold/value` 是标量。AIV 核心先在 `self` dtype 内计算，再由 ACLNN executor 把结果 Cast/ViewCopy 到用户 `out`。 |
| 执行类型 | 核心节点 `ThresholdV2::threshold_v2` 为单 Kernel `AIV_VECTOR`；前后的 `Contiguous`、`Cast`、`ViewCopy` 是 ACLNN 框架节点，不是 ThresholdV2 自有 Device body。没有可达 Cube/AIC/MIX_CV 节点。 |
| Kernel 路线 | R-001 原生 dtype DAG（FP16/FP32/INT8/UINT8/INT16/INT32/INT64）；R-002 BF16 升精度 DAG。两条 Route 都固定 `valueMode=TPL_HAS_VALUE`，并保留框架产生的 `schMode` 模板参数。 |
| 核心资源 | `self/threshold/value/y` GM，`TPipe`，ATVOSS `ElementwiseSch` 管理的 UB tensor pool，`MemOptCfg<LEVEL_2>` DAG 复用；ThresholdV2 自有 workspace 为 0。ACLNN executor 的框架节点 workspace 由 `GetWorkspaceSize()` 汇总。 |
| 来源闭包 | `source_closed`：公开 ACLNN 声明/实现 → `l0op::Threshold` → 950 `ThresholdV2` Host/Tiling → 模板 key → AIV entry → compare/select DAG → 950 构建配置闭合。ATVOSS 通用 tiling/scheduler 与 AscendC primitive 内部按框架边界标识。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=self[i]`，则

```text
mask = (x <= threshold)
out[i] = mask ? value : x
```

这与公开公式 `x > threshold ? x : value` 等价，因此 `x == threshold`、`x < threshold` 都走 `value` 分支。Kernel DAG 用 `CMPMODE_LE=3` 产生 `uint8_t` 比较 mask，再用 `VSEL_TENSOR_SCALAR_MODE=1` 在标量 `value` 和输入 tensor 之间选择。R-002 在 BF16 输入上先把 tensor 和两个标量提升为 FP32，再比较、选择，最后以 `CAST_RINT=1` 回写 BF16。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，0～8 维；可非连续 | BF16、FP16、FP32、INT8、UINT8、INT16、INT32、INT64 | 输入 `x`。第一段接口先检查非空、dtype、最大维数，并要求 shape 与 `out` 相等；空 Tensor 直接返回 `workspaceSize=0`，不加入 ThresholdV2 launch。 |
| `threshold` | ACLNN scalar；L0 转成 `self` dtype 的一元素 tensor | 同公开 dtype 集，需满足与 `self` 的类型推导规则 | 严格比较阈值；L0 的 `ConvertToTensor(threshold, self->GetDataType())` 冻结 Device 标量 dtype。 |
| `value` | ACLNN scalar；L0 转成 `self` dtype 的一元素 tensor | 同公开 dtype 集，需满足与 `self` 的类型推导规则 | `self <= threshold` 时的替代值。公开 ACLNN 参数为必选，所以从本入口可达的 Kernel 只使用 `TPL_HAS_VALUE`。 |
| `out` | 与 `self` shape 相同的 ND；可非连续 | 公开 dtype 集 | ThresholdV2 内部输出与 `self` dtype 相同；随后 executor 调用 `Cast(thresholdOut, out dtype)` 和 `ViewCopy(..., out)`。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | `uniqueExecutor->GetWorkspaceSize()` 汇总整个 executor 图；空输入固定写 0。ThresholdV2 自身在 Tiling 中把唯一 workspace slice 写为 0。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有连续化、核心计算、输出 Cast/ViewCopy 的执行流程；第二段在调用者给定 stream 上执行。 |

### 入口与版本

公开入口是 `aclnnThresholdGetWorkspaceSize`/`aclnnThreshold`。公开文档明确列出 Ascend 950PR/950DT，并把 ACLNN 调用链接到 `ThresholdV2`。L0 `l0op::Threshold` 注册 `ThresholdV2`，把 `self`、由两个 scalar 转成的 tensor 和内部输出加入 AICore launcher。

固定 Commit 中存在两个同语义 `ThresholdV2` 源码候选：默认活动树 `activation/threshold_v2` 与需要 `ENABLE_EXPERIMENTAL=ON` 才选取的 `experimental/activation/threshold_v2`。本卡选择默认活动树；其顶层 CMake 固定 `SUPPORT_COMPUTE_UNIT="ascend950"`、`SUPPORT_TILING_DIR="arch35"`，Kernel CMake 也只把 `arch35/threshold_v2.cpp` 接到 `ascend950`。这是 950 上活动构建支持的最新实现。

`aclnnInplaceThreshold` 与本入口共用 `ExecThresholdGetWorkspaceSize`，但它把 `selfRef` 同时作为输入和输出，属于原地语义变体，本卡不将其并入 Threshold。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnThresholdGetWorkspaceSize` 的实际顺序如下：

1. `OP_CHECK_COMM_INPUT(workspaceSize, executor)` 检查第一段公共输出参数。
2. `CheckParams` 依次检查 `self/out/threshold/value` 非空、`self` dtype 是否属于当前平台 API 列表、输出转换约束（`NpuArch < DAV_3510` 时执行）、`self` 最大 8 维以及 `self/out` shape 相等。
3. `self->IsEmpty()` 时写 `workspaceSize=0` 并释放 executor，不创建任何计算节点。
4. `l0op::Contiguous(self)` 生成连续输入。950 为 `DAV_3510`，因此仅在旧架构执行的 INT16→INT32 前置 Cast 条件为假；`selfCast=selfContiguous`。
5. `l0op::Threshold(selfCast, threshold, value)` 分配与 `selfCast` storage shape/dtype/format 相同的内部输出，把两个 scalar 转成 `self` dtype 的 tensor，并添加 `ThresholdV2` AICore 节点。
6. `l0op::Cast(thresholdOut, out dtype)` 生成目标 dtype；`l0op::ViewCopy(castOut, out)` 写入可能非连续的用户输出。
7. `uniqueExecutor->GetWorkspaceSize()` 汇总图中 workspace，随后把 executor 所有权交给调用者。

任何一步返回空 tensor 或 launcher 添加失败都会立即返回相应 ACLNN 错误，不继续构造后续节点。第二段 `aclnnThreshold` 只调用 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)`。

### 参数、Workspace 与生命周期

ThresholdV2 自有 Tiling 先通过 `TilingPrepareForThreshold` 从平台读取 AIV core 数和 UB 大小到 `ThresholdCompileInfo::{coreNum,ubSize}`。`TilingForThreshold` 只检查 compile info 非空后构造 `ThresholdTiling`；真正的 tile/core/POD 由 `ElewiseBaseTiling::DoTiling<OpDag>` 这个固定仓外的 ATVOSS 通用框架生成。

公开入口始终传 `value`，所以 `CheckShape` 得到 `hasValue=true`。Tiling 按 `inputDtype` 选择 `ThresholdDag<T>` 或 `ThresholdCastDag<bfloat16_t,float>`，把：

- `workspace[0]` 写成 0 字节；
- key 写成 `GET_TPL_TILING_KEY(tiling->scheMode, TPL_HAS_VALUE, dType)`；
- `blockDim` 写成 `tiling->blockNum`。

ACLNN executor 拥有内部连续 tensor、ThresholdV2 内部输出、Cast 输出和 ViewCopy 任务，直到第二段执行结束。用户拥有传入 workspace 与 `out`。所有节点由同一个 executor 在传入 stream 上提交；固定仓没有算子自有跨 Kernel workspace 边。

### 启动图与失败传播

```text
aclnnThresholdGetWorkspaceSize
  -> N0 l0op::Contiguous(self)                    # 框架边界；可能无实际搬运
  -> N1 l0op::Threshold
       -> ConvertToTensor(threshold, self dtype)
       -> ConvertToTensor(value, self dtype)
       -> N1.1 ThresholdV2::threshold_v2          # 单 AIV_VECTOR Kernel
  -> N2 l0op::Cast(thresholdOut, out dtype)        # 框架边界
  -> N3 l0op::ViewCopy(castOut, out)               # 框架边界
aclnnThreshold(workspace, executor, stream)
  -> CommonOpExecutorRun
```

| 节点/边 | producer | consumer | 数据、offset/extent、layout | ordering 与生命周期 | 失败传播 |
|---|---|---|---|---|---|
| N0 连续化 | 用户 `self` | N1.1 | 连续 storage tensor；具体临时 offset/extent 由框架管理 | executor 持有；同一 stream 图顺序 | 返回空指针则第一段终止 |
| scalar tensor | `ConvertToTensor` | N1.1 | `threshold/value` 各一元素、dtype=`self` dtype | executor 持有；在核心节点前可见 | 转换失败则第一段终止 |
| N1.1 核心输出 | `threshold_v2` | N2 | shape/storage format 与连续 `self` 相同，dtype=`self` dtype；ThresholdV2 workspace=0 | 单 Kernel 内 GM→UB→GM；executor stream 保证后续节点读取 | launcher 添加失败则 L0 返回空，第一段终止 |
| N2 Cast 输出 | 框架 Cast | N3 | dtype=`out` dtype；临时 tensor offset/extent 固定仓不可见 | executor 持有至 ViewCopy 消费 | Cast 返回空则第一段终止 |
| N3 用户输出 | ViewCopy | 用户 | 写 `out` 的 storage/view 布局 | 第二段完成后用户可消费；stream 可见性遵循 ACL runtime | ViewCopy 返回空则第一段终止；第二段返回 runtime 状态 |

## Route 与参数设计

### 分核与资源计算

`ThresholdTiling::RunTiling` 不自行重写 element 数、tile、former/tail core 公式，而是把选定 DAG 交给 `ElewiseBaseTiling::DoTiling`。固定 `ops-nn` Commit 能确定的 producer/consumer 边界是：

```text
x/y storage shape + 当前平台信息 + OpDag 的 dtype/Buffer 元信息
  -> ElewiseBaseTiling::DoTiling<OpDag>(EleBaseTilingDataV2)
  -> tiling.scheMode + tiling.blockNum + 其余通用 elewise 字段
  -> key(scheMode, HAS_VALUE, dType)
  -> blockDim = blockNum
  -> ElementwiseSch<schMode, OpDag>(tilingData, TPipe)
```

`EleBaseTilingDataV2` 的完整字段定义、每字段位宽/offset、`scheMode` 选择谓词、tile/loop/former/tail core 公式及 UB 分区都来自仓外 `atvoss` 头文件；固定 Commit 中不可见，不在本卡推测。能确认的上界是 `TilingPrepareForThreshold` 读取 950 AIV core 数与 UB 大小，ThresholdV2 自有 workspace 为 0，且 Host 最终必须得到非空 tiling、成功的 `DoTiling`、有效 `blockNum` 才能启动。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-002 BF16 升精度 | `GET_TPL_TILING_KEY(scheMode, TPL_HAS_VALUE, TPL_BF16)`；`ThresholdCastDag<bfloat16_t,float>` | `inputDtype == DT_BF16 && hasValue`；在 dtype `if/else if` 链中先于 FP16/FP32 判断 | 条件假时按后续 dtype 分支；`DoTiling` 失败则 Tiling 失败 |
| 2 | R-001 原生 dtype | `GET_TPL_TILING_KEY(scheMode, TPL_HAS_VALUE, dType)`；`ThresholdDag<T>` | `inputDtype` 分别映射 FP16/FP32/INT8/UINT8/INT16/INT32/INT64，且 `hasValue=true` | dtype 不在映射表时 `GRAPH_FAILED`；`DoTiling` 失败直接返回失败 |

```text
ValidateParams failed -> GRAPH_FAILED
hasValue = (optional input value exists)  # ACLNN 固定 true
dType = dtypeToTplMap[inputDtype]
if dType == TPL_BF16:
    DoTiling<ThresholdCastDag<bfloat16_t,float>::OpDag>
else if dType in {FP16,FP32,INT8,UINT8,INT16,INT32,INT64}:
    DoTiling<ThresholdDag<T>::OpDag>
else:
    GRAPH_FAILED
key = (tiling.scheMode, TPL_HAS_VALUE, dType)
blockDim = tiling.blockNum
```

模板声明允许 `schMode∈{0,1}`。固定 `ops-nn` Commit 没有 ATVOSS 的 `scheMode` producer 实现，因此本卡保留该模板维度而不伪造 shape/UB 谓词。`valueMode=TPL_NO_VALUE` 的 DAG 和 entry 分支对图模式可能可达，但公开 `aclnnThreshold` 必传 `value`，故不属于本卡 Route。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 原生 dtype | `(schMode,HAS_VALUE,{FP16,FP32,INT8,UINT8,INT16,INT32,INT64})` | 950 活动构建；x/y 同 dtype、同 shape；公开入口 value 存在；`DoTiling<ThresholdDag<T>>` 成功 | 单 AIV 节点 `threshold_v2` | `ElementwiseSch<schMode, ThresholdDag<T>::OpDag>::Init/Process`；DAG 为 CopyIn→Compare(LE)→Select(value,x)→CopyOut | `TPipe` + scheduler tensor pool；DAG `MemLevel::LEVEL_2`；无自有 workspace；`AUTO_SYNC=false` | 通用 scheduler 根据 POD 处理 core/tile/tail；其公式固定仓不可见。失败无后备 Kernel |
| R-002 BF16 升精度 | `(schMode,HAS_VALUE,BF16)` | 950 活动构建；x/y BF16、同 shape；value 存在；`DoTiling<ThresholdCastDag<bf16,float>>` 成功 | 单 AIV 节点 `threshold_v2` | `ElementwiseSch<schMode, ThresholdCastDag<bfloat16_t,float>::OpDag>::Init/Process`；CopyIn→Cast FP32→Compare→Select→RINT Cast BF16→CopyOut | 同上，但 DAG 同时含 BF16 与 FP32 中间值，`MaxDtypeBytes`/Buffer 元信息由 DAG scheduler 计算 | 通用 scheduler tail；失败无后备 Kernel |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空 Tensor | API `self->IsEmpty()` | 避免创建无有效元素的 executor 节点；该路径 workspace=0 | 全部 | 直接成功返回，不启动 Kernel | `activation/threshold/op_host/op_api/aclnn_threshold.cpp::ExecThresholdGetWorkspaceSize` |
| 950 INT16 不升为 INT32 | `GetCurNpuArch() < DAV_3510 && dtype==INT16` 在 950 为假 | 950 ThresholdV2 活动 binary 原生实例化 INT16 | R-001/INT16 | 旧架构才进入前置 Cast；950 保持 INT16 | 同上 |
| x/y dtype 一致 | ThresholdV2 Tiling 要求 `outputDtype == inputDtype` | entry 的模板 `T` 同时绑定 x/y；核心节点不做输出 dtype 转换 | 两条 Route | `GRAPH_FAILED`；ACLNN 的不同用户 out dtype 由核心节点后的框架 Cast 处理 | `ThresholdTiling::CalcOutputDtype` |
| x/y shape 一致 | `EnsureNotScalar(x storage shape) == EnsureNotScalar(y storage shape)` | elementwise 一一对应写回 | 两条 Route | `GRAPH_FAILED` | `ThresholdTiling::CheckShape` |
| dtype 映射 | 仅 8 个 `DT_*` 值进入 `dtypeToTplMap` | entry 只为 8 个模板 dtype 建立 `T` | 两条 Route | `GRAPH_FAILED` | `ThresholdTiling::CalcInputDtype/RunTiling`、`threshold_v2` |
| value 必选于 ACLNN | API 先检查 `value != nullptr`，L0 总是 `ConvertToTensor` 并传第三输入 | 决定 `valueMode=TPL_HAS_VALUE` 与三输入 `sch.Init` | 两条 Route | null 在 API 阶段返回 `ACLNN_ERR_PARAM_NULLPTR` | `CheckNotNull`、`l0op::Threshold` |
| 通用 elewise UB/核数 guard | `ElewiseBaseTiling::DoTiling<OpDag>` 必须成功并写 `blockNum` | tile 必须容纳 DAG 物理 buffer，且 blockDim 必须有效 | 两条 Route | Tiling 返回失败；没有 ThresholdV2 自有 fallback | `ThresholdTiling::RunTiling`；细节为仓外 ATVOSS 边界 |
| 自动同步关闭 | 950 Kernel CMake `AUTO_SYNC false`，全局 binary config 同样写 false | 同步依赖 DAG scheduler/显式机制，不能依赖编译器自动补全 | 两条 Route | 不存在另一个 auto-sync binary | `activation/threshold_v2/op_kernel/CMakeLists.txt::add_kernel_sources`、`scripts/kernel/binary_config/ascendc_config.json` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ThresholdCompileInfo` | `coreNum` | `uint64_t`/64 bit | 声明顺序首字段；精确 `sizeof/alignof/offsetof` 未在固定仓静态断言 | AIV core 个数 | `TilingPrepareForThreshold`: `GetCoreNumAiv()` | 当前 `ThresholdTiling::RunTiling` 不直接读取；仅校验 compile info 非空 | 平台查询失败或 compile info 空则失败 |
| `ThresholdCompileInfo` | `ubSize` | `uint64_t`/64 bit | 声明顺序第二字段；精确 `sizeof/alignof/offsetof` 未在固定仓静态断言 | 字节 | `GetCoreMemSize(CoreMemType::UB, ubSize)` | 当前 wrapper 不直接读取；ATVOSS 从 tiling context 取平台资源的细节不在固定仓 | 平台查询失败则失败 |
| `EleBaseTilingDataV2` | `scheMode` | 固定仓不可见 | 结构由仓外 `atvoss` 声明，`sizeof/alignof/offsetof` 不可从本 Commit 证明 | 枚举 0/1 | `ElewiseBaseTiling::DoTiling<OpDag>` | Host 用于 `GET_TPL_TILING_KEY`；编译期 key 选择 `ElementwiseSch<schMode,...>` | 模板声明只接受 0 或 1；精确选择条件缺失 |
| `EleBaseTilingDataV2` | `blockNum` | 固定仓不可见 | 同上 | block/AIV 核数 | `ElewiseBaseTiling::DoTiling<OpDag>` | Host `SetBlockDim(tiling->blockNum)`；通用 scheduler 的读取位置在仓外 | 必须产生有效 blockDim；失败不 launch |
| `EleBaseTilingDataV2` | 其余通用 elewise 字段 | 字段集合与位宽固定仓不可见 | 不可见 | core/tile/loop/tail/UB 参数 | `ElewiseBaseTiling::DoTiling<OpDag>` | `ElementwiseSch::Init/Process` 接收整个 `tilingData` | 不将不可见字段、offset 或默认值推测补齐 |
| 模板 key | `dType` | `uint64_t` 模板维度 | 非 POD；由 `ASCENDC_TPL_DTYPE_DECL` 生成 | dtype selector | `dtypeToTplMap`：FP16=1、BF16=2、FP32=3、INT8=4、UINT8=5、INT16=6、INT32=7、INT64=8 | entry 的 `std::conditional_t` 还原为 `T` | 默认值 9 不可达，命中则 Tiling 失败 |
| 模板 key | `valueMode` | `uint8_t` 模板维度 | 非 POD | 0/1 | `hasValue ? TPL_HAS_VALUE : TPL_NO_VALUE`；ACLNN 固定 1 | entry 的 `if constexpr` 选择三输入/二输入 DAG | ACLNN 不可达 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 原生 dtype | `(scheMode,HAS_VALUE,dType!=BF16)` | `threshold_v2_tiling_arch35.cpp::ThresholdTiling::RunTiling` | `EleBaseTilingDataV2` + 三维模板 key | `threshold_v2.cpp::threshold_v2<schMode,valueMode,dType>` 的非 BF16 分支 | `threshold_dag.h::ThresholdDag<T>`；`ElementwiseSch`/Vec primitive 为框架边界 | `threshold_v2/CMakeLists.txt`、`op_kernel/CMakeLists.txt`、950 binary JSON |
| R-002 BF16 升精度 | `(scheMode,HAS_VALUE,TPL_BF16)` | 同上 BF16 第一分支 | 同上 | 同一 entry 的 BF16 分支 | `threshold_dag.h::ThresholdCastDag<bfloat16_t,float>` | 同上 |

## Kernel 实现

### 公共执行流程

两条 Route 共用同一个 AIV entry：解析 `EleBaseTilingDataV2`，声明 `KERNEL_TYPE_AIV_ONLY`，由 `dType` 还原 C++ 类型 `T`，在栈上创建 `TPipe`，然后实例化 `ElementwiseSch<schMode,OpDag>`。公开 ACLNN 始终走 `valueMode==TPL_HAS_VALUE`，所以 `sch.Init(x, threshold, value, y)` 后调用 `sch.Process()`。

共同算子 DAG 是：只搬入 `x` tensor；`threshold/value` 以 `ScalarAttr<1>` 表示；比较结果为 `uint8_t` mask；`Select` 的 mode=1 在 scalar 与 tensor 间选择；结果直接作为 `CopyOut` 输入。DAG 选择 `MemOptCfg<MemLevel::LEVEL_2>`，表示交由通用 DAG scheduler 做二级内存优化。固定仓没有 `ElementwiseSch`、`Vec::*` 和 `EleBaseTilingDataV2` 的定义，因此它们的具体 queue/event、tile offset、repeat/mask 与 UB 物理分区属于框架边界；本卡不把逻辑 DAG 节点伪写成可证明的 `InitBuffer` 布局。

### Route R-001：原生 dtype compare/select

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/threshold/op_host/op_api/aclnn_threshold.cpp` | `ExecThresholdGetWorkspaceSize` | 非空，dtype/shape 检查成功，`self` 非空 | 调用 `l0op::Threshold`；失败立即返回 |
| L0 | `activation/threshold/op_host/op_api/threshold.cpp` | `l0op::Threshold` | scalar 转成 `self` dtype tensor；`ADD_TO_LAUNCHER_LIST_AICORE(ThresholdV2,...)` | 创建 ThresholdV2 核心节点 |
| Tiling | `activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.cpp` | `ThresholdTiling::RunTiling` | `dType∈{TPL_FP16,TPL_FP32,TPL_INT8,TPL_UINT8,TPL_INT16,TPL_INT32,TPL_INT64}` 且 `hasValue=true` | `DoTiling<ThresholdDag<T>::OpDag>`；写 key/blockDim |
| Router | `activation/threshold_v2/op_kernel/arch35/threshold_v2.cpp` | `threshold_v2<schMode,valueMode,dType>` | `dType != TPL_BF16 && valueMode == TPL_HAS_VALUE` | `ElementwiseSch<schMode,ThresholdDag<T>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
ExecThresholdGetWorkspaceSize
  -> l0op::Threshold(self, thresholdScalar, valueScalar, executor)
    -> ADD_TO_LAUNCHER_LIST_AICORE(ThresholdV2, x, thresholdTensor, valueTensor, y)
      -> ThresholdTiling::RunTiling
        -> ElewiseBaseTiling::DoTiling<ThresholdDag<T>::OpDag>  # ATVOSS 边界
      -> threshold_v2<schMode,TPL_HAS_VALUE,dType>(x,threshold,value,y,workspace,tiling)
        -> ElementwiseSch<schMode,ThresholdDag<T>::OpDag>::Init(...)
        -> ElementwiseSch<schMode,ThresholdDag<T>::OpDag>::Process()
          -> CopyIn<T>(x)
          -> Compare<uint8_t,T,CMPMODE_LE>(x, thresholdScalar)
          -> Select<uint8_t,T,VSEL_TENSOR_SCALAR_MODE>(mask, valueScalar, x)
          -> CopyOut<T>(y)
```

##### 源码等价伪码

```text
# 伪码；core/tile/offset/tail 循环由仓外 ElementwiseSch 根据 EleBaseTilingDataV2 驱动
T thresholdScalar = thresholdGM[0]
T valueScalar = valueGM[0]
for each valid tile assigned to this AIV core:
    xLocal[0:valid] = GM_x[tileOffset : tileOffset + valid]
    maskLocal[0:valid] = (xLocal[0:valid] <= thresholdScalar)
    yLocal[0:valid] = select(maskLocal, valueScalar, xLocal)
    GM_y[tileOffset : tileOffset + valid] = yLocal[0:valid]
# 只提交 valid 元素；精确 DMA/repeat/mask/tail 参数固定仓不可见
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | scheduler 为每核/每 tile 生成；公式固定仓不可见 | `numel(self)` 个元素 | `T`/ND storage | ACLNN 连续化节点 | `Vec::CopyIn<T>` | executor/用户输入；Kernel 只读 | tail 搬运由 scheduler 保护 |
| GM `threshold` | 基址元素 0 | 1 scalar tensor | `T` | `ConvertToTensor` | `ScalarAttr<1>`/Compare | executor 持有至核心节点完成 | 不按输出域展开成 GM tensor |
| GM `value` | 基址元素 0 | 1 scalar tensor | `T` | `ConvertToTensor` | `ScalarAttr<1>`/Select | 同上 | 标量广播由 DAG primitive 表达 |
| UB `x`/mask/select 逻辑值 | 由 `ElementwiseSch` tensor pool 映射 | 取决于 tile、`OpDag::BufferNum/MaxDtypeBytes` | `T`、`uint8_t`、`T` | CopyIn/Compare/Select | Compare+Select/CopyOut | 单 tile；物理 alias/lifetime 由 `MemLevel::LEVEL_2` 规划 | 固定仓无 `InitBuffer`，不可给出物理 offset/depth |
| GM `y` | scheduler 生成与 x 对应的 tile offset | `numel(self)` 个有效元素 | `T`/ND storage | `Vec::CopyOut<T>` | ACLNN 后置 Cast | executor 内部 tensor | 不写无效 tail |
| workspace | N/A | ThresholdV2 固定 0 字节 | N/A | Host 写 `workspace[0]=0` | Kernel 参数存在但 body 不读取 | 无算子自有 slice | 不与 UB alias |

##### 分层内存布局

N/A：该 AIV entry 的 UB 由仓外 `ElementwiseSch` 根据 `MemOptCfg<MemLevel::LEVEL_2>` 和 DAG 编译期元信息统一分配；固定 `ops-nn` Commit 未包含 `InitBuffer`、BufferId、depth 与字节公式，不能构造真实 `block-beta` 物理布局。可证明的逻辑阶段只有 `x(T) → mask(uint8_t) → selected(T)`。

##### 字节账与复用顺序

```text
UB_total = 固定仓不可见（由 ElementwiseSch + OpDag::BufferNum/MaxDtypeBytes 决定）
UB_total <= A5 UB 的精确选择不等式 = 固定仓不可见
dynamic_ub = N/A（不是 SIMT launch）
workspace_total(ThresholdV2) = 0
workspace_total(ACLNN executor) = uniqueExecutor->GetWorkspaceSize()  # 框架节点汇总，数值依输入视图/dtype
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG tensor pool 物理子区 | CopyIn/Compare/Select | 后继 DAG 节点/CopyOut | 由 `MemOptCfg<LEVEL_2>` 规划 | 通用 scheduler 插入的 release/sync；固定仓不可见 | 必须保持 DAG 依赖和 scheduler 生成的 BufferId，不可自行假设原地 alias |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `Vec::CopyIn<T>` | GM x | Compare | x tile | T 原生精度；有效 tile 长度由 scheduler 提供 |
| 比较 | `Compare<uint8_t,T,3>` | x tile + threshold scalar | Select | uint8 mask | 严格语义通过 `<=` mask 的反向 Select 实现；NaN 行为取决于 T 的 Compare primitive |
| 选择 | `Select<uint8_t,T,1>` | mask/value/x | CopyOut | T 结果 | 无额外 cast；等于阈值选 value |
| UB→GM | `Vec::CopyOut<T>` | select 结果 | GM y | 输出 tile | 只允许写有效 tail；精确 DMA mask 固定仓不可见 |
| 同步 | `AUTO_SYNC=false` + DAG scheduler | 各 DAG 节点 | 后继节点 | tensor pool 生命周期 | entry 无显式 event；实际同步由仓外 scheduler 生成，不能删除或重排 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.cpp` | `ThresholdTiling::RunTiling` | dtype 映射、原生 DAG 实例、workspace=0、key 与 blockDim |
| Kernel entry/router | `activation/threshold_v2/op_kernel/arch35/threshold_v2.cpp` | `threshold_v2<schMode,valueMode,dType>` | AIV_ONLY、T 选择、HAS_VALUE 三输入 Init、Process |
| Device body/helper | `activation/threshold_v2/op_kernel/arch35/threshold_dag.h` | `ThresholdDag<T>` | CopyIn→Compare LE→tensor/scalar Select→CopyOut，LEVEL_2 内存配置 |

### Route R-002：BF16 升精度 compare/select

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/threshold/docs/aclnnThreshold&aclnnInplaceThreshold.md` | 产品/dtype 表 | 950 公开支持 BF16 | 进入 API 实现 |
| L0 | `activation/threshold/op_host/op_api/threshold.cpp` | `l0op::Threshold` | scalar 均转换成 `self` 的 BF16 | 添加 ThresholdV2 BF16 节点 |
| Tiling | `activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.cpp` | `ThresholdTiling::RunTiling` | `dType == TPL_BF16 && hasValue` | `DoTiling<ThresholdCastDag<bfloat16_t,float>::OpDag>` |
| Router | `activation/threshold_v2/op_kernel/arch35/threshold_v2.cpp` | `threshold_v2` | `if constexpr (dType == TPL_BF16)` 且 `valueMode==HAS_VALUE` | `ElementwiseSch<schMode,ThresholdCastDag<T,float>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
ThresholdTiling::RunTiling
  -> DoTiling<ThresholdCastDag<bfloat16_t,float>::OpDag>
threshold_v2<schMode,TPL_HAS_VALUE,TPL_BF16>
  -> ElementwiseSch<schMode,ThresholdCastDag<bfloat16_t,float>::OpDag>::Init/Process
    -> CopyIn<bfloat16_t>(x)
    -> Cast<float,bfloat16_t,CAST_NONE>(x)
    -> ToFloatScalar<float,bfloat16_t>(threshold)
    -> ToFloatScalar<float,bfloat16_t>(value)
    -> Compare<uint8_t,float,CMPMODE_LE>(xFp32, thresholdFp32)
    -> Select<uint8_t,float,VSEL_TENSOR_SCALAR_MODE>(mask, valueFp32, xFp32)
    -> Cast<bfloat16_t,float,CAST_RINT>(selectedFp32)
    -> CopyOut<bfloat16_t>(y)
```

##### 源码等价伪码

```text
# 伪码；tile/core/tail 仍由仓外 ElementwiseSch 驱动
float thresholdFp32 = ToFloat(thresholdBf16GM[0])
float valueFp32 = ToFloat(valueBf16GM[0])
for each valid tile assigned to this AIV core:
    xBf16 = CopyIn(GM_x[tileOffset : tileOffset + valid])
    xFp32 = Cast<float>(xBf16, CAST_NONE)
    mask = (xFp32 <= thresholdFp32)
    selectedFp32 = Select(mask, valueFp32, xFp32)
    yBf16 = Cast<bfloat16_t>(selectedFp32, CAST_RINT)
    CopyOut(GM_y[tileOffset : tileOffset + valid], yBf16)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | scheduler 的对应 tile offset，公式固定仓不可见 | `numel(self)` | BF16/ND storage | ACLNN/Kernel | CopyIn/后置 Cast | executor | tail 由 scheduler 保护 |
| GM threshold/value | 元素 0 | 各 1 BF16 scalar tensor | BF16 | `ConvertToTensor` | `ToFloatScalar` | executor | 只取一次逻辑 scalar，不物化为输出同形状 GM |
| UB x BF16 | scheduler tensor pool | tile 容量固定仓不可见 | BF16 | CopyIn | Cast FP32 | 单 tile | physical BufferId/复用不可见 |
| UB x/select FP32 | scheduler tensor pool | tile 容量×4 字节的逻辑需求；实际容量/alias 不可见 | FP32 | Cast/Select | Compare+Select/降精度 Cast | 单 tile | DAG LEVEL_2 规划可能复用，但不可推测具体子区 |
| UB mask | scheduler tensor pool | tile 容量×1 字节的逻辑值；实际对齐不可见 | uint8 | Compare | Select | Compare→Select | 对齐/physical alias 不可见 |
| workspace | N/A | 0 | N/A | Host | 未读取 | 无 | N/A |

##### 分层内存布局

N/A：与 R-001 相同，固定 Commit 没有 ATVOSS scheduler 的实际 `InitBuffer`。R-002 可证明的逻辑层级为 `BF16 x → FP32 x → uint8 mask/FP32 selected → BF16 y`，不能据此声称这些逻辑值各自占用独立物理 UB。

##### 字节账与复用顺序

```text
UB_total = 固定仓不可见
逻辑活跃 dtype 包含 BF16(2B)、FP32(4B)、mask(1B)，但不能把逻辑容量直接当物理分配
dynamic_ub = N/A
workspace_total(ThresholdV2) = 0
workspace_total(ACLNN executor) = uniqueExecutor->GetWorkspaceSize()
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG tensor pool | CopyIn/Cast/Compare/Select/降精度 Cast | 各后继节点/CopyOut | scheduler 决定 | `MemOptCfg<LEVEL_2>` + scheduler 依赖图 | BF16/FP32 不同元素宽度必须使用 scheduler 生成的 offset/BufferId，不能按同元素数盲目原地覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `Cast<float,bfloat16_t,CAST_NONE>` + `ToFloat` | BF16 x/两个 scalar | FP32 Compare | FP32 tensor/scalars | 比较和选择都在 FP32；输入信息仍受 BF16 表示限制 |
| 比较/选择 | `Compare<uint8_t,float,3>`、`Select<uint8_t,float,1>` | FP32 值 | 降精度 Cast | mask/selected | `<=` 分支选 value；NaN 规则由 Compare primitive 决定 |
| 降精度 | `Cast<bfloat16_t,float,CAST_RINT>` | selected FP32 | CopyOut | BF16 tile | 显式 round-to-integral/rounding mode 常量为 1；精确硬件舍入定义属于 primitive 边界 |
| tail | scheduler valid length | tile producer | 各 Vector 节点/CopyOut | 尾 tile | 精确 repeat/mask/padding 不在固定仓；不得让 padding lane 写 GM |
| 同步 | `AUTO_SYNC=false` + scheduler | 每个 DAG 节点 | 后继节点 | tensor pool | 与 R-001 相同，依赖 scheduler 生成协议 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.cpp` | `ThresholdTiling::RunTiling` BF16 分支 | 选择 `ThresholdCastDag<bfloat16_t,float>`，写统一 key/blockDim/workspace |
| Kernel entry/router | `activation/threshold_v2/op_kernel/arch35/threshold_v2.cpp` | `threshold_v2<schMode,TPL_HAS_VALUE,TPL_BF16>` | BF16 专门实例，HAS_VALUE 时传四个 GM 业务参数给 scheduler |
| Device body/helper | `activation/threshold_v2/op_kernel/arch35/threshold_dag.h` | `ToFloatScalar`、`ThresholdCastDag<T,U>` | tensor/scalar 升 FP32、Compare/Select、RINT 降 BF16 |

## 算子特定优化

### Trick T-001：BF16 比较与选择统一升到 FP32

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：950、`inputDtype==DT_BF16`、公开 ACLNN value 存在。
- Golden 来源：[`activation/threshold_v2/op_kernel/arch35/threshold_dag.h:73-92`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_kernel/arch35/threshold_dag.h#L73-L92)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using OpCopyInXCast = Bind<Vec::Cast<U, T, CAST_NONE>, OpCopyInX>;
using ThresholdScalarHolderCast = Bind<ToFloatScalar<U, T>, ThresholdScalarHolder>;
using ValueScalarHolderCast = Bind<ToFloatScalar<U, T>, ValueScalarHolder>;

using OpCompare = Bind<Vec::Compare<uint8_t, U, CMPMODE_LE>, OpCopyInXCast, ThresholdScalarHolderCast>;
using OpSelect =
    Bind<Vec::Select<uint8_t, U, VSEL_TENSOR_SCALAR_MODE>, OpCompare, ValueScalarHolderCast, OpCopyInXCast>;
using OpSelectResult = Bind<Vec::Cast<T, U, CAST_RINT>, OpSelect>;
```

#### 机制与设计原因

`T=bfloat16_t`、`U=float`。输入 tensor 通过 `Vec::Cast<U,T,CAST_NONE>` 升到 FP32；两个标量通过 `ToFloat` 使用同一 FP32 比较域。`Compare` 的目标是 `uint8_t` mask，mode 固定为 `CMPMODE_LE=3`；`Select` mode 1 接收 mask、FP32 scalar value 与 FP32 tensor x；只在最终结果处用 `CAST_RINT=1` 降回 BF16。该顺序避免在 BF16 中分别比较和选择后反复转换，也保证比较两侧处于同一计算 dtype。源码能证明精度路径，未提供误差或性能百分比。

#### 迁移边界

- 适用：BF16 输入、阈值和值也已按 L0 转成 BF16 scalar tensor的逐元素阈值算子。
- 前提：950 `ThresholdV2` BF16 binary、FP32 Vector Compare/Select、DAG scheduler 能为混合 BF16/FP32 中间值分配 UB。
- 失效：非 BF16 改走 R-001；若输出需要不同 dtype，保持核心 BF16 输出后由 ACLNN Cast，不应擅自改 Kernel ABI。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从依赖关系可确定，不能只把 x 升到 FP32而让 threshold/value 继续以 BF16 primitive 参与 FP32 Compare/Select，也不能删除最终带 `CAST_RINT` 的降精度节点。

### Trick T-002：标量阈值与替代值不按输出 shape 物化

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：公开 ACLNN 的 `threshold/value` 标量经 L0 各转成一个 `self` dtype tensor；DAG 使用 `ScalarAttr<1>`。
- Golden 来源：[`activation/threshold_v2/op_kernel/arch35/threshold_dag.h:40-54`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/threshold_v2/op_kernel/arch35/threshold_dag.h#L40-L54)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using OpCopyInX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
using ThresholdScalarHolder = Placeholder::In1<T, Placeholder::ScalarAttr<1>>;
using ValueScalarHolder = Placeholder::In2<T, Placeholder::ScalarAttr<1>>;

using OpCompare = Bind<Vec::Compare<uint8_t, T, CMPMODE_LE>, OpCopyInX, ThresholdScalarHolder>;
using OpSelectResult =
    Bind<Vec::Select<uint8_t, T, VSEL_TENSOR_SCALAR_MODE>, OpCompare, ValueScalarHolder, OpCopyInX>;
```

#### 机制与设计原因

只有 x 绑定 `Vec::CopyIn<T>`；threshold/value 是标量 holder。Compare 和 Select 接收 scalar 参数，Select mode=1 明确是 tensor/scalar 模式。因此固定源码没有把两个标量广播成与 `self` 同 shape 的 GM tensor，也没有为它们建立逐 tile CopyIn DAG 节点。它减少了核心算子的 GM 搬运和中间物化；标量实际载入、寄存器保存或 cache 行为属于通用 scheduler/primitive 边界。

#### 迁移边界

- 适用：threshold/value 对整个输出域恒定的逐元素选择。
- 前提：L0 保证标量 tensor dtype 与 `self` 一致，Kernel ABI 参数顺序保持 `x,threshold,value,y`，DAG holder 索引保持 In1/In2。
- 失效：若 threshold/value 是逐元素 tensor 或需要广播维度语义，不能复用 `ScalarAttr<1>` Route，必须使用能闭合广播 shape 和搬运的实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能把 `ScalarAttr<1>` 改成普通 tensor holder 后仍声称 workspace、UB 和 GM 搬运不变。

## 模板与实现对比

| 对比维度 | R-001 原生 dtype | R-002 BF16 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | FP16/FP32/各整数，位于 BF16 分支之后 | BF16 第一分支 | `ThresholdTiling::RunTiling` 的 dtype `if/else if` 顺序 |
| 模板与 dtype | `ThresholdDag<T>`，计算 dtype=T | `ThresholdCastDag<bfloat16_t,float>`，计算 dtype=FP32 | BF16 需要显式升/降精度路径 |
| 循环与分核 | `ElementwiseSch<schMode,...>`，通用 POD | 同左 | 两者共用 ATVOSS scheduler；具体公式固定仓不可见 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | `TPipe`+tensor pool；逻辑 T/mask/T；无 SIMT dynamic UB/通信 | `TPipe`+tensor pool；逻辑 BF16/FP32/mask/BF16 | R-002 有混合 dtype 中间值；实际 BufferId/depth 不可见 |
| 内存总占用与复用 | `MemLevel::LEVEL_2`；字节账固定仓不可见；workspace=0 | 同配置但 MaxDtype 为 FP32；字节账不可见；workspace=0 | 不能用逻辑值数量替代实际 scheduler 分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | CopyIn、Compare、Select、CopyOut；纯 AIV Vector | 额外 tensor/scalar Cast 与 RINT Cast；纯 AIV Vector | dtype 决定是否升精度 |
| 同步、精度与 tail | 原生 T 精度；scheduler tail/sync | FP32 比较/选择，RINT 回 BF16；scheduler tail/sync | Kernel CMake `AUTO_SYNC=false`，同步不得脱离 scheduler |
| fallback/失败边界 | dtype/shape/DoTiling 失败即失败，无 Kernel fallback | 同左 | dormant NO_VALUE 分支不是 ACLNN fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档写支持；固定 Commit 无活动 A2 Host/Tiling/Kernel 源码闭包，未生成正式卡 | 公开 ACLNN 文档写支持；固定 Commit 无活动 A3 Host/Tiling/Kernel 源码闭包，未生成正式卡 | 公开支持且默认活动 `activation/threshold_v2` 源码闭合；正式卡覆盖 |
| Host/Launcher | ACLNN 公共实现可见，但核心实现来源缺失 | 同左 | `l0op::Threshold` → arch35 `ThresholdTiling` → `ThresholdV2` |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | 单 AIV Vector Kernel；`ascend950` binary；workspace=0；ATVOSS UB scheduler 边界 |

## 限制与失败边界

- 第一段参数任一为 null 返回 `ACLNN_ERR_PARAM_NULLPTR`；dtype/shape/最大维数不符返回 `ACLNN_ERR_PARAM_INVALID`；内部 tensor/launcher 创建失败返回相应内部错误。
- 公开 API 文档支持空 Tensor，但 950 ThresholdV2 Tiling 的通用 elementwise 路线不接受零元素；ACLNN 在 Tiling 前用 `self->IsEmpty()` 截断，因此复用 Kernel 直调时必须自行保留空域 guard。
- 950 核心 ThresholdV2 要求内部 x/y dtype 和 shape 相同。用户 `out` dtype 转换及非连续布局由后置 Cast/ViewCopy 承担，不属于核心 Kernel ABI。
- `threshold/value` 公开语义是 ACLNN scalar；不要把图模式 ThresholdV2 可接受的可选 value 或普通 tensor shape 能力外推到本 ACLNN Route。
- `GetDtypeSupportList` 通过 `SocVersion` 区间选择 API 列表，固定仓没有 950 `SocVersion` 枚举映射；950 BF16 支持以公开产品表、ThresholdV2 OpDef 与 950 binary config 为准，不推测该框架映射的内部值。
- `EleBaseTilingDataV2`、`ElewiseBaseTiling`、`ElementwiseSch` 和 `Vec::*` 的定义不在固定 `ops-nn` Commit。其 POD 完整 ABI、分核/tile 公式、UB `InitBuffer` 字节账、DMA 参数、同步事件与 tail mask 均为明确框架边界，不能依据名称补齐。
- `AUTO_SYNC=false`；若脱离官方 `ElementwiseSch` 重写 Kernel，必须重新闭合同步与 buffer 生命周期，不能假定编译器自动插入同步。
- A2/A3 只有公开支持与 ACLNN Host 证据，没有固定仓内活动 Kernel/Tiling 闭包，故没有创建 `threshold.md`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/threshold/docs/aclnnThreshold&aclnnInplaceThreshold.md` | `aclnnThresholdGetWorkspaceSize`、产品支持/参数表 | 公开性、A2/A3/950 支持、公式、dtype、shape、两段式入口 |
| 入口实现 | `activation/threshold/op_host/op_api/aclnn_threshold.cpp` | `CheckParams`、`ExecThresholdGetWorkspaceSize`、`aclnnThreshold` | 校验、空域、Contiguous、950 INT16 guard、核心调用、Cast/ViewCopy、workspace/executor |
| L0 launcher | `activation/threshold/op_host/op_api/threshold.cpp` | `l0op::Threshold` | scalar tensor 化、内部输出、`ADD_TO_LAUNCHER_LIST_AICORE(ThresholdV2)` |
| 公开核心说明 | `activation/threshold_v2/README.md` | 产品表、调用说明 | ThresholdV2 语义、公开 950 支持、ACLNN 到 ThresholdV2 的关系 |
| OpDef/平台注册 | `activation/threshold_v2/op_host/threshold_v2_def.cpp` | `ThresholdV2::ThresholdV2`、`OP_ADD` | 8 dtype、ND、动态 shape/rank、ascend950 AICore config |
| Tiling 声明 | `activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.h` | `ThresholdCompileInfo`、`ThresholdTiling` | compile info ABI、Tiling 类字段与方法 |
| Host/Tiling | `activation/threshold_v2/op_host/arch35/threshold_v2_tiling_arch35.cpp` | `ThresholdTiling::*`、`TilingForThreshold`、`TilingPrepareForThreshold` | dtype/shape 校验、DAG 选择、key、blockDim、workspace、平台资源 producer |
| 模板 key | `activation/threshold_v2/op_kernel/arch35/threshold_tiling_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL(ThresholdV2)` | schMode/valueMode/dType 允许值 |
| Device DAG | `activation/threshold_v2/op_kernel/arch35/threshold_dag.h` | `ThresholdDag`、`ThresholdCastDag`、`ToFloatScalar` | Compare/Select 语义、BF16 精度链、MemLevel |
| Kernel entry/body | `activation/threshold_v2/op_kernel/arch35/threshold_v2.cpp` | `threshold_v2` | AIV_ONLY、模板 router、scheduler Init/Process、参数顺序 |
| 950 binary | `activation/threshold_v2/op_host/config/ascend950/threshold_v2_binary.json` | `ThresholdV2` 8 个 binary 项 | 950 dtype binary 接线与三输入一输出 ABI |
| Host 构建 | `activation/threshold_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR`、`add_modules_sources` | 默认活动树只接 ascend950/arch35 |
| Kernel 构建 | `activation/threshold_v2/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch35 kernel、ascend950、impl mode、AUTO_SYNC=false |
| 全局 binary config | `scripts/kernel/binary_config/ascendc_config.json` | `ThresholdV2` entry | ascend950 唯一 compute unit、AUTO_SYNC=false |
| 构建模式开关 | `CMakeLists.txt` | `ENABLE_EXPERIMENTAL` | experimental 默认关闭 |
| 活动树选择 | `cmake/func.cmake` | `add_category_subdirectory` | 默认扫描正式类别树；打开开关时改选 experimental 类别树 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 核心 ABI 为 `x,threshold,value,y,workspace,tiling`；公开 Route value 必选 | 参数顺序、scalar holder In1/In2、key 三维顺序；通用 POD 不可自定义猜测 | `threshold_v2`、`ThresholdDag`、`ASCENDC_TPL_ARGS_DECL` |
| 调用顺序 | Contiguous→ThresholdV2→Cast→ViewCopy | 空域必须在核心 launch 前截断；失败必须阻断后续节点 | `ExecThresholdGetWorkspaceSize` |
| workspace/内存 | ThresholdV2 workspace=0；DAG LEVEL_2；scalar 不按输出 shape 物化 | ACLNN executor workspace 不能硬编码 0；UB 字节/alias 必须由 scheduler 闭合 | `ThresholdTiling::RunTiling`、`ThresholdDag::MemCfg` |
| 同步 | 官方 scheduler 管理 DAG 依赖；binary `AUTO_SYNC=false` | 重写时不得删除 producer→consumer 同步或提前覆盖 tensor pool | `op_kernel/CMakeLists.txt`、`ElementwiseSch` 实例化点 |
| 精度与尾块 | 原生 dtype Route 与 BF16→FP32→BF16 Route 分离；BF16 最终 `CAST_RINT` | 等于阈值选 value；BF16 scalar/tensor 必须在同一 FP32 域比较；tail 只写有效元素 | `ThresholdDag`、`ThresholdCastDag` |
| 失败边界 | 无可达 Kernel fallback；校验、Tiling、launcher 任一失败即终止 | 不把 NO_VALUE dormant 分支或 experimental 候选当 ACLNN fallback | `CheckParams`、`ThresholdTiling::RunTiling`、`l0op::Threshold` |
