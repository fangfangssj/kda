---
type: Operator
title: "Asin 950 ACLNN 算子知识"
description: "说明 aclnnAsin 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/docs/aclnnAsin&aclnnInplaceAsin.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/aclnn_asin.h"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/aclnn_asin.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/asin.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_host/asin_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_host/arch35/asin_tiling_arch35.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/arch35/asin_tilingdata.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/arch35/asin_dag.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/asin_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_host/config/ascend950/asin_binary.json"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel_aicpu/asin_aicpu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel_aicpu/asin_aicpu_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/elewise/elewise_tiling.cpp"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/node.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h"
    role: implementation
  - resource: "https://gitcode.com/cann/asc-devkit/blob/7a7156352f594b5ca79b533f72c5a99f19b87983/impl/basic_api/utils/kernel_utils_mode.h"
    role: implementation
created_at: "2026-08-25T10:08:22Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Asin 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnAsinGetWorkspaceSize` + `aclnnAsin`；本卡不包含 `aclnnInplaceAsin`。 |
| 计算语义 | 对每个逻辑元素计算 `out[i] = asin(self[i])`；核心节点不做 `[-1, 1]` 值域校验。 |
| 输入输出 | `self` 与 `out` 同 shape、ND、至多 8 维且可非连续；950 API 输入支持 FLOAT/FLOAT16/BFLOAT16/DOUBLE/INT8/INT16/INT32/INT64/UINT8/BOOL，输出支持 FLOAT/FLOAT16/BFLOAT16/DOUBLE。 |
| 执行类型 | ACLNN executor 多节点图；核心 `Asin` 为 AIV-only entry 内嵌 SIMT VF（FLOAT/FLOAT16/BFLOAT16）或 AICPU fallback（DOUBLE）。框架 `Contiguous/Cast/ViewCopy` 节点按实际 dtype/连续性出现。无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP32 直接 SIMT；R-002 FP16/BF16 升精度 SIMT；R-003 DOUBLE AICPU。整数/BOOL/UINT8 先由框架 Cast 到 FP32，进入 R-001。 |
| 核心资源 | R-001：GM + 单个 `TBuf<VECCALC>` 的 4 个等长子区 + 1024 SIMT threads；R-002：6 个等长子区 + 1024 SIMT threads；R-003：AICPU GM 输入/输出与 CPU 分片，无 AIV UB。Asin AICore tiling workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN、executor 组图、L0 AICore/AICPU selector、950 Tiling/POD、AIV entry、SIMT body、AICPU body及构建接线已闭合；通用框架节点保留为明确边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设逻辑元素数为 `N = product(self.shape)`。对 `0 <= i < N`，核心结果为 `r[i] = asin(x[i])`。950 上核心计算的实际精度顺序由入口 dtype 决定：

- FLOAT：直接以 `asinf(float)` 计算；
- FLOAT16/BFLOAT16：AIV DAG 先 `Cast<T -> float>`，以 `asinf(float)` 计算，再 `Cast<float -> T>`；
- 整数、UINT8、BOOL：ACLNN 前处理先 Cast 为 FLOAT，再进入 FLOAT 核心；
- DOUBLE：保持 DOUBLE，进入 AICPU 的 `std::asin(double)`；
- 核心结果随后按 `out.dtype` 经 L0 `Cast` 转换，并由 `ViewCopy` 写入可能非连续的 `out`。

源码没有对输入值做 `[-1, 1]` 检查，也没有自行替换 NaN/Inf；这些数值结果由所调用的 `asinf/std::asin` 决定。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；与 `out` 同 shape；公开文档限制不超过 8 维；可非连续 | FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL | 必须非空指针；空 Tensor 在第一段直接返回，workspace 为 0且不组核心任务。非 ND storage format 仅记录 warning，随后仍经 `Contiguous`。 |
| `out` | ND；与 `self` 同 shape；公开文档限制不超过 8 维；可非连续 | FLOAT、FLOAT16、BFLOAT16、DOUBLE | 必须非空指针；dtype 不要求与 `self` 相同，核心结果由末尾 Cast 转为此 dtype。 |
| `workspaceSize` | 标量输出 | `uint64_t` | 由 executor 汇总框架前后处理、临时 Tensor 和实际任务需求；Asin AICore 自身写入的 tiling workspace slice 为 0。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并转移所有权；第二段消费已组好的任务图。 |
| `workspace` | Device 地址 | `void*` | 第二段传入，大小必须采用第一段返回值。 |
| `stream` | runtime stream | `aclrtStream` | `CommonOpExecutorRun` 在该 stream 上按 executor 顺序提交。 |

### 入口与版本

公开入口为未带版本后缀的 `aclnnAsinGetWorkspaceSize/aclnnAsin`，固定仓不存在更高版本的同语义公开入口。本卡只处理新建输出的 `aclnnAsin`；共享内部函数虽也被原地入口调用，但 `aclnnInplaceAsin` 属不同逻辑变体，不纳入路线。

公开产品表列出 A2、A3 与 950 支持。950 的公开性由该表与 API header 共同证明；实现侧由 `Asin` OpDef 仅对 `ascend950` 添加 AICore config、binary config 列出 BF16/F16/F32，以及算子 CMake 的 `SUPPORT_COMPUTE_UNIT "ascend950" "mc62"` 共同闭合。A2/A3 的公开表与当前活动源码接线不闭合，因此不在本文件冒充 950 的复用平台，也不生成无后缀卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecAsinGetWorkspaceSize` 的真实顺序如下：

1. `CHECK_NOT_NULL(self, out)`；创建 `aclOpExecutor`，创建失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. 950/RegBase 采用含 BF16 的输入、输出 dtype 白名单；`CheckParams1In1Out` 校验两侧 dtype 与相同 shape。公开文档另声明最大 8 维，但该函数正文未出现独立维数判断。
3. 若 `self->IsEmpty()`，写 `workspaceSize=0`，转移 executor 并成功返回；不会调用 `Contiguous/Asin/Cast/ViewCopy`。
4. `l0op::Contiguous(self)` 产生连续输入或复用已有连续 Tensor；失败返回内部空指针错误。
5. 取连续输入 dtype。仅当 `npuArch == DAV_3510` 且 dtype 为 BF16/F16/F32 时零拷贝复用，不添加前置 Cast。否则：整数、UINT8、BOOL、BF16 被目标化为 FLOAT；DOUBLE、FLOAT、FLOAT16 保持自身。只有目标 dtype 变化时才调用 L0 Cast。
6. `l0op::Asin(selfCast)` 分配同 shape、同 dtype 的临时输出。L0 selector 对 950 的 BF16/F16/F32 添加 AICore task，对 DOUBLE 添加 AICPU task。
7. `l0op::Cast(asinOutRet, out.dtype)` 总会被调用；Cast L0 在源/目标 dtype 相同的情况下直接返回原 Tensor，否则添加自己的 task。
8. `l0op::ViewCopy(castOut, out)` 负责可能非连续的输出视图。
9. 从 executor 取得总 workspace，转移 executor 所有权；第二段 `aclnnAsin` 只调用 `CommonOpExecutorRun`。

### 参数、Workspace 与生命周期

API 临时 Tensor `selfContiguous -> selfCast -> asinOutRet -> castOut` 均由 `uniqueExecutor` 管理；别名优化会让 `selfCast==selfContiguous`，同 dtype 后 Cast 也可让 `castOut==asinOutRet`。用户 `out` 与 workspace 的生命周期跨第二段执行有效。

Asin AICore 节点的 Tiling 明确写 `workspaceSizes[0]=0`。AICPU OpDef 的通用配置声明 `opInfo.workspaceSize="100"`，其 body 本身不读取 workspace。ACLNN 对外 `workspaceSize` 不是固定的 0 或 100，而是 executor 对实际出现的 Contiguous/Cast/ViewCopy 与临时对象统一求和；固定仓没有在 `aclnn_asin.cpp` 中展开该通用分配算法。

### 启动图与失败传播

非空输入的 executor 图为：

```text
self
  -> [framework Contiguous；连续输入可复用]
  -> [可选 pre-Cast；整数/UINT8/BOOL -> FLOAT，DOUBLE/F16/F32/BF16 不变]
  -> Asin selector
       -> R-001/R-002: AICore task `asin`, blockDim=baseTiling.blockNum,
                       POD=AsinTilingData, workspace slice=0
       -> R-003: AICPU task `Asin`, CPU ParallelFor
  -> [framework post-Cast；仅 core dtype != out.dtype 时产生 task]
  -> [framework ViewCopy；写用户 out]
```

所有节点由同一个 executor 收集并在传入 stream 上按数据依赖提交；中间 Tensor 由 executor 持有至消费完成。任一步返回空指针/非成功状态都会令第一段立即返回，不再追加后续节点；第二段把 `CommonOpExecutorRun` 的状态直接返回。框架边界节点的内部 blockDim、workspace slice 与事件协议由各自实现决定，不由 Asin Host/Tiling 生产。

## Route 与参数设计

### 分核与资源计算

仅 R-001/R-002 使用 950 AICore Tiling。令 `N` 为输出 storage shape 各维乘积，`C_avail=GetCoreNumAiv()`，`U` 为平台 UB 字节数，固定预留 `E=122880+32768=155648` 字节。DAG 编译期给出 `B=BufferNum`、`w_max=MaxDtypeBytes`、`b_min=MinDtypeBits`：

```text
C = min(ceil(N * b_min / 32768), C_avail)
blockFormer = align_up(ceil(N / C), 512)                  # 元素
blockNum = ceil(N / blockFormer)                          # 实际 AIV blockDim
blockTail = N - (blockNum - 1) * blockFormer              # 元素

rawTile = floor((U - E) * 8 / (B * w_max * 8))            # 元素
alignFactor = 256 * 8 / b_min                              # 元素
ubFormer = floor(rawTile / alignFactor) * alignFactor      # 元素
loopFormer = ceil(blockFormer / ubFormer)
loopTail = ceil(blockTail / ubFormer)
tailFormer = blockFormer - (loopFormer - 1) * ubFormer
tailTail = blockTail - (loopTail - 1) * ubFormer
```

R-001 的 DAG 为 `B=4,w_max=4,b_min=32`，故 `alignFactor=64`、`UB_tensor=16*ubFormer` 字节。R-002 为 `B=6,w_max=4,b_min=16`，故 `alignFactor=128`、`UB_tensor=24*ubFormer` 字节。`DoTiling` 要求 `U>=E` 且最终 `ubFormer!=0`；SetTilingData 还把 kernel local-memory 上限设为 `U-32768`。零维输入不会到达此处，因为 ACLNN 空 Tensor 已提前返回。

每个 AIV core 的 GM base 为 `blockFormer * blockIdx * sizeof(dtype)`；非末核使用 former loops/tail，末核使用 tail-block loops/tail。每个 tile 的 SIMT threads 只覆盖 `tileLength` 个有效元素，因此对齐扩出的空间不写 GM。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 直接 SIMT | binary FLOAT；`AsinOpDirect<float>::OpDag` | L0 输入 dtype 为 FLOAT。原始 FLOAT 在 DAV_3510 零拷贝命中；整数/UINT8/BOOL 经 pre-Cast 后也命中。 | 输入为 F16/BF16 到 R-002；DOUBLE 到 R-003。 |
| 2 | R-002 FP16/BF16 升精度 SIMT | binary FLOAT16/BFLOAT16；`AsinOpWithCast<T>::OpDag` | L0 输入 dtype 为 FLOAT16 或 BFLOAT16；DAV_3510 的 `isSupportedKernel` 与 `IsAiCoreSupport` 均为 true。 | DOUBLE 到 R-003；其它 API dtype 已在前处理转为 FLOAT。 |
| 3 | R-003 DOUBLE AICPU | AICPU `AsinCpuKernel` | L0 输入 dtype 为 DOUBLE；`IsAiCoreSupport` 对 DOUBLE 为 false，进入 `ADD_TO_LAUNCHER_LIST_AICPU`。 | AICPU dispatch 的其它 dtype 在 950 ACLNN selector 中不可达；未知 dtype 失败。 |

```text
if empty(self): success, no Route
selfContiguous = Contiguous(self)
if DAV_3510 && dtype in {BF16, F16, F32}:
    selfCast = selfContiguous
else if dtype in {INT8, INT16, INT32, INT64, UINT8, BOOL, BF16}:
    selfCast = Cast(selfContiguous, F32)
else:
    selfCast = selfContiguous  # F16/F32/DOUBLE

if selfCast.dtype in {F32, F16, BF16}: add AICore Asin
else:                                      add AICPU Asin  # A5 API 实际仅 DOUBLE
postCast(coreResult, out.dtype); ViewCopy(..., out)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 直接 SIMT | FLOAT binary，无运行时 TilingKey | 950/DAV_3510；非空；L0 dtype FLOAT；同 shape | 单 AIV kernel；tile 内一次 SIMT VF | `ElementwiseSch<0, AsinOpDirect<float>::OpDag>` -> `AsinCustom<float>` -> `AsinSimtCompute<float>` | 4 个 `ubFormer*4` 子区：MTE2 ping/pong + MTE3 ping/pong；Mutex 管线同步 | former/tail core 与 tileLength；SIMT `i<count`；无其它 fallback |
| R-002 FP16/BF16 升精度 SIMT | F16/BF16 binary，无运行时 TilingKey | 950/DAV_3510；非空；L0 dtype F16 或 BF16；同 shape | 单 AIV kernel；Cast0 -> SIMT VF -> Cast1 | `ElementwiseSch<0, AsinOpWithCast<T>::OpDag>` | 6 个 `ubFormer*4` 子区：MTE2 ping/pong、2 temp、MTE3 ping/pong；Cast/compute 串行锁 | tile 对齐 128 元素；有效 `count` 贯穿 Cast/SIMT/CopyOut |
| R-003 DOUBLE AICPU | `!IsAiCoreSupport(DOUBLE)` | 950；非空；L0 dtype DOUBLE | 单 AICPU task，`ParallelFor` 分片 | `AsinCpuKernel::Compute` -> `ComputeAsinDispatch` -> `ComputeAsin<double>` | 无 AIV UB；输入输出 GM；AICPU runtime 调度 | 空数据由 dispatch 成功返回；不支持 dtype 返回 PARAM_INVALID |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 原生核心 dtype | `npuArch==DAV_3510 && dtype in {BF16,F16,F32}` | 避免 ACLNN 层冗余 pre-Cast，并匹配三个 950 binary | R-001/R-002 | 其它整数类 Cast 到 F32；DOUBLE 到 AICPU | `op_api/aclnn_asin.cpp::isSupportedKernel` |
| AICore selector | 950 `IsRegBase()` 且 dtype in `{F32,F16,BF16}` | OpDef/binary 只接入这三类 AIV body | R-001/R-002 | `AsinAiCpu` | `op_api/asin.cpp::IsAiCoreSupport` |
| UB 预留 | `U >= 155648` 且对齐后 `ubFormer>0` | 120 KiB Ascend API buffer + 32 KiB DCache，剩余空间才可分 DAG buffer | R-001/R-002 | Tiling `GRAPH_FAILED` | `AsinTiling::RunTiling`; `GetMaxElemNum`; `ElewiseTiling` |
| tile 对齐 | R-001 `ubFormer%64==0`；R-002 `ubFormer%128==0` | 以 256-byte repeat 按最小 dtype 对齐 | R-001/R-002 | 取更小的向下对齐 tile；结果为 0 则失败 | `GetMaxElemNum` |
| 每核最小工作量 | `C=min(ceil(N*b_min/32768),C_avail)` | 目标约 4 KiB 最小数据位数/核，避免过多空核 | R-001/R-002 | 由 `blockNum` 收缩实际 launch | `ElewiseTiling` |
| SIMT tail | `threadIdx.x<count` 或按 1024 步进 | 不让对齐后的无效 lane 读取/写出 | R-001/R-002 | 超界 thread 不执行 | `AsinSimtCompute` |
| AICPU 空域 | `GetDataSize()==0` | 避免 `per_unit_size` 分母涉及 `total==0` | R-003 | 直接 `KERNEL_STATUS_OK` | `ComputeAsinDispatch` |

### POD/ABI

`AsinTilingData` 只内嵌一个 `EleBaseTilingData baseTiling`。按字段声明的标准布局，`EleBaseTilingData` 对齐 8 字节、大小 80 字节，外层亦为 80 字节；Host 与 Device 引用同一公共声明。以下 offset 由声明顺序和基础类型对齐确定。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AsinTilingData.baseTiling` | `dim0` | `int64_t`/64 | offset 0 | 元素 | `AdaptEleBaseTilingData: = elewiseTilingData.dim0` | 当前 `ElementwiseSch` 不读取 | shape 乘积；乘法溢出检查固定仓不可见 |
| 同上 | `coreNum` | `int32_t`/32 | offset 8 | core | `static_cast<int32_t>(elewiseTilingData.coreNum)` | 当前 scheduler 不读取 | 源值受 AIV core 数限制；显式窄化 |
| 同上 | `ubFormer` | `int32_t`/32 | offset 12 | 元素/tile | `= elewiseTilingData.ubFormer`，隐式窄化 | `Init` 分配、`Process` full tile | 必须大于 0；64/128 元素对齐 |
| 同上 | `blockFormer` | `int64_t`/64 | offset 16 | 元素/core | `= elewiseTilingData.blockFormer` | `CalcBlockOffset` | 512 元素对齐；末核实际可少于此值 |
| 同上 | `blockNum` | `int64_t`/64 | offset 24 | AIV block | `= elewiseTilingData.blockNum`; `SetBlockDim` 同值 | `Process` 判断末核 | 非空时大于 0 |
| 同上 | `ubLoopOfFormerBlock` | `int64_t`/64 | offset 32 | tile loop | `= ceil(blockFormer/ubFormer)` | 非末核 `loopNum` | 至少 1 |
| 同上 | `ubLoopOfTailBlock` | `int64_t`/64 | offset 40 | tile loop | `= ceil(blockTail/ubFormer)` | 末核 `loopNum` | 至少 1 |
| 同上 | `ubTailOfFormerBlock` | `int64_t`/64 | offset 48 | 元素 | `blockFormer-(loopFormer-1)*ubFormer` | 非末核最后一次 `Run` | `1..ubFormer` |
| 同上 | `ubTailOfTailBlock` | `int64_t`/64 | offset 56 | 元素 | `blockTail-(loopTail-1)*ubFormer` | 末核最后一次 `Run` | `1..ubFormer` |
| 同上 | `elemNum` | `int64_t`/64 | offset 64 | 元素 | `= maxElemNum` | 当前 scheduler 不读取 | 与 `ubFormer` 相同来源 |
| 同上 | `scheMode` | `uint64_t`/64 | offset 72 | 枚举值 | `GetScheMode()=1` | 当前 entry 模板固定 `ElementwiseSch<0,...>`，不读取该字段 | inactive 字段；值 1 不改变当前路线 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 直接 SIMT | API/L0 dtype selector + FLOAT binary | `ExecAsinGetWorkspaceSize`; `l0op::Asin`; `AsinTiling::RunTiling`; `ElewiseTiling` | `AsinTilingData{EleBaseTilingData}` | `asin`, FLOAT `if constexpr` | `ElementwiseSch`; `AsinOpDirect<float>`; `AsinCustom`; `AsinSimtCompute` | `asin_def.cpp`; `asin_binary.json`; `math/asin/CMakeLists.txt` |
| R-002 FP16/BF16 升精度 SIMT | API/L0 dtype selector + F16/BF16 binary | 同上，Tiling 分别实例化 `AsinOpWithCast<half/bfloat16_t>` | 同一 POD，dtype 改变编译期 `BufferNum/minDtypeBits` | `asin`, F16/BF16 `if constexpr` | `Vec::Cast`; `AsinCustom<float>`; `AsinSimtCompute<float>` | 同上，两个独立 binary 项 |
| R-003 DOUBLE AICPU | `IsAiCoreSupport(DOUBLE)==false` | `l0op::Asin` 添加 AICPU task | 无 AIV POD；AICPU context Tensor 元数据 | `REGISTER_CPU_KERNEL("Asin", AsinCpuKernel)` | `ComputeAsinDispatch`; `ComputeAsinKernel<double>` | `asin_aicpu_def.cpp`; `add_all_modules_sources` 收集 AICPU source/opdef |

## Kernel 实现

### 公共执行流程

R-001/R-002 共用单 Kernel AIV 启动：Host 按 dtype 用 atvoss DAG 做 Tiling，entry 解析 `AsinTilingData`，创建一个 `TPipe` 和 `ElementwiseSch`。scheduler 把每核 GM base 设为 `blockFormer*blockIdx`，按 `ubFormer` 遍历 tile；每个 tile 顺序执行 CopyIn、DAG compute、CopyOut，MTE2/MTE3 子区按 `tileIndex&1` ping-pong。

核心反正弦不是经典 Vector API：`AsinCustom<float>` 取得两个 UB 物理地址，用 `asc_vf_call<AsinSimtCompute<float>>(dim3(1024),...)` 进入 SIMT VF。每个 thread 从 `threadIdx.x` 起，以 `blockDim.x` 为步长执行 `asinf`。R-003 不启动 AIV，而由 AICPU runtime 对 DOUBLE 元素区间做 `std::transform(..., std::asin)`。

### Route R-001：FP32 直接 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/asin/op_api/aclnn_asin.cpp` | `ExecAsinGetWorkspaceSize` | 950 `isSupportedKernel(DT_FLOAT)==true`；整数类先 Cast 为 FLOAT | `selfCast` dtype FLOAT |
| L0 | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport`; `Asin` | RegBase 且 FLOAT 在 AICore list | `AsinAiCore` |
| Tiling | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `AsinTiling::RunTiling` | `outputDtype==DT_FLOAT` | `DoTiling<AsinOpDirect<float>::OpDag>` |
| Router | `math/asin/op_kernel/asin_apt.cpp` | `asin` | `DTYPE_X==float` | `ElementwiseSch<0,AsinOpDirect<float>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
AsinTiling::RunTiling -> ElewiseBaseTiling::DoTiling<AsinOpDirect<float>::OpDag>
Host AICore task `Asin` -> __global__ __aicore__ asin(x,y,workspace,tiling)
  -> ElementwiseSch::Init(x,y)
  -> ElementwiseSch::Process
    -> Run(tileOffset,tileLength,pingPong)
      -> ElementwiseSch::CopyIn -> Vec::CopyIn<float> -> DataCopyPad
      -> RunNormalOp<AsinCustom<float>>
        -> AsinCustom -> asc_vf_call<AsinSimtCompute<float>>(1024 threads)
          -> per-thread asinf
      -> ElementwiseSch::CopyOut -> Vec::CopyOut<float> -> DataCopyPad
```

##### 源码等价伪码

```text
# 伪码
coreBase = blockFormer * blockIdx
loopNum = (blockIdx == blockNum-1) ? ubLoopOfTailBlock : ubLoopOfFormerBlock
lastLen = (blockIdx == blockNum-1) ? ubTailOfTailBlock : ubTailOfFormerBlock
offset = 0
for tile = 0 .. loopNum-1:
    count = (tile == loopNum-1) ? lastLen : ubFormer
    pp = tile & 1
    inUB[pp][0:count] = GM_x[coreBase+offset : +count]
    simt_launch(threads=1024):
        for i = threadIdx.x; i < count; i += blockDim.x:
            outUB[pp][i] = asinf(inUB[pp][i])
    GM_y[coreBase+offset : +count] = outUB[pp][0:count]
    offset += ubFormer
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `(blockFormer*blockIdx + tileOffset)*4` bytes | `tileLength*4` bytes | FLOAT/flat ND | ACLNN/framework | CopyIn | 用户/框架；kernel 全程只读 | block base 按 512 元素 |
| TBuf MTE2 ping/pong | `bufId*blockLen`, `blockLen=ubFormer*4` | 两个子区，各 `ubFormer*4` | FLOAT | `DataCopyPad` | SIMT VF | `TPipe`；按 tile parity | `blockCount=1,blockLen=tileLength*4,srcStride=dstStride=0,pad=false` |
| TBuf MTE3 ping/pong | `bufId*blockLen` | 两个子区，各 `ubFormer*4` | FLOAT | SIMT VF | CopyOut | `TPipe`；输出完成后复用 | compute 结果直接落 output buffer，无额外 temp |
| SIMT thread state | `i=threadIdx.x+k*1024` | 1024 threads；寄存器字节固定仓不可见 | FLOAT scalar | thread load | `asinf`/store | 单次 VF call | `i<tileLength` 尾保护 |
| GM `y` | 与 GM `x` 同元素 offset | `tileLength*4` bytes | FLOAT/flat ND | CopyOut | 后续 post-Cast/ViewCopy | executor 临时 Tensor | 仅有效元素写回 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inPing["MTE2 ping · ubFormer×4 B"]:1
  inPong["MTE2 pong · ubFormer×4 B"]:1
  outPing["MTE3 ping · ubFormer×4 B"]:1
  outPong["MTE3 pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
BufferNum = TempCalcNode(0) + InputSize(1)*2 + OutputSize(1)*2 = 4
blockLen = ubFormer * MaxDtypeBytes = ubFormer * 4
UB_tensor = 4 * ubFormer * 4 = 16 * ubFormer bytes
reserved = AscendAPI 122880 + DCache 32768 = 155648 bytes
16*ubFormer <= U - 155648
dynamic_ub = N/A；SIMT VF 直接使用 scheduler 已分配的 UB 子区
Asin_AICore_workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| MTE2 ping/pong | CopyIn/MTE2 | `AsinCustom`/SIMT | 同 parity 的下下个 tile | `GetTensor/ReleaseTensor<VECIN>` 与 `<VECCALC>` Mutex | 当前计算释放后才可再写 |
| MTE3 ping/pong | SIMT compute | CopyOut/MTE3 | 同 parity 的下下个 tile | `<VECCALC>` 后 `<VECOUT>` lock/unlock | 当前搬出释放后才可再写 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn -> compute | `Mutex::Lock/Unlock<PIPE_MTE2>` 与 `<PIPE_V>` | MTE2 | SIMT wrapper | MTE2 buffer id | CopyIn 仅搬 `count*4` 字节 |
| SIMT compute | `asc_vf_call`, 无 barrier/atomic/shared buffer | 每个 SIMT thread | 同一 thread store | output UB 元素 | FLOAT `asinf`; `i<count`；步长 1024 |
| compute -> CopyOut | `<PIPE_V>` 与 `<PIPE_MTE3>` buffer protocol | SIMT store | MTE3 | MTE3 buffer id | 只搬 `count*4` 字节 |
| core/tile tail | POD former/tail fields | Tiling | `Process` | GM 有效域 | 末核/末 tile 选择真实有效长度，不写对齐 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `AsinTiling::RunTiling`; `SetTilingData` | FLOAT 绑定 direct DAG；workspace=0；blockDim=`blockNum` |
| Kernel entry/router | `math/asin/op_kernel/asin_apt.cpp` | `asin`; FLOAT branch | 单 AIV entry 绑定 direct scheduler |
| Device body/helper | `math/asin/op_kernel/arch35/asin_dag.h` | `AsinOpDirect<float>`; `AsinCustom<float>`; `AsinSimtCompute<float>` | CopyIn -> 1024-thread strided `asinf` -> CopyOut |
| Scheduler dependency | `opbase/.../elewise_sch.h` | `ElementwiseSch::Init/Process/Run` | 分核 offset、tile loop、TBuf 分配与管线锁 |

### Route R-002：FP16/BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/asin/op_api/aclnn_asin.cpp` | `isSupportedKernel` | DAV_3510 且 dtype F16/BF16 | 跳过 pre-Cast，保留原 dtype |
| L0 | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport` | RegBase list 包含 F16/BF16 | `AsinAiCore` |
| Tiling | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `AsinTiling::RunTiling` | output F16 或 BF16 | `DoTiling<AsinOpWithCast<T>::OpDag>` |
| Router | `math/asin/op_kernel/asin_apt.cpp` | `asin` | `DTYPE_X==half` 或 `bfloat16_t` | 对应 cast DAG scheduler |

#### Device 数据流

##### 函数级调用链

```text
Host AICore task `Asin` -> asin<T=half|bfloat16_t>
  -> ElementwiseSch<0,AsinOpWithCast<T>::OpDag>::Process
    -> Vec::CopyIn<T> -> DataCopyPad
    -> Vec::Cast<float,T,0> -> AscendC::Cast(...,RoundMode::CAST_NONE,count)
    -> AsinCustom<float>
      -> asc_vf_call<AsinSimtCompute<float>>(1024 threads) -> asinf
    -> Vec::Cast<T,float,1> -> AscendC::Cast(...,RoundMode::CAST_RINT,count)
    -> Vec::CopyOut<T> -> DataCopyPad
```

##### 源码等价伪码

```text
# 伪码；core/tile offset 与 R-001 相同
for each valid tile(count):
    inT = DataCopyPad(GM_x, bytes=count*sizeof(T))
    tmp0_float = Cast(inT, CAST_NONE, count)
    simt_launch(1024):
        for i = threadIdx.x; i < count; i += 1024:
            tmp1_float[i] = asinf(tmp0_float[i])
    outT = Cast(tmp1_float, CAST_RINT, count)
    DataCopyPad(GM_y, outT, bytes=count*sizeof(T))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/y` | `(blockFormer*blockIdx+tileOffset)*sizeof(T)` | `count*sizeof(T)` | F16 或 BF16/flat ND | framework / CopyOut | CopyIn / framework | executor temp | 有效搬运 2 bytes/元素 |
| MTE2 ping/pong | `bufId*blockLen` | 各 `ubFormer*4` bytes，逻辑有效 `count*2` | T | CopyIn | Cast0 | `TPipe` | 物理按最大 dtype 4 B/元素分区；未用后半不参与计算 |
| temp0 | `bufId*blockLen` | `ubFormer*4` | FLOAT | Cast0 | SIMT | scheduler temp；跨 tile 同址复用 | Cast0 消费完 in buffer 后可释放 |
| temp1 | `bufId*blockLen` | `ubFormer*4` | FLOAT | SIMT | Cast1 | scheduler temp；跨 tile同址复用 | SIMT 完成后才降精度 |
| MTE3 ping/pong | `bufId*blockLen` | 各 `ubFormer*4`，逻辑有效 `count*2` | T | Cast1 | CopyOut | `TPipe` | Cast1 直接写 output buffer，避免额外物化 |
| SIMT thread state | `i=threadIdx.x+k*1024` | 1024 threads；寄存器字节固定仓不可见 | FLOAT | tmp0 load | tmp1 store | 单次 VF call | `i<count` |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inPing["MTE2 ping · ubFormer×4 B"]:1
  inPong["MTE2 pong · ubFormer×4 B"]:1
  tmp0["Cast-up temp · ubFormer×4 B"]:1
  tmp1["asin temp · ubFormer×4 B"]:1
  outPing["MTE3 ping · ubFormer×4 B"]:1
  outPong["MTE3 pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
TempCalcNode = 2                  # Cast0 与 Asin 的 float 中间值
BufferNum = 2 + 1*2 + 1*2 = 6
blockLen = ubFormer * 4
UB_tensor = 6 * ubFormer * 4 = 24 * ubFormer bytes
24*ubFormer <= U - 155648
ubFormer % 128 == 0
dynamic_ub = N/A；SIMT 使用 tmp0/tmp1 UB
workspace_total(Asin AICore) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| MTE2 ping/pong | CopyIn | Cast0 | 同 parity 下下个 tile | MTE2/V lock-unlock | Cast0 已消费 |
| temp0 | Cast0 | SIMT | 下一 tile Cast0 | VECCALC lock-unlock；完整 `Run` 顺序 | 当前 SIMT 已消费 |
| temp1 | SIMT | Cast1 | 下一 tile SIMT | VECCALC lock-unlock | Cast1 已消费 |
| MTE3 ping/pong | Cast1 | CopyOut | 同 parity 下下个 tile | V/MTE3 lock-unlock | 当前 CopyOut 已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| T -> FLOAT | `AscendC::Cast`, mode 0 | CopyIn | SIMT | temp0 | mode 0 对应 `CAST_NONE`；处理 `count` 个元素 |
| asin | `asc_vf_call`/SIMT | temp0 | Cast1 | temp1 | 始终 `asinf(float)`；无共享内存、atomic、barrier |
| FLOAT -> T | `AscendC::Cast`, mode 1 | SIMT | CopyOut | MTE3 buffer | mode 1 对应 `CAST_RINT`；F16/BF16 均使用同一降精度 mode |
| tail | POD + SIMT guard | Tiling | scheduler/thread | 所有 buffer/GM | Cast、SIMT、DMA 都传同一有效 `count` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | F16/BF16 branches | 同一 cast DAG；额外预留 155648 B |
| Kernel entry/router | `math/asin/op_kernel/asin_apt.cpp` | half/bfloat16 branches | 两种 binary 绑定同构模板 |
| Device body/helper | `math/asin/op_kernel/arch35/asin_dag.h` | `AsinOpWithCast<T>` | `CopyIn -> Cast0 -> AsinCustom<float> -> Cast1 -> CopyOut` |
| Cast primitive | `opbase/.../util/vec.h`; `asc-devkit/.../kernel_utils_mode.h` | `Vec::Cast`; `RoundMode` | 整数 0/1 分别映射 CAST_NONE/CAST_RINT |

### Route R-003：DOUBLE AICPU

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/asin/op_api/aclnn_asin.cpp` | `ExecAsinGetWorkspaceSize` | DOUBLE 属输出类 dtype，target dtype 保持 DOUBLE | 不产生 pre-Cast |
| L0 | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport`; `Asin` | DOUBLE 不在 RegBase AICore list | `AsinAiCpu` |
| Launcher | 同上 | `ADD_TO_LAUNCHER_LIST_AICPU` | op type `Asin`，1 input/1 output | AICPU task |
| Dispatch | `math/asin/op_kernel_aicpu/asin_aicpu.cpp` | `ComputeAsinDispatch` | `input_type==DT_DOUBLE` | `ComputeAsin<std::double_t>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Asin -> AsinAiCpu -> ADD_TO_LAUNCHER_LIST_AICPU(Asin,input,output)
  -> REGISTER_CPU_KERNEL("Asin", AsinCpuKernel)
  -> AsinCpuKernel::Compute
    -> CheckAsin -> NormalCheck + ExtraCheckAsin
    -> ComputeAsinDispatch
      -> ComputeAsin<double>
        -> ComputeAsinKernel<double>
          -> CpuKernelUtils::ParallelFor
            -> std::transform(..., ScalarAsin<double>)
              -> std::asin(double)
```

##### 源码等价伪码

```text
# 伪码
require 1 input, 1 output
require input.data != null && output.data != null
require input.dtype == output.dtype && input.bytes == output.bytes
if input.bytes == 0: return OK
require dtype in {F16,F32,F64}; A5 ACLNN 当前路线实际为 F64
total = input.NumElements()
cores = GetCPUNum(ctx)
shards = min(max(1, int64(cores)-2), total)
perUnit = total / shards
ParallelFor(total, perUnit):
    for i in [begin,end): output[i] = std::asin(input[i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| AICPU input GM | `input + begin` | `(end-begin)*8` bytes | DOUBLE/flat | framework | `std::transform` | executor Tensor | 各 ParallelFor shard 不重叠 |
| AICPU output GM | `output + begin` | `(end-begin)*8` bytes | DOUBLE/flat | `std::asin` transform | post-Cast/ViewCopy | executor Tensor | 同索引逐元素写 |
| CPU locals | `total,cores,per_unit_size,begin,end` | 固定仓未公开运行时栈字节 | scalar | AICPU body | AICPU body | 调用栈 | 无共享 scratch |
| AICPU workspace declaration | runtime 分配位置固定仓不可见 | 默认配置字符串 `100` bytes | opaque | OpDef/runtime | body 未读取 | runtime | 不与 AIV UB 复用 |

##### 分层内存布局

N/A：此路线不启动 AIV，不分配 `TQue/TBuf/dynamic UB`；实际替代路径是 AICPU runtime 对 GM 输入/输出做 CPU `ParallelFor` 分片。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
AICPU body explicit scratch = 0 bytes
AICPU OpDef default workspace declaration = 100 bytes
ACLNN workspace_total = executor 汇总值，不能仅以 100 bytes 代替
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 无显式中间子区 | N/A | N/A | N/A | `ParallelFor` 返回后 task 完成 | shard 的 `[begin,end)` 必须互不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| AICPU 分片 | `CpuKernelUtils::ParallelFor` | runtime shard 调度 | 返回值 | 输出索引区间 | `per_unit_size=total/min(max(1,cores-2),total)` |
| 标量计算 | `std::asin(double)` | input element | output element | 单元素 | DOUBLE；无额外 cast |
| 空域 | `GetDataSize()==0` | dispatch | 提前返回 | 避免空分片公式 | 返回 OK |
| 失败 | return status | ParallelFor/body | executor | 后续节点 | 非 OK 被 `ComputeAsin` 记录并传播 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/asin/op_api/asin.cpp` | `AsinAiCpu`; `Asin` | DOUBLE 落入 AICPU launcher；无 AIV Tiling |
| Kernel entry/router | `math/asin/op_kernel_aicpu/asin_aicpu.cpp` | `REGISTER_CPU_KERNEL`; `ComputeAsinDispatch` | dtype switch 选择 DOUBLE body |
| Device body/helper | 同上 | `ComputeAsinKernel<double>`; `ScalarAsin<double>` | CPU 分片与 `std::asin` |
| 构建/注册 | `math/asin/op_kernel_aicpu/asin_aicpu_def.cpp`; `cmake/func.cmake` | `ApplyMathAicpuDefaultCfg`; `AICPU_SRCS` | AICPU SO/工作区配置与 source 收集 |

## 算子特定优化

### Trick T-001：950 原生 dtype 跳过 ACLNN 前置 Cast

#### 触发与源码依据

- 适用 Route：R-001（原始 FLOAT）、R-002（FLOAT16/BFLOAT16）
- 触发条件：`GetCurNpuArch()==DAV_3510` 且连续输入 dtype 为 BF16/F16/F32
- Golden 来源：[`math/asin/op_api/aclnn_asin.cpp:139-160`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/aclnn_asin.cpp#L139-L160)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    auto castDtype = selfContiguous->GetDataType();
    const aclTensor* selfCast = nullptr;

    // 判断条件：昇腾950版本且数据类型为Kernel原生支持的类型，则跳过Cast
    if (isSupportedKernel(castDtype)) {
        // 950新增逻辑：kernel支持的数据类型零拷贝复用输入Tensor，避免冗余Cast开销
        selfCast = selfContiguous;
    } else {
        // 原有逻辑优化，目标类型与原类型相同时，跳过Cast算子调用
        DataType targetDtype = castDtype;

        if (!CheckType(castDtype, DTYPE_OUT_LIST) || castDtype == DataType::DT_BF16) {
            targetDtype = DataType::DT_FLOAT;
        }

        if (targetDtype != castDtype) {
            selfCast = l0op::Cast(selfContiguous, targetDtype, uniqueExecutor.get());
        } else {
            selfCast = selfContiguous;
        }
    }
```

#### 机制与设计原因

Host 在组 executor 时直接让 `selfCast` 别名 `selfContiguous`，因此原生三 dtype 不产生独立 pre-Cast Tensor/task。FP16/BF16 所需的升精度被融合在单个 Asin AIV DAG 内，数据只从 GM 搬入一次；FLOAT 直接计算。该机制减少的是可由源码证明的 executor 节点、临时 Tensor 与 GM 往返，而不是声称固定性能比例。

#### 迁移边界

- 适用：950/DAV_3510 上具有自身 BF16/F16/F32 binary、且内部能正确处理精度转换的一元算子。
- 前提：L0 AICore selector、OpDef、binary config 与 kernel dtype specialization 必须同时支持该 dtype。
- 失效：整数/UINT8/BOOL 仍需 Cast 到 FLOAT；DOUBLE 必须走 R-003 AICPU。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能仅因平台是 950 就跳过所有 Cast：固定源码明确把条件限制在 BF16/F16/F32；对整数类直接传入 `Asin` 将不匹配 950 binary，对 DOUBLE 则改变既定 AICPU 路线。

### Trick T-002：半精度核心在单 DAG 内升精度并直接写 output buffer

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：AICore L0 dtype 为 FLOAT16 或 BFLOAT16
- Golden 来源：[`math/asin/op_kernel/arch35/asin_dag.h:75-85`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/arch35/asin_dag.h#L75-L85)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
struct AsinOpWithCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, AsinDag::CastModeToFp32>, OpCopyIn>;
    using OpAsin = Bind<AsinDag::AsinCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, AsinDag::CastModeToBf16>, OpAsin>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
```

#### 机制与设计原因

一次 GM CopyIn 后在 UB 内完成 `T -> FLOAT -> asinf -> T`，避免把 FLOAT 中间结果物化到 GM。`MemLevel::LEVEL_2` 的活跃节点分析只保留两个 FLOAT temp；末次 Cast 的结果直接分配到 MTE3/output 子区。Cast0 的 mode 0 来自常量 `CastModeToFp32`，即 `CAST_NONE`；Cast1 的 mode 1 即 `CAST_RINT`。这同时固定了精度顺序与 buffer 复用边界。

#### 迁移边界

- 适用：输入输出同为 16-bit、超越函数仅有可靠 FP32 body的一元逐元素算子。
- 前提：UB 能容纳 `24*ubFormer` 与 155648-byte 预留，且下游接受 `CAST_RINT` 降精度。
- 失效：FLOAT 应使用 R-001，DOUBLE 使用 R-003；若需保留 FP32 输出，不应在核心 DAG 中降回 16-bit。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码给出的可验证反例是把 BF16 在 ACLNN 层先 Cast 成 FLOAT：950 专门加入 `isSupportedKernel` 分支避免该额外节点；若删除 DAG 内 Cast1，则 kernel binary 的输出 dtype/ABI 又会与 OpDef 不一致。

## 模板与实现对比

| 对比维度 | R-001 FP32 直接 SIMT | R-002 FP16/BF16 升精度 SIMT | R-003 DOUBLE AICPU | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | L0 dtype F32 | L0 dtype F16/BF16 | L0 dtype DOUBLE | 950 AICore binary 只列前三种原生 dtype |
| 模板与 dtype | `AsinOpDirect<float>` | `AsinOpWithCast<T>`，计算 FLOAT | `ComputeAsinKernel<double>` | 兼顾原生 950 binary 与 DOUBLE 公共 API |
| 循环与分核 | AIV block + UB tile + 1024 SIMT thread | 同左，tile 对齐更大 | CPU `ParallelFor` shard | 分别受 UB 与 CPU core 数控制 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单 TBuf，4 子区；thread 寄存器不可见 | 单 TBuf，6 子区；thread 寄存器不可见 | 无 AIV UB | 无 TQue、通信 window、atomic |
| 内存总占用与复用 | `16*ubFormer` + 155648 B 预留 | `24*ubFormer` + 155648 B 预留 | body 无显式 scratch；OpDef 100 B | DAG 活跃节点决定 buffer 数 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + `asc_vf_call/asinf` | DataCopyPad + Cast + `asc_vf_call/asinf` + Cast | `std::transform/std::asin` | 无 NDDMA/通信 |
| 同步、精度与 tail | Mutex pipe；F32；`i<count` | Mutex pipe；T->F32->T；`i<count` | ParallelFor completion；F64 | 全路线仅写有效元素 |
| fallback/失败边界 | Tiling/launcher 失败 | Tiling/launcher 失败 | dtype/context/ParallelFor 失败 | 不存在 Cube fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档称支持，但固定仓缺少活动 Host/Tiling/Kernel 闭包，未建卡 | 公开文档称支持且 API selector 有 910B/910_93 信号，但固定仓缺少对应 Tiling/Kernel/构建接线，未建卡 | 公开支持；source closed |
| Host/Launcher | L0 selector 源码存在，平台实际 kernel producer 不闭合 | BF16 API 开关与 L0 selector 可见，实际 A3 body 不闭合 | DAV_3510 原生三 dtype AICore，DOUBLE AICPU |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 atvoss scheduler + SIMT VF；AICPU fallback |

## 限制与失败边界

- 仅覆盖非原地 `aclnnAsin`；不得把 `aclnnInplaceAsin` 的别名/写回语义并入本卡。
- 公共 API 允许 self/out dtype 不同；核心先按 self 的规范化 dtype 计算，再按 out dtype Cast。复用时不得假设核心总以 out dtype 计算。
- 公开文档声明最大 8 维，但固定的 `CheckParams1In1Out` 正文只显式校验 dtype 与同 shape；维数限制在框架宏中的落实位置固定仓不可见。
- AICore Tiling 将 shape 各维乘入 `int64_t dim0`，固定源码未显示乘法溢出保护；POD 的 `ubFormer` 又从 `int64_t` 写入 `int32_t`，未显示范围检查。
- `U<=155648` 或向下对齐后 `ubFormer==0` 会使 Tiling 失败；`SetLocalMemorySize` 还要求 `U-32768` 可表示为 `uint32_t`，固定仓未额外检查下溢/窄化。
- `asinf/std::asin` 的设备库内部近似、NaN payload、异常标志和具体 ULP 误差不在固定仓可见范围内；本卡不推测。
- 通用 `Contiguous/Cast/ViewCopy/CommonOpExecutorRun` 的内部 workspace 切片、事件和所有可选 kernel body属于框架/其它逻辑算子边界；本卡闭合其调用条件与顺序，不把它们伪造成 Asin 自有 POD。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/asin/docs/aclnnAsin&aclnnInplaceAsin.md` | 产品表、`aclnnAsin*` | 公开平台、dtype、shape、非连续与空 Tensor 约束 |
| 入口声明/约束 | `math/asin/op_api/aclnn_asin.h` | `aclnnAsinGetWorkspaceSize`; `aclnnAsin` | ABI 与 executor 图说明 |
| 入口实现 | `math/asin/op_api/aclnn_asin.cpp` | `ExecAsinGetWorkspaceSize`; `isSupportedKernel` | 校验、前后 Cast、空域、workspace 与 launch 顺序 |
| L0 selector | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport`; `AsinAiCore`; `AsinAiCpu`; `Asin` | AICore/AICPU 分发 |
| Host 注册 | `math/asin/op_host/asin_def.cpp` | `ops::Asin` | 950 dtype、AICore config、kernel file |
| Host/Tiling | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `AsinTiling` | dtype DAG、shape、workspace、local memory、blockDim |
| POD | `math/asin/op_kernel/arch35/asin_tilingdata.h` | `AsinTilingData` | 外层 ABI |
| Kernel DAG/body | `math/asin/op_kernel/arch35/asin_dag.h` | `AsinSimtCompute`; `AsinOp*` | SIMT loop、Cast/direct DAG |
| Kernel entry | `math/asin/op_kernel/asin_apt.cpp` | `asin` | AIV-only entry 与三 dtype specialization |
| Binary config | `math/asin/op_host/config/ascend950/asin_binary.json` | `op_list` | 950 BF16/F16/F32 binaries |
| AICPU body | `math/asin/op_kernel_aicpu/asin_aicpu.cpp` | `AsinCpuKernel`; `ComputeAsinKernel` | DOUBLE dispatch、CPU 分片、错误路径 |
| AICPU 注册 | `math/asin/op_kernel_aicpu/asin_aicpu_def.cpp` | `ops::Asin` | dtype 与 AICPU config |
| 构建接线 | `math/asin/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`; `add_all_modules_sources` | 950/mc62 与 arch35 接线 |
| 构建接线 | `cmake/func.cmake` | `add_all_modules_sources` | API/Tiling/AICPU source 收集 |
| POD 直接依赖 | `pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData` | 字段声明与 ABI 顺序 |
| Tiling 直接依赖 | `pkg_inc/op_common/atvoss/elewise/elewise_tiling.h` | `ElewiseBaseTiling::DoTiling` | DAG buffer 参数生产 |
| Tiling 直接依赖 | `src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling`; `AdaptEleBaseTilingData` | 分核/tile公式与逐字段 producer |
| Scheduler 直接依赖 | `pkg_inc/op_common/atvoss/elewise/elewise_sch.h` | `ElementwiseSch` | GM offset、TBuf、循环、调用与同步 |
| DAG 直接依赖 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | BufferNum/MaxDtypeBytes 与 MemLevel |
| DAG 直接依赖 | `pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo::GetBufferNumLevel2` | 4/6 buffer 字节账 |
| 搬运/Cast 依赖 | `pkg_inc/op_common/atvoss/util/vec.h` | `Vec::CopyIn`; `Vec::Cast`; `Vec::CopyOut` | DataCopyPad 参数与 Cast mode传递 |
| 同步直接依赖 | `pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor`; `ReleaseTensor` | MTE2/V/MTE3 Mutex 协议 |
| Cast enum 依赖 | `impl/basic_api/utils/kernel_utils_mode.h` | `RoundMode` | mode 0/1 的枚举语义 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 可复用 80-byte `AsinTilingData{EleBaseTilingData}` | 字段顺序/位宽/offset；`ubFormer`/`coreNum` 的 32-bit 窄化；inactive 字段仍须保留 | `EleBaseTilingData`; `AsinTilingData` |
| 调用顺序 | `Contiguous -> optional pre-Cast -> Asin -> post-Cast -> ViewCopy` | 空 Tensor 必须在组图前返回；任何节点失败阻断后续 | `ExecAsinGetWorkspaceSize` |
| workspace/内存 | AICore workspace=0；R-001 4 子区；R-002 6 子区 | 155648 B 预留、最大 dtype 4 B、ping/pong与 temp 生命周期 | `SetTilingData`; `ElementwiseSch::Init`; `DAGSch` |
| 同步 | atvoss buffer id 的 MTE2/V/MTE3 Mutex protocol | 不得在 consumer 完成前覆盖同 buffer id；temp 跨 tile仅在完整 Run 后复用 | `GetTensor`; `ReleaseTensor`; `Run` |
| 精度与尾块 | F32 direct；F16/BF16 `CAST_NONE -> asinf -> CAST_RINT`；DOUBLE `std::asin` | `count` 必须贯穿 DMA/Cast/SIMT，thread 必须保留 `i<count` | `AsinOpWithCast`; `AsinSimtCompute`; `ComputeAsinKernel` |
| 失败边界 | dtype、shape、资源、launcher与 CPU dispatch 失败均传播 | 不得把 DOUBLE 塞入 950 AICore；不得忽略 Tiling `ubFormer==0` | `CheckParams1In1Out`; `ElewiseTiling`; `ComputeAsinDispatch` |