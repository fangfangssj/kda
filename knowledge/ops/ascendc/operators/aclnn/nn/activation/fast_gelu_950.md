---
type: Operator
title: "FastGelu 950 ACLNN 算子知识"
description: "说明 aclnnFastGelu 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/docs/aclnnFastGelu.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_api/aclnn_fast_gelu.h"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_api/aclnn_fast_gelu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_api/fast_gelu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_host/fast_gelu_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_host/arch35/fast_gelu_tiling_arch35.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_kernel/arch35/fast_gelu_struct.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_kernel/fast_gelu_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_host/config/ascend950/fast_gelu_binary.json"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu_v2/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/fast_gelu_v2/docs/aclnnFastGeluV2.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/fast_gelu_v2/CMakeLists.txt"
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
  - resource: "https://gitcode.com/cann/asc-devkit/blob/7a7156352f594b5ca79b533f72c5a99f19b87983/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_maskreg_impl.h"
    role: implementation
  - resource: "https://gitcode.com/cann/asc-devkit/blob/7a7156352f594b5ca79b533f72c5a99f19b87983/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h"
    role: implementation
  - resource: "https://gitcode.com/cann/asc-devkit/blob/7a7156352f594b5ca79b533f72c5a99f19b87983/impl/basic_api/dav_3510/kernel_operator_common_impl.h"
    role: implementation
created_at: "2026-08-25T18:45:28Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# FastGelu 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnFastGeluGetWorkspaceSize` + `aclnnFastGelu`；当前默认活动构建没有公开 `aclnnFastGeluV2`。 |
| 计算语义 | 对每个元素计算 `y = x / (1 + exp(-1.702 * x))`。 |
| 输入输出 | `self`/`out` 同 shape、dtype 和 storage format；ND、0～8 维、可非连续；950 支持 FLOAT16、BFLOAT16、FLOAT32。 |
| 执行类型 | 核心节点为单 Kernel `AIV_VECTOR`；FP32 或 FP16/BF16 的 Vector/MicroAPI 路线均无 Cube/AIC/MIX_CV。框架 `Contiguous/ViewCopy` 作为前后处理边界。 |
| Kernel 路线 | R-001 FP32 直接 MicroAPI；R-002 FP16/BF16 先升精度到 FP32，再降回输入 dtype。 |
| 核心资源 | GM 输入/输出；单 `TBuf<VECCALC>` 切为 4 或 6 个等长子区；寄存器 `x/denominator/result`；核心节点固定 16 MiB system workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN、L0 launcher、950 OpDef/Tiling/POD、entry、atvoss scheduler、逐 Route DAG/body、MicroAPI 依赖与构建接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `N = product(self.shape)`，则对 `0 <= i < N`：

```text
denominator[i] = exp(-1.702f * float(self[i])) + 1.0f
result[i]      = float(self[i]) / denominator[i]
out[i]         = cast_to_input_dtype(result[i])
```

FP32 路线不执行 cast，并把 `DivSpecificMode.precisionMode` 设为 `true`；FP16/BF16 路线先以 `Cast<float,T,0>` 升为 FP32，以 `precisionMode=false` 做除法，再以 `Cast<T,float,1>` 降精度。`MaskMergeMode::ZEROING` 使尾轮无效 lane 不保留旧寄存器值；只有当前 mask 覆盖的结果会写回 UB。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；与 `out` 同 shape 和 storage format；0～8 维；可非连续 | FLOAT16、BFLOAT16、FLOAT32 | 必须非空；950 属 RegBase，三种 dtype 均进入 API 白名单。空 Tensor 在第一段直接成功返回。 |
| `out` | ND；与 `self` 同 shape、dtype 和 storage format；可非连续 | 与 `self` 相同 | 必须非空；最终由 `ViewCopy` 写入，因而支持非连续视图。 |
| `workspaceSize` | Host 标量输出 | `uint64_t` | executor 汇总完整任务图所需空间；FastGelu AICore 节点自身登记 `16,777,216` 字节 system workspace。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并转移所有权；保存 `Contiguous -> FastGelu -> ViewCopy` 任务图。 |
| `workspace` | Device 地址 | `void*` | 第二段传入，必须配合第一段返回的大小；核心 entry 对空地址直接返回。 |
| `stream` | runtime stream | `aclrtStream` | `CommonOpExecutorRun` 在该 stream 上执行 executor。 |

### 入口与版本

固定提交的正式公开入口是 `aclnnFastGeluGetWorkspaceSize/aclnnFastGelu`。同仓 `experimental/activation/fast_gelu_v2/` 虽声明 `aclnnFastGeluV2`，但根构建选项 `ENABLE_EXPERIMENTAL` 默认 `OFF`；正式 `activation/fast_gelu_v2/` 的 CMake 又标记 `ACLNNTYPE aclnn_exclude`，只有图模式实现。因此 950 的实际活动公开 ACLNN 版本仍是无后缀 `aclnnFastGelu`，不能把 experimental V2 冒充正式版本，也不能把 V2 的分段多项式 body 接到本入口。

公开文档列出 A2、A3、950 均支持；但固定提交中 `FastGelu` OpDef、CMake、binary config 和 Host/Kernel 只接入 `ascend950/arch35`。950 可形成源码闭环；A2/A3 缺少该提交内的活动 Host/Tiling/POD/Kernel/构建链，因此不生成无后缀卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnFastGeluGetWorkspaceSize` 的真实顺序为：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，`CheckNotNull` 检查 `self/out`；失败分别返回公共输入错误或 `ACLNN_ERR_PARAM_NULLPTR`。
2. 创建 `aclOpExecutor`；失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
3. `CheckDtypeValid` 要求 `self.dtype == out.dtype` 且 dtype 在平台白名单。950 的 `IsRegbase()` 为真，白名单为 FLOAT/FLOAT16/BF16。
4. `CheckShape` 要求 shape 相等且 `self` 不超过 8 维；`CheckFormat` 要求 storage format 相同。
5. 若 `self->IsEmpty()`，写 `workspaceSize=0`，转移 executor 并返回成功，不添加任何 Device 节点。
6. `l0op::Contiguous(self)` 产生连续输入或复用已有连续 Tensor；失败返回内部空指针错误。
7. `l0op::FastGelu(selfContiguous)` 分配同 view shape、同 dtype 的临时输出，并由 `ADD_TO_LAUNCHER_LIST_AICORE(FastGelu, ...)` 添加核心任务。
8. `l0op::ViewCopy(fastGeluResult,out)` 把连续结果写到用户输出视图；失败即停止组图。
9. 读取 executor 总 workspace，转移 executor；第二段仅调用 `CommonOpExecutorRun`。

### 参数、Workspace 与生命周期

`selfContiguous` 与 `fastGeluResult` 都归 `uniqueExecutor` 管理，至少存活至相应下游节点消费完成；连续输入可由框架复用原 Tensor。用户 `out`、workspace 与 stream 必须在第二段执行期间有效。

核心 Tiling 写 `currentWorkspace[0] = SYSWORKSPACE + usrWorkspaceSize = 16,777,216 + 0` 字节。entry 先检查 workspace，调用 `SetSysWorkspace`，再取得 `GetUserWorkspace(workspace)`；950 的 `RESERVED_WORKSPACE` 同为 16 MiB，当前 body 不读写 user workspace，但仍把其非空作为继续执行条件。ACLNN 对外总 workspace 还可能包含框架 `Contiguous/ViewCopy` 节点需求，具体总和以第一段返回值为准。

### 启动图与失败传播

非空输入的单核心 Kernel 图如下：

```text
self
  -> framework Contiguous（连续输入可复用）
  -> FastGelu AIV kernel `fast_gelu`
       blockDim = EleBaseTilingDataV2.blockNum
       POD      = EleBaseTilingDataV2
       workspace slice = 16 MiB system + 0 user
       -> R-001 FP32 或 R-002 FP16/BF16
  -> framework ViewCopy
  -> out
```

任务由同一 executor 按数据依赖在传入 stream 上提交；固定仓没有为 FastGelu 自有节点定义额外 event 或跨 Kernel 边。第一段任一校验、分配或 launcher 添加失败都会立即返回，后续节点不再追加；第二段原样返回 `CommonOpExecutorRun` 状态。entry 的 workspace/user-workspace 空检查是无状态提前终止，不写输出。

## Route 与参数设计

### 分核与资源计算

令 `N` 为输出 storage shape 各维乘积，`C_avail=GetCoreNumAiv()`，`U` 为平台 UB 字节数。DAG 编译期给出 buffer 数 `B`、最大元素字节数 `w_max`、最小元素 bit 数 `b_min`：

```text
C           = min(ceil(N * b_min / 32768), C_avail)
blockFormer = align_up(ceil(N / C), 512)                    # 元素/core
blockNum    = ceil(N / blockFormer)                         # 实际 AIV blockDim
blockTail   = N - (blockNum - 1) * blockFormer              # 元素

rawTile     = floor(U * 8 / (B * w_max * 8))                # 元素
alignFactor = 256 * 8 / b_min                               # 元素
ubFormer    = floor(rawTile / alignFactor) * alignFactor    # 元素/tile
loopFormer  = ceil(blockFormer / ubFormer)
loopTail    = ceil(blockTail / ubFormer)
tailFormer  = blockFormer - (loopFormer - 1) * ubFormer
tailTail    = blockTail - (loopTail - 1) * ubFormer
```

R-001 为 `B=4,w_max=4,b_min=32`，所以 `alignFactor=64`、`UB_total=16*ubFormer` 字节。R-002 为 `B=6,w_max=4,b_min=16`，所以 `alignFactor=128`、`UB_total=24*ubFormer` 字节。Tiling 要求平台 core 数非零、shape 各维非零、UB 不小于 extra size（本算子为 0），且对齐后的 `ubFormer != 0`。ACLNN 空 Tensor 已在进入 Tiling 前返回。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 直接 MicroAPI | `GET_TPL_TILING_KEY(1,TPL_FP32)`；`ElementwiseSch<1,FastGeluNoCast<float>::OpDag>` | 非空 950 输入；API dtype FLOAT；Tiling `outputDtype==DT_FLOAT`。 | FLOAT16/BF16 进入 R-002；其它 dtype 在 API 校验失败。 |
| 2 | R-002 FP16/BF16 升精度 | `GET_TPL_TILING_KEY(1,TPL_FP16/TPL_BF16)`；`ElementwiseSch<1,FastGeluNeedCast<T>::OpDag>` | 非空 950 输入；API dtype FLOAT16 或 BFLOAT16；Tiling 按 dtype 分别实例化 `half/bfloat16_t`。 | 无后续 Route；其它 dtype 在 API 校验失败。 |

```text
if self is empty:
    workspaceSize = 0; success; no Route
require dtype(self) == dtype(out) in {F16, BF16, F32}
require shape/format equal and rank <= 8
scheMode = 1
if dtype == F16: key = (1,TPL_FP16); R-002<half>
else if dtype == BF16: key = (1,TPL_BF16); R-002<bfloat16_t>
else if dtype == F32: key = (1,TPL_FP32); R-001<float>
else: GRAPH_FAILED / API invalid dtype
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 直接 MicroAPI | `(scheMode=1,dType=TPL_FP32)` | 950；非空；x/y FLOAT、同 shape/format | 单 AIV kernel | `FastGeluNoCast<float>::OpDag` -> `FastGeluCustom<float,true>` | 4 个 `ubFormer*4` 子区：MTE2 ping/pong + MTE3 ping/pong；Mutex pipe 锁；计算中间量只在寄存器 | 64 元素 tile 对齐；每轮 `UpdateMask(count)`；未知模板 dtype 无 body |
| R-002 FP16/BF16 升精度 | `(1,TPL_FP16)` / `(1,TPL_BF16)` | 950；非空；x/y 为同一种 F16/BF16、同 shape/format | 单 AIV kernel | `FastGeluNeedCast<T>::OpDag` -> `Cast<float,T,0>` -> `FastGeluCustom<float,false>` -> `Cast<T,float,1>` | 6 个 `ubFormer*4` 子区：MTE2 ping/pong + 2 个 FP32 temp + MTE3 ping/pong；Mutex pipe 锁 | 128 元素 tile 对齐；有效 count 贯穿 Cast/MicroAPI/CopyOut；无其它 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台/API dtype | 950 `IsRegbase()`；dtype 同且属于 `{F32,F16,BF16}` | 只存在三个 950 binary/template dtype | 两 Route | API `ACLNN_ERR_PARAM_INVALID` | `op_api/aclnn_fast_gelu.cpp::GetDtypeSupportList/CheckDtypeValid` |
| 非空域 | `!self->IsEmpty()`；Tiling 各维非零 | 通用 Tiling 的 `C/blockTail/loop` 公式要求正元素域 | 两 Route | ACLNN 空域直接成功；绕过 API 的零维 Tiling 失败 | `aclnnFastGeluGetWorkspaceSize`; `ElewiseTiling` |
| 每核最小量 | `C=min(ceil(N*b_min/32768),C_avail)` | 以最小 dtype bit 数控制约 4 KiB/核，再受 AIV 核数上限约束 | 两 Route | `blockNum` 自动收缩，无另一 Route | `ElewiseTiling` |
| tile 对齐 | R-001 `ubFormer%64==0`；R-002 `ubFormer%128==0` | 对齐到 256-byte repeat 所覆盖的最小 dtype 元素数 | 两 Route | 向下收缩 tile；结果为 0 则 Tiling 失败 | `GetMaxElemNum` |
| workspace | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 需要登记 16 MiB system workspace；即使当前 body 不消费 user 区也执行检查 | 两 Route | entry 直接返回且不写输出 | `fast_gelu` |
| MicroAPI tail | `loopNum=ceil(count/VL)`，每轮 `UpdateMask(count)` post-update | 只允许当前剩余元素 lane 读写；避免对齐尾部污染输出 | 两 Route | 无效 lane zeroing、不提交到 GM | `FastGeluCustom::FastGeluCustom`; `UpdateMaskImpl` |
| 计数窄化 | scheduler 的 `tileLength` 传给 `uint32_t count`，body 再窄化 `loopNum` 为 `uint16_t` | MicroAPI 接口和循环变量位宽固定 | 两 Route | 固定源码无独立溢出 fallback；实际上界来自 UB-derived `ubFormer` | `ElementwiseSch::CallImpl`; `FastGeluCustom` |

### POD/ABI

`EleBaseTilingDataV2` 是 `EleBaseTilingData` 的别名；按字段声明的标准布局，结构对齐 8 字节、大小 80 字节。Host 的 `GetTilingData<EleBaseTilingDataV2>` 与 entry 的 `REGISTER_TILING_DEFAULT/GET_TILING_DATA_WITH_STRUCT` 使用同一声明。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `EleBaseTilingDataV2` | `dim0` | `int64_t`/64 | offset 0 | 元素 | `AdaptEleBaseTilingData: = elewiseTilingData.dim0` | 当前 scheduler 不读取 | shape 乘积；固定源码无显式乘法溢出检查 |
| 同上 | `coreNum` | `int32_t`/32 | offset 8 | AIV core | `static_cast<int32_t>(elewiseTilingData.coreNum)` | 当前 scheduler 不读取 | 源值来自平台 AIV 核数；显式窄化 |
| 同上 | `ubFormer` | `int32_t`/32 | offset 12 | 元素/tile | `= elewiseTilingData.ubFormer` | `Init` 分配和 `Process` full tile | R-001 64、R-002 128 元素对齐；必须非零 |
| 同上 | `blockFormer` | `int64_t`/64 | offset 16 | 元素/core | `= elewiseTilingData.blockFormer` | `CalcBlockOffset` 计算 GM base | 512 元素对齐；末核可少于此值 |
| 同上 | `blockNum` | `int64_t`/64 | offset 24 | AIV block | `= elewiseTilingData.blockNum`; 同值传 `SetBlockDim` | `Process` 判断末核 | 非空时至少 1 |
| 同上 | `ubLoopOfFormerBlock` | `int64_t`/64 | offset 32 | tile loop | `ceil(blockFormer/ubFormer)` | 非末核 `loopNum` | 至少 1 |
| 同上 | `ubLoopOfTailBlock` | `int64_t`/64 | offset 40 | tile loop | `ceil(blockTail/ubFormer)` | 末核 `loopNum` | 至少 1 |
| 同上 | `ubTailOfFormerBlock` | `int64_t`/64 | offset 48 | 元素 | `blockFormer-(loopFormer-1)*ubFormer` | 非末核最后一次 `Run` | `1..ubFormer` |
| 同上 | `ubTailOfTailBlock` | `int64_t`/64 | offset 56 | 元素 | `blockTail-(loopTail-1)*ubFormer` | 末核最后一次 `Run` | `1..ubFormer` |
| 同上 | `elemNum` | `int64_t`/64 | offset 64 | 元素 | `= maxElemNum` | 当前 scheduler 不读取 | inactive 字段；与 `ubFormer` 同源 |
| 同上 | `scheMode` | `uint64_t`/64 | offset 72 | 模板枚举 | `GetScheMode()=1` | 由 Host 编入 TilingKey；模板实例选择 | 声明也允许 0，但 Host 当前不可达 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 直接 MicroAPI | `(1,TPL_FP32)` | `aclnnFastGeluGetWorkspaceSize`; `l0op::FastGelu`; `FastGeluTiling::RunTiling`; `ElewiseTiling` | `EleBaseTilingDataV2` | `fast_gelu<1,TPL_FP32>` | `ElementwiseSch`; `FastGeluNoCast<float>`; `FastGeluCustom<float,true>`; MicroAPI mask/div | `fast_gelu_def.cpp`; `fast_gelu_binary.json`; `activation/fast_gelu/CMakeLists.txt` |
| R-002 FP16/BF16 升精度 | `(1,TPL_FP16/BF16)` | 同上，Tiling 分别实例化 `FastGeluNeedCast<half/bfloat16_t>` | 同一 POD；dtype 改变 DAG 的 buffer 与对齐常量 | `fast_gelu<1,TPL_FP16/BF16>` | `ElementwiseSch`; `Vec::Cast`; `FastGeluCustom<float,false>` | 同上，F16/BF16 各有 binary 条目 |

## Kernel 实现

### 公共执行流程

两个 Route 共用单 AIV Kernel。entry 拒绝空 workspace，登记 system workspace，解析 `EleBaseTilingDataV2`，声明 `KERNEL_TYPE_AIV_ONLY`，创建一个 `TPipe`，再按编译期 `dType` 构造 `ElementwiseSch`。scheduler 将每核 GM base 定位到 `blockFormer*blockIdx*sizeof(T)`，以 `ubFormer` 遍历完整 tile，最后执行一次 tail tile；迭代的 `i&1` 选择 ping/pong 子区。

每个 tile 的公共搬运为 `DataCopyPad`：CopyIn 和 CopyOut 都设置 `blockCount=1`、`blockLen=count*sizeof(T)` 字节、源/目的 stride 为 0；CopyIn `padParams.isPad=false`。DAG 节点以 `Mutex::Lock/Unlock<PIPE_MTE2/PIPE_V/PIPE_MTE3>` 在同一物理子区的 producer/consumer 间交接，最终仅把 `count` 个有效元素写回 GM。

### Route R-001：FP32 直接 MicroAPI

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/fast_gelu/op_api/aclnn_fast_gelu.cpp` | `CheckDtypeValid` | 950 白名单且 `self/out` 同为 FLOAT | 添加 L0 FastGelu |
| Tiling | `activation/fast_gelu/op_host/arch35/fast_gelu_tiling_arch35.cpp` | `FastGeluTiling::RunTiling` | `outputDtype == ge::DT_FLOAT` | `dType=TPL_FP32`; `DoTiling<FastGeluNoCast<float>::OpDag>` |
| Key | 同上 | `GET_TPL_TILING_KEY` | `tiling->scheMode==1,dType==TPL_FP32` | `(1,TPL_FP32)` |
| Router | `activation/fast_gelu/op_kernel/fast_gelu_apt.cpp` | `fast_gelu<schMode,dType>` | `if constexpr(dType==TPL_FP32)` | `ElementwiseSch<1,FastGeluNoCast<float>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::FastGelu -> ADD_TO_LAUNCHER_LIST_AICORE(FastGelu)
  -> fast_gelu<1,TPL_FP32>(x,y,workspace,tiling)
    -> ElementwiseSch::Init(x,y)
    -> ElementwiseSch::Process
      -> Run(tileOffset,tileLength,pingPong)
        -> CopyIn<float>
        -> FastGeluCustom<float,true>
        -> CopyOut<float>
```

##### 源码等价伪码

```text
# 伪码；省略模板元编程
coreBase = blockFormer * blockIdx
loopNum, tail = former_or_tail_core_fields(blockIdx)
for tile in [0, loopNum):
    count = (tile == loopNum-1) ? tail : ubFormer
    pp = tile & 1
    ubIn[pp][0:count] = GM_x[coreBase + tile*ubFormer : ...]
    remain = uint32(count)
    for regLoop in [0, ceil(count / (VECTOR_REG_WIDTH/4))):
        mask = UpdateMask<float>(remain)       # post-update remain
        xReg   = load(ubIn[pp] + regLoop*VL, mask)
        denReg = xReg * (-1.702f)
        denReg = exp(denReg) + 1.0f
        outReg = div<ZEROING,precision=true>(xReg, denReg, mask)
        store(ubOut[pp] + regLoop*VL, outReg, mask)
    GM_y[coreBase + tile*ubFormer : ...] = ubOut[pp][0:count]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `(blockFormer*blockIdx + tile*ubFormer)*4` bytes | `count*4` bytes/tile | FP32/ND flatten | caller | `CopyIn<float>` | caller；执行期间只读 | exact tail byte length |
| UB MTE2 ping/pong | `bufId*blockLen` | 每区 `blockLen=ubFormer*4` bytes，共 2 区 | FP32 flat | `DataCopyPad` | `FastGeluCustom` | scheduler TBuf；tile 奇偶交替 | 256-byte tile 对齐；MTE2↔V 锁 |
| RegTensor `x/denominator/result` | 寄存器，无 UB offset | 每个为一个 FP32 Vector register | FP32 | load/逐步 MicroAPI | 下一计算/store | 单个 regLoop | 中间结果不物化为额外 UB tensor |
| UB MTE3 ping/pong | `bufId*blockLen` | 每区 `ubFormer*4` bytes，共 2 区 | FP32 flat | `FastGeluCustom` | `CopyOut<float>` | scheduler TBuf；写回后可复用 | 计算节点直连 CopyOut；V↔MTE3 锁 |
| GM `y` | 与 x 相同元素 offset | `count*4` bytes/tile | FP32/ND flatten | `CopyOut<float>` | ViewCopy/用户 | executor 临时 Tensor | 只写有效 count |
| workspace | 核心节点 slice 起点 | 16 MiB system + 0 user | bytes | Host Tiling | `SetSysWorkspace`; body 不读 user 区 | executor/用户提供至 kernel 完成 | 950 reserved workspace 边界 |

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
blockLen = ubFormer * MaxDtypeBytes = ubFormer * 4
UB_total = 4 * blockLen = 16 * ubFormer bytes
UB_total <= U，且 ubFormer 向下对齐到 64 元素
dynamic_ub = N/A（AIV TBuf 静态按 Tiling 分配）
workspace_total(core node) = 16,777,216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| MTE2 ping/pong | `CopyIn` | `FastGeluCustom` | 同奇偶的下下个 tile | `Unlock<PIPE_MTE2>` 后 `Lock<PIPE_V>`；compute 消费后释放 V | 不在 V 消费完成前由 MTE2 覆盖 |
| MTE3 ping/pong | `FastGeluCustom` | `CopyOut` | 同奇偶的下下个 tile | compute 释放 V，MTE3 取得锁并在搬出后释放 | 不在 MTE3 搬出完成前覆盖 |
| `denominator` register | `Muls` | `Exp`，再由 `Adds` 原地更新，最后供 `Div` | 下一个 regLoop | 同一 `__VEC_SCOPE__` 顺序 | mask 与当前 remaining count 一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + `PIPE_MTE2` mutex | GM x | Vector compute | MTE2 子区 | blockLen 为 `count*4`；无 padding |
| MicroAPI load/compute | `UpdateMask<float>`、ZEROING | UB input | `Muls->Exp->Adds->Div` | 当前 FP32 register lanes | `VL=VECTOR_REG_WIDTH/4`；mask post-update 覆盖尾轮 |
| division | `Div<float,&mode>`，`mode={ZEROING,true}` | x、denominator | result | FP32 result register | 950 高精度除法实现包含误差补偿；特殊值选择保留首轮 div 结果 |
| UB→GM | `PIPE_MTE3` mutex + `DataCopyPad` | UB output | GM y | MTE3 子区 | 只搬 `count*4` 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/fast_gelu/op_host/arch35/fast_gelu_tiling_arch35.cpp` | `FastGeluTiling::RunTiling`; `ElewiseBaseTiling::DoTiling<FastGeluNoCast<float>::OpDag>` | 选择 FP32 key、blockDim、POD 和 16 MiB workspace |
| Kernel entry/router | `activation/fast_gelu/op_kernel/fast_gelu_apt.cpp` | `fast_gelu<1,TPL_FP32>` | 解析 POD，声明 AIV-only，构造直接 DAG scheduler |
| Device body/helper | `activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h` | `FastGeluCustom<float,true>` | 寄存器化 `Muls/Exp/Adds/Div`，高精度除法与 mask tail |
| Scheduler/搬运 | `pkg_inc/op_common/atvoss/elewise/elewise_sch.h`; `util/vec.h`; `util/sync.h` | `ElementwiseSch`; `CopyIn/CopyOut`; `GetTensor/ReleaseTensor` | 分核/tile 循环、4 子区、exact byte DMA 和 pipe mutex |

### Route R-002：FP16/BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/fast_gelu/op_api/aclnn_fast_gelu.cpp` | `CheckDtypeValid` | 950 白名单且两侧同为 FLOAT16 或 BFLOAT16 | 添加 L0 FastGelu |
| Tiling | `activation/fast_gelu/op_host/arch35/fast_gelu_tiling_arch35.cpp` | `FastGeluTiling::RunTiling` | `DT_FLOAT16` 优先于 `DT_BF16` 分支 | 分别 `TPL_FP16`/`TPL_BF16` 与 `FastGeluNeedCast<T>` |
| Key | 同上 | `GET_TPL_TILING_KEY` | `scheMode=1,dType in {TPL_FP16,TPL_BF16}` | 两个 dtype key 合并到相同行为 Route |
| Router | `activation/fast_gelu/op_kernel/fast_gelu_apt.cpp` | `fast_gelu<schMode,dType>` | 前两个 `if constexpr` 分支 | `ElementwiseSch<1,FastGeluNeedCast<half/bfloat16_t>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::FastGelu -> ADD_TO_LAUNCHER_LIST_AICORE(FastGelu)
  -> fast_gelu<1,TPL_FP16/BF16>(x,y,workspace,tiling)
    -> ElementwiseSch::Init(x,y)
    -> ElementwiseSch::Process
      -> Run(tileOffset,tileLength,pingPong)
        -> CopyIn<T>
        -> Cast<float,T,0>
        -> FastGeluCustom<float,false>
        -> Cast<T,float,1>
        -> CopyOut<T>
```

##### 源码等价伪码

```text
# 伪码；T 为 half 或 bfloat16_t
coreBase = blockFormer * blockIdx
for each tile(count, pp):
    ubInT[pp][0:count] = GM_x_T[coreBase + tile*ubFormer : ...]
    ubFp32A[0:count] = Cast<float,T,mode=0>(ubInT[pp], count)
    remain = uint32(count)
    for regLoop in [0, ceil(count / (VECTOR_REG_WIDTH/4))):
        mask = UpdateMask<float>(remain)
        xReg   = load(ubFp32A + regLoop*VL, mask)
        denReg = exp(xReg * -1.702f) + 1.0f
        outReg = div<ZEROING,precision=false>(xReg, denReg, mask)
        store(ubFp32B + regLoop*VL, outReg, mask)
    ubOutT[pp][0:count] = Cast<T,float,mode=1>(ubFp32B, count)
    GM_y_T[coreBase + tile*ubFormer : ...] = ubOutT[pp][0:count]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `(blockFormer*blockIdx + tile*ubFormer)*2` bytes | `count*2` bytes | F16/BF16 flat | caller | `CopyIn<T>` | caller；执行期间只读 | exact tail bytes |
| UB MTE2 ping/pong | `bufId*blockLen` | 每区 `ubFormer*4` bytes，共 2 区；逻辑只用前 `count*2` | T flat | `DataCopyPad` | `Cast<float,T,0>` | scheduler TBuf | 物理按最大 dtype 4 B/元素分区 |
| UB FP32 temp A | `bufId*blockLen` | `ubFormer*4` bytes | FP32 flat | CastIn | `FastGeluCustom` | 当前 tile | CastIn 消费 T 输入后独立存活 |
| RegTensor `x/denominator/result` | 寄存器 | 三个 FP32 Vector register | FP32 | load/MicroAPI | store | 单 regLoop | denominator 原地更新；无额外 UB |
| UB FP32 temp B | `bufId*blockLen` | `ubFormer*4` bytes | FP32 flat | `FastGeluCustom` | CastOut | 当前 tile | 与 temp A 生命周期重叠，不能别名 |
| UB MTE3 ping/pong | `bufId*blockLen` | 每区 `ubFormer*4` bytes，共 2 区；逻辑只写 `count*2` | T flat | CastOut | `CopyOut<T>` | scheduler TBuf | MTE3 ping/pong |
| GM `y` | 与 x 相同元素 offset | `count*2` bytes | F16/BF16 flat | CopyOut | ViewCopy/用户 | executor 临时 Tensor | 只写有效 count |
| workspace | 核心 slice 起点 | 16 MiB system + 0 user | bytes | Host Tiling | entry | 至 kernel 完成 | body 不消费 user 区 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inPing["MTE2 ping · ubFormer×4 B"]:1
  inPong["MTE2 pong · ubFormer×4 B"]:1
  fpA["FP32 temp A · ubFormer×4 B"]:1
  fpB["FP32 temp B · ubFormer×4 B"]:1
  outPing["MTE3 ping · ubFormer×4 B"]:1
  outPong["MTE3 pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
blockLen = ubFormer * MaxDtypeBytes = ubFormer * 4
UB_total = 6 * blockLen = 24 * ubFormer bytes
UB_total <= U，且 ubFormer 向下对齐到 128 元素
dynamic_ub = N/A（AIV TBuf 静态按 Tiling 分配）
workspace_total(core node) = 16,777,216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| MTE2 ping/pong | CopyIn | CastIn | 同奇偶的下下个 tile | MTE2/V mutex | CastIn 未消费前不得覆盖 |
| FP32 temp A | CastIn | FastGeluCustom | 下一 tile 的 CastIn | compute 节点输入释放 | FastGeluCustom 完成读后才覆盖 |
| FP32 temp B | FastGeluCustom | CastOut | 下一 tile 的 Custom | compute 节点输入释放 | CastOut 完成读后才覆盖 |
| MTE3 ping/pong | CastOut | CopyOut | 同奇偶的下下个 tile | V/MTE3 mutex | DMA 完成前不得覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + MTE2 mutex | GM T | CastIn | MTE2 子区 | 搬 `count*2` 字节，无 padding |
| 升精度 | `Vec::Cast<float,T,0>` | UB T | FP32 temp A | temp A | `count` 贯穿；源码把整数 0 转为 `RoundMode` |
| 核心计算 | MicroAPI mask + `DivSpecificMode{ZEROING,false}` | FP32 temp A | FP32 temp B | FP32 registers/temp | Muls/Exp/Adds/Div 全为 FP32；tail mask 逐 VL 更新 |
| 降精度 | `Vec::Cast<T,float,1>` | FP32 temp B | UB T output | MTE3 子区 | 源码把整数 1 转为 `RoundMode`；固定仓未在算子文件解释该数值枚举名 |
| UB→GM | MTE3 mutex + `DataCopyPad` | UB T output | GM y | MTE3 子区 | 搬 `count*2` 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/fast_gelu/op_host/arch35/fast_gelu_tiling_arch35.cpp` | `FastGeluTiling::RunTiling`; `DoTiling<FastGeluNeedCast<T>::OpDag>` | 分别选择 F16/BF16 key，产生 6-buffer tile 上界 |
| Kernel entry/router | `activation/fast_gelu/op_kernel/fast_gelu_apt.cpp` | `fast_gelu<1,TPL_FP16/BF16>` | dtype 映射到 half/bfloat16_t DAG |
| Device body/helper | `activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h` | `FastGeluNeedCast<T>`; `FastGeluCustom<float,false>` | Cast→FP32 寄存器公式→Cast，普通除法模式 |
| Scheduler/搬运 | `pkg_inc/op_common/atvoss/elewise/elewise_sch.h`; `util/vec.h`; `util/sync.h` | `ElementwiseSch`; `Vec::Cast`; `CopyIn/CopyOut` | 6 子区、生命周期、DMA 与同步 |

## 算子特定优化

### Trick T-001：寄存器内融合 logistic 近似并只做一次 UB 读写

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：进入 `FastGeluCustom<float,highPrecisionDiv>` 的任一有效 tile/regLoop。
- Golden 来源：[`activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h:44-53`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
    mask = AscendC::MicroAPI::UpdateMask<T, AscendC::MicroAPI::RegTraitNumOne>(count);
    AscendC::MicroAPI::DataCopy(x, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
    AscendC::MicroAPI::Muls(denominator, x, value1, mask);
    AscendC::MicroAPI::Exp(denominator, denominator, mask);
    AscendC::MicroAPI::Adds(denominator, denominator, value2, mask);
    AscendC::MicroAPI::Div<T, &mode>(result, x, denominator, mask);
    AscendC::MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), result, mask);
}
```

#### 机制与设计原因

一个 regLoop 只从 UB 装入一次 `x`、向 UB 写回一次结果；`-1.702*x -> exp -> +1 -> x/denominator` 的中间量全部保留在 `RegTensor`，`denominator` 被原地覆盖。这样消除了公式各阶段的 UB 中间 Tensor 和相应搬运。`vlSize=VECTOR_REG_WIDTH/sizeof(float)` 决定相邻寄存器块地址，`UpdateMask(count)` 同时 post-update 剩余元素，使最后一轮只处理真实 tail。

#### 迁移边界

- 适用：逐元素、输入输出一一对应且能在寄存器串联完成的激活函数。
- 前提：950 MicroAPI 支持所需 `Muls/Exp/Adds/Div`；中间 dtype 为 FP32；每轮 mask 必须一致贯穿全部 primitive。
- 失效：需要跨元素归约、跨 tile 状态或中间值被多个后续分支长期消费时，必须重新做 UB/同步账，不能直接套用该融合。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从源码可确认不得遗漏 `UpdateMask` 或用对齐后的 `ubFormer` 代替当前 `count`，否则尾轮可能访问无效元素。

### Trick T-002：按输出精度分配除法精度预算

#### 触发与源码依据

- 适用 Route：R-001 使用高精度除法；R-002 使用普通除法但整个公式仍在 FP32 中执行。
- 触发条件：Tiling/entry 的 dtype 分支分别实例化 `FastGeluCustom<float,true>` 或 `FastGeluCustom<float,false>`。
- Golden 来源：[`activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h:60-83`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
struct FastGeluNoCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpFastGelu = Bind<FastGeluCustom<float, true>, OpCopyIn>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpFastGelu>;
    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};

template <typename T>
struct FastGeluNeedCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastIn = Bind<Vec::Cast<float, T, 0>, OpCopyIn>;
    using OpFastGelu = Bind<FastGeluCustom<float, false>, CastIn>;
    using CastOut = Bind<Vec::Cast<T, float, 1>, OpFastGelu>;
```

#### 机制与设计原因

FP32 输出没有末端量化可掩盖误差，因此模板将 `precisionMode=true` 传给 950 `Div`；该模式的固定实现用首轮除法、FMA 残差和二次除法做误差补偿，并对 Inf/NaN/0 选择首轮结果。FP16/BF16 路线先提升到 FP32，避免在低精度下计算 `exp/div`，但选择 `precisionMode=false`，最后按模板 mode 1 降精度。精度策略在 Host dtype 路由时完成模板特化，Device 内没有运行时 dtype 分支。

#### 迁移边界

- 适用：输出 dtype 已知、可以按最终精度对昂贵 primitive 做编译期特化的逐元素算子。
- 前提：高精度路线必须有对应平台实现和额外寄存器/指令预算；低精度路线仍需先证明升降精度顺序。
- 失效：若 FP16/BF16 结果也要求与 FP32 高精度除法完全一致，必须改用 `highPrecisionDiv=true` 并重新验证；当前固定源码没有该 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得仅因 R-002 的主体使用 FP32 就声称它与 R-001 数值路径相同，因为 `DivSpecificMode.precisionMode` 明确不同，且 R-002 还有最终降精度。

## 模板与实现对比

| 对比维度 | R-001 FP32 直接 MicroAPI | R-002 FP16/BF16 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | FLOAT | FLOAT16 或 BFLOAT16 | Host 按 output dtype 顺序选择，API 已保证输入输出同 dtype |
| 模板与 dtype | `FastGeluNoCast<float>` | `FastGeluNeedCast<half/bfloat16_t>` | 是否需要 FP32 Cast |
| 循环与分核 | `b_min=32`，tile 64 元素对齐 | `b_min=16`，tile 128 元素对齐 | 通用 Tiling 以最小 dtype bit 数计算核数与 repeat 对齐 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4 个 TBuf 子区 + 3 个 FP32 RegTensor；无 dynamic UB/通信 | 6 个 TBuf 子区 + 3 个 FP32 RegTensor；无 dynamic UB/通信 | Cast 引入两个同时存活的 FP32 temp |
| 内存总占用与复用 | `16*ubFormer` 字节；计算输出直连 MTE3 | `24*ubFormer` 字节；两个 FP32 temp 串联 | DAG 最大存活节点和 MTE2/MTE3 ping-pong |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM `count*4` 字节；MicroAPI 公式 | GM `count*2` 字节；Vector Cast + 同一 MicroAPI 公式 | 外层 dtype 不同；无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | MTE2/V/MTE3 mutex；FP32 高精度 Div；mask tail | 同步相同；FP32 普通 Div；降回 T；mask tail | 输出精度决定除法模式与 Cast |
| fallback/失败边界 | 非 FLOAT 去 R-002 或 API 失败 | 非 F16/BF16 去 R-001 或 API 失败 | 无 Device runtime fallback；unknown key/body 不产出 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档称支持；固定提交内 `source_closed` 缺失 | 公开 ACLNN 文档称支持；固定提交内 `source_closed` 缺失 | 公开支持且源码闭合 |
| Host/Launcher | 当前活动 OpDef/CMake/binary 未接入 | 当前活动 OpDef/CMake/binary 未接入 | `ascend950` + `arch35`，两段式 ACLNN 到单 AIV kernel |
| Kernel/资源 | 固定提交不可见，未推测 | 固定提交不可见，未推测 | atvoss ElementwiseSch；4/6 子区；16 MiB system workspace |

## 限制与失败边界

- API 要求非空指针、输入输出同 dtype/shape/storage format、dtype 属于 FLOAT/FLOAT16/BFLOAT16，且输入不超过 8 维；空 Tensor 成功且 workspace 为 0。
- Tiling 展平 shape 时以 `int64_t` 连乘，固定源码没有显式乘法溢出检查；`ubFormer` 写入 `int32_t`，tileLength 继续传为 `uint32_t count`，MicroAPI loop 数写为 `uint16_t`，没有独立窄化 fallback。
- `scheMode` 声明允许 0/1，但通用 `GetScheMode()` 固定返回 1；模式 0 的 binary/template 选择即使被声明，也不是当前 Host 可达 Route。
- entry 对 workspace 或 user-workspace 空指针直接返回，不上报 Device 错误；调用者必须使用第一段 workspace/executor。
- A2/A3 的公开支持表与固定提交的实际 Host/Tiling/Kernel 构建接线不闭合，因此未生成 `fast_gelu.md`；不得从 950 `arch35` 路径推测复用。
- `aclnnFastGeluV2` 仅存在于默认关闭的 experimental 树；正式 `activation/fast_gelu_v2` 排除 ACLNN，且其分段多项式语义不同，不纳入本卡。
- 本次知识提取未执行编译、真机、数值或性能测试；源码可证明机制，不代表已测性能增益。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/fast_gelu/docs/aclnnFastGelu.md` | 产品支持表、API 参数表 | A2/A3/950 公共支持声明、dtype/shape/format/空域约束 |
| 入口声明/约束 | `activation/fast_gelu/op_api/aclnn_fast_gelu.h` | `aclnnFastGeluGetWorkspaceSize/aclnnFastGelu` | 公开两段式签名 |
| 入口实现 | `activation/fast_gelu/op_api/aclnn_fast_gelu.cpp` | `GetDtypeSupportList`; `CheckParams`; 两段式函数 | 平台 dtype、前后处理、executor/workspace、失败路径 |
| 入口实现 | `activation/fast_gelu/op_api/fast_gelu.cpp` | `l0op::FastGelu` | 临时输出与 AICore launcher |
| Host/Tiling/Launcher | `activation/fast_gelu/op_host/fast_gelu_def.cpp` | `FastGelu::FastGelu` | 950 OpDef、dtype/format、`fast_gelu_apt` |
| Host/Tiling/Launcher | `activation/fast_gelu/op_host/arch35/fast_gelu_tiling_arch35.cpp` | `FastGeluTiling::RunTiling`; `TilingFuncFastGelu` | dtype→DAG/key、blockDim、16 MiB workspace |
| Host/Tiling/Launcher | `activation/fast_gelu/op_kernel/arch35/fast_gelu_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 与 dtype 模板域 |
| Kernel entry/body | `activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h` | `FastGeluCustom`; `FastGeluNoCast/NeedCast` | 公式、cast、除法模式、mask tail |
| Kernel entry/body | `activation/fast_gelu/op_kernel/fast_gelu_apt.cpp` | `fast_gelu` | AIV entry、POD、workspace、dtype router |
| 构建接线 | `activation/fast_gelu/op_host/config/ascend950/fast_gelu_binary.json` | `FastGelu_bf16/fp16/fp32` | 三个 950 binary 条目 |
| 构建接线 | `activation/fast_gelu/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`; `add_modules_sources` | 仅 ascend950/arch35 的活动实现 |
| 构建接线 | `CMakeLists.txt` | `ENABLE_EXPERIMENTAL` | experimental 默认关闭，排除 V2 ACLNN 候选 |
| 入口声明/约束 | `experimental/activation/fast_gelu_v2/docs/aclnnFastGeluV2.md` | `aclnnFastGeluV2GetWorkspaceSize/aclnnFastGeluV2` | V2 ACLNN 仅在 experimental 树公开，且数学语义为另一分段近似 |
| 构建接线 | `experimental/activation/fast_gelu_v2/CMakeLists.txt` | `add_modules_sources` | V2 ACLNN 候选属于 experimental 选择分支 |
| 构建接线 | `activation/fast_gelu_v2/CMakeLists.txt` | `ACLNNTYPE aclnn_exclude` | 正式 V2 目录不发布 ACLNN |
| Host/Tiling/Launcher | `pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData/V2` | 80-byte POD 字段 |
| Host/Tiling/Launcher | `pkg_inc/op_common/atvoss/elewise/elewise_tiling.h` | `ElewiseBaseTiling::DoTiling` | DAG buffer 元数据到 Tiling 参数 |
| Host/Tiling/Launcher | `src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling`; `AdaptEleBaseTilingData` | 分核/tile/loop/tail 公式与字段 producer |
| Kernel entry/body | `pkg_inc/op_common/atvoss/elewise/elewise_sch.h` | `ElementwiseSch` | TBuf、分核循环、DAG 遍历、生命周期 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch`; `GetBufferIds` | buffer 数、最大 dtype、buffer 映射 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo` | 最大存活节点与 MTE/temp buffer 计数 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/vec.h` | `Vec::Cast/CopyIn/CopyOut` | cast 模式和 exact byte DataCopyPad |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 mutex 协议 |
| Kernel entry/body | `impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_maskreg_impl.h` | `UpdateMaskImpl` | mask post-update 与 tail |
| Kernel entry/body | `impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | `DivPrecisionImpl` | 950 高精度除法误差补偿 |
| Kernel entry/body | `impl/basic_api/dav_3510/kernel_operator_common_impl.h` | `SetSysWorkspace/GetUserWorkspace` | 950 system/user workspace 分界 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 可复用 `EleBaseTilingDataV2` 的 80-byte elementwise ABI | 字段顺序、位宽、单位、`scheMode=1` 和 inactive 字段均不得私改 | `EleBaseTilingData`; `AdaptEleBaseTilingData` |
| 调用顺序 | `Contiguous -> FastGelu -> ViewCopy`；kernel 内 `CopyIn -> DAG compute -> CopyOut` | 空域短路、失败即停止追加，以及 Cast 前后次序必须保留 | `aclnnFastGeluGetWorkspaceSize`; `ElementwiseSch::Run` |
| workspace/内存 | 通用分核/tile 公式、MTE ping/pong 与 temp buffer 计数可复用 | R-001 的 4 子区、R-002 的 6 子区、每区按最大 dtype 4 B/元素、16 MiB system workspace不可漏算 | `ElewiseTiling`; `ElementwiseSch::Init`; `FastGeluTiling::RunTiling` |
| 同步 | MTE2/V/MTE3 mutex 交接可复用 | 同一 bufId 的 lock/unlock 顺序与 input 释放点不得提前 | `GetTensor/ReleaseTensor`; `PrepareArgs/PostArgs` |
| 精度与尾块 | FP32 中的 fused formula、`UpdateMask` tail 可复用 | FP32 `precisionMode=true`；F16/BF16 升精度、`precisionMode=false`、降精度；每轮必须用剩余 `count` | `FastGeluCustom`; `FastGeluNoCast/NeedCast` |
| 失败边界 | API/Host 失败终点和 entry 空 workspace 终点可复用 | 不得为未支持 dtype、A2/A3 或 experimental V2 编造 fallback | `CheckParams`; `FastGeluTiling::RunTiling`; `fast_gelu` |

- [GET_TILING_DATA_WITH_STRUCT](../../../../apis/simd_api/basic_api/kernel_tiling/get_tiling_data_with_struct.md)
- [REGISTER_TILING_DEFAULT](../../../../apis/simd_api/basic_api/kernel_tiling/register_tiling_default.md)
- [RoundMode](../../../../apis/simd_api/basic_api/reg_vector_compute/aux_data_types/round_mode.md)
- [SetSysWorkSpace](../../../../apis/simd_api/basic_api/resource_management/workspace/set_sys_work_space.md)
