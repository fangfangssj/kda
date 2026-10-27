---
type: Operator
title: "InplaceIndexFill 950 ACLNN 算子知识"
description: "说明 aclnnInplaceIndexFill 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/index_fill_d/docs/aclnnIndexFill&aclnnInplaceIndexFill.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/index_fill_d/op_api/aclnn_index_fill.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/index_fill_d/op_api/index_fill.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_simd.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_simt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_simt_dense_indices.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/inplace_index_fill_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simt_dense_indices.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_host/inplace_index_fill_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_host/config/ascend950/inplace_index_fill_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:55:07Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceIndexFill 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceIndexFillGetWorkspaceSize` + `aclnnInplaceIndexFill`，公开 ACLNN 两段式接口 |
| 计算语义 | 把 `selfRef` 视为 `[P,N,Q]`；对每个合法 `indices[k]`（负值加 `N`）及所有 `p,q`，原地写 `selfRef[p,indices[k],q]=value`，越界索引跳过 |
| 输入输出 | `selfRef/value`: FP16/FP32/BF16/FP64/INT8/UINT8/INT16/INT32/INT64/BOOL；`index`: INT32/INT64；ND，`selfRef` 原地输出 |
| 执行类型 | 单 Kernel；R-001、R-002 为 SIMT，R-003 为 AIV_VECTOR Sort/MicroAPI + SIMT；入口固定 `KERNEL_TYPE_AIV_ONLY`，无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 SIMD-name mask 路线；R-002 通用 SIMT 稀疏/稠密双路径；R-003 DenseIndices Sort+Unique 路线 |
| 核心资源 | GM 原地 `x`、`indices/value`；workspace 前 `N` 字节 mask；R-003 另有双缓冲 `TQue`、排序与 unique `TBuf` |
| 来源闭包 | source_closed：公开 ACLNN→L0 `InplaceIndexFill`→arch35 优先级模板→POD→APT router→三个实际 body→ascend950 构建接线均闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令归一化轴 `d = dim >= 0 ? dim : dim + rank(selfRef)`，并折叠 `P=prod(shape[0:d])`、`N=shape[d]`、`Q=prod(shape[d+1:])`。每个索引 `j` 先执行 `j = j < 0 ? j + N : j`；仅当 `0 <= j < N` 时，把所有 `p∈[0,P)`、`q∈[0,Q)` 的 `selfRef[(p*N+j)*Q+q]` 写成由 `value` 转成 `selfRef` dtype 的标量。重复索引写入相同值，因此结果确定；越界索引不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，0–8 维，允许 ACLNN 非连续 view | FP16/FP32/BF16/FP64/INT8/UINT8/INT16/INT32/INT64/BOOL | 输入与原地输出；API 先转连续临时量，最终 `ViewCopy` 回原 view |
| `dim` | 标量属性 | int64 | 非空 rank 下须在 `[-rank,rank)`；负轴加 rank |
| `index` | API 接受 tensor；Kernel 按一维元素序列读取 | INT32/INT64 | 负索引加 `N`；仍越界则跳过 |
| `value` | `aclScalar`，转换成单元素 tensor | 可转换到 `selfRef` dtype | Kernel 读取元素 0 |
| `workspaceSize/executor` | Host 输出 | — | executor 持有 `Contiguous`、核心节点和 `ViewCopy`；workspace 由 executor 汇总 |

### 入口与版本

固定仓公开文档同时列出 A2、A3、Ascend 950；950 API 代码在 `IsRegbase()` 为真且 `isInplace` 时选择 `l0op::InplaceIndexFill`，没有版本后缀，是该平台可达的最新公开入口。本文不覆盖 `aclnnIndexFill`、`aclnnInplaceIndexFillTensor` 或非原地逻辑算子。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecIndexFillGetWorkspaceSize` 顺序执行：公共输出指针检查；`self/index/value/out` 非空；950 `self` dtype 在 10 类型集合、`index` 为 INT32/INT64、原地 `out=selfRef` 的 shape/dtype 一致、rank/dim 合法。`selfRef` 为空时返回 workspace 0；`index->Size()==0` 时只登记 `ViewCopy(self,out)`。其余场景对 `self/index` 做 `Contiguous`，0 维临时 `UnsqueezeNd(...,0)`，把 scalar `value` 转为 `self` dtype。950 的 `NeedTranspose` 恒假；随后登记 `l0op::InplaceIndexFill`，最后以 `ViewCopy(indexFillOut,selfRef)`提交到原 view。

### 参数、Workspace 与生命周期

L0 以 `OP_INPUT(self,indices,value)`、`OP_OUTPUT(self)`、`OP_ATTR(dim)` 登记核心节点。Tiling 从平台取得 AIV 核数和 UB；三模板 workspace 均为 `libApiWorkspace + N*sizeof(int8_t)`，前 `N` 字节由 Kernel 作为 mask，且 `SetScheduleMode(1)` 支持 `SyncAll`。executor 拥有连续临时量与 workspace 需求；调用第二段 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)` 后按同一 stream 顺序执行核心节点和 `ViewCopy`。

### 启动图与失败传播

```text
aclnnInplaceIndexFillGetWorkspaceSize
  -> Contiguous/Unsqueeze/ConvertToTensor（按需，框架边界）
  -> l0op::InplaceIndexFill / ADD_TO_LAUNCHER_LIST_AICORE
     -> [K1] inplace_index_fill(x, indices, value, y_alias_x, workspace, tiling)
  -> ViewCopy(K1.x, selfRef)（原 view 非连续时负责提交）
aclnnInplaceIndexFill
  -> CommonOpExecutorRun(..., stream)
```

K1 是唯一算子自有 Device 节点，`blockDim=usedCoreNum`，POD 依 Route 不同；`y` 参数未被 body 读取，实际直接写 `x`。workspace mask 的 owner 是调用者/executor，K1 内从清零到最后一次 `MaskBasedFill` 有效；`SyncAll` 保证清零→置位→消费。任一登记/转换返回空指针即第一段失败，不继续登记后继；第二段由 `CommonOpExecutorRun` 返回执行状态。

## Route 与参数设计

### 分核与资源计算

共同输入推导为 `P,N,Q,indicesNum,numel=P*N*Q`。SIMT 核数从 2048 threads/core 开始计算 `min(coreNum,numel/threadNum)`；若不足平台核数一半则把线程数依次减半至 128，非空至少 1 核。R-001 另按 `P*indicesNum` 均分、在核利用率低于一半且每份 Q 仍至少 `2048/xDtypeSize` 元素时增加 `qUsedCoreNum`，但当前 Kernel body 只消费其中 `P/N/Q/indicesNum` 与 `usedCoreNum`。R-003 的 `indicesUbFactor` 从 `ubSize-128KiB` 对齐后估算，并在并行度不足时收缩到至少 `max(N+256,512)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 SIMD-name mask | priority 1；template 0；dtype mode 4/8 | `Q*xDtypeSize > 128` | R-003 |
| 2 | R-003 DenseIndices | priority 6；template 2；dtype mode 4/8 | `numel!=0 && Q*xDtypeSize<=128 && P*Q*indicesNum>2048 && indicesNum>=5*N` | R-002 |
| 3 | R-002 通用 SIMT | priority 10；template 1；dtype mode 4/8 | `IsCapable()==true` 兜底；body 内再按 `indicesNum*5<=N` 分稀疏/稠密 | 无；Tiling/校验失败 |

```text
if Q * sizeof(x) > 128: R-001
else if numel != 0 and P*Q*indicesNum > 2048 and indicesNum >= 5*N: R-003
else: R-002
dtype_mode = sizeof(x) <= 4 ? 4 : 8
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `(TEMPLATE_MODE=0,DTYPE_MODE=4/8)` | 950 且 `Q*sizeof(x)>128` | 单 K1 | `InplaceIndexFillSimd<X,DTYPE_INDICES,uint32/64_t>` | `N` 字节 GM mask；两次 `SyncAll`；SIMT 2048 threads | indices 与全量元素均 stride-loop；越界跳过 |
| R-002 | `(1,4/8)` | 前两模板不命中 | 单 K1 | `InplaceIndexFillSimtImpl<X,I,uint32/64_t>` | 稀疏子路不使用 mask；稠密子路用 `N` 字节 mask 和两次同步 | `indicesNum*5<=N` 随机写，否则 mask 顺序写 |
| R-003 | `(2,4/8)` | Dense 三谓词全真 | 单 K1 | `InplaceIndexFillDenseIndicesImpl<X,I,uint32/64_t>` | `N` 字节 mask；indicesQueue depth 2 + sort/unique TBuf；三处同步 | indices 分块尾块；无任务核退出构 mask 后仍参加同步 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Q 字节阈值 | `Q*xDtypeSize>128` | 选择优先级 1 路线；当前 body 实际仍是 mask+SIMT | R-001 | R-003/R-002 | `InplaceIndexFillTilingSimd::IsCapable` |
| 极密阈值 | `indicesNum>=5*N` 且总任务 `>2048` | Sort+Unique 成本只在重复多、任务大时摊销 | R-003 | R-002 | `InplaceIndexFillTilingSimtDenseIndices::IsCapable` |
| 运行时稀疏阈值 | `indicesNum*5<=N` | 遍历 `indicesNum*P*Q` 少于全量 mask 扫描 | R-002 | 同 Route 稠密子路 | `InplaceIndexFillSimtImpl::Process` |
| COM_T 位宽 | `sizeof(x)<=4` | 地址/元素计数用 uint32，否则 uint64 | 全部 | 对应 dtype mode | `GetTilingKey`、`ComputeTypeGet` |
| Dense 本地内存 | `localMemory=ubSize-128KiB` | 预留 DCache；Sort/Unique 需要显式 UB | R-003 | Tiling/launch 资源边界 | `DoIndicesTilingTask`、`PostTiling` |
| 空域 | API `selfRef->IsEmpty()` 提前返回 | 不启动核心 Kernel | 全部 | workspace 0 | `ExecIndexFillGetWorkspaceSize` |

### POD/ABI

所有结构仅含连续 `int64_t` 字段，固定源码所示 ABI 为 8 字节对齐。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `InplaceIndexFillSimtTilingData` | `tilingKeySimt` | int64/64 | sizeof=56, align=8, offset=0 | key | `SetTilingData=GetTilingKey()` | router 模板选择已在编译期；body不读 | 1/4 或 1/8 |
| 同上 | `p` | int64/64 | offset=8 | 元素轴乘积 | `preDimProduct` | `Process/CallMaskBasedFill` | 非负 |
| 同上 | `n` | int64/64 | offset=16 | 元素 | `dimSize` | 索引归一化、mask extent | 非负 |
| 同上 | `q` | int64/64 | offset=24 | 元素轴乘积 | `postDimProduct` | 地址反算 | 非负 |
| 同上 | `indicesNum` | int64/64 | offset=32 | 元素 | indices dim0 | 子路选择和循环 | 非负 |
| 同上 | `coreNum` | int64/64 | offset=40 | core | 平台 AIV 核数 | 当前 body 不读 | >0 |
| 同上 | `usedCoreNum` | int64/64 | offset=48 | core | `CalcSimtUsedCoreNum` | 间接成为 blockDim | >=1 |
| `InplaceIndexFillSimtDenseIndicesTilingData` | `tilingKeySimt` | int64/64 | sizeof=64, align=8, offset=0 | key | Dense `GetTilingKey` | router | 2/4 或 2/8 |
| 同上 | `p` | int64/64 | offset=8 | 元素轴乘积 | Host | final fill | 非负 |
| 同上 | `n` | int64/64 | offset=16 | 元素 | Host | mask/归一化 | >0（Dense gate） |
| 同上 | `q` | int64/64 | offset=24 | 元素轴乘积 | Host | final fill | 非负 |
| 同上 | `indicesNum` | int64/64 | offset=32 | 元素 | Host | 分块 | >0 |
| 同上 | `coreNum` | int64/64 | offset=40 | core | Host | 未直接读 | >0 |
| 同上 | `usedCoreNum` | int64/64 | offset=48 | core | Host | blockDim | >=1 |
| 同上 | `indicesUbFactor` | int64/64 | offset=56 | indices 元素 | `DoIndicesTilingTask` | `BuildIndicesMask` | 正数且按 dtype block 对齐 |
| `InplaceIndexFillSimdTilingData` | `tilingKey` | int64/64 | sizeof=136, align=8, offset=0 | key | SIMD `GetTilingKey` | router | 0/4 或 0/8 |
| 同上 | `preDimProduct` | int64/64 | offset=8 | 元素轴乘积 | Host | `Process` | 非负 |
| 同上 | `dimSize` | int64/64 | offset=16 | 元素 | Host | mask extent | 非负 |
| 同上 | `postDimProduct` | int64/64 | offset=24 | 元素轴乘积 | Host | final fill | 非负 |
| 同上 | `indicesNum` | int64/64 | offset=32 | 元素 | Host | mask fill | 非负 |
| 同上 | `perBlockData` | int64/64 | offset=40 | `(P*indicesNum)/core` | `BlockTiling` | 当前 body 不读 | inactive |
| 同上 | `tailBlockData` | int64/64 | offset=48 | 任务 | `BlockTiling` | 当前 body 不读 | inactive |
| 同上 | `tailBlockNum` | int64/64 | offset=56 | core | `BlockTiling` | 当前 body 不读 | inactive |
| 同上 | `qBlockFactor` | int64/64 | offset=64 | Q 元素 | `BlockTiling` | 当前 body 不读 | inactive |
| 同上 | `qUsedCoreNum` | int64/64 | offset=72 | core | `BlockTiling` | 当前 body 不读 | inactive |
| 同上 | `usedCoreNum` | int64/64 | offset=80 | core | `BlockTiling` | blockDim | >=1 |
| 同上 | `qBufferSize` | int64/64 | offset=88 | 字节 | `UBTiling` | 当前 body 不读 | inactive |
| 同上 | `indicesBufferSize` | int64/64 | offset=96 | 字节 | `UBTiling` | 当前 body 不读 | inactive |
| 同上 | `indicesUbFactor` | int64/64 | offset=104 | 元素 | `UBTiling` | 当前 body 不读 | inactive |
| 同上 | `qUbFactor` | int64/64 | offset=112 | 元素 | `UBTiling` | 当前 body 不读 | inactive |
| 同上 | `qLoopSize` | int64/64 | offset=120 | loop | `UBTiling` | 当前 body 不读 | inactive |
| 同上 | `qUbTailFactor` | int64/64 | offset=128 | 元素 | `UBTiling` | 当前 body 不读 | inactive；末字段 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | template 0, dtype 4/8 | `inplace_index_fill_tiling_simd.cpp` | `InplaceIndexFillSimdTilingData` | `inplace_index_fill_apt.cpp::DispatchSimd` | `inplace_index_fill_simd.h` + common | `inplace_index_fill/CMakeLists.txt` ascend950/arch35 |
| R-002 | template 1, dtype 4/8 | `inplace_index_fill_tiling_simt.cpp` | `InplaceIndexFillSimtTilingData` | `DispatchSimt` | `inplace_index_fill_simt.h` + common | 同上 |
| R-003 | template 2, dtype 4/8 | `inplace_index_fill_tiling_simt_dense_indices.cpp` | Dense POD | `DispatchDenseIndices` | dense header + common Sort/Unique | 同上 |

## Kernel 实现

### 公共执行流程

APT entry 从编译期 `TEMPLATE_MODE/DTYPE_MODE` 选择 POD 与 dispatch；`ComputeTypeGet` 对 8 字节 x 使用 `int64_t` 位模式，其余保留 `DTYPE_X`，COM_T 为 uint32/uint64。三路线都只写 `x`，`y` 是未消费的 ABI 参数。mask 路线遵循 `GM indices → workspace[N] mask → SIMT 遍历 x 原地写`；Dense 路线在 mask 前增加 `GM indices → VECIN TQue → Sort/MicroAPI Unique → mask`。

### Route R-001：SIMD-name mask 路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `inplace_index_fill_tiling_simd.cpp` | `IsCapable` | `Q*xDtypeSize>128`，priority 1 | template 0 |
| Router | `inplace_index_fill_apt.cpp` | `DispatchSimd` | template 0 + dtype 4/8 | `InplaceIndexFillSimd` |

#### Device 数据流

##### 函数级调用链

```text
inplace_index_fill -> DispatchSimd -> InplaceIndexFillSimd::Init/Process
  -> InitMaskZero -> SyncAll
  -> VF_CALL<InplaceIndexFillSimdMaskFill> -> SyncAll
  -> CallMaskBasedFill -> VF_CALL<MaskBasedFillSimt>
```

##### 源码等价伪码

```text
parallel_zero(mask[0:N]); SyncAll
for idx = globalThread; idx < indicesNum; idx += globalThreads:
  j = indices[idx]; if j<0: j+=N; if 0<=j<N: mask[j]=1
SyncAll
for i = globalThread; i < P*N*Q; i += globalThreads:
  pn=i/Q; p=pn/N; n=pn-p*N
  if mask[n]: x[i]=value
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `i` | `P*N*Q` | X/PNQ | caller | final SIMT | caller；原地 | 无中间输出 |
| workspace mask | 0 | `N` bytes | int8[N] | InitMaskZero/MaskFill | final SIMT | executor；K1 全程 | 每核清零片按 32B ceil |

##### 分层内存布局

N/A：该 Route 没有 AIV `InitBuffer`；实际替代路径是 GM workspace mask 与 SIMT 寄存器标量。

##### 字节账与复用顺序

```text
UB_total = 0（算子显式 AIV buffer）
dynamic_ub = N/A（VF_CALL 固定 2048 threads，未传 dynUBufSize）
workspace_total = libApiWorkspace + N bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mask[0:N] | clear + idempotent set-1 | final fill | K1 结束后 workspace 可复用 | 两次 `SyncAll` | schedule mode 1，所有核参与 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| clear→set | `SyncAll` | `InitOutput` | mask fill | mask | 清零片末尾按 32B ceil，workspace 用户区只声明 N 字节，固定仓未展示框架额外 padding |
| set→read | `SyncAll` | mask fill | final fill | mask | 重复索引同值写，无 atomic |
| fill | SIMT stride | all threads | K1 end | x | X 原 dtype 位值，不 cast；越界跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/inplace_index_fill_tiling_simd.cpp` | `BlockTiling/UBTiling/SetTilingData` | 生产 blockDim、P/N/Q 与遗留 UB 字段 |
| Kernel entry/router | `op_kernel/inplace_index_fill_apt.cpp` | `DispatchSimd` | template/dtype→COM_T |
| Device body/helper | `op_kernel/arch35/inplace_index_fill_simd.h` | `Process` | mask 构建与原地 fill |

### Route R-002：通用 SIMT 双路径

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `inplace_index_fill_tiling_simt.cpp` | `IsCapable` | 恒 true，priority 10 | template 1 |
| Device | `inplace_index_fill_simt.h` | `Process` | `indicesNum*5<=N` | 稀疏随机写，否则稠密 mask |

#### Device 数据流

##### 函数级调用链

```text
inplace_index_fill -> DispatchSimt -> Init -> Process
  -> sparse: VF_CALL<ComputeByIndices>
  -> dense: BuildIndicesMask -> CallMaskBasedFill
```

##### 源码等价伪码

```text
if 5*K <= N:
  for i over K*P*Q by global thread stride:
    slice=i/(P*Q); inner=i-slice*(P*Q); j=normalize(indices[slice])
    if valid: p=inner/Q; q=inner-p*Q; x[p*N*Q+j*Q+q]=value
else:
  clear mask; set valid indices; SyncAll; sequential-mask fill as R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x | sparse `(p*N+j)*Q+q`; dense `i` | P*N*Q | X | caller | SIMT | caller | direct in-place |
| indices | `slice`/`idx` | K | I | caller | SIMT | caller | stride-loop |
| mask | workspace+0 | N bytes | int8 | dense Build | dense fill | executor | sparse inactive |

##### 分层内存布局

N/A：无 AIV buffer；SIMT 以寄存器保存 magic/shift、value 和坐标，稠密子路使用 GM mask。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A
workspace_total=libApiWorkspace+N bytes（即使 sparse 子路不读 mask，Host 仍统一申请）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mask | dense Build | dense fill | K1 后 | SyncAll | sparse 时 inactive |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| sparse | 无跨核同步 | 每线程 | K1 end | 不同/重复 x 地址 | 重复写同值；负索引归一化，越界跳过 |
| dense | `SyncAll`×2 | clear/set | set/read | mask | 与 R-001 同 |
| 除法 | magic+shift `UintDiv` | helper | 地址反算 | 坐标 | COM_T 位宽由 x size 选择 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/inplace_index_fill_tiling_simt.cpp` | `CalcSimtUsedCoreNum/SetTilingData` | 线程/核利用率推导 |
| Kernel entry/router | `op_kernel/inplace_index_fill_apt.cpp` | `DispatchSimt` | template 1 映射 |
| Device body/helper | `op_kernel/arch35/inplace_index_fill_simt.h` | `Process/ComputeByIndices` | 稀疏/稠密运行时分支、地址式 |

### Route R-003：DenseIndices Sort+Unique

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `inplace_index_fill_tiling_simt_dense_indices.cpp` | `IsCapable` | `numel!=0 && Q*size<=128 && P*Q*K>2048 && K>=5*N` | template 2 |
| Router | `inplace_index_fill_apt.cpp` | `DispatchDenseIndices` | dtype mode 4/8 | Dense body |

#### Device 数据流

##### 函数级调用链

```text
DispatchDenseIndices -> Init -> InitMaskZero
  -> Process -> SyncAll -> BuildIndicesMask
     -> CopyInIndices -> SortAndUnique -> ComputeUniqueIdNum(MicroAPI) -> SetMaskSimt
  -> SyncAll -> CallMaskBasedFill
```

##### 源码等价伪码

```text
clear mask; SyncAll
factor = POD.indicesUbFactor; split ceil(K/factor) blocks across min(blocks,blockNum)
for assigned block:
  DMA indices[offset:offset+len] into queued UB
  radix_sort; compare adjacent vector lanes; gather unique positions
  SIMT for each unique position: normalize index; if valid mask[index]=1
SyncAll
SIMT stride over P*N*Q; if mask[nIdx] write value
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indicesQueue | queue slot | `factor*sizeof(I)` each, depth 2 | I | DataCopyPad | Sort | TPipe per core | EnQue/DeQue/Free |
| indicesBuff | UB | `ceil((factor+1)/vfLen)*vfLen*sizeof(I)+32` | I + sentinels | Sort | unique/SetMask | TPipe | 32B front sentinel |
| uniqueIdBuff | UB | `factor*4` | int32 positions | MicroAPI GatherMask | SetMaskSimt | TPipe | separate TBuf |
| mask | workspace 0 | N bytes | int8 | SetMaskSimt | final fill | executor | cross-core shared |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  iq["indicesQueue · depth=2 · 2*factor*sizeof(I)"]:2
  sort["indicesBuff · ceil((factor+1)/vfLen)*vfLen*sizeof(I)+32"]:1
  uniq["uniqueIdBuff · factor*4"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*factor*sizeof(I)
         + ceil((factor+1)/vfLen)*vfLen*sizeof(I)+32
         + factor*4
localMemorySize = platformUB - 128KiB
dynamic_ub = N/A
workspace_total = libApiWorkspace + N
```

Host 的 `maxUbIndicesNum` 以 `(N_BUFFER+1)=3` 倍 indices 空间估算，而 Device 明示 queue depth 2、sort buffer、int32 unique buffer；固定源码未给出静态断言证明上式始终不超过 `ubSize-128KiB`，迁移时必须重新核账。

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | CopyIn | SortAndUnique | FreeTensor 后下轮 Alloc | EnQue/DeQue/Free | depth 2 |
| sort TBuf | Sort/sentinel | unique + SetMask | 下一块 sort | V/S events and synchronous VF call | SetMask 完成后覆盖 |
| mask | all cores | final fill | K1 后 | SyncAll | 同值写 1 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→Sort | TQue + `PIPE_MTE2_S` | CopyIn | Sort | queue slot | tail len 单独传入 |
| Sort→Unique | `PIPE_V_S/S_V` | radix sort/sentinel | MicroAPI compare/gather | sort buffer | INT32/64 exact |
| mask graph | `SyncAll` before/after build | clear/build | build/fill | GM mask | 越界在 SetMask 跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/inplace_index_fill_tiling_simt_dense_indices.cpp` | `DoIndicesTilingTask/PostTiling` | factor、local memory、blockDim |
| Kernel entry/router | `op_kernel/inplace_index_fill_apt.cpp` | `DispatchDenseIndices` | template 2 映射 |
| Device body/helper | `op_kernel/arch35/inplace_index_fill_simt_dense_indices.h` | `BuildIndicesMask` | 分块、Queue、同步 |
| Device body/helper | `op_kernel/arch35/inplace_index_fill_common.h` | `SortAndUnique/ComputeUniqueIdNum` | radix sort 与 MicroAPI 去重 |

## 算子特定优化

### Trick T-001：稀疏索引直接随机写、稠密索引转 mask 顺序写

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`indicesNum*5<=N` 走直接路径，否则走 mask 路径
- Golden 来源：[`inplace_index_fill_simt.h:156-187`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simt.h#L156-L187)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tilingData_->indicesNum * 5 <= tilingData_->n) {
    COM_T process_num = static_cast<COM_T>(tilingData_->indicesNum * p * q);
    COM_T slice_size = static_cast<COM_T>(p * q);
    Simt::VF_CALL<InplaceIndexFillSimtComputeByIndices<T_X, INDEX_TYPE, COM_T>>(
        Simt::Dim3{SIMT_THREAD_NUM}, x,
        (__gm__ INDEX_TYPE*)indicesGm_.GetPhyAddr(), fillValue, process_num,
        p, n, q, slice_size, shift, magic, shift_q, magic_q);
} else {
    BuildIndicesMask(workspace);
    InplaceIndexFillCommon::CallMaskBasedFill<T_X, INDEX_TYPE, COM_T>(
        x, (__gm__ int8_t*)maskGm_.GetPhyAddr(), fillValue, n, p, q);
}
```

#### 机制与设计原因

稀疏时工作量从 `P*N*Q` 降为 `K*P*Q`，接受随机写；稠密时把重复/乱序索引压成 N 字节 mask，再顺序扫描 x，避免随 K 增长的重复写。magic/shift 来自 `P*Q` 和 `Q`，替代坐标反算中的普通整数除法。

#### 迁移边界

- 适用：索引填充且重复写同值的确定性语义。
- 前提：越界跳过、负索引加 N、workspace mask 与跨核同步可用。
- 失效：写值依赖索引顺序或每次写值不同，不能用 mask 合并。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：极密索引先 Sort+Unique 再置 mask

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：`Q*sizeof(x)<=128 && P*Q*K>2048 && K>=5*N`
- Golden 来源：[`inplace_index_fill_simt_dense_indices.h:184-211`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simt_dense_indices.h#L184-L211)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
pipe_->InitBuffer(indicesQueue_, SIMT_DB_BUFFER, indicesUbFactor * sizeof(INDEX_TYPE));
pipe_->InitBuffer(indicesBuff_, indicesBuffSize);
pipe_->InitBuffer(uniqueIdBuff_, indicesUbFactor * sizeof(int32_t));
for (int64_t i = start; i < static_cast<int64_t>(end); i++) {
    offset = i * indicesUbFactor;
    if (i == totalBlocks - 1) {
        CopyInIndices(offset, tailIndicesUbFactor);
        FillMaskWithUniqueIndices(tailIndicesUbFactor);
        break;
    }
    CopyInIndices(offset, indicesUbFactor);
    FillMaskWithUniqueIndices(indicesUbFactor);
}
```

#### 机制与设计原因

每个 indices 分块在 UB 中 radix sort，以相邻比较和 `GatherMask` 只保留唯一位置，再由 SIMT 写 mask；对大量重复索引减少共享 mask 的同地址写次数。双缓冲 Queue 支持搬运生命周期与计算解耦，但固定源码没有量化收益。

#### 迁移边界

- 适用：可交换、幂等的“置位/写同值”索引操作。
- 前提：可为 sentinel、排序输出、unique ID 和 depth-2 输入队列提供足够 UB。
- 失效：索引顺序影响结果或 UB 字节账不满足时应回 R-002。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 SIMD-name | R-002 SIMT | R-003 DenseIndices | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | Q 字节>128 / 1 | 兜底 / 10 | 极密 / 6 | 数字越小优先级越高 |
| 模板与 dtype | mode0, COM32/64 | mode1, COM32/64 | mode2, COM32/64 | x size<=4 选 COM32 |
| 循环与分核 | mask+全量 stride | 稀疏 K*P*Q 或稠密全量 | indices 分块去重+全量 | 降低遍历或重复写 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无显式 UB | 无显式 UB | depth2 Queue+2 TBuf | 仅 Dense 物化排序中间量 |
| 内存总占用与复用 | N-byte mask | sparse 不用；dense N-byte mask | mask + 显式 UB | workspace 全部 Host 统一申请 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT VF | SIMT VF | DataCopyPad、Sort、MicroAPI、SIMT | 无 NDDMA/通信 |
| 同步、精度与 tail | SyncAll×2 | sparse 无；dense×2 | Queue event+SyncAll | 原 dtype 位值写 |
| fallback/失败边界 | →Dense/SIMT | Tiling 失败 | →SIMT | 资源和谓词决定 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 支持；见无后缀卡 | 公开 ACLNN 支持；见无后缀卡 | 支持 |
| Host/Launcher | 原地 API 内部用非原地 IndexFill+ViewCopy | 同 A2 | `IsRegbase()` 直接登记 InplaceIndexFill |
| Kernel/资源 | arch32 AIV Vector、5 keys | 同 A2 | arch35 SIMT/Sort+Unique 三模板、N-byte mask |

## 限制与失败边界

- 950 Tiling 明确拒绝空 `x/indices` shape，但 ACLNN 在空 self 或空 index 前已短路；直接绕过 ACLNN 时不能依赖该短路。
- `P/N/Q/numel` 使用 int64 连乘，固定仓未见溢出检查；SIMD tiling 的多项 UB 字段当前 Device 不消费，不应作为真实 AIV buffer 复刻。
- R-003 Host factor 估算与 Device 明示的 depth-2 Queue + 两 TBuf 字节账没有固定源码静态证明；移植必须重新验证 UB 上界。
- `InitMaskZero` 尾片按 32 字节向上取整，而用户 mask 公式是 N 字节；固定仓未展示这段额外写的 padding/隔离契约。
- 框架 `Contiguous/Unsqueeze/ViewCopy/CommonOpExecutorRun` 的内部 Kernel 与 workspace 切片属于固定仓不可见框架边界；算子自有 selector、POD、K1 和同步已闭合。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/index_fill_d/docs/aclnnIndexFill&aclnnInplaceIndexFill.md` | 产品支持、接口原型 | 950 公开性与语义 |
| 入口实现 | `index/index_fill_d/op_api/aclnn_index_fill.cpp` | `ExecIndexFillGetWorkspaceSize/aclnnInplaceIndexFill` | 校验、前处理、950 分支 |
| 入口实现 | `index/index_fill_d/op_api/index_fill.cpp` | `l0op::InplaceIndexFill` | 核心节点登记 |
| Host/Tiling/Launcher | `index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_base.cpp` | base/registry | P/N/Q、平台、workspace 基础 |
| Host/Tiling/Launcher | `index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_simd.cpp` | SIMD template | R-001 |
| Host/Tiling/Launcher | `index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_simt.cpp` | SIMT template | R-002 |
| Host/Tiling/Launcher | `index/inplace_index_fill/op_host/arch35/inplace_index_fill_tiling_simt_dense_indices.cpp` | Dense template | R-003 |
| Kernel entry/body | `index/inplace_index_fill/op_kernel/inplace_index_fill_apt.cpp` | entry/dispatch | 六模板实例 |
| Kernel entry/body | `index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simd.h` | `InplaceIndexFillSimd` | R-001 mask/SIMT body |
| Kernel entry/body | `index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simt.h` | SIMT body | 双路径 |
| Kernel entry/body | `index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_simt_dense_indices.h` | Dense body | 分块去重 |
| Kernel entry/body | `index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_common.h` | common helpers | mask、Sort/MicroAPI |
| Host/Tiling/Launcher | `index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_struct.h` | 三个 POD class | 字段顺序与 ABI |
| Host/Tiling/Launcher | `index/inplace_index_fill/op_kernel/arch35/inplace_index_fill_tiling_key.h` | `ASCENDC_TPL_SEL` | 六个模板组合 |
| 构建接线 | `index/inplace_index_fill/op_host/inplace_index_fill_def.cpp` | `AddConfig("ascend950")` | 950 OpDef 与 dtype |
| 构建接线 | `index/inplace_index_fill/op_host/config/ascend950/inplace_index_fill_binary.json` | `op_type/op_list` | 20 dtype binary 组合 |
| 构建接线 | `index/inplace_index_fill/CMakeLists.txt` | SUPPORT_COMPUTE_UNIT | ascend950/arch35 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三结构全 int64 顺序字段 | template 对应结构、COM_T 位宽 | `inplace_index_fill_struct.h` |
| 调用顺序 | clear→set/unique→fill | Dense 三个 SyncAll；mask 两次同步 | `Process/BuildIndicesMask` |
| workspace/内存 | mask 位于 workspace 起点 | N 字节公式、Dense UB 重新核账 | `GetWorkspaceSize/InitMaskZero` |
| 同步 | schedule mode 1 + SyncAll | 所有启动核必须参加同步 | `GetWorkspaceSize`、common |
| 精度与尾块 | x 原 dtype 位写；indices 负值加 N | 越界跳过、tail length 传 Sort | 各 SIMT body |
| 失败边界 | 前处理空域短路、模板兜底 | 不绕过 ACLNN 假定空 index 可达 Tiling | API + Tiling base |

- [GatherMask](../../../../apis/simd_api/basic_api/memory_vector_compute/compare_and_select/gather_mask.md)
- [Sort](../../../../apis/simd_api/adv_api/sort_operations/sort.md)
