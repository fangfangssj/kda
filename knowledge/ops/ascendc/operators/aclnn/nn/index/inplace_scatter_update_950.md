---
type: Operator
title: "InplaceScatterUpdate 950 ACLNN 算子知识"
description: "说明 aclnnInplaceScatterUpdate 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/docs/aclnnInplaceScatterUpdate.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_api/aclnn_scatter_update.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_api/aclnn_scatter_update.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_api/scatter.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_graph/scatter_proto.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/scatter_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/arch35/scatter_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/arch35/scatter_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/arch35/scatter_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_kernel/scatter_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_kernel/arch35/scatter.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_kernel/arch35/scatter_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_kernel/arch35/scatter_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_kernel/arch35/scatter_deterministic.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/config/ascend950/scatter_binary.json
    role: implementation
created_at: "2026-08-25T16:59:16Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceScatterUpdate 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceScatterUpdateGetWorkspaceSize` → `aclnnInplaceScatterUpdate`，公开 ACLNN 两段式接口，无 V2 入口。 |
| 计算语义 | 在 `axis` 轴把 `updates` 写入 `data`；一维 `indices[i]` 给出 batch `i` 的轴内起点，二维 `indices[i]={batchDst, axisOffset}` 同时改写目标 batch 和轴内起点。未覆盖元素保持 `data` 原值。 |
| 输入输出 | `data/updates` 为 ND、rank 2–8、同 dtype；`indices` 为 INT32/INT64、rank 0–2；950 还支持 UINT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8。输出原地回写 `data`。 |
| 执行类型 | 算子自有节点始终 `KERNEL_TYPE_AIV_ONLY`；R-001/R-002/R-003 为 AIV_VECTOR 数据搬运，R-004 为 SIMT，R-005/R-006 为 AIV DMA + `asc_vf_call` SIMT 的 Hybrid。无 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 二维索引确定性；R-002 SIMD mode-0；R-003 SIMD mode-1；R-004 小域单核 SIMT；R-005 多核 32 位地址 Hybrid；R-006 多核 64 位地址 Hybrid。 |
| 核心资源 | executor 条件插入 Contiguous/ViewCopy；Scatter 节点请求 16 MiB system workspace；Device 使用 GM、AIV UB Queue/TBuf，或 1024-thread SIMT。 |
| 来源闭包 | source_closed：公开入口、L0 launcher、950 OpDef/build、Tiling/POD、全部可达 router/body/helper 已闭合；Contiguous/ViewCopy、executor 调度和 AscendC primitive 是明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令原始 `data/updates` rank 为 `r`，`a = axis < 0 ? r + axis : axis`。Tiling 把它们折叠为四维：

```text
data4    = [D0, product(D[1:a]), D[a], product(D[a+1:r])]
updates4 = [U0, product(U[1:a]), U[a], product(U[a+1:r])]
```

并把内部轴固定成 `-2`。对一维索引，目标是 `data4[i,o,indices[i]+s,c] = updates4[i,o,s,c]`；对二维索引，目标是 `data4[indices[i,0],o,indices[i,1]+s,c] = updates4[i,o,s,c]`。源码仍保留 `axis==-1` 的四维地址分支，但公开入口经过 `MergeDims` 后实际写入 `axis=-2`。Scatter schema 把输入、输出均命名为 ref `var`，Kernel 注释为 “output reuse input”，且 body 只覆盖被更新片段；具体 ref storage 绑定由 launcher 框架完成，固定仓不展开该框架实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `data` | ND，rank 2–8 | INT8、UINT8、FLOAT16、FLOAT32、INT32、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8 | 输入/输出 ref；非空；950 dtype 列表由 `IsRegbase()` 分支和 OpDef/config 共同支撑。 |
| `indices` | ND，rank 0–2；二维时 dim1=2 | INT32、INT64 | 非空、仅非负且不得越界；rank 0 经 `EnsureNotScalar` 进入一维实现，且要求 `updates.dim0=1`。 |
| `updates` | ND，rank 与 `data` 相同 | 与 `data` 相同 | `U0=indices.dim0` 且 `U0<=D0`；除 batch 与 scatter 轴外各维相等。 |
| `axis` | Host 标量 | INT64 | `(-r,r)` 且不能为 0；归一化后必须在 `[0,r)`，再折叠为内部 `-2`。 |
| `workspaceSize/executor` | Host 输出 | uint64_t*/opaque | 返回整个 executor 图的 workspace 与生命周期对象。 |
| `workspace/stream` | Device 地址/运行流 | opaque | 第二段必须使用第一段返回的大小与 executor；同一 executor 图在给定 stream 上执行。 |

### 入口与版本

固定 Commit 只有无版本后缀的 `aclnnInplaceScatterUpdate`，因此它是 950 实际支持的最新公开版本。API 依次执行三个 `Contiguous`、L0 `Scatter` 和 `ViewCopy`。公开文档明确列出 Ascend 950PR/950DT，OpDef 只注册 `ascend950`，顶层 CMake 指定 `SUPPORT_COMPUTE_UNIT "ascend950"` 与 `arch35`，binary config 覆盖所有 950 dtype/indices 组合，平台证据一致。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckNotNull` 拒绝 `data/indices/updates == nullptr`。
2. `GetDtypeSupportListBySocVersion` 在 regbase/950 选择 9 种 data dtype；indices 只允许 INT32/INT64，updates 必须与 data 同 dtype。
3. `CheckShape` 拒绝空 Tensor、data/updates rank 不同、indices rank 不在 0–2，以及标量 indices 时 `updates.dim0!=1`。
4. `CREATE_EXECUTOR` 后，按 data、indices、updates 顺序添加 `l0op::Contiguous`；任一节点构造失败立即返回。
5. `l0op::Scatter` 分配与 data view shape/dtype 相同的 ref 输出描述，`ScatterAiCore` 以 `reduce="update"`、原始 `axis` 添加 AICore launcher。
6. `l0op::ViewCopy(scatterUpdateRes, data)` 把结果提交回原始 view。第一段返回 executor 聚合 workspace；第二段由 `CommonOpExecutorRun` 在 stream 上运行。

更严格的 axis、折叠 shape、二维 indices dim1 等检查位于 Tiling；失败返回 `GRAPH_FAILED` 并阻断 Scatter launch。

### 参数、Workspace 与生命周期

Scatter Tiling 固定 `workspaceSize_=16*1024*1024` 字节，`PostTiling` 写入 workspace slice 0；entry 要求传入 workspace 非空，调用 `SetSysWorkspace/GetUserWorkspace`，但各 Route 不读写 user slice。ACLNN 返回值是整个 executor 图的聚合 workspace，还可能包括三个 Contiguous、Scatter ref 输出及 ViewCopy 的框架分配；其精确拼接 offset 固定仓不可见。调用者拥有 `workspace`，executor 从第一段释放给调用者并在第二段消费；Scatter POD 与临时 tensor 随 executor/stream 生命周期有效。

### 启动图与失败传播

```text
N0 data Contiguous  ─┐
N1 indices Contiguous ├─(executor 顺序、框架临时 tensor)─> N3 Scatter
N2 updates Contiguous ┘                                  │
                                                         └─> N4 ViewCopy -> 原 data view
```

N0/N1/N2/N4 是允许的框架边界：若输入本已连续，框架可复用其 storage；否则生成临时连续 tensor，具体 Kernel、offset 与 workspace slice 固定仓不可见。N3 是唯一算子自有 Device 节点：Host 添加符号 `ADD_TO_LAUNCHER_LIST_AICORE(Scatter, ...)`，entry 为 `scatter(x,indices,updates,y,workspace,tiling)`，`blockDim=aivCoreNum`，单个 `ScatterTilingData`。executor 按构造顺序在传入 stream 上调度，任一节点创建失败阻断后续；运行期内部错误传播由 `CommonOpExecutorRun` 框架处理，固定仓不展开事件/cleanup 细节。

## Route 与参数设计

### 分核与资源计算

定义 `B=U0, O=product(U[1:a]), S=U[a], C=product(U[a+1:r])`，`srcStride=S*C`，`E=dtypeSize`，平台 AIV 核数 `P`、UB 字节 `UB`：

```text
deterministic: normCol=max(ceil(C/P), 128/E); used=ceil(C/normCol)
SIMD: split=B*O (mode0) 或 B*O*S (mode1);
      blockFactor=ceil(split/P); used=ceil(split/blockFactor); tail=split-(used-1)*blockFactor
generic: total=B*O*S*C（可能先 8B 打包）
         total<=1024: used=1
         total<=P*1024: blockFactor=1024, used=ceil(total/1024)
         else: blockFactor=ceil(total/P), used=P
```

非 SIMD 路径在内部轴为 -2 且 `C` 可被 `8/E` 整除时，把 `C` 除以 `8/E` 并把 `E` 提升为 8，即按 64-bit 包搬运连续元素；地址域 `total>INT32_MAX` 时选择 64-bit `T3`。SIMD 可用 `loopLength=floorAlign((UB-8KiB-4096)/2, ubBlockSize)/E`；generic Device 可用 `tiling.ubSize=UB-32KiB-8KiB`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 确定性二维索引 | 1000 | `context.GetDeterministic() && originalIndicesRank==2` | R-002/R-003 检查 |
| 2 | R-003 SIMD mode-1 | 200 / `ScatterSimd<IDX_T,1>` | 非 R-001，`S*C*E>1024 && B*O<P && S*C>loopLength && C*E>8192` | R-002 |
| 3 | R-002 SIMD mode-0 | 100 / `ScatterSimd<IDX_T,0>` | 非 R-001，`S*C*E>1024` 且未命中 mode-1 | R-004/R-005/R-006 |
| 4 | R-004 小域单核 SIMT | 300–313 | 非 SIMD，8B 打包后 `total<=1024` | R-005/R-006 |
| 5 | R-006 64 位地址 Hybrid | 14–27 | 非 SIMD、`total>1024 && total>2147483647` | R-005 |
| 6 | R-005 32 位地址 Hybrid | 0–13 | 非 SIMD、`1024<total<=2147483647` | 未知 dtype tuple 得 key 255，router 无 body |

```text
if deterministic && indicesRank==2: key=1000
else if S*C*E>1024:
  if B*O<P && S*C>loopLength && C*E>8192: key=200
  else: key=100
else:
  if axis==-2 && C%(8/E)==0: C/=8/E; E=8
  total=B*O*S*C
  base=map(address64, indices_dtype, original_data_dtype) 或 packed-int64 key
  if total<=1024: key=base+300
  else: key=base
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1000 | 950；deterministic；indices rank=2 | 单 Scatter AIV 节点 | `ScatterDeterministic<VAR_T,DTYPE_INDICES>` | indices queue + updates bind queue；MTE2→S、MTE2→MTE3 | 列分核，最后核/最后 loop 独立长度 |
| R-002 | 100 | 非确定性；`S*C*E>1024`；mode-1 条件假 | 单 Scatter AIV 节点 | `ScatterSimd<DTYPE_INDICES,0>` | 4096B index TBuf + depth-2 byte queue；MTE2→S | 按 B*O 分核，列超 tile 再切分 |
| R-003 | 200 | R-002 前提且四个 mode-1 条件全真 | 单 Scatter AIV 节点 | `ScatterSimd<DTYPE_INDICES,1>` | 同 R-002 | 按 B*O*S 分核，避免 B*O 核并行不足 |
| R-004 | 300–313 | 非 SIMD；打包后 total<=1024 | 单 Scatter SIMT 节点 | `ScatterSimt<T1,T2,uint32_t>` → `SimtOneCoreCompute0..3` | 1024B parameter TBuf；1024 threads 直接 GM load/store | thread `idx<total`；无 fallback |
| R-005 | 0–13 | 非 SIMD；1024<total<=INT32_MAX | 单 Scatter Hybrid 节点 | `Scatter<T1,T2,uint32_t>` → `SimtCompute0..3` | depth-2 indices/updates queues + 1024B TBuf；UB barrier 后 SIMT | former/tail core；UB loop；32-bit magic division |
| R-006 | 14–27 | 非 SIMD；total>INT32_MAX | 单 Scatter Hybrid 节点 | `Scatter<T1,T2,uint64_t>` → `SimtUint64Compute0..3` | 同 R-005，参数项变 64-bit | 64-bit地址与 `Simt::UintDiv`；无更大 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| deterministic 优先 | `GetDeterministic() && rank(indices)==2` | 二维 indices 可按输出列分核，避免多个 core 写同一连续列片 | R-001 | SIMD/generic | `ScatterTiling::GetShapes/DoOpTiling` |
| SIMD 阈值 | `S*C*E>1024` | 连续更新片足够大时整片 DMA 比逐元素 SIMT 更直接 | R-002/R-003 | R-004–R-006 | `DoOpTiling` |
| mode-1 并行度 | `B*O<P && S*C>loopLength && C*E>8192` | mode-0 的 B*O 域不足核且单片需切分，改按 B*O*S 分核 | R-003 | R-002 | `DoOpTiling` |
| 8B 打包 | `axis==-2 && C%(8/E)==0` | 连续内轴按 int64 word 搬运，减少线程元素数且保持 bit pattern | R-004–R-006 | 原 dtype 实例 | `PromoteDtype/GetTilingKey` |
| 单核 SIMT | packed `total<=1024` | 一个 1024-thread `asc_vf_call` 覆盖全部更新，跳过 GM→UB payload DMA | R-004 | Hybrid | `GetTilingParam/setSimtTilingKey` |
| 地址位宽 | `total>2147483647` | 32-bit `addr/dstOffset` 不能覆盖整个更新域 | R-006 | R-005 | `GetTilingParam/GetTilingKey` |
| UB 拆分 | generic `indicesBytes>tiling.ubSize/2` | indices 与 updates depth-2 queues 必须共享 `UB-40KiB` | R-005/R-006 | 每 UB loop 重载 indices | `Scatter::Process` |
| workspace 非空 | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 初始化 system workspace；空指针直接无输出 | 全 Route | return | `scatter` |

### POD/ABI

`ScatterTilingData` 由外部 `tilingdata_base.h` 宏生成；固定 `ops-nn` 仓未包含宏展开定义，因此 `sizeof/alignof/offsetof` 不可验证。下表严格保留字段声明顺序、源码位宽、producer 与可达 consumer；“未读”表示字段虽序列化但该 Device body 不读取。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| ScatterTilingData | tilingKey | uint32_t/32 | 宏展开不可见 | key | `PostTiling:GetTilingKey()` | entry `TILING_KEY_IS` | 未知 key 无 body |
| ScatterTilingData | ubSize | uint32_t/32 | 同上 | byte | `SetTilingData: UB-32KiB-8KiB` | R-005/R-006 `Scatter::Process` | 仅 Hybrid |
| ScatterTilingData | aivCoreNum | uint32_t/32 | 同上 | core | 分核后 `SetTilingData` | 全 Route core guard/尾核 | `blockDim` 同值 |
| ScatterTilingData | tailCoreNum | uint32_t/32 | 同上 | core | generic `GetTilingParam` | R-005/R-006 former/tail | 非 Hybrid 未读 |
| ScatterTilingData | blockFactor | int64_t/64 | 同上 | element/row | SIMD/generic 分核 | R-002–R-006 | R-001 未读 |
| ScatterTilingData | axis | int32_t/32 | 同上 | dim id | `MergeDims` 后 -2 | SIMT/Hybrid 地址分支 | SIMD/R-001地址已固化 |
| ScatterTilingData | indicesDim | uint32_t/32 | 同上 | rank/tuple width | `GetShapeAttrsInfo` | 全 Route 选 1/2 index | scalar经 EnsureNotScalar |
| ScatterTilingData | inputDim0 | int64_t/64 | 同上 | element | `inputNewShape[0]` | SIMT/Hybrid GM extent | 乘法溢出未显式检查 |
| ScatterTilingData | inputDim1 | int64_t/64 | 同上 | element | `inputNewShape[1]` | 全 Route 地址 | 同上 |
| ScatterTilingData | inputDim2 | int64_t/64 | 同上 | element | `inputNewShape[2]` | 全 Route 地址 | 同上 |
| ScatterTilingData | inputDim3 | int64_t/64 | 同上 | element/packed word | `inputNewShape[3]` | 全 Route 地址 | 可被 8B 打包缩小 |
| ScatterTilingData | updatesDim0 | int64_t/64 | 同上 | element | `updatesNewShape[0]` | 全 Route循环/extent | 同上 |
| ScatterTilingData | updatesDim1 | int64_t/64 | 同上 | element | `updatesNewShape[1]` | 全 Route循环/地址 | 同上 |
| ScatterTilingData | updatesDim2 | int64_t/64 | 同上 | element | `updatesNewShape[2]` | 全 Route循环/地址 | 同上 |
| ScatterTilingData | updatesDim3 | int64_t/64 | 同上 | element/packed word | `updatesNewShape[3]` | 全 Route循环/地址 | 可被 8B 打包缩小 |
| ScatterTilingData | tailBlockData | int64_t/64 | 同上 | row | `DoSimdTiling` | R-002/R-003 last core | 仅 SIMD |
| ScatterTilingData | loopLength | int64_t/64 | 同上 | element | SIMD UB公式 | R-002/R-003 tile | 仅 SIMD |
| ScatterTilingData | indicesUbSize | int64_t/64 | 同上 | byte | SIMD 固定4096 | R-002/R-003 indexBuf/maxIndex | 仅 SIMD |
| ScatterTilingData | dtypeSize | int64_t/64 | 同上 | byte/element | 原 dtype 或 promoted 8 | R-002/R-003 byte offset | 其余 Route 未读 |
| ScatterTilingData | simtUsedCore | int64_t/64 | 同上 | core | oneCore `set=1` | R-004 `ScatterSimt::Init` | 仅 R-004 |
| ScatterTilingData | simtPerCoreNum | int64_t/64 | 同上 | element | oneCore `set=total` | R-004 | 仅 R-004 |
| ScatterTilingData | simtTailCoreNum | int64_t/64 | 同上 | element | oneCore `set=total` | R-004 | 仅 R-004 |
| ScatterTilingData | simtThreadNum | int64_t/64 | 同上 | thread | oneCore `set=1024` | Device 未读；模板常量 THREAD_NUM=1024 | 冗余字段 |
| ScatterTilingData | normBlockColNum | int64_t/64 | 同上 | element | `DoDeterministicTiling` | R-001列 offset | 仅 R-001 |
| ScatterTilingData | indicesUbFactor | int64_t/64 | 同上 | element | deterministic UB分配 | R-001 queue/loop | 仅 R-001 |
| ScatterTilingData | updatesColUbFactor | int64_t/64 | 同上 | element | deterministic UB分配 | R-001 queue/loop | 仅 R-001 |
| ScatterTilingData | indicesLoop | int64_t/64 | 同上 | loop | `ceil(indicesSize/factor)` | R-001外循环 | 仅 R-001 |
| ScatterTilingData | indicesTailLoopNum | int64_t/64 | 同上 | element | deterministic tail | R-001最后 indices loop | 仅 R-001 |
| ScatterTilingData | updatesNormBlockColLoop | int64_t/64 | 同上 | loop | deterministic normal core | R-001 normal core | 仅 R-001 |
| ScatterTilingData | updatesTailBlockColLoop | int64_t/64 | 同上 | loop | deterministic tail core | R-001 last core | 仅 R-001 |
| ScatterTilingData | updatesNormBlockTailLoopSize | int64_t/64 | 同上 | element | deterministic normal core | R-001 normal core tail | 仅 R-001 |
| ScatterTilingData | updatesTailBlockTailLoopSize | int64_t/64 | 同上 | element | deterministic tail core | R-001 last core tail | 仅 R-001 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1000 | `scatter_tiling_arch35.cpp:GetShapes/DoDeterministicTiling` | ScatterTilingData deterministic fields | `scatter_apt.cpp:scatter` | `scatter_deterministic.h:ScatterDeterministic` | `scatter_def.cpp` + `CMakeLists.txt` + ascend950 JSON |
| R-002 | 100 | `DoOpTiling/DoSimdTiling` | SIMD fields, mode0 | `scatter` | `scatter_simd.h:ScatterSimd<IDX_T,0>` | 同上 |
| R-003 | 200 | `DoOpTiling/DoSimdTiling` | SIMD fields, mode1 | `scatter` | `scatter_simd.h:ScatterSimd<IDX_T,1>` | 同上 |
| R-004 | 300–313 | `PromoteDtype/GetTilingParam/GetTilingKey` | oneCore fields | `scatter` | `scatter_simt.h:ScatterSimt` + `scatter.h:CalcIndex*` | 同上 |
| R-005 | 0–13 | `GetTilingParam/InitTilingKeyMap/GetTilingKey` | generic fields | `scatter` | `scatter.h:Scatter<T1,T2,uint32_t>` | 同上 |
| R-006 | 14–27 | 同 R-005，`isUint64=true` | generic fields | `scatter` | `scatter.h:Scatter<T1,T2,uint64_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 先验证 workspace，设置 system workspace、解析 POD，创建 `TPipe`，声明 AIV-only task，然后严格按 key 路由到一个 body。各 body 都把 `updates` 的 bit pattern 写到 ref 输出，不做算术、cast、round 或 atomic；数据精度等于输入 dtype。indices 越界不在 Device 防护，重复目标也不做冲突消解。所有 Route 共享单 Scatter Kernel 拓扑；框架 Contiguous 与 ViewCopy 位于其前后。

### Route R-001：确定性二维索引 AIV 搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `scatter_tiling_arch35.cpp` | `GetShapes` | `context_->GetDeterministic() && indicesDimSize==2` | `isDeterministic_=true` |
| Tiling | 同上 | `DoOpTiling/GetTilingKey` | deterministic 在 SIMD 判断前返回 | key 1000 |
| Router | `scatter_apt.cpp` | `scatter` | `TILING_KEY_IS(1000)` | `ScatterDeterministic<VAR_T,DTYPE_INDICES>` |

#### Device 数据流

##### 函数级调用链

```text
ScatterAiCore -> scatter(..., key=1000)
  -> ScatterDeterministic::Init
  -> ScatterDeterministic::Process
    -> CopyInIndices -> ProcessUpdates
      -> CopyInUpdates -> CopyOutUpdates
```

##### 源码等价伪码

```text
for each indices tile i:
  DMA indices[i*indicesUbFactor : +indicesCount] GM->UB; wait MTE2->S
  for j in [0,indicesCount) step 2:
    batchDst=indices[j]; axisStart=indices[j+1]
    for outer o, scatter s, local column tile c:
      src=((globalPair*O+o)*S+s)*C + blockIdx*normCol + c
      dst=((batchDst*inputO+o)*inputS+(axisStart+s))*C + blockIdx*normCol + c
      DMA updates[src:src+n] GM->UB; wait MTE2->MTE3; DMA UB->y[dst:dst+n]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices GM | `i*indicesUbFactor` | indicesSize elements | INDICES_T, pairs | API/Contiguous | CopyInIndices | executor | chunked |
| indicesQue | 0 | `indicesUbFactor*sizeof(INDICES_T)` | INDICES_T | MTE2 | scalar GetValue | TPipe, per tile | depth1 |
| updates GM | formula above | `updatesCount` | VAR_T ND | updates | CopyInUpdates | executor | contiguous column |
| updatesQueue | 0 | `updatesColUbFactor*sizeof(VAR_T)` | VAR_T | MTE2 | MTE3 | TPipe, per column tile | depth1 bind VECIN/VECOUT |
| y GM | formula above | ref data extent | VAR_T ND | CopyOutUpdates | ViewCopy/data | executor | disjoint column cores |
| system workspace | framework offset | 16 MiB requested | byte | runtime | system runtime | API caller | user slice unused |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  idx["indicesQue · depth=1 · indicesUbFactor*sizeof(INDICES_T)"]:1
  upd["updatesQueue bind · depth=1 · updatesColUbFactor*sizeof(VAR_T)"]:1
```

##### 字节账与复用顺序

```text
UB_total = indicesUbFactor*sizeof(INDICES_T) + updatesColUbFactor*sizeof(VAR_T)
Host 保证两项由 UB 分支互补计算，均按 32B 对齐边界切分，合计 <= UB
dynamic_ub = N/A；AIV Queue 路径
workspace_total(Scatter node) = 16 MiB system workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| indicesQue slot | CopyInIndices | scalar GetValue/ProcessUpdates | 下一 indices tile | MTE2→S + FreeTensor | 完成当前 tile 所有 pair |
| updatesQueue slot | CopyInUpdates | CopyOutUpdates | 下一 column tile | MTE2→MTE3 + FreeTensor | CopyOut完成后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| indices | MTE2_S flag | DataCopyPad | GetValue前 WaitFlag | indices UB | 原整数；最后 tile=`indicesTailLoopNum` |
| updates | MTE2_MTE3 flag | GM→UB | UB→GM前 WaitFlag | updatesQueue | bitwise搬运；normal/tail core各自 loop/tail size |
| 跨核 | 分列无 atomic | Host column split | 各 core 独立 | y column slice | 重叠 indices pair 仍可能覆盖同一目标，公开约束不保证重复结果 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_tiling_arch35.cpp` | `DoDeterministicTiling` | 计算列分核、两类 UB factor 与四个 tail 字段 |
| Kernel entry/router | `op_kernel/scatter_apt.cpp` | key 1000 | 映射 `VAR_T/DTYPE_INDICES` |
| Device body/helper | `op_kernel/arch35/scatter_deterministic.h` | `Process/ProcessUpdates` | 二维 pair、Queue DMA、显式 event |

### Route R-002：SIMD mode-0 大连续片搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `scatter_tiling_arch35.cpp` | `DoOpTiling` | 非R-001且 `S*C*E>1024` | `simdTemp=1` |
| Tiling | 同上 | mode-1 if | mode-1复合条件为假 | key100 |
| Router | `scatter_apt.cpp` | key100 | `TILING_KEY_SIMD` | `ScatterSimd<DTYPE_INDICES,0>` |

#### Device 数据流

##### 函数级调用链

```text
scatter -> ScatterSimd<IDX_T,0>::Init -> Process
  -> NoSplitColProcessForMode0 或 SplitColProcessForMode0
    -> GetIndex -> CopyIn/DataCopyPad -> CopyOut/DataCopyPad
```

##### 源码等价伪码

```text
core covers rows in flattened B*O
load needed indices chunk to 4096B TBuf; wait MTE2->S
if align(S*C*E)<=loopLength*E:
  DMA multiple update rows into depth-2 queue; for each row DMA its valid S*C*E bytes to mapped y
else:
  for each row and C/S column tile: DMA updates tile GM->queue->mapped y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indexBuf | chunk by batch id | 4096B | IDX_T | MTE2 | scalar address | TPipe | MTE2_S |
| inQueue | row/tile offset | `2*loopLength*E` total | int8 bytes | MTE2 | MTE3 | TPipe | depth2 bind |
| y GM | `((batchDst*inputO+outer)*inputS+axisDst)*inputC*E` | valid copied bytes | raw bytes | CopyOut | ref output | executor | no arithmetic |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inq["inQueue · depth=2 · 2*loopLength*E bytes"]:2
  idx["indexBuf · 4096 bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*loopLength*E + 4096
loopLength=floorAlign((UB-8192-4096)/2, ubBlockSize)/E
=> UB_total <= UB-8192（保留8KiB）
dynamic_ub=N/A; workspace=16MiB system
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| indexBuf | GetIndex CopyIn | scalar GetValue | batch超出缓存窗口 | MTE2_S | 当前 batch 位于窗口 |
| queue slot | CopyIn | CopyOut | 下一 row/tile | EnQue/DeQue/FreeTensor | CopyOut完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| index | MTE2_S | DataCopyPad | GetValue | indexBuf | INT32/INT64原值 |
| payload | TQueBind lifecycle | MTE2 | MTE3 | byte queue | 仅有效 `cols*E`，padding不写回 |
| core tail | `curCoreData` | Tiling | last core | rows | `tailBlockData` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_tiling_arch35.cpp` | `DoOpTiling/DoSimdTiling` | mode0阈值、tile与分核 |
| Kernel entry/router | `op_kernel/scatter_apt.cpp` | key100 | `ScatterSimd<IDX_T,0>` |
| Device body/helper | `op_kernel/arch35/scatter_simd.h` | mode0 methods | B*O分核、整片或切列 DMA |

### Route R-003：SIMD mode-1 并行展开搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `scatter_tiling_arch35.cpp` | `DoOpTiling` | `S*C*E>1024 && B*O<P && S*C>loopLength && C*E>8192` | `simdTemp=2`，key200 |
| Router | `scatter_apt.cpp` | key200 | `TILING_KEY_SIMD_PERF` | `ScatterSimd<IDX_T,1>` |

#### Device 数据流

##### 函数级调用链

```text
scatter -> ScatterSimd<IDX_T,1>::Init -> Process
  -> NoSplitColProcessForMode1 或 SplitColProcessForMode1
    -> GetIndex -> CopyInAndOut
```

##### 源码等价伪码

```text
flatten work as B*O*S, so each core covers individual scatter rows
derive batch, outer, scatterI from srcIdx
axisDst=scatterI+index[batch] (or 2D pair)
copy C*E bytes, or loopLength-element column tiles, updates GM->UB->y GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indexBuf | 同R-002 | 4096B | IDX_T | MTE2 | scalar | TPipe | cache window |
| inQueue | `(srcIdx*C+j*loopLength)*E` | depth2×`loopLength*E` | int8 | MTE2 | MTE3 | TPipe | tile reuse |
| y GM | mapped `(batch,outer,axisDst,column)` | C*E valid | raw bytes | CopyOut | ref output | executor | per scatter row |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inq["inQueue · depth=2 · 2*loopLength*E bytes"]:2
  idx["indexBuf · 4096 bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*loopLength*E + 4096 <= UB-8192
dynamic_ub=N/A; workspace=16MiB system
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| indexBuf/queue | 同R-002 | 同R-002 | 下一窗口/列tile | MTE2_S与Queue生命周期 | B*O*S 分核后目标片不冲突时安全 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| index/payload | 同R-002 | DMA | scalar/MTE3 | UB | byte搬运，无cast |
| core tail | `tailBlockData` | Host | last core | B*O*S rows | last row/tile精确长度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_tiling_arch35.cpp` | mode-1 branch | 四条件同时触发 |
| Kernel entry/router | `op_kernel/scatter_apt.cpp` | key200 | mode1实例 |
| Device body/helper | `op_kernel/arch35/scatter_simd.h` | mode1 methods | B*O*S分核与C列切片 |

### Route R-004：小域单核 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `scatter_tiling_arch35.cpp` | `GetTilingParam` | 非SIMD且 packed total<=1024 | oneCoreTemp=true，used=1 |
| Tiling | 同上 | `setSimtTilingKey` | oneCoreTemp | base key+300 |
| Router | `scatter_apt.cpp` | 300–313 | dtype/indices key | `ScatterSimt<...,uint32_t>` |

#### Device 数据流

##### 函数级调用链

```text
scatter -> ScatterSimt::Init -> Process
  -> DataSyncBarrier<UB>
  -> asc_vf_call<SimtOneCoreCompute0..3>(dim3(1024))
```

##### 源码等价伪码

```text
write factor/coef/magic/shift arrays to 1024B TBuf; UB barrier
for thread=threadIdx.x; thread<total; thread+=1024:
  decode updates address using magic multiply/shift
  read indices directly from GM
  compute destination offset
  y[dst]=updates[thread] directly in GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| simtBuf | factors 0; coefs `3*sizeof(T3)`; shifts `11*sizeof`; magic `14*sizeof` | fixed 1024B | uint32 params | scalar AIV | SIMT threads | TPipe, launch | subviews alias one TBuf |
| indices/updates GM | thread-derived | total | T2/T1 | inputs | SIMT | executor | direct GM |
| y GM | decoded dst | one element/thread | T1 | SIMT | ref output | executor | repeated indices race |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  factor["factor[3] · 12B"]:1
  coef["coef[8] · 32B"]:1
  shift["shift[3] · 12B"]:1
  magic["magic[3] · 12B; TBuf总容量1024B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 1024 bytes
dynamic_ub = N/A（asc_vf_call固定1024 threads；参数在AIV TBuf）
workspace=16MiB system
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| simtBuf subviews | `Process` scalar stores | SIMT threads | Kernel结束 | `DataSyncBarrier<UB>` | barrier后调用SIMT |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| params | UB DSB | AIV stores | asc_vf_call | factor arrays | uint32地址；无cast |
| threads | `idx<batchNum` | 1024 threads | loop guard | updates domain | total<=1024，每线程最多一元素 |
| writes | direct GM | thread | none/无atomic | y element | 重复目标结果不保证 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_tiling_arch35.cpp` | `GetTilingParam/setSimtTilingKey` | 小域映射300系 |
| Kernel entry/router | `op_kernel/scatter_apt.cpp` | keys300–313 | dtype模板 |
| Device body/helper | `scatter_simt.h` + `scatter.h` | `ScatterSimt/CalcIndex2/3` | 直接GM SIMT与magic除法 |

### Route R-005：多核 32 位地址 Hybrid

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `scatter_tiling_arch35.cpp` | `GetTilingParam` | 非SIMD、1024<total<=INT32_MAX | partial/all cores，isUint64=false |
| Tiling | 同上 | `InitTilingKeyMap/GetTilingKey` | `(false,indicesDtype,inputDtype)` 或 packed key12/13 | key0–13 |
| Router | `scatter_apt.cpp` | keys0–13 | dtype | `Scatter<T1,T2,uint32_t>` |

#### Device 数据流

##### 函数级调用链

```text
scatter -> Scatter<T1,T2,uint32_t>::Init -> Process
  -> DataCopyPad(indices, updates)
  -> DataSyncBarrier<UB>
  -> ScatterComputeUint32
    -> asc_vf_call<SimtCompute0..3>(1024 threads)
```

##### 源码等价伪码

```text
core selects blockFactor (tail cores one element shorter or remainder)
derive startOffset,indexStart,indexEnd
partition available UB into depth2 indices and updates queues
for each updates UB tile:
  load required indices once, or reload when indices exceed half UB
  DMA updates GM->UB
  publish factor/coef/magic arrays; UB barrier
  SIMT threads decode each UB element and scatter directly UB->y GM
  free queues; advance startOffset by updatesBufferSize/sizeof(T1)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices queue | indexStart*indicesDim | `2*indicesBufferSize=indicesUbSize` | T2 | MTE2 | SIMT | TPipe/core | 64B rounded, depth2 |
| updates queue | startOffset | `2*updatesBufferSize=updatesUbSize` | T1 | MTE2 | SIMT | TPipe/tile | 64B floor, depth2 |
| simtBuf | same subviews R-004 | 1024B | uint32 | AIV | SIMT | TPipe | DSB |
| y GM | decoded offset | per UB element | T1 | SIMT | ref output | executor | direct store |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  idx["indices queue · depth=2 · indicesUbSize bytes"]:1
  upd["updates queue · depth=2 · updatesUbSize bytes"]:2
  prm["simtBuf · 1024B"]:1
```

##### 字节账与复用顺序

```text
queue bytes = indicesUbSize + updatesUbSize
indicesUbSize = ceil64(indicesBytes) 或 ceil64(tiling.ubSize/2)
updatesUbSize = floor64(tiling.ubSize-indicesUbSize)
UB_total = indicesUbSize + updatesUbSize + 1024
tiling.ubSize=UB-40KiB；1024B simtBuf落在预留空间内
dynamic_ub=N/A；workspace=16MiB system
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| indices slot | DataCopyPad | SIMT | 下一core tile或重载 | EnQue/DeQue/FreeTensor | 若大indices，每loop重载后才free |
| updates slot | DataCopyPad | SIMT | 下一UB loop | queue lifecycle | asc_vf_call完成后FreeTensor |
| simtBuf | AIV | SIMT | 下一UB loop | UB barrier + synchronous call boundary | 参数消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue | EnQue/DeQue | MTE2 | SIMT | payload/index UB | DataCopyPad有效字节 |
| params | UB DSB | scalar stores | SIMT | param TBuf | uint32 magic地址 |
| core tail | blockCoreNum | Tiling | Process | updates interval | partial-core remainder或全核缺口 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_tiling_arch35.cpp` | `GetTilingParam/GetTilingKey` | 32-bit route与tail cores |
| Kernel entry/router | `op_kernel/scatter_apt.cpp` | keys0–13 | T1/T2/uint32实例 |
| Device body/helper | `op_kernel/arch35/scatter.h` | `Scatter::Process/SimtCompute*` | 动态UB分区、DMA+SIMT Hybrid |

### Route R-006：多核 64 位地址 Hybrid

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `scatter_tiling_arch35.cpp` | `GetTilingParam` | packed total>2147483647 | `isUint64=true` |
| Tiling | 同上 | `GetTilingKey` | uint64 tuple/packed key | 14–27 |
| Router | `scatter_apt.cpp` | 14–27 | key | `Scatter<T1,T2,uint64_t>` |

#### Device 数据流

##### 函数级调用链

```text
scatter -> Scatter<T1,T2,uint64_t>::Process
  -> ScatterComputeUint64
    -> asc_vf_call<SimtUint64Compute0..3>
      -> CalcUint64Index2/3 -> Simt::UintDiv
```

##### 源码等价伪码

```text
与R-005相同的core/UB循环和DMA网络
区别：startOffset/indexStart/idx/dstOffset及factor/coef/magic为64位，
SIMT用Simt::UintDiv(value,magic,shift)恢复二维/三维坐标后写GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices/updates queues | 同R-005 | 同R-005 | T2/T1 | DMA | SIMT64 | TPipe | depth2 |
| simtBuf | factor/coef/shift/magic 64-bit subviews | 1024B | uint64 | AIV | SIMT64 | TPipe | DSB |
| y GM | uint64 decoded dst | element | T1 | SIMT64 | ref output | executor | direct store |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  idx["indices queue · depth=2 · indicesUbSize bytes"]:1
  upd["updates queue · depth=2 · updatesUbSize bytes"]:2
  prm["simtBuf · 1024B · 64-bit params"]:1
```

##### 字节账与复用顺序

```text
UB_total = indicesUbSize + updatesUbSize + 1024；同R-005上界
dynamic_ub=N/A；workspace=16MiB system
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queues/simtBuf | 同R-005 | 64-bit SIMT | 下一UB loop | Queue + UB DSB | 当前asc_vf_call完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/params | 同R-005 | MTE2/AIV | SIMT64 | UB | payload bitwise不变 |
| address | `Simt::UintDiv` | magic/shift | thread | >2^31 element域 | 64-bit offset；未提供>int64 shape fallback |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_tiling_arch35.cpp` | `isUint64/GetTilingKey` | INT32_MAX边界 |
| Kernel entry/router | `op_kernel/scatter_apt.cpp` | keys14–27 | uint64 T3实例 |
| Device body/helper | `op_kernel/arch35/scatter.h` | `SimtUint64Compute*/CalcUint64Index*` | 64位地址解码 |

## 算子特定优化

### Trick T-001：按 8 字节打包连续内轴

#### 触发与源码依据

- 适用 Route：R-004、R-005、R-006。
- 触发条件：非 SIMD，内部 `axis==-2` 且 `updatesDim3 % (8/dtypeSize)==0`。
- Golden 来源：[`index/scatter/op_host/arch35/scatter_tiling_arch35.cpp:371-380`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/arch35/scatter_tiling_arch35.cpp#L371-L380)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int32_t factor = PROMOTE_DTYPE_SIZE / dtypeSize;
if (axis == SECOND_LAST_DIM && updatesNewShape[DIM3] % factor == 0) {
  updatesNewShape[DIM3] /= factor;
  inputNewShape[DIM3] /= factor;
  dtypeSize = PROMOTE_DTYPE_SIZE;
}
```

#### 机制与设计原因

Host 把连续内轴的 `8/E` 个原元素合成一个 8B word，router 选择 `T1=int64_t` 的 key12/13/26/27。Kernel 只做 bitwise load/store，不解释数值，因此一次 SIMT 迭代搬运 8B 并同步缩小地址域；这减少迭代数，也可能把 total 降到单核 SIMT Route。没有 cast/round。

#### 迁移边界

- 适用：纯搬运型 scatter，内轴连续且整除 8B。
- 前提：950、内部 axis=-2、输出与输入 dtype/布局一致。
- 失效：不能整除或走 SIMD 时保留原 dtype key。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在非连续内轴或有数值计算时直接把 dtype 改成 int64；固定仓未提供其它可验证反例。

### Trick T-002：低并行 mode-0 自动展开 scatter 轴

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：`B*O<P && S*C>loopLength && C*E>8192`，且已满足 `S*C*E>1024`。
- Golden 来源：[`index/scatter/op_host/arch35/scatter_tiling_arch35.cpp:536-546`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter/op_host/arch35/scatter_tiling_arch35.cpp#L536-L546)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t boNum = updatesNewShape[DIM0] * updatesNewShape[DIM1];
if (boNum < aivCoreNum && srcStride > loopLength && updatesNewShape[DIM3] * dtypeSize > SIMD_PERF_SIZE) {
  simdTemp = SIMD_PERF_TEMP;
}
```

#### 机制与设计原因

mode-0 只把 `B*O` 当分核行；当该域小于核数且一行超过 tile 时，mode-1 把 `S` 也纳入分核域，Kernel 地址解码随之从 `(batch,outer)` 变为 `(batch,outer,scatter)`。它提高可用核粒度且仍保持连续 C 列 DMA。

#### 迁移边界

- 适用：scatter 轴较大、末轴连续、外层并行度不足的数据搬运。
- 前提：950，C 列单片超过 8KiB，S*C 超过当前 UB tile。
- 失效：条件任一不成立使用 R-002。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若无条件按 B*O*S 分核，短 C 列会增加调度粒度；固定仓以四个 guard 限制该变化，未提供真机反例。

## 模板与实现对比

| 对比维度 | R-001 确定性 | R-002/003 SIMD | R-004 SIMT | R-005/006 Hybrid | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | deterministic+2D最高 | 大连续片 | packed total<=1024 | 其余多核 | 先语义确定性，再搬运形态/规模 |
| 模板与 dtype | VAR_T/INDICES_T | IDX_T；payload int8 bytes | T1/T2/u32 | T1/T2/u32或u64 | payload均不数值转换 |
| 循环与分核 | 按C列 | B*O或B*O*S | 1 core×1024 threads | element core×UB loop×1024 threads | 连续DMA与随机写权衡 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth1两Queue | depth2 bind+4096B TBuf | 1024B TBuf | depth2两Queue+1024B TBuf | 无通信/dynamic UB |
| 内存总占用与复用 | 两factor互补占UB | `2*loopLength*E+4096` | 1024B | 动态二分`tiling.ubSize`+1024B | 均由Host上界约束 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad | DataCopyPad | asc_vf_call直访GM | DataCopyPad+asc_vf_call | 无NDDMA/通信 |
| 同步、精度与 tail | MTE2_S/MTE2_MTE3 | Queue+MTE2_S | UB DSB/thread guard | Queue+UB DSB | bitwise，无cast |
| fallback/失败边界 | 重复目标未保证 | 切列 | 无更小fallback | 64-bit route | 未知key静默无body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定 Commit 未提供实现接线闭包 | 同 A2 | 公开文档、OpDef、CMake、binary config 全部支持 |
| Host/Launcher | ACLNN 层可见；平台 Host/Tiling/构建不可见 | 同 A2 | `arch35 ScatterTiling`，6 Route |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | AIV_VECTOR/SIMT/Hybrid；AIV-only entry；16MiB system workspace |

## 限制与失败边界

- data/updates 不得为空、rank 必须相同；indices rank 0–2，二维 dim1=2；axis 不能0且归一化后必须合法。
- indices 只允许非负且不得越界；Device 不做越界 guard。重复 indices 的结果公开文档明确不保证，所有随机写 Route 都无 atomic。
- shape 连乘、`size_t` 折叠及多处 int64 乘法没有显式溢出检查；R-005 只保证 packed total 不超过 INT32_MAX，R-006 仍受 int64 地址表达式范围限制。
- entry 对 workspace 或 user workspace 空指针直接 return，未知 key 255 也静默无 body；Host dtype 校验通常阻止未知 tuple。
- ref 输出预置/别名、Contiguous/ViewCopy Kernel 和 executor workspace 聚合属于固定仓外框架边界；固定仓能证明 Scatter schema/ref语义及 body 的局部覆盖，不能给出框架 storage offset。
- A2/A3 虽有公开支持声明，但固定 Commit 只构建 `ascend950`，因此未生成无后缀卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter/docs/aclnnInplaceScatterUpdate.md` | 产品表、原型、约束 | 950公开性、语义、dtype/shape、重复索引 |
| 入口声明/约束 | `index/scatter/op_api/aclnn_scatter_update.h` | 两段式声明 | ABI符号 |
| 入口实现 | `index/scatter/op_api/aclnn_scatter_update.cpp` | `CheckParams/GetWorkspaceSize/aclnnInplaceScatterUpdate` | 前处理与executor图 |
| 入口实现 | `index/scatter/op_api/scatter.cpp` | `ScatterAiCore/Scatter` | L0 launcher与out描述 |
| 入口实现 | `index/scatter/op_graph/scatter_proto.h` | `REG_OP(Scatter)` | ref var语义与I/O schema |
| Host/Launcher | `index/scatter/op_host/scatter_def.cpp` | `Scatter::Scatter` | ascend950注册与dtype |
| Host/Launcher | `index/scatter/op_host/arch35/scatter_tiling.cpp` | `Tiling4Scatter/TilingPrepareForScatter` | AIV核/UB producer与注册 |
| Host/Launcher | `index/scatter/op_host/arch35/scatter_tiling_arch35.h` | `ScatterTilingData/ScatterTiling` | POD声明与Host状态 |
| Host/Launcher | `index/scatter/op_host/arch35/scatter_tiling_arch35.cpp` | `DoOpTiling/GetTilingKey/PostTiling` | 校验、Route、分核、workspace |
| Kernel entry/body | `index/scatter/op_kernel/scatter_apt.cpp` | `scatter` | key router与AIV-only声明 |
| Kernel entry/body | `index/scatter/op_kernel/arch35/scatter.h` | `Scatter/SimtCompute*` | Hybrid、UB与SIMT地址 |
| Kernel entry/body | `index/scatter/op_kernel/arch35/scatter_simd.h` | `ScatterSimd` | 两条DMA route |
| Kernel entry/body | `index/scatter/op_kernel/arch35/scatter_simt.h` | `ScatterSimt` | 小域直接GM SIMT |
| Kernel entry/body | `index/scatter/op_kernel/arch35/scatter_deterministic.h` | `ScatterDeterministic` | 确定性二维索引搬运 |
| 构建接线 | `index/scatter/CMakeLists.txt` | SUPPORT_COMPUTE_UNIT/TILING_DIR | 仅ascend950/arch35 |
| 构建接线 | `index/scatter/op_host/config/ascend950/scatter_binary.json` | Scatter op_list | 950二进制dtype组合 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 32字段声明顺序、Route inactive字段 | 宏展开布局固定仓不可见，不得臆造offset；Host/Device必须共享头 | `ScatterTilingData` |
| 调用顺序 | Contiguous×3→Scatter→ViewCopy | 任一构造失败阻断；第二段必须使用原executor | `aclnnInplaceScatterUpdateGetWorkspaceSize` |
| workspace/内存 | Scatter请求16MiB；各Route UB公式 | workspace必须非空；executor聚合offset不可外推 | `GetWorkspaceSize/scatter` |
| 同步 | MTE2_S、MTE2_MTE3、Queue、UB DSB | scalar读/GM写前不能删除对应等待；SIMT前参数必须可见 | 各Route body |
| 精度与尾块 | bitwise搬运、8B打包、former/tail | 不得加入数值cast；只提交有效tail字节/元素 | `PromoteDtype` 与各 Process |
| 失败边界 | dtype/shape/axis失败、unknown key、越界约束 | 不得把重复索引描述为确定结果；A2/A3不得复用950 body宣称闭包 | `CheckParams/ScatterTiling/scatter` |
