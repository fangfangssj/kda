---
type: Operator
title: "GatherV2 950 ACLNN 算子知识"
description: "说明 aclnnGatherV3 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/docs/aclnnGatherV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/docs/aclnnGatherV3.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_api/aclnn_gather_v3.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_api/aclnn_gather_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_api/gather_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_host/arch35/gather_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_host/arch35/gather_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_host/arch35/gather_v2_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/gather_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simd_two_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simd_last_gather.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_ga_all_load.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simt_two_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simd_pcie.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simd_two_dim_pcie.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel_aicpu/gather_v2_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel_aicpu/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_host/gather_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:37:03Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GatherV2 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnGatherV3GetWorkspaceSize` / `aclnnGatherV3`，公开 ACLNN V3；同族 V2 较旧，950 取 V3。 |
| 计算语义 | 将 `self` 折叠为 `[B,P,G,A]`、`index` 折叠为 `[B,I]`，输出 `[B,P,I,A]`，`out[b,p,i,a]=self[b,p,index[b,i],a]`。 |
| 输入输出 | `self/out` 为 ND 且同 dtype；公开 V3 接受 FLOAT/FLOAT16/BFLOAT16、INT8/16/32/64、UINT8、BOOL、DOUBLE、COMPLEX64/128；`index` 为 INT32/INT64；rank 不超过 8。 |
| 执行类型 | 单节点：AIV_VECTOR、SIMT 或 AICPU；没有 Cube/AIC/MIX_CV 节点。 |
| Kernel 路线 | 空 Tensor 在 API 层零 Kernel 返回；非空为 R-001 二维 SIMD、R-002 末轴全载、R-003 GA 全载、R-004 二维 SIMT、R-005 二维 PCIe SIMD、R-006 通用 SIMD、R-007 通用 SIMT、R-008 通用 PCIe SIMD、R-009 COMPLEX128 AICPU。 |
| 核心资源 | AIV 路线使用 GM、UB `TQue/TQueBind/TBuf` 与寄存器；SIMT 直接 GM，动态本地内存仅由 launch 配置保留；AICPU 使用 CPU 地址空间；公共 Tiling workspace 固定报 16 MiB，但 Device body 不消费 `workspace`。 |
| 来源闭包 | `source_closed`：ACLNN V3→`l0op::GatherV2`→AICore/AICPU selector→950 Tiling/POD→`gather_v2` router→全部可达 body 与构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令归一化后的 `dim=d`、`batchDims=q`。`B=prod(self[0:q])`，`P=prod(self[q:d])`，`G=self[d]`，`A=prod(self[d+1:])`，`I=numel(index)/B`。对合法索引 `k=index[b,i]`，结果为 `out[b,p,i,a]=self[b,p,k,a]`。950 Host 允许负 `batchDims` 并归一化；V3 通过 `l0op::GatherV2(..., batchDims)` 固定 `negative_index_support=false`。AIV/SIMT body 对越界索引写零；AICPU fallback 对负索引只加一次 `G`，且随后只校验 `k<G`、未校验 `k>=0`：`k>=G` 返回参数错误，而原值 `<-G` 时仍可能形成负地址。这是两条执行边界的可见差异与固定源码限制。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，rank 0～8（公开文档写 1～8，源码保留标量分支） | V3 支持表所列 dtype | 非连续输入先 `Contiguous`；标量按 rank 1 参与 axis 校验。 |
| `dim` | Host 标量 | INT64 | `[-rank(self), rank(self)-1]`，负值加 rank；标量用有效 rank 1。 |
| `index` | ND，rank 0～8 | INT32/INT64 | 非连续输入先 `Contiguous`；950 输出 shape 从 `index[batchDims:]` 拼接；每个 batch 前缀须与 `self` 相同。 |
| `batchDims` | Host 标量 | INT64 | 950 支持负值，归一化后必须 `<rank(self)` 且 `<=dim`。 |
| `mode` | Host 标量 | INT64 | 文档定义 0/1/2；950 `IsRegbase()` 分支不把它传给 L0，故不参与 950 Route。小输入会把局部变量改为 0，但仍不消费。 |
| `out` | ND，`self[:dim]+index[batchDims:]+self[dim+1:]` | 与 `self` 相同 | 计算结果经 `ViewCopy` 写入，允许非连续输出。 |

### 入口与版本

公开文档同时列出 `aclnnGatherV2` 与 `aclnnGatherV3`，且两者均公开支持 Ascend 950；V3 增加 `batchDims` 与 `mode`，是该架构实际支持的最新公开版本。`aclnnGatherV3GetWorkspaceSize` 在 950（`IsRegbase()`）直接调用 `l0op::GatherV2`，而不是旧架构的 `GatherV2WithImplMode`。低层 AICore op 名仍是 `GatherV2`，不代表回退到 ACLNN V2。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序是：创建 executor；检查 `self/index/out` 非空、`self==out` dtype、dtype 白名单、rank≤8、dim 范围与输出 shape；950 再检查/归一化 `batchDims` 及 batch 前缀 shape。空 `self` 或空 `index` 直接返回 `workspaceSize=0`，不加入 Kernel。非空时依次对 `self`、`index` 调用 `Contiguous`；源码构造了零维 index 的 `UnsqueezeNd` 结果 `indexParam`，但 950 V3 实际传给 L0 的是 `indexContiguous`，这是固定源码的真实调用。随后 `l0op::GatherV2` 分配连续输出，Host 把归一化 axis 转为 INT64 长度 1 Tensor，并按 dtype 选择 AICore 或 AICPU，最后加入 `ViewCopy`。

### 参数、Workspace 与生命周期

`l0op::GatherV2` 的临时输出由 executor 管理；`Contiguous` 结果、axis Tensor、Gather 输出和 `ViewCopy` 均与 executor 同寿命。Tiling 将 AIV 核数与 UB 大小取自 950 platform/compile info，`PostTiling` 设置 `blockDim=needCoreNum`。`GetWorkspaceSize` 对所有非空 AICore Route 固定写 `16*1024*1024` 字节；`gather_v2` entry 虽接收 `workspace`，全部 body 均不引用它，因此它是框架分配且在本 Kernel 中未消费的保留区。第二段接口由 `CommonOpExecutorRun` 在调用者 stream 上按 executor 顺序运行所有节点。

### 启动图与失败传播

```text
aclnnGatherV3GetWorkspaceSize
  -> Contiguous(self) -> Contiguous(index)
  -> l0op::GatherV2
       -> IsAiCoreSupport
          true: ADD_TO_LAUNCHER_LIST_AICORE(GatherV2)
                -> GatherTiling/Gatherv2TilingBase::DoTiling
                -> gather_v2<<<needCoreNum>>>(x,index,axis,y,workspace,tiling)
          false (A5 API 交集仅 COMPLEX128): ADD_TO_LAUNCHER_LIST_AICPU(GatherV2)
                -> GatherV2CpuKernel::Compute
  -> ViewCopy(gatherResult,out)
aclnnGatherV3 -> CommonOpExecutorRun(..., stream)
```

单次调用只启动一个 Gather 核心节点；`ViewCopy`、`Contiguous` 是允许的框架边界，不是 Gather 算法节点。节点失败会使对应 `CHECK_RET/OP_CHECK` 返回，阻断后续加入或执行；executor 拥有临时 Tensor 和 workspace，stream 顺序保证 Gather 结果在 `ViewCopy` 前可见。

## Route 与参数设计

### 分核与资源计算

Host 先做 dtype 合宽：若尾部连续字节数可整除 8/4/2，则把 `improveDtypeSize` 提升到 8/4/2，并相应缩小逻辑 `innerSize`；这是按字节搬运，不改变结果位模式。`ySize=B*P*I*A`（使用合宽后的元素单位）。SIMT 按输出元素分核，并把每核量对齐 `threadNum`；SIMD 按 `[B,P,I]` 行分核，`maxElement=floor_align((UB-8192),ubBlock)/(2*dtypeSize)`。二维 SIMD 按 `I` 分核；末轴与 GA 路线分别按 `P×I` 维度做二维分核。所有 AICore Route 的 `blockDim=needCoreNum`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 2D SIMD | `1000000299` | `B==1 && P==1` 且 `A*D>=2048`，或 `D=1/2/4` 时分别 `A*D>=218/360/620`；并且 `B*P*I>=aivNum/2` | R-002 |
| 2 | R-002 last gather | `1100000000+D` | `B==1 && A==1 && G*D` 可全载，`D<=8`，`I/G>1`，`I*D>=32`，且 B8/B16 时 `G<65536`、其余 `G<INT32_MAX`；公开 V3 950 的 NIS 固定 false | R-003 |
| 3 | R-003 GA all-load | `3000` | `B==1 && I/G>1`；`32<A*D<2048`；若 `D==8` 则 `align(A*D)*G>512`；扣除 help、双输出区、最小 index 区和一个 `GA` 载入区后 UB 非负；公开 V3 950 的 NIS 固定 false | R-004/R-005 |
| 4 | R-004/R-005 2D SIMT/PCIe | `2000000000+D+(addr64?100:0)` | 前述路线均不命中且 `B==1 && P==1`；所有地址不在 `[0x40000000000,0x80000000000)` 用 SIMT，否则同 key 改走 PCIe SIMD | R-006 |
| 5 | R-006 general SIMD | `1000000099` | 前述二维路线不命中；`(D=1 && A*D>=218)\|\|(D=2 && >=360)\|\|(D=4 && >=620)\|\|(A*D>=2048)`，且 `UB>=16384`、`B*P*I>=aivNum/2` | R-007/R-008 |
| 6 | R-007/R-008 general SIMT/PCIe | `GET_TILINGKEY(addrType,dtypeKey)` | 其余 AICore case；本地地址用 SIMT，任一 I/O/axis 地址落入 PCIe 区间时改走 PCIe SIMD | Tiling 失败 |
| 7 | R-009 AICPU | 无 AICore key | `IsAiCoreSupport==false`；950 V3 API 白名单与 L0 白名单的差集为 COMPLEX128 | 失败 |

```text
IsSimdTwoDim -> 2D SIMD
else if IsLastGatherAndFullLoad -> last-gather SIMD
else if IsGaAllLoad -> GA-all-load SIMD
else if B==1 && P==1 -> 2D SIMT key -> [PCIe address ? PCIe SIMD : SIMT]
else if general SIMD predicate -> general SIMD
else -> general SIMT key -> [PCIe address ? PCIe SIMD : SIMT]
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `1000000299` | 表中二维 SIMD 谓词 | 单 AIV_VECTOR | `Gatherv2SimdTwoDim<DTYPE_INDICES>` | 双缓冲 bind queue + index TBuf，event 保护 | 行/列切分；越界填零 |
| R-002 | `110000000{1,2,4,8}` | 末轴、G 可全载，NIS=false | 单 AIV_VECTOR | `Gatherv2SimdLastGather<X_T,INDICES_T,false>` | input/index/output queues + cast TBuf | `ubRows/ubCols` 双尾块 |
| R-003 | `3000` | GA 路线谓词，NIS=false | 单 AIV_VECTOR | `Gatherv2GaAllLoad<INDICES_T,false>` | X/index TBuf、256B helper、depth=2 output queue | X/index/output 三层尾循环 |
| R-004 | 2D SIMT keys | 2D 且地址非 PCIe | 单 SIMT | `Gatherv2SimtTwoDim<X_T,INDICES_T,uint32/64_t>` | 直接 GM；`dynUB=UB-128KiB`，body 不分配 | thread stride 尾保护 |
| R-005 | 与 R-004 同 key | 任一地址在 PCIe 区间 | 单 AIV_VECTOR | `Gatherv2SimdTwoDimPcie<INDICES_T>` | Device 重新按 `UB-128KiB` 建双缓冲+8KiB index | 行/列切分 |
| R-006 | `1000000099` | general SIMD 谓词 | 单 AIV_VECTOR | `Gatherv2Simd<INDICES_T>` | 双缓冲 bind queue + 8KiB index | 行/列切分 |
| R-007 | generic key | 其余且非 PCIe | 单 SIMT | `Gatherv2<X_T,INDICES_T,uint32/64_t>` | 直接 GM；`dynUB=UB-128KiB` | 每核最后段、thread stride |
| R-008 | 与 R-007 同 key | 其余且 PCIe 地址 | 单 AIV_VECTOR | `Gatherv2SimdPcie<INDICES_T>` | Device 重新计算双缓冲 + 8KiB index | 行/列切分 |
| R-009 | L0 dtype selector | COMPLEX128、非空 | 单 AICPU | `GatherV2CpuKernel::Compute`→`DoGatherV2Compute<complex<double>,Index>` | CPU 循环与 `BiggerMemCpy` | `k>=G` 报错；归一化后 `k<0` 未被显式拒绝 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype 合宽 | 尾部字节可被 8/4/2 整除 | 用较宽整数搬运连续位模式，减少索引和搬运迭代 | 全部 AICore | 保持原 dtype size | `Gatherv2TilingBase::XDtypeImprove` |
| SIMD 起点 | B8/B16/B32 为 218/360/620 字节，通用 2048 字节 | 短行 SIMT 更合适，长行 DMA/SIMD 更合适；仅源码分流机制，未做性能验证 | R-001/R-006 | SIMT/专用路线 | `IsSimdTwoDim`、`DoOpTiling` |
| PCIe 地址 | 任一地址在 `[0x40000000000,0x80000000000)` | router 禁用 direct-GM SIMT，改用显式 DMA 到 UB | R-004/005、R-007/008 | 同 key 另一 body | `gather_v2` |
| GA bank padding | `align(Abytes)==256 or 128` 时再加一个 UB block | 避免 gather bank conflict | R-003 | N/A，Route 内调整 | `IsGaAllLoad`、`GaAllLoadTiling` |
| last-gather DB | UB 公式满足且 `blockFactor>1` | input queue depth 2；否则 depth 1 | R-002 | 同 Route 单缓冲 | `LastGatherUbTiling` |
| 地址位宽 | 总输入/输出超过 `INT32_MAX` | router 实例化 `uint64_t` 地址计算 | R-004/R-007 | `uint32_t` 实例 | `GetTilingKey` |
| UB 本地内存 | SIMT 路线 `SetLocalMemorySize(UB-128KiB)` | 给 dcache/本地内存留保留区 | R-004/R-007 | 其它 Route 不设置 | `PostTiling` |

### POD/ABI

以下结构都由 `BEGIN_TILING_DATA_DEF` 宏生成；字段源码类型/顺序可见，但宏定义在固定仓外部依赖中，故 `sizeof/alignof/offsetof` 精确值固定仓不可见。表中 “ABI” 均指按声明顺序序列化、精确偏移不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `GatherV2TilingData` | `needCoreNum` | int64/64 | 顺序 1 | core | `CalcCoreElement/CalcSimdTiling` | 通用 SIMD/SIMT core guard | >0 |
| 同上 | `threadNum` | int64/64 | 顺序 2 | thread/core | `CalcCoreElement` | SIMT launch | 128～2048（FPGA 分支除外） |
| 同上 | `batchSize` | int64/64 | 顺序 3 | batch | `MargeAxis` | 通用地址公式 | ≥1 |
| 同上 | `outerSize` | int64/64 | 顺序 4 | row group | `MargeAxis` | 通用地址公式 | ≥1 |
| 同上 | `gatherDimSize` | int64/64 | 顺序 5 | element | `MargeAxis` | index bound/x offset | 非空 Route >0 |
| 同上 | `gatherSize` | int64/64 | 顺序 6 | index element/batch | `MargeAxis` | output/index mapping | ≥1 |
| 同上 | `innerSize` | int64/64 | 顺序 7 | 合宽元素 | `MargeAxis` | row bytes/address | ≥1 |
| 同上 | `xSize` | int64/64 | 顺序 8 | 原 dtype element | `MargeAxis` | 仅初始化读取/日志；核心 body不依赖 | 可大于 32 位 |
| 同上 | `indicesSize` | int64/64 | 顺序 9 | element | `MargeAxis` | 通用 SIMD 初始化 | ≥1 |
| 同上 | `ySize` | int64/64 | 顺序 10 | 合宽元素 | `MargeAxis` | 通用 SIMD 初始化 | ≥1 |
| 同上 | `perCoreElements` | int64/64 | 顺序 11 | 输出元素或 `[P,I]` 行 | 各分核函数 | core extent | Route 相关 |
| 同上 | `lastCoreElements` | int64/64 | 顺序 12 | 同上 | 各分核函数 | 尾核/余数 | Route 相关 |
| 同上 | `negativeIndexSupport` | int64/64 | 顺序 13 | bool | `GetAttrsInfoAndCheck` | SIMT/SIMD | V3 950 为 0 |
| 同上 | `supportOutOfBoundIndex` | int64/64 | 顺序 14 | bool | 固定置 1 | 固定仓 body未读取 | inactive |
| 同上 | `maxElement` | int64/64 | 顺序 15 | 合宽元素 | `CalcSimdTiling` | UB容量/列切分 | 对 SIMT inactive |
| 同上 | `indiceUbSize` | int64/64 | 顺序 16 | byte | `CalcSimdTiling` 固定 8192 | index TBuf | R-006 |
| 同上 | `dtypeSize` | int64/64 | 顺序 17 | byte/合宽元素 | 各 tiling | byte offset/模板 key | 1/2/4/8 |
| `GatherV2LastTilingData` | `needCoreNum` | int16/16 | 顺序 1 | core | `LastGatherTiling` | tail core | 窄化要求 core 数≤INT16_MAX |
| 同上 | `indicesNum` | int16/16 | 顺序 2 | queue depth | `LastGatherTiling`=2 | `indexQue_` | 2 |
| 同上 | `splitIndices` | int16/16 | 顺序 3 | bool | 固定 1 | body 未读取 | inactive |
| 同上 | `inputNum` | int16/16 | 顺序 4 | queue depth | `LastGatherUbTiling`=1/2 | `inQueue_` | UB guard |
| 同上 | `splitMode` | int16/16 | 顺序 5 | enum | `indiceUbSize>=gFactor*indexSize ? 0:1` | `Process` | 0/1 |
| 同上 | `coreInCols` | int16/16 | 顺序 6 | core subgroup | `LastGatherTiling` | p/g block mapping | ≥1 |
| 同上 | `inputUbSize` | int32/32 | 顺序 7 | byte/buffer | `LastGatherUbTiling` | input queue | UB block aligned |
| 同上 | `outUbSize` | int32/32 | 顺序 8 | byte/buffer | 同上 | depth=2 output queue | UB block aligned |
| 同上 | `indiceUbSize` | int32/32 | 顺序 9 | byte/buffer | 同上 | index queue | UB block aligned |
| 同上 | `indiceCastUbSize` | int32/32 | 顺序 10 | byte | 同上 | int64→int32 TBuf | 至少 trait 对齐区 |
| 同上 | `ubCols` | int32/32 | 顺序 11 | index column | 同上 | vector cols/output stride | vector-width aligned |
| 同上 | `ubRows` | int32/32 | 顺序 12 | input row | 同上 | input loop | ≥1 |
| 同上 | `gatherDimSize` | int32/32 | 顺序 13 | element | `LastGatherTiling` | gather bound | Route guard `<INT32_MAX` |
| 同上 | `gatherSize` | int64/64 | 顺序 14 | element | 同上 | GM stride | ≥1 |
| 同上 | `blockFactor` | int64/64 | 顺序 15 | P row/core | 同上 | core GM base | ≥1 |
| 同上 | `tailBlockFactor` | int64/64 | 顺序 16 | P row | 同上 | last core | ≥1 |
| 同上 | `gFactor` | int64/64 | 顺序 17 | index/core-group | 同上 | indices/output base | ≥1 |
| `GatherV2GaAllLoadTilingData` | `gSize` | int64/64 | 顺序 1 | element | `GaAllLoadTiling` | X layout/index bound | >0 |
| 同上 | `aSize` | int64/64 | 顺序 2 | byte | 同上 | DMA有效行长 | 33～2047 route guard |
| 同上 | `aSizeAligned` | int64/64 | 顺序 3 | byte | 同上 | UB row stride | UB-block aligned，加 bank padding |
| 同上 | `indicesSize` | int64/64 | 顺序 4 | element | 同上 | output stride | >0 |
| 同上 | `usedCoreNum` | int64/64 | 顺序 5 | core | 同上 | core guard | `pOuter*indicesOuter` |
| 同上 | `indicesOuter` | int64/64 | 顺序 6 | core group | 同上 | block mapping | ≥1 |
| 同上 | `normalCoreIndicesNum` | int64/64 | 顺序 7 | element/core | 同上 | index loop | ≥1 |
| 同上 | `tailCoreIndicesNum` | int64/64 | 顺序 8 | element | 同上 | tail core | ≥1 |
| 同上 | `pOuter` | int64/64 | 顺序 9 | core group | 同上 | p tail判断 | ≥1 |
| 同上 | `normalCoreGaNum` | int64/64 | 顺序 10 | P row/core | 同上 | X loop | ≥1 |
| 同上 | `tailCoreGaNum` | int64/64 | 顺序 11 | P row | 同上 | tail core | ≥1 |
| 同上 | `xBufferSize` | int64/64 | 顺序 12 | byte | 同上 | `xBuf_` | UB-block aligned |
| 同上 | `indicesBufferSize` | int64/64 | 顺序 13 | byte | 同上 | `indicesBuf_` | UB-block aligned |
| 同上 | `yBufferSize` | int64/64 | 顺序 14 | byte/buffer | 同上 | depth=2 `yQueue_` | UB余量的一半 |
| `GatherV2TilingDataSimdTwoDim` | `needCoreNum` | int16/16 | 顺序 1 | core | `SimdTwoDimTiling` | core guard | 窄化边界同上 |
| 同上 | `negativeIndexSupport` | int16/16 | 顺序 2 | bool | 同上 | index fix | V3 为 0 |
| 同上 | `indiceFactor` | int32/32 | 顺序 3 | index element | `8192/indexSize` | index TBuf/loop | >0 |
| 同上 | `dtypeSize` | int32/32 | 顺序 4 | byte | Tiling | byte offsets | 1/2/4/8 |
| 同上 | `gatherDimSize` | int64/64 | 顺序 5 | element | Tiling | bounds | >0 |
| 同上 | `gatherSize` | int64/64 | 顺序 6 | element | Tiling | split | >0 |
| 同上 | `innerSize` | int64/64 | 顺序 7 | 合宽 element | Tiling | row width | >0 |
| 同上 | `blockFactor` | int64/64 | 顺序 8 | index/core | Tiling | start/end | 可为 0 |
| 同上 | `tailBlockFactor` | int64/64 | 顺序 9 | remainder | Tiling | former cores | `<aivNum` |
| 同上 | `maxElement` | int64/64 | 顺序 10 | element/buffer | Tiling | UB split | >0 |
| `GatherV2TilingDataSimtTwoDim` | `needCoreNum` | int16/16 | 顺序 1 | core | `SimtTwoDimTiling` | SIMT/PCIe core guard | 窄化边界 |
| 同上 | `negativeIndexSupport` | int16/16 | 顺序 2 | bool | 同上 | index fix | V3 为 0 |
| 同上 | `threadNum` | int32/32 | 顺序 3 | thread/core | 同上 | SIMT launch | 128～2048 |
| 同上 | `gatherDimSize` | int64/64 | 顺序 4 | element | 同上 | bound | >0 |
| 同上 | `innerSize` | int64/64 | 顺序 5 |合宽 element | 同上 | address | >0 |
| 同上 | `perCoreElements` | int64/64 | 顺序 6 | output element | 同上 | core extent | >0 |
| 同上 | `lastCoreElements` | int64/64 | 顺序 7 | output element | 同上 | tail | >0 |
| 同上 | `gatherSize` | int64/64 | 顺序 8 | element | 同上 | PCIe Device 重分核 | >0 |
| 同上 | `dtypeSize` | int64/64 | 顺序 9 | byte | 同上 | router/template byte width | 1/2/4/8 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | SIMD 2D | `IsSimdTwoDim/SimdTwoDimTiling` | `GatherV2TilingDataSimdTwoDim` | `gather_v2_apt.cpp::gather_v2` | `gather_v2_simd_two_dim.h` | `CMakeLists.txt`, `gather_v2_def.cpp` |
| R-002 | last keys | `IsLastGatherAndFullLoad/LastGatherTiling` | `GatherV2LastTilingData` | 同上 | `gather_v2_simd_last_gather.h` | 同上 |
| R-003 | 3000 | `IsGaAllLoad/GaAllLoadTiling` | `GatherV2GaAllLoadTilingData` | 同上 | `gather_v2_ga_all_load.h` | 同上 |
| R-004/005 | SIMT 2D keys + address selector | `IsSimtTwoDim/SimtTwoDimTiling` | `GatherV2TilingDataSimtTwoDim` | 同上 | `gather_v2_simt_two_dim.h` / `gather_v2_simd_two_dim_pcie.h` | 同上 |
| R-006 | SIMD key | `DoOpTiling/CalcSimdTiling` | `GatherV2TilingData` | 同上 | `gather_v2_simd.h` | 同上 |
| R-007/008 | generic key + address selector | `DoOpTiling/CalcCoreElement` | `GatherV2TilingData` | 同上 | `gather_v2.h` / `gather_v2_simd_pcie.h` | 同上 |
| R-009 | `IsAiCoreSupport=false` | `op_api/gather_v2.cpp` | AICPU context attrs | `GatherV2CpuKernel::Compute` | `InitGatherV2Info/CopyGatherV2Data` | `op_kernel_aicpu/CMakeLists.txt` |

## Kernel 实现

### 公共执行流程

AICore 路线共享 `gather_v2(x,indices,axis,y,workspace,tiling)`，entry 固定声明 `KERNEL_TYPE_AIV_ONLY`，按 TilingKey 解析对应 POD，再实例化唯一 body。所有 body 将输出域分给 `GetBlockIdx()`；SIMD 路线把数据搬到 UB 后写回，SIMT 路线由 `asc_vf_call(dim3(threadNum))` 让 thread 以 `blockDim.x` 为步长直接处理 GM。所有索引在真正读取 X 前检查；本地 AICore 越界输出零。

### Route R-001：二维行搬运 SIMD

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/gather_v2_tiling.cpp` | `IsSimdTwoDim` | `B=P=1`、行字节阈值和规模谓词 | key `1000000299` |
| Router | `op_kernel/gather_v2_apt.cpp` | `gather_v2` | key 命中 | `Gatherv2SimdTwoDim<DTYPE_INDICES>` |

#### Device 数据流

##### 函数级调用链

```text
gather_v2 -> Gatherv2SimdTwoDim::Init -> Process
  -> NoSplitColProcess|SplitColProcess -> GetIndex -> CopyIn/Duplicate -> CopyOut
```

##### 源码等价伪码

```text
split I across cores; cache up to indiceFactor indices in indexBuf
for each output row/chunk:
  k = index[i]; if negative-support then k += G
  valid ? DMA x[k*A + chunk] : Duplicate(0)
  enqueue same bound queue and DMA to y[i*A + chunk]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQueue_` | allocator | `2*maxElement*D` bytes | byte row | CopyIn/Duplicate | CopyOut | pipe, per tile | depth 2、VECIN/VECOUT bind |
| `indexBuf_` | 0 | `indiceFactor*sizeof(index)` = 8192 bytes | index | GM DMA | scalar `GetValue` | pipe, core lifetime | MTE2→S event |
| GM y | `(i*A+chunk)*D` | valid bytes only | ND byte view | CopyOut | caller | caller | tail `cols` |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  rows["inQueue bind · depth=2 · 2*maxElement*D"]:2
  idx["indexBuf · indiceFactor*sizeof(index)=8192"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*maxElement*D + 8192
maxElement = floor_align((UB-8192),ubBlock)/(2*D)
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bind queue slot | CopyIn/Duplicate | CopyOut | 下一 tile | EnQue/DeQue/FreeTensor，V→MTE3 event | depth=2 |
| 最近 3 行 | CopyIn | `Copy` 重复行 | 同 tile 后续行 | MTE2→V event | xIndex 与缓存槽匹配 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| index | MTE2→S | DMA | scalar load | indexBuf | 无 cast |
| output | V→MTE3 | Copy/Duplicate | DataCopyPad | bound queue | 位拷贝；越界填零；最后 chunk 有效字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `SimdTwoDimTiling` | 分核和 UB 公式 |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | `SIMD_TILING_KEY_TWO_DIM` | POD→body |
| Device body/helper | `op_kernel/arch35/gather_v2_simd_two_dim.h` | `Process/NoSplitColProcess/SplitColProcess` | 行缓存、重复行复用、tail |

### Route R-002：末轴全载 Vector Gather

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/gather_v2_tiling.cpp` | `IsLastGatherAndFullLoad` | `B=1,A=1` 且 G 可入 UB、I/G>1 等 | router 定义 8 个 dtype/NIS key；公开 V3 实际只产出 4 个 NIS=false key |
| Router | `op_kernel/gather_v2_apt.cpp` | last-gather branches | `D∈{1,2,4,8}` 与 NIS | 对应 `Gatherv2SimdLastGather` |

#### Device 数据流

##### 函数级调用链

```text
Init -> Process -> SplitIndicesProcess|NoSplitIndicesProcess
 -> CopyInX + CopyInIndices -> [int64 Cast to int32] -> ConvertNegIndices
 -> GatherComputeMultiRows -> MicroAPI::DataCopyGather -> CopyOut
```

##### 源码等价伪码

```text
map block to (pBlock,gBlock); load ubRows full G rows
load ubCols indices (or cache all); int64 index casts to int32
for each vector register of indices and each resident row:
  gatherMask = (0 <= index < G)
  DataCopyGather(xRow,index,gatherMask); store only valid output lanes
DMA rows*cols with src/dst strides; repeat over row/column tails
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQueue_` | p row block | `inputNum*inputUbSize` | X | GM DMA | gather | pipe | depth 1/2 |
| `indexQue_` | gBlock range | `indicesNum*indiceUbSize` | INT32/64 | GM DMA | cast/gather | pipe | depth 2 |
| `indexBuf_` | 0 | `indiceCastUbSize` | INT32 | Cast | gather | pipe | int64 only materially used |
| `outQueue_` | allocator | `2*outUbSize` | X | Vector gather | GM DMA | pipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["inQueue · inputNum*inputUbSize"]:1
  iq["indexQueue · indicesNum*indiceUbSize"]:1
  cast["indexBuf · indiceCastUbSize"]:1
  yq["outQueue · 2*outUbSize"]:1
```

##### 字节账与复用顺序

```text
UB_total = inputNum*inputUbSize + indicesNum*indiceUbSize + indiceCastUbSize + 2*outUbSize
UB_total < platform UB（Host LastGatherUbTiling 的选择式）
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input slot | CopyInX | all column tiles for those rows | next row tile | DeQue/FreeTensor | column inner loop完成 |
| indexBuf | Cast/ConvertNegIndices | gather loops | next index tile | queue ordering | int64 input先 Cast |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| index cast | `Cast(...CAST_NONE)` | INT64 index queue | gather | INT32 TBuf | Route guard `G<INT32_MAX` |
| gather | mask + packed store | index/x registers | output queue | valid lanes | B8 uses packed B16 store；out-of-range mask不取 X |
| output | queue | gather | DataCopyPad | row/col tail | `srcStride/dstStride` 由 `ubCols/gatherSize` 给出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `LastGatherUbTiling/LastGatherTiling` | DB与二维分核 |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | last-gather key branches | dtype/NIS specialization |
| Device body/helper | `op_kernel/arch35/gather_v2_simd_last_gather.h` | `LoadIndices/GatherComputeMultiRowsVF` | index mask、cast、VGather、tail |

### Route R-003：GA 全载与向量索引变换

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/gather_v2_tiling.cpp` | `IsGaAllLoad/GaAllLoadTiling` | 中等 A 字节、B=1、复用度与 UB 余量；NIS=false | 3000 |
| Router | `op_kernel/gather_v2_apt.cpp` | GA branches | NIS key | `Gatherv2GaAllLoad` |

#### Device 数据流

##### 函数级调用链

```text
Init -> InitializationX -> [xLoop: xProcess -> indicesLoop: IndicesProcess -> FixIndices -> yProcess]
 -> GatherProcessWithGA2|GatherProcessVfWithGA|GatherProcess -> CopyOutY
```

##### 源码等价伪码

```text
split block into pIndex and indicesIndex
zero sentinel row in each GA tile; DMA G rows with padded A stride
load index tile; optionally add G for negative; invalid -> -1
transform index to (index+1)*aSizeAligned so invalid maps to zero sentinel
vector gather/copy one or multiple P rows into depth-2 yQueue; strip padding on copyout
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `xBuf_` | P tile | `xBufferSize` | byte，`[P,G+1,Aaligned]` | zero+DMA | gather | pipe | sentinel + padding |
| `indicesBuf_` | index tile | `indicesBufferSize` | index→in-place INT32 offsets | DMA/FixIndices | gather | pipe | MTE2→V |
| `tmpIndexBuf_` | 0 | 256 B | UINT32 | GenIndexBuf | GA2 gather | pipe | helper pattern |
| `yQueue_` | allocator | `2*yBufferSize` | byte padded rows | vector gather | CopyOutY | pipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xb["xBuf · xBufferSize"]:1
  ib["indicesBuf · indicesBufferSize"]:1
  hb["tmpIndexBuf · 256B"]:1
  yb["yQueue · 2*yBufferSize"]:1
```

##### 字节账与复用顺序

```text
UB_total = xBufferSize + indicesBufferSize + 256 + 2*yBufferSize
yBufferSize = (UB-256-indicesBufferSize-xBufferSize)/2
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xBuf | InitializationX/CopyInX | all index tiles for current P tile | next xLoop | MTE3→MTE2 与 MTE2→V events | y tiles已写出 |
| indicesBuf | CopyInIndices/FixIndices | yProcess | next index tile | MTE2→V | consumer完成 |
| yQueue slot | gather | CopyOutY | next y tile | EnQue/DeQue/FreeTensor | depth=2 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| x/index reuse | MTE3→MTE2 | prior copyout | next input DMA | shared TBuf | 防止提前覆盖 |
| DMA→Vector | MTE2→V | X/index DMA | Fix/gather | x/index buffers | index INT64 低 32 位用于变换，Route 保证 G 可表示 |
| invalid index | sentinel row | FixIndices | gather | X row 0 sentinel | 输出确定为零 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `GaAllLoadTiling` | P×I 分核、buffer 账 |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | `3000` | `NIS=false` specialization |
| Device body/helper | `op_kernel/arch35/gather_v2_ga_all_load.h` | `FixIndicesVF/GatherProcessWithGA2VF/Process` | sentinel、GA gather、DMA loop mode |

### Route R-004：二维本地 GM SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/gather_v2_tiling.cpp` | `IsSimtTwoDim` | `B=P=1` 且优先路线未命中 | SIMT 2D key |
| Router | `op_kernel/gather_v2_apt.cpp` | address branch | 全部地址非 PCIe | `Gatherv2SimtTwoDim` |

#### Device 数据流

##### 函数级调用链

```text
gather_v2 -> Gatherv2SimtTwoDim::Process -> asc_vf_call<GatherSimt<NIS>>(threadNum)
```

##### 源码等价伪码

```text
for lane=threadIdx.x; lane<coreElements; lane+=blockDim.x:
  yIndex=coreBase+lane; i=fast_div(yIndex,A); a=yIndex-i*A
  k=index[i]; optional negative fix
  y[yIndex] = valid(k) ? x[k*A+a] : 0
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/index/y | 伪码公式 | 每线程 1 element | 合宽 X/index | GM load | thread/store | caller | 无 UB buffer |

##### 分层内存布局

N/A：SIMT body 直接访问 GM；Host 设置 local memory 为 `UB-128KiB`，源码 body 未显式分配该区。

##### 字节账与复用顺序

```text
UB_total = 0（显式）
dynamic_ub/local_memory = UB-128KiB
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | thread 独占输出元素 | core extent不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread loop | 无 barrier | GM load | same thread store | 单一 y 元素 | fast division；尾部 `lane<coreElements` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `SimtTwoDimTiling` | thread/core 配置 |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | SIMT 2D branch | PCIe runtime guard |
| Device body/helper | `op_kernel/arch35/gather_v2_simt_two_dim.h` | `GatherSimt/Process` | thread mapping、越界零 |

### Route R-005：二维 PCIe 显式 DMA

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `op_kernel/gather_v2_apt.cpp` | SIMT 2D key branch | 任一地址落入 PCIe 区间 | `Gatherv2SimdTwoDimPcie` |

#### Device 数据流

##### 函数级调用链

```text
Init(Device 重算分核/UB) -> Process -> NoSplitColProcess|SplitColProcess
 -> GetIndex -> DataCopyPad/Duplicate -> CopyOut
```

##### 源码等价伪码

与 R-001 的行/列搬运相同，但分核与 `ubAvailable` 在 Device 用 `GetBlockNum/GetUBSizeInBytes-128KiB` 重算，避免 PCIe 地址由 SIMT 直接解引用。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| bind queue | allocator | `2*ubAvailable*D` | byte | DMA/zero | DMA out | pipe | depth 2 |
| index TBuf | 0 | 8192 B | index | PCIe/GM DMA | scalar | pipe | MTE2→S |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  rows["bind queue · 2*ubAvailable*D"]:2
  idx["indexBuf · 8192B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*floor_align((UB-128KiB-8192),block)/(2) + 8192
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue/index | DMA | CopyOut/scalar | next tile | queue + MTE2→S/V→MTE3 | 与 R-001 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| PCIe DMA | `DataCopyPad` + events | remote/GM | Vector/scalar | queue/index | 位拷贝、越界零、列 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `SimtTwoDimTiling` | 共享 POD |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | PCIe address guard | 同 key 换 body |
| Device body/helper | `op_kernel/arch35/gather_v2_simd_two_dim_pcie.h` | `Init/Process` | Device 资源重算与 DMA |

### Route R-006：通用行搬运 SIMD

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/gather_v2_tiling.cpp` | `DoOpTiling/CalcSimdTiling` | 通用 SIMD 阈值且工作量≥半数 AIV | `1000000099` |

#### Device 数据流

##### 函数级调用链

```text
Gatherv2Simd::Process -> NoSplitColProcess|SplitColProcess -> GetIndex -> CopyIn/Copy/Duplicate -> CopyOut
```

##### 源码等价伪码

按 `[B,P,I]` 行分核，解码 `batchI/outerI/gatherI`，缓存 8KiB indices；合法行从 `((b*P+p)*G+k)*A` DMA，最近三个相同 xIndex 直接从当前 UB tile 复制，越界清零，再写 `((b*P+p)*I+i)*A`。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| bind queue | allocator | `2*maxElement*D` | byte rows | CopyIn/Copy/zero | CopyOut | pipe | depth 2 |
| index TBuf | batch/index offset | 8192 B | INT32/64 | DMA | scalar | pipe | refill按 batch 边界 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  rows["bind queue · 2*maxElement*D"]:2
  idx["indexBuf · 8192B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*maxElement*D + 8192 <= UB
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 最近三行 | DMA | UB `Copy` | tile结束 | MTE2→V | 仅相同 xIndex |
| queue slot | row producer | CopyOut | next tile | queue+V→MTE3 | depth=2 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/Vector/output | MTE2→S、MTE2→V、V→MTE3 | input/index | scalar/Copy/CopyOut | buffers | 纯位搬运；行/列 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `CalcSimdTiling` | 行分核、UB公式 |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | `SIMD_TILING_KEY` | `Gatherv2Simd` |
| Device body/helper | `op_kernel/arch35/gather_v2_simd.h` | `GetIndex/NoSplitColProcess/SplitColProcess` | index缓存、最近三行复用 |

### Route R-007：通用本地 GM SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/gather_v2_tiling.cpp` | `DoOpTiling/CalcCoreElement` | 其余 AICore case | generic key |
| Router | `op_kernel/gather_v2_apt.cpp` | generic branches | 地址非 PCIe | `Gatherv2<X_T,INDICES_T,U>` |

#### Device 数据流

##### 函数级调用链

```text
Gatherv2::Process -> GetUintDivMagicAndShift(x3) -> asc_vf_call<GatherSimt<NIS>>(threadNum)
```

##### 源码等价伪码

```text
for lane in core output elements:
  decode [b,p,i,a] with three magic divisions
  k=index[b,i]; optional negative fix
  xIndex=((b*P+p)*G+k)*A+a
  y = valid(k) ? x[xIndex] : 0
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/index/y | 公式如上 | 1 element/thread iteration | X/index | GM | thread | caller | 32/64-bit address specialization |

##### 分层内存布局

N/A：显式 UB 为 0；Host 仅设置 `local_memory=UB-128KiB`。

##### 字节账与复用顺序

```text
UB_total = 0（显式）
dynamic_ub/local_memory = UB-128KiB
workspace_total = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | 每线程独立 y | core range不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | GM | GM | 单输出元素 | thread-stride尾保护，越界零 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | `CalcCoreElement/GetTilingKey` | thread/core与地址位宽 |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | generic keys | dtype/address实例 |
| Device body/helper | `op_kernel/arch35/gather_v2.h` | `GatherSimt/Process` | 通用四维映射 |

### Route R-008：通用 PCIe 显式 DMA

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `op_kernel/gather_v2_apt.cpp` | generic key address branch | 任一地址在 PCIe 区间 | `Gatherv2SimdPcie` |

#### Device 数据流

##### 函数级调用链

```text
Gatherv2SimdPcie::Init(Device 重算资源) -> Process
 -> NoSplitColProcess|SplitColProcess -> GetIndex -> DMA/Copy/zero -> CopyOut
```

##### 源码等价伪码

与 R-006 相同的 `[B,P,I]` 行映射和最近三行复用；区别是 Device 扣除 128KiB dcache 后自行重算 `aivNum/blockFactor/maxElement`，并强制通过 `DataCopyPad` 访问 PCIe 区间。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| bind queue | allocator | `2*maxElement*D` | byte | DMA/Copy/zero | DMA out | pipe | depth 2 |
| index TBuf | 0 | 8192 B | index | DMA | scalar | pipe | event protected |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  rows["bind queue · 2*maxElement*D"]:2
  idx["indexBuf · 8192B"]:1
```

##### 字节账与复用顺序

```text
maxElement=floor_align(UB-128KiB-8192,block)/(2*D)
UB_total=2*maxElement*D+8192
workspace_total=16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue/index/recent rows | DMA | Copy/CopyOut | next tile | events+queue | 同 R-006 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| PCIe搬运 | MTE2→S/V、V→MTE3 | DMA | scalar/vector/output | UB | 位拷贝、越界零、行/列 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/gather_v2_tiling.cpp` | general SIMT Tiling | 共享 POD |
| Kernel entry/router | `op_kernel/gather_v2_apt.cpp` | PCIe address guard | 同 key 换 SIMD body |
| Device body/helper | `op_kernel/arch35/gather_v2_simd_pcie.h` | `Init/GetIndex/Process` | Device资源重算、PCIe DMA |

### Route R-009：COMPLEX128 AICPU fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0 | `op_api/gather_v2.cpp` | `IsAiCoreSupport` | 950 AICore列表不含 COMPLEX128 | `GatherV2AiCPU` |
| AICPU | `op_kernel_aicpu/gather_v2_aicpu.cpp` | `IndicesCompute` | X=COMPLEX128，index=INT32/64 | 模板计算 |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICPU -> GatherV2CpuKernel::Compute
 -> GetInputAndCheck -> IndicesCompute<Index>
 -> DoGatherV2Compute<complex<double>,Index>
 -> InitGatherV2Info -> CopyGatherV2Data -> BiggerMemCpy
```

##### 源码等价伪码

```text
for b in B, p in P, i in I:
  k=index[b,i]; if k<0 then k+=G
  require k<G else PARAM_INVALID
  memcpy(y[(b*P+p)*I+i], x[(b*P+p)*G+k], A*sizeof(complex<double>))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| CPU input/output | 伪码公式 | `slice_size=A*sizeof(T)` | ND contiguous | `BiggerMemCpy` | output | framework | 无 AIV UB |

##### 分层内存布局

N/A：AICPU 路线不使用 AIV UB/TQue；使用 AICPU tensor data pointers。

##### 字节账与复用顺序

```text
UB_total = N/A（AICPU）
dynamic_ub = N/A
workspace_total = executor/AICPU task framework 决定，固定仓未给出独立 slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | memcpy | output | N/A | 顺序 CPU 循环 | 每个 slice不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU copy | 顺序循环 | input slice | output | slice | 纯位拷贝；负 index 只加一次 G；仅显式校验 `k<G` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_api/gather_v2.cpp` | `GatherV2/GatherV2AiCPU` | dtype selector与 AICPU launcher |
| Kernel entry/router | `op_kernel_aicpu/gather_v2_aicpu.cpp` | `GatherV2CpuKernel::Compute/IndicesCompute` | index与X dtype分派 |
| Device body/helper | 同上 | `CopyGatherV2Data` | 三重循环、地址、memcpy、错误边界 |

## 算子特定优化

### Trick T-001：按尾部字节整除进行 dtype 合宽

#### 触发与源码依据

- 适用 Route：R-001～R-008。
- 触发条件：原 dtype 小于 8/4/2 字节，且 `innerSize*originalDtypeSize` 能被该宽度整除，按 8→4→2 优先。
- Golden 来源：[`index/gather_v2/op_host/arch35/gather_v2_tiling.cpp:312-333`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_host/arch35/gather_v2_tiling.cpp#L312-L333)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if ((xDtypeSize < INPUT_DTYPE_B64) && (lastAxisBytes % INPUT_DTYPE_B64) == 0) {
  improveDtypeSize_ = INPUT_DTYPE_B64;
  return INPUT_DTYPE_B64;
}
if ((xDtypeSize < INPUT_DTYPE_B32) && (lastAxisBytes % INPUT_DTYPE_B32) == 0) {
  improveDtypeSize_ = INPUT_DTYPE_B32;
  return INPUT_DTYPE_B32;
}
```

#### 机制与设计原因

Host 同时放大搬运元素宽度并把 `innerSize` 除以相同比例，保持每行字节数和 GM 地址不变。router 再按 `D` 选择 int8/16/32/64 body，因此一次循环搬更多连续字节，且无需数值 Cast。

#### 迁移边界

- 适用：只需位拷贝、尾部连续且满足整除的 gather/切片。
- 前提：不改变对齐后地址语义；模板类型只作 byte carrier。
- 失效：尾部字节不能整除更宽 dtype 时退到原宽度；数值计算算子不能照搬。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若只改 `dtypeSize` 而不同比例缩小 `innerSize`，会改变每行字节范围和索引地址；固定仓代码成对更新两者。

### Trick T-002：越界索引映射到 UB 零哨兵

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：GA 全载路线；`FixIndicesVF` 把非法 index 置 -1，再统一执行 `(index+1)*aSizeAligned`。
- Golden 来源：[`index/gather_v2/op_kernel/arch35/gather_v2_ga_all_load.h:458-474`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_ga_all_load.h#L458-L474)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::Duplicate(tmpReg, int32_t(-1), fixMask);
Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesReg, tmpReg, fixMask);
AscendC::MicroAPI::Adds(indicesReg, indicesReg, 1, preg);
AscendC::MicroAPI::Muls(indicesReg, indicesReg, aSizeAligned, preg);
```

#### 机制与设计原因

X 的每个 `[G,A]` tile 前留一条已清零、按 `aSizeAligned` 对齐的哨兵行。非法 index 经 `-1→0` 偏移后落到哨兵，合法 index 经 `+1` 落到真实行；这样 VGather 主循环无需为每个 index 分支，padding 与输出零语义同时闭合。

#### 迁移边界

- 适用：越界输出有确定零值、输入可在 UB 多留一行的间接寻址。
- 前提：所有非法 index 必须先规范为 -1，哨兵在每次覆盖 X buffer 前重新清零。
- 失效：UB 不足或越界应报错时不能采用，转其它 Route/AICPU 语义。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不预留/清零哨兵却仍执行 `index+1` 会把非法索引读成未初始化数据；固定仓 `InitializationX` 明确先清零该子区。

### Trick T-003：同一 tile 最近三行复用

#### 触发与源码依据

- 适用 Route：R-001、R-006、R-008 及其二维 PCIe 对应体。
- 触发条件：当前合法 `xIndex` 与本 tile 最近三个已 DMA 的行起点之一相同。
- Golden 来源：[`index/gather_v2/op_kernel/arch35/gather_v2_simd.h:214-230`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/gather_v2/op_kernel/arch35/gather_v2_simd.h#L214-L230)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (unlikely(xIndex == preIdx0 || xIndex == preIdx1 ||xIndex == preIdx2)) {
  int32_t backStep = (xIndex == preIdx0) ? backOffset0 : (xIndex == preIdx1) ? backOffset1 : backOffset2;
  Copy(xLocal[j * colsAlign], xLocal[backStep], tilingData_->innerSize * tilingData_->dtypeSize);
} else {
  CopyIn(xLocal[j * colsAlign], xGm_, offset, 1, tilingData_->innerSize * tilingData_->dtypeSize);
}
```

#### 机制与设计原因

聚集索引重复时，把已在当前 UB tile 中的完整行复制到新行槽，避免再次从 GM/PCIe 读取。`MTE2→V` event 保证先前 DMA 对 UB `Copy` 可见；该机制只在同一已分配 tile 内复用，不跨 tile 保存地址。

#### 迁移边界

- 适用：索引有局部重复且输出行不重叠的 gather。
- 前提：缓存 key 必须是完整源行起点，行容量按 `colsAlign` 预留。
- 失效：跨 tile 或分列时当前子区不包含整行，必须重新 DMA。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若只比较 index 而忽略 batch/outer 形成的完整 `xIndex`，可能跨源行错误复用；固定仓比较的是最终 `xIndex`。

## 模板与实现对比

| 对比维度 | R-001～R-003 专用 AIV | R-004～R-008 通用/PCIe | R-009 AICPU | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 长二维、末轴高复用、中等 A 高复用 | 二维/通用、SIMD阈值、PCIe地址 | COMPLEX128 | Host shape/UB优先，router再看地址 |
| 模板与 dtype | byte宽度/NIS/index specialization | 32/64地址、index、byte宽度 | C++ `complex<double>`+index | 语义相同但介质不同 |
| 循环与分核 | 行/列或 P×I | thread-stride或行/列 | B×P×I 顺序 | 工作集与并行度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | route-specific UB与 RegTensor | SIMT 直接GM；PCIe用双缓冲 | N/A | 地址可达性与工作集 |
| 内存总占用与复用 | 精确见各 Route | `UB-128KiB` 保留或双缓冲 | CPU memory | Host/Device各自资源 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、DataCopyGather、Copy | SIMT direct load/store或DMA | BiggerMemCpy | 无 NDDMA/通信 |
| 同步、精度与 tail | queue/events/masks | thread guard或events | 顺序循环 | 均为位拷贝，无浮点计算 |
| fallback/失败边界 | 不命中继续下一 Route | AICore最终通用 Route | 不支持 dtype/越界报错 | selector闭合 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 V3 支持，但本卡不覆盖 | 公开 V3 支持，但本卡不覆盖 | 公开 V3 支持且 source_closed |
| Host/Launcher | V3 限制 `dim=0,batchDims=0,mode=1`；固定仓可见 API/L0 selector | 同 A2 | `IsRegbase()` 支持通用 dim/batchDims，mode不进入 L0 |
| Kernel/资源 | A2/A3 AICore Tiling/Kernel 未在固定仓闭合，故未发布无后缀卡 | 同 A2 | arch35 AIV/SIMT/PCIe 与 AICPU fallback 全闭合 |

## 限制与失败边界

- 950 最新入口为 V3；A2/A3 也公开 V3，但固定仓的活动 CMake 只编译 `ascend950/mc62` 与 `arch35`，旧架构 AICore body/tiling 不可见，所以 `gather_v2.md` 未生成。
- `G==0` 的 empty key 在 Tiling/router 中存在，但公开 V3 先由 `HasEmptyTensor` 以 workspace 0 返回，故不属于 ACLNN 可达 Route；同理 `IsAfterGdimFullLoad` 要求 `B=P=1`，却排在无条件接收该二维条件的 `IsSimtTwoDim` 之后，固定决策树中不可达。两者只保留为审计结论，未进入 Route 正文。
- NIS=true 的 last-gather/GA key 在 router 中存在，但 `aclnnGatherV3` 950 调用 `l0op::GatherV2` 时采用默认 `negativeIndexSupport=false`，故正向公开链路只包含 NIS=false key。
- 950 V3 的 `mode` 在 API 文档中有 0/1/2，但 Regbase 分支不传入 L0；不可把旧架构 impl-mode 行为推断到 950。
- AIV/SIMT body 对越界写零；AICPU 对负 index 只加一次 `G`，且固定源码仅检查归一化值 `<G`：`k>=G` 报错，原值 `<-G` 后仍为负数却未被显式拒绝，存在形成负源地址的风险。调用者若依赖越界行为，必须保留执行路线差异，且不得把 AICPU 负越界描述成已完整校验。
- V3 Markdown 参数表只列到 COMPLEX64，公开头文件注释与 API 实现白名单还列出 COMPLEX128、BFLOAT16；本卡按可执行入口实现闭包保留二者，并显式记录这一公开文档差异。
- POD 宏的精确 `sizeof/alignof/offsetof` 在固定仓依赖边界外不可见；字段声明顺序、类型、producer 与 consumer 已闭合。
- `workspace=16MiB` 是 Tiling 固定返回，当前 entry/body 不消费；不得擅自删除，因为 executor ABI 和框架可能依赖该报告值。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 版本族审计 | `index/gather_v2/docs/aclnnGatherV2.md` | 产品表/函数原型 | 同族旧版 V2 的公开性 |
| 入口声明/约束 | `index/gather_v2/docs/aclnnGatherV3.md` | 产品表/函数原型 | 950公开性、V3参数 |
| 入口声明/约束 | `index/gather_v2/op_api/aclnn_gather_v3.h` | `aclnnGatherV3*` | API ABI与计算图 |
| 入口实现 | `index/gather_v2/op_api/aclnn_gather_v3.cpp` | `CheckParams/aclnnGatherV3GetWorkspaceSize` | 950前处理与节点顺序 |
| 入口实现 | `index/gather_v2/op_api/gather_v2.cpp` | `IsAiCoreSupport/GatherV2` | AICore/AICPU分发 |
| Host/Tiling/Launcher | `index/gather_v2/op_host/arch35/gather_v2_tiling.h` | PODs/`Gatherv2TilingBase` | ABI字段与模板 |
| Host/Tiling/Launcher | `index/gather_v2/op_host/arch35/gather_v2_tiling.cpp` | `DoOpTiling/GetTilingKey/PostTiling` | Route、资源、workspace |
| Host/Tiling/Launcher | `index/gather_v2/op_host/arch35/gather_v2_tiling_arch35.cpp` | `GatherTiling/IMPL_OP_OPTILING` | 注册与 platform parse |
| Kernel entry/body | `index/gather_v2/op_kernel/gather_v2_apt.cpp` | `gather_v2` | key/router/PCIe guard |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2.h` | `Gatherv2` | 通用 SIMT |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_simd.h` | `Gatherv2Simd` | 通用 SIMD |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_simd_two_dim.h` | `Gatherv2SimdTwoDim` | 二维 SIMD |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_simd_last_gather.h` | `Gatherv2SimdLastGather` | 末轴 VGather |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_ga_all_load.h` | `Gatherv2GaAllLoad` | GA全载/哨兵 |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_simt_two_dim.h` | `Gatherv2SimtTwoDim` | 二维 SIMT |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_simd_pcie.h` | `Gatherv2SimdPcie` | 通用 PCIe DMA |
| Kernel entry/body | `index/gather_v2/op_kernel/arch35/gather_v2_simd_two_dim_pcie.h` | `Gatherv2SimdTwoDimPcie` | 二维 PCIe DMA |
| Kernel entry/body | `index/gather_v2/op_kernel_aicpu/gather_v2_aicpu.cpp` | `GatherV2CpuKernel` | COMPLEX128 fallback |
| 构建接线 | `index/gather_v2/op_kernel_aicpu/CMakeLists.txt` | `add_aicpu_cust_kernel_modules/target_sources` | AICPU body 编译接线 |
| 构建接线 | `index/gather_v2/op_host/gather_v2_def.cpp` | `AICore().AddConfig("ascend950")` | 950 opFile 接线 |
| 构建接线 | `index/gather_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950/arch35活动构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | shape折叠与多 POD 分路线 | 字段顺序、位宽、单位；SIMT/PCIe共用POD但body资源重算 | `gather_v2_tiling.h` |
| 调用顺序 | Contiguous→Gather→ViewCopy | 950必须调用 `GatherV2` 而非 impl-mode 入口 | `aclnn_gather_v3.cpp` |
| workspace/内存 | 各 Route UB公式与固定16MiB workspace | depth乘数、128/32KiB保留、workspace owner | `PostTiling`、各 `Init` |
| 同步 | queue 与 MTE2/V/MTE3 events | index scalar读取、UB Copy、输出DMA前等待 | SIMD/GA body |
| 精度与尾块 | 纯位搬运，int64 index仅专用路线可安全窄化 | 越界零/AICPU错误差异，former/tail core与行列tail | 各 `Process` |
| 失败边界 | Host校验、Tiling失败、AICPU越界错误 | 不得把公开支持等同于旧架构 source_closed | API/Tiling/AICPU symbols |

- [GetBlockNum](../../../../apis/simd_api/basic_api/tool_interface/system_resources_and_variables/get_block_num.md)
- [GetUBSizeInBytes](../../../../apis/simd_api/basic_api/tool_interface/system_resources_and_variables/get_ub_size_in_bytes.md)
