---
type: Operator
title: "MoeDistributeDispatch 950 ACLNN 算子知识"
description: "说明 aclnnMoeDistributeDispatchV4 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v3/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v3/op_host/op_tiling/arch35/moe_distribute_dispatch_v3_tiling_a5.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v3/op_kernel/moe_distribute_dispatch_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v3/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:28:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MoeDistributeDispatch 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMoeDistributeDispatchV4GetWorkspaceSize/aclnnMoeDistributeDispatchV4`；公开 ACLNN 两段式入口 |
| 计算语义 | 按专家路由扩展 token，可选多种低精度/动态量化，经 EP AllToAllV 分发并生成 combine 元数据 |
| 输入输出 | ND `x[BS,H]`、`expertIds[BS,K]`；七个输出为 token、scale、assist 和计数 |
| 执行类型 | 每条 Route 均为单 Kernel `HYBRID(AIV_VECTOR + 非 Cube HCCL window/MTE 通信)`；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 MTE fullmesh_v1 状态窗；R-002 MTE fullmesh_v2 DataAsFlag |
| 核心资源 | Mc2 context GM window、GM workspace、UB/TQue/TBuf；无 L1/L0/Cube |
| 来源闭包 | source_closed：V4 公开入口经 base 创建 context，闭合到 950 Tiling、V3 entry、复用的 V2 body 与构建接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对每个有效 `(token,k)`，根据 `expertIds` 计算目标专家/EP rank，重排并通过 Mc2 window 发送；接收端压紧为 `expandX`，并输出 `[源EP rank, 源token索引, k]` 等 combine 辅助信息。非量化路径保留输入低精度；量化路径按 `quantMode` 和 `scales` specialization 生成目标 INT8/FP8/FP4 数据及尺度。固定仓公开文档规定的运算次序优先于未公开内部模板分支。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[BS,H]`/ND | FP16/BF16 及 950 文档允许低精度 | 必选，950 common Tiling 检查 `1024≤H≤8192` |
| `expertIds` | `[BS,K]`/ND | INT32 | 必选，`1≤K≤16`；fullmesh_v2 `K≤12` |
| `scalesOptional` | ND | FP32 | smooth/量化尺度；存在性形成 `ScaleMode` |
| `xActiveMaskOptional` | `[BS]` 或 `[BS,K]` | BOOL | fullmesh_v1 可消费；fullmesh_v2 受限 |
| `expertScalesOptional` | 2D/ND | FP32 | public fullmesh 不使用 hierarchy expert scale 语义 |
| `elasticInfoOptional` | 1D/ND | INT32 | 950 V4 public fullmesh 路线不支持 |
| `performanceInfoOptional` | 1D/ND | INT64 | base 在 950 强制传空 |
| `groupEp` | string | char | 必选 EP group；base 据此创建 `Mc2Context` |
| `groupTp/tpWorldSize/tpRankId` | string/scalar | char/INT64 | 950 V4 public路线不支持 TP，Tiling key固定 `HasTp=false` |
| 专家属性 | scalar | INT64 | `moeExpertNum`、shard/shared/zero/copy/const；公开约束逐项校验 |
| `quantMode` | scalar | INT64 | 950 文档支持 0..4；决定输出 dtype、ScaleMode 和 body 模板 |
| `globalBs/expertTokenNumsType` | scalar | INT64 | window 上界；0=cumsum、1=count |
| `commAlg` | string | char | 公开范围仅空、`fullmesh_v1`、`fullmesh_v2` |
| 七个输出 | ND | 文档规定的低精度/FP32/INT32/INT64 | expandX、dynamicScales、assistInfo、expertTokenNums、EP/TP counts、expandScales |

### 入口与版本

V4 README/公开文档的产品表覆盖 Ascend 950；v3 V5 README 明确 Ascend 950 不支持，因此 950 最新公开版本仍是 V4。V4 cpp 转发到 base；base 在 950 public MTE 分支调用 `Mc2Context::GetMc2ContextTensor(groupEp,"moe_distribute_v2",hcclBuffSize,...)` 并执行 inner V3。这里的“V3”是内部算子/Kernel 复用，不把公开入口升级为不支持 950 的 V5。

## Host/Launcher 执行流程

### 参数校验与前处理

`DispatchCheckNotNull` 检查必选输入、七个输出和非空 `groupEp`；`quantMode==2` 要求动态尺度输出，group 名长度须 `<128`。base 识别 950（DAV_3510），将 performanceInfo 清空；公开 `commAlg` 三值都走 MTE，构造 Mc2 context 后调用 inner V3。OpDef 做 `AutoContiguous`，950 V3 wrapper 将 input/attr 索引映射给 common Tiling。

### 参数、Workspace 与生命周期

Host 写 `MoeDistributeDispatchV2TilingData`，workspace=`16 MiB + 512*aivNum*aivNum`。后项由各 AIV 以 512-byte 间隔写接收计数。context 持有 EP 通信 window 和 data-state 双半窗；executor、context、workspace、tiling 在同一 stream 上存活到 Kernel 完成。

### 启动图与失败传播

V4 GetWorkspaceSize → base 创建 context/inner V3 → `MoeDistributeDispatchV3TilingFuncA5` → common Tiling → 单节点 `moe_distribute_dispatch_v3` → V2 或 FullMesh body。节点使用全部 AIV，POD 为 `MoeDistributeDispatchV2TilingData`，输入/七个输出直接绑定公开 tensor，workspace 为 `16 MiB+512*aivNum²`；没有跨 Kernel 边。context/window、workspace 与 tiling 由 executor 在同一 stream 上持有至节点完成；任一 ACLNN/base/Tiling/launch 失败阻断唯一节点，不支持组合无 fallback。

## Route 与参数设计

### 分核与资源计算

950 使用全部 AIV core；Tiling 计算 `axisMaxBS=globalBs/epWorldSize`、`hAlignWinSize=Align512(hScaleIdxSize)`、双半窗和每专家窗口。公共 workspace 是 `16 MiB+512*AIV²`。fullmesh_v2 以 `ceil(hOutSizeAlign/SPLIT_BLOCK_DATA_SIZE)*SPLIT_BLOCK_SIZE`（常量 480/512 字节）计算物理通信 stride，并校验 status、双 H buffer、双通信 buffer、expertIds 和量化临时区总 UB。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 MTE fullmesh_v1 | `A5,MTE,NO_FULLMESH,HasTp=false,Q,S` | V4 950；commAlg 为空或 fullmesh_v1；shape/quant/window 通过 | fullmesh_v2 → R-002 |
| 2 | R-002 MTE fullmesh_v2 | `A5,MTE,ENABLE_FULLMESH,HasTp=false,Q,S` | commAlg=fullmesh_v2，K≤12 且 UB/window 通过 | 失败，无 fallback |

```text
V4 base on A5 public commAlg -> create Mc2Context -> inner V3
if commAlg==fullmesh_v2 and guards pass: R-002
else if commAlg in {empty,fullmesh_v1}: R-001
else: ACLNN/Tiling failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 950/MTE/no-fullmesh | 950 V4，公开 fullmesh_v1，quant 0..4合法 | AIV_VECTOR+HCCL window | `MoeDistributeDispatchV2<X,Y,Q,S,false>` | 独立32B status、SyncAll、workspace count | 按有效count压紧 |
| R-002 | 950/MTE/fullmesh | 950 V4，fullmesh_v2，K≤12、无TP | AIV_VECTOR+HCCL window | `MoeDistributeDispatchV2FullMesh<X,Y,Q,S,false>` | 512B DataAsFlag、core role barrier | 480B payload tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 public MTE | commAlg 仅空/v1/v2 | 公开文档未承诺 CCU | 全部 | API域外/失败 | V4 docs/base |
| fullmesh_v2 K | `K≤12` | DataAsFlag 索引/UB 上界 | R-002 | 失败 | common Tiling |
| TP 禁用 | `tpWorldSize!=2` | 950 public body固定 HasTp=false | 全部 | 失败 | Tiling key |
| performance 清空 | 950 时 nullptr | public 950不闭合性能计时buffer | 全部 | 不产生该路径 | base |
| UB/window | 计算值≤容量 | queue、状态、量化区并存 | R-001/002 | 失败 | UB/WinSize check |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| V3 launch | `mc2Context` | GM_ADDR/64 | 第1参数 | byte address | V4 base/inner V3 | body Init | 必选 |
| 同上 | `x` | GM_ADDR/64 | 第2参数 | byte address | inner V3 | route body | 必选 |
| 同上 | `expertIds` | GM_ADDR/64 | 第3参数 | byte address | inner V3 | sort/router | 必选 |
| 同上 | `scales` | GM_ADDR/64 | 第4参数 | byte address | optional binding | quant/smooth | 可空 |
| 同上 | `xActiveMask` | GM_ADDR/64 | 第5参数 | byte address | optional binding | mask path | 可空 |
| 同上 | `expertScales` | GM_ADDR/64 | 第6参数 | byte address | optional binding | internal arg | public fullmesh inactive |
| 同上 | `elasticInfo` | GM_ADDR/64 | 第7参数 | byte address | base/inner | V2 body | public 950为空 |
| 同上 | `performanceInfo` | GM_ADDR/64 | 第8参数 | byte address | base强制空 | perf path | public 950为空 |
| 同上 | `expandXOut` | GM_ADDR/64 | output1 | byte address | executor | LocalWindowCopy | 必选 |
| 同上 | `dynamicScalesOut` | GM_ADDR/64 | output2 | byte address | executor | quant writer | 条件有效 |
| 同上 | `assistInfoOut` | GM_ADDR/64 | output3 | byte address | executor | metadata writer | 必选 |
| 同上 | `expertTokenNumsOut` | GM_ADDR/64 | output4 | byte address | executor | count writer | 必选 |
| 同上 | `epSendCountsOut` | GM_ADDR/64 | output5 | byte address | executor | EP count writer | 公开语义recv |
| 同上 | `tpSendCountsOut` | GM_ADDR/64 | output6 | byte address | executor | TP count writer | public 950为空 |
| 同上 | `expandScalesOut` | GM_ADDR/64 | output7 | byte address | executor | scale writer | 条件有效 |
| 同上 | `workspaceGM` | GM_ADDR/64 | 倒数第2 | byte address | executor | scratch/count | 容量按Tiling |
| 同上 | `tilingGM` | GM_ADDR/64 | 最后 | byte address | 950 Tiling | GET_TILING_DATA | key同构建 |
| `MoeDistributeDispatchV2TilingData` | `mc2InitTiling` | opaque | 第1子结构；内部字段固定仓不可见 | block | Mc2 Tiling | context init | 不推测 |
| 同上 | `mc2CcTiling1` | opaque | 第2子结构 | block | EP Tiling | EP channel | 不重排 |
| 同上 | `mc2CcTiling2` | opaque | 第3子结构 | block | TP Tiling | inactive TP channel | 保留ABI |
| 同上 | `moeDistributeDispatchV2Info` | V2Info | 第4子结构 | aggregate | common/950 Tiling | bodies | 见下列字段组 |
| `MoeDistributeDispatchV2Info` | `epWorldSize` | uint32/32 | 首字段；无offsetof断言 | rank | 950 Tiling | EP loops | >0 |
| 同上 | `tpWorldSize` | uint32/32 | 顺序字段 | rank | 950 Tiling | TP guard | public 950=1 |
| 同上 | `epRankId` | uint32/32 | 顺序字段 | rank id | 950 Tiling | metadata | 合法rank |
| 同上 | `tpRankId` | uint32/32 | 顺序字段 | rank id | 950 Tiling | TP metadata | inactive=0 |
| 同上 | `expertShardType` | uint32/32 | 顺序字段 | enum | attr | layout | 公开约束 |
| 同上 | `sharedExpertNum` | uint32/32 | 顺序字段 | expert | attr | A/window | 公开约束 |
| 同上 | `sharedExpertRankNum` | uint32/32 | 顺序字段 | rank | attr | layout | 公开约束 |
| 同上 | `moeExpertNum` | uint32/32 | 顺序字段 | expert | attr | sort/count | 上界 |
| 同上 | `quantMode` | uint32/32 | 顺序字段 | enum | attr | quant template | 0..4 |
| 同上 | `globalBs` | uint32/32 | 顺序字段 | token | attr | axisMaxBS | 防溢出 |
| 同上 | `bs` | uint32/32 | 顺序字段 | token | x shape | loops | route上界 |
| 同上 | `k` | uint32/32 | 顺序字段 | expert/token | expertIds shape | gather/sort | ≤16/12 |
| 同上 | `h` | uint32/32 | 顺序字段 | element/token | x shape | stride | 1024..8192 |
| 同上 | `a` | uint32/32 | 顺序字段 | output token | Tiling derivation | output range | 防溢出 |
| 同上 | `aivNum` | uint32/32 | 顺序字段 | core | platform info | core roles | >0 |
| 同上 | `isTokenMask` | bool/8 | 顺序字段 | flag | mask rank | mask body | conditional |
| 同上 | `isExpertMask` | bool/8 | 顺序字段 | flag | mask rank | mask body | conditional |
| 同上 | `hasElasticInfo` | bool/8 | 顺序字段 | flag | optional input | elastic path | false |
| 同上 | `isPerformance` | bool/8 | 顺序字段 | flag | base清空 | perf path | false |
| 同上 | `isQuant` | bool/8 | 顺序字段 | flag | quantMode | buffer allocation | mode!=0 |
| 同上 | `isMc2Context` | bool/8 | 顺序字段 | flag | 950 wrapper=true | context window | true |
| 同上 | `reserved1` | bool/8 | 顺序字段 | flag | zero/default | 未消费 | 保持0 |
| 同上 | `reserved2` | bool/8 | 顺序字段 | flag | zero/default | 未消费 | 保持0 |
| 同上 | `totalUbSize` | uint64/64 | C++对齐后 | byte | platform info | allocation guard | 上界 |
| 同上 | `totalWinSizeEp` | uint64/64 | 顺序字段 | byte | context/Tiling | EP window | buffer内 |
| 同上 | `totalWinSizeTp` | uint64/64 | 顺序字段 | byte | Tiling | TP window | inactive=0 |
| 同上 | `expertTokenNumsType` | uint32/32 | 顺序字段 | enum | attr | count writer | 0/1 |
| 同上 | `zeroComputeExpertNum` | int32/32 | 顺序字段 | expert | attrs sum | layout | ≥0 |
| 同上 | `maxSizeForUbBuffer` | uint32/32 | 顺序字段 | byte | UB formula | InitBuffer | ≤totalUbSize |
| 同上 | `scalesRow` | uint64/64 | 顺序字段 | row | scales shape | quant index | conditional |
| 同上 | `scalesCol` | uint64/64 | 顺序字段 | column | scales shape | quant index | conditional |
| 同上 | `scalesTypeSize` | uint32/32 | 顺序字段 | byte | dtype | scale stride | conditional |
| 同上 | `scalesCount` | uint64/64 | 末字段 | element | shape product | bounds | 防溢出 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 950/MTE/no-fullmesh | `v3/...tiling_a5.cpp`→common Tiling | V2 tiling+V3 ABI | `v3/op_kernel/moe_distribute_dispatch_v3.cpp` | `v2/op_kernel/moe_distribute_dispatch_v2.h` | v3 `CMakeLists.txt` |
| R-002 | 950/MTE/fullmesh | 同上 | 同上 | 同上 | `...moe_distribute_dispatch_v2_full_mesh.h` | 同上 |

## Kernel 实现

### 公共执行流程

V4 base 创建的 context 作为 V3 entry 首参数；entry 按 `<false,QuantMode,ScaleMode,FullMesh,MTE,A5>` 实例化复用 body。两 route 均在 UB 统计/重排，向 context window 通信，等待就绪后压紧输出；无 Cube/L1/L0。

### Route R-001：MTE fullmesh_v1 状态窗

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `...v2_base.cpp` | base | 950且public MTE | context+inner V3 |
| Tiling | `...v3_tiling_a5.cpp` | 950 func | empty/fullmesh_v1 | 950 no-fullmesh key |
| Router | `...v3.cpp` | entry | 950,MTE,NO_FULLMESH | V2 body |

#### Device 数据流

##### 函数级调用链

```text
moe_distribute_dispatch_v3 -> V2::Init -> Process
  -> AlltoAllDispatch -> SetStatus -> WaitDispatch -> LocalWindowCopy -> SetExpertTokenNums
```

##### 源码等价伪码

```text
take EP window from Mc2Context; select dataState half
scatter token+metadata, write 32B status and SyncAll
poll/reduce counts through workspace; compact valid records to outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| EP window | `dataState*totalWinSizeEp/2+...` | half-window | token+triple | dispatch | local copy | context | 512-align |
| status | window status area | 32B record | flag/count | SetStatus | WaitDispatch | context | SyncAll |
| workspace | `(512*AIV)*writer` / `512*reader` | 512*AIV² | counts | writers | reducer | executor | separated |
| UB | status/index/gather/quant queues | Tiling formula | mixed | DMA/vector | DMA/vector | kernel | events/reuse |
| L1/L0 | N/A | 0 | N/A | N/A | N/A | N/A | 未使用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  status["statusBuf/waitStatusBuf · aligned status"]:1
  ids["expertIdsBuf · Align(BS*K*4)"]:1
  mask["gatherMaskTBuf · maxSize"]:1
  quant["receiveFloat/smoothScale · conditional 2*maxSize"]:1
  inq["xInQueue · depth=bufferNum · hAlignSize"]:2
  outq["xOutQueue · depth=bufferNum · hOutAlignUbSize"]:2
```

##### 字节账与复用顺序

```text
workspace_total = 16 MiB + 512*aivNum*aivNum
expertPerSizeOnWin = (globalBs/epWorldSize) * Align512(hScaleIdxSize)
UB_total = status + expertIds + gatherMask + conditional quant*2*maxSize + bufferNum*(hAlignSize+hOutAlignUbSize)
bufferNum = (double-buffer total > 190KiB) ? 1 : 2; dynamic_ub = N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `gatherMaskTBuf_` | mask/gather | route筛选 | 后续临时计算 | 阶段顺序+PipeBarrier | mask已消费 |
| `receiveDataCastFloatBuf_`→`dstExpBuf_` | quant temp | quant | index阶段 | 编译期分支+阶段结束 | quant结果已发送 |
| `smoothScalesBuf_`→`subExpBuf_` | scale temp | scale consumer | index阶段 | 同上 | scale已消费 |
| EP half-window | dispatch | LocalWindowCopy | 下轮dataState | status+SyncAll | 当前半窗完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dispatch | `SyncAll<true>`+status polling | scatter | WaitDispatch | payload/count | 32B atomic record |
| quant | mode-specific cast/scale | vector | communication | token/scale | 只处理有效mask |
| UB reuse | hard events/TQue | DMA/vector | next stage | queues | valid tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...v3_tiling_a5.cpp` | 950 func | 950/MTE key |
| Kernel entry/router | `...v3.cpp` | `MoeDistributeDispatchV2` | quant/dtype实例 |
| Device body/helper | `...v2.h` | `Process/WaitDispatch` | status/window/copy |

### Route R-002：MTE fullmesh_v2 DataAsFlag

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `...v2_tiling.cpp` | `isSetFullMeshV2` | commAlg=v2、K≤12、无TP | 950 fullmesh key |
| Router | `...v3.cpp` | entry | 950,MTE,ENABLE_FULLMESH | FullMesh body |

#### Device 数据流

##### 函数级调用链

```text
entry -> FullMesh::Init -> Process
  -> AllToAllDispatch || CalCumSum -> PipeBarrier -> LocalWindowCopy
```

##### 源码等价伪码

```text
stride=ceil(hOutSizeAlign/SPLIT_BLOCK_DATA_SIZE)*SPLIT_BLOCK_SIZE
scatter [32B arrival flag|480B payload] records while other cores prefix-sum
barrier; poll embedded flags; strip every 32B gap and commit valid output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| DataAsFlag window | token*stride | ceil(size/480)*512 | 32B+480B | scatter | copy | context | 512B |
| workspace | per-AIV | 512*AIV² | counters | cumsum | copy | executor | disjoint |
| UB | H/comm/index/quant | computed upper bound | mixed | AIV | AIV | kernel | conditional double-buffer |
| L1/L0 | N/A | 0 | N/A | N/A | N/A | N/A | 未使用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inq["inQueue · depth=2 · 2*hAlignSize"]:1
  outb["outBuf · 2*hCommuSize"]:1
  ids["expertIds/gatherMask · aligned"]:1
  quant["quant/reuse · conditional 2*maxSize"]:1
  wait["wait/status/cumsum buffers · aligned"]:2
  arena["remaining tBuf · up to 190KiB total"]:2
```

##### 字节账与复用顺序

```text
hCommuSize = Ceil(hOutSizeAlign,SPLIT_BLOCK_DATA_SIZE)*SPLIT_BLOCK_SIZE
workspace_total = 16 MiB + 512*aivNum*aivNum
UB_total = 32B state + 2*hAlignSize + 2*hCommuSize + expertIds/gatherMask + conditional quant/reuse + wait/cumsum <= 190KiB
dynamic_ub = N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `receiveDataCastFloatBuf`→`dstExpBuf_` | quant | quant consumer | index阶段 | 阶段结束 | quant结果已发出 |
| `smoothScalesBuf`→`subExpBuf_` | scale计算 | scale consumer | index阶段 | 阶段结束 | scale已消费 |
| DataAsFlag record | remote scatter | LocalWindowCopy | 下轮dataState | embedded flag+PipeBarrier | 末块只取有效payload |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| role join | PipeBarrier | scatter/cumsum | copy | counts/window | all roles complete |
| arrival | embedded flag | remote | local copy | each block | final valid bytes |
| gap strip | 480B blockLen+32B stride | window | output | payload | padding不外露 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...v2_tiling.cpp` | fullmesh_v2 checks | key/UB/window guard |
| Kernel entry/router | `...v3.cpp` | `MoeDistributeDispatchV2FullMesh` | 950/MTE实例 |
| Device body/helper | `...v2_full_mesh.h` | `Process/LocalWindowCopy` | DataAsFlag/strip |

## 算子特定优化

### Trick T-001：V4 public group 在 Host 转成 Mc2Context

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：950 V4，公开 MTE commAlg
- Golden 来源：[`moe_distribute_dispatch_v2_base.cpp:189-192`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t hcclBuffSize = 0;
const char *opName = "moe_distribute_v2";
auto aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context);
```

#### 机制与设计原因

公开 API 保留 group 名，Host 在 executor 内物化 context，Device 因而复用 V3 context entry/body，避免 Kernel 再解析字符串或构造通信资源。

#### 迁移边界

- 适用：950 V4 public MTE 路线。
- 前提：context 生命周期覆盖 stream 上 Kernel，op tag 与 buffer size 保持一致。
- 失效：A2 使用独立 A2 entry；V5 public 不支持 950。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

将内部 V3 entry 误记为 950 公开 V5 会违反 v3 README 的平台产品表。

### Trick T-002：DataAsFlag 合并 payload 到达状态

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：fullmesh_v2、K≤12、无TP
- Golden 来源：[`moe_distribute_dispatch_v2_full_mesh.h:431-432`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
```

#### 机制与设计原因

480-byte payload 与 32-byte flag 组成 512-byte 记录，消费者逐块轮询后跨 gap 搬运，减少独立状态表寻址。

#### 迁移边界

- 适用：共享 512/480/32 ABI 的 window 通信。
- 前提：Host 和 Device 用相同 stride。
- 失效：fullmesh_v1 使用独立 status route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

忽略 32-byte gap 会污染输出并错位下一 token。

## 模板与实现对比

| 对比维度 | R-001 fullmesh_v1 | R-002 fullmesh_v2 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | empty/v1 | v2+guards | public commAlg |
| 模板与 dtype | `V2<X,Y,Q,S,false>`，quant 0..4 | `FullMesh<X,Y,Q,S,false>`，quant 0..4 | quantMode/输出dtype |
| 循环与分核 | 全AIV按rank/expert分发 | scatter/cumsum core分组 | 状态协议 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | TQue/TBuf+独立status window | TQue/TBuf+DataAsFlag window | InitBuffer/stride |
| 内存总占用与复用 | 16MiB+512*AIV²，quant buffer别名 | 同workspace，190KiB UB guard | common Tiling |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Vector+MTE window | Vector+512B record/480B copy | 无SIMT/NDDMA/Cube |
| 同步、精度与 tail | SyncAll+32B status | barrier+embedded flag | flag粒度 |
| fallback/失败边界 | v2转R-002 | guard失败 | 无隐式fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本卡不覆盖 | 本卡不覆盖 | V4 最新公开支持；V5不支持 |
| Host/Launcher | group→A2 inner | V5 context入口 | V4 group→context→inner V3 |
| Kernel/资源 | 专用A2 bodies | A3 V3 entry | 950 specialization复用V2 bodies |

## 限制与失败边界

- 正式 Route 只覆盖公开文档列出的空/fullmesh_v1/fullmesh_v2。固定源码虽接有 `commAlg="ccu"`、950 CCU Tiling/router/body，但它不在 V4 公开支持域，故不发布为 ACLNN Route。
- V5 README 明确 950 不支持；内部 V3 复用不能据此推导公开 V5 950 支持。
- 950 public 路线不支持 TP、elastic/performance；fullmesh_v2 受 K/UB/window guard 限制。
- 固定仓未给出 Mc2 opaque tiling 子结构内部定义和完整 C++ offsetof；不得推测。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.h` | V4 API | 公开签名 |
| 入口声明/约束 | `mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md` | V4 docs | 950公开约束/commAlg |
| 入口声明/约束 | `mc2/moe_distribute_dispatch_v3/README.md` | 产品表 | V5不支持950 |
| 入口实现 | `mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.cpp` | V4 funcs | base转发 |
| 入口实现 | `mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp` | base | 950 context/inner V3 |
| Host/Tiling/Launcher | `mc2/moe_distribute_dispatch_v3/op_host/op_tiling/arch35/moe_distribute_dispatch_v3_tiling_a5.cpp` | 950 func | 950 wrapper/key |
| Host/Tiling/Launcher | `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` | common func | guards/resources |
| POD | `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h` | V2 structs | POD ABI |
| Kernel entry/body | `mc2/moe_distribute_dispatch_v3/op_kernel/moe_distribute_dispatch_v3.cpp` | v3 entry | 950 router |
| Kernel entry/body | `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h` | V2 Process | v1 body |
| Kernel entry/body | `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h` | FullMesh Process | v2 body |
| 构建接线 | `mc2/moe_distribute_dispatch_v3/CMakeLists.txt` | dependencies | V3复用V2接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | V4 950通过context复用V3 entry | 首参数context、V2Info字段顺序 | v3 entry |
| 调用顺序 | 重排→通信→等待→压紧 | v1/v2状态协议不可交换 | route Process |
| workspace/内存 | 两body共用预算 | 512/480/32 stride | FullMesh Init |
| 同步 | 共用Mc2 window | status与embedded flag不可混用 | WaitDispatch/LocalCopy |
| 精度与尾块 | quant模板复用 | mode/ScaleMode、cast次序、valid count | entry branches |
| 失败边界 | public commAlg早筛 | CCU不作为public route，V5不支持950 | docs/base/README |