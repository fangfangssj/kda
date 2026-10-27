---
type: Operator
title: "AlltoAllvGroupedMatMul A3 ACLNN 算子知识"
description: "说明 aclnnAlltoAllvGroupedMatMulV2 在 A3 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_tiling_a3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_graph/allto_allv_grouped_mat_mul_gen_task_training.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_host/op_tiling/mc2_calc_num_blocks.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_comm_params.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/scheduler/a2av_gmm_scheduler.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/hccl_a2av_op.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_permute_engine.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/compute/quant_grouped_matmul.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/grouped_matmul/op_kernel/quant_block_sch.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_host/config/ascend910_93/allto_allv_grouped_mat_mul_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:28:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a3]
cann_versions: ["9.1.0"]
---

# AlltoAllvGroupedMatMul A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize` / `aclnnAlltoAllvGroupedMatMulV2`，公开 ACLNN V2 两段式入口 |
| 计算语义 | `ataOut=AlltoAllv(gmmX)`；`gmmY=GroupedMatMul(ataOut,gmmWeight)`；可选 `mmY=mmX×mmWeight`，可选输出 expert-first `permuteOut` |
| 输入输出 | FP16/BF16、ND；`gmmX(BSK,H1)`、`gmmWeight(e,H1,N1)` 或转置存放、`gmmY(A,N1)`；共享专家三件套可选 |
| 执行类型 | `MIX_CV` + runtime HCCL AICPU service task；单个 AscendC kernel 使用 `KERNEL_TYPE_MIX_AIC_1_2` |
| Kernel 路线 | R-001 AICPU 通信先行融合；generated task callback 配置 AICPU/KFC stream，四个 transpose tuple key 均在执行矩阵逐项闭合 |
| 核心资源 | GM、user workspace、AIC Matmul 的 L1/L0A/L0B/L0C；请求 permute 时每 AIV 两块 32768B UB |
| 来源闭包 | `source_closed`：V2 ACLNN→generated inner bridge（框架边界）→OpDef/Tiling→POD/key→MIX kernel→HCCL/permute/GMM specialization→构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `P=epWorldSize`，`send[r,e]`/`recv[r,e]` 表示到/来自 rank `r`、本地专家 `e` 的 token 数，`BSK=sum(send)`，`A=sum(recv)`。AIV0 以 `axis=H1` 提交分专家批次的 `AlltoAllV`，通信结果在 workspace 中按 `rank→expert` 排列。AIC 为每个 `(expert,rank)` group 构造 A offset table，直接对 rank-first 通信结果做 GMM；仅当 `permuteOutFlag=true` 时，全体 AIV 才另行把它重排成 expert-first `permuteOut(A,H1)`。

共享专家路径为 `mmY=mmX(BS,H2)×mmWeight(H2,N2)`。AIC 执行 local MM 时，AIV 已越过该 AIC-only body并提交通信，因此两者核间并行。非量化 specialization 的 scale/bias 均为 `nullptr`，`aQuantMode=bQuantMode=DEFAULT`、`hasBias=false`、activation=NONE。底层 `MatmulImpl` 的累加与舍入细节在固定仓未包含的 AscendC 库中，不把 wrapper 的 `float` scale 类型误写成累加类型。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gmmX` | `(BSK,H1)` / ND / 2D | FP16/BF16 | 必选；`0<BSK<52428800`，`0<H1<65536` |
| `gmmWeight` | `(e,H1,N1)` 或转置 `(e,N1,H1)` / ND / 3D | 同 `gmmX` | 文档一般约束 `1<=e<=48`；A3 还受 `e*P<=256`、`P>=8` 约束，实际 `e<=min(48,floor(256/P))<=32`；K 维等于 H1，`0<N1<65536` |
| counts tensor 两项 | N/A | N/A | 保留参数；V2 API 与 tiling 当前都只接受 `nullptr` |
| `mmX/mmWeight/mmY` | `(BS,H2)`、`(H2,N2)` 或转置、`(BS,N2)` / ND | 同 `gmmX` | 三者全空或全非空；`0<H2<=12288`；`BSK%BS==0` 且 `BSK/BS∈[2,8]` |
| `group` | string | char | 必选，长度 1..127 |
| `commMode` | string | char | A3 仅接受 `"ai_cpu"`；头注释中的空串/`"aicpu"` 不被实现接受 |
| `epWorldSize` | 标量 | int64 | A3 公开范围 `{8,16,32,64,128}` |
| `sendCounts/recvCounts` | 长度 `e*P` | int64 list | 非空、等长、非负；和分别为 BSK/A；实现 POD 限制长度 `<=256` |
| transpose 两项 | 标量 | bool | 默认 false；无共享专家时 Host 强制 `transMmWeight=false` |
| `permuteOutFlag` | 标量 | bool | 与 `permuteOut` 是否存在严格一致 |
| `gmmY` | `(A,N1)` / ND | FP16/BF16 | 必选；dtype 同 `gmmX` |
| `permuteOut` | `(A,H1)` / ND | FP16/BF16 | flag=true 时必选，否则必须为空 |

### 入口与版本

固定 Commit 同时公开 V1/V2；V2 文档明确相对 V1 新增 `commMode`，故选 V2 为同平台最新公开入口。第一段完成本文件可见的指针/可选项/group/counts/commMode检查，再调用 generated `aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize`；第二段把 A3 HCCL server设为 AICPU后调用 generated inner 执行。generated bridge 不在固定仓，是框架边界；算子自有 selector、POD、workspace和 Device body均可追溯。

## Host/Launcher 执行流程

### 参数校验与前处理

API 顺序是 `CheckParams→CheckSendAndRecv→CheckAndHandleCommMode→inner GetWorkspaceSize`。OpDef 对输入声明 `AutoContiguous()`；generated bridge 的具体 contiguous 节点不可见。Tiling 顺序是 `GetContextAttr→CheckShapeDims→CheckDType→CheckShapeRelation→CheckShapeSize→CheckAttrsShapeSize/Relation→CheckSendRecvDataVolumn→CheckMKN`。A3 `NeedToCheckCounts=true`，逐项要求 `0<=send<=BSK`、`0<=recv<=A`。任一步失败即 `GRAPH_FAILED`，没有 Device fallback。

### 参数、Workspace 与生命周期

```text
S = sizeof(FP16/BF16) = 2
commOut = align_up(A*H1*S, 512)
userWorkspace = commOut + groupList(8eP) + aOffset(8eP) + xScaleOffsetReserve(8eP) + pointerReserve(512)
workspace[0] = libApiWorkSpaceSize + align_up(A*H1*2,512) + 24eP + 512
```

Device 用 `SetSysWorkspace`/`GetUserWorkspace` 取得 user slice。共享 MM 与 routing GMM 时序复用 commOut 之后的 metadata区；通信只写 commOut，二者不重叠。workspace、输入输出和 executor 活到传入 stream 上 kernel完成；generated bridge 的精确释放点固定仓不可见。

### 启动图与失败传播

```text
V2 GetWorkspaceSize -> generated inner -> A3 tiling(priority 0)
  -> HCCL AICPU tiling + two optional TCubeTiling + workspace/blockDim/key
V2(stream) -> SetHcclServerType(AICPU) -> generated task callback(AICPU/KFC stream) -> generated launch
  -> one allto_allv_grouped_mat_mul MIX kernel, blockDim=min(AIC,AIV/2)
     AIC: optional local MM -> wait comm flag -> routing GMM
     AIV0: submit all AlltoAllV batches -> wait each handle
     all AIV: barrier -> set flag -> optional permute
     all: final barrier; AIV HCCL Finalize
```

API/inner/tiling 失败时不 launch；entry 遇到空 workspace/user workspace 直接返回无输出。HCCL handle 的运行错误传播属于 HCCL/runtime边界，固定 Device body没有自定义错误码回传分支。

## Route 与参数设计

### 分核与资源计算

`blockDim=min(aicCoreNum,aivCoreNum/2)`，entry 声明 AIC:AIV=1:2。`CalcExpertNum` 规则：`e<=4→1`；`ceil(e/4)<=4→4`；否则默认4，当 `(BSK/P)*N1<20*1024*1024` 时取 `max(1,min(e,128/P,e/2))`。每批实际 expert 数为 `min(expertNum,e-start)`，routing GMM group数为 `actualExpertNum*P`。

非量化 TCube：`baseM=align16(min(M,256))`；`baseN` 从 `min(N,256)` 按 B layout 对齐；`baseK=align16(min(64,K))`。`baseM*baseN*4` 超 L0C时缩 baseN。`stepM=stepN=1`，`dbL0A=dbL0B=2`，`dbL0C` 按 `2*baseM*baseN*4<=L0C` 取2否则1。L1 以 `baseA=baseM*baseK*2`、`baseB=baseN*baseK*2` 推出 `depthA1/depthB1` 与 `stepKa/stepKb`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 AICPU MIX | `GET_TPL_TILING_KEY(gmmT,mmT,1)` | A3；`commMode=ai_cpu`；`gmmT=transGmmWeight`；`mmT=isNeedMM?transMmWeight:false` | 参数/资源不满足即失败；无其它通信 route |

```text
commMode != ai_cpu -> fail
mmT = isNeedMM ? transMmWeight : false
(gmmT,mmT,1) in {(0,0,1),(1,0,1),(0,1,1),(1,1,1)} -> R-001 exact specialization
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001/NN | `(0,0,1)` | A3 FP16/BF16 ND；shared可无或非转置 | 单 MIX：AIV HCCL/permute，AIC local/routing GMM | `HcclA2avOp<T,true,1>`；两条 `bTrans=false` | rank-first commOut；flag8 | expert/M/N/K tail；失败终止 |
| R-001/TN | `(1,0,1)` | routing B转置；shared可无或非转置 | 同上 | routing `bTrans=true`，local false | B offset=`nOffset*K` | 同上 |
| R-001/NT | `(0,1,1)` | shared必须存在且 B转置 | 同上 | routing false，local true | local phase先复用compute区 | 同上 |
| R-001/TT | `(1,1,1)` | shared必须存在；两条 B转置 | 同上 | 两条 `bTrans=true` | 两套独立 TCubeTiling | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 核配比 | `min(AIC,AIV/2)>0` | MIX 1:2 | R-001 | tiling失败 | `mc2tiling::GetNumBlocks` |
| counts容量 | `len=eP<=256` | POD数组固定256 | R-001 | tiling失败 | `CheckAttrsShapeSize/Relation` |
| Device每卡expert缓存 | A3由`P>=8`与`eP<=256`推出`e<=32` | `expertTokenNum_[MAX_EXPERT_PER_RANK]`仅32项 | R-001 | Host已有约束使越界Route不可达 | `TaskTilingInfo`,`QuantGroupedMatmul::Init` |
| GMM batch | `expertNum*P<=128` | `GMMArray[128]` | R-001 | helper失败 | `CalcExpertNum`,`SetInputParams` |
| L0C | `baseM*baseN*4<=L0C` | accumulator tile | R-001 | 缩baseN或失败 | `CalBasicBlock` |
| L1 | `depthA1*baseA+depthB1*baseB<=L1` | A/B stage | R-001 | 降depth/step或失败 | `CalL1Depth` |
| permute UB | flag=true才 `2*32768B` | GM↔UB双缓冲 | R-001 | flag=false跳过 | `PermuteData` |
| 空 expert batch | token和为0 | 避免空GMM | R-001 | 直接return | `QuantGroupedMatmul::Process` |

### POD/ABI

`Mc2InitTiling`、`Mc2CcTiling`、`TCubeTiling` 来自固定仓未包含的 SDK 头，含它们的外层 numeric `sizeof/offsetof` 不可见；下表保留可证明的内部布局和符号 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| launch | 11个GM参数 | `GM_ADDR`/64bit | entry参数顺序 | 地址 | generated inner | entry/scheduler | counts tensor地址不消费；workspace必需 |
| outer POD | `hcclA2avTilingInfo` | SDK HCCL structs | offset0，size不可见 | bytes | `SetHcclTiling` | `InitV2/SetCcTilingV2` | engine=AICPU |
| outer POD | `taskTilingInfo` | `TaskTilingInfo` | symbolic outer offset；`sizeof=2160,align=8` | mixed | `PostTiling` | scheduler/comm/GMM | counts<=256 |
| Task | 12个shape/resource字段 | uint64×12 | offsets 0,8,...,88 | 元素/字节/核 | `PostTiling` | 各body | inactive BS/H2/N2=0 |
| Task | 四个loop字段 | uint32×4 | offsets 96,100,104,108 | expert/loop | `CalcExpertNum/PostTiling` | scheduler | expertNum>=1 |
| Task | `sendCnt` | int32[256] | offset112，1024B | token | int64 attr窄化 | HCCL/GMM | 非负；和=BSK |
| Task | `recvCnt` | int32[256] | offset1136，1024B；Task总2160B | token | int64 attr窄化 | HCCL/GMM | 非负；和=A |
| outer POD | `isPermuteOut,isNeedMM` | bool | Task之后，numeric outer offset不可见 | flag | `PostTiling` | scheduler | false不读可选GM |
| outer POD | 两套GMM tiling | `GMMQuantTilingData` | 各含24B params+1536B arrays+SDK TCube | tiling | helper | local/routing GMM | inactive mm不消费 |
| GMM params | 4×uint32 + flags | 24B，align4 | groupNum@0；modes@8/12；flags@16..23 | count/enum | helper | GMM body | no quant/no bias/SPLIT_M |
| GMM array | m/k/n list | int32[128]×3 | offsets0/512/1024；1536B | 元素 | helper | `SetMNK` | groupNum<=128 |
| Cube tiling | M/N/K/base/step/depth/db/core | SDK struct | numeric ABI不可见 | mixed | `DoLibApiTiling` | Matmul/block scheduler | 同SDK构建 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001四tuple | `(gmmT,mmT,1)` | `AlltoAllvGmmTilingStructA3`→`RunFusionKernelTiling/GetTilingKey` | outer POD + 两套GMM POD | `allto_allv_grouped_mat_mul<gmmT,mmT,1>` | scheduler→HCCL/permute/GMM→Matmul | A3 binary JSON + CMake |

## Kernel 实现

### 公共执行流程

entry 设置 system workspace、读取 POD并实例化 `HcclA2avOp<DTYPE_GMM_X,true,1>` 以及 local/routing 两个 `QuantGroupedMatmul`。AIV 跳过 Cube body并提交通信；AIC可先做local MM。每批仅AIV0 wait HCCL handle，全部AIV barrier后以flag8通知AIC。AIC从rank-first commOut直接做routing GMM；AIV发flag后做可选permute，因此两者可重叠。末尾 `SyncAll<false>`，仅AIV执行HCCL Finalize。

### Route R-001：AICPU 通信先行 MIX

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/...v2.cpp` | `CheckAndHandleCommMode` | 非DAV_3510只接受`ai_cpu` | comm enum=1，否则失败 |
| Host | `...tiling.cpp` | `GetTilingKey` | 两个effective transpose bool + AICPU | 四个tuple之一 |
| Launch task | `op_graph/...gen_task_training.cpp` | `CalcParamFunc/GenTaskFunc` | 非950平台 | AICPU KFC server/stream任务 |
| Router | `op_kernel/...cpp` | entry模板 | tuple编译实例 | scheduler；无unknown fallback |

#### Device 数据流

##### 函数级调用链

```text
entry -> Scheduler::Init
  -> [AIC] localComputeOp.Process(0) || [AIV0] commOp.Launch(all batches)
  -> [AIV0] Wait -> [all AIV] SyncAll<true>/SetFlag(8)
  -> [AIC] WaitFlag(8)/computeOp.Process
  -> [AIV optional] PermuteData/TileDataCopyLoopDoubleBuf
  -> End/SyncAll<false>/AIV Finalize
```

##### 源码等价伪码

```text
if workspace/userWorkspace null: return
commOut=userWorkspace; computeWS=commOut+align512(A*H1*2)
if AIC && isNeedMM: local GMM
if AIV0: submit AlltoAllV for every expert batch
for batch:
  AIV0 Wait(handle); all AIV barrier+SetFlag(8); optional rank→expert permute
  AIC WaitFlag(8); build recv groupList/A offsets; routing GMM
all-core barrier; AIV Finalize
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM gmmX | HCCL send offsets | `BSK*H1*2` | ND rank→expert slices | caller | AIV0 HCCL | caller/stream | count×H1元素 |
| commOut | user+0 | `align512(A*H1*2)` | rank-first | HCCL | AIC GMM/AIV permute | executor WS | 与compute区分离 |
| groupList/A/xScale表 | commOut末尾+0/8eP/16eP | 各8eP | int64/uint64 | AIC | block scheduler | 当前batch | xScale表非量化保留不读 |
| pointer reserve | +24eP | 512B | 非量化实际写slot 0/1/3三个16B descriptor；slot2量化scale未用 | `BuildPtrTable` | GMM | 当前调用 | 固定reserve；local阶段ptr起点为`+8e`，routing阶段改写为`+24eP` |
| permute UB | AIV local | 2×32768B | FP16/BF16 raw tile | GM load | GM store | 当前kernel | 双缓冲，flag控制分配 |
| L1 | Matmul内部 | `depthA1*baseM*baseK*2 + depthB1*baseN*baseK*2` | Cube A/B stage | GM | L0A/B | 当前Matmul | <=L1 |
| L0A/B | Matmul内部 | 各`2*baseM*baseK*2` / `2*baseN*baseK*2`上界 | Cube tiles | L1 | Matmul | tile | exact库排布不可见 |
| L0C | Matmul内部 | `dbL0C*baseM*baseN*4` | accumulator tile | Matmul | GetTensorC | tile | <=L0C |
| GM outputs | token/N offsets | `A*N1*2`等 | ND | AIC/AIV | caller | caller/stream | 只写有效tail |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM gmmX"] --> H["AIV0 HCCL"] --> C["workspace commOut rank-first"]
  C --> L1["AIC GM→L1"] --> L0["L0A/L0B→Matmul→L0C"] --> Y["GM gmmY"]
  C --> UB["AIV UB A/B"] --> P["GM permuteOut"]
  M["GM mmX/mmWeight"] --> LM["AIC local Matmul"] --> MY["GM mmY"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total = libApiWorkSpaceSize + align512(A*H1*2) + 24eP + 512
L1_visible = depthA1*baseM*baseK*2 + depthB1*baseN*baseK*2
L0C = dbL0C*baseM*baseN*4
permute_UB = isPermuteOut ? 65536 : 0 bytes/AIV
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| comm→GMM | HCCL | 当前batch连续slice；batch内rank→expert | FP16/BF16 ND | AIC GMM | handle wait→AIV barrier→flag8 | WS/End | 当前batch完成 |
| comm→permute | HCCL | 同一batch slice | batch内rank→expert | AIV permute | 同上 | WS/permute完成 | `pairIdx`按源码`GetBlockIdx()`和`aivCoreNum`分段；MIX下物理subblock展开语义属于AscendC边界 |
| computeWS local→routing | local GMM metadata | 24eP+512 | tables | routing重写 | AIC程序序 | WS | local返回 |
| UB ping-pong | prior store | 32768B buffer | raw | next load | MTE3_MTE2 wait；load→store MTE2_MTE3 | AIV | final store wait |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| HCCL | handle/Wait | AIV0 | AIV0 | commOut batch | FP16/BF16，count×H1 |
| AIV→AIC | `SyncAll<true>` + mode2 flag8 PIPE_V→PIPE_MTE2 | all AIV | AIC | batch slice | 每批复用flag |
| permute | MTE events | GM load/store | same AIV | two UB buffers | last `realLength`，zero skip |
| GMM | SetSingleShape/Iterate/GetTensorC | L0C | GM | M/N/K tile | base tail；最后group可细分空闲核 |
| 空组 | token sum==0 | counts | routing | N/A | 直接return |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling.cpp` | `CalcExpertNum`,`RunFusionKernelTiling`,`PostTiling` | batch、workspace、POD/key |
| Kernel entry/router | `op_kernel/...cpp` | `allto_allv_grouped_mat_mul<gmmT,mmT,1>` | MIX 1:2和实例 |
| Device body/helper | `a2av_gmm_scheduler.h` | `ProcessCommRoundMode` | overlap、wait/flag、optional permute |
| Device body/helper | `hccl_a2av_op.h` | `LaunchCommData/Wait/PermuteData` | counts/offset/HCCL |
| Device body/helper | `gqmm_cube_on_the_fly.h` | `Mc2GmmASWKernel::Process` | Cube调度/tail/Matmul |

## 算子特定优化

### Trick T-001：按 expert 与每-rank MN 自适应合并批次

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：`e,P,BSK,N1` 进入 `CalcExpertNum`。
- Golden 来源：[`allto_allv_grouped_mat_mul_tiling.cpp:35-62`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp#L35-L62)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (e == 0 || e == 1) {
    return 1U;
}
if (e <= SMALL_EXPERT_THRESHOLD) {
    return 1U;
}
uint32_t originalLoopCount = static_cast<uint32_t>((e + DEFAULT_MERGED_EXPERT_NUM - 1U)
    / DEFAULT_MERGED_EXPERT_NUM);
if (originalLoopCount <= ORIGINAL_LOOP_THRESHOLD) {
    return DEFAULT_MERGED_EXPERT_NUM;
}
```

#### 机制与设计原因

同一值控制 HCCL handle批次、GMM `groupNum=expertNum*P` 和tail。大 e且每-rank MN较小时扩大批次，同时受128项GMM数组与至少两轮约束，保持通信/计算调度和POD容量一致。

#### 迁移边界

- 适用：rank×expert group共享同一M/N/K上界tiling。
- 前提：`expertNum*P<=128`、`eP<=256`。
- 失效：专家K/N/layout不同或group数越界，必须重做POD/tiling。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

`groupNum>128` 时 `SetInputParams` 直接失败，不能无限增大 batch。

### Trick T-002：AIV→AIC 显式交接并与可选 permute 重叠

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：每个 expert batch；permute还要求flag=true。
- Golden 来源：[`a2av_gmm_scheduler.h:115-130`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/scheduler/a2av_gmm_scheduler.h#L115-L130)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if ASCEND_IS_AIV {
    if (GetBlockIdx() == 0) {
        if constexpr (MX_QUANT_MODE) {
            commOp.WaitScale(start);
        }
        commOp.Wait(start);
    }
    SyncAll<true>();
    AscendC::CrossCoreSetFlag<SYNC_MODE_AIV_TO_AIC, PIPE_V>(
        SYNC_FLAG_ID_COMM_DONE + SYNC_FLAG_AIV1_OFFSET * GetSubBlockIdx());
    if (isPermuteOut_) {
        commOp.PermuteData(start, actualExpertNum, permuteOutGm_);
    }
}
if ASCEND_IS_AIC {
    AscendC::CrossCoreWaitFlag<SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(SYNC_FLAG_ID_COMM_DONE);
```

#### 机制与设计原因

HCCL wait与AIV barrier先保证commOut可见，再用PIPE_V→PIPE_MTE2 flag交给AIC。flag先于permute，故AIC routing GMM与AIV重排可并行，且只读同一输入、写不同输出。

#### 迁移边界

- 适用：单MIX kernel中AIV通信producer与AIC consumer共享GM。
- 前提：保留A3 mode2、flag8、barrier与关闭auto-sync时的显式协议。
- 失效：多kernel或配比变化需重新证明可见性。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

删除wait/barrier/flag会让AIC读取未完成slice；源码无安全fallback。

### Trick T-003：rank-first commOut 直接做GMM

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：routing GMM；permute输出独立可选。
- Golden 来源：[`quant_grouped_matmul.h:210-243`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/compute/quant_grouped_matmul.h#L210-L243)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
} else if constexpr (!isLocal && isA2avGmm) {
    const auto *opCnt = &tilingData_->taskTilingInfo.recvCnt[0];
    __gm__ uint64_t *aGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(aGroupOffsetTableGm_);
    for (uint32_t e = 0; e < expertNum; e++) {
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            uint32_t absExpertIdx = startExpertIdx + e;
            uint64_t cnt = static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]);
            groupListPtr[GROUP_LIST_INDEX + e * epWorldSize_ + r] = static_cast<int64_t>(cnt);
        }
    }
```

#### 机制与设计原因

每个 `(expert,rank)` 是一个GMM group，A offset table指向rank-first源，weight只在expert边界推进；因此routing计算不依赖完整permute。未请求permute时也不分配其64KiB/AIV UB。

#### 迁移边界

- 适用：GMM scheduler接受rank维group和外部A offset table。
- 前提：groupList expert-major，A offsets指向rank-first，weight按expert推进。
- 失效：下游只接受连续expert tensor时必须保留permute。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把rank-first数据当普通expert-first连续矩阵会配错权重。

## 模板与实现对比

| 对比维度 | R-001 NN | R-001 TN | R-001 NT | R-001 TT | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `(0,0,1)` | `(1,0,1)` | `(0,1,1)`需shared | `(1,1,1)`需shared | exact tuple |
| 执行族/节点图 | MIX/HCCL/optional local+permute/GMM | 同 | 同 | 同 | runtime flags控制可选子图 |
| M/N/K、tile与scheduler | 两B非转置 | routing转置 | local转置 | 两B转置 | 两套TCubeTiling |
| workspace/L1/L0/UB | 公共公式 | 同 | local先复用 | 同 | commOut/compute分区 |
| AIC/AIV交接 | mode2 flag8 | 同 | 同 | 同 | wait→barrier→flag |
| 同步、精度与tail | expert+Matmul tail | routing transpose | local transpose | 两者 | FP16/BF16无量化 |
| fallback/失败边界 | 校验失败 | 同 | 无shared不可达 | 无shared不可达 | 无Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档明确不支持 | 本卡：支持 | 支持，独立`_950`卡 |
| Host/Launcher | N/A | ascend910_93/arch22；仅ai_cpu；P={8,16,32,64,128} | arch35；ai_cpu/ccu；公开P={2,4,8,16,32,64} |
| Kernel/资源 | N/A | sync mode2，subblock flag offset0；AICPU | mode4，flags 8/24；CCU或AICPU |

## 限制与失败边界

- V2文档约束段称`e*P`最大384，但同一文档参数表和固定Host/POD均规定`MAX_EXPERT_SIZE=256`，Host显式拒绝更长数组；实现复用按`<=256`。
- A3 virtual `GetEpWorldSizeOptional()`没有目录内调用点；通用`GetContextAttr`另行硬编码同一集合，当前结果一致。
- counts从int64 attr写入int32 POD，代码没有独立`INT32_MAX`检查；不得依赖溢出窄化。
- outer POD中的HCCL/TCube numeric ABI、Matmul内部L0排布及累加舍入固定仓不可见；必须用同一SDK。
- 本次未执行编译、真机、数值或性能测试；文档提示A3单卡通信小于2MiB可能性能劣化，这不是功能fallback。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md` | V2原型/产品表 | 最新入口、平台、shape |
| 目录公开约束 | `mc2/allto_allv_grouped_mat_mul/README.md` | 产品表/约束说明 | A3/950支持、`e<=32`与`eP<=256`安全边界 |
| 入口实现 | `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp` | API checks/V2函数 | 校验、server选择 |
| OpDef | `mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_def.cpp` | `AlltoAllvGroupedMatMul` | IO/attr/平台注册 |
| Host/Tiling | `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp` | `CalcExpertNum`,`RunFusionKernelTiling`,`PostTiling` | selector/workspace/POD/key |
| Host平台 | `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_tiling_a3.cpp` | A3 tiling struct | A3策略 |
| Launch task | `mc2/allto_allv_grouped_mat_mul/op_graph/allto_allv_grouped_mat_mul_gen_task_training.cpp` | `AlltoAllvGroupedMatMulCalcParamFunc/GenTaskFunc` | A3 AICPU server与KFC stream任务 |
| 分核 | `mc2/common/op_host/op_tiling/mc2_calc_num_blocks.h` | `mc2tiling::GetNumBlocks` | `min(AIC,AIV/2)`和零核失败 |
| POD/ABI | `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling.h` | outer POD | ABI顺序 |
| POD公共结构 | `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h` | `TaskTilingInfo`/同步常量 | 数组容量、字段顺序、A3 mode2/flag8 |
| Kernel entry/body | `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul.cpp` | entry | MIX/模板实例 |
| Scheduler | `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/scheduler/a2av_gmm_scheduler.h` | scheduler | task graph/workspace/flag |
| 通信 | `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/hccl_a2av_op.h` | HCCL op | counts/offset/wait |
| 通信参数 | `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_comm_params.h` | `CalcA2avCommBeforeParams` | batch/rank/expert count和offset |
| Permute | `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_permute_engine.h` | permute | 分核/UB/tail |
| GMM wrapper | `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/compute/quant_grouped_matmul.h` | `QuantGroupedMatmul` | rank-first offsets |
| Cube body | `mc2/3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h` | `Mc2GmmASWKernel` | Cube/tail/Matmul |
| Cube scheduler | `mc2/3rd/grouped_matmul/op_kernel/quant_block_sch.h` | `QuantASWBlockSch` | group/core/tile/GM offset映射 |
| GMM POD | `mc2/3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h` | `GMMQuantTilingData` | params/数组/TCube顺序与定长数组 |
| Cube tiling | `mc2/3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.cpp` | base/L1 tiling | base/depth/step/db |
| 构建配置 | `mc2/allto_allv_grouped_mat_mul/op_host/config/ascend910_93/allto_allv_grouped_mat_mul_binary.json` | FP16/BF16 entries | dtype/layout |
| 构建接线 | `mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt` | modules/options | 依赖与显式同步编译 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Task内部2160B、两套GMM POD | outer SDK ABI、counts256、tuple/build一致 | POD structs |
| 调用顺序 | AIV通信/AIC local并行→逐批交接→routing/permute | wait/barrier先于读取；Finalize在End | scheduler |
| workspace/内存 | `libApi+align512(AH1*2)+24eP+512` | offset、owner、local→routing复用 | Host/scheduler Init |
| 同步 | A3 mode2 flag8及MTE events | auto-sync关闭时保留显式协议 | scheduler/permute/CMake |
| 精度与尾块 | 非量化FP16/BF16、expert/Matmul/permute tail | 不推断外部Matmul round | GMM/permute bodies |
| 失败边界 | 仅ai_cpu、P集合、shape/count/resource | unknown key/空workspace不fallback | API/tiling/entry |