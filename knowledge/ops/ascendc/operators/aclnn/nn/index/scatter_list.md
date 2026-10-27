---
type: Operator
title: "ScatterList A2/A3 ACLNN 算子知识"
description: "说明 aclnnScatterList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/docs/aclnnScatterList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_host/scatter_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_host/scatter_list_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_host/scatter_list_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_transpose_small.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_transpose_more.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_transpose_large.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_neg_small.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_neg_more.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_neg_large.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_neg_large_e.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_neg_dim2_large.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rsbse.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rlbse.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rlbse_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rsble.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rlble.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rlble_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_host/config/ascend910b/scatter_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_host/config/ascend910_93/scatter_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T16:59:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ScatterList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnScatterListGetWorkspaceSize` / `aclnnScatterList`，公开两段式 ACLNN，固定仓未出现更高版本后缀。 |
| 计算语义 | 对 `varRef[n]` 做稀疏覆盖更新；`indice[n]` 给出轴内起点，二维 indice 的第二值给出有效更新长度，`mask[n]==0` 跳过。`reduce` 仅允许 `"update"`。 |
| 输入输出 | `varRef` 为同 shape、同 dtype 的 ND TensorList；`updates[0]==len(varRef)`；`indice` 为 rank 1 或 2 的 INT32/INT64；可选 `mask` 为 UINT8 rank 1；输出原位回写 `varRef`。 |
| 执行类型 | 单 Kernel `scatter_list`；全部 Route 均为 `AIV_VECTOR`。100–102 使用 `TransDataTo5HDImpl`，其余路线以 AIV MTE/DataCopy 为主；无 Cube/AIC、SIMT、NDDMA、通信或 `MIX_CV` 节点。 |
| Kernel 路线 | R-001～R-014：key 100/101/102/103/104/105/106/107/200/210/211/220/230/231。 |
| 核心资源 | GM TensorList 指针表；depth=1 的 indice/updates/可选 mask `VECIN` Queue；100–102 另有 data/transpose `TBuf`，211/231 另有 32 B tail `TBuf`；Host 固定声明 32 B user workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档与 `support_aclnn` OpDef、Host/Tiling/POD、全部 14 个 router/body、A2/A3 binary config 和全局 AscendC 构建接线闭合；自动生成的 ACLNN inner bridge 作为框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

将 `updates` 视为 `[N, D1, D2, D3]` 的展平逻辑域，将每个 `varRef[n]` 视为 `[D1, VD2, D3]`。`N=len(varRef)=updates.shape[0]`。当 `mask` 缺省或 `mask[n]==1` 时：

- rank-1 `indice`：以 `indice[n]` 为 `axis` 方向起点，覆盖长度为 `D2`；等价地 `varRef[n][d1, indice[n]+d2, d3] = updates[n,d1,d2,d3]`。
- rank-2 `indice`：`indice[n,0]` 为起点，`indice[n,1]` 为有效长度 `L`，仅覆盖 `d2<L`；Kernel 不写其余位置。
- `axis` 先按 `axis<0 ? updates.rank+axis : axis` 归一化，必须满足 `0<axis<updates.rank`。Host 将轴前维折为 `D1`、轴维记为 `D2/VD2`、轴后维折为 `D3`。

公开文档把接口描述为广播关系，但固定 Host 实现实际要求：所有 `varRef` shape 完全一致，轴前后折叠长度分别与 updates 相等，且 `D2<=VD2`。索引越界未在 Kernel 内防护，属于调用方约束。公开文档标明默认非确定性，可由运行时系统参数请求确定性；固定仓未给出该运行时策略的算子内实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | TensorList；每项 ND，Host 要求每项 shape 相同 | INT8/INT16/INT32/UINT8/UINT16/UINT32/FLOAT16/BFLOAT16/FLOAT | 输入/输出原位对象；非空；dtype 与 updates 相同。 |
| `indice` | ND；实际 Tiling 仅 rank 1 `[N]` 或 rank 2 `[N,2]` | INT32/INT64 | 首维必须为 `N`；rank 2 第二列为有效长度；越界不支持。 |
| `updates` | ND，rank≥2，首维 N | 与 `varRef` 相同 | 各维必须为正；轴前/后与 var 等长，轴长不大于 var。 |
| `maskOptional` | 缺省或 rank 1 `[N]` | UINT8 | 缺省表示全部更新；Device 仅在值为 1 时更新，100–107 的判断写作 `==0` 跳过。 |
| `reduceOptional` | Host 字符串 | string | 缺省 `"update"`，Tiling 对其他值返回失败。 |
| `axis` | Host 标量 | int64_t | 文档默认 -2；归一化后必须落在 `[1, updates.rank-1]`。 |
| `workspaceSize/executor` | Host 输出 | uint64_t / executor | 第一段产生；算子 Tiling 固定 user workspace 为 32 B。 |

### 入口与版本

固定仓公开唯一版本族为 `aclnnScatterListGetWorkspaceSize` + `aclnnScatterList`，无 `V2` 或更高公开符号。API 文档给出两段式原型、A2/A3 支持表和错误码；`ScatterList` OpDef 的 `ascend910b`、`ascend910_93` config 均设置 `aclnnSupport.value=support_aclnn`。固定仓未提交 ACLNN inner bridge C++，其 tensor 连续化、executor 壳和第二段框架 launch 作为自动生成边界；算子拥有的 `ScatterList` Host/Tiling、POD、entry、body 与构建接线均已追到实际实现。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口先拒绝空 tensor/TensorList、非法 var/indice dtype、非 `update` reduce 和 var/updates dtype 不同。进入算子 Tiling 后按源码顺序：读取 AIV 核数与 UB；读取动态 var 实例数并验证非空、各项 shape 相等；令 `indiceIndex=N`、`updatesIndex=N+1`；验证 updates/var shape 与 axis；验证 indice rank/shape；验证可选 mask rank/shape；最后选择 Route、写 POD、`blockDim`、key 和 32 B workspace。任一步失败均返回 `GRAPH_FAILED`，不启动 Kernel。

非连续 tensor 如何被生成式 ACLNN bridge 转为底层输入固定仓不可见；这不改变下游 ND storage shape、TensorList 指针表 ABI 与实际 Host selector。

### 参数、Workspace 与生命周期

`Tiling4ScatterList` 以平台 `AIV core count` 和 UB bytes 为资源上界。所有 shape 乘积、对齐、Queue bytes 与 key 使用 `int64_t`；源码未对 shape 连乘单独做溢出检查。Tiling 写 `workspaces[0]=32`；Kernel entry 要求传入 workspace 非空并通过 `GetUserWorkspace` 得到非空 `userWS`，随后各 Route class 不再读写该地址。因此 32 B 是 launch ABI/框架 user-workspace 保留，不是跨 Kernel 数据边。executor 与 workspace 由第一段返回/调用方分配，第二段在给定 stream 上异步启动，调用方在 executor 完成前保持输入、输出、workspace 有效。

### 启动图与失败传播

单 Kernel 启动图：

```text
aclnnScatterListGetWorkspaceSize
  -> generated ACLNN bridge (固定仓不可见)
    -> ScatterList / Tiling4ScatterList
      -> POD + blockDim=useCoreNum + tilingKey + workspace[0]=32
aclnnScatterList(workspace, 32, executor, stream)
  -> scatter_list(var-list, indice, updates, mask, varOut, workspace, tiling)
    -> GetUserWorkspace(workspace)
    -> key router -> Route::Init -> Route::Process
    -> 原位更新 var-list
```

节点只有 `scatter_list`：grid=`useCoreNum`，POD=`ScatterListTilingData`，输入为 var TensorList 指针表/indice/updates/可选 mask，输出通过 var 指针原位提交；`varOut` 参数未被 body 使用。workspace 无 producer→consumer 数据边。Host Tiling 失败阻断 launch；entry 遇到 null workspace、null user workspace 或未知 key 静默返回，固定 Device entry 不写错误码。

## Route 与参数设计

### 分核与资源计算

令 `B=D0*D1`、`blockElems=32/sizeof(T)`、`indiceUbSize=align(indice元素数,32/sizeof(I))*sizeof(I)`、`maskUbSize=mask存在?align(N,32):0`。普通路线初始 `preCoreBatchNum=ceil(B/coreNum)`、`useCoreNum=ceil(B/preCoreBatchNum)`、`lastCoreBatchNum=B-preCoreBatchNum*(useCoreNum-1)`；220 会把域改为 `B*D2` 再分核。

对最后轴路线，`maxUb=UB-indiceUbSize-maskUbSize`。100–102 为 transpose 专化；失败后 103–107 按 `maxUb` 是否能容纳每核、整 `D2`、单个对齐行或仅分片 `D3` 依次降级。非最后轴路线令 `maxUpdatesUbCount=(UB-indiceUbSize-maskUbSize-128)/sizeof(T)`，依次选择 200/210/220/230 或未对齐 211/231。所有 Queue depth 均为 1；无 double buffer。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 transpose-small | 100 / `ScatterListTransposeSmall` | last-axis 分支；`VD2`、`D2` 均 block 对齐，`D3==1`，indice rank1，`sizeof(T)!=8`，且 `maxUb>=dataUb+perCoreUpdatesUb+transposeUb` | R-002 |
| 2 | R-002 transpose-more | 101 / `ScatterListTransposeMore` | 同上但 R-001 未命中，且 `maxUb>=dataUb+oneBatchUpdatesUb+transposeUb` | R-003 |
| 3 | R-003 transpose-large | 102 / `ScatterListTransposeLarge` | 同上但 one-batch transpose 仍放不下，按 VD2 分片 | R-004 |
| 4 | R-004 pad-small | 103 / `ScatterListNegSmall` | fallback；`D3` block 对齐且 `maxUb>=perCoreUpdatesUb` | R-005 |
| 5 | R-005 pad-more | 104 / `ScatterListNegMore` | 前项失败；`maxUb>=D2*align(D3)*sizeof(T)` 且 `D2<=4095` | R-006 |
| 6 | R-006 pad-large | 105 / `ScatterListNegLarge` | 前项失败；`maxUb>=align(D3)*sizeof(T)`，按 D2 分片且每片≤4095 | R-007/008 |
| 7 | R-007 large-E | 106 / `ScatterListNegLargeE` | 上述均失败，`maxUb<align(D3)*sizeof(T)`，indice rank1，按 D3 分片 | 失败边界 |
| 8 | R-008 large-E-rank2 | 107 / `ScatterListNegDim2Large` | 与 106 相同但 indice rank2 | 失败边界 |
| 9 | R-009 row-small-batch | 200 / `ScatterListRSBSE` | 非 last-axis；`D3` block 对齐且 `preCoreBatchNum*D2*D3<=maxUpdatesUbCount` | R-010 |
| 10 | R-010 row-large-batch | 210 / `ScatterListRLBSE` | 对齐且前项失败，`D2*D3<=maxUpdatesUbCount` | R-012/013 |
| 11 | R-011 row-large-batch-pad | 211 / `ScatterListRLBSEPad` | 非 last-axis；`D3` 不对齐且 `align(D2*D3)<=maxUpdatesUbCount` | R-014 |
| 12 | R-012 row-split-D2 | 220 / `ScatterListRSBLE` | 对齐、200/210 失败，`B<coreNum` 且 `ceil(B*D2/coreNum)*D3<=maxUpdatesUbCount` | R-013 |
| 13 | R-013 row-loop | 230 / `ScatterListRLBLE` | 对齐路线其余情况，按 `D2*D3` 分片 | 失败边界 |
| 14 | R-014 row-loop-pad | 231 / `ScatterListRLBLEPad` | `D3` 不对齐且 211 失败，按 `D2*D3` 分片 | 失败边界 |

```text
if ChooseGetOtherData() == true:             # A2/A3 上 last-axis 进入
  if transpose-small predicates: 100
  elif transpose-more predicates: 101
  elif transpose-large predicates: 102
  elif D3 aligned and per-core updates fit: 103
  elif whole D2*aligned-D3 fits and D2<=4095: 104
  elif one aligned-D3 row fits: 105
  else: 107 when indice rank==2, otherwise 106
else:
  if D3 aligned:
    if per-core B slice fits: 200
    elif one B item fits: 210
    elif split-D2 guard fits: 220
    else: 230
  else:
    if aligned one B item fits: 211
    else: 231
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 100 | 见选择表；T 为 1/2/4 B | 单 AIV `scatter_list` | `ScatterListTransposeSmall<T,I>`；router 按 sizeof 映射 int8/half/float | 全 per-core updates + data/transpose；MTE2→V→MTE3 | R-002 |
| R-002 | 101 | 同 transpose 守卫，一次一 B item | 单 AIV | `ScatterListTransposeMore<T,I>` | Queue 复用；每 item GM→UB | R-003 |
| R-003 | 102 | transpose 守卫，UB 不足 | 单 AIV | `ScatterListTransposeLarge<T,I>` | VD2 分片；双 TBuf 非双缓冲 | R-004 |
| R-004 | 103 | last-axis fallback、per-core fit | 单 AIV | `ScatterListNegSmall<T,I>` | 直接 `DataCopy(Pad)`，无 calc TBuf | R-005 |
| R-005 | 104 | whole D2 fits | 单 AIV | `ScatterListNegMore<T,I>` | 一个 batch slice 复用 updates Queue | R-006 |
| R-006 | 105 | one D3 row fits | 单 AIV | `ScatterListNegLarge<T,I>` | D2 loop，tail 用 byte-length Pad | R-007/008 |
| R-007 | 106 | D3 也需分片，rank1 | 单 AIV | `ScatterListNegLargeE<T,I>` | D3 loop | 无更多 fallback |
| R-008 | 107 | D3 也需分片，rank2 | 单 AIV | `ScatterListNegDim2Large<T,I>` | 按运行时 length 分片，DataCopyPad | 无更多 fallback |
| R-009 | 200 | aligned、per-core fit | 单 AIV | `ScatterListRSBSE<T,I>` | 一次搬入整 core slice | R-010 |
| R-010 | 210 | aligned、one batch fit | 单 AIV | `ScatterListRLBSE<T,I>` | 每 batch Queue 生命周期 | R-012/013 |
| R-011 | 211 | unaligned、one batch fit | 单 AIV | `ScatterListRLBSEPad<T,I>` | updates+32 B tail TBuf | R-014 |
| R-012 | 220 | aligned、可沿 D2 增核 | 单 AIV | `ScatterListRSBLE<T,I>` | 每核若干 D3 行 | R-013 |
| R-013 | 230 | aligned、需 loop | 单 AIV | `ScatterListRLBLE<T,I>` | `eachLoopNum` 分片 | 无更多 fallback |
| R-014 | 231 | unaligned、需 loop | 单 AIV | `ScatterListRLBLEPad<T,I>` | 分片 + 32 B tail TBuf | 无更多 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| MovePad 能力 | A2/A3 对应 `ASCEND910B`/regbase，`supportMovePad=true` | Device 可用字节长度 `DataCopyPad`，last-axis 可进入 100–107 | R-001～008 | 普通 200～231 | `scatter_list_tiling.cpp::GetPlatformData/ChooseGetOtherData` |
| 32 B block | `updatesOneBlock=32/sizeof(T)` | 普通 `DataCopy` 的 block 对齐 | 全 Route | pad 或更细分片 | `GetVarAndUpdateData` |
| transpose dtype | `sizeof(T)!=8` | router 只实例化 1/2/4 B transpose body | R-001～003 | R-004～008 | `GetDataNeg` + `scatter_list` |
| repeat/片长 | transpose large 将 `eachPreLoopEle` 对齐 16；1 B 再对齐 32；pad large 限制 D2 tile≤4095 | `TransDataTo5HDImpl` repeat 与 DataCopy blockCount 范围 | R-003/R-006 | 更小分片 | `GetDataNeg/GetOtherDataNegMovePad` |
| 单核回退 | 非 MovePad 平台且非对齐小块时可能 `totalCoreNum=1` | 防止多核非对齐覆盖；A2/A3 MovePad 路线不触发 | 非 A2/A3 事实 | 单核 | `GetOtherData` |
| workspace 非空 | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 在解析 tiling 前检查 | 全 Route | 静默 return | `scatter_list` |
| mask 值 | 200～231 仅 `mask==1` 更新；100～107 仅 `mask==0` 跳过 | 防止无效 batch 写 GM | 全 Route | 不写对应 var | 各 body `CopyOut*` |

### POD/ABI

`ScatterListTilingData` 由 35 个连续声明的 `int64_t` 字段组成；每字段 64 bit。固定仓不包含外部 `register/tilingdata_base.h` 宏定义，因此 `sizeof/alignof/offsetof` 的编译器实体固定仓不可见；Host `set_*`、序列化顺序与 Device `ParseTilingData` 字段逐一同序闭合。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| ScatterListTilingData | `dim0Count` | int64/64 | 第1字段；精确 offsetof 固定仓不可见 | element-axis count | updates dim0 | batch→var list index | >0 |
| 同上 | `dim1Count` | int64/64 | 第2字段 | flattened rows | `ReShape/CalculateParams` | batch div/mod | >0 |
| 同上 | `varDim2Count` | int64/64 | 第3字段 | elements | var axis extent | transpose/strided copy count | ≥D2 |
| 同上 | `dim2Count` | int64/64 | 第4字段 | elements | updates axis extent | loops/split | >0 |
| 同上 | `dim3Count` | int64/64 | 第5字段 | elements | suffix product | copy length/offset | >0 |
| 同上 | `dim3CountAlign` | int64/64 | 第6字段 | blocks或elements，Route 相关 | CeilAlign/CeilDiv | pad stride | inactive route 为0或重解释值 |
| 同上 | `updatesOneBlock` | int64/64 | 第7字段 | elements/32 B | `32/sizeof(T)` | align/tail | 4/8/16/32 |
| 同上 | `indiceDims` | int64/64 | 第8字段 | rank | indice shape | EqLen/NotEqLen | 1或2 |
| 同上 | `indiceCount` | int64/64 | 第9字段 | aligned elements | indice count align | GM→UB | ≥N或2N |
| 同上 | `indiceUbSize` | int64/64 | 第10字段 | bytes | count*sizeof(I) | InitBuffer | 32 B aligned |
| 同上 | `maskCount` | int64/64 | 第11字段 | aligned elements | optional mask | GM→UB | 缺省0 |
| 同上 | `maskUbSize` | int64/64 | 第12字段 | bytes | optional mask | conditional InitBuffer | 缺省0 |
| 同上 | `srcBatchStride` | int64/64 | 第13字段 | elements | D2*D3 | updates offset/copy | >0 |
| 同上 | `srcBatchStrideAlign` | int64/64 | 第14字段 | elements | align(src stride)或tail align | pad/transpose tail | Route 相关 |
| 同上 | `dstBatchStride` | int64/64 | 第15字段 | elements | VD2*D3 | var row offset | >0 |
| 同上 | `useCoreNum` | int64/64 | 第16字段 | AIV cores | ceil division | Process guard | blockDim 同值 |
| 同上 | `preCoreBatchNum` | int64/64 | 第17字段 | batch items/core | ceil(B/cores) | core base | >0 |
| 同上 | `lastCoreBatchNum` | int64/64 | 第18字段 | batch items | remainder | last core | >0 |
| 同上 | `eachLoopNum` | int64/64 | 第19字段 | loops | large-route ceil division（部分路线存前段 loop 数） | inner loop | Route 相关 |
| 同上 | `eachPreLoopEle` | int64/64 | 第20字段 | elements | UB tile | per-loop copy | Route 相关 |
| 同上 | `eachLastLoopEle` | int64/64 | 第21字段 | elements | tail remainder | final copy | Route 相关 |
| 同上 | `eachLastLoopEleAlign` | int64/64 | 第22字段 | elements | tail CeilAlign | tail GM→UB | Route 相关 |
| 同上 | `updatesCount` | int64/64 | 第23字段 | elements | regular route tile | DataCopy input | 200–231 |
| 同上 | `updatesUbSize` | int64/64 | 第24字段 | bytes | tile*sizeof(T) | updates Queue | 全 Route |
| 同上 | `dataUbSize` | int64/64 | 第25字段 | bytes | transpose calc | data TBuf | 100–102，其余0 |
| 同上 | `transposeUbSize` | int64/64 | 第26字段 | bytes | transpose calc | transpose TBuf | 100–102，其余0 |
| 同上 | `transRepeatTimes` | int64/64 | 第27字段 | repeats | aligned VD2 / 16 | TransDataTo5HD | 100–102 |
| 同上 | `transRepeatTimesTail` | int64/64 | 第28字段 | repeats | tail aligned /16 | large tail transpose | 102 |
| 同上 | `updateDim23Align` | int64/64 | 第29字段 | elements | aligned update tile | GM→UB/transpose | 100–105 |
| 同上 | `preCoreUpdateDim23` | int64/64 | 第30字段 | elements | per-core/tile stride | updates base | 100/103/105 |
| 同上 | `varDim3Stride` | int64/64 | 第31字段 | bytes或32 B blocks，按 DataCopy API | `(VD3-D3)*sizeof(T)`；非 MovePad 分支改 block | strided MTE | API 必须匹配 Route |
| 同上 | `varDim3Count` | int64/64 | 第32字段 | elements | var suffix product | GM offset | >0 |
| 同上 | `dim3CountSize` | int64/64 | 第33字段 | bytes或blocks，按 Route | `D3*sizeof(T)`；非 MovePad 分支为 blocks | `blockLen` | API 必须匹配 Route |
| 同上 | `eachLastSize` | int64/64 | 第34字段 | bytes或elements，按 consumer | Tiling tail formula | tail/transpose | Route 相关 |
| 同上 | `tilingKey` | int64/64 | 第35字段 | enum value | selector | ParseTilingData/诊断；router 用 launch key | 100…231 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～003 | 100/101/102 | `GetDataNeg` | transpose/UB/core 字段 | `scatter_list` | `scatter_list_transpose_{small,more,large}.h` + base transpose helper | A2/A3 binary JSON + `ascendc_config.json` |
| R-004～008 | 103…107 | `GetOtherDataNegMovePad` | pad/tile/core 字段 | `scatter_list` | `scatter_list_neg_{small,more,large,large_e}.h`、`scatter_list_neg_dim2_large.h` | 同上 |
| R-009～014 | 200/210/211/220/230/231 | `GetOtherDataNeg2` 及各 Get*Data | regular tile/core 字段 | `scatter_list` | `scatter_list_{rsbse,rlbse,rlbse_pad,rsble,rlble,rlble_pad}.h` | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 共用 entry 参数顺序 `(var, indice, updates, mask, varOut, workspace, tiling)`。entry 先验证 workspace，再 `GET_TILING_DATA`，按 key 实例化一个 class。class `Init` 读取 `GetBlockIdx`、解析全部 POD、绑定 GM，按 `mask!=nullptr` 条件分配 mask Queue；`Process` 用 `useCoreNum` 防越界并区分 former/last core。`GetTensorAddr` 从 TensorList 描述首地址读取指针表 offset，再按 `dim0Idx` 取真实 var GM 地址。Queue 的 `Alloc/EnQue/DeQue/FreeTensor` 管理同一 depth=1 子区；MTE→Scalar、MTE2→MTE3、MTE2→Vector、Vector→MTE3 由显式 event `SetFlag/WaitFlag` 建立依赖。没有跨 Kernel workspace 边、atomic 或跨核 barrier。

### Route R-001：transpose-small

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetDataNeg` | aligned VD2/D2，D3=1，rank1，T非8B，三块 UB 总和可容纳 | key100，否则101 |
| Router | `op_kernel/scatter_list.cpp` | `scatter_list` | `TILING_KEY_IS(100)`；按 sizeof(T) 选 int8/half/float | `ScatterListTransposeSmall` |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListTransposeSmall::Init -> Process
  -> CopyIn(all per-core updates + global indice/mask)
  -> CopyOut -> GM var gather -> TransposeB{8,16,32}
     -> overwrite transposed slice -> inverse transpose -> strided GM var store
```

##### 源码等价伪码

```text
for local_b in core_batch:
  n=global_b/D1; skip if mask[n]==0
  load VD2 strided var lanes -> dataUb
  transpose(dataUb, transposeUb); replace D2 slice from resident updatesUb
  inverse_transpose(transposeUb, dataUb); store VD2 blocks to var[n]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updates Queue | `core*preCoreUpdateDim23` | `updatesUbSize` B | T/flat | CopyIn | transpose replace | pipe/整 core | depth1 |
| data/transpose TBuf | local | 各 `dataUbSize`/`transposeUbSize` B | T/blocked transpose | GM/transpose | inverse/store | pipe/Process | 两独立子区 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  indice["indiceQ · depth=1 · indiceUbSize"]:1
  mask["maskQ · depth=1 · maskUbSize(可选)"]:1
  updates["updatesQ · depth=1 · updatesUbSize"]:1
  data["dataBuf · dataUbSize"]:1
  trans["transposeBuf · transposeUbSize"]:1
```

##### 字节账与复用顺序

```text
UB_total = indiceUbSize + maskUbSize + updatesUbSize + dataUbSize + transposeUbSize <= platform UB
dynamic_ub = N/A
workspace_total = 32 B（未承载中间数据）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| data/transpose | GM load/transpose | inverse transpose/store | 下一 batch | MTE2→V、V→MTE3、MTE3→MTE2 | 单 batch 串行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| transpose pipeline | hard events | MTE2/Vector | Vector/MTE3/next MTE2 | TBuf与GM | 纯位搬运，不 cast；T 为1/2/4B；repeat=1 时 stride 清零 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetDataNeg` | key100 与总 UB 守卫 |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | `scatter_list` | sizeof→模板映射 |
| Device body/helper | `op_kernel/scatter_list_transpose_small.h` / `scatter_list_base.h` | `CopyOut` / `TransposeB*` | gather-transpose-replace-inverse-store |

### Route R-002：transpose-more

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetDataNeg` | R-001失败，one-batch updates + 两TBuf可容纳 | key101，否则102 |
| Router | `op_kernel/scatter_list.cpp` | `scatter_list` | key101 + sizeof(T) | `ScatterListTransposeMore` |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListTransposeMore::Process -> CopyIn(indice/mask)
  -> per batch: DataCopy updates -> CopyOutMore -> TransposeB* -> GM store
```

##### 源码等价伪码

```text
for b in core_batch:
  wait previous MTE3; load one update slice
  gather var strided column; transpose; replace; inverse; store
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | `global_b*srcBatchStride` | one aligned batch | T | per-loop MTE2 | transpose | whole Process | 每 batch 覆盖 |
| data/transpose | local | 各自 Tiling bytes | T/blocked | var/transpose | store | Process | 与 R-001 同步 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  indice["indiceQ · 1 · indiceUbSize"]:1
  mask["maskQ · 1 · maskUbSize"]:1
  updates["updatesQ · 1 · updatesUbSize"]:1
  data["dataBuf · dataUbSize"]:1
  trans["transposeBuf · transposeUbSize"]:1
```

##### 字节账与复用顺序

```text
UB_total = indiceUbSize + maskUbSize + updatesUbSize + dataUbSize + transposeUbSize <= UB
dynamic_ub=N/A; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | per-batch MTE2 | transpose replace | next batch | MTE3→MTE2 | store完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| batch reuse | `Mte3ToMte2` | GM store | next update load | updatesQ/dataBuf | 无算术；mask跳过不写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetDataNeg` | one-batch UB guard |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key101 | More 模板 |
| Device body/helper | `op_kernel/scatter_list_transpose_more.h` | `CopyOutMore/TransposeB*` | Queue 循环复用 |

### Route R-003：transpose-large

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetDataNeg` | transpose predicate 且 `!istransMore`；tile 对齐16（1B再对齐32） | key102 |
| Router | `op_kernel/scatter_list.cpp` | key102 | sizeof(T) | Large body |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListTransposeLarge::Process -> CopyIn(indice/mask)
  -> CopyOut -> for VD2 tile: CopyOutPre -> GM load/update -> TransposeB* -> GM store
```

##### 源码等价伪码

```text
for b:
  for tile in [full tiles, tail]:
    load aligned update tile; gather corresponding VD2 var tile
    transpose/replace/inverse; store only onceNum valid rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | `global_b*VD2 + tile*eachPreLoopEle` | `eachPreLoopEle*sizeof(T)` | T | MTE2 | transpose | tile | tail用srcBatchStrideAlign |
| data/transpose | local | `eachPreLoopEle*blockElems*sizeof(T)` 各一份 | blocked | gather | store | tile | repeat tail 单独字段 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · tile"]:1
  data["dataBuf · tile×block"]:1
  trans["transposeBuf · tile×block"]:1
```

##### 字节账与复用顺序

```text
UB_total = indiceUbSize + maskUbSize + updatesUbSize + dataUbSize + transposeUbSize <= UB
workspace_total=32 B; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 全部 tile buffer | `CopyOutPre` | GM store | next inner loop | MTE3→MTE2 | `onceNum/copyNum/repeatTimes` 使用相同 tile |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| full/tail | events + separate repeat fields | current tile | next tile | TBuf | tail `onceNum=eachLastLoopEle`，搬入对齐、写出仅有效 rows |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetDataNeg` | tile/repeat/tail 推导 |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key102 | Large 映射 |
| Device body/helper | `op_kernel/scatter_list_transpose_large.h` | `CopyOutPre` | tile 地址与有效 tail |

### Route R-004：pad-small

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | D3 block 对齐且 per-core updates fit | key103 |
| Router | `op_kernel/scatter_list.cpp` | key103 | 任意配置 dtype | NegSmall |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListNegSmall::CopyIn -> CopyOut
  -> DataCopySmallPad(A2/A3) -> GM var
```

##### 源码等价伪码

```text
load whole core updates; for b: read indice[/length]; DataCopyPad strided rows directly to var
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Queue组 | core updates + full indice/mask | `updatesUbSize+indiceUbSize+maskUbSize` | flat T/I/U8 | CopyIn | direct GM copy | core | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ · indiceUbSize"]:1
  mask["maskQ · maskUbSize"]:1
  updates["updatesQ · updatesUbSize"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; dynamic_ub=N/A; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | one CopyIn | all batch writes | Process结束 | FreeTensor | 只读驻留 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue→scalar/MTE3 | MTE2→S | CopyIn | indice/mask read | Queue | rank2 byte-length Pad 精确写有效 length |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | key103 |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key103 | NegSmall |
| Device body/helper | `op_kernel/scatter_list_neg_small.h` | `DataCopySmallPad` | rank1/rank2 strided写 |

### Route R-005：pad-more

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | R-004失败，整 D2×align(D3) fit 且 D2≤4095 | key104 |
| Router | `op_kernel/scatter_list.cpp` | key104 | direct | NegMore |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListNegMore::Process -> CopyIn(indice/mask)
  -> per batch GM updates -> updatesQ -> CopyOutSmall/DataCopyPad -> var
```

##### 源码等价伪码

```text
for b: wait previous store; padded-load one update slice; byte-strided padded-store to var
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | global_b*srcStride | D2*align(D3)*sizeof(T) | padded rows | DataCopyPad | CopyOutSmall | Process | batch复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · whole D2"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | batch load | GM store | next batch | MTE3→MTE2 | 串行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pad load/store | `DataCopyPad` + events | GM | GM | updatesQ | `dim3CountSize` 为 bytes，rank2 length 限定 blockLen |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | key104/D2上限 |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key104 | NegMore |
| Device body/helper | `op_kernel/scatter_list_neg_more.h` | `CopyOutSmall` | padded row网络 |

### Route R-006：pad-large

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | 一行 aligned D3 fit；D2 tile=`min(maxUb/rowBytes,4095)` | key105 |
| Router | `op_kernel/scatter_list.cpp` | key105 | direct | NegLarge |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListNegLarge::CopyOut -> D2 tile loop
  -> CopyOutOnce -> padded GM load -> padded strided GM store
```

##### 源码等价伪码

```text
for batch: for D2 tile including tail: load tile; apply rank-dependent offset/length; store valid rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | `global_b*srcStride+tile*eachPreLoopEle*D3` | tile*align(D3)*sizeof(T) | padded | MTE2 | MTE3 | tile | tail复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · D2 tile"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | CopyOutOnce load | same-call store | next tile | MTE3→MTE2 | tile≤4095 rows |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| D2 tail | Pad API | MTE2 | MTE3 | updatesQ | final `copyNum=eachLastLoopEle`，rank2 blockLen由运行时 length裁剪 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | D2 tile |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key105 | NegLarge |
| Device body/helper | `op_kernel/scatter_list_neg_large.h` | `CopyOutOnce` | 分片地址/Pad tail |

### Route R-007：large-E

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNegMovePad` | 单个 aligned D3 也不 fit；rank1 | key106 |
| Router | `op_kernel/scatter_list.cpp` | key106 | direct | NegLargeE |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListNegLargeE::ProcessPerCore -> CopyIn(indice/mask)
  -> CopyOut -> D3 segment loop -> CopyOnce/CopyLast -> var
```

##### 源码等价伪码

```text
flatten D1*D2 into batch; for each: segment D3 by eachPreLoopEle; direct copy to indice offset + segment
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | core/batch D3 base + segment | max aligned segment | T | MTE2 | MTE3 | segment | tail aligned load |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · D3 segment"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | CopyOnce/Last | immediate store | next segment | MTE3→MTE2 | valid tail bytes only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tail | `DataCopyPad` on A2/A3 | aligned MTE2 | byte-exact MTE3 | final segment | `eachLastSize=validBytes` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | fallback 106 | D3 tile |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key106 | NegLargeE |
| Device body/helper | `op_kernel/scatter_list_neg_large_e.h` | `CopyOnce/CopyLast` | segment offset/tail |

### Route R-008：large-E-rank2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | fallback key rewrite | 106资源条件且 `indiceDims==2` | key107 |
| Router | `op_kernel/scatter_list.cpp` | key107 | direct | NegDim2Large |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListNegDim2Large::CopyOut
  -> read offset,length -> CopyOutDim2 -> runtime segment loop -> DataCopyPad
```

##### 源码等价伪码

```text
for batch: L=indice[n,1]; if align(L)<=tile copy once else split L; final store blockLen=valid*sizeof(T)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | segment base | `updatesUbSize` | T | GM | byte exact var store | segment | length运行时裁剪 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ · [offset,length]"]:1
  mask["maskQ"]:1
  updates["updatesQ · length segment"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | CopyOnce | DataCopyPad | next segment | events | A2/A3满足Pad capability |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| runtime tail | length-derived align | indice scalar | MTE | last segment | 只写 `copyOutNum*sizeof(T)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | key107 rewrite | rank2专化 |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key107 | body映射 |
| Device body/helper | `op_kernel/scatter_list_neg_dim2_large.h` | `CopyOutDim2` | 运行时length分片 |

### Route R-009：row-small-batch

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNeg2/GetRSBSEData` | D3 aligned 且整 core slice fit | key200 |
| Router | `op_kernel/scatter_list.cpp` | key200 | direct | RSBSE |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListRSBSE::CopyIn(core slice) -> CopyOutEqLen/NotEqLen -> var
```

##### 源码等价伪码

```text
resident core updates; for B item: compute n,d1; if enabled direct contiguous copy D2*D3 or L*D3
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | preCoreBatchIdx*srcStride | preCoreSrcCount*sizeof(T) | T contiguous | CopyIn | all core stores | core | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · whole core slice"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | CopyIn | all batch writes | end | FreeTensor | D3 aligned |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| direct store | MTE2→MTE3 | updatesQ | var GM | core slice | rank2写L*D3，无额外Pad |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetRSBSEData` | core slice fit |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key200 | RSBSE |
| Device body/helper | `op_kernel/scatter_list_rsbse.h` | `CopyOutEqLen/NotEqLen` | contiguous scatter |

### Route R-010：row-large-batch

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetRLBSEData` | aligned；core slice不fit而单 batch fit | key210 |
| Router | `op_kernel/scatter_list.cpp` | key210 | direct | RLBSE |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListRLBSE::ProcessEqLen/NotEqLen
  -> per batch CopyIn -> CopyOut -> FreeTensor
```

##### 源码等价伪码

```text
for B item: load one D2*D3 update; read index/mask; direct contiguous store; reuse queues
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Queue组 | per batch + full metadata | `srcBatchStride*sizeof(T)+metadata` | flat | CopyIn | CopyOut | iteration | 每 batch alloc/free |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · one batch"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| all Queues | CopyIn | CopyOut | next batch | FreeTensor + MTE3→MTE2 | depth1串行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| loop | Queue + hard event | MTE2 | MTE3/next alloc | Queue | exact contiguous count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetRLBSEData` | single batch fit |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key210 | RLBSE |
| Device body/helper | `op_kernel/scatter_list_rlbse.h` | `Process*/CopyOut*` | per-batch lifecycle |

### Route R-011：row-large-batch-pad

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetRLBSEPadData` | D3 unaligned，aligned batch fit | key211 |
| Router | `op_kernel/scatter_list.cpp` | key211 | direct | RLBSEPad |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListRLBSEPad::Process* -> CopyIn -> CopyOutPad -> var
```

##### 源码等价伪码

```text
for batch: copy aligned update block; if valid count unaligned, bulk aligned prefix + byte-exact DataCopyPad tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | one aligned batch | `srcBatchStrideAlign*sizeof(T)` | T | MTE2 | tail writer | batch | depth1 |
| tempBuf | local | 32 B | T | legacy fallback | tail RMW | Process | A2/A3 Pad path通常不读 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ"]:1
  temp["tempBuf · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize+32<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | batch load | prefix/tail store | next batch | FreeTensor/events | valid count计算无误 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| unaligned tail | aligned DataCopy + DataCopyPad | updatesQ | var | 最后不足32 B | 仅写 valid bytes，避免邻域覆盖 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetRLBSEPadData` | aligned batch buffer |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key211 | Pad body |
| Device body/helper | `op_kernel/scatter_list_rlbse_pad.h` | `CopyOutPad` | bulk+tail策略 |

### Route R-012：row-split-D2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetOtherDataNeg2/GetRSBLEData` | aligned，B<cores，沿 D2 重分核后每核 D3 slice fit | key220 |
| Router | `op_kernel/scatter_list.cpp` | key220 | direct | RSBLE |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListRSBLE::CopyIn -> CopyOutEqLen/NotEqLen
  -> each core batch item maps to n,d1,d2 -> direct D3 store
```

##### 源码等价伪码

```text
global_item in B*D2; derive n,d1,d2; if d2<runtime L, copy one D3 vector
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | preCoreBatchIdx*D3 | preCoreBatchNum*D3*sizeof(T) | T rows | CopyIn | per-item store | core | aligned D3 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · D3 rows"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ | core CopyIn | all per-item stores | end | FreeTensor | D3 block aligned |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| rank2 length | scalar guard | indiceQ | MTE3 | D2 item | `d2<L` 才写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetRSBLEData` | B*D2重新分核 |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key220 | RSBLE |
| Device body/helper | `op_kernel/scatter_list_rsble.h` | `CopyOut*` | item→n,d1,d2映射 |

### Route R-013：row-loop

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetRLBLEData` | aligned regular路线其余情况 | key230 |
| Router | `op_kernel/scatter_list.cpp` | key230 | direct | RLBLE |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListRLBLE::ProcessEqLen/NotEqLen
  -> batch loop -> inner tile CopyIn -> CopyOut -> FreeTensor
```

##### 源码等价伪码

```text
for batch: for tile of D2*D3: load updatesCount; compute valid overlap with runtime L*D3; write min(remaining,tile)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | batch*srcStride+tile*eachPreLoopEle | `updatesCount*sizeof(T)` | T | CopyIn | CopyOut | tile | final有效长度另传 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · flat tile"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| all queues | tile CopyIn | tile CopyOut | next tile | FreeTensor + events | `haveCopyCount<needCopyCount` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| final overlap | min remaining | indice length | DataCopy | tail | `copyCount=min(need-have, tileValid)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetRLBLEData` | loop/tile/tail |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key230 | RLBLE |
| Device body/helper | `op_kernel/scatter_list_rlble.h` | `Process*/CopyOutNotEqLen` | overlap裁剪 |

### Route R-014：row-loop-pad

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/scatter_list_tiling.cpp` | `GetRLBLEPadData` | D3 unaligned 且 aligned whole batch 不fit | key231 |
| Router | `op_kernel/scatter_list.cpp` | key231 | direct | RLBLEPad |

#### Device 数据流

##### 函数级调用链

```text
scatter_list -> ScatterListRLBLEPad::Process* -> inner CopyIn
  -> CopyOutEqLen/NotEqLen -> CopyOutPad -> var
```

##### 源码等价伪码

```text
for flat tile: determine valid overlap; aligned prefix direct copy; final non-block tail DataCopyPad
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| updatesQ | flat tile | `updatesUbSize` | T | MTE2 | prefix/tail | tile | depth1 |
| tempBuf | local | 32 B | T | legacy RMW | tail | Process | Pad支持路径不需要RMW |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  indice["indiceQ"]:1
  mask["maskQ"]:1
  updates["updatesQ · flat tile"]:1
  temp["tempBuf · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=indiceUbSize+maskUbSize+updatesUbSize+32<=UB; workspace_total=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| updatesQ/tempBuf | tile load/tail merge | GM store | next tile | events/FreeTensor | `copyCountAlign`覆盖合法子区 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pad tail | prefix `DataCopy` + tail `DataCopyPad` | updatesQ | var | 最后block | 不写越过 valid count 的相邻 var |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scatter_list_tiling.cpp` | `GetRLBLEPadData` | loop/tail align |
| Kernel entry/router | `op_kernel/scatter_list.cpp` | key231 | RLBLEPad |
| Device body/helper | `op_kernel/scatter_list_rlble_pad.h` | `CopyOutPad/CopyOutNotEqLen` | overlap+byte tail |

## 算子特定优化

### Trick T-001：对齐前缀与字节精确尾块分离

#### 触发与源码依据

- 适用 Route：R-011、R-014。
- 触发条件：有效 `copyCount` 不是 32 B block 整数倍；A2/A3 的 `IsDataCopyPadSupport()` 为 true。
- Golden 来源：[`index/scatter_list/op_kernel/scatter_list_rlbse_pad.h:167-177`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_kernel/scatter_list_rlbse_pad.h#L167-L177)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (IsDataCopyPadSupport()) {
    if (copyCountAlign > m_tilingData.updatesOneBlock) {
        alignCount = copyCountAlign - m_tilingData.updatesOneBlock;
        tailCount = copyCount - alignCount;
        DataCopy(varGm[dstGmOffset], updatesUb, alignCount);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(T1)), 0, 0, 0};
        DataCopyPad(varGm[dstGmOffset + alignCount], updatesUb[alignCount], copyParams);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyCount * sizeof(T1)), 0, 0, 0};
        DataCopyPad(varGm[dstGmOffset], updatesUb, copyParams);
    }
```

#### 机制与设计原因

大部分对齐数据继续走元素计数的 `DataCopy`；仅最后不足一个 block 的 `tailCount*sizeof(T1)` 交给字节长度 `DataCopyPad`。这样避免为了保存相邻 var 元素而对整个 tile 做 GM read-modify-write，也避免用对齐长度写坏更新区外数据。`updatesOneBlock=32/sizeof(T1)` 来自 Host，`copyCountAlign=ceil(copyCount,updatesOneBlock)` 来自当前 rank/length/tile，故 prefix 与 tail 恰好覆盖有效区且不重叠。

#### 迁移边界

- 适用：目的 GM 区域尾部不足 DMA block、平台支持 byte-length MovePad 的直接覆盖更新。
- 前提：A2/A3、有效 byte length 可表示为 `uint32_t`、prefix 已按 block 对齐，源 UB 至少有对齐搬入容量。
- 失效：平台不支持 MovePad 时必须使用源码中的 32 B `tempBuf` read-modify-write fallback；更大数据仍由 R-011/R-014 的 UB selector 分流。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码给出的反例边界是直接用 `copyCountAlign` 向 GM 写出：会覆盖有效更新区外的 var 尾部。实现必须使用有效 `copyCount` 派生 byte tail，不能把 UB padding 当有效输出。

### Trick T-002：Host 以 UB 容量分三级物化 updates

#### 触发与源码依据

- 适用 Route：R-004～R-008。
- 触发条件：last-axis 路线退出 transpose 专化后，依次比较 per-core、整 D2、单 aligned D3 行与更小 D3 segment 的 bytes。
- Golden 来源：[`index/scatter_list/op_host/scatter_list_tiling.cpp:331-367`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_list/op_host/scatter_list_tiling.cpp#L331-L367)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if ((dim3Count % updatesOneBlock == 0) && (maxUbSize >= updatesUbSize)) {
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PSMALL);
    preCoreUpdateDim23 = preCoreBatchNum * srcBatchStride;
    return ge::GRAPH_SUCCESS;
}
if ((maxUbSize >= updatesUbSizeLarge) && (updatesUbSizeLarge / eSizeAlign <= MAX_COUNT)) {
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PMORE);
    updatesUbSize = updatesUbSizeLarge;
    return ge::GRAPH_SUCCESS;
}
if (maxUbSize >= updatesUbSizeLargeE) {
    tilingKey = static_cast<int64_t>(ScatterListTilingKey::TILINGKEY_PLARGE);
```

#### 机制与设计原因

selector 把同一稀疏覆盖语义分为“整 core slice 常驻”“每 batch 常驻”“D2 tile”“D3 segment”。UB 足够时减少重复 metadata/updates 搬运；UB 紧张时只缩小 updates 物化粒度，不改变 indice/mask 常驻与地址公式。`MAX_COUNT=4095` 同时约束二维搬运的 blockCount，避免 Device 才发现 API 参数超界。

#### 迁移边界

- 适用：只需数据搬运、可按相邻维度连续切片、metadata 能常驻 UB 的 scatter/update。
- 前提：Host 与 Device 对元素/字节单位一致，tile 对齐保证合法 DMA，最后片有独立有效长度。
- 失效：transpose 守卫成立时应先用 R-001～003；非 last-axis 应走 R-009～014 的不同地址网络。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但 selector 明确禁止在 `D2>4095` 时使用 key104，不能省略该 Host guard。

## 模板与实现对比

| 对比维度 | R-001～003 transpose | R-004～008 last-axis direct/pad | R-009～014 row routes | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | D3=1、rank1、T为1/2/4B、对齐 | transpose失败，按 UB 粒度降级 | 非 last-axis，按 D3 对齐与 batch/tile fit | axis、dtype size、UB、alignment |
| 模板与 dtype | sizeof映射 int8/half/float 位容器 | `DTYPE_VAR/DTYPE_INDICE` | `DTYPE_VAR/DTYPE_INDICE` | transpose helper只实现1/2/4 B |
| 循环与分核 | B分核；Large 再分 VD2 | B或B*D2分核；D2/D3 loop | B或B*D2分核；flat update loop | 扩大并行或减小UB |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 Queue + 2 TBuf | 3 Queue | 3 Queue；Pad再加32 B TBuf | 无dynamic UB/通信 window |
| 内存总占用与复用 | metadata+updates+2 calc buffer | metadata+updates tile | metadata+updates tile(+32) | 所有depth1，显式串行复用 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy + TransDataTo5HDImpl | DataCopy/DataCopyPad | DataCopy/DataCopyPad | 全为AIV；无其他执行族 |
| 同步、精度与 tail | MTE2↔V↔MTE3，repeat tail | MTE2↔MTE3，byte tail | Queue/event，Pad或length overlap | 纯位搬运，无cast/累加 |
| fallback/失败边界 | 依次到 direct/pad | 106/107为终点 | 230/231为终点 | null workspace/未知key静默返回 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档支持；OpDef `ascend910b` 与 binary config闭合 | 公开 ACLNN 文档支持；OpDef `ascend910_93` 与 binary config闭合 | 公开 ACLNN 文档明确不支持；尽管 README/OpDef/config有实现信号，构成发布冲突，不在本卡发布 |
| Host/Launcher | 共用 `Tiling4ScatterList`；MovePad capability | 共用 `Tiling4ScatterList`；MovePad capability | 不适用正式 ACLNN 卡 |
| Kernel/资源 | 共用14 key与 `scatter_list` bodies | 共用14 key与 `scatter_list` bodies | 实现信号不提升为公开支持 |

## 限制与失败边界

- `varRef` 不得为空且每项 shape 必须完全相同；updates/var 的轴前后折叠长度必须一致，`D2<=VD2`；这比 API 文档的泛化广播表述更严格。
- indice 仅支持 rank 1 或 `[N,2]`；mask 仅支持 `[N]`；reduce 仅支持 `update`；axis 归一化后不得为0或超出 rank。
- 固定 Kernel 不检查 indice 起点、length 是否负数/越界，也不检查目的区间重叠；调用者必须满足公开原型约束。Device entry 对 null workspace/user workspace 或未知 key 静默 return。
- shape 连乘、byte 乘法与部分 `uint32_t blockLen` 窄化没有显式溢出保护；复用时必须保持 Host 已验证范围。
- 公开 ACLNN inner bridge、非连续 tensor 的具体 Contiguous/ViewCopy 序列、executor 内部 owner 实现固定仓不可见；已作为生成式框架边界，不推测补齐。
- 950 公开支持表与实现接线冲突，因此不生成 `_950` 正式卡。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter_list/docs/aclnnScatterList.md` | `aclnnScatterList*` | API原型、A2/A3支持、950不支持、约束 |
| 平台说明 | `index/scatter_list/README.md` | 产品支持表 | Host实现信号及与950 ACLNN文档冲突 |
| 入口实现 | `index/scatter_list/op_host/scatter_list_def.cpp` | `ScatterList::ScatterList` | dtype、A2/A3 config、support_aclnn |
| Host/Tiling/Launcher | `index/scatter_list/op_host/scatter_list_tiling.h` | `ScatterListTilingData` | POD与key枚举 |
| Host/Tiling/Launcher | `index/scatter_list/op_host/scatter_list_tiling.cpp` | `Tiling4ScatterList` | 校验、分核、14 Route、workspace |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list.cpp` | `scatter_list` | entry/router |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_base.h` | `ParseTilingData/GetTensorAddr/TransposeB*` | POD消费、TensorList ABI、同步与transpose helper |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_transpose_small.h` | `ScatterListTransposeSmall` | key100 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_transpose_more.h` | `ScatterListTransposeMore` | key101 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_transpose_large.h` | `ScatterListTransposeLarge` | key102 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_neg_small.h` | `ScatterListNegSmall` | key103 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_neg_more.h` | `ScatterListNegMore` | key104 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_neg_large.h` | `ScatterListNegLarge` | key105 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_neg_large_e.h` | `ScatterListNegLargeE` | key106 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_neg_dim2_large.h` | `ScatterListNegDim2Large` | key107 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_rsbse.h` | `ScatterListRSBSE` | key200 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_rlbse.h` | `ScatterListRLBSE` | key210 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_rlbse_pad.h` | `ScatterListRLBSEPad` | key211 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_rsble.h` | `ScatterListRSBLE` | key220 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_rlble.h` | `ScatterListRLBLE` | key230 |
| Kernel entry/body | `index/scatter_list/op_kernel/scatter_list_rlble_pad.h` | `ScatterListRLBLEPad` | key231 |
| 构建接线 | `index/scatter_list/op_host/config/ascend910b/scatter_list_binary.json` | `op_type=ScatterList` | A2 dtype二进制实例 |
| 构建接线 | `index/scatter_list/op_host/config/ascend910_93/scatter_list_binary.json` | `op_type=ScatterList` | A3 dtype二进制实例 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `name=ScatterList` | A2/A3 compute unit与auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 35个64-bit字段同序序列化/解析 | 字段顺序、Route相关单位、inactive字段不可误读 | `ScatterListTilingData/ParseTilingData` |
| 调用顺序 | Tiling→单 entry→Init→Process→CopyIn/CopyOut | null workspace检查在tiling解析前；key未知不执行 | `Tiling4ScatterList/scatter_list` |
| workspace/内存 | user workspace固定32 B且不承载中间数据；Queue depth1 | workspace仍必须非空；UB按分配容量而非有效长度计算 | `SetTilingData/scatter_list/InitBuffer` |
| 同步 | Queue生命周期 + MTE/Vector hard events | 覆盖 Queue/TBuf 前必须保留 producer→consumer event | `ScatterListBase::Mte*` |
| 精度与尾块 | 纯位搬运，无cast；rank2 length与Pad tail限制有效输出 | 不得用aligned UB长度写GM；transpose仅1/2/4 B | `CopyOutPad/TransposeB*` |
| 失败边界 | Host校验失败阻断launch；Device null/unknown静默return | 索引越界和连乘溢出不能靠Kernel补救 | `Tiling4ScatterList/scatter_list` |
