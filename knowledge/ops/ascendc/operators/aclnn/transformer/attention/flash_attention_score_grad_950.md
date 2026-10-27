---
type: Operator
title: "FlashAttentionScoreGrad 950 ACLNN 算子知识"
description: "说明 aclnnFlashAttentionScoreGradV4 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV4.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_api/flash_attention_score_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_entry_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_deter.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_cube.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_vec.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_empty_tensor_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_presfmg_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_nz_post.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T00:00:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# FlashAttentionScoreGrad 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnFlashAttentionScoreGradV4GetWorkspaceSize` + `aclnnFlashAttentionScoreGradV4`，公开二段式 ACLNN |
| 计算语义 | 从 Q/K/V、dY、前向 softmax 统计量与 attention 输出重算 P、dS，产生 dQ/dK/dV；可选 rope、sink、PSE、mask、dropout 与 scale。 |
| 输入输出 | BSH/SBH/BNSD/BSND/TND；主输入 FP16/BF16/FP32；输出 dQ/dK/dV，另有可选 dQ-rope/dK-rope/dPSE/dSink。 |
| 执行类型 | 非空计算是单 launch `MIX_CV`：RegBase AIC Cube 与 AIV Vector 通过 workspace/跨核 flag 交接；全空 API 终点不 launch，empty key 终点是 AIV clear。 |
| Kernel 路线 | R-001 空输出；R-002 TND/实际序列 RegBase；R-003 普通 RegBase |
| 核心资源 | user/system workspace、GM、L1/L0A/L0B/L0C、UB/寄存器、跨核 flag |
| 来源闭包 | `source_closed`：V4 文档/API→L0 launcher→DAV_3510 Tiling/POD→APT entry→RegBase Cube/Vector body→构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `A=QKᵀ`。`pseType=1` 时 `S=Mask(scale·(A+pse))`，其它值时 `S=Mask(scale·A+pse)`；随后 `P=Dropout(Softmax(S),keepProb)`、`Y=PV`。反向核心为 `dV=PᵀdY`、`dP=dYVᵀ`、`dS=SoftmaxGrad(dP,P)`、`dQ=scale·dSK`、`dK=scale·dSᵀQ`。sink 存在时 softmax 最大值和分母加入 sink，并额外归约 `dSink`。矩阵乘累加与梯度 workspace 使用 FP32；低精度输出在 post 阶段转换。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| query/keyIn/value/dy | BSH、SBH、BNSD、BSND、TND | FP16/BF16/FP32 | 必选；输出 shape 与对应输入相等，`headNum>0`。 |
| softmaxMax/softmaxSum/attentionIn | 前向输出契约 | softmax 统计量 FP32；attention 与主输入族一致 | attentionIn 必选；其余由 Host/Tiling 检查。 |
| pseShift/dropMask/attenMask/sink | 广播 PSE、dropout mask、attention mask、1-D sink | 主输入族/UINT8/BOOL或UINT8/文档类型 | 可选；`paddingMask` 是保留参数。 |
| prefix/actualSeqQLen/actualSeqKvLen/startIdx | 一维累计长度/索引 | INT64 | TND/变长路线依赖；L0 转成 ND tensor。 |
| dScaleQ/K/V/Dy/O | 标量 tensor | FLOAT32 | 可选反量化 scale；缺省时 L0 分配空 FLOAT32 tensor占位，RegBase key/body按输入 dtype 决定是否消费。 |
| scale/keepProb/tokens/headNum/layout/pseType | 标量 | double/int64/string | `keepProb∈(0,1]`；seed/offset 可生成 dropout。 |
| dq/dk/dv、rope 梯度、dpse/dsink | 对应输入或广播结果 | 主输出族；dsink FP32 | dq/dk/dv 必选，其余随可选输入启用。 |

### 入口与版本

V4 是固定提交中 950 最新公开 `FlashAttentionScoreGrad` 入口；产品表只把 Ascend 950PR/950DT 标为支持，A2/A3 明确不支持。V4 合并定长和变长入口，并加入 rope、sink、seed/offset、反量化 scale 和输出 dtype 控制。执行段只调用 `CommonOpExecutorRun`。

## Host/Launcher 执行流程

### 参数校验与前处理

V4 第一段创建 executor，检查 Q/K/V/dY/attentionIn/dQ/dK/dV 非空、`headNum>0` 及输出 shape。内部执行 `ContiguousInputTensor`/`ContiguousOptionalInputTensor`，必要时 reshape、Pad D、Transpose；`l0op::FlashAttentionScoreGrad` 分配七个逻辑输出，把 `aclIntArray` 转为 INT64 tensor，执行 `INFER_SHAPE` 与一次 `ADD_TO_LAUNCHER_LIST_AICORE`。最后通过 transpose/slice/reshape/ViewCopy 提交输出。

### 参数、Workspace 与生命周期

executor 持有连续化、Pad/Transpose 和 ViewCopy 节点。算子 user workspace 由 Tiling 写入 `workspaces[0]`：先留 `RESERVED_WORKSPACE_SIZE=64 KiB`，再按 `GM_ALIGN=512 B` 切 dQ/dK/dV FP32 区、可选 drop-mask 展开、softmax-grad、dSink 与确定性归并区，末尾加 `WORKSPACE_BUFFER=20 MiB`。workspace、executor 与 backing tensor 必须保持到同一 stream 执行完成。

### 启动图与失败传播

```text
V4 GetWorkspaceSize
  -> dQ/dK/dV 全空且 dPSE 缺省/空：workspace=0，返回 executor（无节点）
  -> otherwise Contiguous/Pad/Transpose
  -> l0op::FlashAttentionScoreGrad
     -> INFER_SHAPE
     -> ADD_TO_LAUNCHER_LIST_AICORE(FlashAttentionScoreGrad) [唯一算子 launch]
        -> TilingFlashAttentionGradScore
        -> flash_attention_score_grad<...>
           -> empty clear，或 RegbaseFAG(pre -> AIC/AIV task graph -> post)
  -> Post/ViewCopy
V4(workspace, executor, stream) -> CommonOpExecutorRun
```

任一参数、shape、连续化、infer、launcher、Tiling、workspace 或 executor 失败都立即返回，不消费未初始化中间区。

## Route 与参数设计

### 分核与资源计算

Tiling 读取 AIC/AIV 数及 UB/L1/L0/L2。空输出按元素数在 AIV 间 former/tail 分配。正常路线从 B/N2/G/S1/S2/D/D1/T1/T2 推导 S1/S2/D tile、outer、`splitAxis`（BN2GS1S2、BN2S2、BN2）、blockStarts/Ends 和 sparse/deterministic key。普通非确定性 blockDim 由 `blockOuter*2` 与 AIC/AIV 比计算；确定性使用 AIC 数。含全核同步的路线设置 schedule mode 1。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 空输出 | API 无 launch 终点，或 `IsEmptyTensor=1` | dQ/dK/dV 全空且 dPSE 缺省/空时 API 返回 workspace=0；否则 DAV_3510 下任一主/rope 输出 storage size 为 0 时生成 empty key | 非空进入 registry |
| 2 | R-002 变长 RegBase | Varlen RegBase，priority 900 | actualSeqQLen tensor 存在且 shape size 非 0 | R-003 |
| 3 | R-003 普通 RegBase | Normal RegBase，priority 950 | DAV_3510 且 `softmaxInLayout==""` | Tiling 失败 |

```text
if dQ/dK/dV empty && (dPSE absent || empty): R-001 no-launch
else if DAV_3510 && IsEmptyOutput: R-001 empty-key
else if actualSeqQLen exists && size != 0: R-002
else if DAV_3510 && softmaxInLayout == "": R-003
else: GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | API no-launch / empty key | 950，全输出空或主/rope output-empty | 无 Device 节点，或 AIV clear | API early return / `EmptyTensorRegbase<T>` | workspace=0，或 100 MiB 保留槽且 GM 清零 | immediate return / former-tail |
| R-002 | 19 段 RegBase key | actual length 存在，TND/prefix 有效 | 单 launch MIX_CV，pre→main→post | `Kernel{Deter}<FAGBlockCube,FAGBlockVec>` | FP32 workspace，CrossCore flag，TND prefix/swizzle | actual sequence 裁剪 |
| R-003 | 同一 19 段 key | 无实际序列；950 normal | 同上 | BN2GS1S2/BN2S2/BN2 specialization | L2/sparse 可 swizzle，部分 D 可 NZ | 逻辑 tail；失败关闭 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 变长优先 | actualSeqQLen 非空 | 需要 prefix block 映射 | R-002 | R-003 | Varlen `IsCapable` |
| splitAxis | `SetSplitAxis` + sparse/BN2 multi-block | 决定 AIC tile 映射 | R-002/R-003 | BN2GS1S2 或失败 | `DoSparse` |
| NZ Fixpipe | BN2GS1S2，64<D<128，D%16!=0，非 FP32/FP8，swizzle，S 足够大 | 非对齐 D 的 NZ 写出/post 恢复 | R-002/R-003 | ND | `isNzOut` |
| TND swizzle | BN2S2、B<129、无零序列/尾零且 L2/无效块触发 | 重排 block | R-002 | 普通 mapping | `isTndSwizzle` |
| 确定性 | runtime deterministic + sparse 分类 | 固定归并 workspace/flag | R-002/R-003 | 非确定性 body | `GetDeterSparseTilingKey` |
| schedule | 使用 SyncAll | 要求所有核同时启动 | R-002/R-003 | 特定 TND BN2 例外 | `PostTiling` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| BaseParamsRegbase | b/n2/g/s1/s2/d/d1 | int64_t/64，各字段独立 | 顺序标准布局；固定仓未给 sizeof/offsetof 常量 | 元素 | Init/SaveTilingData | Cube/Vec 地址 | D≤768 |
| 同上 | scaleValue/keepProb | float/32，各字段独立 | 同上 | 标量 | attrs | softmax/dropout | keepProb (0,1] |
| 同上 | layout/pse/mask/sparse switches | uint32_t/32，各字段独立 | 同上 | 枚举 | tiling | vector/body | inactive 不读 |
| SplitCoreParamsRegbase | s1Outer/s2Outer/blockOuter | int64/uint32，各字段独立 | 同上 | tile/block | DoSplit/DoSparse | scheduler | 非零 |
| BlockNumListParamsRegbase | blockStarts/blockEnds | `int64_t[36]` 各数组 | 连续数组 | task | Host | per-core scheduler | unused 清零 |
| PreParamsRegbase | q/k/v/sink Factor/Total/Tail | uint64_t/64，各字段独立 | 8B 对齐，有保留 padding | 元素 | DoPreTiling | pre | 不越界 |
| PostParamsRegbase | dq/dk/dvWorkSpaceOffset | uint64_t/64，各字段独立 | 标准布局 | byte | GetWorkspaceSize | main/post | GM 对齐 |
| 同上 | drop/sfmg/dsink/deter offsets | uint64_t/64，各字段独立 | 标准布局 | byte | workspace planner | active stages | inactive 不读 |
| TCubeTiling | baseM/N/K、step/depth/share size | 外部 POD | 本仓未给 offsetof | tile/byte | Matmul tiling API | FAGBlockCube | 与 key 同步 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | API empty / `IsEmptyOutput` | API + top-level `RunEmptyTilingRegbase` | no POD / EmptyTensor POD | no entry / APT entry | no body / EmptyTensor body | op_host CMake |
| R-002 | priority 900 | Varlen RegBase | RegBase aggregate + prefix | APT→RegbaseFAG | kernel/deter + block cube/vec | Ascend950 options |
| R-003 | priority 950 | Normal RegBase | RegBase aggregate | APT→RegbaseFAG | 同上 | 同上 |

## Kernel 实现

### 公共执行流程

APT entry 从编译期 key 解出 splitAxis、dtype、TND、drop/PSE/mask、S1/S2/D 模板、deter、rope、NZ、swizzle。非空路线进入 `RegbaseFAG`：pre 清零/展开 mask 或计算 softmax-grad；main 在 AIC 做矩阵乘、AIV 做 mask/dropout/softmax-grad；post 将 FP32/NZ workspace 转成最终输出。AIC/AIV task 双缓冲用固定 flag 交接，不是多个 Host kernel。

### Route R-001：空输出清零

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | op_api/aclnn_flash_attention_score_grad.cpp | `aclnnFlashAttentionScoreGradV4GetWorkspaceSize` | dQ/dK/dV 全空且 dPSE 缺省/空 | workspace=0，无 launcher |
| Tiling | op_host/flash_attention_score_grad_tiling.cpp | `IsEmptyOutput` | 已构图后输出 size=0 | empty tiling |
| Router | op_kernel/flash_attention_score_grad_apt.cpp | `IsEmptyTensor` | 编译常量 1 | clear + return |

#### Device 数据流

##### 函数级调用链

```text
APT entry -> EmptyTensorRegbase<T>::Init -> Process -> GM gradients = 0
```

##### 源码等价伪码

```text
for enabled output: choose former/tail count; Duplicate(UB,0); DataCopy(UB, GM range)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM outputs | former/tail prefix | output elements | T/ND | AIV | API output | executor | tail |
| workspace/L1/L0 | N/A | 保留 workspace；无 Cube | N/A | N/A | N/A | executor | N/A |
| UB | per-core tile | body 固定块 | T | Duplicate | DataCopy | loop | tail mask |

##### 分层内存布局

```mermaid
flowchart LR
  Z["AIV zero"] --> UB["UB tail tile"] --> O["GM gradients"]
```

##### 字节账、交接边与复用顺序

```text
API 全空终点：workspace_total=0，且无 Device body
empty key 终点：workspace_total=100 MiB（顶层 Tiling 约定；清零 body 不消费语义 slice）
L1/L0 = N/A；UB = 当前清零 tile（固定仓未给 tiling 常量）
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| UB→GM | Duplicate | former/tail | T/ND | output | AIV pipe | loop | Host count 裁剪 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| clear | AIV 顺序 | Duplicate | DataCopy | output shard | 精确零，former/tail 不重叠 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | op_api/aclnn_flash_attention_score_grad.cpp / op_host/flash_attention_score_grad_tiling.cpp | V4 empty return / `RunEmptyTilingRegbase` | no-launch 或 former/tail |
| Kernel entry/router | op_kernel/flash_attention_score_grad_apt.cpp | empty branch | dtype instance |
| Device body/helper | op_kernel/arch35/flash_attention_score_grad_empty_tensor_regbase.h | Process | output clear |

### Route R-002：TND/实际序列 RegBase MIX_CV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35/...varlen_regbase.cpp | IsCapable | DAV_3510 && actualSeqQLen != null && size!=0 | priority 900 |
| Router | flash_attention_score_grad_apt.cpp | template key | nonempty 19-field key | RegbaseFAG |

#### Device 数据流

##### 函数级调用链

```text
entry -> RegbaseFAG -> Pre/Presfmg + SyncAll
  -> Kernel{Deter}<FAGBlockCube,FAGBlockVec>::Process
  -> SyncAll -> PostRegbase/NzPost
```

##### 源码等价伪码

```text
pre(actualSeqPrefix); sync_all
for task in blockStarts[core]..blockEnds[core]:
  AIC writes mm1/mm2 FP32 tile and flags C1/C2
  AIV waits, applies mask/pse/dropout/softmaxGrad, writes dS and flags V3/V4
  AIC waits, computes dQ/dK/dV and Fixpipe
sync_all; post_cast_or_nz_restore(valid prefixes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | TND prefix+head+tile | T1/T2×N×D | input ND | caller | Cube/Vec | caller | actual seq |
| gradient workspace | POD offsets | align(T*N,128)×D×4 或 per-core | FP32 ND/NZ | AIC | post | executor | disjoint |
| task workspace | reserved+ping/pong | baseMN/sfmgdInner | FP32 | AIC/AIV | peer | kernel | taskId&1 |
| L1/L0 | TCubeTiling | base/step/depth | fractal/FP32 | MTE/Matmul | Fixpipe | tile | DB |
| UB/register | vector tiles | S1cv×S2inner+temp | FP32/mask | GM/Fixpipe | vector | task | mask tail |

##### 分层内存布局

```mermaid
flowchart LR
  GM["GM Q/K/V/dY"] --> L1["AIC L1/L0 MM"] --> W["FP32 mm workspace"]
  W --> V["AIV mask+softmax-grad"] --> DS["FP32 dS"] --> C["AIC gradient MM"]
  C --> P["AIV post"] --> O["GM gradients"]
```

##### 字节账、交接边与复用顺序

```text
workspace = 64 KiB + align512(dQ)+align512(dK)+align512(dV)
          + optional align(dropMask)+align(sfmg*4)+align(sink*4)+deter + WORKSPACE_BUFFER
WORKSPACE_BUFFER = 20 MiB
TND qSize=align(T1*N1,128)*D；k/v analogous
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| C1/C2→V2 | AIC | ping/pong tile | FP32 | AIV | C1/C2 flags | task | next-round wait |
| V3/V4→C3/C5 | AIV | dS tile | FP32 | AIC | V3/V4 flags | task | MTE3 before set |
| grad→post | AIC | POD byte offsets | FP32 ND/NZ | AIV | SyncAll | launch | valid prefix |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cube→Vector | CrossCore flags | AIC | AIV | mm tile | FP32, S tail |
| Vector→Cube | PIPE_MTE3/MTE1 flags | AIV | AIC | dS | FP32 |
| stages | SyncALLCores + schedule mode 1 | pre/main | main/post | workspace | actual prefix |
| output | post cast/NZ restore | FP32 | output | gradients | low precision last cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp | Varlen methods | prefix/swizzle/deter |
| Kernel entry/router | op_kernel/flash_attention_score_grad_apt.cpp | entry | full key |
| Device body/helper | arch35/flash_attention_score_grad_kernel.h | Process | AIC/AIV graph |

### Route R-003：普通 RegBase MIX_CV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35/...normal_regbase.cpp | IsCapable | DAV_3510 && softmaxInLayout empty | priority 950 |
| Tiling | 同文件 | DoOpTiling | split/sparse/workspace 成功 | key/blockDim/POD |

#### Device 数据流

##### 函数级调用链

```text
entry -> RegbaseFAG -> optional pre -> Kernel/KernelDeter<BlockCube,BlockVec> -> optional post
```

##### 源码等价伪码

```text
derive task from B*N2*G*S1Outer*S2Outer; skip sparse-invalid blocks
run AIC<->AIV ping-pong graph; NZ route restores ND; otherwise cast/copy logical region
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | ((b*N+h)*S+t)*D | B×N×S×D | ND | caller | body | caller | D/S tail |
| workspace | POD offsets | q/k/v+optional slices | FP32 ND/NZ | body | post | executor | GM align |
| L1/L0/UB | tile-derived | specialization | fractal/FP32 | Cube/Vec | peer | task | flags |

##### 分层内存布局

```mermaid
flowchart LR
  GM --> C["AIC MM"] --> W["FP32 workspace"] --> V["AIV"] --> C2["AIC grad MM"] --> P["post"] --> O["GM"]
```

##### 字节账、交接边与复用顺序

```text
qSize=((B*N1-1)*S1+align(S1,128))*D
kSize=((B*N2-1)*S2+align(S2,128))*D
vSize=((B*N2-1)*S2+align(S2,128))*D1
each gradient slice=align(size*4,GM_ALIGN)
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| AIC↔AIV tile | BlockCube/Vec | taskId&1 | FP32 | peer | CrossCore flags | task | consume before overwrite |
| main→post | Fixpipe | POD offsets | FP32/NZ | post | SyncAll | launch | valid tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tasks | CrossCore/PIPE | AIC/AIV | peer | ping-pong | FP32 accum |
| sparse | validity predicate | POD | scheduler | invalid blocks | skip |
| output | post mask/cast | main | GM | logical region | no padding write |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35/flash_attention_score_grad_tiling_normal_regbase.cpp | DoOpTiling/GetWorkspaceSize/GetTilingKey | split/workspace/key |
| Kernel entry/router | arch35/flash_attention_score_grad_entry_regbase.h | RegbaseFAG | specialization |
| Device body/helper | arch35/flash_attention_score_grad_block_cube.h + flash_attention_score_grad_block_vec.h | `FAGBlockCube/FAGBlockVec` | compute bodies |

## 算子特定优化

### Trick T-001：AIC/AIV taskId 双缓冲与显式跨核 flag

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：非空 RegBase key。
- Golden 来源：[`flash_attention_score_grad_kernel.h:79-121`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel.h#L79-L121)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
```

#### 机制与设计原因

奇偶 task 使用两组 flag。生产者覆盖下一轮 workspace 前等待消费者释放同槽，AIV 读取 mm1/mm2 后反向发出可复用信号，把 Cube 与 softmax-grad 组成同 launch 流水。

#### 迁移边界

- 适用：固定 AIC:AIV 比、每 task 双槽 workspace 的 MIX_CV。
- 前提：flag ID、taskId 奇偶、PIPE 方向、offset 同时迁移。
- 失效：生产/消费次数、核比例或 CopyOut 时序改变时重做同步证明。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只复制 `SetFlag` 而漏掉下一轮 `WaitFlag[(taskId+1)&1]` 会覆盖未消费槽。

### Trick T-002：L2/无效块触发 swizzle，并为非对齐 D 选择 NZ

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：L2 超限或大量无效块且 blockOuter=aicNum；NZ 另要求 BN2GS1S2、64<D<128、D%16!=0、非 FP32/FP8、S1/S2 达阈值。
- Golden 来源：[`flash_attention_score_grad_tiling_normal_regbase.cpp:430-453`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp#L430-L453)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bool isExceedL2Cache = CheckExceedL2Cache();
bool isLargeInvalidBlk = CheckIsLargeInvalidBlk(fBaseParams);
fBaseParams.enableSwizzle =
    (isExceedL2Cache || isLargeInvalidBlk) && fBaseParams.blockOuter == fBaseParams.aicNum;
```

#### 机制与设计原因

Host 把 L2 容量与 sparse 无效块密度编码进调度；窄 D 非对齐区间用 NZ Fixpipe 写出并由 AIV post 恢复 ND，避免把不同 layout 行为混入同一 specialization。

#### 迁移边界

- 适用：同 DAV_3510 资源模型和 NZ post。
- 前提：阈值、dtype 排除、key 位和 post POD 一致。
- 失效：FP32/FP8、D≤64/D≥128、非 BN2GS1S2 或无 swizzle 时走 ND。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓已有 fallback：条件不满足时 `isNzOut=false`；不得只改 Device layout 而漏掉 Host key/post。

## 模板与实现对比

| 对比维度 | R-001 空输出 | R-002 变长 RegBase | R-003 普通 RegBase | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 顶层提前返回 | 900 | 950 | 空域、实际长度 |
| 执行族/节点图 | AIV clear | MIX_CV pre/main/post | MIX_CV pre/main/post | 是否有 attention 域 |
| M/N/K、tile 与 scheduler | N/A | actual prefix/sparse | B/N/S split | layout/length |
| workspace/L1/L0/UB | 无 Cube | FP32+TND/deter | FP32+NZ/swizzle | feature 开关 |
| AIC/AIV 或 kernel 交接 | 无 | CrossCore 双槽 | CrossCore 双槽 | 同一 task graph |
| 同步、精度与 tail | former/tail | SyncAll+prefix | SyncAll+logical tail | 防越界 |
| fallback/失败边界 | return | 转 normal | 失败 | registry/key |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V4 文档明确不支持 | V4 文档明确不支持 | 950PR/950DT 支持 |
| Host/Launcher | 最新见无后缀 V3 卡 | 最新见无后缀 V3 卡 | DAV_3510 Normal/Varlen |
| Kernel/资源 | arch22，不外推 | arch22，不外推 | APT/RegBase/MIX_CV |

## 限制与失败边界

- V4 产品表是平台边界；`arch35` 文件存在不能单独证明公开支持。
- layout 仅 BSH/SBH/BNSD/BSND/TND；shape、rope 成对性、sink 维度、keepProb、headNum 非法时失败。
- Tiling key 有 19 个维度；未知 key 无运行时通用 fallback。
- workspace offset 是字节，逻辑元素数不能替代对齐后 FP32/NZ 容量。
- POD 编译后 `sizeof/alignof/offsetof` 数值固定仓未静态列出，本卡不推测。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV4.md | V4 原型/产品表 | 950 最新公开入口 |
| 入口实现 | attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp | V4GetWorkspaceSize | 校验/executor |
| L0 launcher | attention/flash_attention_score_grad/op_api/flash_attention_score_grad.cpp | l0op::FlashAttentionScoreGrad | 单节点 launch |
| Host/Tiling | attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp | TilingFlashAttentionGradScore | 空域/registry |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp | Varlen | R-002 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp | Normal | R-003 |
| Host/POD | attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h | constants/base params | 64 KiB reserved、512B 对齐、20 MiB buffer、36-core 上界 |
| POD | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h | RegBase POD | ABI |
| Kernel entry | attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_apt.cpp | entry | key/router |
| Kernel router | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_entry_regbase.h | RegbaseFAG | pre/main/post |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel.h | Kernel::Process | flag graph |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_deter.h | KernelDeter::Process | 确定性归并/flag graph |
| Cube body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_cube.h | FAGBlockCube | Matmul/Fixpipe producer |
| Vector body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_vec.h | FAGBlockVec | mask/dropout/softmax-grad consumer/producer |
| Empty body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_empty_tensor_regbase.h | EmptyTensorRegbase::Process | R-001 clear |
| Pre body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_presfmg_regbase.h | pre/sfmg | mask/dropout/softmax-grad 前置阶段 |
| Post body | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h | PostRegbase | FP32 梯度 cast/copy |
| NZ post | attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_nz_post.h | NzPost | NZ→ND 恢复与 tail |
| 构建接线 | attention/flash_attention_score_grad/op_host/CMakeLists.txt | Ascend950 options | dependency/options |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | base/split/pre/post POD | 顺序、位宽、单位、36-core 数组、inactive 开关 | tiling_data_regbase.h |
| 调用顺序 | API→L0→Tiling→单 entry→pre/main/post | 单算子 launch；SyncAll/pipe lifetime | V4/RegbaseFAG |
| workspace/内存 | FP32 gradient/mask/sfmg/sink/deter slices | offset、align、ND/NZ、owner | GetWorkspaceSize |
| 同步 | taskId 双槽 flags | ID、PIPE、奇偶槽、set/wait | Kernel::Process |
| 精度与尾块 | FP32 accum，post cast，actual prefix | 不提前 cast、不写 padding | cube/vec/post |
| 失败边界 | selector/key/workspace/shape 失败关闭 | 不运行未注册 key | registry/APT |