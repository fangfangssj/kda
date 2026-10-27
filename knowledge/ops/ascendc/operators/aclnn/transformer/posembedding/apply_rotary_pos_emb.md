---
type: Operator
title: "ApplyRotaryPosEmb A2/A3 ACLNN 算子知识"
description: "说明 aclnnApplyRotaryPosEmbV2GetWorkspaceSize/aclnnApplyRotaryPosEmbV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/op_api/aclnn_apply_rotary_pos_emb_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/op_api/aclnn_apply_rotary_pos_emb_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmbV2.md
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/README.md
    role: primary
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_small.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab_cast.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/config/ascend910b/apply_rotary_pos_emb_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/config/ascend910_93/apply_rotary_pos_emb_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/cmake/level2_ops_list
    role: implementation
created_at: "2026-08-24T16:30:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ApplyRotaryPosEmb A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | CANN 9.1.0 最新公开接口 `aclnnApplyRotaryPosEmbV2GetWorkspaceSize` / `aclnnApplyRotaryPosEmbV2`；V2 相比旧接口新增 `rotaryMode` |
| 计算语义 | 对 query、key 的前 `D_r=cos.shape[-1]` 个元素执行 half RoPE：`y=x*cos+rotate_half(x)*sin`，结果原地更新；A2/A3 公开约束仅允许 `rotaryMode="half"` |
| 输入输出 | BSND四维或TND三维ND；四输入dtype相同且为FP16/BF16/FP32；`D_r`和query/key的`D`均为64或128；V2文档与通用README对是否允许`D>D_r`表述不同，Tiling/Kernel具备前缀旋转路线 |
| 执行类型 | `AIV_VECTOR`；入口显式声明 `KERNEL_TYPE_AIV_ONLY`，全部 key 仅使用 DataCopy 与 Vector API |
| Kernel 路线 | R-001 small FP16/FP32；R-002 small BF16→FP32；R-003 batched aggregate-output FP16/FP32；R-004 batched one-copy/one-output FP16/FP32；R-005 batched BF16→FP32 |
| 核心资源 | GM 原地 query/key、1-depth 或 2-depth VECIN、1-depth VECOUT、VECCALC `TBuf`、BF16 额外 FP32 cos/sin `TBuf`；workspace 仅用于取得 user workspace，有效 body 不读写 user workspace |
| 来源闭包 | `source_closed`：V2 声明/转发、OpDef/shape、A2/A3 Tiling/POD、key 1/3/4 router、五个 dtype 实现族和构建/二进制配置均闭合；生成式 `aclnnInner` 桥接体作为固定仓不可见边界单列 |
| 验证状态 | 已按固定 Commit 完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试 |

## 接口与语义

### 计算语义

设每个 query/key head 的完整末维为 `D`，实际旋转长度为 `D_r=cos.shape[-1]`，A2/A3 只接受 half 模式。对 `x` 的有效前缀分成 `x0=x[...,0:D_r/2]`、`x1=x[...,D_r/2:D_r]`，旋转量为 `rot=(-x1,x0)`，输出有效前缀为 `x*cos+rot*sin`。当 `D>D_r` 时 Kernel 只搬运并覆盖前 `D_r`，其余后缀保持调用前内容；query 与 key 分别使用自己的 N 轴，cos/sin 的 N 为 1并沿 head 广播。公开接口把结果原地回写 `queryRef`、`keyRef`，OpDef 内部输出同名且 shape 与相应输入一致。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `queryRef` | layout 1 为 `[B,S,QN,D]`；layout 4 为 `[T,QN,D]`；ND | FP16/BF16/FP32 | 必选输入输出；非空；`D` 为64或128；V2文档要求与key的D相等；实现中 `D>D_r` 时只更新前缀 |
| `keyRef` | `[B,S,KN,D]` 或 `[T,KN,D]`；ND | 与 query 相同 | 必选输入输出；与 query 的 B/S 或 T、D 相同，N 可不同；非空 |
| `cos` | `[B,S,1,D_r]` 或 `[T,1,D_r]`；ND | 与 query 相同 | 必选只读；A2/A3要求B/S或T与query/key相等、N=1、`D_r`为64或128；V2文档要求与sin的D相等，Tiling另检查`D>=D_r` |
| `sin` | shape 与 cos 相同；ND | 与 query 相同 | 必选只读；与 cos 完全同 shape；half 模式前半段参与负号 |
| `layout` | Host 标量 | int64 | V2 文档和旧 Tiling 接受 1-BSND、4-TND；默认值由 OpDef 定义为 1 |
| `rotaryMode` | Host C 字符串 | char 指针 | V2 新增；公开 A2/A3 约束只允许 `"half"`，默认 schema 属性为 half；旧 Tiling 未读取该属性 |
| `workspaceSize` | Host 输出 | uint64 | 第一段返回；Host Tiling 写入 `max(GetLibApiWorkSpaceSize(),16 MiB)` |
| `executor` | Host 输出/输入 | `aclOpExecutor` 指针 | 第一段生成、第二段消费；具体内部节点与连续化临时对象的代码由生成式 inner 桥接提供，固定仓不可见 |
| `workspace` | Device 地址 | void 指针 | 第二段传入；Kernel 要求非空并调用 `GetUserWorkspace`，随后五个 body 均不访问该地址 |
| `stream` | runtime stream | `aclrtStream` | 第二段提交 executor；同步仅限同一 Kernel 内 TPipe queue 生命周期 |

### 入口与版本

公开版本族包含旧 `aclnnApplyRotaryPosEmb*` 和 V2；CANN 9.1.0 在 A2/A3 的最新入口是 V2。`aclnnApplyRotaryPosEmbV2GetWorkspaceSize` 原样转发 query/key/cos/sin/layout/rotaryMode 到 `aclnnInnerApplyRotaryPosEmbGetWorkspaceSize`，第二段转发 workspace/executor/stream 到 `aclnnInnerApplyRotaryPosEmb`。算子 schema 的内部类型仍是 `ApplyRotaryPosEmb`，属性次序为 layout、rotary_mode。固定仓没有 `aclnnInner_apply_rotary_pos_emb.h` 的生成结果及函数体，因此 API 层的精确错误码分支、Contiguous 节点顺序、原地 alias 建图和 stream runner 实现不可见；公开文档给出 NULL、dtype/format/shape、layout、rotaryMode 的失败类别，正文不把文档约束冒充为可见函数体。

## Host/Launcher 执行流程

### 参数校验与前处理

公开层先执行 V2 wrapper 到生成式 inner。可见 OpDef 把四输入标为 REQUIRED/ND/AutoContiguous，允许 FP16/FP32/BF16，输出 query/key 为 REQUIRED/ND，并定义 layout 默认 1、rotary_mode 默认 half。旧 A2/A3 `ApplyRotaryPosEmbTiling::GetInputParams/CheckParams` 再逐项检查：四输入非空 tensor；rank 均为 3或4且一致；layout 只能为 1或4并与 rank 对应；query/key/cos 的 B/S 或 T 相同；cos/sin shape 相等；`D_r` 为 64或128、query `D` 为64或128且不小于 `D_r`；cos N 为1；四 dtype 相同且属于三种支持类型；总元素数均非零；`QN+KN` 不超过 uint8 最大值。旧 Tiling 不读取 rotary_mode，故非 half 的 A2/A3 拒绝点只能归于固定仓不可见的 inner 校验。

### 参数、Workspace 与生命周期

`TilingPrepare4ApplyRotaryPosEmb` 取得 AIV 核数、UB 字节数和 SoC；运行时 `GetPlatformInfo` 取得同类信息，并把 workspace 定为平台 LibApi workspace 与 16 MiB 的较大者。`SetTilingData` 把 38 个 int64 字段序列化到 raw tiling buffer，设置 `blockDim=useCoreNum`、key 1/3/4与 workspace[0]。Kernel 仅以 workspace 非空、`GetUserWorkspace(workspace)` 非空作为启动保护，route `Init` 参数虽然接收 user workspace，但不保存也不读写；workspace 所有权和释放仍由 ACLNN executor/调用者协议决定。executor 必须从第一段保留到第二段同一执行序列完成，stream 仅由固定仓不可见的 inner runner 消费。

### 启动图与失败传播

启动图为 `V2GetWorkspaceSize → aclnnInner...GetWorkspaceSize → ApplyRotaryPosEmb OpDef/InferShape → Tiling4ApplyRotaryPosEmb → ApplyRotaryPosMembaseEmbTilingClass::DoOpTiling → apply_rotary_pos_emb<<<useCoreNum>>> → route body`，第二段由 `aclnnApplyRotaryPosEmbV2 → aclnnInnerApplyRotaryPosEmb` 提交。可见 Host 任一空指针、shape、dtype、UB、核数、`QN+KN` 检查失败均返回 `GRAPH_FAILED`，不写有效 launch；Kernel workspace/userWS 为空或未知 key 时静默 return。inner 如何把图错误映射为 aclnnStatus、如何处理 AutoContiguous/原地回写失败，固定仓不可见。

## Route 与参数设计

### 分核与资源计算

令 `AB=B*S`（TND 时内部置 B=1、S=T），`preCoreBatch=ceil(AB/totalCoreNum)`，`useCoreNum=ceil(AB/preCoreBatch)`，尾核 batch 为 `AB-(useCoreNum-1)*preCoreBatch`。`oneBlock=32/dtypeSize`，BF16 的计算元素按 FP32 令 `oneBlockFp32=8`，其余为 `oneBlock`；`mask=min(D_r,64)` 对 FP32/BF16、`min(D_r,128)` 对 FP16。基础字节为 `qPart1=(QN+KN)*D_r*castSize`、`cosPart1=D_r*dtypeSize`、`q2q1Part1=(QN+KN)*D_r*castSize`、`sin1=D_r*castSize`，其中 BF16 `castSize=4`。

small 条件使用 `speUb=2*qPart1+2*cosPart1+2*q2q1Part1+isBF16*2*sin1`。非 small 时 `oneLoop=2*qPart1+4*cosPart1+qPart1+2*q2q1Part1+isBF16*2*sin1` 必须为正且不超过 UB；`times=floor(UB/oneLoop)`，剩余空间产生 `shengMte=floor((UB%oneLoop)/(3*qPart1+4*cosPart1))`，BF16 强制为0，`times>preCoreBatch` 时截断并清零 `shengMte`。`preCBatchB=times+shengMte` 决定每轮搬运 batch；former/tail core 与轮内 `comBatch` 均用 `ComputeTimes` 形成整轮加末轮。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 small FP16/FP32 | key 1；`ARPESmall<T,T>` | `preCoreBatch==1 && speUb<=totalUbSize && dtype!=BF16` | BF16 同条件到 R-002；small 条件失败继续 R-003/R-004 |
| 2 | R-002 small BF16→FP32 | key 1；`ARPESmall<bfloat16_t,float>` | `preCoreBatch==1 && speUb<=totalUbSize && dtype==BF16` | small 条件失败到 R-005 |
| 3 | R-003 aggregate-output FP16/FP32 | key 3；`ARPEComputeAB<T,T>` | 非 small、dtype 非 BF16，`ComputeAB` 得到 `shengMte>0` | `shengMte==0` 到 R-004；UB 公式失败终止 |
| 4 | R-004 one-copy/one-output FP16/FP32 | key 4；`ARPEComputeABCast<T,T>` | 非 small、dtype 非 BF16，且 `shengMte==0` | BF16 到 R-005；UB 公式失败终止 |
| 5 | R-005 batched BF16→FP32 | key 4；`ARPEComputeABCast<bfloat16_t,float>` | 非 small且 dtype BF16；`ComputeAB` 强制 `shengMte=0` | UB 公式失败终止 |

```text
先执行 CheckParams；失败则不启动
计算分核、mask、基础 UB 字节和 speUb
若 preCoreBatch==1 且 speUb 不超过 UB：BF16 -> R-002，否则 -> R-001
否则计算 oneLoop；若 oneLoop 非法 -> GRAPH_FAILED
计算 times 与 shengMte
若 BF16 -> R-005
否则若 shengMte 大于 0 -> R-003
否则 -> R-004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 实际模板/类 | body/helper | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 small FP16/FP32 | key 1 | A2/A3；FP16/FP32；`preCoreBatch==1 && speUb<=totalUbSize` | `ARPESmall<T,T>` | `ProcessPerCore→CastCopyIn→ComputeTotary` | 输入/输出 depth1；2个计算 TBuf；cos/sin repeat 广播 | `D>D_r` 时只搬写前缀；small不满足进入 batched |
| R-002 small BF16→FP32 | key 1 | A2/A3；BF16；同一 small 谓词 | `ARPESmall<bfloat16_t,float>` | `ComputeBF16→LocalTensor2NewTensor` | depth1；按 FP32 为 Q/K 缓冲预留；额外 cos/sin FP32 TBuf | 前缀写回；small不满足进入 R-005 |
| R-003 aggregate-output FP16/FP32 | key 3 | A2/A3；非 BF16；非 small；`shengMte>0` | `ARPEComputeAB<T,T>` | `ProcessPreCore/LastCore→CopyIn→ComputeTotary→CopyOut` | 输入 depth2；`times` 大小计算 TBuf；`times+shengMte` 结果聚合后一次 MTE3 | former/tail core 与计算子轮各有末轮；UB余量不足转 R-004 |
| R-004 one-copy/one-output FP16/FP32 | key 4 | A2/A3；非 BF16；非 small；`shengMte==0` | `ARPEComputeABCast<T,T>` | `CopyIn→ComputeElse→ComputeTotary` | 输入 depth2；每搬入轮立即输出；无额外 cast TBuf | former/tail末轮；无其它 fallback |
| R-005 batched BF16→FP32 | key 4 | A2/A3；BF16；非 small；`isCast` 强制 `shengMte=0` | `ARPEComputeABCast<bfloat16_t,float>` | `ComputeBF16→CTotaryBF16` | 输入 depth2；Q/K按 FP32 容量；额外 cosC/sinC；FP32计算后舍入 | former/tail末轮；oneLoop超限直接失败 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| small 全载 | `preCoreBatch==1 && speUb<=totalUbSize` | 单核单 batch 可把 Q/K、cos/sin、旋转结果和两个乘法中间量一次放入 UB | R-001/R-002 | 进入 `ComputeAB` | `apply_rotary_pos_emb_tiling.cpp:Compute` |
| batched 最低容量 | `0 < oneLoop <= totalUbSize` | 至少容纳一批 depth2 输入、depth1 输出、两个计算 TBuf；BF16 还需两个 cast TBuf | R-003/R-004/R-005 | `GRAPH_FAILED` | `apply_rotary_pos_emb_tiling.cpp:ComputeAB` |
| MTE3 聚合余量 | `shengMte=floor(shengUb/(3*qPart1+4*cosPart1))`；仅非BF16且大于0 | 剩余 UB 只扩展输入和输出，不扩展计算 TBuf，从而多次 MTE2/计算后一次 MTE3 | R-003 | `shengMte==0` 进入 R-004；BF16 固定 R-005 | `ComputeAB` |
| repeatTimes 位宽 | `qkcNum=QN+KN<=UINT8_MAX` | `Mul` 的 repeatTimes 最终为 `uint8_t` | 全部 Route | `GRAPH_FAILED` | `Compute` |
| repeat 方向 | batched body 中比较 `qkcNum>=comBatchBB/preCBatchB` | 选择沿 head 还是沿 batch 做 repeat，并同步改变 `src1RepStride` 的广播轴 | R-003/R-004/R-005 | 走同 Route 的另一个 helper | `CTotaryBF32/ComputeTotary` |
| 部分旋转 | `qDim3>lastDim` | 完整 head 后缀不参与 RoPE，搬运参数必须跨过未旋转区 | 全部 Route | `qDim3==lastDim` 使用连续 DataCopy | `CopyInQK/CopyInPartialRotation/CopyOutPartialRotation` |
| 活跃核与尾核 | `blockIdx<useCoreNum`；最后核单独使用 `lastCoreBatch` | 防止多启动核越界，并限制末核/末轮有效 batch | 全部 Route | 无效核 return | 三个 Route 类的 `Process` |

### POD/ABI

`TILING_DATA_FIELD_DEF` 的宏实现来自仓外编译依赖；固定仓只能证明字段声明顺序和全部字段均为 int64，不能证明生成结构的精确 `sizeof/alignof/offsetof`。下表逐字段给出 producer 与真实 consumer，序号从0开始。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ApplyRotaryPosEmbTilingData` | `useCoreNum` | int64/64 | `sizeof/alignof/offsetof` 固定仓不可见；序号0 | 核 | `Compute: ceil(AB/preCoreBatch)` | 三个 route `Process` 活跃核判断 | 正数且不超过 AIV 核数 |
| 同上 | `lastDim` | int64/64 | 固定仓不可见；序号1 | 元素 | `Compute: cosDim3` | 搬运、旋转、Mul/Add 长度 | 64或128 |
| 同上 | `halfNum` | int64/64 | 固定仓不可见；序号2 | 元素 | `Compute: lastDim/2` | half 交换和 `Muls` | 32或64 |
| 同上 | `preCBatchB` | int64/64 | 固定仓不可见；序号3 | batch/轮 | `ComputeAB: times+shengMte` | batched route 每个整轮搬运数 | 必须为正 |
| 同上 | `preCBatchL` | int64/64 | 固定仓不可见；序号4 | batch | `preCoreBatch-preCBatchB*preCLTimes` | former core 末轮 | 1至preCBatchB |
| 同上 | `lastCBatchL` | int64/64 | 固定仓不可见；序号5 | batch | `lastCoreBatch-preCBatchB*lastCLTimes` | tail core 末轮 | 1至preCBatchB |
| 同上 | `comBatchBB` | int64/64 | 固定仓不可见；序号6 | batch/计算子轮 | `ComputeAB: times` | key3 内层 Vector 子轮 | 正数 |
| 同上 | `comBatchBBL` | int64/64 | 固定仓不可见；序号7 | batch | `preCBatchB-preCBBTimes*comBatchBB` | key3 整搬运轮的计算尾块 | 1至comBatchBB |
| 同上 | `comBatchBLL` | int64/64 | 固定仓不可见；序号8 | batch | `preCBatchL-preCBLTimes*comBatchBB` | key3 former core 末轮计算尾块 | 同上 |
| 同上 | `comBatchLLL` | int64/64 | 固定仓不可见；序号9 | batch | `lastCBatchL-preCLLTimes*comBatchBB` | key3 tail core 末轮计算尾块 | 同上 |
| 同上 | `qPart1Ub` | int64/64 | 固定仓不可见；序号10 | 字节/queue slot | `Compute` 基础值；`ComputeAB` 乘 preCBatchB | `InitBuffer(qIn/qOut)` | 总 UB 公式约束 |
| 同上 | `q2q1Part1Ub` | int64/64 | 固定仓不可见；序号11 | 字节/TBuf | 基础值；batched 乘 times | `InitBuffer(mul1/mul2)` | BF16 按 FP32 容量 |
| 同上 | `cosPart1Ub` | int64/64 | 固定仓不可见；序号12 | 字节/queue slot | 基础值；batched 乘 preCBatchB | cos/sin VECIN | 两队列分别占用 |
| 同上 | `sin1UbSize` | int64/64 | 固定仓不可见；序号13 | 字节/TBuf | 基础值；batched 乘 times | BF16 `cosCast/sinCast` 或 `cosC/sinC` | 非 BF16 不分配 |
| 同上 | `preCLTimes` | int64/64 | 固定仓不可见；序号14 | 整轮数 | `ComputeTimes(preCoreBatch,preCBatchB)` | former core 外循环 | 可为0 |
| 同上 | `lastCLTimes` | int64/64 | 固定仓不可见；序号15 | 整轮数 | `ComputeTimes(lastCoreBatch,preCBatchB)` | tail core 外循环 | 可为0 |
| 同上 | `preCBBTimes` | int64/64 | 固定仓不可见；序号16 | 子轮数 | `ComputeTimes(preCBatchB,comBatchBB)` | key3 整轮内计算循环 | 可为0 |
| 同上 | `preCBLTimes` | int64/64 | 固定仓不可见；序号17 | 子轮数 | `ComputeTimes(preCBatchL,comBatchBB)` | key3 former 末轮内循环 | 可为0 |
| 同上 | `preCLLTimes` | int64/64 | 固定仓不可见；序号18 | 子轮数 | `ComputeTimes(lastCBatchL,comBatchBB)` | key3 tail 末轮内循环 | 可为0 |
| 同上 | `qCoreOffset` | int64/64 | 固定仓不可见；序号19 | 元素/核 | `preCoreBatch*QN*qD` | q GM 核起点 | 使用完整 qD |
| 同上 | `kCoreOffset` | int64/64 | 固定仓不可见；序号20 | 元素/核 | `preCoreBatch*KN*kD` | k GM 核起点 | 使用完整 kD |
| 同上 | `cosCoreOffset` | int64/64 | 固定仓不可见；序号21 | 元素/核 | `preCoreBatch*coscdNum` | cos/sin GM 核起点 | A2/A3 无 B 广播 |
| 同上 | `qcdNum` | int64/64 | 固定仓不可见；序号22 | 元素/batch | `QN*lastDim` | q 在拼接 UB 的长度 | 旋转前缀，不含后缀 |
| 同上 | `kcdNum` | int64/64 | 固定仓不可见；序号23 | 元素/batch | `KN*lastDim` | k 在拼接 UB 的长度 | 同上 |
| 同上 | `coscdNum` | int64/64 | 固定仓不可见；序号24 | 元素/batch | `1*cosDim3` | cos/sin 连续搬运长度 | 等于lastDim |
| 同上 | `qkcNum` | int64/64 | 固定仓不可见；序号25 | head 数 | `QN+KN` | Vector repeatTimes与拼接跨度 | Host 限制不超过255 |
| 同上 | `mulNum` | int64/64 | 固定仓不可见；序号26 | 元素或repeat跨度 | small 为 `(QN+KN)*D_r`；batched 为原值乘2再除oneBlockFp32 | Add 长度或 SmallQC repeat stride | route 解释不同，不得混用 |
| 同上 | `qcdHalfNum` | int64/64 | 固定仓不可见；序号27 | 循环次数/元素 | 初值 `QN*D_r/2`；small 改为 `D_r/mask` | small `aa` 循环 | key间语义不同 |
| 同上 | `dstRepSBr` | int64/64 | 固定仓不可见；序号28 | 32 B block/repeat | `D_r/oneBlockFp32` | Unary/Binary repeat stride | 整除成立 |
| 同上 | `blockLenQ` | int64/64 | 固定仓不可见；序号29 | 32 B block | small `D_r/2/oneBlockFp32`；batched `QN*D_r/oneBlock` | DataCopy q blockLen | key间语义不同 |
| 同上 | `srcStrideK` | int64/64 | 固定仓不可见；序号30 | 32 B block | `KN*D_r/oneBlock` | q/k 拼接 DataCopy stride | small 路径不读取 |
| 同上 | `blockLenq2q1` | int64/64 | 固定仓不可见；序号31 | 32 B block | `D_r/2/oneBlockFp32` | half 两段交换 DataCopy | batched route |
| 同上 | `mask` | int64/64 | 固定仓不可见；序号32 | Vector 元素 | `min(D_r,64或128)` | Mul mask | 与计算 dtype 对应 |
| 同上 | `qcNum` | int64/64 | 固定仓不可见；序号33 | head 数 | query N | partial DataCopy blockCount | 正数，且参与qkc上限 |
| 同上 | `kcNum` | int64/64 | 固定仓不可见；序号34 | head 数 | key N | partial DataCopy与 KN==1 分支 | 正数 |
| 同上 | `qDim3` | int64/64 | 固定仓不可见；序号35 | 元素 | query 完整 D | partial 分支与 GM batch stride | 64或128 |
| 同上 | `kDim3` | int64/64 | 固定仓不可见；序号36 | 元素 | key 完整 D | k GM batch stride | 可见检查未直接比较 qD/kD，文档要求相等 |
| 同上 | `blockMoveQ` | int64/64 | 固定仓不可见；序号37 | 32 B block | `D_r/oneBlock` | partial prefix DataCopy blockLen/stride | D_r 按 block 对齐 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 small FP16/FP32 | key1、非BF16 | `apply_rotary_pos_emb_tiling.cpp:Compute` | `ApplyRotaryPosEmbTilingData` small 语义 | `apply_rotary_pos_emb.cpp:apply_rotary_pos_emb` | `apply_rotary_pos_emb_small.h:ARPESmall::Process/CopyInQK/ComputeTotary` | `op_host/CMakeLists.txt:add_modules_sources`；910b/910_93 binary config |
| R-002 small BF16→FP32 | key1、BF16 | `apply_rotary_pos_emb_tiling.cpp:Compute` | 同结构，FP32 UB 容量 | 同一 entry 的 BF16 编译分支 | `ARPESmall::ComputeBF16`、`ApplyRotaryPosEmbBase::LocalTensor2NewTensor` | 同 R-001 |
| R-003 aggregate-output FP16/FP32 | key3 | `apply_rotary_pos_emb_tiling.cpp:ComputeAB` | batched 字段，`shengMte>0` | entry key3 分支 | `apply_rotary_pos_emb_compute_ab.h:ARPEComputeAB::Process/CopyIn/ComputeTotary/CopyOut` | 同 R-001 |
| R-004 one-copy/one-output FP16/FP32 | key4、非BF16 | `apply_rotary_pos_emb_tiling.cpp:ComputeAB` | batched 字段，`shengMte=0` | entry key4 非BF16实例 | `apply_rotary_pos_emb_compute_ab_cast.h:ARPEComputeABCast::Process/CopyIn/ComputeElse` | 同 R-001 |
| R-005 batched BF16→FP32 | key4、BF16 | `apply_rotary_pos_emb_tiling.cpp:ComputeAB` | batched 字段，BF16 cast buffers | entry key4 BF16实例 | `ARPEComputeABCast::ComputeBF16/CTotaryBF16`、base reinterpret helper | 同 R-001 |

## Kernel 实现

### 公共执行流程

统一 entry 声明 `KERNEL_TYPE_AIV_ONLY`，先拒绝空 workspace和空 user workspace，再反序列化同一 POD。所有 route 用 `GetBlockIdx` 定位 `(B,S)` 或 T batch，将 query、key 按 N 轴拼入 qIn UB；cos/sin 只搬一份 head并在 Vector repeat 中广播。half 旋转先把后半移到前半、前半移到后半，并对 sin 的前半取负，再分别执行 `rotate*sin`、`x*cos`、Add。TQue 的 EnQue/DeQue 是 GM搬入→Vector消费和 Vector产出→GM搬出的同步边；没有跨核通信、Atomic或 workspace 数据交换。

### Route R-001：small FP16/FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmbV2.md | A2/A3 产品约束 | layout 1/4、half、非空、相同 dtype、D 为64或128 | 合法请求进入 inner；违约返回参数错误 |
| Tiling | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ApplyRotaryPosEmbTiling::Compute` | `preCoreBatch==1 && speUb<=totalUbSize`，且编译 dtype 非 BF16 | key1，R-001 |
| Router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `apply_rotary_pos_emb` | `ORIG_DTYPE_QUERY!=DT_BF16 && TILING_KEY_IS(1)` | `ARPESmall<DTYPE_QUERY,DTYPE_QUERY>` |

#### Device 数据流

##### 函数级调用链

```text
apply_rotary_pos_emb -> ARPESmall<T,T>::Init
  -> ARPESmall<T,T>::Process -> ProcessPerCore -> CastCopyIn
    -> CopyInQK -> ComputeTotary -> DataCopy原地回写
```

##### 源码等价伪码

```text
源码等价伪码：
blockIdx=GetBlockIdx；若 blockIdx>=useCoreNum 则返回
qBase=blockIdx*qCoreOffset；kBase=blockIdx*kCoreOffset；csBase=blockIdx*cosCoreOffset
分配 q/cos/sin queue slot；搬 cos/sin 的 D_r 元素
若 qD>D_r：按 QN/KN 个 block 只搬每个 head 前 D_r；否则连续拼接 Q 后 K
EnQue 后 DeQue；对 sin[0:D_r/2] 执行 Muls(-1)
把每个 head 后半搬到 out 前半、前半搬到 out 后半
for aa in [0,D_r/mask)：
  Mul(rot*sin, mask, repeat=QN+KN, repStride=D_r对应block)
  Mul(x*cos,   mask, repeat=QN+KN, cos repStride=0)
Add 全部 (QN+KN)*D_r 元素；EnQue/DeQue 输出
按完整或前缀 DataCopy 写回 qBase/kBase；释放 queue slot
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM query/key | `q=blockIdx*qCoreOffset`；`k=blockIdx*kCoreOffset` | 每核 `QN*D_r`、`KN*D_r` 有效元素 | FP16或FP32，head-major | 调用者 | `CopyInQK`/原地写回 | 调用者/整次 launch | partial 时 blockCount=QN/KN、blockLen=`blockMoveQ`、srcStride同值 |
| GM cos/sin | `blockIdx*cosCoreOffset` | 各 `D_r` 元素 | 同 dtype，N=1 | 调用者 | Vector广播 | 调用者/本核 | 连续 DataCopy |
| qIn/qOut TQue | TPipe 管理，q后接k | 每 slot `qPart1Ub=(QN+KN)*D_r*sizeof(T)`，depth1 | VECIN/VECOUT | DataCopy/Vector | Compute/CopyOut | 本核/一次处理 | EnQue/DeQue保护；两 queue 不别名 |
| cos/sin TQue | TPipe 管理 | 各 `cosPart1Ub=D_r*sizeof(T)`，depth1 | VECIN | GM DataCopy | Mul | 本核/一次处理 | 广播通过 src1RepStride=0 |
| mul1/mul2 TBuf | TPipe 管理 | 各 `q2q1Part1Ub=(QN+KN)*D_r*sizeof(T)` | VECCALC | Mul | Add | 本核/一次处理 | 无显式 queue；Add 后可复用 |
| workspace | `N/A（body 不形成地址）` | Host 至少16 MiB但有效 extent=0 | N/A | Host分配 | 仅 entry 非空检查 | 调用者/launch | 不与UB/GM别名 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  qin["qInQueue · depth=1 · Q"]:2
  cosin["cosInQueue · depth=1 · C"]
  sinin["sinInQueue · depth=1 · C"]
  qout["qOutQueue · depth=1 · Q"]:2
  mul1["mul1 · Q"]:3
  mul2["mul2 · Q"]:3
```

其中 `Q=qPart1Ub=(QN+KN)*D_r*sizeof(T)`，`C=cosPart1Ub=D_r*sizeof(T)`；`mul1/mul2` 的容量均为 `q2q1Part1Ub=Q`。六块分配逐项对应 `ARPESmall::Init`。

##### 字节账与复用顺序

```text
UB_total = qIn(Q) + qOut(Q) + cosIn(C) + sinIn(C) + mul1(Q) + mul2(Q)
         = 4*Q + 2*C
         = speUb
UB_total <= totalUbSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| qIn queue slot | `CopyInQK` | `ComputeTotary` | 本核结束，无下一轮 | `EnQue→DeQue→FreeTensor` | Add 的原值项完成后才释放 |
| qOut queue slot | half 交换与 Add | GM `DataCopy` | 本核结束 | `EnQue→DeQue→FreeTensor` | Q/K 两段均写回后释放 |
| mul1/mul2 | 两次 `Mul` | `Add` | 本核结束 | 同一 pipe 指令顺序 | Add 消费前不得覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | q/cos/sin `EnQue→DeQue` | DataCopy | `CastCopyIn` | 三个 VECIN slot | FP16或FP32原精度；D_r整block，无元素尾 |
| UB计算 | Vector依赖顺序 | half DataCopy、Muls | 两次Mul后Add | out/mul TBuf | mask FP16最多128、FP32最多64；`D_r/mask` 循环覆盖全D_r |
| UB→GM | qOut `EnQue→DeQue` | Add | DataCopy | VECOUT slot | 仅有效 D_r 前缀；后缀不写 |
| 复用 | `FreeTensor` | GM写回完成 | 下一次分配 | 全部 queue slot | 单 batch/核，无跨迭代 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ApplyRotaryPosEmbTiling::Compute` | small 谓词、key1、mask和搬运字段 |
| Kernel entry/router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `apply_rotary_pos_emb` key1非BF16 | 实例化 `ARPESmall<T,T>`且任务族为AIV_ONLY |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_small.h | `ARPESmall::CopyInQK/CastCopyIn` | Q/K拼接、queue同步和原地写回 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_small.h | `ARPESmall::ComputeTotary` | half交换、Mul repeat参数与Add |

### Route R-002：small BF16→FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_def.cpp | `ApplyRotaryPosEmb` | ascend910b/910_93 config允许 BF16 | 进入旧 Tiling |
| Tiling | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `Compute` | small 谓词成立且 `qDtype==DT_BF16` | key1，castSize=4、mask最多64 |
| Router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | key1 BF16分支 | `ORIG_DTYPE_QUERY==DT_BF16 && TILING_KEY_IS(1)` | `ARPESmall<DTYPE_QUERY,float>` |

#### Device 数据流

##### 函数级调用链

```text
apply_rotary_pos_emb -> ARPESmall<bfloat16_t,float>::Init
  -> Process -> CastCopyIn -> CopyInQK
    -> ComputeTotary -> ComputeBF16 -> LocalTensor2NewTensor -> 原地CopyOut
```

##### 源码等价伪码

```text
源码等价伪码：
若 blockIdx>=useCoreNum 返回；按 q/k/cosCoreOffset 搬 BF16 到 depth1 queues
Cast 拼接 QK 到 mul2 FP32；把 FP32 后半/前半交换到 qOut物理区的FP32视图
Cast cos/sin 到独立 FP32 TBuf；对 sin前半执行 Muls(-1)
for aa in [0,D_r/mask)：以 mask、repeat=QN+KN、cos/sin广播 stride 做两次 FP32 Mul
FP32 Add 后 Cast(CAST_ROUND) 回 BF16 qOut queue
若 partial 仅按 blockMoveQ 写每个head前缀，否则连续写 Q/K；释放 slot
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM四输入/query-key输出 | 与R-001相同 `blockIdx*CoreOffset` | 输入 BF16；有效Q/K前缀 `(QN+KN)*D_r` | BF16/ND | 调用者 | DataCopy/原地回写 | 调用者 | D_r按32B对齐 |
| qIn/qOut queues | q后接k | qIn/qOut slot各 `qPart1Ub=(QN+KN)*D_r*4 B`，实际元素BF16但为FP32 reinterpret留容量 | BF16与FP32物理视图 | CopyIn/Cast | Compute/Cast回写 | 本核 | depth1、queue同步 |
| mul1/mul2 TBuf | TPipe管理 | 各 `(QN+KN)*D_r*4 B` | FP32 | Cast/Mul | Add/Cast | 本核 | `LocalTensor2NewTensor`仅重解释地址 |
| cosCast/sinCast TBuf | TPipe管理 | 各 `D_r*4 B` | FP32 | Cast cos/sin | FP32 Mul | 本核 | BF16编译分支条件分配 |
| cos/sin queues | TPipe管理 | 各 `D_r*2 B`，depth1 | BF16 | DataCopy | Cast | 本核 | dequeue后才释放 |
| workspace | `N/A（body不使用）` | 有效0 B | N/A | Host | entry guard | launch | 仅非空条件 |

##### 分层内存布局

```mermaid
block-beta
  columns 8
  qin["qInQueue · depth=1 · Q"]:2
  qout["qOutQueue · depth=1 · Q"]:2
  cosin["cosIn · C"]
  sinin["sinIn · C"]
  coscast["cosCast · S"]
  sincast["sinCast · S"]
  mul1["mul1 · M"]:4
  mul2["mul2 · M"]:4
```

`Q=M=(QN+KN)*D_r*4 B`，`C=D_r*2 B`，`S=D_r*4 B`。qIn/qOut 虽以 BF16 tensor 访问，但 Host 按 `castDtypeSize=4` 分配，允许 `LocalTensor2NewTensor` 在同一物理区创建 FP32 视图。

##### 字节账与复用顺序

```text
UB_total = 2*Q + 2*C + 2*M + 2*S
         = speUb(BF16)
UB_total <= totalUbSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| qIn 物理区 | BF16 DataCopy；Cast消费后由FP32视图承接 `x*cos` | `Cast`读取BF16；`Add`读取FP32乘积 | 本核结束 | qIn `DeQue→FreeTensor` | 必须先把BF16原值Cast到mul2，之后才能用同区作FP32 Mul目的地 |
| qOut 物理区 | FP32 half交换，随后 BF16 Cast写回 | GM CopyOut | 本核结束 | qOut `EnQue→DeQue→FreeTensor` | FP32计算结束后才允许 CAST_ROUND 覆盖 |
| cosCast/sinCast | BF16→FP32 Cast | 两次FP32 Mul | 本核结束 | 仅BF16编译分支分配 | Mul完成前不得复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | VECIN EnQue/DeQue | BF16 DataCopy | FP32 Cast | q/cos/sin slot | 无隐式补值；D_r整block |
| 计算 | Vector顺序 | CAST_NONE | FP32 Mul/Add | mul/cast TBuf | mask最多64；最后 CAST_ROUND 到BF16 |
| 输出 | VECOUT EnQue/DeQue | BF16 Cast | DataCopy | qOut slot | partial只提交D_r前缀 |
| 复用 | FreeTensor | CopyOut完成 | 后续分配 | queue slot | 单次处理无跨轮事件 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `Compute` | BF16 castSize/mask与small key |
| Kernel entry/router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `ARPESmall<DTYPE_QUERY,float>` | BF16 key1实例 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_small.h | `ARPESmall::ComputeBF16` | BF16→FP32→BF16计算顺序 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_base.h | `LocalTensor2NewTensor` | qIn/qOut物理UB重解释为FP32 |

### Route R-003：aggregate-output FP16/FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `Compute` | small条件失败 | 调用 `ComputeAB` |
| Tiling | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ComputeAB` | 非BF16且剩余UB算得 `shengMte>0` | key3 |
| Router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | key3分支 | `ORIG_DTYPE_QUERY!=DT_BF16 && TILING_KEY_IS(3)` | `ARPEComputeAB<T,T>` |

#### Device 数据流

##### 函数级调用链

```text
apply_rotary_pos_emb -> ARPEComputeAB<T,T>::Init -> Process
  -> ProcessPreCore或ProcessLastCore -> CopyIn
    -> CopyInQK或CopyInPartialRotation -> ComputeTotary -> CopyOut或CopyOutPartialRotation
```

##### 源码等价伪码

```text
源码等价伪码：
若无效核返回；former核循环preCLTimes，tail核循环lastCLTimes，再各处理末轮
每轮 coreBatchIndex 计算 q/k/cos GM地址；搬 preCBatchB 个batch到depth2 queues
若partial，按batch与head分块搬前D_r；KN==1时用专门stride拼接
对每batch的sin前半Muls(-1)；按 blockLenq2q1 交换每个拼接head两半
for j in整comBatch子轮并处理尾子轮：
  若 qkcNum>=comBatch，按batch循环、repeat=qkcNum；否则按head循环、repeat=comBatch
  mask由dtype与D_r给定；分别Mul旋转项和原值项，再Add该子轮全部元素
完成整个搬入轮后只EnQue一次大输出并CopyOut；partial仅写前缀
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM Q/K | 核起点加 `coreBatchIndex*preCBatchB*qcdNum或kcdNum` | 每轮 `preCBatchB*QN/KN*D_r` | FP16/FP32 | 调用者 | CopyInQK | launch | partial改用完整D的batch stride |
| GM cos/sin | `blockIdx*cosCoreOffset+coreBatchIndex*preCBatchB*coscdNum` | `preCBatchB*D_r` | 同dtype | 调用者 | Vector广播 | launch | 连续搬运 |
| q/cos/sin VECIN | TPipe管理 | q slot `qPart1Ub`，cos/sin各`cosPart1Ub`，depth2 | 同dtype | DataCopy | DeQue/Compute | 每轮 | ping-pong容量由Host计入2倍 |
| qOut VECOUT | TPipe管理 | `qPart1Ub`，depth1 | 同dtype | 所有子轮Add | CopyOut | 每轮 | key3聚合多次计算后一次输出 |
| mul1/mul2 TBuf | TPipe管理 | 各 `q2q1Part1Ub=times*(QN+KN)*D_r*sizeof(T)` | 同dtype | Mul | Add | 子轮 | 子轮复用，偏移从0写 |
| workspace | `N/A` | 有效0 B | N/A | Host | entry guard | launch | 不参与route |

##### 分层内存布局

```mermaid
block-beta
  columns 8
  qin["qInQueue · depth=2 · BQ"]:2
  cosin["cosInQueue · depth=2 · BC"]:2
  sinin["sinInQueue · depth=2 · BC"]:2
  qout["qOutQueue · depth=1 · BQ"]:2
  mul1["mul1 · tQ"]:4
  mul2["mul2 · tQ"]:4
```

令单 batch 的 `Q=(QN+KN)*D_r*sizeof(T)`、`C=D_r*sizeof(T)`，`t=times`，`m=shengMte>0`，`B=t+m=preCBatchB`。输入和输出按 B 批容量分配，两个计算 TBuf 只按 t 批容量分配。

##### 字节账与复用顺序

```text
UB_total = 2*(B*Q) + 4*(B*C) + 1*(B*Q) + 2*(t*Q)
         = t*(5*Q + 4*C) + m*(3*Q + 4*C)
         <= totalUbSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| depth2 q/cos/sin slot | `CopyInQK` | 当前外轮全部计算子轮 | 下一外轮 `AllocTensor` | 各输入 `EnQue→DeQue→FreeTensor` | `preCBBTimes` 和尾子轮完成后才释放 |
| mul1/mul2 | 当前 t 批 `Mul` | 当前子轮 `Add` | 下一计算子轮从 offset 0 覆盖 | 同一 pipe 指令顺序 | Add 已把结果写入 qOut 对应偏移 |
| qOut 聚合区 | 每个子轮 `Add` 写不同偏移 | 单次 `CopyOut` | 下一外轮 | 全部子轮后 `EnQue→DeQue→FreeTensor` | `m` 只扩展聚合区，不得扩大 t 批 TBuf 的单次计算范围 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入 | depth2 EnQue/DeQue | GM DataCopy | Vector | q/cos/sin slot | block参数来自POD，partial保留后缀 |
| 子轮计算 | 同pipe指令序 | half交换/Muls | Mul/Add | qOut和TBuf | FP16或FP32原精度；mask覆盖D_r |
| 聚合输出 | VECOUT EnQue/DeQue | 全部子轮完成 | CopyOut | 大qOut slot | former/tail core及末轮由三组tail字段保护 |
| 复用 | FreeTensor | CopyOut | 下一外轮Alloc | depth2/1 slot | 无跨核同步 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ComputeAB` | `shengMte>0`选key3及双层loop字段 |
| Kernel entry/router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `ARPEComputeAB<T,T>` | key3仅非BF16可达 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab.h | `ProcessPreCore/ProcessLastCore/CopyIn` | former/tail与外轮/子轮控制 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab.h | `LargeQC/SmallQC/ComputeTotary` | repeat方向、mask、Mul/Add |

### Route R-004：one-copy/one-output FP16/FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `Compute` | small条件失败且非BF16 | 调用ComputeAB |
| Tiling | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ComputeAB` | `shengMte==0` | key4 |
| Router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | key4非BF16 | `ORIG_DTYPE_QUERY!=DT_BF16 && TILING_KEY_IS(4)` | `ARPEComputeABCast<T,T>` |

#### Device 数据流

##### 函数级调用链

```text
apply_rotary_pos_emb -> ARPEComputeABCast<T,T>::Init -> Process
  -> ProcessPreCore或ProcessLastCore -> CopyIn
    -> CopyInPartialRotation可选 -> ComputeElse -> ComputeTotary -> CopyOutPartialRotation可选
```

##### 源码等价伪码

```text
源码等价伪码：
按former/tail core选择preCLTimes或lastCLTimes并处理末轮
for 每个coreBatchIndex：计算 qOffset=blockIdx*qCoreOffset+coreBatchIndex*preCBatchB*qcdNum，k/cos同理
搬 preCBatchB 个batch的Q/K/cos/sin；若partial则按完整D跨步
立即DeQue；sin前半按每batch执行Muls(-1)
按每head的half block交换到qOut；根据 qkcNum 与 preCBatchB 选择repeat方向
执行两次Mul和一次Add，当前搬入轮立即EnQue/DeQue并写回Q/K
释放输入与输出slot后进入下一轮
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM输入输出 | 同R-003的核起点与batch偏移 | 每轮`preCBatchB*(QN或KN)*D_r` | FP16/FP32 | 调用者 | CopyIn/CopyOut | launch | partial用blockMoveQ |
| q/cos/sin VECIN | TPipe管理 | q=`qPart1Ub`、cos/sin=`cosPart1Ub`，depth2 | 同dtype | DataCopy | ComputeElse | 每轮 | ping-pong |
| qOut VECOUT | TPipe管理 | `qPart1Ub`，depth1 | 同dtype | Add | 立即CopyOut | 每轮 | 与key3差异是每搬入轮一一输出 |
| mul1/mul2 TBuf | TPipe管理 | 各`q2q1Part1Ub` | 同dtype | Mul | Add | 每轮 | 无BF16 cast buffer |
| workspace | `N/A` | 有效0 B | N/A | Host | entry guard | launch | 不访问 |

##### 分层内存布局

```mermaid
block-beta
  columns 8
  qin["qInQueue · depth=2 · BQ"]:2
  cosin["cosInQueue · depth=2 · BC"]:2
  sinin["sinInQueue · depth=2 · BC"]:2
  qout["qOutQueue · depth=1 · BQ"]:2
  mul1["mul1 · BQ"]:4
  mul2["mul2 · BQ"]:4
```

本 Route 中 `shengMte=0`，所以 `B=preCBatchB=times`。令 `Q=(QN+KN)*D_r*sizeof(T)`、`C=D_r*sizeof(T)`；一次搬入覆盖 B 批，计算与输出也覆盖同一 B 批。

##### 字节账与复用顺序

```text
UB_total = 2*(B*Q) + 4*(B*C) + 1*(B*Q) + 2*(B*Q)
         = B*(5*Q + 4*C)
         = times*oneLoop
         <= totalUbSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| depth2输入slot | `CopyIn` | `ComputeElse` | 下一外轮 | `EnQue→DeQue→FreeTensor` | 当前轮 Add 完成后释放 |
| mul1/mul2 | 两次 Mul | 当前轮 Add | 下一外轮 | 指令顺序 | Add完成前不得覆盖 |
| qOut slot | half交换和Add | 当前轮立即 CopyOut | 下一外轮 | `EnQue→DeQue→FreeTensor` | Q/K写回完成后才释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | depth2 queue | GM DataCopy | ComputeElse | q/cos/sin slot | D_r完整mask，partial跨D后缀 |
| 计算 | Vector顺序 | Muls/交换 | Mul/Add | qOut/TBuf | FP16或FP32原精度，无cast |
| 输出 | VECOUT queue | Add | DataCopy | qOut slot | 每轮立即提交；末轮batch取POD tail |
| 复用 | FreeTensor | CopyOut完成 | 下一轮Alloc | 所有slot | 不需显式event/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ComputeAB` | key4与轮字段 |
| Kernel entry/router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `ARPEComputeABCast<T,T>` | key4非BF16实例 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab_cast.h | `CopyIn/ComputeElse` | 一轮搬入、计算、输出 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab_cast.h | `LargeQCFP16/SmallQCFP16/CTotaryBF32` | dtype与repeat方向 |

### Route R-005：batched BF16→FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `Compute` | BF16且small失败 | ComputeAB |
| Tiling | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ComputeAB` | `isCast`令`shengMte=0`并选key4 | R-005 |
| Router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | key4 BF16分支 | `ORIG_DTYPE_QUERY==DT_BF16 && TILING_KEY_IS(4)` | `ARPEComputeABCast<DTYPE_QUERY,float>` |

#### Device 数据流

##### 函数级调用链

```text
apply_rotary_pos_emb -> ARPEComputeABCast<bfloat16_t,float>::Init -> Process
  -> ProcessPreCore或ProcessLastCore -> CopyIn
    -> ComputeBF16 -> ComputeTotary -> CTotaryBF16 -> LocalTensor2NewTensor -> BF16 CopyOut
```

##### 源码等价伪码

```text
源码等价伪码：
按core/外轮循环；qOffset=blockIdx*qCoreOffset+coreBatchIndex*preCBatchB*qcdNum，k/cos同理
搬BF16 Q/K/cos/sin；若partial则走按完整D跨步的分支
DeQue后分配BF16 qOut；Cast QK到mul2 FP32
在qOut物理UB的FP32视图中交换half；Cast cos/sin到FP32 TBuf并对sin前半Muls(-1)
若 qkcNum>=preCBatchB 按batch循环，否则按head循环；mask最多64
FP32两次Mul与Add；Cast(CAST_ROUND)回BF16 qOut
释放输入，queue提交输出；写回有效前缀并释放
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM Q/K/cos/sin | 核起点加外轮偏移 | Q/K前缀与 `preCBatchB*D_r` cos/sin | BF16 | 调用者 | CopyIn | launch | partial跨完整D |
| q/cos/sin queues | TPipe管理 | q slot=`qPart1Ub`，cos/sin=`cosPart1Ub`，depth2 | BF16，q预留FP32容量 | DataCopy | Cast | 每轮 | queue同步 |
| qOut queue | TPipe管理 | `qPart1Ub`，depth1 | BF16存储/FP32物理视图 | FP32计算后Cast | CopyOut | 每轮 | 容量按4B元素预留 |
| mul1/mul2 TBuf | TPipe管理 | 各`q2q1Part1Ub=times*(QN+KN)*D_r*4 B` | FP32 | Cast/Mul | Add/Cast | 每轮 | 计算后复用 |
| cosC/sinC TBuf | TPipe管理 | 各`sin1UbSize=times*D_r*4 B` | FP32 | Cast | Mul | 每轮 | 仅BF16编译分支 |
| workspace | `N/A` | 有效0 B | N/A | Host | entry guard | launch | 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 10
  qin["qInQueue · depth=2 · BQ"]:2
  cosin["cosIn · depth=2 · BC"]:2
  sinin["sinIn · depth=2 · BC"]:2
  qout["qOut · depth=1 · BQ"]:2
  cosc["cosC · BS"]
  sinc["sinC · BS"]
  mul1["mul1 · BQ"]:5
  mul2["mul2 · BQ"]:5
```

令 `B=times=preCBatchB`、`Q=(QN+KN)*D_r*4 B`、`C=D_r*2 B`、`S=D_r*4 B`。qIn/qOut 按 FP32 容量分配，cos/sin 输入保持 BF16 容量，cosC/sinC保存升精度结果。

##### 字节账与复用顺序

```text
UB_total = 3*(B*Q) + 4*(B*C) + 2*(B*Q) + 2*(B*S)
         = B*(5*Q + 4*C + 2*S)
         = times*oneLoop(BF16)
         <= totalUbSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| qIn/qOut 物理区 | qIn先装BF16并在Cast后承接 `x*cos`；qOut承接FP32 half交换 | FP32 Add 与最终BF16 CopyOut | 下一外轮 | queue生命周期 | 两区均按4B元素计算；BF16源被消费后才允许FP32目的写入 |
| mul2 | 首先保存 q 的 FP32 Cast，后保存最终 FP32 Add | q 原值 Mul、CAST_ROUND | 下一外轮 | 指令顺序 | q 原值被消费后才能由 Add 覆盖 |
| cosC/sinC | BF16 Cast | FP32 Mul | 下一外轮 | 当前计算结束 | 两个 TBuf 仅BF16实例存在，非BF16 Route不得引用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | depth2 EnQue/DeQue | BF16 DataCopy | Cast | 三个输入slot | former/tail core及末轮batch受POD保护 |
| 计算 | Vector依赖 | CAST_NONE | FP32 Mul/Add | 四个TBuf与qOut视图 | mask最多64；half交换按blockLenq2q1 |
| 舍入 | `Cast(...CAST_ROUND)` | FP32 Add | qOut EnQue | BF16输出 | 唯一显式降精度点 |
| 输出/复用 | VECOUT DeQue、FreeTensor | BF16 qOut | GM DataCopy/下一轮 | qOut slot | partial只写前D_r，无跨核同步 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `ComputeAB` | BF16强制key4、FP32容量字段 |
| Kernel entry/router | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `ARPEComputeABCast<DTYPE_QUERY,float>` | BF16 key4实例 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab_cast.h | `ComputeBF16/CTotaryBF16` | Cast、repeat、Add与舍入 |
| Device body/helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_base.h | `LocalTensor2NewTensor` | q/qOut UB的FP32重解释 |

## 算子特定优化

### Trick T-001：用剩余 UB 扩大聚合输出批次

#### 触发与源码依据

- 适用 Route：R-003；R-004/R-005 是该机制不成立时的真实去向。
- 触发条件：small 路线未命中；`oneLoop` 合法；非 BF16；基础 `times` 分配后剩余 UB 还能容纳至少一个 `3*qPart1Ub+4*cosPart1Ub` 扩展批次，即 `shengMte>0`。
- Golden 来源：[`apply_rotary_pos_emb_tiling.cpp:325-346`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
ge::graphStatus ApplyRotaryPosEmbTiling::ComputeAB(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    // 一个batch搬运及计算使用的UB size (2: ping-pong)
    int64_t oneLoop = params.qPart1Ub * 2 + params.cosPart1Ub * 4 + params.qPart1Ub + params.q2q1Part1Ub * 2 +
                      static_cast<int64_t>(params.isCast) * (params.sin1UbSize * 2);
    OP_LOGD(context->GetNodeName(), "oneLoop ub size %ld", oneLoop);
    OP_CHECK_IF(oneLoop > params.totalUbSize || oneLoop <= 0,
                OP_LOGE(context->GetNodeName(), "oneLoop is too large or small than 0"), return ge::GRAPH_FAILED);
    int64_t times = params.totalUbSize / oneLoop; // ub一次可以计算的batch数量
    int64_t shengUb = params.totalUbSize % oneLoop;
    // shengMte：如果有剩余ub空间，再计算几个单batch后一起搬出，减少mte3次数
    int64_t shengMte = shengUb / (params.qPart1Ub * 2 + params.cosPart1Ub * 4 + params.qPart1Ub);
    shengMte = params.isCast ? 0 : shengMte;
    if (times > params.preCoreBatch) { // UB可以一次放下当前核要出的batch数量
        times = params.preCoreBatch;
        shengMte = 0;
    }
    if (shengMte == 0 || params.isCast) { // 搬入搬出一一对应情况
        params.tilingKey = static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB_CAST);
    } else { // 存在多次mte2搬入，计算，一次mte3搬出的场景
        params.tilingKey = static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB);
    }
```

#### 机制与设计原因

基础 `times` 同时为输入、输出和两个计算 TBuf 预留一批完整 `oneLoop`。余量扩展时不再增加 `mul1/mul2`，只增加 depth2 Q/K、cos/sin 和聚合 qOut，因此每个额外 batch 的边际成本是 `3Q+4C`。R-003 让计算 TBuf 以 t 批为窗口反复覆盖，同时把 t+m 批结果写入 qOut 的不同偏移，最后一次 MTE3 写回，源码注释明确目标是减少 MTE3 次数。BF16 需要额外 FP32 cast 区且 qOut 存在重解释，Host 直接把 `shengMte` 清零，避免把非 cast 的边际容量模型错误套用到 BF16。

#### 迁移边界

- 适用：具有“计算窗口小于聚合输出窗口”、结果可在 UB 不重叠累积的纯 Vector 算子。
- 前提：Host 能精确证明输入/输出扩展成本、计算 TBuf 可覆盖复用、所有子轮完成后再 MTE3；仅适用于 A2/A3 本卡 POD 和非 BF16。
- 失效：`shengMte==0` 使用 R-004；BF16 使用 R-005；`oneLoop` 非法直接失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在 `shengMte==0` 或 BF16 时强制 key 3：固定源码明确将这些场景路由到 key 4，强制聚合会让 qOut/input 的 B 批容量与 Host 计算不一致。

### Trick T-002：只搬运和回写旋转前缀

#### 触发与源码依据

- 适用 Route：R-001/R-002；R-003/R-004/R-005 使用等价的 `CopyInPartialRotation/CopyOutPartialRotation`。
- 触发条件：`qDim3>lastDim`，即完整 head 维 D 大于 `D_r=cos.shape[-1]`。
- Golden 来源：[`apply_rotary_pos_emb_small.h:127-145`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_small.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ARPESmall<T1, T2>::CopyInQK(const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    DataCopy(cosUb, cosGm[blockIdx * tilingData->cosCoreOffset], tilingData->coscdNum);
    DataCopy(sinUb, sinGm[blockIdx * tilingData->cosCoreOffset], tilingData->coscdNum);
    if(tilingData->qDim3 > tilingData->lastDim) { // 仅操作前64个元素
        DataCopy(qUb, qGm[blockIdx * tilingData->qCoreOffset], copyInQ); // tiling侧qCoreOffset修改
        DataCopy(qUb[tilingData->qcdNum], kGm[blockIdx * tilingData->kCoreOffset], copyInK);
    }
    else {
        DataCopy(qUb, qGm[blockIdx * tilingData->qCoreOffset], tilingData->qcdNum);
        DataCopy(qUb[tilingData->qcdNum], kGm[blockIdx * tilingData->kCoreOffset], tilingData->kcdNum);
    }

    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);
```

#### 机制与设计原因

partial 路线把 `blockCount` 设为 QN/KN、`blockLen=blockMoveQ=D_r/oneBlock`，并用 `srcStride=blockMoveQ` 跳过每个 head 的未旋转后缀；UB 内仍把 Q 和 K 的旋转前缀紧密拼接。CopyOut 使用对偶参数，仅覆盖 GM 中每个 head 的前 `D_r`，所以后缀无需搬入、无需计算，也不会被输出覆盖。`qCoreOffset/kCoreOffset` 仍以完整 D 计算，保证下一 batch/head 的 GM 起点没有被紧密 UB 布局误导。

#### 迁移边界

- 适用：输出为原地更新且只有连续前缀参与计算的 head-major ND 数据。
- 前提：`D_r` 为 32B block 对齐，Host 同时提供完整 D 的 GM stride和旋转前缀的 blockLen；Q/K后缀语义是保持原值。
- 失效：全 D 旋转时走连续 DataCopy；非连续旋转域不能复用本参数，需要新的 gather/reorder 路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能用紧密的 `D_r` 作为 GM batch/head stride；固定实现只在 UB 中紧密拼接，GM 偏移始终使用完整 `qDim3/kDim3`。混用两者会从下一 head 的错误地址读写。

### Trick T-003：按 head 数与 batch 数切换 Vector repeat 方向

#### 触发与源码依据

- 适用 Route：R-003；R-004/R-005 的 `ComputeTotary` 保留同一分支结构。
- 触发条件：比较 `qkcNum` 与当前计算子轮 `comBatchBB`；两边分别选择沿 head 或沿 batch 做 repeat。
- Golden 来源：[`apply_rotary_pos_emb_compute_ab.h:215-233`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ARPEComputeAB<T1, T2>::CTotaryBF32(
    const int64_t j, const int64_t comBatchBB, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    if (tilingData->qkcNum >= comBatchBB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQC(j, comBatchBB, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData, repeatParams);
    } else {
        repeatParams.dstRepStride = tilingData->mulNum;
        repeatParams.src0RepStride = tilingData->mulNum;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQC(j, comBatchBB, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData, repeatParams);
    }
    Add(outUb[j * tilingData->comBatchBB * tilingData->qkcNum * tilingData->lastDim], mul1Ub, mul2Ub,
        comBatchBB * tilingData->qkcNum * tilingData->lastDim);
```

#### 机制与设计原因

当 head 总数不少于 batch 子轮时，`LargeQC` 以每个 batch 为外循环、以 `qkcNum` 为 repeatTimes，cos/sin 的 `src1RepStride=0` 在各 head 间广播；当 batch 更多时，`SmallQC` 以 head 为外循环、以 `comBatchBB` 为 repeatTimes，cos/sin stride 改成 `dstRepSBr` 以跨 batch 前进。这样 repeatTimes 选择更长的维度，同时保持 Q/K 和 cos/sin 的真实布局；`dstRepSBr=D_r/oneBlockFp32`、`mulNum` 和两个循环上界都来自 POD，不能只复制分支形式。

#### 迁移边界

- 适用：二维可交换循环中一个操作数沿 head 广播、另一个维度是 batch 的 Vector 逐元素计算。
- 前提：两种顺序都保持连续 mask 段，repeatTimes 不超过 API 位宽，repeat stride 能由真实布局表达。
- 失效：布局转置、cos/sin 不再按 batch/head广播或任一方向 stride 超限时必须使用新的分块方式。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能固定 `src1RepStride=0`：在 `qkcNum<comBatchBB` 分支中 cos/sin 必须按 batch 前进；固定广播会让多个 batch 错用同一组位置编码。

### Trick T-004：BF16 以 FP32 物理视图完成旋转和累加

#### 触发与源码依据

- 适用 Route：R-002/R-005。
- 触发条件：编译期 `ORIG_DTYPE_QUERY==DT_BF16`；Host 使用 `castDtypeSize=4`、FP32 mask和额外 cos/sin TBuf。
- Golden 来源：[`apply_rotary_pos_emb_compute_ab_cast.h:250-284`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab_cast.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ARPEComputeABCast<T1, T2>::CTotaryBF16(
    const int64_t preCBatchB, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize, LocalTensor<T1>& sinSize,
    LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T2> cosCUb = cosC.Get<T2>();
    LocalTensor<T2> sinCUb = sinC.Get<T2>();
    LocalTensor<T2> mul2Ub = mul2.Get<T2>();
    LocalTensor<T2> mul1Ub = mul1.Get<T2>();
    Cast(mul2Ub, qSize, RoundMode::CAST_NONE, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
    copyInq2q1.blockCount = preCBatchB * tilingData->qkcNum;

    LocalTensor<T2> qoutSizeFP32;
    this->LocalTensor2NewTensor(qoutSizeFP32, outUb);
    DataCopy(qoutSizeFP32, mul2Ub[tilingData->halfNum], copyInq2q1);
    DataCopy(qoutSizeFP32[tilingData->halfNum], mul2Ub, copyInq2q1);
    Cast(cosCUb, cosSize, RoundMode::CAST_NONE, preCBatchB * tilingData->lastDim);
    Cast(sinCUb, sinSize, RoundMode::CAST_NONE, preCBatchB * tilingData->lastDim);

    Muls(sinCUb, sinCUb, T2(-1.0), tilingData->halfNum, preCBatchB, mulRepeatP);
    LocalTensor<T2> qUbFP32;
    this->LocalTensor2NewTensor(qUbFP32, qSize);
    if (tilingData->qkcNum >= preCBatchB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQC(preCBatchB, mul1Ub, qoutSizeFP32, sinCUb, qUbFP32, mul2Ub, cosCUb, tilingData, repeatParams);
    } else {
        repeatParams.dstRepStride = tilingData->mulNum;
        repeatParams.src0RepStride = tilingData->mulNum;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQC(preCBatchB, mul1Ub, qoutSizeFP32, sinCUb, qUbFP32, mul2Ub, cosCUb, tilingData, repeatParams);
    }

    Add(mul2Ub, mul1Ub, qUbFP32, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
    Cast(outUb, mul2Ub, RoundMode::CAST_ROUND, preCBatchB * tilingData->qkcNum * tilingData->lastDim);
```

#### 机制与设计原因

Q/K、cos、sin先以 `CAST_NONE` 升到 FP32，half交换、负号、两次 Mul 与 Add 全部在 FP32 完成，最后唯一一次 `CAST_ROUND` 回 BF16。`LocalTensor2NewTensor` 不分配新内存，而是把 Host 已按4B元素预留的 qIn/qOut 物理区解释成 FP32 tensor：qOut视图承接half交换，qIn在其BF16原值已Cast到mul2后改作 `x*cos` 的FP32目的地。这避免再增加两块FP32结果 TBuf，但要求 UB 字节账始终使用 `castDtypeSize=4`。`mul2` 先保存Q/K的FP32 Cast，原值项消费后又承接最终Add，构成有明确指令顺序的复用。

#### 迁移边界

- 适用：BF16输入、可在 UB 中升精度计算且最终只需 BF16 输出的逐元素融合算子。
- 前提：物理 buffer 按 FP32 容量和对齐分配，重解释 helper 不改变地址；BF16源值先被Cast到独立TBuf，原物理区之后才作为FP32计算目的地覆盖。
- 失效：若 UB 按 BF16 字节分配、需要保留 FP32 输出或量化舍入模式不同，不能复用此布局；BF16不使用R-003的余量聚合。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能因 qIn/qOut 的 C++ 元素类型是 BF16 就按2B元素计算容量；固定 Host 与 `InitBuffer` 的配合依赖4B预留，缩小后 FP32 视图的 half交换和 Vector计算会越过实际分配范围。

## 模板与实现对比

| 对比维度 | R-001 small FP16/FP32 | R-002 small BF16 | R-003 aggregate FP16/FP32 | R-004 immediate FP16/FP32 | R-005 batched BF16 | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | small首先命中 | small首先命中且BF16 | 非small、非BF16、`shengMte>0` | 非small、非BF16、`shengMte=0` | 非small、BF16 | 先全载；否则按UB余量和cast区选batched实现 |
| 模板与 dtype | `ARPESmall<T,T>` | `ARPESmall<BF16,float>` | `ARPEComputeAB<T,T>` | `ARPEComputeABCast<T,T>` | `ARPEComputeABCast<BF16,float>` | BF16独立升精度；key3不实例化BF16 |
| 循环与分核 | 每核单batch | 每核单batch | core→外轮→t批计算子轮 | core→外轮，一轮完整计算 | core→外轮，一轮完整FP32计算 | `preCoreBatch` 与 UB 可容纳批数决定层数 |
| Queue/TBuf | 全depth1；2个TBuf | depth1；4个TBuf | 输入depth2、输出depth1、2个t批TBuf | 同左但TBuf/输出均B批 | 输入depth2；4个FP32 TBuf | key3只扩大输入/输出聚合区；BF16增加cast区 |
| UB总占用 | `4Q+2C` | `2Q+2C+2M+2S` | `t(5Q+4C)+m(3Q+4C)` | `B(5Q+4C)` | `B(5Q+4C+2S)` | 全部与Host `speUb/oneLoop/shengMte`同式 |
| 搬运/Vector API | 单batch；cos/sin沿head广播 | 单batch；FP32 Mul/Add | 多次MTE2/计算，一次MTE3；repeat方向自适应 | 每轮MTE2与MTE3一一对应 | 每轮一一对应；FP32后舍入 | UB余量是否足以扩大聚合输出决定MTE3策略 |
| 同步、精度与 tail | Queue顺序；原dtype | Queue顺序；BF16→FP32→BF16 | depth2 Queue；former/tail与计算子轮尾块 | depth2 Queue；former/tail外轮 | depth2 Queue；BF16升精度 | 无跨核同步；tail由POD末轮字段保护 |
| fallback/失败边界 | small失败进入batched | small失败进入R-005 | `shengMte==0`转R-004 | 无后续Route | 无后续Route | oneLoop超UB、qkcNum超uint8、非法shape均Host失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 V2；ascend910b 配置 | 支持 V2；ascend910_93 配置 | 支持 V2，但使用独立 `_950` 卡 |
| Host/Launcher | 共用旧 Tiling：layout 1/4、half、非空、D为64/128 | 与A2共用同一源码、POD、key 1/3/4 | arch35 Tiling支持四种layout、三种mode和D至1024，POD/key完全不同；950公开文档称支持空Tensor，但固定源码会拒绝非正维，详见独立卡 |
| Kernel/资源 | `apply_rotary_pos_emb.cpp`，AIV_ONLY，TQue/TBuf Vector API | 与A2相同 | `apply_rotary_pos_emb_apt.cpp`与RegBase MicroAPI；不得复用本卡ABI/UB公式 |

## 限制与失败边界

- A2/A3 公开语义仅允许 half；旧 Tiling 源码不读取 rotary_mode，实际拒绝非 half 的生成式 ACLNN 校验代码固定仓不可见，不能从 Kernel 推断其错误分支。
- 通用 README 的A2/A3约束写“四输入最后一维必须相等”，V2专用API文档只明确query/key彼此相等、cos/sin彼此相等，而可见Tiling和五条Kernel Route实现了`D>D_r`的前缀旋转。本文把后者标为固定源码可达实现能力，不把它改写成没有文档差异的统一公开约束。
- 可见 Tiling 比文档少一项：它检查 query D、cos D与相对大小，但未显式检查 key D 等于 query D；公开文档要求相等，inner 是否预先检查固定仓不可见。
- 四输入禁止空 tensor；总元素为0直接 `GRAPH_FAILED`。`QN+KN` 必须不超过255，因为 Vector `repeatTimes` 最终窄化为 uint8。
- `oneLoop` 必须为正且不超过平台UB；所有乘法使用 int64，但固定源码未提供更大的 shape 乘法溢出保护或 raw tiling buffer 容量失败后的 ACLNN 映射。
- Kernel 对空 workspace/user workspace和未知 key静默返回；它写传入的 q/k GM 对象而未使用 q_out/k_out 参数，原地 alias如何由inner建立固定仓不可见。
- 生成式 `aclnnInner_apply_rotary_pos_emb.h`、`aclnnInnerApplyRotaryPosEmb*` 函数体、runtime launch宏展开、`TILING_DATA_FIELD_DEF` 布局实现、平台精确UB/AIV核数和实测数值/性能均固定仓不可见或未验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | posembedding/apply_rotary_pos_emb/op_host/op_api/aclnn_apply_rotary_pos_emb_v2.h | `aclnnApplyRotaryPosEmbV2GetWorkspaceSize/aclnnApplyRotaryPosEmbV2` | 两段式公开签名 |
| 入口实现 | posembedding/apply_rotary_pos_emb/op_host/op_api/aclnn_apply_rotary_pos_emb_v2.cpp | 两个 V2 wrapper | 参数原样转发至inner |
| 公开说明 | posembedding/apply_rotary_pos_emb/README.md | 产品支持、计算公式和A2/A3约束 | A2/A3/950平台边界、half语义和公开UB公式 |
| 公开约束 | posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmbV2.md | 产品支持、参数与失败表 | A2/A3 layout/mode/shape与原地语义 |
| Host schema | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_def.cpp | `ApplyRotaryPosEmb/OP_ADD` | dtype、AutoContiguous、属性和SoC kernel文件 |
| Shape | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_infershape.cpp | `ApplyRotaryPosEmbInferShape` | 输出shape广播推导 |
| Host/Tiling/Launcher | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.cpp | `CheckParams/Compute/ComputeAB/SetTilingData` | 校验、key、POD、blockDim、workspace |
| POD声明 | posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_tiling.h | `ApplyRotaryPosEmbTilingData` | 38字段顺序与类型 |
| Kernel entry/body | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb.cpp | `apply_rotary_pos_emb` | AIV_ONLY与key/dtype router |
| Kernel body | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_small.h | `ARPESmall` | small route搬运、Vector与cast |
| Kernel body | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab.h | `ARPEComputeAB` | key3聚合输出route |
| Kernel body | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_compute_ab_cast.h | `ARPEComputeABCast` | key4及BF16 route |
| Kernel helper | posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_base.h | `LocalTensor2NewTensor` | BF16 FP32物理视图 |
| A2二进制配置 | posembedding/apply_rotary_pos_emb/op_host/config/ascend910b/apply_rotary_pos_emb_binary.json | `ApplyRotaryPosEmb` op_list | 三种dtype与默认属性二进制 |
| A3二进制配置 | posembedding/apply_rotary_pos_emb/op_host/config/ascend910_93/apply_rotary_pos_emb_binary.json | `ApplyRotaryPosEmb` op_list | 与A2相同签名配置 |
| 构建接线 | posembedding/apply_rotary_pos_emb/op_host/CMakeLists.txt | `add_modules_sources`/`add_ops_compile_options` | Host、Tiling、Kernel模块和auto-sync |
| 构建接线 | posembedding/apply_rotary_pos_emb/CMakeLists.txt | 子目录遍历 | 接入op_host/op_graph |
| 构建接线 | posembedding/CMakeLists.txt | `add_subdirectory` | 接入posembedding类别 |
| API安装 | cmake/level2_ops_list | `aclnn_apply_rotary_pos_emb_v2.h` | V2公开头进入level2列表 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | A2/A3五路线共用38个int64字段 | 字段顺序、key间复用字段的不同单位、仓外宏布局不可猜测 | `ApplyRotaryPosEmbTilingData`、`SetTilingData` |
| 调用顺序 | wrapper→inner→Tiling→entry→Init/Process | 第一段executor必须供第二段使用；entry先做workspace/userWS保护 | V2 wrapper、`Tiling4ApplyRotaryPosEmb`、`apply_rotary_pos_emb` |
| workspace/内存 | 五个body都不读写user workspace | workspace非空仍是entry条件；UB大小必须使用对应route/dtype公式 | `GetPlatformInfo`、各route `InitBuffer` |
| 同步 | TQue EnQue/DeQue/FreeTensor模式可复用 | depth1/depth2、输入消费后释放、输出写回后释放顺序不可改 | `CopyInQK/CopyIn/CopyOut` |
| 精度与尾块 | FP16/FP32保持dtype，BF16升FP32后CAST_ROUND | mask、repeat stride、half交换、partial只写前缀不可跨route套用 | `ComputeTotary/ComputeBF16/CTotaryBF16` |
| 失败边界 | Host检查和Kernel早退事实可复用 | 非half校验、alias建图、aclnnStatus映射固定仓不可见；不能自行补齐 | `CheckParams/ComputeAB`、V2 wrappers、entry guards |