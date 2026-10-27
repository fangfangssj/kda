---
type: Operator
title: "ForeachNonFiniteCheckAndUnscale A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachNonFiniteCheckAndUnscale 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/docs/aclnnForeachNonFiniteCheckAndUnscale.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_base_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/foreach_non_finite_check_and_unscale.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/foreach_non_finite_check_and_unscale_n_d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/config/ascend910b/foreach_non_finite_check_and_unscale_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
created_at: "2026-08-25T15:47:27Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachNonFiniteCheckAndUnscale A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize` + `aclnnForeachNonFiniteCheckAndUnscale`，公开两段式 ACLNN，无额外版本后缀 |
| 计算语义 | 遍历 `scaledGrads`；若任一原值为 Inf/NaN，则只把 `foundInf[0]` 置为 `1.0f`，否则保持其原值；同时原地执行 `scaledGrads[i][j] *= invScale[0]` |
| 输入输出 | `scaledGrads`：1～256 个连续 ND Tensor，FP32/FP16/BF16、各 Tensor dtype 相同；`foundInf`/`invScale`：单元素 FP32 ND；前两者为原地可见结果 |
| 执行类型 | 单 Kernel：`foreach_non_finite_check_and_unscale`，`KERNEL_TYPE_AIV_ONLY`，实际执行族 `AIV_VECTOR` |
| Kernel 路线 | R-001 FP32 直算；R-002 FP16/BF16 升精度计算 |
| 核心资源 | TensorList GM 指针表、原地 GM、双缓冲 `TQue`、Reduce 临时 Queue、32 B 对齐尾块；Host 固定申请 32 B user workspace，但 body 不读写该区 |
| 来源闭包 | `source_closed`：公开 ACLNN 契约经 OpDef/自动生成 ACLNN bridge 边界闭合到 Tiling selector、POD、AIV entry/body 与 binary/CMake 接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对列表中每个元素的**缩放前值** `x` 执行：

```text
if exponent_bits(abs(float(x))) == 0xff:
    foundInf[0] = 1.0f
scaledGrads[i][j] = cast_T(float(x) * invScale[0])
```

FP32 Route 直接在 FP32 UB 上归约和乘法；FP16/BF16 Route 先把一个对齐 tile 升为 FP32，完成非有限值检测和乘法，再以 `CAST_RINT` 降回原 dtype。多个 AIV core 可能并发写 `foundInf[0]`，但唯一写值均为 `1.0f`，没有“未发现时写 0”路径。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `scaledGrads` | TensorList；单 Tensor rank 0～8，ND | FP32/FP16/BF16 | 列表长度 `(0,256]`；所有 Tensor dtype 相同；公开 ACLNN 不支持空 Tensor 和非连续 Tensor；原地写反缩放结果 |
| `foundInf` | 单元素，rank 0～8，ND | FP32 | 输入/输出；发现 Inf/NaN 时置 1，否则不写；不支持空或非连续 Tensor |
| `invScale` | 单元素，rank 0～8，ND | FP32 | 只读缩放因子；文档允许非连续，OpDef 的 `AutoContiguous()` 属于生成 bridge 的框架前处理边界 |
| `workspaceSize`/`executor` | Host 标量/对象 | `uint64_t*`/`aclOpExecutor**` | 第一段返回；executor 封装单算子流程 |
| `workspace`/`stream` | Device 地址/运行时流 | `void*`/`aclrtStream` | 第二段使用；Tiling 请求 32 B workspace；同一 stream 上启动单 Kernel |

### 入口与版本

固定仓只公开无后缀 `aclnnForeachNonFiniteCheckAndUnscale` 两段式接口，没有更高版本号。API 文档与 README 均明确支持 Atlas A2 与 A3。固定仓未公开该 ACLNN 的手写 `op_api/*.cpp`；`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 接线，OpDef 的动态输入与 `AutoContiguous()` 供框架生成 inner bridge，因此 ACLNN 参数对象到 `ForeachNonFiniteCheckAndUnscale` Host 节点的生成代码是明确框架边界，不把它推测为手写逻辑。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 ACLNN 第一段检查空指针、支持 dtype、TensorList 长度、列表 dtype 一致性及两个标量 Tensor 的元素数；文档给出的失败码包括 `161001`、`161002` 和 Tiling 错误 `561002`。
2. OpDef 将三个输入声明为 ND，并对 `scaled_grads`、`found_inf`、`inv_scale` 标记 `AutoContiguous()`；公开契约仍明确原地 `scaledGrads` 与 `foundInf` 不支持非连续 Tensor。
3. Tiling 将计算节点输入数减 2 得到列表长度；逐 Tensor 读取 storage shape 元素数和 dtype，拒绝列表为空、长度超过 256、空 Tensor、dtype 不一致或不属于 FP32/FP16/BF16；再验证 `found_inf` 与 `inv_scale` 是 FP32。A2/A3 此份 membase Tiling 未在源码中再次检查两者 shape=1，该约束由公开 ACLNN 校验边界给出。
4. 任一校验、平台信息、UB 切分失败都返回 `GRAPH_FAILED`，不会启动 Kernel。

### 参数、Workspace 与生命周期

令 `B=32 B`，`e=B/sizeof(T)`，第 `i` 个 Tensor 的实际元素数为 `n_i`，对齐元素数 `a_i=ceil(n_i/e)*e`，`A=sum(a_i)`：

- `needCoreNum=min(A/e, platformCoreNum)`；由于每个公开输入非空，所以 `A/e>0`。
- 先把 `A/e` 个 32 B block 按商和余数分给各 core，再将扁平 block 区间映射成 `(tensorStart,tensorStartOffset,tensorEnd,tensorEndOffset)`。
- `SetBlockDim(needCoreNum)`；workspace slice 只有 `[0,32)`，owner 是 ACLNN 调用方按第一段结果分配的 Device workspace，生命周期覆盖第二段执行。
- entry 必须得到非空 `workspace`，再经 `GetUserWorkspace(workspace)` 得到非空地址；body 接收但不使用 `userWS`。空指针直接返回，固定仓 entry 没有 Device 侧错误回传。

### 启动图与失败传播

```text
aclnn...GetWorkspaceSize
  -> 生成 ACLNN bridge / ForeachNonFiniteCheckAndUnscale Host 节点
  -> Tiling4ForeachNonFiniteCheckAndUnscale
     -> priority 10000 membase template
  -> workspaceSize = 32 B, executor
aclnn...(workspace, executor, stream)
  -> [同一 stream] foreach_non_finite_check_and_unscale<<<blockDim=needCoreNum>>>
     -> 原地 scaledGrads GM + 条件写 foundInf GM
```

单一节点 N1 的 entry 为 `foreach_non_finite_check_and_unscale(scaled_grads,found_inf,inv_scale,workspace,tiling)`，POD 为 `ForeachNonFiniteCheckAndUnscaleTilingData`。输入与输出交接介质就是调用方 TensorList 指向的 GM：Kernel 结束前原地写回 `scaledGrads`，发现非有限值时写 `foundInf[0]`。没有跨 Kernel 边；stream 的 enqueue/完成语义和 executor 内部生命周期属于 ACLNN 框架边界。Tiling 失败阻断 N1；entry 的 workspace guard 失败则静默不计算。

## Route 与参数设计

### 分核与资源计算

Host 的优先级表按整数升序遍历；membase priority 10000 先于 regbase priority 30000。A2/A3 上 `IsRegbaseSocVersion(context)==false`，membase 接受，regbase 不再执行。

设平台 UB 字节数为 `U`，POD 序列化字节数为 `P=tilingData.GetDataSize()`，`C=floor((U-P)/2/32)*32`：

```text
coefficient = 2 (FP32) or 6 (FP16/BF16)
predictS = floor(256*sizeof(T) / (coefficient*256*sizeof(T)+32) * C)
S = floor(predictS/32)*32
srcCount = S/sizeof(T)
R = ceil(ceil(2*srcCount/64)/8)*8*4 bytes
guard: coefficient*S + R <= C
```

`S=scaledGradsUbSize` 是每个 Queue slot 的输入/输出 tile 字节数，`R=reduceTempValUbSize` 是每个 Reduce 临时 slot 的字节数。Device 的 `maxDataCount=S/sizeof(T)`，每 Tensor 循环 `ceil(dataCount/maxDataCount)` 次，最后 tile 只提交实际 `tempCount` 元素。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 直算 | membase priority 10000；key 1；`ND<float>` | 非 regbase NPU；列表 dtype 全为 FP32；其他公开/Tiling 校验通过；UB guard 成立 | dtype 不符则 key 2/3；资源或参数失败则终止 |
| 2 | R-002 FP16/BF16 升精度 | membase priority 10000；key 2/3；`ND<half>`/`ND<bfloat16_t>` | 非 regbase NPU；列表 dtype 全为 FP16 或 BF16；BF16 entry 还受 `__CCE_AICORE__>200` 且架构不为 3003/3113 的编译 guard；UB guard 成立 | 未知 key 无 body；参数/资源失败终止 |

```text
priority 10000 ForeachNonFiniteCheckAndUnscaleMembaseTiling
  if IsRegbaseSocVersion(context): GRAPH_PARAM_INVALID -> priority 30000
  else:
    validate
    dtype FP32 -> key 1 -> R-001
    dtype FP16 -> key 2 -> R-002
    dtype BF16 -> key 3 -> R-002 (subject to entry compile guard)
    otherwise -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 直算 | key 1 | 非 regbase；1～256 个非空连续 ND FP32 Tensor；两个单元素 FP32 标量；UB guard | 单 AIV Kernel N1 | `ForeachNonFiniteCheckAndUnscaleND<float>::Process` | `copyInQueue`/`copyOutQueue`/`tempValQueue` 均 depth 2；V→S 等待归约标量 | 尾块 GM→UB 补 0、仅按实际长度回写；未知 key 静默结束 |
| R-002 FP16/BF16 升精度 | key 2/3 | 同上但统一 FP16/BF16；BF16 满足编译 guard | 单 AIV Kernel N1 | `ND<half>` 或 `ND<bfloat16_t>`，含两个 Cast 阶段 | 在 R-001 基础上增加 depth-2 FP32 `calcInQueue`/`calcOutQueue`；Queue 保证阶段所有权 | 32 B 对齐计算，实际长度回写；不满足 BF16 guard 时 key 3 分支不实例化 body |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| NPU 族 | `!IsRegbaseSocVersion(context_)` | membase body 使用传统 Vector API | R-001/R-002 | regbase template（950 卡） | `ForeachNonFiniteCheckAndUnscaleMembaseTiling::IsCapable` |
| 列表与空域 | `0<totalTensorCount<=256` 且每个 `n_i!=0` | POD 数组固定 256；membase 分核要求至少一个 block | 两 Route | `GRAPH_FAILED` | `ForeachNonFiniteCheckAndUnscaleTiling::Init` |
| 32 B 分核 | `a_i=ceil(n_i/e)*e` | DMA block 与 core 边界不切出小于 32 B 的分区 | 两 Route | 无其他 Route | `AssignDataToEachCore` |
| UB 上界 | `coefficient*S+R<=C` | depth 2 后实际总 UB 不超过 `U-P` | 两 Route | `GRAPH_FAILED` | `DivideUbMemory` |
| BF16 编译 guard | key 3 且 `__CCE_AICORE__>200` 且 arch 非 3003/3113 | 只在支持 BF16 的编译目标实例化 body | R-002 key 3 | 分支无操作 | Kernel entry lines 40-45 |
| workspace | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 在解析 key 前要求 user workspace 地址 | 两 Route | entry 直接返回 | Kernel entry lines 24-30 |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 宏实现位于固定仓外部框架头，故固定仓可确认字段声明顺序、类型和数组长度，但无法证明 `sizeof/alignof/offsetof`；Device 通过同名注册结构解析，Host 以 `SaveToBuffer` 序列化。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachNonFiniteCheckAndUnscaleTilingData` | `scaledGradsUbSize` | `uint32_t`/32 | 第 1 字段；精确 offset/总 sizeof 固定仓不可见 | 字节/slot | `set_scaledGradsUbSize(S)` | `ParseTilingData`; `InitBuffer`; `maxDataCount` | 32 B 向下对齐 |
| 同上 | `reduceTempValUbSize` | `uint32_t`/32 | 第 2 字段；精确 offset 固定仓不可见 | 字节/slot | `set_reduceTempValUbSize(R)` | `InitBuffer(tempValQueue,2,R)` | 4 B 倍数且按 32 B 归约结果对齐公式产生 |
| 同上 | `tensorDataCountList` | `int64_t[256]` | 第 3 字段 | 元素 | `GetStorageShape().GetShapeSize()` 逐项写 | `Process` 计算实际 tensor 末端 | 未使用尾项零初始化，不应读取 |
| 同上 | `tensorStartList` | `uint16_t[64]` | 第 4 字段 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 活跃 core 范围；最大 Tensor 索引 255 |
| 同上 | `tensorEndList` | `uint16_t[64]` | 第 5 字段 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 含端点 |
| 同上 | `tensorStartOffsetList` | `int64_t[64]` | 第 6 字段 | 元素 | `AssignDataToEachCore` | 首 Tensor `cursorStart` | 基于对齐扁平域 |
| 同上 | `tensorEndOffsetList` | `int64_t[64]` | 第 7 字段 | 元素 | `AssignDataToEachCore` | 末 Tensor `cursorEnd` 上限 | 含端点；再受实际 `n_i-1` 限制 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | priority 10000/key 1 | `..._tiling.cpp::Init/RunBigKernelTiling` | `ForeachNonFiniteCheckAndUnscaleTilingData` | `op_kernel/...cpp::foreach_non_finite_check_and_unscale` | `..._n_d.h::ND<float>` | `op_kernel/CMakeLists.txt`; ascend910b/910_93 binary JSON |
| R-002 | priority 10000/key 2/3 | 同上 | 同上 | 同上 key 2/3 | `..._n_d.h::ND<half/bfloat16_t>` | 同上，三个 dtype binary 条目 |

## Kernel 实现

### 公共执行流程

entry 显式标记 `KERNEL_TYPE_AIV_ONLY`，解析 POD 后按 key 构造模板。每个 core 根据 POD 的首尾 Tensor/offset 遍历自己的区间；TensorList GM 首个 `uint64_t` 是指针表偏移，右移 3 转为 `uint64_t` 项数，再取第 `i` 个 Tensor 地址。每个 Tensor 分 tile 执行 `CopyIn -> [CastToFloat] -> Compute -> [CastToOriginalType] -> CopyOut`。非有限值检测读取反缩放前的 FP32 `computeInLT`，与乘法共享同一次 GM→UB 输入。

### Route R-001：FP32 直算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host selector | `op_host/..._tiling.cpp` | `MembaseTiling::IsCapable` | `IsRegbaseSocVersion(context_)` 为 false | 采用 priority 10000 |
| Host key | 同上 | `GetTilingKeyVal` | `dataType==DT_FLOAT` | key 1 |
| Router | `op_kernel/...cpp` | `foreach_non_finite_check_and_unscale` | `TILING_KEY_IS(1)` | `ND<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_non_finite_check_and_unscale(..., key=1)
  -> ND<float>::Init -> ParseTilingData
  -> ND<float>::Process
     -> GetTensorAddr
     -> SingleTensorProcess
        -> CopyIn -> Compute(ReduceMax/ReduceMin, Muls) -> CopyOut
```

##### 源码等价伪码

```text
for tensor i in [tensorStart, tensorEnd]:
  start = (i==tensorStart ? startOffset : 0)
  end = min(n_i-1, i==tensorEnd ? endOffset : n_i-1)
  GM = tensor_ptr[i] + start
  for tile k in [0, ceil((end-start+1)/maxDataCount)):
    valid = min(maxDataCount, remaining)
    aligned = ceil(valid/(32/4))*(32/4)
    CopyIn(valid); pad right with zero if unaligned
    if !haveFoundInf:
      max = ReduceMax(input[0:aligned]); V_S wait; bit-test max
      if finite: min = ReduceMin(input[0:aligned]); V_S wait; bit-test min
      if nonfinite: foundInf[0]=1; haveFoundInf=true
    output[0:aligned] = input[0:aligned] * invScaleVal
    CopyOut(valid)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList GM | `dataAddr + ((*dataAddr)>>3) + i` | 256 指针以内 | `uint64_t` 指针表 | ACLNN runtime | `GetTensorAddr` | 调用方；整次 launch | 8 B 指针 |
| `scaledGradsGM` | `tensorPtr[i]+cursorStart+k*maxDataCount` | `valid` 元素 | FP32 ND | 调用方/前一 tile 回写 | `CopyIn`/`CopyOut` | 原地 Tensor | 尾 tile pad，不回写 padding |
| `copyInQueue` | Queue 管理 | `2*S` | FP32 | `CopyIn` | `Compute` | `TPipe`，Alloc→EnQue→DeQue→Free | depth 2 |
| `copyOutQueue` | Queue 管理 | `2*S` | FP32 | `Compute` | `CopyOut` | 同上 | depth 2 |
| `tempValQueue` | Queue 管理 | `2*R` | FP32 | Reduce workspace/result | scalar `GetValue(0)` | 仅 `!haveFoundInf` 的 Compute 内 Alloc→Free | depth 2 分配，实际一次占一个 slot |
| workspace | user slice `[0,32)` | 32 B | 未解释字节 | Host/runtime | body 不消费 | executor launch 期间 | entry 仅检查地址 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["copyInQueue · depth=2 · 2*S"]:1
  outQ["copyOutQueue · depth=2 · 2*S"]:1
  tempQ["tempValQueue · depth=2 · 2*R"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*S + 2*S + 2*R = 4*S + 2*R
Host guard: 2*S + R <= floor((U-P)/2/32)*32
=> UB_total <= U-P
dynamic_ub = N/A
workspace_total = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `copyInQueue` slot | `CopyIn` | `Compute` | 后续 tile `AllocTensor` | EnQue/DeQue/FreeTensor | Compute 完成后才 Free |
| `copyOutQueue` slot | `Muls` | `CopyOut` | 后续 tile | EnQue/DeQue/FreeTensor | GM CopyOut 完成后 Free |
| `tempValQueue` slot | `ReduceMax/Min` | Scalar `GetValue(0)` | 下一次需要检测的 tile | V→S flag/wait 后读取，随后 Free | `haveFoundInf` false 才分配 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + MTE2→S flag/wait + Queue | MTE2 | Queue consumer | 补零完成及 Queue 所有权 | `blockLen=valid*4` B；右补到 8 元素 |
| 归约→Scalar | `ReduceMax/Min` + V→S flag/wait | Vector | `GetValue(0)` | Reduce 结果 | FP32；指数域 `0xff` 判 Inf/NaN |
| 乘法 | `Muls(...,aligned)` | Vector | output Queue | 反缩放结果 | FP32；padding 也算但不回写 |
| UB→GM | `DataCopyPad`/`DataCopy` + Queue | MTE3 | FreeTensor | 输出 slot | 只提交 `valid` 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/..._tiling.cpp` | `RunBigKernelTiling/DivideUbMemory` | key、blockDim、S/R、32 B workspace |
| Kernel entry/router | `op_kernel/...cpp` | key 1 分支 | AIV-only 到 `ND<float>` |
| Device body/helper | `op_kernel/..._n_d.h` | `ND<float>::CopyIn/Compute/CopyOut` | pad、归约、短路、原地乘法与尾写 |

### Route R-002：FP16/BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host key | `op_host/..._tiling.cpp` | `GetTilingKeyVal` | `DT_FLOAT16`/`DT_BF16` | key 2/3 |
| Router | `op_kernel/...cpp` | key 2/3 分支 | key 2；或 key 3 + compile guard | `ND<half>`/`ND<bfloat16_t>` |
| Device compile branch | `op_kernel/..._n_d.h` | `ORIG_DTYPE_SCALED_GRADS != DT_FLOAT` | 编译期 true | 启用 calcIn/calcOut 与 Cast |

#### Device 数据流

##### 函数级调用链

```text
entry key 2/3 -> ND<T>::Init/Process
  -> CopyIn<T>
  -> CastToFloat<T>
  -> Compute<float>: ReduceMax/Min + Muls
  -> CastToOriginalType<T>(CAST_RINT)
  -> CopyOut<T>
```

##### 源码等价伪码

```text
valid/aligned 按 32 B 的 T 元素数计算
copyInT = GM tile, right-pad zero
calcInF32[0:aligned] = Cast<T->float>(copyInT)
若本 core 尚未发现非有限值：对 calcInF32 ReduceMax/Min 并位检测
calcOutF32 = calcInF32 * invScaleVal
copyOutT = Cast<float->T, CAST_RINT>(calcOutF32)
GM tile[0:valid] = copyOutT[0:valid]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `copyInQueue` | Queue | `2*S` | T | DMA | `CastToFloat` | Alloc→Free | depth 2 |
| `calcInQueue` | Queue | `2*(2*S)=4*S` | FP32 | `CastToFloat` | `Compute` | Alloc→Free | 每个 T tile 的 FP32 容量翻倍 |
| `calcOutQueue` | Queue | `4*S` | FP32 | `Muls` | `CastToOriginalType` | Alloc→Free | depth 2 |
| `copyOutQueue` | Queue | `2*S` | T | 降精度 Cast | DMA | Alloc→Free | depth 2 |
| `tempValQueue` | Queue | `2*R` | FP32 | Reduce | Scalar | 条件 Alloc→Free | 与 R-001 相同 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["copyIn · 2*S"]:1
  calcIn["calcIn FP32 · 4*S"]:1
  calcOut["calcOut FP32 · 4*S"]:1
  outQ["copyOut · 2*S"]:1
  tempQ["reduce temp · 2*R"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*S + 4*S + 4*S + 2*S + 2*R = 12*S + 2*R
Host guard: 6*S + R <= C
=> UB_total <= U-P
dynamic_ub = N/A
workspace_total = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `copyIn` slot | DMA | `CastToFloat` | 后续 tile | Queue + FreeTensor | Cast 已完成 |
| `calcIn` slot | `CastToFloat` | Reduce/Muls | 后续 tile | Queue + Compute Free | Muls 已消费 |
| `calcOut` slot | `Muls` | 降精度 Cast | 后续 tile | Queue + Cast Free | Cast 已消费 |
| `copyOut` slot | 降精度 Cast | DMA | 后续 tile | Queue + CopyOut Free | 实际长度已提交 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `Cast(...,CAST_NONE,aligned)` | copyIn Queue | calcIn Queue | T→FP32 | padding 0 同步进入 FP32 |
| 归约/乘法 | 同 R-001 | FP32 Vector | Scalar/下一 Queue | FP32 中间值 | 检测发生在缩放前 |
| 降精度 | `Cast(...,CAST_RINT,aligned)` | calcOut Queue | copyOut Queue | FP32→T | 最近整数舍入模式由源码固定 |
| 尾写 | `DataCopyPad` | copyOut Queue | GM | 原地结果 | 只写 `valid` 个 T 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/..._tiling.cpp` | `GetTilingKeyVal/DivideUbMemory` | key 2/3 与 coefficient 6 |
| Kernel entry/router | `op_kernel/...cpp` | key 2/3 | dtype 实例与 BF16 编译 guard |
| Device body/helper | `op_kernel/..._n_d.h` | `CastToFloat/Compute/CastToOriginalType` | FP32 检测/乘法及 `CAST_RINT` 降精度 |

## 算子特定优化

### Trick T-001：每 core 首次发现后跳过后续归约但继续反缩放

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：当前 core 的任一先前 tile 已由 max/min 指数位检测确认非有限值
- Golden 来源：[`foreach_non_finite_check_and_unscale_n_d.h:189-213`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/foreach_non_finite_check_and_unscale_n_d.h#L189-L213)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (!haveFoundInf) { // Inside the same Core, just find it once.
    LocalTensor<float> workLocal = tempValQueue.AllocTensor<float>();
    ReduceMax<float>(workLocal, computeInLT, workLocal, dataCount, false);
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    float maxValue = workLocal.GetValue(0);
    if (IsNonFinite(maxValue)) {
        foundInfGM.SetValue(0, 1.0);
        haveFoundInf = true;
    }
```

#### 机制与设计原因

`haveFoundInf` 是 core 私有状态。一旦该 core 已将全局 flag 置 1，后续 tile 不再分配 Reduce 临时 Tensor，也不再执行 max/min 与 V→S 标量等待；`Muls` 和 CopyOut 始终继续，保持“检测 + 全列表反缩放”的完整语义。它减少的是确认之后的归约与同步，不减少必须完成的 GM 搬运和乘法。

#### 迁移边界

- 适用：输出 flag 为单调、幂等的 sticky true，且主变换仍必须遍历全部输入。
- 前提：各 core 只写相同 true 值；false 情况必须保持输入 flag，不可在 core 尾部清零。
- 失效：若需要非有限值计数、位置或每 tile 结果，不能短路归约。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16 tile 在 FP32 中融合检测与缩放

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`ORIG_DTYPE_SCALED_GRADS != DT_FLOAT`
- Golden 来源：[`foreach_non_finite_check_and_unscale_n_d.h:238-257`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/foreach_non_finite_check_and_unscale_n_d.h#L238-L257)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<T> copyInLT = copyInQueue.DeQue<T>();
LocalTensor<float> calcInLT = calcInQueue.AllocTensor<float>();
Cast(calcInLT, copyInLT, RoundMode::CAST_NONE, dataCount);
calcInQueue.EnQue(calcInLT);
copyInQueue.FreeTensor(copyInLT);
```

#### 机制与设计原因

一个 T tile 只从 GM 装载一次；升为 FP32 后，同一 `calcInLT` 同时供 ReduceMax/Min 和 `Muls` 使用，避免为“检测”和“反缩放”分别加载输入。`calcOutLT` 最后以 `CAST_RINT` 降回原 dtype，精度顺序是 T→FP32 检测/乘法→T。

#### 迁移边界

- 适用：低精度输入需要同时做全局检测与逐元素变换。
- 前提：UB 能容纳两个 depth-2 FP32 Queue，Host 使用 coefficient 6 做字节上界。
- 失效：UB guard 失败直接 Tiling 失败，固定仓没有小 UB fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 直算 | R-002 FP16/BF16 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2/3 | dtype switch；同属 priority 10000 |
| 模板与 dtype | `ND<float>` | `ND<half/bfloat16_t>` | 是否需要显式 FP32 Cast |
| 循环与分核 | 相同 32 B block/core/tile | 相同 | 数据域映射共用 |
| Queue/TBuf/寄存器/dynamic UB | 3 个 depth-2 Queue | 5 个 depth-2 Queue | 低精度增加 FP32 中间 Queue |
| 内存总占用与复用 | `4S+2R` | `12S+2R` | Host coefficient 2/6 配对 |
| 搬运/Vector API | Copy、Reduce、Muls | Copy、Cast、Reduce、Muls、Cast | 低精度升精度路径 |
| 同步、精度与 tail | FP32；V→S；32 B pad | T→FP32→T；同样 tail | 检测在缩放前 |
| fallback/失败边界 | 资源失败即终止 | BF16 还受编译 guard | 无小 tile fallback Route |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开支持，见 `_950` 卡 |
| Host/Launcher | 共用非 regbase priority 10000 Tiling | 共用非 regbase priority 10000 Tiling | regbase priority 30000；POD core 数组扩至 80 |
| Kernel/资源 | membase AIV body | membase AIV body | arch35 MicroAPI body，另有 max/min TBuf 与 cache flush |

## 限制与失败边界

- `scaledGrads` 长度必须为 1～256，各 Tensor dtype 相同且非空；公开 ACLNN 的原地输入不支持非连续 Tensor。
- FP32/FP16/BF16 之外的 dtype、`foundInf`/`invScale` 非 FP32、平台 UB 为 0 或 UB guard 失败都会终止 Tiling；没有备用 Route。
- membase Tiling 没有再次验证 `foundInf`/`invScale` 元素数为 1；该校验仅由公开 ACLNN bridge 契约支撑。
- Host 使用 `int64_t` 累加对齐元素数，但 `GetNeedCoreNum` 把商构造成 `uint32_t`，固定源码没有显式溢出 guard。
- entry 对空 workspace/user workspace 直接返回且不写错误；unknown key 也无计算。
- 多 core 对 `foundInf[0]` 的写入不是 atomic；源码仅依赖所有写者都写相同的 `1.0f`，固定仓未给出独立内存模型证明。
- 固定仓未公开生成 ACLNN bridge、executor 内部 stream enqueue 代码以及 TilingData 宏的精确 ABI 布局；这些均作为框架边界标记，不推测补齐。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/.../docs/aclnnForeachNonFiniteCheckAndUnscale.md` | 两段式函数原型 | ACLNN 参数、错误与公开平台 |
| 平台/语义 | `foreach/.../README.md` | 产品表/公式 | A2/A3/950 支持和算子语义 |
| Host OpDef | `foreach/.../op_host/foreach_non_finite_check_and_unscale_def.cpp` | `ops::ForeachNonFiniteCheckAndUnscale` | dtype/format/AutoContiguous/平台注册 |
| Host 入口 | `foreach/.../op_host/foreach_non_finite_check_and_unscale_base_tiling.cpp` | `Tiling4ForeachNonFiniteCheckAndUnscale` | Tiling registry 入口 |
| Host/Tiling/POD | `foreach/.../op_host/foreach_non_finite_check_and_unscale_tiling.h` | 两个 TilingData 定义 | POD 字段与容量 |
| Host/Tiling | `foreach/.../op_host/foreach_non_finite_check_and_unscale_tiling.cpp` | membase selector/tiling | A2/A3 校验、分核、UB、key |
| Kernel entry | `foreach/.../op_kernel/foreach_non_finite_check_and_unscale.cpp` | `foreach_non_finite_check_and_unscale` | AIV-only key router |
| Kernel body | `foreach/.../op_kernel/foreach_non_finite_check_and_unscale_n_d.h` | `ForeachNonFiniteCheckAndUnscaleND` | 完整传统 Vector 实现 |
| binary 接线 | `foreach/.../op_host/config/ascend910b/foreach_non_finite_check_and_unscale_binary.json` | 三个 dtype 条目 | A2/A3 dtype binary 信号 |
| 构建接线 | `foreach/.../op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend910b/910_93 使用 membase source |
| selector 框架 | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 升序与失败传播 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 两个 size、元素数及每 core 首尾四数组 | 字段声明顺序、位宽、数组 256/64、元素/字节单位 | `ForeachNonFiniteCheckAndUnscaleTilingData` |
| 调用顺序 | CopyIn→可选升精度→Compute→可选降精度→CopyOut | 检测必须读取缩放前值；乘法不能因已发现而跳过 | `SingleTensorProcess` |
| workspace/内存 | 32 B workspace；depth 2 Queue 字节公式 | coefficient 2/6 必须与实际 Queue 配对 | `DivideUbMemory/Init` |
| 同步 | Queue 生命周期与 Reduce 后 V→S | 标量读取前必须等待 Vector | `Compute` |
| 精度与尾块 | 低精度在 FP32 计算，`CAST_RINT` 返回；尾块补 0 | CopyOut 只能写实际元素数 | `CastToFloat/CastToOriginalType/CopyOut` |
| 失败边界 | selector fallthrough 与 `GRAPH_FAILED` | 不得把 unknown key、空 workspace 当成功计算 | selector/entry |