---
type: Operator
title: "ForeachMinimumScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachMinimumScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/op_api/aclnn_foreach_minimum_scalar_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/op_api/aclnn_foreach_minimum_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/op_api/foreach_minimum_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/arch35/foreach_minimum_scalar_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/foreach_minimum_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/config/ascend950/foreach_minimum_scalar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:48:04Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMinimumScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMinimumScalarV2GetWorkspaceSize` / `aclnnForeachMinimumScalarV2`，公开 ACLNN V2；旧版公开产品表明确不支持 950。 |
| 计算语义 | 对 TensorList 每项逐元素计算 `y_t[i]=min(x_t[i],scalar)`；浮点 SIMT helper显式传播 NaN并在 ±0 相等时选择 -0。 |
| 输入输出 | x/out 为 ND TensorList，FP16/FP32/INT32/BF16；scalar 为 `aclScalar`，Host 转单元素 Tensor，BF16 x 的 scalar Tensor 是 FP32。 |
| 执行类型 | 单 Kernel `SIMT`：AIV entry按 dtype模板调用一个 1024-thread VF；全局 thread grid以 grid-stride 直接读写 GM。 |
| Kernel 路线 | R-001 FP32、R-002 INT32、R-003 FP16/BF16→FP32计算；四个 schMode selector均落到同一 SIMT launch拓扑。 |
| 核心资源 | TensorList GM指针表、tiling GM数组、线程局部标量；无源码拥有的 TQue/TBuf、显式 shared/dynamic UB或 workspace。Host保留 `UB-128 KiB` local-memory launch额度。 |
| 来源闭包 | `source_closed`：V2 ACLNN、L0 launcher、950公开支持、arch35 Tiling/POD/template selector、SIMT entry/body、dtype config与构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x=[x_0,...,x_{N-1}]`，每个有效索引计算 `y_t[i]=min(x_t[i],scalar)`。FP32直接在 float 上调用 `SimtMinimum(float,float)`；FP16/BF16先以 `static_cast<float>` 升精度，比较后 `static_cast<T>` 写回；INT32调用整数重载。浮点重载按源码顺序先返回任一 NaN操作数，再比较大小，相等时优先返回带负号的操作数，从而 `min(+0,-0)` 和 `min(-0,+0)` 均得到 -0。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，每项 ND、0–8维 | FP16/FP32/INT32/BF16，列表内一致 | 必选；V2文档称支持空 TensorList。API源码逐项执行 `Contiguous`，但公开表把非连续 x 标为不支持，复用时遵循公开约束。 |
| `scalar` | `aclScalar` | x=FP32/BF16时 FP32/DOUBLE；x=FP16时 FP16/DOUBLE；x=INT32时 INT32/INT64 | Host校验数值范围后转换成 Tensor；BF16对应 FP32 Tensor，其余转换为 x dtype。 |
| `out` | TensorList，每项 ND、0–8维 | 与 x 一致 | API要求逐项 shape相等，公开文档不支持非连续 out；Tiling只按 x元素数写输出。 |
| `workspaceSize` | Host输出 | `uint64_t` | executor全图 workspace；核心 950 Tiling明确写0。 |
| `executor`/`workspace`/`stream` | 两段式资源 | 框架类型 | 第一段构图并转移 executor所有权，第二段在 stream运行。 |

### 入口与版本

公开旧版 `aclnnForeachMinimumScalar` 接收 scalar Tensor，其产品表对 Ascend 950PR/950DT 标为不支持；公开 V2改为 `aclScalar`，产品表对 A2/A3/950均标支持。因此 950只提取 V2。V2 Host仍排队内部 `ForeachMinimumScalar`，950由构建接线选择 arch35 Tiling和 SIMT Kernel，不复用 arch22 Vector body。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段依次创建 executor；检查 x/scalar/out非空指针；`GetDtypeSupportList` 在 950的 RegBase架构返回 FP32/FP16/BF16/INT32；检查列表dtype一致、out匹配及 scalar dtype组合；检查每项≤8维、x/out对应 shape相等、存储格式非私有。非空列表逐项调用 `l0op::Contiguous`，再以 `CheckScalarValueValid` 检查 scalar可表示范围；BF16 x调用 `ConvertToTensor(scalar,DT_FLOAT)`，其它 dtype转换成 x dtype；最后 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachMinimumScalar)`。任一步失败立即返回。

源码意图在 x列表大小为0时返回 `workspaceSize=0` 且不排队 Kernel；但 `CheckDtypeValid` 在该分支前读取 `(*x)[0]`，固定仓缺少 `aclTensorList::operator[]` 实现，空列表是否安全不可由本仓闭合。

### 参数、Workspace 与生命周期

Contiguous结果 TensorList与 scalar Tensor由 executor持有；第一段用 `GetWorkspaceSize` 返回整个图需求并把所有权转给调用者。950核心 Tiling `currentWorkspace[0]=0`，Kernel参数仍保留 workspace但 body不读。Tiling调用 `SetLocalMemorySize(uint32_t(ubSize-128*1024))`；这是 launcher的 local-memory额度 producer，SIMT源码没有显式切片或共享数组，其内部消费布局属于 `VF_CALL`框架边界。源码只检查 `ubSize!=0`，未检查 `ubSize>=128 KiB`。

### 启动图与失败传播

```text
aclnnForeachMinimumScalarV2GetWorkspaceSize
  -> [每个 x 项可选] Contiguous
  -> ConvertToTensor(scalar,target dtype)
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachMinimumScalar)
aclnnForeachMinimumScalarV2
  -> CommonOpExecutorRun(stream)
    -> foreach_minimum_scalar<schMode><<<blockDim=needCoreNum, localMem=UB-128KiB>>>
      -> VF_CALL<OpForeachMinimumScalarSimt<T,S>>(threads=1024)
```

只有一个核心 Device节点，没有 Kernel间 workspace/GM交接边。该节点输入 x TensorList GM、scalar Tensor GM、out TensorList GM与2064 B POD；workspace slice为0。executor/stream保证前处理节点先于核心节点可见；前序或 launch失败由 `CommonOpExecutorRun` 返回并阻断完成语义，具体框架 event/offset固定仓不可见。

## Route 与参数设计

### 分核与资源计算

Tiling读取 AIV核数 `C` 与 UB大小 `U`，读取 TensorList实例数 `N`（要求`N<=256`），并把每项 storage元素数写入 `tensorElements[t]`，累加 `E=ΣtensorElements[t]`。若 `E==0`，`needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(C,ceil(E/1024))`。Device不读取 `needCoreNum` 或 `totalElements`，实际 grid由 launch的 blockDim体现。每个 block内 VF固定1024线程，因此全局线程数 `G=blockDim*1024`；每个 Tensor都由 `idx=global_tid; idx<count; idx+=G`覆盖。

workspace为0；local memory launch额度为 `uint32_t(U-131072)`。SIMT body没有 `InitBuffer`、Queue、shared数组、atomic或barrier，逐线程仅持有 x/s/result和索引变量。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | schMode 0，`Process<float,float>` | `dataType==DT_FLOAT`→`GET_TPL_TILING_KEY(0)` | FP16→R-003，INT32→R-002，BF16→R-003 |
| 2 | R-003 FP16升精度 SIMT | schMode 1，`Process<half,half>` | `dataType==DT_FLOAT16`→selector 1 | INT32→R-002，BF16→R-003 |
| 3 | R-002 INT32 SIMT | schMode 2，`Process<int32_t,int32_t>` | `dataType==DT_INT32`→selector 2 | BF16→R-003 |
| 4 | R-003 BF16升精度 SIMT | schMode 3，`Process<bfloat16_t,float>` | `dataType==DT_BF16`→selector 3 | 其它 dtype Tiling失败 |

```text
FP32 -> schMode 0 -> R-001
else FP16 -> schMode 1 -> R-003<half,half>
else INT32 -> schMode 2 -> R-002
else BF16 -> schMode 3 -> R-003<bfloat16_t,float>
else -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 SIMT | schMode 0 | 950 RegBase；FP32；API及N≤256校验通过 | 单 entry→单1024-thread VF | `Process<float,float>`→`Op...<float,float>`→`SimtMinimum(float)` | GM直读写；线程局部float；无同步 | `idx<count`自然 tail；未知 selector无 body |
| R-002 INT32 SIMT | schMode 2 | 950 RegBase；INT32；其余同上 | 同拓扑 | `Process<int32_t,int32_t>`→整数重载 | GM直读写；无atomic/barrier | grid-stride tail |
| R-003 FP16/BF16升精度 SIMT | schMode 1/3 | FP16或BF16；BF16 scalar Tensor为FP32 | 同拓扑 | `ComputeType<T>::type=float`，`static_cast<float>`→float helper→`static_cast<T>` | 无FP32 UB物化，值在线程局部 | 每元素guard；降精度round mode固定仓未命名 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList定长 | `tensorNum<=256` | POD仅有256个 `int64_t` | 全部 | Tiling `GRAPH_FAILED` | `ForeachMinimumScalarTilingFunc` |
| 每核最少1024元素启发式 | `blockDim=min(C,ceil(E/1024))` | 控制全局SIMT block数 | 全部 | E=0固定1 block | 同函数 |
| VF线程数 | `THREAD_NUM=1024`，`LAUNCH_BOUND(1024)` | VF实例与 grid-stride固定 | 全部 | 无其它threads fallback | `OpForeachMinimumScalarSimt`/`VF_CALL` |
| local memory保留 | `SetLocalMemorySize(uint32_t(U-128KiB))` | 预留DCache 128 KiB | 全部 | 源码仅检查U非0；U<128KiB会无符号下溢后窄化 | Tiling函数 |
| dtype selector | 仅FP32/FP16/INT32/BF16 | entry只有四个编译期实例 | 全部 | Tiling报unsupported dtype | Tiling/entry |
| 空 Tensor | 每 Tensor `count==0`直接continue | 避免地址解析与循环 | 全部 | 继续下一 Tensor | `OpForeachMinimumScalarSimt` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMinimumScalarTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8，sizeof 2064 | AIV block数 | E=0写1，否则 `min(C,ceil(E/1024))` | body不读；对应`SetBlockDim` | C先校验非0，窄化到int32无显式上界检查 |
| `ForeachMinimumScalarTilingData` | `tensorCount` | `int32_t`/32 | offset 4 | Tensor个数 | `static_cast<int32_t>(tensorNum)` | `Process`传给VF | 0–256 |
| `ForeachMinimumScalarTilingData` | `totalElements` | `int64_t`/64 | offset 8 | 元素 | `Σnumel` | body不读 | 累加无溢出检查 |
| `ForeachMinimumScalarTilingData` | `tensorElements` | `int64_t[256]` | offset 16，2048 B | 元素/Tensor | 循环写 storage shape size，其余 memset为0 | VF按`t<tensorCount`读 `tensorElements[t]` | 每项负值未显式检查；shape producer应给非负数 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 SIMT | schMode 0 | `op_host/arch35/foreach_minimum_scalar_tiling.cpp::ForeachMinimumScalarTilingFunc` | `ForeachMinimumScalarTilingData` | `op_kernel/arch35/foreach_minimum_scalar.cpp::foreach_minimum_scalar<0>` | `foreach_minimum_scalar_simt.h::Process<float,float>/Op.../SimtMinimum(float)` | host/kernel CMake + ascend950 binary config |
| R-002 INT32 SIMT | schMode 2 | 同上 | 同 POD | `foreach_minimum_scalar<2>` | `Process<int32_t,int32_t>`及int重载 | 同上 |
| R-003 FP16/BF16升精度 SIMT | schMode 1/3 | 同上 | 同 POD | `foreach_minimum_scalar<1/3>` | `ComputeType<half/bfloat16_t>`、`Process`、float重载 | 同上 |

## Kernel 实现

### 公共执行流程

entry由编译期 `schMode`选择 `Process<T,S>`。每个 outer block先从 scalar GM读取一次 `S scalarVal`，并以 `offsetof`取得 tiling GM中 `tensorElements`数组地址；随后以 `Dim3(1024)`调用 VF。VF按 Tensor顺序遍历；空 Tensor直接跳过，非空 Tensor通过 TensorList首个 `uint64_t`偏移定位指针表。全局线程编号和全局线程总数形成 grid-stride，线程只写属于自己的idx，因此无需atomic/barrier，也没有跨线程中间内存。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_minimum_scalar_v2.cpp` | `GetDtypeSupportList/CheckDtypeValid` | RegBase且x FP32，scalar FP32/DOUBLE | scalar转FP32 Tensor并排队 |
| Tiling | `op_host/arch35/foreach_minimum_scalar_tiling.cpp` | `ForeachMinimumScalarTilingFunc` | `dataType==DT_FLOAT` | `GET_TPL_TILING_KEY(0)` |
| Router | `op_kernel/arch35/foreach_minimum_scalar.cpp` | `foreach_minimum_scalar<schMode>` | `schMode==0` | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_minimum_scalar<0>
  -> Process<float,float>
    -> scalarVal=*scalarGm; elemCounts=&tiling.tensorElements
    -> VF_CALL<OpForeachMinimumScalarSimt<float,float>>(threads=1024)
      -> SimtGetTensorAddr<float>
      -> grid-stride loop -> SimtMinimum(float,float) -> yData[idx]
```

##### 源码等价伪码

```text
# 源码等价伪码
scalarVal = *(float*)scalar
global_tid = blockIdx*threadNum + threadIdx
stride = threadNum*blockNum
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0: continue
  xData=TensorListPtr<float>(x,t); yData=TensorListPtr<float>(y,t)
  for idx=global_tid; idx<count; idx+=stride:
    a=xData[idx]; b=scalarVal
    if isnan(a): r=a
    else if isnan(b): r=b
    else if a<b: r=a
    else if b<a: r=b
    else: r=signbit(a) ? a : b
    yData[idx]=r
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y TensorList GM | `*(ptrTable+t)+idx` | `tensorElements[t]` | FP32/ND | caller/Contiguous、thread store | SIMT thread/caller | 调用期 | 每线程离散4 B load/store，无显式DMA |
| scalar GM | element0 | 4 B | FP32 | executor转换 | outer `Process` | executor | 每 outer block读一次 |
| tiling GM | offset16+t*8 | 2064 B struct | int64 counts | Host | Process/VF | launch期只读 | `offsetof`定位数组 |
| 线程局部 | 寄存器/编译器栈 | a,b,r,tid,stride | FP32/uint64 | 当前线程 | 当前线程 | 单迭代 | 固定仓不暴露字节数 |
| local memory | runtime额度 | `uint32_t(U-131072)` | 框架内部 | Host launch | VF runtime | Kernel期 | 源码无切片，不能计入自有buffer |
| workspace | N/A | 0 B | N/A | Host | 无 | N/A | 无 |

##### 分层内存布局

未使用 AIV UB Queue/TBuf。实际替代路径是 SIMT线程直接GM load→线程局部值→GM store；launcher设置 `local_memory_size=uint32_t(U-131072)`，但 `VF_CALL`内部布局固定仓不可见。

##### 字节账与复用顺序

```text
UB_total(source-owned TQue/TBuf/LocalTensor) = 0 B
dynamic_ub/local_memory launch额度 = uint32_t(platform_UB - 131072)
workspace_total = 0 B（核心节点）
register_bytes = 固定仓不可见；线程局部值仅活跃于单元素迭代
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程局部a/r | 当前idx load/比较 | 当前idx store | 同线程下一idx | 顺序程序语义 | 不跨线程共享 |
| y GM元素 | 唯一global_tid序列 | caller后续stream节点 | 本算子不覆盖第二次 | idx同余类互斥 | block/thread数在launch期固定 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程映射 | grid-stride | launch grid | 当前线程 | y元素独占 | `idx<count`处理任意尾数 |
| 比较 | `SimtMinimum(float)` | GM load | GM store | 单元素 | 显式NaN传播与-0 tie |
| 跨节点 | stream/executor | 前处理 | 核心launch | 连续化/scalar Tensor | 具体event固定仓不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_minimum_scalar_tiling.cpp` | `ForeachMinimumScalarTilingFunc` | blockDim、POD、local memory、selector |
| Kernel entry/router | `op_kernel/arch35/foreach_minimum_scalar.cpp` | `foreach_minimum_scalar<0>` | FP32实例 |
| Device body/helper | `op_kernel/arch35/foreach_minimum_scalar_simt.h` | `Process/Op.../SimtMinimum(float)` | scalar读取、线程循环、浮点边界 |

### Route R-002：INT32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_minimum_scalar_v2.cpp` | `CheckDtypeValid` | x INT32，scalar INT32/INT64且值在int32范围 | scalar转INT32 Tensor |
| Tiling | `arch35/foreach_minimum_scalar_tiling.cpp` | Tiling函数 | `dataType==DT_INT32` | selector 2 |
| Router | `arch35/foreach_minimum_scalar.cpp` | entry | `schMode==2` | `Process<int32_t,int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_minimum_scalar<2>
  -> Process<int32_t,int32_t>
    -> VF_CALL<OpForeachMinimumScalarSimt<int32_t,int32_t>>(1024)
      -> grid-stride -> SimtMinimum(int32_t,int32_t) -> GM store
```

##### 源码等价伪码

```text
# 外层Tensor与grid-stride同R-001
a = xData[idx]; b = scalarVal
r = (a <= b) ? a : b
yData[idx] = r
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | TensorListPtr+t/idx | count×4 B | INT32/ND | caller/thread | thread/caller | 调用期 | 直接GM load/store |
| scalar GM | element0 | 4 B | INT32 | Host转换 | Process | executor | 每block一次 |
| 线程局部/local memory/workspace | 同R-001 | source-owned UB=0，local额度U-128KiB，workspace=0 | INT32/索引 | 当前线程/Host | 当前线程/VF | 迭代/Kernel | 无共享复用 |

##### 分层内存布局

未使用 AIV UB；SIMT线程直接 GM→寄存器局部→GM。`VF_CALL` local-memory内部布局固定仓不可见。

##### 字节账与复用顺序

```text
UB_total = 0 B source-owned
dynamic_ub/local_memory = uint32_t(U-131072)
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread local | 当前idx | 当前idx store | 下一idx | 顺序语义 | 线程私有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 比较 | `(a<=b)?a:b` | INT32 load | store | 单元素 | 无cast/溢出算术；idx<count tail |
| 并发 | disjoint grid-stride | launch | threads | y元素 | 无atomic/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_minimum_scalar_tiling.cpp` | dtype selector | INT32→2 |
| Kernel entry/router | `op_kernel/arch35/foreach_minimum_scalar.cpp` | `foreach_minimum_scalar<2>` | INT32模板 |
| Device body/helper | `op_kernel/arch35/foreach_minimum_scalar_simt.h` | `SimtMinimum(int32_t)` | 整数比较与grid-stride |

### Route R-003：FP16/BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_minimum_scalar_v2.cpp` | `CheckDtypeValid/Exec...` | FP16 scalar FP16/DOUBLE；BF16 scalar FP32/DOUBLE | 分别转FP16/FP32 Tensor |
| Tiling | `arch35/foreach_minimum_scalar_tiling.cpp` | Tiling函数 | FP16→selector1；BF16→selector3 | R-003两个实例 |
| Router | `arch35/foreach_minimum_scalar.cpp` | entry | schMode 1/3 | `Process<half,half>` / `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_minimum_scalar<1 or 3>
  -> Process<T,S> -> scalarVal=*scalarGm
  -> VF_CALL<OpForeachMinimumScalarSimt<T,S>>(1024)
    -> ComputeType<T>::type == float
    -> static_cast<float>(x/scalar) -> SimtMinimum(float) -> static_cast<T> -> yGM
```

##### 源码等价伪码

```text
# Tensor/grid循环同R-001
float xVal = static_cast<float>(xData[idx])
float sVal = static_cast<float>(scalarVal)
float result = SimtMinimum(xVal,sVal)
yData[idx] = static_cast<T>(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | TensorListPtr+t/idx | count×2 B | FP16或BF16/ND | caller/thread | thread/caller | 调用期 | 直接GM访问 |
| scalar GM | element0 | FP16 2 B或FP32 4 B | S | Host转换 | Process | executor | 每block一次 |
| thread locals | 私有 | xVal/sVal/result各float | FP32 | casts/helper | cast/store | 单迭代 | 不物化到源码UB |
| local memory/workspace | N/A自有布局 | local额度U-128KiB；workspace0 | 框架 | Host | VF runtime | Kernel | 固定仓不可见布局 |

##### 分层内存布局

未使用 AIV UB Queue/TBuf。FP32中间值是线程局部标量，不存在共享 FP32 tile。

##### 字节账与复用顺序

```text
UB_total(source-owned) = 0 B
dynamic_ub/local_memory = uint32_t(U-131072)
workspace_total = 0 B
register_bytes = 固定仓不可见（三个FP32值的逻辑生命周期可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local FP32值 | 当前元素casts/helper | 当前元素store | 下一idx | 顺序语义 | 无跨线程地址共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `static_cast<float>` | FP16/BF16 GM与S | helper | thread local | 每元素FP32比较 |
| 比较 | float helper | xVal/sVal | result | thread local | NaN传播、-0 tie同R-001 |
| 降精度 | `static_cast<T>` | result | y store | 单元素 | 固定源码未命名round mode；idx<count保护tail |
| 并发 | disjoint grid-stride | grid | threads | y元素 | 无barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_minimum_scalar_tiling.cpp` | dtype selector | FP16/BF16→1/3 |
| Kernel entry/router | `op_kernel/arch35/foreach_minimum_scalar.cpp` | schMode1/3 | T/S模板组合 |
| Device body/helper | `op_kernel/arch35/foreach_minimum_scalar_simt.h` | `ComputeType`/`Op...` | 在线程局部升/降精度 |

## 算子特定优化

### Trick T-001：一个全局 SIMT grid复用到每个 Tensor

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：非空 Tensor项；`blockDim`与固定1024线程已由Host/模板确定。
- Golden 来源：[`foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_simt.h:116-127`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_simt.h#L116-L127)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
    AscendC::Simt::GetThreadIdx());
uint64_t stride = static_cast<uint64_t>(
    AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
```

#### 机制与设计原因

每个 Tensor都复用同一个全局线程grid，thread `k`处理所有满足 `idx≡k mod G` 的元素。TensorList只需一次 Kernel launch，不需要为每项构造POD分段或AIV Queue；`idx<count`天然覆盖任意尾数，且不同线程同余类不相交，所以无需atomic/barrier。Host以总元素数每1024个元素增加一个block，最多平台AIV核数。

#### 迁移边界

- 适用：各输出元素独立、Tensor间无依赖的列表逐元素操作。
- 前提：每项count可用uint64循环比较，输出不与其它线程目标重叠，TensorList指针表ABI一致。
- 失效：归约、邻域、跨Tensor状态或原地读写冲突需要额外同步/划分。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若把 stride误写成单block线程数，多个block会重复写同一输出元素。

### Trick T-002：浮点最小值显式固化 NaN 与有符号零

#### 触发与源码依据

- 适用 Route：R-001、R-003
- 触发条件：比较类型为float；FP16/BF16先升到float。
- Golden 来源：[`foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_simt.h:58-83`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_simt.h#L58-L83)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isnan(a)) {
    return a;
}
if (isnan(b)) {
    return b;
}
```

#### 机制与设计原因

helper避免直接使用可能对NaN/±0有不同约定的普通比较内建：先传播NaN，再做双向小于，最后以`signbit(a)`选择相等值中的负号。这样FP32及升精度后的FP16/BF16路线具有源码可审计的特殊值次序；代价/收益未做性能实测。

#### 迁移边界

- 适用：要求IEEE风格NaN传播和`min(+0,-0)=-0`的浮点minimum。
- 前提：输入已按目标语义升到float，降精度仍能表示NaN与负零。
- 失效：若目标算子采用`fmin`的单NaN忽略规则或不同tie语义，不能复用该helper。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供另一实现反例；简单写`a<b?a:b`会在NaN和`a=-0,b=+0`时改变本源码定义的结果。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 INT32 | R-003 FP16/BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode0 | schMode2 | schMode1/3 | dtype决策树 |
| 模板与 dtype | `<float,float>` | `<int32_t,int32_t>` | `<half,half>`/`<bfloat16_t,float>` | scalar Tensor ABI及ComputeType |
| 循环与分核 | 统一grid-stride | 同左 | 同左 | 单launch覆盖列表 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无Queue，float局部 | int局部 | FP32局部 | local额度相同，源码布局不可见 |
| 内存总占用与复用 | source-owned UB=0 | 同左 | 同左 | 直接GM load/store |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT直接访问 | SIMT直接访问 | SIMT + static_cast | 无Vector DMA/NDDMA/通信 |
| 同步、精度与 tail | NaN/-0 helper | int比较 | FP32 helper后降精度 | 全部idx<count、无barrier |
| fallback/失败边界 | 未知selector无body | 同左 | 同左 | unsupported dtype在Tiling失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2公开支持 | V2公开支持 | V2公开支持；旧版公开不支持 |
| Host/Launcher | arch22公共Foreach Tiling，列表/核上限50 | 同A2 | arch35专用Tiling，列表上限256、每1024元素一block |
| Kernel/资源 | AIV Vector双Queue，BF16使用FP32 Queue | 同A2 | SIMT直接GM，1024线程，无源码拥有的UB Queue/workspace |

## 限制与失败边界

- 950 Tiling显式限制 TensorList≤256；`totalElements`以`int64_t`累加且没有溢出检查。`tensorElements`负值也无显式guard，依赖合法shape producer。
- `SetLocalMemorySize`只检查UB非0而没有检查UB≥128 KiB；减法发生在无符号`uint64_t`并窄化到`uint32_t`，迁移到不同资源规格时必须补同等前置保证。
- 文档声明空 TensorList可用，但 API在`Size()==0`判断前访问`(*x)[0]`；该访问器实现固定仓不可见，不能证明空列表安全路径。
- 文档不支持非连续x/out；尽管源码对x调用Contiguous，本卡不扩大公开支持。Contiguous与executor内部workspace/event切片固定仓不可见。
- FP16/BF16降精度使用C++ `static_cast<T>`，固定源码未给出硬件round mode；不能假定与A2/A3的`CAST_RINT`完全一致。
- 950 source-owned workspace为0，executor仍可能因框架前处理返回非零workspace。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalarV2.md` | 产品表/V2原型 | 950公开支持与V2约束 |
| 入口声明/约束 | `foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalar.md` | 旧版产品表 | 旧版950明确不支持 |
| 入口声明/约束 | `foreach/foreach_minimum_scalar/op_host/op_api/aclnn_foreach_minimum_scalar_v2.h` | V2 C ABI | 两段式声明 |
| 入口实现 | `foreach/foreach_minimum_scalar/op_host/op_api/aclnn_foreach_minimum_scalar_v2.cpp` | `Exec.../Check*` | 校验、前处理、executor |
| 入口实现 | `foreach/foreach_minimum_scalar/op_host/op_api/foreach_minimum_scalar_v2.cpp` | `l0op::ForeachMinimumScalarV2` | 核心launcher排队 |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar/op_host/arch35/foreach_minimum_scalar_tiling.cpp` | `ForeachMinimumScalarTilingFunc` | POD、blockDim、workspace、local memory、selector |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_tiling_data.h` | POD struct | ABI布局 |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_tiling_key.h` | template args/selection | schMode集合 |
| Kernel entry/body | `foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar.cpp` | templated entry | selector到Process |
| Kernel entry/body | `foreach/foreach_minimum_scalar/op_kernel/arch35/foreach_minimum_scalar_simt.h` | `Process/Op.../SimtMinimum` | 完整SIMT body |
| 构建接线 | `foreach/foreach_minimum_scalar/op_host/foreach_minimum_scalar_def.cpp` | 950 AICore config | dtype/format/动态shape注册 |
| 构建接线 | `foreach/foreach_minimum_scalar/op_host/config/ascend950/foreach_minimum_scalar_binary.json` | 四个bin条目 | 950 dtype组合 |
| 构建接线 | `foreach/foreach_minimum_scalar/op_host/CMakeLists.txt` | ascend950→arch35 | Host Tiling选择 |
| 构建接线 | `foreach/foreach_minimum_scalar/op_kernel/CMakeLists.txt` | arch35 kernel source | 950 Device构建 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 显式2064 B POD可直接复用 | offset 0/4/8/16、256项int64数组、Host/Device同struct | `ForeachMinimumScalarTilingData` |
| 调用顺序 | V2 scalar转Tensor后复用内部Foreach op | BF16 scalar Tensor必须FP32；前处理先于核心stream节点 | `Exec...` / `l0op::ForeachMinimumScalarV2` |
| workspace/内存 | 核心零workspace、直接GM | local memory=`uint32_t(U-128KiB)`及其资源前提；executor workspace不可写死0 | Tiling函数 / SIMT body |
| 同步 | disjoint grid-stride免atomic/barrier | `tid`必须含blockIdx且stride必须含blockNum | `OpForeachMinimumScalarSimt` |
| 精度与尾块 | FP16/BF16在线程局部FP32比较 | NaN/-0 helper顺序、`idx<count`、降精度static_cast语义 | `ComputeType/SimtMinimum/Op...` |
| 失败边界 | N>256或unsupported dtype在Tiling失败 | 空列表访问、UB减128KiB、totalElements溢出边界 | API Check/Tiling函数 |
