---
type: Operator
title: "ForeachMinimumScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachMinimumScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/docs/aclnnForeachMinimumScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/foreach_minimum_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/arch35/foreach_minimum_scalar_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/config/ascend950/foreach_minimum_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:07:57Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMinimumScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMinimumScalarListGetWorkspaceSize` + `aclnnForeachMinimumScalarList`，公开 ACLNN 两段式接口；固定仓未出现带版本后缀的新入口。 |
| 计算语义 | `y_i[j]=min(x_i[j],scalars[i])`；浮点显式传播任一操作数的 NaN。 |
| 输入输出 | ND TensorList `x/y`：FP16/FP32/INT32/BF16；内部 scalar Tensor：FP32 或 INT64；最多 256 个 Tensor。 |
| 执行类型 | 单节点 `SIMT`：AIV entry 内 `VF_CALL` 1024 threads/block；无 Cube/AIC 计算或 `MIX_CV` 节点。 |
| Kernel 路线 | R-001 FP32/FP32；R-002 FP16/FP32；R-003 INT32/INT64；R-004 BF16/FP32。 |
| 核心资源 | TensorList GM 指针表、scalar GM、tiling GM；线程直接 load/compare/store GM，无 `TQue/TBuf`、shared/dynamic UB 或 operator workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档/ACLNN 构建接线→OpDef/infer→arch35 Tiling/POD/key→SIMT router/body、线程映射、精度、tail 与构建均闭合；自动生成 ACLNN bridge 属框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`、`scalars=[s_0,...,s_{n-1}]`。对每项扁平元素 `j<numel(x_i)`，`MinimumWithNaN(a,b)` 先检测 `a!=a || b!=b`：若 `x` 为 NaN 返回 `x`，否则若 scalar 为 NaN 返回 scalar；都非 NaN 时返回 `a<=b?a:b`。FP16/BF16 先升到 FP32 比较，INT32 与 INT64 scalar 在 INT64 比较，最后转回输出 T。Kernel 只写 `numel(x_i)` 个元素；额外 output 容量不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND、0～8 维 | FP32/FP16/INT32/BF16 | 列表内 dtype 一致；支持空 Tensor；`AutoContiguous()`。950 Tiling 显式要求 Tensor 数 `<=256`。 |
| `scalars` | ACL ScalarList；内部为连续 scalar Tensor | ACL 支持 FP32/DOUBLE/INT64；950 binary 为 FP32 或 INT64 | 个数等于 Tensor 数；浮点/BF16 使用 FP32，INT32 使用 INT64；DOUBLE→FP32 bridge 细节固定仓不可见。 |
| `out` | TensorList；每项 ND、0～8 维 | 与 `x` 相同 | Tensor 数相同，公开接口不支持非连续 output；shape-size 不小于 input，公共 infer 默认完全复制 shape。 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段完成校验并生成执行器。 |
| `workspace/stream` | Device workspace/执行流 | 指针/句柄 | 第二段异步执行；operator Tiling workspace 为 0。 |

### 入口与版本

公开 API 产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 选择 arch35，Kernel CMake 为 ascend950 选择 `arch35/foreach_minimum_scalar_list.cpp`，binary JSON 覆盖四个 dtype 组合。`ACLNNTYPE aclnn` 生成的 bridge 源码不在固定仓，作为框架边界；其后所有算子拥有节点已闭合。固定仓没有其它 ACLNN 版本族。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段按公开错误表检查空指针、dtype、TensorList 内一致性、output 对应关系和最大 8 维。
2. OpDef 约束动态 `x/y`、ScalarList `scalars`、ND 与 `AutoContiguous`；公共 infer 要求动态输入/输出数量相等并复制 shape/dtype。
3. arch35 Tiling 获取 AIV core 数和 UB bytes；任一为 0 则失败。
4. 从动态输入 instance 取得 `tensorNum`，从第一个 input desc 取得统一 dtype；`tensorNum>256` 失败。
5. 对每个 Tensor 写 `tensorElements[i]=storageShape.GetShapeSize()` 并累计 `totalElements`；dtype 不在四类中失败。

### 参数、Workspace 与生命周期

设 `N=sum_i tensorElements[i]`、`C=GetCoreNumAiv()`。`N==0` 时 `needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(C,ceil(N/1024))`。Tiling workspace slice 为 0 bytes；`SetLocalMemorySize(ubSize-128KiB)` 向 launch/runtime 声明本地内存预算，但 SIMT body 没有 shared/dynamic UB 分配或访问。POD 和 input/output/scalar 在 Kernel 执行期间有效；ACLNN caller 拥有总 workspace，executor 可能包含框架 Contiguous 临时空间，其合并布局固定仓不可见。

### 启动图与失败传播

```text
GetWorkspaceSize
  -> 生成 ACLNN bridge / OpDef / infer（生成层为框架边界）
  -> ForeachMinimumScalarListTilingFunc
     -> POD + schMode key + blockDim + localMemorySize + workspace[0]=0
Execute(stream)
  -> foreach_minimum_scalar_list<schMode><<<blockDim, stream>>>
     -> Process<T,S>
        -> VF_CALL<OpForeachMinimumScalarListSimt<T,S>>(Dim3(1024), ...)
```

单节点直接从 GM 读 `x/scalars/tiling` 并写 `y`；无 Kernel 间 edge。Host/Tiling 任一空指针、平台、数量或 dtype 检查失败返回 `GRAPH_FAILED` 并阻断 launch。单 stream 维持生成 executor 内部前处理→Kernel→后处理顺序；生成任务的具体事件/切片固定仓不可见。

## Route 与参数设计

### 分核与资源计算

Host 以总元素数每 1024 元素至少配置一个 AIV block，最多平台 AIV 核数。每个 block 的 `THREAD_NUM=1024`；全局线程号 `tid=blockIdx*threadNum+threadIdx`，全局步长 `stride=threadNum*blockNum`。每个 Tensor 都从相同 `tid` 开始执行 `idx=tid; idx<count; idx+=stride`，因此每个 Tensor 独立进行 grid-stride 覆盖。无 tile、former/tail core 或 UB 容量计算；`idx<count` 是唯一元素尾保护。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | schMode 0；`Process<float,float>` | `dataType==DT_FLOAT` | 下一 dtype 分支 |
| 2 | R-002 FP16→FP32 SIMT | schMode 1；`Process<half,float>` | `dataType==DT_FLOAT16` | 下一 dtype 分支 |
| 3 | R-003 INT32→INT64 SIMT | schMode 2；`Process<int32_t,int64_t>` | `dataType==DT_INT32` | 下一 dtype 分支 |
| 4 | R-004 BF16→FP32 SIMT | schMode 3；`Process<bfloat16_t,float>` | `dataType==DT_BF16` | 其他 dtype `GRAPH_FAILED` |

```text
FP32 -> GET_TPL_TILING_KEY(0) -> compile-time schMode 0
FP16 -> GET_TPL_TILING_KEY(1) -> compile-time schMode 1
INT32 -> GET_TPL_TILING_KEY(2) -> compile-time schMode 2
BF16 -> GET_TPL_TILING_KEY(3) -> compile-time schMode 3
otherwise -> GRAPH_FAILED（不 launch）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | schMode0 | FP32/FP32 scalar；`n<=256`；950 | 单 SIMT Kernel | `Process<float,float>`→`Op...<float,float>` | 直接 GM；无 barrier/atomic/shared | grid-stride `idx<count` |
| R-002 | schMode1 | FP16/FP32 scalar；同上 | 单 SIMT Kernel | `Process<half,float>`，`C=float` | 逐元素 FP16→FP32→FP16 | 同上 |
| R-003 | schMode2 | INT32/INT64 scalar；同上 | 单 SIMT Kernel | `Process<int32_t,int64_t>`，`C=int64_t` | 逐元素 INT64 比较后 cast INT32 | 同上；无 scalar range guard |
| R-004 | schMode3 | BF16/FP32 scalar；同上 | 单 SIMT Kernel | `Process<bfloat16_t,float>`，`C=float` | 逐元素 BF16→FP32→BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachMinimumScalarListTilingFunc` |
| 每核最少域 | `blockDim=min(C,ceil(N/1024))` | 与 `THREAD_NUM=1024` 配合，目标每 block 至少约 1024 总元素 | 全部 | `N==0` 特判 blockDim1 | `MIN_PER_CORE/THREAD_NUM` |
| 线程覆盖 | `idx=count` 上界，stride=`1024*blockNum` | 无 DMA padding，线程直接处理有效元素 | 全部 | 超出线程不执行 | `OpForeachMinimumScalarListSimt` |
| UB 保留 | `SetLocalMemorySize(uint32_t(ubSize-128KiB))` | 为 runtime/compiler 扣除 DCache | 全部 | 无显式 `ubSize>=128KiB` guard；无 fallback | `DCACHE_SIZE/SetLocalMemorySize` |
| 总元素累加 | `int64_t totalElements += numel` | 仅用于 blockDim producer | 全部 | 固定仓无 int64 overflow 检查 | Tiling loop |
| INT32 scalar 范围 | 无 Host 谓词 | body 在 INT64 比较后转 INT32 | R-003 | 无 fallback | `ComputeType<int32_t>/static_cast<T>` |

### POD/ABI

`ForeachMinimumScalarListTilingData` 是普通 standard-layout 结构：`alignof=8`、`sizeof=2064` bytes；offset 由字段类型与声明顺序确定。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMinimumScalarListTilingData` | `needCoreNum` | `int32_t`/32 | offset 0，size 4 | blocks | `N==0?1:min(C,ceil(N/1024))` | 不直接读取；通过 `SetBlockDim` 的 launch grid 生效 | `C>0`；inactive POD copy 仍写值 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4，size 4 | Tensors | `int32_t(tensorNum)` after `<=256` | `Process`→SIMT outer loop | 0～256 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8，size 8 | elements | Tiling 累加全部 storage numel | Device 不读取 | 仅 Host 分核；无 overflow guard |
| 同上 | `tensorElements` | `int64_t[256]` | offset 16，size 2048 | elements/Tensor | Tiling 逐项写，结构先 `memset_s(0)` | `Process` 取 GM offset；SIMT `tensorElements[t]` | 活动前 `tensorCount` 项；空 Tensor 为 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode0 | arch35 `ForeachMinimumScalarListTilingFunc` FP32 branch | arch35 POD | entry constexpr branch | `Process<float,float>`→SIMT body/MinimumWithNaN | ascend950 binary + arch35 CMake |
| R-002 | schMode1 | 同上 FP16 branch | 同上 | entry constexpr branch | `Process<half,float>`→SIMT body | 同上 |
| R-003 | schMode2 | 同上 INT32 branch | 同上 | entry constexpr branch | `Process<int32_t,int64_t>`→SIMT body | 同上 |
| R-004 | schMode3 | 同上 BF16 branch | 同上 | entry constexpr branch | `Process<bfloat16_t,float>`→SIMT body | 同上 |

## Kernel 实现

### 公共执行流程

entry 使用模板参数 `schMode` 做编译期 router。`Process` 不复制 POD：它计算 `offsetof(...,tensorElements)` 得到 tiling GM 内数组地址，读取 `tensorCount`，再以 `Dim3(1024)` 调用 SIMT VF。每个线程对所有 Tensor 循环；空 Tensor `continue`，非空时经 TensorList pointer-table 解码 `x_i/y_i`，读取 `scalars[i]`，用同一全局 tid/stride 遍历元素。线程各写互不重叠的 `y_i[idx]`，无需 atomic/barrier；无跨节点 workspace/synchronization。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/...tiling_arch35.cpp` | `ForeachMinimumScalarListTilingFunc` | `dataType==DT_FLOAT` | template key 0 |
| Router | `op_kernel/arch35/foreach_minimum_scalar_list.cpp` | schMode branch | `schMode==TILING_KEY_FLOAT(0)` | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_minimum_scalar_list<0>
  -> Process<float,float>
     -> VF_CALL<OpForeachMinimumScalarListSimt<float,float>>(Dim3(1024), ...)
        -> SimtGetTensorAddr<float>
        -> MinimumWithNaN<float>
```

##### 源码等价伪码

```text
tid=blockIdx*1024+threadIdx; stride=1024*blockNum
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0 continue
  xData=pointerTable(x,t); yData=pointerTable(y,t); s=scalars_fp32[t]
  for idx=tid; idx<uint64(count); idx+=stride:
    yData[idx] = (isnan(x)||isnan(s)) ? (isnan(x)?x:s) : (x<=s?x:s)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tiling GM | base+16 | 256×INT64 array | POD | Host | Process/SIMT | runtime | 直接 GM pointer |
| x/y GM | pointer-table `[t]`, element `[idx]` | count elements | FP32 flat ND | caller/Kernel | Kernel/caller | caller | 单元素 load/store，无 padding |
| scalar GM | `scalars[t]` | 1 FP32/thread/Tensor | FP32 | bridge | thread compare | input | 每线程每 Tensor 读取一次 |
| registers | thread locals `count/s/tid/stride/a/b` | 固定仓不可见字节数 | scalar | thread | thread | 线程调用期 | 无跨线程共享 |
| dynamic UB/shared/workspace | N/A | 0 body bytes | N/A | N/A | N/A | N/A | `localMemorySize` 未被 body 分配 |

##### 分层内存布局

N/A：该 SIMT Route 没有 AIV `TQue/TBuf`、dynamic UB 或 shared buffer；数据经线程标量寄存器在 GM→compare→GM 间流动。

##### 字节账与复用顺序

```text
UB_total=N/A（无 InitBuffer）
dynamic_ub=N/A（VF_CALL 未传/访问 dynamic buffer）
workspace_total=0 bytes for operator Kernel
Host localMemorySize=uint32(UB_bytes-131072)，但 body allocation=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar locals | GM load | MinimumWithNaN/store | 下一 idx/t | 线程程序顺序 | 不跨线程共享，无 barrier 要求 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程覆盖 | grid-stride | launch grid | loop condition | y element | `idx<count`，无越界 lane |
| compare/store | thread program order | GM loads | GM store | 单元素 | FP32；显式 NaN 传播；无 atomic/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_minimum_scalar_list_tiling_arch35.cpp` | `ForeachMinimumScalarListTilingFunc` | blockDim、POD、key0 |
| Kernel entry/router | `arch35/foreach_minimum_scalar_list.cpp` | schMode0 branch | FP32 specialization |
| Device body/helper | `foreach_minimum_scalar_list_simt.h` | `Process/Op.../MinimumWithNaN` | 1024 threads、grid-stride、NaN 语义 |

### Route R-002：FP16 升 FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35 tiling | `ForeachMinimumScalarListTilingFunc` | `DT_FLOAT16` | key1 |
| Router | arch35 entry | schMode1 branch | `TILING_KEY_FLOAT16` | `Process<half,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry<1> -> Process<half,float> -> VF_CALL<Op...<half,float>>(1024)
  -> SimtGetTensorAddr<half> -> ComputeType<half>::type=float
  -> MinimumWithNaN<float> -> static_cast<half>
```

##### 源码等价伪码

```text
for 每线程命中的 idx:
  a=float(x_half[idx]); b=float(scalar_fp32[t]); r=MinimumWithNaN(a,b); y_half[idx]=half(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | table[t]+idx | count | FP16 | caller/thread | thread/caller | caller | scalar load/store |
| scalar GM | t | 1 | FP32 | bridge | thread | input | 每线程每 Tensor 一次 |
| thread locals | a,b,r | 寄存器字节不可见 | FP32 | cast/compare | store cast | thread | 迭代复用 |
| UB/workspace | N/A | 0 | N/A | N/A | N/A | N/A | 无 buffer |

##### 分层内存布局

N/A：SIMT 线程只使用 GM 与标量寄存器，没有 AIV UB Queue/dynamic UB/shared。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic_ub=N/A；workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 thread locals | FP16/FP32 cast | MinimumWithNaN/FP16 cast | 下一 idx | thread order | 单线程私有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast/compare | C++ `static_cast` | GM load | compare/store | thread local | FP16→FP32 compare→FP16；源码未指定独立 round enum |
| tail | `idx<count` | tid/stride | body | GM extent | 精确元素保护，无 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 tiling | FP16 key branch | schMode1 |
| Kernel entry/router | arch35 entry | `Process<half,float>` | 模板类型 |
| Device body/helper | arch35 SIMT header | `ComputeType<half>/Op...` | FP32 compare与线程 tail |

### Route R-003：INT32 升 INT64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35 tiling | dtype branch | `DT_INT32` | key2 |
| Router | arch35 entry | schMode2 branch | `TILING_KEY_INT32` | `Process<int32_t,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry<2> -> Process<int32_t,int64_t> -> VF_CALL<Op...>(1024)
  -> ComputeType<int32_t>::type=int64_t
  -> MinimumWithNaN<int64_t> -> static_cast<int32_t>
```

##### 源码等价伪码

```text
a=int64(x_int32[idx]); b=scalars_int64[t]
r=(a<=b?a:b)  # integer a!=a/b!=b 恒 false
y_int32[idx]=int32(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | table[t]+idx | count | INT32 | caller/thread | thread/caller | caller | scalar load/store |
| scalar GM | t | 1 | INT64 | bridge | thread | input | 原值参与 INT64 比较 |
| registers | a/b/r | 字节不可见 | INT64 | loads/cast | compare/store | thread | 迭代复用 |
| UB/workspace | N/A | 0 | N/A | N/A | N/A | N/A | 无 buffer |

##### 分层内存布局

N/A：SIMT INT Route 无 UB/shared，仅用线程 INT64 标量中间值。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic_ub=N/A；workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| INT64 thread locals | GM loads/cast | compare/INT32 cast | 下一 idx | thread order | 单线程私有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compare | `MinimumWithNaN<int64_t>` | INT64 locals | cast/store | single element | INT64 精确比较；NaN branch 对整数恒不命中 |
| output cast | `static_cast<int32_t>` | compare | GM y | output element | scalar 小于 INT32_MIN 时 r 可能越界；Host 无 range guard |
| tail | `idx<count` | grid | body | extent | 无越界访问 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 tiling | INT32 key branch | key2、blockDim |
| Kernel entry/router | arch35 entry | `Process<int32_t,int64_t>` | scalar 类型 INT64 |
| Device body/helper | arch35 SIMT header | `ComputeType<int32_t>/MinimumWithNaN` | INT64 比较后降 INT32 |

### Route R-004：BF16 升 FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35 tiling | dtype branch | `DT_BF16` | key3 |
| Router | arch35 entry | schMode3 branch | `TILING_KEY_BF16` | `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry<3> -> Process<bfloat16_t,float> -> VF_CALL<Op...>(1024)
  -> ComputeType<bfloat16_t>::type=float
  -> MinimumWithNaN<float> -> static_cast<bfloat16_t>
```

##### 源码等价伪码

```text
for 每线程命中的 idx:
  a=float(x_bf16[idx]); b=float(scalar_fp32[t]); r=MinimumWithNaN(a,b); y_bf16[idx]=bf16(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | table[t]+idx | count | BF16 | caller/thread | thread/caller | caller | scalar load/store |
| scalar GM | t | 1 | FP32 | bridge | thread | input | 每线程每 Tensor 一次 |
| thread locals | a/b/r | 字节不可见 | FP32 | cast/load | compare/cast | thread | 迭代复用 |
| UB/workspace | N/A | 0 | N/A | N/A | N/A | N/A | 无 buffer |

##### 分层内存布局

N/A：SIMT BF16 Route 不物化 UB scratch，FP32 中间量只在线程局部生存。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic_ub=N/A；workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 thread locals | BF16 cast/scalar load | MinimumWithNaN/BF16 cast | 下一 idx | thread order | 单线程私有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast/compare | C++ `static_cast` | GM load | compare/store | single element | BF16→FP32 compare→BF16；显式 NaN 传播 |
| tail | `idx<count` | grid | body | GM extent | 无 padding/无效 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 tiling | BF16 key branch | key3 |
| Kernel entry/router | arch35 entry | `Process<bfloat16_t,float>` | BF16/FP32 模板 |
| Device body/helper | arch35 SIMT header | `ComputeType<bfloat16_t>/MinimumWithNaN` | FP32 compare、NaN 与 tail |

## 算子特定优化

### Trick T-001：所有 Tensor 复用同一 1024-thread grid-stride launch

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：`tensorCount<=256`，任意非空 Tensor；空 Tensor 被跳过。
- Golden 来源：[`foreach_minimum_scalar_list_simt.h:81-90`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_simt.h#L81-L90)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
    AscendC::Simt::GetThreadIdx());
uint64_t stride = static_cast<uint64_t>(
    AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    yData[idx] = static_cast<T>(MinimumWithNaN(
        static_cast<C>(xData[idx]),
        static_cast<C>(scalarVal)));
}
```

#### 机制与设计原因

Host 只 launch 一次 Kernel；Device 每个线程对每个 Tensor 使用相同全局 tid/stride，从而无需为列表中每个 Tensor 建立独立 launch 或 Tiling 区间。`idx<count` 同时覆盖任意 Tensor 长度和尾元素，不需要 padding DMA 或 tail Kernel。线程只做一读一写和标量比较，不物化 UB tile。

#### 迁移边界

- 适用：TensorList 中各项互相独立、每元素计算常数且允许直接 GM 访问的 foreach 算法。
- 前提：TensorList 指针表 ABI 与 `SimtGetTensorAddr` 一致，所有线程可读取 scalar/POD，输出元素无写冲突。
- 失效：需要 Tensor 内归约、邻域共享或合并 DMA 时必须增加同步/局部存储，不能直接复用此无 barrier 结构。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：ComputeType 在寄存器内升精度而不分配 UB scratch

#### 触发与源码依据

- 适用 Route：R-002、R-003、R-004。
- 触发条件：T 为 half、int32_t 或 bfloat16_t。
- Golden 来源：[`foreach_minimum_scalar_list_simt.h:43-55`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_simt.h#L43-L55)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
struct ComputeType { using type = T; };
template <> struct ComputeType<half> { using type = float; };
template <> struct ComputeType<bfloat16_t> { using type = float; };
template <> struct ComputeType<int32_t> { using type = int64_t; };

template <typename C>
__simt_callee__ inline C MinimumWithNaN(C a, C b)
{
    if (a != a || b != b) {
        return (a != a) ? a : b;
    }
    return (a <= b) ? a : b;
}
```

#### 机制与设计原因

模板在单元素 load 后把 FP16/BF16 提升到 FP32、INT32 提升到 INT64，然后执行显式 NaN/最小值逻辑并立刻降回 T。中间值只在单线程表达式内生存，避免为升精度建立 UB Queue 或二次搬运；同时 FP16/BF16 与 FP32 scalar 在同一计算类型比较。

#### 迁移边界

- 适用：中间状态仅为少数标量、无需跨元素共享的 SIMT 逐元素算法。
- 前提：目标 cast 语义满足 API，INT64 结果在降 INT32 前应处于可表示范围。
- 失效：多步算法需要大量同时存活中间量时寄存器压力/语义可能要求 UB；INT64 scalar 越过 INT32 下界时本实现无范围 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | schMode0 | schMode1 | schMode2 | schMode3 | x dtype 单层决策树 |
| 模板与 dtype | T/C/S=float | T=half,C/S=float | T=int32,C/S=int64 | T=bf16,C/S=float | `ComputeType` 与 binary scalar dtype |
| 循环与分核 | 1024-thread grid-stride | 同左 | 同左 | 同左 | 公共 SIMT body |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread FP32 locals | thread FP32 locals | thread INT64 locals | thread FP32 locals | 无 Queue/shared/dynamic UB |
| 内存总占用与复用 | GM direct | GM direct | GM direct | GM direct | 单元素线程生命周期 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar load/store | 同左 | 同左 | 同左 | 不用 DMA tile |
| 同步、精度与 tail | FP32/NaN | FP32 compare | INT64 compare→INT32 | FP32 compare | 无 barrier；`idx<count` |
| fallback/失败边界 | unsupported dtype Host fail | 同左 | scalar range 无 guard | 同左 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持（公开 Ascend 950PR/DT） |
| Host/Launcher | arch22 公共 Tiling，最多 50 Tensor | 同 A2 | arch35 独立 Tiling，最多 256 Tensor、`ceil(N/1024)` blocks |
| Kernel/资源 | AIV Vector、UB 双 Queue | 同 A2 | SIMT 1024 threads、直接 GM、无 UB Queue |

## 限制与失败边界

- `tensorNum>256`、coreNum/UB 为 0、POD/shape/desc 指针为空或 dtype 不支持时 Tiling 失败，无 Device fallback。
- `totalElements` 的 INT64 累加、`ubSize-128KiB` 的无符号减法无显式 overflow/underflow guard，依赖已注册 950 平台和合法实际 shape。
- INT32 Route 以 INT64 比较后转 INT32；当 scalar 小于 INT32_MIN 且成为最小值时，固定仓未给出范围约束或 fallback，不得假设可移植结果。
- output 额外容量不写；output 非连续不支持。input 连续化任务、DOUBLE scalar 到 FP32 的生成 bridge、TensorList/ScalarList pointer-table producer 与 executor workspace 合并布局固定仓不可见。
- `SetLocalMemorySize` 声明的 runtime 预算不等于 body 分配；当前 body 无 dynamic UB/shared。SIMT intrinsic `VF_CALL` 的更底层调度实现属于 AscendC framework boundary。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_minimum_scalar_list/docs/aclnnForeachMinimumScalarList.md` | 两段式原型/产品表 | 公开 API、950 支持、参数与错误 |
| 入口实现 | `foreach/foreach_minimum_scalar_list/op_host/foreach_minimum_scalar_list_def.cpp` | `ForeachMinimumScalarList` | OpDef、dtype、ScalarList、ascend950 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` | output 数/shape/dtype |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar_list/op_host/arch35/foreach_minimum_scalar_list_tiling_arch35.cpp` | `ForeachMinimumScalarListTilingFunc` | 平台、POD、blockDim、key、workspace |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_tiling_data.h` | `ForeachMinimumScalarListTilingData` | 950 ABI |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 集合 |
| Kernel entry/body | `foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list.cpp` | templated entry | constexpr key router |
| Kernel entry/body | `foreach/foreach_minimum_scalar_list/op_kernel/arch35/foreach_minimum_scalar_list_simt.h` | `Process/Op.../MinimumWithNaN` | 指针表、1024 threads、计算/精度/tail |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_host/config/ascend950/foreach_minimum_scalar_list_binary.json` | `op_list` | 950 四 dtype binary |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_host/CMakeLists.txt` | `SUPPORT_TILING_DIR` | ascend950→arch35，ACLNN |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/...` | 950 Kernel 接线，`AUTO_SYNC false` |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064-byte POD：count + 256 numel | offset 0/4/8/16、Tensor 上限 256 | `ForeachMinimumScalarListTilingData` |
| 调用顺序 | entry→Process→VF_CALL；每线程逐 Tensor→逐 idx | `Dim3(1024)` 与 blockDim/stride 必须一致 | `Process/OpForeachMinimumScalarListSimt` |
| workspace/内存 | operator workspace 0，直接 GM | 不得把 localMemorySize 误当 body UB 分配 | Tiling `workspace/SetLocalMemorySize` |
| 同步 | 无 atomic/barrier；元素独占写 | 仅在每个 `(t,idx)` 唯一映射时成立 | grid-stride loop |
| 精度与尾块 | FP16/BF16→FP32、INT32→INT64；显式 NaN | cast 顺序与 `idx<count` 不得改变 | `ComputeType/MinimumWithNaN` |
| 失败边界 | Host 数量/platform/dtype 失败阻断 launch | 不得为 >256、低 UB、溢出或 INT64 range 编造 fallback | `ForeachMinimumScalarListTilingFunc` |