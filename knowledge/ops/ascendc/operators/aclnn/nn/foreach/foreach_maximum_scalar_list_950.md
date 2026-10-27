---
type: Operator
title: "ForeachMaximumScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachMaximumScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/docs/aclnnForeachMaximumScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/foreach_maximum_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/arch35/foreach_maximum_scalar_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/config/ascend950/foreach_maximum_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:47:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMaximumScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMaximumScalarListGetWorkspaceSize` + `aclnnForeachMaximumScalarList`，公开 ACLNN 两段式接口，无更高版本后缀 |
| 计算语义 | `y_i[j] = max(x_i[j], scalars_i)`；浮点 SIMT helper 显式传播 NaN 并处理相等零，整数 helper 使用 `a>=b?a:b` |
| 输入输出 | ND TensorList；FLOAT16/FLOAT32/INT32/BFLOAT16；最多 256 个 Tensor 的 POD/Host guard；每个 Tensor 0–8 维、可为空 |
| 执行类型 | 单 Kernel：SIMT；AIV launch wrapper 内 `VF_CALL`，全部可达 Device 计算均无 Cube/AIC 或 `MIX_CV` |
| Kernel 路线 | R-001 FP32；R-002 FP16→FP32→FP16；R-003 INT32→INT64→INT32；R-004 BF16→FP32→BF16 |
| 核心资源 | TensorList/Scalar/POD GM 直读、输出 GM 直写；每 block 1024 SIMT threads；workspace 0；无显式 Queue/TBuf/shared/atomic |
| 来源闭包 | `source_closed`：公开 ACLNN 文档经框架生成 executor 边界，闭合到 950 OpDef/config、arch35 Tiling/POD/template key、SIMT entry/body 和 ascend950 构建接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 TensorList 中每个 Tensor 使用同位置 scalar 广播：

```text
for t in [0,tensorCount):
  for idx in [0,tensorElements[t]):
    y[t][idx] = maximum(x[t][idx], scalars[t])
```

浮点 `SimtMaximum(a,b)` 先检查 `isnan(a)`、再检查 `isnan(b)`；非 NaN 时比较大小，相等且 `a==0` 时返回 `a+b`，其余相等值返回 `a`。INT32 路线先扩展到 INT64，使用 `a>=b?a:b`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，每项 ND、0–8 维，可为空 | FLOAT16/FLOAT32/INT32/BFLOAT16，列表内一致 | 非连续输入由 OpDef `AutoContiguous()` 进入框架边界；Host 使用 storage shape numel |
| `scalars` | ScalarList，元素数等于 Tensor 数 | ACLNN 浮点侧支持 FLOAT32/DOUBLE，算子 binary 为 FLOAT32；INT32 对应 INT64 | generated bridge 的 DOUBLE→FLOAT32 细节固定仓不可见；Kernel 直接按模板 `S` 读取连续 scalar GM |
| `out` | TensorList，每项 ND、0–8 维 | 与 `x` 相同 | 输出不支持非连续；infer-shape 默认复制输入 shape，Kernel 只写 `tensorElements[t]` |

### 入口与版本

公开产品表明确 Ascend 950PR/950DT 支持。OpDef 注册 `ascend950`，Host CMake 将其映射到 `arch35`，Kernel CMake只为 ascend950 选择 `arch35/foreach_maximum_scalar_list.cpp`，binary config 给出四个 dtype 实例。固定提交没有带后缀的新 ACLNN 版本。手写 ACLNN inner bridge 不在仓内，严格作为框架生成的 executor/Contiguous 边界；算子自有链路已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段完成空指针、公开 dtype/shape/TensorList 约束检查并建立 executor；`x` 的非连续输入由 `AutoContiguous()` 处理，`out` 必须连续。
2. arch35 `GetPlatformInfo` 读取 AIV core 数 `C` 与 UB 字节 `UB`，任一为 0 或 platform 指针为空即失败。
3. 从动态输入 0 的 instance info 得到 `tensorNum`；超过 256 直接 `GRAPH_FAILED`。从第一个动态输入 desc 得到 dtype。
4. POD 先整体 `memset_s(...,0)`，随后写 `tensorCount`、逐 Tensor `tensorElements[i]=storageShape.GetShapeSize()` 与 `totalElements`。
5. dtype 决策树选择 schMode 0/1/2/3；其它 dtype 报 unsupported 并失败。workspace[0]=0，local memory size 设置为 `UB-128 KiB`。

### 参数、Workspace 与生命周期

设 `E=sum tensorElements[t]`，`C` 为平台 AIV core 数：

```text
E == 0: N = 1
E > 0 : N = min(C, ceil(E/1024))
blockDim = N
threadsPerBlock = 1024
global SIMT threads = N*1024
local_memory_size = UB - 128 KiB
workspace_total = 0
```

POD、TensorList 指针表、scalar Tensor 与输出由 executor/调用者持有至 stream 中 K0 完成。可见 SIMT body 不声明 shared/dynamic UB，也不读取 local-memory 子区；`SetLocalMemorySize` 是 launch 资源设置而不是 body 中可见的缓冲布局。

### 启动图与失败传播

```text
aclnnForeachMaximumScalarListGetWorkspaceSize
  -> framework executor / AutoContiguous boundary
  -> ForeachMaximumScalarListTilingFunc
     [platform/POD/tensorNum/dtype/workspace 指针失败 => 不产生 K0]
aclnnForeachMaximumScalarList(workspace=null,size=0,executor,stream)
  -> K0 foreach_maximum_scalar_list<schMode><<<blockDim=N>>>
       -> Process<T,S>
          -> VF_CALL<OpForeachMaximumScalarListSimt<T,S>>(Dim3(1024), ...)
```

单节点无 workspace/GM 中间边。K0 失败直接结束 executor；没有后继节点或 cleanup Kernel。stream 提供 host launch 与调用者后续工作的顺序，K0 内线程对不同 `(t,idx)` 地址写入，不需要 atomic。

## Route 与参数设计

### 分核与资源计算

Host 的 `N` 只按总元素数以每核至少 1024 元素估算；它不为每个 Tensor 分别切核。Device 每个 block 的 1024 threads 对每个 Tensor 都使用相同全局步长：

```text
tid = blockIdx*1024 + threadIdx
stride = 1024*blockDim
idx = tid, tid+stride, ... < tensorElements[t]
```

因此无 former/tail core POD；每个 Tensor 的尾部由 `idx<count` 循环条件保护。workspace=0。`local_memory_size=UB-128KiB` 的 `uint32_t` 写入没有 `UB>=128KiB` 显式 guard。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | schMode 0 | `dataType == DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16/FP32 SIMT | schMode 1 | `dataType == DT_FLOAT16` | 检查 INT32 |
| 3 | R-003 INT32/INT64 SIMT | schMode 2 | `dataType == DT_INT32` | 检查 BF16 |
| 4 | R-004 BF16/FP32 SIMT | schMode 3 | `dataType == DT_BF16` | 其它 dtype：`GRAPH_FAILED` |

```text
DT_FLOAT   -> GET_TPL_TILING_KEY(schMode=0) -> Process<float,float>
DT_FLOAT16 -> GET_TPL_TILING_KEY(schMode=1) -> Process<half,float>
DT_INT32   -> GET_TPL_TILING_KEY(schMode=2) -> Process<int32_t,int64_t>
DT_BF16    -> GET_TPL_TILING_KEY(schMode=3) -> Process<bfloat16_t,float>
else       -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | FLOAT32、tensorCount<=256、合法 950 | 单 K0 SIMT | `Process<float,float>`→`Op...<float,float>` | GM 直读写，无 UB/atomic/barrier | global-stride `idx<count`; 无 fallback |
| R-002 FP16 | schMode 1 | FLOAT16、同上 | 单 K0 SIMT | `Process<half,float>`；`ComputeType<half>=float` | FP16 load→FP32 register→FP16 store | 同上 |
| R-003 INT32 | schMode 2 | INT32 + INT64 scalar、同上 | 单 K0 SIMT | `Process<int32_t,int64_t>`；`ComputeType<int32_t>=int64_t` | INT32 load→INT64 register→INT32 store | 同上；scalar 超 INT32 值域无 guard |
| R-004 BF16 | schMode 3 | BF16、同上 | 单 K0 SIMT | `Process<bfloat16_t,float>`；compute FP32 | BF16 load→FP32 register→BF16 store | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数 | `tensorNum <= 256` | POD `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachMaximumScalarListTilingFunc` |
| 每核最小工作量 | `N=min(C,ceil(E/1024))` | 匹配每 block 1024 SIMT threads，避免总线程显著超过 E | 全部 | E=0 强制 N=1 | Tiling func；`THREAD_NUM` |
| AIV core/UB 非零 | `coreNum!=0 && ubSize!=0` | 需要 blockDim 与 local-memory launch 参数 | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| local memory 减法 | `SetLocalMemorySize(uint32_t(UB-128KiB))` | 为 DCACHE 预留 128 KiB | 全部 | 源码无 `UB>=128KiB` guard；依赖 950 平台事实 | `DCACHE_SIZE`, Tiling func |
| TensorList ABI | 首个 `uint64_t` 为指针表相对字节 offset，右移 3 转槽位 | SIMT 直接解引用第 t 个 Tensor 地址 | 全部 | 不满足即地址解释错误，无 Device fallback | `SimtGetTensorAddr` |
| 0 元素 Tensor | `count==0` | 避免地址读取和无效写入 | 全部 | `continue` | `OpForeachMaximumScalarListSimt` |

### POD/ABI

`ForeachMaximumScalarListTilingData` 是标准字段顺序的显式 struct；由字段类型可闭合 `alignof=8`、`sizeof=2064`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMaximumScalarListTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8 | core | Tiling: `N` | 可见 body 不读取；blockDim 已携带 N | E=0 时 1 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4 | Tensor 个数 | `int32_t(tensorNum)` | `Process`→SIMT 外层循环 | 0–256 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素 | shapes 求和 | 可见 body 不读取 | Host `int64_t` 累加无显式溢出检查 |
| 同上 | `tensorElements[256]` | `int64_t[256]` | offset 16；2048 B；struct sizeof 2064 | 每 Tensor 元素 | `storageShape.GetShapeSize()` | `elemCounts[t]`/`count` | 仅 `[0,tensorCount)` 读取，空项为 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | arch35 `ForeachMaximumScalarListTilingFunc` | 显式 POD；N/1024 | templated entry mode 0 | `Process<float,float>` + SIMT body/helper | ascend950 binary/Host/Kernel CMake |
| R-002 | schMode 1 | 同上 | 同上 | entry mode 1 | `Process<half,float>` + `ComputeType` | 同上 |
| R-003 | schMode 2 | 同上 | 同上 | entry mode 2 | `Process<int32_t,int64_t>` + int helper | 同上 |
| R-004 | schMode 3 | 同上 | 同上 | entry mode 3 | `Process<bfloat16_t,float>` + `ComputeType` | 同上 |

## Kernel 实现

### 公共执行流程

templated entry 读取 tiling struct，并保留指向 GM POD 的 `tilingGm`。按编译期 schMode 选择唯一 `Process<T,S>`；`Process` 通过 `offsetof(...,tensorElements)` 得到 GM 数组指针，读取 `tensorCount`，以 `Dim3(1024)` 调用 SIMT VF。SIMT 外层遍历所有 Tensor；每个线程用 `(blockIdx,threadIdx)` 形成全局 tid，对当前 Tensor 走 global-stride loop。TensorList 地址由首个相对 offset 和第 t 个 64 位指针槽解析。所有结果直接写 y GM，无 Queue、workspace、atomic 或跨 Kernel 同步。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_maximum_scalar_list_tiling_arch35.cpp` | `ForeachMaximumScalarListTilingFunc` | `dataType==DT_FLOAT` | schMode 0 |
| Router | `op_kernel/arch35/foreach_maximum_scalar_list.cpp` | `schMode==TILING_KEY_FLOAT` | mode 0 | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
K0 foreach_maximum_scalar_list<0>
  -> Process<float,float>
    -> VF_CALL<OpForeachMaximumScalarListSimt<float,float>>(1024 threads)
      -> for each t: SimtGetTensorAddr<float>(x/y,t)
      -> global-stride load -> SimtMaximum(float,float) -> store
```

##### 源码等价伪码

```text
for t in [0,tensorCount):
  count = tensorElements[t]; if count==0 continue
  xData/yData = tensorListAddr(t); scalar = scalars_fp32[t]
  tid = blockIdx*1024 + threadIdx; stride = blockDim*1024
  for idx=tid; idx<count; idx+=stride:
    yData[idx] = SimtMaximum(float(xData[idx]), float(scalar))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| POD GM | offset 4/16 | tensorCount + 256 counts | struct | Host Tiling | Process/SIMT | executor/K0 | 8 B struct alignment |
| x/y GM | `tensor_ptr(t)+idx` | count FP32 | flattened ND | caller/thread store | thread/caller | caller | unique idx，无冲突 |
| scalar GM | `scalars+t` | 1 FP32 | contiguous | framework | each active thread | executor/K0 | 同 Tensor 多线程只读 |
| SIMT registers | per thread | xVal/sVal/result | FP32 | load/compute | store | 当前 loop iteration | 固定仓不提供寄存器字节数 |

##### 分层内存布局

N/A：该 SIMT body 不分配 AIV Queue/TBuf/shared/dynamic UB；实际路径为 GM→每线程寄存器→GM。Host 仍设置 `local_memory_size=UB-128KiB`，可见 body 未访问它。

##### 字节账与复用顺序

```text
UB_total = 0 B（可见 body）
dynamic_ub explicit VF argument = N/A
local_memory_size launch setting = UB - 128 KiB（body 无 consumer）
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-thread registers | GM load/SimtMaximum | SimtMaximum/store | 下一 idx | 同一线程程序顺序 | store 后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread mapping | global-stride loop | Host blockDim/1024 | loop condition | y[t][idx] 唯一性 | `idx<count` 保护尾部 |
| maximum | branch helper | FP32 loads | FP32 store | thread registers | NaN 按 x 后 scalar 顺序传播；相等零返回 `a+b` |
| synchronization | N/A | 独立线程 | N/A | 无共享写地址 | 不需要 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_maximum_scalar_list/op_host/arch35/foreach_maximum_scalar_list_tiling_arch35.cpp` | Tiling func | N、POD、workspace/local memory、mode 0 |
| Kernel entry/router | `foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list.cpp` | entry `<0>` | 路由 float/float |
| Device body/helper | `foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_simt.h` | `Process`, `Op...<float,float>`, `SimtMaximum(float,float)` | VF 线程、地址、精度/tail |

### Route R-002：FP16 经 FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35 Tiling file | `ForeachMaximumScalarListTilingFunc` | `DT_FLOAT16` | schMode 1 |
| Router | arch35 entry | mode 1 branch | `<half,float>` | R-002 |

#### Device 数据流

##### 函数级调用链

```text
K0<1> -> Process<half,float> -> VF_CALL<Op...<half,float>>
  -> half x load -> static_cast<float>
  -> FP32 scalar load -> SimtMaximum(float,float)
  -> static_cast<half> -> y GM
```

##### 源码等价伪码

```text
使用与 R-001 完全相同的 t/tid/stride/idx；每个有效 idx 在寄存器中执行
float xVal=half(x[idx]); float sVal=scalars[t]; float r=SimtMaximum(xVal,sVal); y[idx]=half(r)。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensor_ptr(t)+idx` | count | FP16 flattened ND | caller/thread | thread/caller | K0 | idx<count |
| scalar GM | `scalars+t` | 1 | FP32 | framework | thread | K0 | 不先转 FP16 |
| registers | per thread | FP32 x/s/result | FP32 | cast/load | maximum/store cast | iteration | 无共享资源 |

##### 分层内存布局

N/A：GM→FP32 thread registers→GM，无 Queue/TBuf/shared/dynamic UB；Host local-memory setting 无可见 consumer。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; local_memory_size=UB-128KiB; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | load/cast | maximum/store cast | next idx | thread program order | 当前 FP16 store 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | `ComputeType<half>::type=float` | FP16/FP32 loads | FP16 store | registers | FP32 compare，最终 static_cast 到 FP16；源码未命名 round mode |
| tail/sync | `idx<count`; no barrier | thread mapping | store | unique y idx | 空 Tensor `continue` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling file | dtype decision | FP16→mode 1 |
| Kernel entry/router | arch35 entry | `Process<half,float>` | 模板类型 |
| Device body/helper | arch35 SIMT header | `ComputeType<half>`, `Op...` | FP32 compute 与 global-stride tail |

### Route R-003：INT32 经 INT64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35 Tiling file | dtype decision | `DT_INT32` | schMode 2 |
| Router | arch35 entry | mode 2 branch | `<int32_t,int64_t>` | R-003 |

#### Device 数据流

##### 函数级调用链

```text
K0<2> -> Process<int32_t,int64_t> -> VF_CALL
  -> INT32 x load -> INT64 xVal
  -> INT64 scalar load -> SimtMaximum(int64_t,int64_t)
  -> static_cast<int32_t> -> y GM
```

##### 源码等价伪码

```text
对每个有效 idx：int64 xVal=x_int32[idx]; int64 sVal=scalars_int64[t];
int64 result=(xVal>=sVal)?xVal:sVal; y_int32[idx]=static_cast<int32_t>(result)。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensor_ptr(t)+idx` | count | INT32 flattened ND | caller/thread | thread/caller | K0 | unique idx |
| scalar GM | `scalars+t` | 1 | INT64 | framework | thread | K0 | 直接 INT64 load |
| registers | per thread | INT64 x/s/result | INT64 | extend/load | compare/narrow store | iteration | 无共享资源 |

##### 分层内存布局

N/A：GM→INT64 thread registers→GM；无 Queue/TBuf/shared/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; local_memory_size=UB-128KiB; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | extend/load | compare/narrow store | next idx | thread program order | store 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | INT64 compare | INT32 x + INT64 scalar | INT32 store | registers | scalar 超 INT32 时 result 可能超输出值域，源码无 guard |
| tail/sync | `idx<count`; no atomic | thread mapping | unique store | y idx | 空 Tensor continue |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling file | dtype decision | INT32→mode 2 |
| Kernel entry/router | arch35 entry | `Process<int32_t,int64_t>` | 模板类型 |
| Device body/helper | arch35 SIMT header | `ComputeType<int32_t>`, integer `SimtMaximum` | INT64 compare/INT32 store |

### Route R-004：BF16 经 FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch35 Tiling file | dtype decision | `DT_BF16` | schMode 3 |
| Router | arch35 entry | mode 3 branch | `<bfloat16_t,float>` | R-004 |

#### Device 数据流

##### 函数级调用链

```text
K0<3> -> Process<bfloat16_t,float> -> VF_CALL
  -> BF16 x load -> FP32 register
  -> FP32 scalar -> SimtMaximum(float,float)
  -> static_cast<bfloat16_t> -> y GM
```

##### 源码等价伪码

```text
同 R-002 的 global-stride 和 FP32 maximum，只把 GM 输入/输出类型换为 bfloat16_t。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensor_ptr(t)+idx` | count | BF16 flattened ND | caller/thread | thread/caller | K0 | idx<count |
| scalar GM | `scalars+t` | 1 | FP32 | framework | thread | K0 | 保持 FP32 |
| registers | per thread | FP32 x/s/result | FP32 | cast/load | maximum/cast store | iteration | 无共享资源 |

##### 分层内存布局

N/A：GM→FP32 thread registers→GM；无 Queue/TBuf/shared/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; local_memory_size=UB-128KiB; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | load/cast | maximum/store cast | next idx | thread order | 当前 store 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | `ComputeType<bfloat16_t>=float` | BF16/FP32 loads | BF16 store | registers | FP32 maximum；最终 static_cast，源码未命名 round mode |
| tail/sync | `idx<count`; no barrier | thread mapping | store | unique y idx | 空 Tensor continue |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling file | dtype decision | BF16→mode 3 |
| Kernel entry/router | arch35 entry | `Process<bfloat16_t,float>` | 模板类型 |
| Device body/helper | arch35 SIMT header | `ComputeType<bfloat16_t>`, float `SimtMaximum` | FP32 compute 与 BF16 store |

## 算子特定优化

### Trick T-001：跨全部 block 的 global-stride 线程映射复用于每个 Tensor

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003、R-004
- 触发条件：950 arch35 SIMT body；Host 设置 `blockDim=N`，VF 固定 `THREAD_NUM=1024`
- Golden 来源：[`foreach_maximum_scalar_list_simt.h:96-107`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_simt.h#L96-L107)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
    AscendC::Simt::GetThreadIdx());
uint64_t stride = static_cast<uint64_t>(
    AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    C xVal = static_cast<C>(xData[idx]);
    C sVal = static_cast<C>(scalarVal);
    C result = SimtMaximum(xVal, sVal);
    yData[idx] = static_cast<T>(result);
}
```

#### 机制与设计原因

Host 只根据总元素数决定 N，Device 不生成每 Tensor 的分核表。每个 Tensor 复用 `N*1024` 个全局线程，线程 `tid` 以总线程数为 stride 覆盖互不重叠的 idx；这样只需要 `tensorElements[]`，无需 per-core start/end POD、UB 搬运和原子写。`idx<count` 同时承担任意 Tensor 长度的 tail 保护。

#### 迁移边界

- 适用：TensorList 各项独立、每个输出元素只依赖同位置输入和单个广播 scalar 的逐元素算子。
- 前提：TensorList GM 指针表 ABI、每 block 1024 VF threads、输出无跨线程归约/写冲突。
- 失效：需要邻域、归约、跨 Tensor 状态或共享中间量时，不能继续用无 barrier/atomic 的直接映射。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时若把 stride 错写为仅 `threadNum`，多个 block 会覆盖相同输出地址，破坏当前无 atomic 的安全前提。

### Trick T-002：显式固定浮点 NaN 与相等零语义

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-004
- 触发条件：`ComputeType<T>::type=float`
- Golden 来源：[`foreach_maximum_scalar_list_simt.h:48-65`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_simt.h#L48-L65)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isnan(a)) {
    return a;
}
if (isnan(b)) {
    return b;
}
if (a > b) {
    return a;
}
if (b > a) {
    return b;
}
if (a == 0.0f) {
    return a + b;
}
return a;
```

#### 机制与设计原因

helper 避免直接依赖通用 `max` 对 NaN 和 signed-zero 的实现差异：输入 NaN 优先返回输入，scalar NaN 次之；相等零通过加法合并符号，其余相等值保留输入。FP16/BF16 路线在 FP32 中使用同一 helper，再转换回输出 dtype，从源码层统一三条浮点 Route 的分支顺序。

#### 迁移边界

- 适用：要求与该算子相同 NaN 优先级和相等零处理的浮点 maximum。
- 前提：比较发生在 FP32；降精度输出必须接受模板的 `static_cast<T>` 语义。
- 失效：若目标 API 规定忽略 NaN、不同 signed-zero 规则或其它 tie-break，必须替换 helper。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能在复用时用未证明等价的通用 `max` 替换这段有序分支。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | mode 0 | mode 1 | mode 2 | mode 3 | dtype 决策树 |
| 模板与 dtype | float/float/C=float | half/float/C=float | int32/int64/C=int64 | bf16/float/C=float | `ComputeType` 与 scalar ABI |
| 循环与分核 | N×1024 global stride | 同左 | 同左 | 同左 | 与 dtype 无关的调度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread FP32 regs | thread FP32 regs | thread INT64 regs | thread FP32 regs | 无 Queue/TBuf/shared/dynamic UB |
| 内存总占用与复用 | 可见 UB 0 | 0 | 0 | 0 | GM 直读写 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT load/store | SIMT + cast | SIMT + extend/narrow | SIMT + cast | 全部单 SIMT 节点 |
| 同步、精度与 tail | explicit float max | FP32 compare | INT64 compare | FP32 compare | idx<count；无 barrier/atomic |
| fallback/失败边界 | 无 fallback | 无 | scalar 值域风险 | 无 | unsupported dtype Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持 |
| Host/Launcher | arch22 公共 Tiling、最多 50 Tensor、workspace 32 B | 同 A2 | arch35 自有 Tiling、最多 256 Tensor、workspace 0、每核阈值 1024 |
| Kernel/资源 | AIV Vector Queue；FP16 原生 half Maxs，BF16 FP32 | 同 A2 | SIMT GM 直读写；FP16/BF16 FP32，INT32 INT64；显式 NaN/zero helper |

## 限制与失败边界

- `tensorNum>256`、platform AIV core/UB 为 0、关键 context/POD/workspace 指针为空或 dtype 不在四种支持集合时，arch35 Tiling 返回 `GRAPH_FAILED`；没有 Device fallback。
- `totalElements` 为 int64 累加，固定仓没有 shape numel 求和溢出检查。`UB-128KiB` 也没有下界检查，依赖 ascend950 平台 UB 配置。
- ACLNN 浮点 ScalarList 可为 DOUBLE，但 fixed repo 不含 generated bridge 的 DOUBLE→FLOAT32 细节；binary 与 Kernel 的 scalar 类型为 FLOAT32。
- INT32 Route 在 INT64 中取 maximum 后转回 INT32；scalar 超 INT32 范围时结果转换行为没有接口级闭合，调用者不得依赖。
- FP16/BF16 输出使用 C++ `static_cast<T>`，固定仓没有命名具体 round mode；只可复用“FP32 比较后转换”这一事实。
- `needCoreNum`、`totalElements` 写入 POD 但可见 Device body 不读取；不得假设它们影响 SIMT 地址公式。local-memory launch 设置也无可见 body consumer。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_maximum_scalar_list/docs/aclnnForeachMaximumScalarList.md` | ACLNN 原型、产品表 | API/平台/dtype/shape/错误语义 |
| 入口实现 | `foreach/foreach_maximum_scalar_list/op_host/foreach_maximum_scalar_list_def.cpp` | `ForeachMaximumScalarList` | ascend950 OpDef、AutoContiguous |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`, `InferDataType4ForeachCommon` | 动态列表输出 shape/dtype 推导 |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar_list/op_host/arch35/foreach_maximum_scalar_list_tiling_arch35.cpp` | `GetPlatformInfo`, Tiling func | N/POD/workspace/local memory/mode producer |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_tiling_data.h` | explicit POD struct | ABI 字段 |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_tiling_key.h` | template schMode declaration | 四个 mode 与模板选择 |
| Kernel entry/body | `foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list.cpp` | templated entry | mode→`Process<T,S>` |
| Kernel entry/body | `foreach/foreach_maximum_scalar_list/op_kernel/arch35/foreach_maximum_scalar_list_simt.h` | `Process`, `Op...`, helpers | VF launch、线程映射、地址/精度/tail |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_host/config/ascend950/foreach_maximum_scalar_list_binary.json` | 四个 binary 条目 | 950 dtype 实例 |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_host/CMakeLists.txt` | ascend950→arch35 | Host Tiling 选择 |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_kernel/CMakeLists.txt` | ascend950 kernel source | 950 entry 选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B POD、256 counts、TensorList pointer table | offsets 0/4/8/16、首字相对 offset 右移 3 | POD struct；`SimtGetTensorAddr` |
| 调用顺序 | entry→Process→VF_CALL→Tensor loop→thread loop | mode 与 `<T,S>` 对应，VF threads=1024 | entry；`Process`; `Op...` |
| workspace/内存 | workspace 0，GM direct，visible UB 0 | local-memory setting `UB-128KiB` 不等于可随意复用的 body buffer | Tiling func；SIMT body |
| 同步 | 无 barrier/atomic，靠唯一 idx 和线程程序顺序 | stride 必须是 `threadNum*blockNum` | SIMT loop |
| 精度与尾块 | float helper；FP16/BF16 FP32；INT32 INT64 | NaN/zero 分支序、output cast、`idx<count` | `ComputeType`, `SimtMaximum`, `Op...` |
| 失败边界 | Host 对 tensorNum/dtype/platform 失败 | 不虚构 fallback；不扩展 256 Tensor | Tiling func |