---
type: Operator
title: "ForeachCosh 950 ACLNN 算子知识"
description: "说明 aclnnForeachCosh 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 SIMT 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/docs/aclnnForeachCosh.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/foreach_cosh_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/arch35/foreach_cosh.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/config/ascend950/foreach_cosh_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:31:16Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachCosh 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachCoshGetWorkspaceSize` → `aclnnForeachCosh`；固定仓未保存自动生成 inner bridge。 |
| 计算语义 | 对 TensorList 中全部元素执行 `y[t][i]=cosh(x[t][i])`。 |
| 输入输出 | ND TensorList；FP16/FP32/BF16；输入可非连续且由框架连续化，输出连续。 |
| 执行类型 | 单 Kernel、单 SIMT VF 节点；每 block 512 threads，线程直接从 GM 读写。 |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16→FP32 SIMT；R-003 BF16→FP32 SIMT。 |
| 核心资源 | TensorList GM 指针表、POD 内每 tensor 元素数；无显式 AIV UB/TQue/TBuf、无 dynamic shared UB、无用户 workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN → 生成边界 → 950 OpDef/arch35 Tiling/POD/template key → templated `foreach_cosh` → `VF_CALL<OpForeachCoshSimt<T>>` → ascend950 config/build。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设列表长度 `L=tensorCount`，第 `t` 项元素数 `N_t=tensorElements[t]`：

```text
for t in [0,L):
  for i in [0,N_t): out[t][i] = coshf(float(x[t][i])) converted to T
```

FP32 路线的转换函数是恒等；FP16/BF16 显式转 FP32 后调用 `coshf`，再转回输入 dtype。空 Tensor 被跳过。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；ND；rank 0–8 | FLOAT32/FLOAT16/BFLOAT16，列表内一致 | 支持空 Tensor；950 Tiling 强制 `tensorCount<=256`；非连续由 `AutoContiguous` 框架边界处理。 |
| `out` | TensorList；ND；容量≥对应输入 | 与 `x` 相同且列表内一致 | 输出必须连续；SIMT 只写每项前 `N_t` 个元素。 |
| `workspaceSize/executor` | 两段式 Host 输出 | `uint64_t*` / `aclOpExecutor**` | executor 内部布局固定仓不可见；arch35 Tiling 将算子 workspace 设为0。 |
| `workspace/stream` | Device地址/stream | `void*` / `aclrtStream` | Kernel 参数包含 workspace 但 body 不读取；单节点在调用 stream 上执行。 |

### 入口与版本

固定仓只有未编号的公开 `aclnnForeachCosh` 版本族。README 产品表明确支持 Ascend 950PR/950DT，ACLNN 文档同样列 950；OpDef 注册 `ascend950`，Host CMake 将 950 指向 arch35，Kernel CMake 将 `arch35/foreach_cosh.cpp` 仅接到 ascend950，binary config 同时列 FP16/FP32/BF16，证据无冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 ACLNN 第一段检查指针、dtype、shape/rank、列表内 dtype 与输入输出一致性；其自动生成实现不在固定仓。
2. OpDef 注册动态 `x/y`、ND、FP16/FP32/BF16、`AutoContiguous`；公开约束要求输出本身连续。
3. `ForeachCoshTilingFunc` 取得 AIV core 数和 UB bytes；任一为0直接 `GRAPH_FAILED`。
4. 从 input instance 0 取得 `tensorNum`，从首个 dynamic input desc 取得 dtype；`tensorNum>256` 失败。
5. POD 先全0，再逐项写 storage `numel` 和 `totalElements`；dtype 映射到 `schMode=0/1/2` 的模板 key，其他 dtype失败。

arch35 Tiling 本身没有读取输出 desc/shape；公开错误语义依赖 ACLNN/OpDef通用层，固定仓不可见的生成边界不能被误写成该函数的显式检查。

### 参数、Workspace 与生命周期

```text
E = Σ tensorElements[t]
needCoreNum = E==0 ? 1 : min(aivCoreNum, ceil(E/1024))
blockDim = needCoreNum
localMemorySize = UB_platform - 128 KiB
workspace[0] = 0
threadsPerBlock = 512（VF_CALL内部常量）
totalLogicalThreads = blockDim * 512
```

`needCoreNum` 与 `totalElements` 写入 POD，但当前 Device body不读取；实际线程映射依赖 launch 的 `blockDim` 与 `VF_CALL` 512。POD、TensorList地址表与 executor 在 Kernel 完成前保持有效；无中间 workspace owner。

### 启动图与失败传播

```text
aclnnForeachCoshGetWorkspaceSize [生成边界]
  -> ForeachCoshTilingFunc -> template key/POD/blockDim/localMemorySize
  -> executor
aclnnForeachCosh(...,stream)
  -> K0 foreach_cosh<schMode><<<blockDim,stream>>>(x,y,unused_workspace,tiling)
     -> NsForeachCosh::Process<T>
     -> VF_CALL<OpForeachCoshSimt<T>>(Dim3(512), ...)
```

| 节点/边 | producer | consumer | 介质与范围 | ordering / owner / failure |
|---|---|---|---|---|
| K0 | executor | templated `foreach_cosh` | x/y TensorList GM、2064 B POD、blockDim | 调用stream；Tiling/launch错误直接返回。 |
| 跨节点边 | N/A | N/A | 单Kernel，无workspace edge | N/A。 |

## Route 与参数设计

### 分核与资源计算

Host 只用总元素数选择 block 数，每个 block 固定 512 threads。SIMT 内对每个 Tensor 都重新计算：

```text
tid = blockIdx*threadNum + threadIdx
stride = threadNum*blockNum
for idx=tid; idx<N_t; idx+=stride: compute one element
```

因此同一全局线程 id 对列表中的每个非空 Tensor 各负责一条 grid-stride 序列；没有按 tensor 重新分配 block，也没有跨 tensor prefix-sum。

`SetLocalMemorySize(UB-128KiB)` 是 launch资源上限；当前 SIMT body 没有显式 dynamic UB/shared数组，也未传 dynamic size 参数。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `schMode=0` / `foreach_cosh<0>` | `dataType==DT_FLOAT` | dtype判断下一分支。 |
| 2 | R-002 FP16→FP32 SIMT | `schMode=1` / `foreach_cosh<1>` | `dataType==DT_FLOAT16` | 下一分支。 |
| 3 | R-003 BF16→FP32 SIMT | `schMode=2` / `foreach_cosh<2>` | `dataType==DT_BF16` | 其它dtype `GRAPH_FAILED`。 |

```text
Host dtype: FLOAT ? GET_TPL_KEY(schMode=0)
          : FLOAT16 ? GET_TPL_KEY(schMode=1)
          : BF16 ? GET_TPL_KEY(schMode=2)
          : GRAPH_FAILED
Compile-time router: schMode 0->float, 1->half, 2->bfloat16_t
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode0 | FP32；tensorCount≤256；rank≤8；950 | K0单SIMT VF，512 threads/block | `Process<float>`→`OpForeachCoshSimt<float>` | GM直读写；无显式UB/barrier/atomic | `idx<N_t` grid-stride保护；空Tensor continue。 |
| R-002 FP16 | schMode1 | FP16；其余同上 | 同 | `Process<half>`→SIMT half specialization | 单元素 half→float→half | 同。 |
| R-003 BF16 | schMode2 | BF16；其余同上 | 同 | `Process<bfloat16_t>`→SIMT BF16 specialization | 单元素 BF16→float→BF16 | 同。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor上限 | `tensorNum<=256` | POD数组固定256 | 全部 | `GRAPH_FAILED` | `ForeachCoshTilingFunc` |
| 最少每core 1024 | `blockDim=min(coreNum,ceil(E/1024))` | 限制小元素域使用的block数 | 全部 | E=0强制1；无其它fallback | 同上 |
| 512 threads | `LAUNCH_BOUND(512)` 与 `Dim3(512)` | 固定SIMT block size | 全部 | 无可选模板 | `OpForeachCoshSimt`; `Process` |
| local memory reserve | `SetLocalMemorySize(UB-128KiB)` | 为DCache保留128KiB | 全部 | UB小于128KiB时无显式下溢guard | `ForeachCoshTilingFunc` |
| direct GM | body不分配shared/dynamic UB | 每线程独立元素，无协作数据 | 全部 | N/A | `OpForeachCoshSimt` |
| 空Tensor | `count==0` | 避免解析/读写tensor数据 | 全部 | `continue` | 同上 |

### POD/ABI

`ForeachCoshTilingData` 是无 pack 的标准字段序列；按固定字段类型/顺序，ABI 为 `alignof=8`、`sizeof=2064`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCoshTilingData` | `needCoreNum` | int32/32 | offset0；sizeof4 | blocks | E=0写1，否则min(core,ceil(E/1024)) | 当前body不读；inactive字段 | int64→int32前由platform core限制。 |
| 同上 | `tensorCount` | int32/32 | offset4；sizeof4 | tensors | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT outer loop | 0..256。 |
| 同上 | `totalElements` | int64/64 | offset8；sizeof8 | elements | 逐tensor求和 | 当前body不读；inactive字段 | Host加法无显式overflow检查。 |
| 同上 | `tensorElements[256]` | int64[256] | offset16；sizeof2048 | elements/tensor | 每项storage `GetShapeSize()`；其余memset为0 | `elemCounts[t]` | 只读前tensorCount项。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode0 | `ForeachCoshTilingFunc` | `ForeachCoshTilingData`; blockDim; localMemorySize | `foreach_cosh<0>` | `Process<float>`; `OpForeachCoshSimt<float>`; `coshf` | ascend950 binary config + arch35 Kernel CMake |
| R-002 | schMode1 | 同上 | 同上 | `foreach_cosh<1>` | half `ToFloat32/FromFloat32`; SIMT body | 同上 |
| R-003 | schMode2 | 同上 | 同上 | `foreach_cosh<2>` | BF16 `ToFloat32/FromFloat32`; SIMT body | 同上 |

## Kernel 实现

### 公共执行流程

Entry 按编译期 `schMode` 选择 T。`Process<T>` 用 `offsetof(ForeachCoshTilingData,tensorElements)` 在 GM POD 中定位元素数数组，读 `tensorCount`，以 `VF_CALL` 启动512线程。每个线程顺序遍历所有 tensor；对每个非空项解析 TensorList 指针表，执行 grid-stride循环并直接写输出 GM。没有 Queue、barrier、atomic、workspace 或中间节点。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp` | `ForeachCoshTilingFunc` | `dataType==DT_FLOAT` | schMode0 key |
| Router | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh.cpp` | `foreach_cosh<schMode>` | `if constexpr(schMode==0)` | `Process<float>` |
| Device | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h` | `OpForeachCoshSimt<float>` | count>0, idx<count | direct GM coshf |

#### Device 数据流

##### 函数级调用链

```text
foreach_cosh<0> -> NsForeachCosh::Process<float>
  -> VF_CALL<OpForeachCoshSimt<float>>(Dim3(512),tensorCount,elemCounts,x,y)
     -> per tensor SimtGetTensorAddr<float>
     -> per-thread grid-stride load -> coshf -> store
```

##### 源码等价伪码

```text
tid=blockIdx*512+threadIdx; stride=512*blockNum
for t=0..tensorCount-1:
  if elemCounts[t]==0: continue
  xT=TensorListAddr<float>(x,t); yT=TensorListAddr<float>(y,t)
  for idx=tid; idx<elemCounts[t]; idx+=stride:
    yT[idx]=coshf(xT[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x tensor pointer table | `base + (*base>>3) + t` in uint64 slots | tensorCount addresses | uint64 addresses | framework | `SimtGetTensorAddr` | caller/framework | shift3 assumes8B entries |
| x GM | `xData[idx]` | N_t float | flat ND storage | caller | current thread | caller | scalar direct load |
| register scalar | thread-local `xVal/xF32/yF32` | one element | float | load/coshf | coshf/store | thread lifetime | no shared alias |
| y GM | `yData[idx]` | N_t float prefix | flat ND storage | current thread | caller | caller | unique idx per global thread sequence |

##### 分层内存布局

N/A：本 SIMT body 未分配 AIV `TQue/TBuf`、shared/dynamic UB 或通信 window；替代路径是 thread-local scalar/register 加直接 GM load/store。寄存器物理字节数由编译器决定，固定仓不可见。

##### 字节账与复用顺序

```text
UB_explicit = 0 B
dynamic_ub = 0 B（VF_CALL无dynamic-size参数；body无shared数组）
localMemorySize launch上限 = UB_platform - 131072 B
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local scalars | GM load/coshf | coshf/GM store | 下一idx循环 | 单线程程序顺序 | 每线程独占；无需barrier。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Thread mapping | grid-stride | launch dimensions | loop | output index ownership | `idx<N_t`处理任意尾。 |
| Compute | `coshf` | x float load | y store | scalar | FP32。 |
| Inter-thread | N/A | each thread | N/A | disjoint idx | 无barrier/atomic。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp` | `ForeachCoshTilingFunc` | blockDim/POD/schMode0。 |
| Kernel entry/router | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh.cpp` | `foreach_cosh<0>` | float映射。 |
| Device body/helper | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h` | `Process<float>`, `OpForeachCoshSimt<float>` | 512线程、地址表、grid-stride、coshf。 |

### Route R-002：FP16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | arch35 Tiling | `ForeachCoshTilingFunc` | `DT_FLOAT16` | schMode1 |
| Router | arch35 entry | `foreach_cosh<1>` | constexpr mode1 | `Process<half>` |
| Device | arch35 SIMT | `ToFloat32<half>/FromFloat32<half>` | `is_same_v<T,half>` | `__half2float` / `__float2half` |

#### Device 数据流

##### 函数级调用链

```text
foreach_cosh<1> -> Process<half> -> VF_CALL<OpForeachCoshSimt<half>>(512)
  -> load half -> __half2float -> coshf -> __float2half -> store half
```

##### 源码等价伪码

```text
for each t and idx=globalTid+k*globalThreads, idx<N_t:
  xf=__half2float(xData[idx])
  yf=coshf(xf)
  yData[idx]=__float2half(yf)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | idx grid-stride | N_t half | half flat ND | caller/thread | thread/caller | caller | scalar direct access |
| thread scalars | one half + two float logical values | compiler registers | half/float | load/conversion/cosh | next op/store | thread | physical registers fixed仓不可见 |

##### 分层内存布局

N/A：无显式 UB/shared buffer；直接 GM 与 thread-local scalar路径。

##### 字节账与复用顺序

```text
UB_explicit=0 B; dynamic_ub=0 B; workspace_total=0 B
localMemorySize=UB_platform-131072 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | conversion/coshf | next scalar op | next idx | program order | no cross-thread sharing |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Convert in | `__half2float` | half load | coshf | scalar | FP32升精度。 |
| Compute/out | coshf + `__float2half` | float | half store | scalar | 转换round细节由SIMT intrinsic定义，源码未传round mode。 |
| Tail | idx bound | global thread id | loop | GM index | 任意N_t，无padding写。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp` | dtype branch | schMode1。 |
| Kernel entry/router | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh.cpp` | mode1 branch | half type。 |
| Device body/helper | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h` | half conversions/body | FP32 coshf与grid-stride。 |

### Route R-003：BF16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | arch35 Tiling | `ForeachCoshTilingFunc` | `DT_BF16` | schMode2 |
| Router | arch35 entry | `foreach_cosh<2>` | constexpr mode2 | `Process<bfloat16_t>` |
| Device | arch35 SIMT | BF16 conversion branches | `is_same_v<T,bfloat16_t>` | static_cast float/BF16 |

#### Device 数据流

##### 函数级调用链

```text
foreach_cosh<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachCoshSimt<bfloat16_t>>(512)
  -> load BF16 -> static_cast<float> -> coshf -> static_cast<BF16> -> store
```

##### 源码等价伪码

```text
for each nonempty tensor and owned idx:
  xf=float(xData[idx])
  yf=coshf(xf)
  yData[idx]=bfloat16_t(yf)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | idx grid-stride | N_t BF16 | BF16 flat ND | caller/thread | thread/caller | caller | scalar direct access |
| thread scalars | one BF16 + float intermediates | compiler registers | BF16/float | load/casts/coshf | store | thread | register bytes fixed仓不可见 |

##### 分层内存布局

N/A：无显式 UB/shared buffer；直接 GM 与 thread-local scalar路径。

##### 字节账与复用顺序

```text
UB_explicit=0 B; dynamic_ub=0 B; workspace_total=0 B
localMemorySize=UB_platform-131072 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | static_cast/coshf | next scalar op | next idx | program order | per-thread private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Convert in | `static_cast<float>` | BF16 load | coshf | scalar | FP32升精度。 |
| Convert out | `static_cast<bfloat16_t>` | coshf float | GM store | scalar | round mode未在源码显式选择。 |
| Tail | idx bound | thread mapping | loop | GM | 任意N_t。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp` | BF16 branch | schMode2。 |
| Kernel entry/router | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh.cpp` | mode2 branch | BF16 type。 |
| Device body/helper | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h` | BF16 conversions/body | FP32 coshf与direct GM。 |

## 算子特定优化

### Trick T-001：全局线程对每个 Tensor 复用同一 grid-stride 映射

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：任意 `tensorCount<=256`；每个非空 Tensor。
- Golden 来源：[`foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h:89-109`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h#L89-L109)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    T xVal = xData[idx];
    float xF32 = ToFloat32<T>(xVal);
    float yF32 = coshf(xF32);
    yData[idx] = FromFloat32<T>(yF32);
}
```

#### 机制与设计原因

单次 VF launch 覆盖整个 TensorList；每个线程对每项 Tensor 复用相同 `tid/stride`，直接标量 GM 读写。它避免为每个 Tensor 创建单独 Kernel 节点，也不需要 prefix-sum/workspace 来把异长列表压成一个物理数组。每个 `idx` 只被一个全局线程序列访问，因此无 atomic/barrier。

#### 迁移边界

- 适用：TensorList 中各元素完全独立、直接GM标量API可用的逐元素算子。
- 前提：地址表协议一致；每项Tensor可扁平化；单个线程顺序遍历TensorList可接受。
- 失效：需要Tensor间协作、归约、共享缓存或合并DMA时应选其它Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Host 以每核至少1024元素限制小任务block数

#### 触发与源码依据

- 适用 Route：全部。
- 触发条件：`totalElements>0`。
- Golden 来源：[`foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp:95-103`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp#L95-L103)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t needCoreNum = std::min(coreNum, Ops::Base::CeilDiv(totalElements, MIN_PER_CORE));
tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
```

#### 机制与设计原因

blockDim 不直接取满平台 core，而按 `ceil(E/1024)` 截断；随后每 block 仍有512 threads。源码可证明这减少小元素域的启动 block 数，并维持 grid-stride覆盖，但固定仓未提供性能数据。

#### 迁移边界

- 适用：同类SIMT逐元素Kernel。
- 前提：`MIN_PER_CORE=1024` 与512线程、单元素成本的组合经目标算子重新评估；不能视为通用常数。
- 失效：每元素计算/访存成本显著不同或需要固定block数时应重新Tiling。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16→FP32 | R-003 BF16→FP32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode0 | schMode1 | schMode2 | Host dtype分支。 |
| 模板与 dtype | float | half | bfloat16_t | compile-time specialization。 |
| 循环与分核 | 相同512-thread grid-stride | 同 | 同 | shape不参与Route。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread scalars；0 dynamic UB | 同 | 同 | direct GM。 |
| 内存总占用与复用 | 编译器寄存器，不可见 | 同 | 同 | 无显式共享内存。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + coshf | GM + half casts + coshf | GM + BF16 casts + coshf | 全部SIMT。 |
| 同步、精度与 tail | FP32，无barrier | FP32 compute，half输出 | FP32 compute，BF16输出 | idx bound统一tail。 |
| fallback/失败边界 | unsupported dtype Host失败 | 同 | 同 | 无Device fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 同 | 公开支持，本文目标。 |
| Host/Launcher | arch22公共Tiling、最多50 tensor/core | 同 | arch35专用Tiling、最多256 tensor。 |
| Kernel/资源 | AIV双缓冲UB | 同 | 512-thread SIMT、direct GM、workspace0。 |

## 限制与失败边界

- `tensorCount>256` 明确 Tiling失败；`totalElements` 的 int64累加没有显式溢出检查。
- `ubSize-128KiB` 在写 local memory size前无显式下溢guard；受支持950资源应满足，但固定仓没有fallback。
- arch35 Tiling不显式验证output；公开ACLNN错误语义由自动生成/通用框架层承担，其实现固定仓不可见。
- `needCoreNum` 与 `totalElements` 是POD中当前body未读取字段；复用ABI时仍不得删除或换序。
- FP16/BF16输出转换的具体round模式未作为源码参数显式给出，只能归于SIMT intrinsic语义。
- 输出容量可大于输入，但仅输入长度前缀被写；无额外区间初始化。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_cosh/docs/aclnnForeachCosh.md` | 两段式原型 | API参数、错误语义、dtype/shape。 |
| 入口声明/约束 | `foreach/foreach_cosh/README.md` | 产品支持表 | 950公开支持。 |
| 入口实现 | `foreach/foreach_cosh/op_host/foreach_cosh_def.cpp` | `ForeachCosh`, `OP_ADD` | OpDef、AutoContiguous、ascend950注册。 |
| 构建接线 | `foreach/foreach_cosh/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `SUPPORT_TILING_DIR` | ascend950选择arch35 Tiling并接入ACLNN生成。 |
| Host/Tiling/Launcher | `foreach/foreach_cosh/op_host/arch35/foreach_cosh_tiling.cpp` | `ForeachCoshTilingFunc` | platform、POD、block、workspace、local memory、key。 |
| Host/Tiling/Launcher | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_tiling_data.h` | `ForeachCoshTilingData` | POD字段/容量/ABI。 |
| Host/Tiling/Launcher | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode候选与模板选择。 |
| Kernel entry/body | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh.cpp` | `foreach_cosh<schMode>` | compile-time router。 |
| Kernel entry/body | `foreach/foreach_cosh/op_kernel/arch35/foreach_cosh_simt.h` | `Process`, `OpForeachCoshSimt`, conversions | 512-thread launch、地址、循环、精度/tail。 |
| 构建接线 | `foreach/foreach_cosh/op_host/config/ascend950/foreach_cosh_binary.json` | `op_list` | 950三dtype binary条目。 |
| 构建接线 | `foreach/foreach_cosh/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950选择arch35 source。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B标准字段POD | 字段顺序/offset、256容量；inactive字段仍保留 | `ForeachCoshTilingData`; `offsetof` consumer |
| 调用顺序 | entry→Process→VF_CALL→Tensor loop→idx loop | 512 threads与地址表解析 | `Process`; `OpForeachCoshSimt` |
| workspace/内存 | workspace0、direct GM | localMemorySize reserve；无虚构UB | `ForeachCoshTilingFunc`; SIMT body |
| 同步 | 无barrier/atomic，靠disjoint idx | stride必须等于threadNum*blockNum | `OpForeachCoshSimt` |
| 精度与尾块 | low precision升到FP32 coshf | 转换顺序与`idx<count` | `ToFloat32`; `FromFloat32`; idx loop |
| 失败边界 | platform/resource/dtype/tensorCount失败阻断 | 不得绕过tensorCount256和unsupported dtype | `GetPlatformInfo`; `ForeachCoshTilingFunc` |
