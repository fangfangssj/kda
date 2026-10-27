---
type: Operator
title: "ForeachLog 950 ACLNN 算子知识"
description: "说明 aclnnForeachLog 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/docs/aclnnForeachLog.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_host/foreach_log_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_host/arch35/foreach_log_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_kernel/arch35/foreach_log_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_kernel/arch35/foreach_log_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_kernel/arch35/foreach_log.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_kernel/arch35/foreach_log_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_host/config/ascend950/foreach_log_binary.json
    role: implementation
created_at: "2026-08-25T15:45:22Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachLog 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachLogGetWorkspaceSize` + `aclnnForeachLog`，公开 ACLNN 两段式接口 |
| 计算语义 | 对 TensorList 的每个有效元素计算自然对数 |
| 输入输出 | ND TensorList；FP32、FP16、BF16；输入输出 dtype 一致；最多 256 个列表项 |
| 执行类型 | 单 Kernel，唯一节点为 SIMT |
| Kernel 路线 | R-001 FP32 `logf`；R-002 FP16 `hlog`；R-003 BF16 `hlog` |
| 核心资源 | TensorList GM 指针表、GM POD；1024 threads/block；配置 local memory `UB-128 KiB`，body 无显式 shared/dynamic UB 使用；算子 workspace 0 B |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→生成式 ACLNN 框架边界→950 OpDef/arch35 Tiling/POD/template key→SIMT entry/body→ascend950 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x=[x_0,...,x_{n-1}]` 中每个张量，950 Kernel 对 `0<=j<numel(x_i)` 执行 `y_i[j]=ln(x_i[j])`。每种 dtype 直接加载和存回相同 dtype；FP32 调 `logf`，FP16/BF16 调各自 `hlog` overload。固定仓没有这些 math intrinsic 对非法域、NaN/Inf 与近似精度的内部实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 每项 0～8 维 ND；可为空 Tensor | FP32/FP16/BF16 | 非空指针；列表内 dtype 一致；输入可非连续并由 `AutoContiguous` 框架边界处理 |
| `out` | 每项 0～8 维 ND；容量不小于对应输入 | 与 `x` 一致 | 输出不支持非连续；只写输入元素域 |
| `workspaceSize` | 标量 | `uint64_t*` | 950 算子 Tiling slice 为 0；最终值仍可能包含生成框架输入连续化空间 |
| `executor` | 执行器 | `aclOpExecutor**` | 第一段返回，第二段在指定 stream 提交 |

### 入口与版本

固定提交的公开最新版本是无后缀 `aclnnForeachLog`。公开 README/API 表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 映射到 `arch35`，Kernel CMake指定 `arch35/foreach_log.cpp`，950 binary JSON 覆盖三种 dtype。不存在“文档不支持、实现接线支持”冲突。ACLNN inner bridge 是仓内生成框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 文档给出空指针、dtype、列表内 dtype、输入输出 dtype、shape 容量和维数错误。OpDef 为输入输出声明 ND、FP32/FP16/BF16 和 `AutoContiguous`。arch35 Tiling 读取 AIV 核数与 UB，任一为 0 则失败；读取动态输入实例数，显式拒绝 `tensorNum>256`；读取第一个输入描述的 dtype，逐项写 storage `numel` 并求和。算子自有 950 Tiling 没有再次遍历输出 shape/dtype 的手写检查，相关校验属于公开 ACLNN/框架边界。

### 参数、Workspace 与生命周期

设 `N=sum_i numel(x_i)`、平台 AIV 核数 `C`、平台 UB 字节 `U`。若 `N=0`，`needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(C,ceil(N/1024))`。Tiling 把 `workspace[0]` 设为 0，并设置 `localMemorySize=uint32(U-128*1024)`。Kernel 参数中的 workspace 未被读取。POD、x/y TensorList GM 描述与输出 buffer 必须保持到 stream 完成；生成式 executor 的临时对象释放细节固定仓不可见。

### 启动图与失败传播

单节点启动图：`aclnnForeachLog` executor → 模板化 `foreach_log<schMode><<<needCoreNum, stream>>>` → 每个 block 内 `asc_vf_call<OpForeachLogSimt<T>>(dim3(1024),...)`。无 Kernel 间 workspace/GM 边；SIMT 线程直接从输入 GM 读并写输出 GM。平台、POD、tensor count、dtype 或 workspace 指针获取失败均令 Tiling 返回 `GRAPH_FAILED` 并阻断 launch；不支持的 template key 不会实例化有效 dtype body。

## Route 与参数设计

### 分核与资源计算

Host 只以所有 Tensor 的总元素数决定 grid 数，不预分配每核 Tensor 区间。每个 SIMT block 有 1024 threads；线程全局编号 `tid=blockIdx.x*blockDim.x+threadIdx.x`，全局步长 `stride=blockDim.x*gridDim.x=1024*needCoreNum`。对列表中的每个 Tensor，所有线程重复使用同一个 `tid/stride` 遍历该 Tensor 的 `idx=tid,tid+stride,...`。因此大 Tensor 由全 grid 条带化，小 Tensor 仅前若干线程有效。body 没有显式 UB buffer、shared memory、atomic 或 barrier。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `schMode=0` | `dataType==DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16 SIMT | `schMode=1` | `dataType==DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BF16 SIMT | `schMode=2` | `dataType==DT_BF16` | 其它 dtype Tiling 失败 |

```text
ForeachLogTilingFunc:
  DT_FLOAT   -> GET_TPL_TILING_KEY(schMode=0)
  DT_FLOAT16 -> GET_TPL_TILING_KEY(schMode=1)
  DT_BF16    -> GET_TPL_TILING_KEY(schMode=2)
  other      -> GRAPH_FAILED
build/template selector -> foreach_log<schMode>
  0 -> Process<float>; 1 -> Process<half>; 2 -> Process<bfloat16_t>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | FP32、ND、每项≤8维、列表≤256、950 | 单 SIMT | `Process<float>`→`OpForeachLogSimt<float>`→`logf` | GM direct，1024 threads/block；无 barrier/atomic/显式 UB | `idx<count` 保护；不支持 dtype 失败 |
| R-002 | schMode 1 | FP16，其余同 R-001 | 单 SIMT | `Process<half>`→`hlog` | 同上 | 同上 |
| R-003 | schMode 2 | BF16，其余同 R-001 | 单 SIMT | `Process<bfloat16_t>`→`hlog` | 同上 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 每核最少 1024 总元素 | `blockDim=min(C,ceil(N/1024))`，N=0 特判 1 | 匹配 1024 SIMT threads/block，避免为很小总域启动全部核 | 全部 | 无备用 Route | `MIN_PER_CORE`; core split |
| TensorList 上限 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `MAX_TENSOR_NUM`; `ForeachLogTilingData` |
| local memory reserve | `SetLocalMemorySize(uint32(U-128KiB))` | 为 DCache 留 128 KiB；body 无显式 local address | 全部 | `U<128KiB` 无显式 Guard，窄化边界未防护 | `DCACHE_SIZE`; `SetLocalMemorySize` |
| 1024 threads | 编译常量 `THREAD_NUM=1024`、`LAUNCH_BOUND(1024)` | thread/grid stride 与 launch 必须一致 | 全部 | 无其它线程配置 | `OpForeachLogSimt`; `Process` |
| 空 Tensor | `count==0 -> continue` | 避免解码/访问空项数据地址 | 全部 | 继续下一 Tensor | `OpForeachLogSimt` |

### POD/ABI

`ForeachLogTilingData` 是普通字段顺序结构；按源码声明与目标常规 8 B 对齐，字段 offsets 为 0/4/8/16，总大小 2064 B、对齐 8 B。entry 同时用注册宏解析一份局部 `tilingData`，实际 body 则通过 GM 指针直接读取同一结构。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachLogTilingData` | `needCoreNum` | `int32_t`/32 | offset 0 | core | `needCoreNum` | body 不读取；launch 使用 `SetBlockDim` 的同值 | N=0 为1；否则≤C |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT 外层 Tensor 循环 | 0～256 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素 | `sum(numel)` | body 不读取；只用于 Host 分核 | 求和溢出无显式 Guard |
| 同上 | `tensorElements` | `int64_t[256]`/每项64 | offset 16；结构 sizeof 2064 | 元素/列表项 | 每项 `storageShape.GetShapeSize()` | GM `offsetof` 定位后由 SIMT 读 `count` | inactive 尾项先 memset 为0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | `ForeachLogTilingFunc` | `ForeachLogTilingData` | `foreach_log<0>` | `Process<float>`; `OpForeachLogSimt<float>`; `LogFunc<float>` | 950 arch35 Kernel CMake + ascend950 JSON |
| R-002 | schMode 1 | 同上 | 同上 | `foreach_log<1>` | `Process<half>`; `hlog` | 同上 |
| R-003 | schMode 2 | 同上 | 同上 | `foreach_log<2>` | `Process<bfloat16_t>`; `hlog` | 同上 |

## Kernel 实现

### 公共执行流程

模板 entry 解析 POD 并以 `schMode` 编译期选择 dtype。`Process<T>` 通过 `offsetof(ForeachLogTilingData,tensorElements)` 在 GM POD 中取得 count 数组，将 `tensorCount`、数组、x/y TensorList 描述传给 `asc_vf_call`。SIMT body 顺序扫描 TensorList；空项跳过，否则解码第 `t` 个实际 GM 指针。每线程以全 grid stride 遍历该 Tensor，单次迭代完成标量 load→math intrinsic→store。没有跨线程数据依赖、shared buffer、barrier、atomic 或跨 Kernel边。

### Route R-001：FP32 SIMT logf

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_log_tiling.cpp` | `ForeachLogTilingFunc` | `dataType==DT_FLOAT` | schMode 0 |
| Router | `op_kernel/arch35/foreach_log.cpp` | `foreach_log<schMode>` | `schMode==TILING_KEY_FLOAT(0)` | `Process<float>` |
| Body | `foreach_log_simt.h` | `LogFunc<float>` | template `T=float` | `logf(x)` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log<0> -> Process<float>
  -> asc_vf_call<OpForeachLogSimt<float>>(dim3(1024),...)
    -> for each tensor -> SimtGetTensorAddr<float>(x/y)
      -> for idx=tid; idx<count; idx+=1024*gridDim: y[idx]=logf(x[idx])
```

##### 源码等价伪码

```text
tid = blockIdx*1024 + threadIdx
stride = 1024*gridDim
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0: continue
  xT=decode(xList,t); yT=decode(yList,t)
  for idx=tid; idx<count; idx+=stride:
    yT[idx]=logf(xT[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList descriptor | `base + (*base>>3) + t` | `tensorCount` 个64位地址 | pointer table | framework | `SimtGetTensorAddr` | Kernel launch | 首字为指针表字节 offset |
| x/y GM | `tensorPtr[t]+idx` | `tensorElements[t]` | FP32/ND | caller/thread load | `logf`/thread store | stream | `idx<count` tail guard |
| local memory | N/A（body 无显式地址） | launch budget `U-128KiB` | N/A | Tiling | 无直接 consumer | launch | 不伪造 shared 布局 |

##### 分层内存布局

N/A：SIMT body 直接 GM load/store，没有 AIV Queue/TBuf、显式 shared/dynamic UB 或中间 workspace；仅 launcher 配置 `localMemorySize=U-128 KiB`。

##### 字节账与复用顺序

```text
UB_explicit = 0
dynamic/local_memory_budget = uint32(U - 128*1024)
workspace_total(算子 slice) = 0
GM traffic per valid element = 4 B load + 4 B store
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | 每线程处理互不重叠 idx，无 barrier | `tid` 唯一且 stride 为全线程数 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load→compute→store | 线程内程序顺序 | GM load | `logf`/GM store | 单元素 | FP32；`idx<count` 保护尾部；无跨线程同步 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_log_tiling.cpp` | `ForeachLogTilingFunc` | grid、POD、local memory、schMode 0 |
| Kernel entry/router | `op_kernel/arch35/foreach_log.cpp` | `foreach_log<0>` | FP32 body 映射 |
| Device body/helper | `foreach_log_simt.h` | `OpForeachLogSimt<float>`; `LogFunc<float>` | grid-stride 与 `logf` |

### Route R-002：FP16 SIMT hlog

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_log_tiling.cpp` | `ForeachLogTilingFunc` | `DT_FLOAT16` | schMode 1 |
| Router/body | `foreach_log.cpp`; `foreach_log_simt.h` | `foreach_log<1>`; `LogFunc<half>` | `T=half` | `hlog(x)` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log<1> -> Process<half> -> asc_vf_call<OpForeachLogSimt<half>>(1024)
  -> TensorList decode -> grid-stride loop -> half load -> hlog -> half store
```

##### 源码等价伪码

```text
与 R-001 相同，元素类型改为 half，GM 地址按 2 B 元素递增，数学函数为 hlog。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `ptr[t]+idx` | count | FP16/ND | GM/thread | hlog/store | stream | 每元素2 B；tail guard |
| explicit local/workspace | N/A | 0 B显式；local budget同公共流程 | N/A | N/A | N/A | launch | 无复用 |

##### 分层内存布局

N/A：与 R-001 相同，无显式 SIMT 中间 buffer。

##### 字节账与复用顺序

```text
UB_explicit=0; local budget=uint32(U-128KiB); workspace=0
GM traffic per valid element=2 B load+2 B store
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | 线程索引互斥 | `idx<count` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 单元素 | 线程内顺序 | FP16 load | `hlog`→FP16 store | 单 idx | 源码无显式升精度/cast；尾部条件保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_log_tiling.cpp` | dtype branch | schMode 1 |
| Kernel entry/router | `foreach_log.cpp` | `foreach_log<1>` | half 映射 |
| Device body/helper | `foreach_log_simt.h` | `LogFunc<half>` | `hlog` 与 grid-stride |

### Route R-003：BF16 SIMT hlog

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_log_tiling.cpp` | `ForeachLogTilingFunc` | `DT_BF16` | schMode 2 |
| Router/body | `foreach_log.cpp`; `foreach_log_simt.h` | `foreach_log<2>`; `LogFunc<bfloat16_t>` | `T=bfloat16_t` | `hlog(x)` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log<2> -> Process<bfloat16_t> -> asc_vf_call<OpForeachLogSimt<bfloat16_t>>(1024)
  -> TensorList decode -> grid-stride loop -> BF16 load -> hlog -> BF16 store
```

##### 源码等价伪码

```text
与 R-001 相同，T=bfloat16_t；每个线程直接调用 hlog，源码没有 A2/A3 的显式 BF16->FP32->BF16 pipeline。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `ptr[t]+idx` | count | BF16/ND | GM/thread | hlog/store | stream | 每元素2 B；tail guard |
| explicit local/workspace | N/A | 0 B显式 | N/A | N/A | N/A | launch | local budget配置但未显式消费 |

##### 分层内存布局

N/A：没有显式 BF16/FP32 中间 buffer，`hlog` 内部资源固定仓不可见。

##### 字节账与复用顺序

```text
UB_explicit=0; local budget=uint32(U-128KiB); workspace=0
GM traffic per valid element=2 B load+2 B store
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | 线程索引互斥 | `idx<count` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 单元素 | 线程内顺序 | BF16 load | `hlog`→BF16 store | 单 idx | 源码无显式 cast/round；intrinsic 内部精度不可见；tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_log_tiling.cpp` | dtype branch | schMode 2 |
| Kernel entry/router | `foreach_log.cpp` | `foreach_log<2>` | BF16 映射 |
| Device body/helper | `foreach_log_simt.h` | `LogFunc<bfloat16_t>` | 直接 `hlog` 与线程循环 |

## 算子特定优化

### Trick T-001：每个 Tensor 复用统一全 grid 条带映射

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：`tensorCount<=256`，每项元素独立，launch 为 `needCoreNum` blocks×1024 threads。
- Golden 来源：[`foreach/foreach_log/op_kernel/arch35/foreach_log_simt.h:71-89`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_kernel/arch35/foreach_log_simt.h#L71-L89)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t t = 0; t < tensorCount; t++) {
    int64_t count = tensorElements[t];
    if (count == 0) {
        continue;
    }

    __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

    uint64_t tid = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint64_t stride = static_cast<uint64_t>(blockDim.x * gridDim.x);

    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        T val = xData[idx];
        yData[idx] = LogFunc<T>(val);
    }
}
```

#### 机制与设计原因

Host 只保存每项 count，不生成 256×core 的切片表。每个 Tensor 都使用相同全局 `tid/stride`；这避免 per-core Tensor 起止 POD和分支，且不需要 atomic/barrier，因为不同线程的 `idx` 不重叠。空项在解码数据指针前跳过。`blockDim=min(C,ceil(total/1024))` 控制总线程规模，而单项 tail 由 `idx<count` 自然消除。

#### 迁移边界

- 适用：TensorList 中各元素完全独立的 unary SIMT 算子。
- 前提：所有 Tensor 共用 dtype/操作；每次 store 地址由唯一 `tid+k*stride` 产生；列表项上限与 POD 一致。
- 失效：归约、邻域、跨 Tensor 依赖或需要 Tensor-specific launch 配置时，必须引入同步或不同分发。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若把 `stride` 误写为单 block 的 1024，会让多个 block 重复写同一元素。

### Trick T-002：按总元素数限制 block 数并固定 1024 线程

#### 触发与源码依据

- 适用 Route：全部
- 触发条件：`N>0` 时 `needCoreNum=min(C,ceil(N/1024))`；N=0 固定1。
- Golden 来源：[`foreach/foreach_log/op_host/arch35/foreach_log_tiling.cpp:95-111`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log/op_host/arch35/foreach_log_tiling.cpp#L95-L111)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (totalElements == 0) {
    tiling->needCoreNum = 1;
    context->SetBlockDim(1);
} else {
    int64_t needCoreNum = std::min(coreNum, Ops::Base::CeilDiv(totalElements, MIN_PER_CORE));
    tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
    context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
}

size_t* currentWorkspace = context->GetWorkspaceSizes(1);
OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
currentWorkspace[0] = WS_SYS_SIZE;

context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));
```

#### 机制与设计原因

`MIN_PER_CORE=1024` 与 SIMT `THREAD_NUM=1024` 对齐：总元素少时不启动超过 `ceil(N/1024)` 的 block，多时上限为平台 AIV 核数。算子不使用 workspace 中间态，Tiling 将其置零，并预留 128 KiB DCache 后设置 local memory budget。

#### 迁移边界

- 适用：固定 1024-thread、全 grid stride 的逐元素 SIMT 工作。
- 前提：总元素和不溢出 `int64_t`；平台 UB≥128 KiB；每项负载差异可由全 grid 重复遍历接受。
- 失效：需要 shared memory、不同 threads/block 或 Tensor-specific 调度时须重新推导 block/local memory。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不可只改 `THREAD_NUM` 而保留 Host 的 `MIN_PER_CORE=1024`，否则 launch 粒度与分核假设不一致。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | dtype branch |
| 模板与 dtype | `float/logf` | `half/hlog` | `bfloat16_t/hlog` | math intrinsic overload |
| 循环与分核 | 相同 grid-stride | 同左 | 同左 | shape 无关模板 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无显式 buffer | 同左 | 同左 | scalar SIMT direct GM |
| 内存总占用与复用 | GM+POD；local budget未显式用 | 同左 | 同左 | workspace 0 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT load/logf/store | load/hlog/store | load/hlog/store | 无 Vector/NDDMA/通信 |
| 同步、精度与 tail | 线程内 FP32，idx guard | 线程内 FP16，idx guard | 线程内 BF16，idx guard | 无 barrier/atomic |
| fallback/失败边界 | 无 | 无 | 无 | unsupported dtype Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；见无后缀卡 | 公开支持；见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 common Foreach Tiling，≤50 Tensor/core | 同 A2 | arch35 独立 Tiling，≤256 Tensor，按总元素/1024 分 block |
| Kernel/资源 | AIV Vector Queue；BF16 显式 FP32 cast | 同 A2 | SIMT direct GM；无显式 UB Queue；workspace 0 |

## 限制与失败边界

- 950 TensorList 最多 256 项；超过即 Tiling 失败。`totalElements` 的 int64 求和、`U-128KiB` 再窄化到 uint32 没有显式溢出/下溢 Guard。
- block 数按所有 Tensor 总元素计算，但每个 Tensor 都由整个 grid 重新扫描；列表中很多极小 Tensor 时的调度代价没有固定源码性能数据，不能声称收益。
- `needCoreNum`、`totalElements` 被序列化但 Device body不读取；不得误认为存在 POD 定义的 per-core slice。
- ACLNN 生成 bridge、最终 workspace 合计、`asc_vf_call` 框架 launch 细节及 `logf/hlog` 内部精度/资源属于固定仓不可见边界。
- 本次未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_log/docs/aclnnForeachLog.md` | 两段式 ACLNN 原型 | API、shape/dtype/平台/错误 |
| 入口实现 | `foreach/foreach_log/op_host/foreach_log_def.cpp` | `ForeachLog`; `OP_ADD` | 950 OpDef、AutoContiguous |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`; `IMPL_OP_INFERSHAPE(ForeachLog)` | 输出数量、shape 与 dtype 传播 |
| Host/Tiling/Launcher | `foreach/foreach_log/op_host/arch35/foreach_log_tiling.cpp` | `ForeachLogTilingFunc` | 平台、POD、grid、workspace、local memory、selector |
| Host/Tiling/Launcher | `foreach/foreach_log/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950 到 arch35 的活动映射 |
| Host/Tiling/Launcher | `foreach/foreach_log/op_kernel/arch35/foreach_log_tiling_data.h` | `ForeachLogTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_log/op_kernel/arch35/foreach_log_tiling_key.h` | `ForeachLog` template args | schMode 0/1/2 |
| Kernel entry/body | `foreach/foreach_log/op_kernel/arch35/foreach_log.cpp` | `foreach_log<schMode>` | 编译期 dtype router |
| Kernel entry/body | `foreach/foreach_log/op_kernel/arch35/foreach_log_simt.h` | `Process`; `OpForeachLogSimt`; `LogFunc` | 线程映射、GM、math、tail |
| 构建接线 | `foreach/foreach_log/op_kernel/CMakeLists.txt` | 950 `add_kernel_sources` | ascend950 选择 arch35 source |
| 构建接线 | `foreach/foreach_log/op_host/config/ascend950/foreach_log_binary.json` | `ForeachLog op_list` | 950 三 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B plain POD 保存 count 与每项元素数 | 字段顺序、256 项、int32/int64 位宽 | `ForeachLogTilingData` |
| 调用顺序 | entry→Process→asc_vf_call→逐 Tensor grid-stride | 1024 threads 与全 grid stride | `Process`; `OpForeachLogSimt` |
| workspace/内存 | workspace 0；body direct GM；local budget U-128KiB | 不得虚构 UB/shared；保留 DCache reserve | `WS_SYS_SIZE`; `SetLocalMemorySize` |
| 同步 | 无跨线程同步，地址天然互斥 | stride 必须覆盖全 grid | `tid/stride` 公式 |
| 精度与尾块 | FP32 logf；FP16/BF16 hlog；idx<count | 不得外推 intrinsic 内部 cast/round | `LogFunc<T>` |
| 失败边界 | >256、非法 dtype、平台信息失败阻断 launch | 无 Device fallback | `ForeachLogTilingFunc` |