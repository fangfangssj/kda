---
type: Operator
title: "ForeachAbs 950 ACLNN 算子知识"
description: "说明 aclnnForeachAbs 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/docs/aclnnForeachAbs.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_host/foreach_abs_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_host/arch35/foreach_abs_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_host/arch35/foreach_abs_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/arch35/foreach_abs.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/arch35/foreach_abs_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/arch35/foreach_abs_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_host/config/ascend950/foreach_abs_binary.json
    role: implementation
created_at: "2026-08-25T15:28:45Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAbs 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAbsGetWorkspaceSize` + `aclnnForeachAbs`，无版本后缀的公开两段式 ACLNN；固定仓未出现更新的同语义版本。 |
| 计算语义 | 对张量列表逐张量、逐元素计算 `y_i[j] = \|x_i[j]\|`。 |
| 输入输出 | 动态 TensorList；`FLOAT16/FLOAT32/BFLOAT16`，ND；输入可非连续并由框架 `AutoContiguous` 处理，公开接口不接受非连续输出。 |
| 执行类型 | 单 Kernel：AIV entry 内以 `VF_CALL` 启动 SIMT 线程函数；实际逐元素节点为 SIMT，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP16、R-002 FP32、R-003 BF16；key 分别为 0、1、2。 |
| 核心资源 | GM TensorList 地址表、GM POD；1024 SIMT threads/block；无 AIV Queue/TBuf、无 dynamic UB 数据区、算子 Kernel workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档 → `ForeachAbs` OpDef/框架生成 bridge 边界 → arch35 Tiling/POD → 模板 key → `foreach_abs` entry → `Process`/SIMT body → ascend950 构建与 binary config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`、`y=[y_0,...,y_(n-1)]`，每个有效输入元素满足 `y_i[j]=abs(x_i[j])`。950 实体用比较与减法实现：普通类型返回 `v>=0 ? v : 0-v`；BF16 先转 FP32，执行同一表达式，再转回 BF16。由此，源码对 `-0` 的行为是保留 `-0`，NaN 走 `0-NaN` 分支；固定仓没有给出 NaN payload/sign 的额外承诺。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，各 Tensor 0～8 维，ND | FP16/FP32/BF16，列表内一致 | 非空指针；支持空 Tensor；OpDef 标记 `AutoContiguous()`。 |
| `out` | TensorList，各 Tensor 0～8 维，ND | 与 `x` 相同，列表内一致 | 非空指针；每个输出 storage shape size 不小于对应输入；公开文档声明非连续输出不支持。Kernel 只写输入元素数范围。 |
| `workspaceSize` | 标量 | `uint64_t*` | 第一段返回 executor 所需 Device workspace；算子 Tiling 明确写 0，但框架前处理的总 workspace 细分在固定仓不可见。 |
| `executor` | 不适用 | `aclOpExecutor**` | 第一段产生，第二段消费，承载框架生成的计算流程。 |
| `workspace/stream` | Device 地址/stream | 指针 | 第二段按同一 stream 提交 executor。 |

### 入口与版本

固定仓公开的最新且唯一同语义入口是无后缀 `aclnnForeachAbs`。公开原型位于 `docs/aclnnForeachAbs.md`；仓内没有手写 ACLNN bridge，`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 生成接口接线，因此 bridge/executor 封装是明确的框架边界。`ForeachAbs` OpDef 的动态输入输出、dtype、ND 与 `AutoContiguous` 配置继续闭合到 Tiling 与 Kernel。README 和 ACLNN 文档均公开支持 Ascend 950PR/950DT，OpDef、Host arch35 映射、Kernel CMake 与 ascend950 binary config 一致，无发布冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段接口检查 `x/out` 空指针、dtype 支持/一致性、列表内 dtype、0～8 维和输出容量；公开错误为 `161001/161002/561002`。
2. OpDef 将动态 `x/y` 限定为 FP16/FP32/BF16、ND，并对输入输出声明 `AutoContiguous`；公开契约仍明确禁止非连续输出。
3. 框架生成 bridge 将 TensorList 编码成 GM 地址表并进入 `ForeachAbsTilingFunc`。生成代码不可见，但算子拥有的 selector、POD、entry 和 body 均已闭合。
4. Tiling 从 compile info 读取 AIV core 数和 UB；无有效 compile info 时回退到平台信息。任一值无效，或物理 UB `<=128 KiB`，返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

Tiling 扫描 `tensorNum` 个动态输入，累加 storage numel 并写前缀和 `cumulativeOffsets[i]`。`tensorNum>256` 失败。非空域时：

```text
perCoreElements = align_up(max(1024, ceil(totalElements / coreNum)), 32)  # 元素
needCoreNum = min(coreNum, max(1, ceil(totalElements / perCoreElements)))
blockDim = needCoreNum
localMemorySize = physicalUB - 128 KiB
kernel_workspace[0] = 0
```

空域固定 `blockDim=1`、`perCoreElements=0`、workspace 0。`perCoreElements` 写入 POD 但 Device 当前实现不读取；SIMT 用 `blockDim × 1024` 线程的全局步长划分。TensorList、POD、输出和 executor/workspace 由调用方/框架持有，至少存活到 stream 中任务完成。

### 启动图与失败传播

```text
aclnnForeachAbsGetWorkspaceSize
  -> framework-generated ForeachAbs bridge
    -> ForeachAbsTilingFunc / TilingParseForForeachAbs
      -> one launch: foreach_abs<schMode>(xList, yList, workspace, tiling)
        -> Process<T>
          -> VF_CALL<OpForeachAbsSimtKernel<T, IDX_T>>(Dim3(1024), ...)
```

单节点输入为 GM TensorList `x` 和 POD，输出为 GM TensorList `y`。没有 Kernel 间 edge；`workspace` 参数不被读取，Tiling slice 为 offset 0、extent 0。第二段以指定 stream 提交；Tiling/launch 失败阻止节点执行，生成 executor 的更细失败清理在固定仓不可见。

## Route 与参数设计

### 分核与资源计算

`totalElements` 是所有 storage numel 的 64 位和，`cumulativeOffsets[i]` 是包含第 `i` 个 Tensor 的前缀和。Host 保证 block 数不超过平台 AIV core 数并把每核目标向 32 元素对齐；Device 不使用 `perCoreElements`，而令每个 SIMT thread 从 `blockIdx*1024+threadIdx` 开始，以 `1024*blockNum` 为步长遍历扁平拼接域。每个元素线性扫描最多 256 个前缀和来确定 Tensor 与局部下标。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 SIMT | key/schMode 0，`Process<half>` | 第一个输入 descriptor 为 `DT_FLOAT16`；空列表时 Tiling 默认 dtype 为 `DT_FLOAT`，不会命中此项 | dtype FP32/BF16 分支；非法 dtype 由 OpDef/API 拒绝 |
| 2 | R-002 FP32 SIMT | key/schMode 1，`Process<float>` | 第一个输入为 `DT_FLOAT`；或 `tensorNum==0` 时默认 dtype | dtype BF16 分支 |
| 3 | R-003 BF16 SIMT | key/schMode 2，`Process<bfloat16_t>` | 第一个输入为 `DT_BF16` | 非支持 dtype 不存在已注册实例 |

```text
dtype == FLOAT16  -> schMode 0 -> Process<half>
dtype == FLOAT32  -> schMode 1 -> Process<float>
dtype == BFLOAT16 -> schMode 2 -> Process<bfloat16_t>
otherwise         -> Host helper returns 0, but public/OpDef contract must reject before launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 0 | 950、FP16、Tensor 数≤256、UB>128 KiB | 单 entry + SIMT VF | `foreach_abs<0>` → `Process<half>` → `OpForeachAbsSimtKernel<half,int32/int64>` | 直接 GM load/store；无 barrier/atomic/workspace | `flatIdx<totalElements`；总元素决定 32/64 位 index |
| R-002 FP32 | 1 | 同上，FP32 | 同上 | `foreach_abs<1>` → `Process<float>` | 同上 | 同上 |
| R-003 BF16 | 2 | 同上，BF16 | 同上 | `foreach_abs<2>` → BF16 specialization | GM BF16→寄存器 FP32→BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `static_cast<int32_t>(tensorNum) > 256` 失败 | POD 前缀和固定 256 项 | 全部 | `GRAPH_FAILED` | `foreach_abs_tiling_arch35.cpp::ForeachAbsTilingFunc` |
| DCache 保留 | `ubSize > 128*1024` | Tiling 仅把扣除后的值设为 local memory；当前 body 不分配 dynamic UB | 全部 | `GRAPH_FAILED` | `ForeachAbsTilingFunc` |
| 索引位宽 | `totalElements <= INT32_MAX` | 避免常见规模使用 64 位循环变量 | 全部 | 大域走 `IDX_T=int64_t`，不是失败 | `foreach_abs_simt.h::Process` |
| 空域 | `totalElements==0` | 保持合法 launch，同时所有线程被循环 guard 挡住 | 全部 | blockDim 1、0 次 store | `ForeachAbsTilingFunc` / `OpForeachAbsSimtKernel` |
| 地址表格式 | 首字 `tensorPtrOffset` 必须为 8 字节倍数 | Device 以 `offset>>3` 找指针数组 | 全部 | 固定仓没有 Device fallback | `foreach_abs_simt.h::SimtGetTensorAddr` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAbsTilingData` | `needCoreNum` | int32/32 | offset 0；结构 align 8、sizeof 2072 | core | `needCoreNum` 或空域 1 | 当前 body 不读取 | 与 blockDim 冗余 |
| 同上 | `tensorCount` | int32/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | SIMT 线性扫描上界 | 0～256 |
| 同上 | `totalElements` | int64/64 | offset 8 | 元素 | storage numel 之和 | index 位宽与循环终点 | Host 求和未显式检查溢出 |
| 同上 | `perCoreElements` | int64/64 | offset 16 | 元素/core | Host 对齐公式；空域 0 | 当前 body 不读取 | inactive producer 信息 |
| 同上 | `cumulativeOffsets` | int64[256] | offset 24，2048 bytes | 元素前缀和 | 循环中 `totalElements += numel` 后写第 i 项 | `cumOffsets[t]`/`cumOffsets[t-1]` | 仅前 `tensorCount` 项有效 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 0 | `ForeachAbsTilingFunc`/`GetTilingKeyByDtype` | `ForeachAbsTilingData` | `foreach_abs<0>` | `Process<half>`、`OpForeachAbsSimtKernel<half,IDX_T>` | Kernel CMake ascend950 + ascend950 binary JSON |
| R-002 | 1 | 同上 | 同上 | `foreach_abs<1>` | `Process<float>`、SIMT body | 同上 |
| R-003 | 2 | 同上 | 同上 | `foreach_abs<2>` | `Process<bfloat16_t>`、`AbsCompute<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 并按编译期 `schMode` 选择 `T`。`Process<T>` 从 POD 取总元素、Tensor 数和前缀和；根据总元素选择 int32/int64 index，然后以每 block 1024 threads 启动 SIMT VF。线程把整个 TensorList 看成串接的一维域，通过前缀和恢复 `(tensorId,localIdx)`，从地址表取真实 GM 指针，单次读、计算、单次写。不同线程写唯一扁平索引，无 atomic、barrier 或 workspace 交接。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach/foreach_abs/op_host/arch35/foreach_abs_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key 0 |
| Router | `foreach/foreach_abs/op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<schMode>` | `schMode==TILING_KEY_FLOAT16` | `Process<half>` |
| Device | `foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h` | `Process<half>` | total≤INT32_MAX | int32 index；否则 int64 |

#### Device 数据流

##### 函数级调用链

```text
foreach_abs<0>(x,y,workspace,tiling)
  -> Process<half>
    -> VF_CALL<OpForeachAbsSimtKernel<half, IDX_T>>(Dim3(1024))
      -> SimtGetTensorAddr<half>(x/y,tensorId)
      -> AbsCompute<half>::Compute
```

##### 源码等价伪码

```text
for flat = blockIdx*1024 + threadIdx; flat < total; flat += 1024*blockNum:
  scan t=0..tensorCount-1 until flat < cumulativeOffsets[t]
  prev = t==0 ? 0 : cumulativeOffsets[t-1]
  local = flat-prev
  v = xAddrTable[t][local]
  yAddrTable[t][local] = (v >= half(0)) ? v : half(0)-v
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y address table | base + `(*base>>3)` + tensorId | tensorCount×8-byte pointers | uint64 address table | framework bridge | `SimtGetTensorAddr` | executor/launch | first offset divisible by 8 |
| GM x/y tensor | `ptr[t]+flat-prev` | input numel elements | FP16 ND storage | caller / thread | thread / output | caller through stream completion | unique element per thread |
| GM POD | offsets 0～2071 | 2072 bytes | ABI struct | Tiling | `Process`/SIMT | launch | align 8 |

##### 分层内存布局

N/A：该 SIMT 节点没有 AIV Queue/TBuf 或 shared/dynamic UB 数据区；标量与索引仅存于线程私有寄存器，寄存器字节数在固定仓不可见。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub_data = 0  # SetLocalMemorySize(physicalUB-128KiB) 是 launch 上限，body 未取用
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程寄存器 `val` | GM load | `AbsCompute` | 下一循环迭代 | 单线程程序序 | 每个 flat index 唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT grid-stride | 无 barrier/atomic | thread GM load | 同线程 GM store | 独占 flat index | FP16 比较/减法；`flat<total` 处理尾部与空域 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_abs_tiling_arch35.cpp` | `ForeachAbsTilingFunc` | 产生 key、blockDim、前缀和、0 workspace |
| Kernel entry/router | `op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<0>` | key 0 映射 `half` |
| Device body/helper | `op_kernel/arch35/foreach_abs_simt.h` | `OpForeachAbsSimtKernel<half,IDX_T>` | 1024-thread grid-stride 与直接 GM 读写 |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_abs_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT` | key 1 |
| Router | `op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<schMode>` | schMode 1 | `Process<float>` |
| Device | `op_kernel/arch35/foreach_abs_simt.h` | `Process<float>` | total≤INT32_MAX | int32，否则 int64 |

#### Device 数据流

##### 函数级调用链

```text
foreach_abs<1> -> Process<float>
  -> VF_CALL<OpForeachAbsSimtKernel<float, IDX_T>>(Dim3(1024))
    -> SimtGetTensorAddr<float> -> AbsCompute<float>::Compute
```

##### 源码等价伪码

```text
按 R-001 的 flat/thread/tensorId/local 映射；
v:float = x[t][local]; y[t][local] = (v>=0.0f) ? v : 0.0f-v
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y tensor | `ptr[t]+local` | input numel×4 bytes | FP32 ND | caller/thread | thread/caller | stream 完成前有效 | unique flat index |
| GM POD/address table | 同 R-001 | POD 2072 bytes + pointer array | ABI/uint64 | Host/framework | Device | launch | 无 UB staging |

##### 分层内存布局

N/A：无 AIV UB/shared memory；只使用 GM 与线程私有寄存器。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub_data=0; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `val` 寄存器 | GM load | compare/subtract | 下一 loop | 程序序 | 唯一 flat index |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | GM read | GM write | 单元素 | FP32 原精度；grid-stride guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_abs_tiling_arch35.cpp` | `GetTilingKeyByDtype` | FP32→1 |
| Kernel entry/router | `op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<1>` | 映射 float |
| Device body/helper | `op_kernel/arch35/foreach_abs_simt.h` | `AbsCompute<float>` | 原类型比较与减法 |

### Route R-003：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_abs_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_BF16` | key 2 |
| Router | `op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<schMode>` | schMode 2 | `Process<bfloat16_t>` |
| Device | `op_kernel/arch35/foreach_abs_simt.h` | `AbsCompute<bfloat16_t>` | 模板特化 | BF16→FP32→BF16 |

#### Device 数据流

##### 函数级调用链

```text
foreach_abs<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachAbsSimtKernel<bfloat16_t, IDX_T>>(Dim3(1024))
    -> AbsCompute<bfloat16_t>::Compute
```

##### 源码等价伪码

```text
v_bf16 = x[t][local]
f32 = float(v_bf16)
abs_f32 = (f32>=0.0f) ? f32 : 0.0f-f32
y[t][local] = bfloat16_t(abs_f32)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `ptr[t]+local` | input numel×2 bytes | BF16 ND | caller/thread | thread/caller | stream completion | direct load/store |
| thread temporaries | scalar | `v_bf16,f32,absF32` | BF16/FP32 | thread | specialization | one iteration | register bytes invisible |

##### 分层内存布局

N/A：BF16 的 FP32 临时量是线程标量，不是 dynamic UB/shared buffer。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub_data=0; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 scalar | BF16 cast | compare/subtract/downcast | next flat index | program order | one thread owns scalar |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16 compute | scalar cast sequence | BF16 load | BF16 store | one value | FP32 中计算后降为 BF16；具体 round mode 固定仓未显式命名；grid guard 处理尾部 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_abs_tiling_arch35.cpp` | `GetTilingKeyByDtype` | BF16→2 |
| Kernel entry/router | `op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<2>` | 映射 bfloat16_t |
| Device body/helper | `op_kernel/arch35/foreach_abs_simt.h` | `AbsCompute<bfloat16_t>` | 显式 FP32 中间值 |

## 算子特定优化

### Trick T-001：用拼接域 grid-stride 避免逐 Tensor Kernel 启动

#### 触发与源码依据

- 适用 Route：R-001～R-003
- 触发条件：950 arch35 的全部合法 TensorList；`tensorCount<=256`。
- Golden 来源：[`foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h:63-84`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h#L63-L84)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (IDX_T flatIdx = static_cast<IDX_T>(
        AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum()
        + AscendC::Simt::GetThreadIdx());
     flatIdx < totalElements;
     flatIdx += static_cast<IDX_T>(
        AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
    int32_t tensorId = tensorCount - 1;
    IDX_T prevCumSum = 0;
    for (int32_t t = 0; t < tensorCount; t++) {
        if (flatIdx < static_cast<IDX_T>(cumOffsets[t])) {
            tensorId = t;
            prevCumSum = (t > 0) ? static_cast<IDX_T>(cumOffsets[t - 1]) : static_cast<IDX_T>(0);
            break;
        }
    }
```

#### 机制与设计原因

Host 只生成一个前缀和 POD 与一个 Kernel launch；`blockDim×1024` 个线程覆盖所有 Tensor 的拼接域，因此避免为列表中每个 Tensor 分别提交 Kernel。代价是每个元素线性扫描前缀和；它依赖 `tensorCount<=256`。grid-stride 步长来自 `GetThreadNum()*GetBlockNum()`，尾部由 `flatIdx<totalElements` 精确保护。

#### 迁移边界

- 适用：多个同 dtype Tensor 的简单逐元素 unary 运算。
- 前提：地址表 ABI、单调前缀和、总元素 64 位可表示、每个输出元素独占写。
- 失效：列表超过 256、需要 Tensor 间归约/同步或输出别名会产生写冲突时必须改写调度。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：按总元素选择 32/64 位 SIMT 索引

#### 触发与源码依据

- 适用 Route：R-001～R-003
- 触发条件：`totalElements<=INT32_MAX` 走 int32，否则走 int64。
- Golden 来源：[`foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h:98-109`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h#L98-L109)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (totalElements <= static_cast<int64_t>(INT32_MAX)) {
    using IDX_T = int32_t;
    AscendC::Simt::VF_CALL<OpForeachAbsSimtKernel<T, IDX_T>>(
        AscendC::Simt::Dim3(THREAD_NUM),
        static_cast<IDX_T>(totalElements), tensorCount,
        cumOffsets, x, y);
} else {
    using IDX_T = int64_t;
```

#### 机制与设计原因

常见规模的 loop、前缀比较和 local index 使用 32 位整数；超过边界才保留 64 位寻址能力。阈值来自 `INT32_MAX`，不是经验 shape 条件。

#### 迁移边界

- 适用：元素总数可在 Host/POD 准确统计的 grid-stride Kernel。
- 前提：32 位路径的所有参与索引均不超过 `INT32_MAX`。
- 失效：总元素更大时必须走源码中的 int64 specialization，不能窄化。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的可验证错误边界是对大于 `INT32_MAX` 的总元素强制窄化；官方实现显式改走 int64。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 0 | key 1 | key 2 | dtype 唯一映射 |
| 模板与 dtype | `Process<half>` | `Process<float>` | `Process<bfloat16_t>` specialization | binary config 三个 dtype |
| 循环与分核 | colspan：相同 1024-thread grid-stride、前缀和定位 | 同左 | 同左 | totalElements/blockDim |
| Queue/TBuf/寄存器/dynamic UB/通信 window | scalar register；无 UB | 同左 | BF16+FP32 scalar | 无共享数据区 |
| 内存总占用与复用 | GM direct | GM direct | GM direct | workspace 0 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT GM load/store | 同左 | 同左 + scalar casts | 不使用 Vector DMA staging |
| 同步、精度与 tail | 原类型比较/减法 | 原类型比较/减法 | FP32 中计算再 BF16 | unique index + loop guard |
| fallback/失败边界 | int64 index fallback | 同左 | 同左 | Tensor≤256、UB>128KiB |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 foreach Tiling | arch22 公共 foreach Tiling | arch35 前缀和 Tiling，key 0/1/2 |
| Kernel/资源 | AIV Vector、TQue/TBuf | AIV Vector、TQue/TBuf | SIMT 直接 GM，无 UB staging |

## 限制与失败边界

- 950 Tensor 数最多 256；单 Tensor 维数最多 8；所有输入及输出 dtype 必须一致，输出容量不小于输入。
- Host 对 `totalElements` 和累计 offset 的 int64 加法没有显式溢出检查；超出 int64 表示域不在闭合的合法范围。
- `perCoreElements` 是 ABI 字段但当前 Device 不读取；不可据此实现每核连续 slice。
- 地址表首字 offset 的构造属于框架边界；Device 明确假定其可右移 3 位定位 uint64 指针数组。
- 物理 UB 必须大于 128 KiB，尽管当前 SIMT body 不使用所设置的 local memory 数据区。
- 生成 ACLNN bridge 的具体 workspace 合并与 executor cleanup 固定仓不可见；算子 Tiling workspace 明确为 0。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_abs/docs/aclnnForeachAbs.md` | 两段式原型/产品表 | API、错误与平台 |
| 入口实现 | `foreach/foreach_abs/op_host/foreach_abs_def.cpp` | `ops::ForeachAbs` | dtype/format/AutoContiguous/ascend950 config |
| Host/Tiling | `foreach/foreach_abs/op_host/arch35/foreach_abs_tiling_arch35.cpp` | `ForeachAbsTilingFunc` | selector、分核、POD、workspace |
| POD | `foreach/foreach_abs/op_host/arch35/foreach_abs_tiling_arch35.h` | `ForeachAbsTilingData` | Host ABI |
| POD mirror | `foreach/foreach_abs/op_kernel/arch35/foreach_abs_tiling_data.h` | `ForeachAbsTilingData` | Device ABI |
| Kernel entry | `foreach/foreach_abs/op_kernel/arch35/foreach_abs.cpp` | `foreach_abs<schMode>` | key→T router |
| Kernel body | `foreach/foreach_abs/op_kernel/arch35/foreach_abs_simt.h` | `Process`/`OpForeachAbsSimtKernel` | SIMT loop、地址、精度 |
| 模板 key | `foreach/foreach_abs/op_kernel/arch35/foreach_abs_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL` | schMode 0/1/2 实例 |
| Host 构建 | `foreach/foreach_abs/op_host/CMakeLists.txt` | `SUPPORT_TILING_DIR` | ascend950→arch35 |
| Kernel 构建 | `foreach/foreach_abs/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 arch35 source |
| binary config | `foreach/foreach_abs/op_host/config/ascend950/foreach_abs_binary.json` | `op_list` | 三个 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 前缀和定位 Tensor | 字段顺序、int64 offset、256 上限 | `ForeachAbsTilingData` |
| 调用顺序 | entry→Process→VF_CALL | `Dim3(1024)` 与 blockDim 的组合 | `Process` |
| workspace/内存 | 直接 GM、workspace 0 | 地址表编码、owner 存活期 | `SimtGetTensorAddr` |
| 同步 | 独占元素无需 atomic/barrier | 不得引入输出写冲突 | `OpForeachAbsSimtKernel` |
| 精度与尾块 | 32/64 位 index specialization | BF16 FP32 中间值、`flat<total` | `AbsCompute`/`Process` |
| 失败边界 | Tensor≤256、UB>128KiB | 不得把 Host 失败改成 silent truncation | `ForeachAbsTilingFunc` |