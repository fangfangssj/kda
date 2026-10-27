---
type: Operator
title: "ForeachAsin 950 ACLNN 算子知识"
description: "说明 aclnnForeachAsin 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/docs/aclnnForeachAsin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/foreach_asin_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/arch35/foreach_asin_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/arch35/foreach_asin.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/arch35/foreach_asin_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/config/ascend950/foreach_asin_binary.json
    role: implementation
created_at: "2026-08-25T15:31:05Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAsin 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAsinGetWorkspaceSize` + `aclnnForeachAsin`，公开 ACLNN 两段式接口，固定仓仅公开无版本后缀版本。 |
| 计算语义 | 对 tensor list 中每个有效元素执行 `y[t][i] = asin(x[t][i])`。 |
| 输入输出 | ND 动态 tensor list；FP16、FP32、BF16；输入经 `AutoContiguous`，输出要求连续。 |
| 执行类型 | 单 Kernel SIMT：AICore entry 内对每个相交 tensor 发起一个 512-thread `Simt::VF_CALL`；无 Cube/AIC/MIX_CV 节点。 |
| Kernel 路线 | R-001 FP32 `asinf`；R-002 FP16→FP32 `asinf`→FP16；R-003 BF16→FP32 `asinf`→BF16。 |
| 核心资源 | GM tensor-list 描述与输入/输出；无显式 TQue/TBuf/shared/dynamic UB；Host 声明 16 MiB workspace 并设置 `UB-128 KiB` local memory，但 body 不寻址二者。 |
| 来源闭包 | source_closed：公开 ACLNN 文档→OpDef→arch35 Tiling/POD/key→template entry→SIMT body→950 构建与 binary config 已闭合；ACLNN inner bridge、list descriptor 和 SIMT math intrinsic 内部为框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 list 长度为 `L`，第 `t` 个 tensor 的存储元素数为 `N_t`。对每个 `0 <= i < N_t`，`out[t][i]=asin(x[t][i])`。Host 将各 tensor 按累计 offset 串成总长 `E=sum(N_t)` 的逻辑一维域，仅用于 core 切分；SIMT 实际仍以每个 tensor 的独立 GM 指针和局部索引读写。`|x|>1` 的数值结果由 `asinf` intrinsic 决定，固定仓未展开 intrinsic。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | tensor list；ND；每项 0~8 维 | FP16、FP32、BF16，list 内一致 | 文档允许空 tensor和非连续输入；OpDef `.AutoContiguous()` 提供连续化边界；null、非法 dtype、list 内不一致或维度 >8 失败。 |
| `out` | tensor list；ND；每项 0~8 维 | 与 `x` 一致 | 输出不支持非连续；公开文档要求对应 shape size 不小于输入，公共 infer-shape 路径实际逐项复制输入 shape/dtype。 |
| 属性 | N/A | N/A | 无属性。 |

### 入口与版本

公开 API 是无版本后缀的 `aclnnForeachAsinGetWorkspaceSize`/`aclnnForeachAsin`，固定仓没有更高公开版本。手写 inner bridge 不在仓内，由 `ACLNNTYPE aclnn` 构建框架生成。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef `AddConfig("ascend950")`、Host CMake `ascend950→arch35`、Kernel CMake 的 arch35 source 与 ascend950 binary config 均一致，不存在文档/实现冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 第一段完成 null、dtype、shape、list 内一致性检查并建立 executor；输入非连续时由 OpDef 的 `.AutoContiguous()` 转换，输出必须连续。arch35 `ForeachAsinTilingFunc` 获取 AIV core 数 `C` 与 UB 字节数 `U`，要求 `C>0`、`U>0` 且 `U>128 KiB`。它读取动态输入实例数 `L`，遍历每项 storage shape，写累计元素 offset 和 `E`；再按 dtype 生成 template key。null platform/shape/desc/POD/workspace 指针、非法 dtype 或 local-memory 设置失败均返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

第一段返回 executor 和 workspace size。arch35 Tiling 固定写 `workspaces[0]=16777216` B（16 MiB），并调用 `SetLocalMemorySize(U-128 KiB)`；Kernel entry 接收 `workspace` 但不读取，SIMT body也无 workspace 参数，因此 16 MiB 是框架/AscendC tools 声明，算子语义数据 0 B。调用者持有 workspace、x/y GM 与 executor，至少覆盖第二段向 stream 提交及 Kernel 完成；固定仓未展开 executor 内部 owner/cleanup。

### 启动图与失败传播

单 Device 节点：Host 以 `blockDim=needCoreNum` 启动 template `foreach_asin<dtype>(xList,yList,workspace,tiling)`。每个 AIV block 在 `Process<T>` 中得到本核全局逻辑区间，与每个 tensor 的累计区间求交；对每个非空交集顺序调用一个 `Simt::VF_CALL<ForeachAsinSimtKernel<T>>(Dim3(512), localStart, localEnd, xAddr, yAddr)`。不同 tensor VF call 的 ordering 是同一 entry 中的程序顺序；没有跨 Kernel workspace 边、event 或 atomic。任一 Host/Tiling 失败阻断 entry；SIMT body无显式错误返回。

## Route 与参数设计

### 分核与资源计算

`P=128 KiB` DCache 预留，`U_actual=U-P`。若 `E=0`，`perCoreElements=1024`、`needCoreNum=1`；否则 `raw=max(1024,ceil(E/C))`，`perCoreElements=ceil(raw/32)*32`，`needCoreNum=ceil(E/perCoreElements)`。第 `b` 核处理 `[b*perCoreElements, min((b+1)*perCoreElements,E))`，空核立即返回。该公式保证单核至少 1024 元素且边界以 32 元素对齐，但没有按 dtype 字节数对齐。`SetLocalMemorySize(U_actual)` 是 launch 资源设置，body 无显式 UB 分配。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP32)`；`foreach_asin<0>` | 首个动态输入 desc dtype `DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16 SIMT | `GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP16)`；`foreach_asin<1>` | dtype `DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BF16 SIMT | `GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_BF16)`；`foreach_asin<2>` | dtype `DT_BF16` | 其它 dtype `GRAPH_FAILED` |

```text
dtype == DT_FLOAT   -> symbolic template key FP32 -> foreach_asin<0> -> Process<float>
else DT_FLOAT16     -> symbolic template key FP16 -> foreach_asin<1> -> Process<half>
else DT_BF16        -> symbolic template key BF16 -> foreach_asin<2> -> Process<bfloat16_t>
else -> GRAPH_FAILED，不 launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | symbolic FP32 key | FP32、ND、0~8 维、950 | 单 entry；每个相交 tensor 一个 512-thread VF call | `foreach_asin<0>`→`Process<float>`→`ForeachAsinSimtKernel<float>` | 每线程直接 GM load/asinf/store；无 barrier/atomic/UB | 线程循环 `i<localEnd`；无 fallback |
| R-002 FP16 | symbolic FP16 key | FP16，其余同上 | 同上 | `<1>`→`Process<half>`→SIMT half | 寄存器内 half→float→half | 同上 |
| R-003 BF16 | symbolic BF16 key | BF16，其余同上 | 同上 | `<2>`→`Process<bfloat16_t>`→SIMT BF16 | 寄存器内 BF16→float→BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| DCache 预留 | `U>128*1024` | 只把剩余 local memory 交给 Kernel launch | 全部 | `GRAPH_FAILED` | `ForeachAsinTilingFunc` |
| 最小核粒度 | `perCore=max(1024,ceil(E/C))` 后按 32 元素对齐 | 控制 block 数与每个 VF call 的段长 | 全部 | 无另一路线 | 同上 |
| 静态 tensor 数 | POD offset 数组长度 257，即最多表达 256 tensor | `Process` 读取 `[t]`/`[t+1]` | 全部 | Host 未显式检查 `L<=256`；禁止假设自动 fallback | `FOREACH_ASIN_MAX_TENSOR_NUM`, Tiling loop |
| 线程数 | `THREAD_NUM=512`，`LAUNCH_BOUND(512)`，VF call `Dim3(512)` | thread `i=localStart+tid+k*threadNum` | 全部 | 无动态 selector | `ForeachAsinSimtKernel`, `Process` |
| tensor 交集 | `segStart<segEnd` | 只对本核和 tensor 的非空交集 launch VF | 全部 | 空交集 continue；空核 return | `Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAsinTilingData` | `needCoreNum` | `int32_t`/32 | 首字段；Host/Device 共用同一头文件；精确 `sizeof/alignof/offsetof` 的目标编译器结果固定仓不可见 | core | `ceil(E/perCoreElements)` 或 1；同时写 blockDim | Device 不读取；blockDim 已消费 | `int64→int32`，核数来自平台 |
| 同上 | `totalElements` | `int64_t`/64 | 声明次序第 2；精确 offset 固定仓不可见 | 元素 | 累加每项 `GetShapeSize()` | `Process` 截断 `coreEnd` 与空核 guard | 加法无溢出检查 |
| 同上 | `perCoreElements` | `int64_t`/64 | 声明次序第 3；精确 offset 固定仓不可见 | 元素/core | `align32(max(1024,ceil(E/C)))` | `coreStart/coreEnd` | E=0 时 1024 |
| 同上 | `tensorNum` | `int32_t`/32 | 声明次序第 4；精确 offset 固定仓不可见 | tensor 数 | `static_cast<int32_t>(GetInstanceNum())` | tensor 循环上界 | 无 `<=256` 或窄化检查 |
| 同上 | `tensorCumulativeOffset[257]` | `int64_t[257]` | 末字段，逻辑 payload 2056 B；精确 offset/尾 padding 固定仓不可见 | 元素前缀和 | `[0]=0`, `[i+1]=sum_{0..i}N` | tensor 区间、局部 offset | 仅容纳 256 tensor + sentinel |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | template dtype 0 | `arch35/foreach_asin_tiling_arch35.cpp::ForeachAsinTilingFunc` | `ForeachAsinTilingData`, 512 threads | `arch35/foreach_asin.cpp::foreach_asin<0>` | `Process<float>`, `ForeachAsinSimtKernel<float>`, `ComputeAsin<float>` | arch35 source→ascend950；FP32 binary entry |
| R-002 FP16 | dtype 1 | 同上 | 同上 | `<1>` | half specialization | 同 target；FP16 binary entry |
| R-003 BF16 | dtype 2 | 同上 | 同上 | `<2>` | BF16 specialization | 同 target；BF16 binary entry |

## Kernel 实现

### 公共执行流程

template entry 用 `REGISTER_TILING_DEFAULT`/`GET_TILING_DATA_WITH_STRUCT` 解析 POD，并按编译期 dtype 调用 `Process<T>`。`Process` 将 `GetBlockIdx` 映射为全局元素区间，构造 `ListTensorDesc` 解析 x/y list，遍历 `tensorNum` 个累计区间求交。每个交集发起 512-thread SIMT VF；线程 `tid` 从 `localStart+tid` 开始，以 `GetThreadNum()` 为步长遍历到 `localEnd`，每次只做一个 GM load、`ComputeAsin` 和 GM store。无 shared/dynamic UB、barrier、warp primitive、atomic、workspace 边或跨核归并。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_asin_def.cpp` | `ForeachAsin` | `DT_FLOAT`, ascend950 | arch35 Tiling |
| Host/Tiling | `foreach_asin_tiling_arch35.cpp` | `ForeachAsinTilingFunc` | `dtype==DT_FLOAT` | FP32 template key |
| Router | `arch35/foreach_asin.cpp` | `foreach_asin<dtype>` | `dtype==FOREACH_ASIN_TPL_DTYPE_FP32` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_asin<0>(xList,yList,workspace,tiling)
  -> Process<float>
    -> ListTensorDesc::GetDataPtr<float>
    -> Simt::VF_CALL<ForeachAsinSimtKernel<float>>(Dim3(512), ...)
      -> ComputeAsin<float> -> asinf
```

##### 源码等价伪码

```text
# 伪码
global = [blockIdx*perCore, min((blockIdx+1)*perCore,E))
if global.start >= E: return
for t in 0..tensorNum-1:
  seg = intersect(global, [prefix[t],prefix[t+1]))
  if empty: continue
  local = seg - prefix[t]
  VF 512 threads:
    for i=local.start+threadIdx; i<local.end; i+=threadNum:
      y[t][i] = asinf(x[t][i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `xList.GetDataPtr<float>(t)+i` | 每线程一个元素 | FP32/ND continuous | 调用者 | thread load | 调用者/stream | 无显式合并搬运 API |
| y GM | `yList.GetDataPtr<float>(t)+i` | 每线程一个元素 | FP32/ND | thread store | 下游 | 调用者/stream | 每元素唯一线程 |
| thread register | `val/result` | 2 个标量概念生命周期；物理寄存器数不可见 | FP32 | load/asinf | asinf/store | thread | 循环迭代覆盖 |
| local memory | N/A：无显式 UB/shared 分配 | Host 设置 `U-128KiB`，body 0 B 显式寻址 | N/A | Host launch | 固定仓无 consumer | Kernel | 不可作为可复用 buffer |
| workspace | N/A：body 不读取 | 16 MiB Host 声明 | N/A | executor | 无 | 调用者 | 禁止放置算法中间量 |

##### 分层内存布局

N/A：该 SIMT 节点没有 AIV `TQue/TBuf`、dynamic UB 或 shared-memory 布局；实际替代路径是线程直接 GM load→标量寄存器 `asinf`→GM store。

##### 字节账与复用顺序

```text
UB_explicit = 0 B
dynamic_ub = N/A（VF_CALL 只给出 Dim3(512)，未传 dynUBufSize）
local_memory_launch_limit = U - 128 KiB（Host 设置；body 无显式消费）
workspace_total = 16 MiB declared; 0 B read by Kernel
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar `val` | GM load | `asinf` | 下一循环迭代 | 单线程程序顺序 | store 前 result 保持有效 |
| y 元素 | 唯一 `i` 对应线程 | 下游 Kernel | 不在本 Kernel 重写 | `i=localStart+tid+k*512` 不重叠 | Host 核区间不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread loop | 无 barrier/atomic | GM load | `asinf`/store | 单元素 | FP32；`i<localEnd` 是尾保护 |
| VF calls | 同一 `Process` 程序顺序 | 前一 tensor VF | 下一 tensor VF | 各自 GM 区间 | 区间互不重叠，无跨 tensor 同步数据 |
| stream | ACLNN launch | Kernel | 后续 stream task | x/y 可见性 | executor 内部细节不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_asin_tiling_arch35.cpp` | `ForeachAsinTilingFunc` | core、prefix、key、workspace/local memory |
| Kernel entry/router | `op_kernel/arch35/foreach_asin.cpp` | `foreach_asin<0>` | FP32 compile-time route |
| Device body/helper | `op_kernel/arch35/foreach_asin_simt.h` | `Process<float>`, `ForeachAsinSimtKernel<float>`, `ComputeAsin<float>` | VF launch、thread loop、asinf |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_asin_def.cpp` | `ForeachAsin` | `DT_FLOAT16`, ascend950 | arch35 Tiling |
| Host/Tiling | `ForeachAsinTilingFunc` | dtype branch | `dtype==DT_FLOAT16` | FP16 template key |
| Router | `foreach_asin<dtype>` | constexpr branch | template dtype FP16 | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_asin<1> -> Process<half>
  -> VF_CALL<ForeachAsinSimtKernel<half>>(512 threads)
    -> ComputeAsin<half>: __half2float -> asinf -> __float2half
```

##### 源码等价伪码

```text
# 伪码；核/tensor/线程映射同 R-001
half v = x[t][i]
float f = __half2float(v)
float r = asinf(f)
y[t][i] = __float2half(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | list pointer + `i` | 每线程 1 FP16 | FP16/ND | caller/thread | thread/downstream | stream | 唯一线程写 |
| thread scalars | half input/output + FP32 `f/result` | 标量；物理寄存器不可见 | mixed | load/conversion/asinf | store | thread iteration | 程序顺序复用 |
| local/workspace | 无显式 consumer | `U-128KiB`; 16 MiB | N/A | Host | N/A | launch/caller | 不参与算法 |

##### 分层内存布局

N/A：无 AIV UB、dynamic UB 或 shared buffer；线程以寄存器标量完成 FP16→FP32→FP16。

##### 字节账与复用顺序

```text
UB_explicit = 0 B
dynamic_ub = N/A
workspace_total = 16 MiB declared; 0 B read
per-thread logical live values = one half input + two FP32 temporaries + one half output
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 scalar | `__half2float`/`asinf` | `asinf`/`__float2half` | 下一 i | 单线程顺序 | 转换结束后才覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | 无 barrier | half load | float conversion/asinf/half conversion | thread scalar | 以 FP32 `asinf` 计算；`__float2half` 舍入模式由 intrinsic 定义，固定仓未显式命名 |
| tail | `i<localEnd` | index mapping | load/store | 最后 segment | 无越界 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_asin_tiling_arch35.cpp` | FP16 dtype branch | symbolic key |
| Kernel entry/router | `arch35/foreach_asin.cpp` | `foreach_asin<1>` | `Process<half>` |
| Device body/helper | `foreach_asin_simt.h` | `ComputeAsin<half>` | FP32 升精度链 |

### Route R-003：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_asin_def.cpp` | `ForeachAsin` | `DT_BF16`, ascend950 | arch35 Tiling |
| Host/Tiling | `ForeachAsinTilingFunc` | dtype branch | `dtype==DT_BF16` | BF16 template key |
| Router | `foreach_asin<dtype>` | constexpr branch | template dtype BF16 | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_asin<2> -> Process<bfloat16_t>
  -> VF_CALL<ForeachAsinSimtKernel<bfloat16_t>>(512 threads)
    -> ComputeAsin<bfloat16_t>: __bfloat162float -> asinf -> __float2bfloat16
```

##### 源码等价伪码

```text
# 伪码；核/tensor/线程映射同 R-001
bfloat16 v = x[t][i]
float f = __bfloat162float(v)
float r = asinf(f)
y[t][i] = __float2bfloat16(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | list pointer + `i` | 每线程 1 BF16 | BF16/ND | caller/thread | thread/downstream | stream | 唯一线程写 |
| thread scalars | BF16 input/output + FP32 `f/result` | 标量；物理寄存器不可见 | mixed | load/conversion/asinf | store | thread iteration | 程序顺序复用 |
| local/workspace | 无显式 consumer | `U-128KiB`; 16 MiB | N/A | Host | N/A | launch/caller | 不参与算法 |

##### 分层内存布局

N/A：无 AIV UB、dynamic UB 或 shared buffer；线程以寄存器标量完成 BF16→FP32→BF16。

##### 字节账与复用顺序

```text
UB_explicit = 0 B
dynamic_ub = N/A
workspace_total = 16 MiB declared; 0 B read
per-thread logical live values = one BF16 input + two FP32 temporaries + one BF16 output
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 scalar | BF16 conversion/`asinf` | `asinf`/BF16 conversion | 下一 i | 单线程顺序 | 转换完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | 无 barrier | BF16 load | FP32 `asinf`/BF16 conversion | thread scalar | FP32 计算；降精度 intrinsic 的具体 round mode 固定仓未显式命名 |
| tail | `i<localEnd` | thread index | GM access | 最后 segment | 精确末尾保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_asin_tiling_arch35.cpp` | BF16 dtype branch | symbolic key |
| Kernel entry/router | `arch35/foreach_asin.cpp` | `foreach_asin<2>` | BF16 Process |
| Device body/helper | `foreach_asin_simt.h` | `ComputeAsin<bfloat16_t>` | FP32 升精度链 |

## 算子特定优化

### Trick T-001：全 list 前缀和分核并仅 launch 相交 tensor 段

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：Tiling 已生成 `tensorCumulativeOffset`，每核取得自己的全局逻辑区间。
- Golden 来源：[`foreach_asin_simt.h:104-128`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/arch35/foreach_asin_simt.h#L104-L128)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t segStart = (coreStart > tensorStart) ? coreStart : tensorStart;
int64_t segEnd = (coreEnd < tensorEnd) ? coreEnd : tensorEnd;
if (segStart >= segEnd) { continue; }
```

#### 机制与设计原因

Host 只传 257 项前缀和；Device 用区间交集把一个均衡 core 范围映射回真实 tensor，而不是为每个 tensor 固定分核。小 tensor 可共享同一 core，跨 tensor 的 core 只对非空交集发起 VF，减少空段 launch，同时保持每个 VF 内 GM 地址连续。

#### 迁移边界

- 适用：tensor 间无依赖、可视为串联元素域的逐元素 list 算子。
- 前提：前缀和不溢出、list 长度不超过 256、各 core 全局区间不重叠。
- 失效：跨 tensor 归约/通信、不同计算语义或需要 tensor 独占同步时必须改变 launch graph。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：512 线程网格步进直接 GM 计算

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：当前核与 tensor 交集非空。
- Golden 来源：[`foreach_asin_simt.h:68-73`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/arch35/foreach_asin_simt.h#L68-L73)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int64_t i = localStart + static_cast<int64_t>(Simt::GetThreadIdx());
     i < localEnd;
     i += static_cast<int64_t>(Simt::GetThreadNum())) {
```

#### 机制与设计原因

固定 512 threads 覆盖任意长度 segment，每个线程以 thread count 为步长继续处理；`i<localEnd` 同时完成尾保护。路径不物化 UB tile，也不需要 barrier/atomic，适合独立标量 `asinf`；FP16/BF16 只在单线程寄存器内升精度。

#### 迁移边界

- 适用：无邻域依赖、每个输出唯一写者的逐元素 SIMT 算子。
- 前提：GM 直接访问满足平台 SIMT 约束，线程数和 `LAUNCH_BOUND(512)` 保持一致。
- 失效：需要跨线程共享、Vector tile 复用或归约时必须引入真实 buffer/同步路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | FP32 key，首分支 | FP16 key，次分支 | BF16 key，第三分支 | Host dtype selector |
| 模板与 dtype | `Process<float>` | `Process<half>` | `Process<bfloat16_t>` | storage dtype specialization |
| 循环与分核 | 32 元素对齐 core 段；512-thread stride | 同左 | 同左 | dtype 不改变调度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅线程 FP32 标量 | half + FP32 标量 | BF16 + FP32 标量 | 无显式 UB/shared |
| 内存总占用与复用 | 显式 UB 0 | 显式 UB 0 | 显式 UB 0 | 直接 GM 路径 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM scalar + `asinf` | GM scalar + convert/asinf/convert | 同左，BF16 intrinsic | 全部 SIMT |
| 同步、精度与 tail | 无 barrier；FP32 | 无 barrier；FP32 计算 | 无 barrier；FP32 计算 | `i<localEnd` tail |
| fallback/失败边界 | 无 Device fallback | 同左 | 同左 | 非三 dtype Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持；README/API 与 ascend950 接线一致 |
| Host/Launcher | arch22 公共 Tiling | arch22 公共 Tiling | arch35 专用前缀和 POD、16 MiB workspace、local memory 设置 |
| Kernel/资源 | AIV Vector Queue/Asin | AIV Vector Queue/Asin | SIMT VF 512 threads，直接 GM，无显式 UB |

## 限制与失败边界

- `ForeachAsinTilingData` 只能容纳 256 个 tensor 的 257 个前缀 offset；Tiling 未检查 `tensorNum<=256`，且将 `uint64_t` 实例数窄化到 `int32_t`。入口复用必须补上限制，不能依赖未见框架行为。
- `totalElements` 累加和 `totalElements+maxCoreNum-1`、`+perCoreElements-1` 没有有符号溢出检查；超大 shape 行为不可推测。
- `needCoreNum` POD 字段不被 Device 读取；正确 launch 依赖 `SetBlockDim(needCoreNum)` 与 POD 的 `perCoreElements` 同步。
- 16 MiB workspace 和 `U-128 KiB` local memory 均无 Kernel 显式 consumer；不要把它们解释为算法中间 buffer，也不要擅自缩减框架保留值。
- 多个 tensor 的 VF call 位于同一 entry 循环内；固定仓只证明程序顺序，未公开 SIMT VF runtime 的内部调度/完成协议。
- ACLNN inner bridge、executor launch/cleanup、`ListTensorDesc` 内部 ABI、`asinf` 和 FP16/BF16 conversion intrinsic 的具体实现与降精度 round mode 固定仓不可见。
- 卡内 16 MiB 是 ForeachAsin Device 节点的 Tiling workspace 声明；若输入触发 `AutoContiguous`，ACLNN executor 总 workspace 的额外框架开销固定仓不可见。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_asin/docs/aclnnForeachAsin.md` | API 原型、参数/错误表 | 公开入口、语义、约束 |
| 平台/语义 | `foreach/foreach_asin/README.md` | 产品表、公式 | 950 公开支持 |
| OpDef | `foreach/foreach_asin/op_host/foreach_asin_def.cpp` | `ForeachAsin`, `OP_ADD` | dtype/layout/AutoContiguous/ascend950 |
| infer-shape | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`, `InferDataType4ForeachCommon` | 输出实例数、shape 与 dtype 推导 |
| Host/Tiling | `foreach/foreach_asin/op_host/arch35/foreach_asin_tiling_arch35.cpp` | `ForeachAsinTilingFunc`, `TilingParseForForeachAsin` | core、UB、prefix、key、workspace/local memory |
| POD | `foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_data.h` | `ForeachAsinTilingData` | ABI 字段与容量 |
| selector | `foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三 dtype template key |
| Kernel entry | `foreach/foreach_asin/op_kernel/arch35/foreach_asin.cpp` | `foreach_asin<dtype>` | compile-time router |
| Kernel body | `foreach/foreach_asin/op_kernel/arch35/foreach_asin_simt.h` | `Process`, `ForeachAsinSimtKernel`, `ComputeAsin` | core/tensor/thread 映射与数值链 |
| 构建接线 | `foreach/foreach_asin/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 Kernel source |
| 构建接线 | `foreach/foreach_asin/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_asin/op_host/config/ascend950/foreach_asin_binary.json` | 三项 `op_list` | 950 FP16/FP32/BF16 binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Host/Device 共用的 5 字段、257 前缀项 struct | 字段顺序、类型、元素单位、256 上限；目标编译器精确布局不可见 | `ForeachAsinTilingData` |
| 调用顺序 | entry→Process→逐 tensor VF→thread loop | 每个 VF 的 512 threads 与局部区间 | `foreach_asin<dtype>`, `Process` |
| workspace/内存 | 算法直接 GM；显式 UB 0 | 16 MiB framework workspace 与 128 KiB DCache 预留 | `ForeachAsinTilingFunc` |
| 同步 | 单线程程序顺序，无 barrier/atomic | 核区间与 thread 索引必须唯一覆盖 | `ForeachAsinSimtKernel` |
| 精度与尾块 | FP16/BF16 以 FP32 `asinf` 计算 | conversion intrinsic、`i<localEnd`、不越界 | `ComputeAsin`, thread loop |
| 失败边界 | Host null/dtype/UB 检查阻断 launch | >256 tensor、算术溢出无显式处理 | `ForeachAsinTilingFunc` |