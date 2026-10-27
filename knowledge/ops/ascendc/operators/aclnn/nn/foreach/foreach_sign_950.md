---
type: Operator
title: "ForeachSign 950 ACLNN 算子知识"
description: "说明 aclnnForeachSign 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/docs/aclnnForeachSign.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_host/foreach_sign_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_host/arch35/foreach_sign_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_kernel/arch35/foreach_sign_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_kernel/arch35/foreach_sign_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_kernel/arch35/foreach_sign.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_kernel/arch35/foreach_sign_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:21:40Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSign 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSignGetWorkspaceSize` + `aclnnForeachSign`，公开 ACLNN 两段式接口；固定仓未发现更高版本。 |
| 计算语义 | 对 TensorList 中每个有效元素执行 `(0<x)-(x<0)`，得到 `1/0/-1`；浮点 NaN 在当前 SIMT body 中两次比较均 false，写 0。 |
| 输入输出 | `x/out` 为 ND TensorList；FLOAT16、FLOAT32、BFLOAT16、INT8、INT32、INT64；列表最多 256 项；每项输出只写输入元素数。 |
| 执行类型 | 单 Kernel，唯一节点为 SIMT；一个 AIV block 内用 `VF_CALL` 发起 512 threads，无 Cube/AIC、MIX_CV、NDDMA 或通信节点。 |
| Kernel 路线 | R-001 统一 SIMT grid-stride，六个 dtype key 仅实例化类型不同。 |
| 核心资源 | TensorList GM 描述、寄存器标量 `val/signVal`；无 `TQue/TBuf`、shared/dynamic UB；Host 设置 local memory 为 `UB-128 KiB` 并返回 16 MiB tools workspace。 |
| 来源闭包 | source_closed：公开 API 文档/OpDef → 950 arch35 Tiling/POD/key → SIMT entry/body → ascend950 构建接线闭合；自动生成 ACLNN bridge 是框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x=[x_0,...,x_{n-1}]` 中每个 Tensor 的每个有效元素：

```text
signVal = (T(0) < x_i[j]) - (x_i[j] < T(0))
y_i[j] = static_cast<T>(signVal)
```

正数写 `1`，正/负零写 `0`，负数写 `-1`。对浮点 NaN，IEEE 比较在源码注释与表达式中均为 false，当前 950 body 写 `0`；这是实现事实，公开接口公式没有单列 NaN。每项 `y_i` 只写 `[0,tensorElements[i])`，即使调用约束允许输出 storage 元素数更大，额外区域也不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND，公开 rank 0–8 | FLOAT16/FLOAT32/BFLOAT16/INT8/INT32/INT64，列表内一致 | 支持空 Tensor和非连续输入；OpDef `AutoContiguous()`。950 Tiling 接受 `tensorNum` 0–256。 |
| `out` | TensorList；ND；对应 storage 元素数不小于输入 | 与 `x` 相同，列表内一致 | 支持空 Tensor；公开接口不支持非连续输出；Kernel 不写超出输入元素数的尾区。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 950 Tiling 固定返回 `16*1024*1024` B tools workspace。 |
| `executor` | Host 执行器输出 | `aclOpExecutor**` | 第一段返回；生成实现固定仓不可见。 |
| `workspace` | Device 地址 | N/A | 第二段传给 Kernel；当前 body 未读取。 |
| `stream` | Runtime stream | `aclrtStream` | 单节点异步提交的 ordering/可见性载体。 |

### 入口与版本

公开文档仅有无版本后缀的 `aclnnForeachSignGetWorkspaceSize`/`aclnnForeachSign`，固定仓没有更高公开版本。文档产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 把 `ascend950` 映射到 `arch35`，Kernel CMake 指定 `arch35/foreach_sign.cpp`，三层证据一致。生成的 ACLNN header/inner bridge 与 executor 实现不在固定仓，按框架边界记录；算子自有选择、POD、Kernel 和构建均可追溯。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开第一段接口要求 `x/out` 非空、dtype 属于六种集合且对应相同、列表内 dtype 一致、rank≤8、输出 storage 元素数不小于输入；错误映射见公开文档。
2. OpDef 将输入/输出声明为动态 ND TensorList 并启用 `AutoContiguous()`。输入非连续的实际 materialize 序列与输出检查在 generated bridge 中，固定仓不可见。
3. 公共 infer-shape/dtype 要求输入实例数等于输出数并逐项复制；950 自有 Tiling 直接读取 dynamic input instance count 和 storage shape。
4. `GetPlatformInfo` 获取 `GetCoreNumAiv()` 与 UB 大小，核数≤0或 UB=0 失败。
5. `ForeachSignTilingFunc` 检查 `tensorNum` 在 `[0,256]`；非空列表从第 0 项取 `xDtype`，空列表默认 FLOAT；遍历所有输入写 `tensorElements[i]` 和最大元素数。它未再次逐项检查 dtype/output shape，依赖公开入口/OpDef 契约。

### 参数、Workspace 与生命周期

记 950 AIV 核数 `C`、最大单 Tensor 元素数 `M=max_i n_i`、UB 字节数 `U`：

```text
perCoreElements = align_up_32(max(1024, ceil(M/C)))  # 元素，不是字节
needCoreNum = max(1, min(C, ceil(M/perCoreElements)))
blockDim = needCoreNum
localMemorySize = uint32(U - 128*1024)
workspaceSize = 16*1024*1024 B
```

POD 的 `needCoreNum` 同时写入，但 Device body 不读取；真实 grid 来自 `SetBlockDim(needCoreNum)`。workspace 由调用者在第一、二段之间分配并保持到 stream 完成；当前 entry 保留参数但不使用。`localMemorySize` 是 launch 元数据，当前 SIMT body 没有显式 dynamic/shared UB 分配。

### 启动图与失败传播

```text
aclnnForeachSignGetWorkspaceSize(x,out)
  -> generated ACLNN/OpDef boundary
  -> ForeachSignTilingFunc
     -> GetPlatformInfo
     -> dtype key + ForeachSignTilingData + blockDim/localMemory/workspace
  -> workspaceSize=16 MiB, executor

aclnnForeachSign(workspace,16MiB,executor,stream)
  -> K0: foreach_sign<schMode><<<needCoreNum>>>(xDesc,yDesc,workspace,tiling)
     -> NsForeachSign::Process<T>
        -> per Tensor: VF_CALL<OpForeachSignSimt<T>>(512 threads)
```

| 节点/边 | producer / consumer | 交接介质 | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| K0 | executor / templated `foreach_sign` | x/y ListTensorDesc、2056 B POD、blockDim | 单 stream 单节点 | caller 持有 x/y；runtime 持有 tiling launch data | Tiling/launch 失败阻断唯一节点。 |
| workspace 参数 | first-stage / K0 | 16 MiB GM，当前 K0 不读写 | 无算法依赖 | caller 至 stream 完成 | 分配失败在 launch 前；无 producer/consumer edge。 |
| K0 输出 | SIMT threads / caller | 每项 y GM `[0,n_i)` | stream 完成后可见 | caller | 无后续算子自有节点。 |

## Route 与参数设计

### 分核与资源计算

Host 不按总元素数分核，而按最大单 Tensor `M` 选择 grid。K0 的所有 blocks 都依次遍历同一个 TensorList；对每个非空 Tensor，thread 的全局线性 id 为：

```text
tid = blockIdx * 512 + threadIdx
gridStride = 512 * blockDim
for idx = tid; idx < tensorElements[i]; idx += gridStride:
    y_i[idx] = sign(x_i[idx])
```

`perCoreElements` 只用于估算 `blockDim`，不写入 POD；32 是元素对齐单位而非 DMA 字节对齐，因为 Device 不使用 DMA/TQue。若 `M=0`，`needCoreNum=1`。Tiling 将 local memory 设为 `U-128 KiB` 以扣除 DCache，但 body 没有对应动态 UB consumer。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 统一 SIMT grid-stride | schMode 0 FLOAT、1 FP16、2 INT8、3 INT32、4 INT64、5 BF16 | `tensorNum==0` 时默认 FLOAT key 0；否则按第 0 个输入 desc 的 dtype 依次 `if/else if` 选择六 key | 其它 dtype `GRAPH_FAILED`；模板列表外 key 无 body。 |

```text
tensorNum==0 -> xDtype=FLOAT -> key 0 -> Process<float>（循环 0 次）
else FLOAT   -> 0 -> Process<float>
else FP16    -> 1 -> Process<half>
else INT8    -> 2 -> Process<int8_t>
else INT32   -> 3 -> Process<int32_t>
else INT64   -> 4 -> Process<int64_t>
else BF16    -> 5 -> Process<bfloat16_t>
else         -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 统一 SIMT | 0/1/2/3/4/5 | 950；`0<=tensorNum<=256`；dtype 为六种之一；各 shape size 可由 int64 表示 | 单 K0 SIMT；每 Tensor 一次 device-side `VF_CALL` | `foreach_sign<schMode>` → `Process<T>` → `OpForeachSignSimt<T>` | GM 直接 load/store；寄存器 `val/signVal`；无 Queue、shared/dynamic UB、barrier、atomic | `idx<count` 是逐线程 tail guard；空 Tensor continue；非法 dtype Host 失败。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数量 | `0<=tensorNum<=256` | POD `tensorElements[256]` | R-001 | `GRAPH_FAILED` | `ForeachSignTilingFunc` |
| 平台核数 | `GetCoreNumAiv()>0` | SIMT grid 至少一个 block | R-001 | `GRAPH_FAILED` | `GetPlatformInfo` |
| UB/DCache | `ubSize!=0`；实际表达式还隐含 `ubSize>=128KiB` | `uint32(ubSize-128KiB)` 避免无符号下溢 | R-001 launch | 固定 Host 未显式检查该下界；依赖 950 平台资源 | `GetPlatformInfo`、`SetLocalMemorySize` |
| 最小每核工作量 | `perCore=max(1024,ceil(M/C))` 再元素 32 对齐 | 避免为小 Tensor 启动全部 cores | R-001 | 产生较小 blockDim，无 fallback | `ForeachSignTilingFunc` |
| dtype 一致性 | 950 Tiling 只读取第 0 项 dtype | `Process<T>` 用同一 T 解释所有 ListTensor 地址 | R-001 | 必须由公开 ACLNN/OpDef 契约保证；绕过入口不安全 | `ForeachSignTilingFunc`、`Process<T>` |
| 线程尾部 | `idx<elemCount`，步长 `512*blockDim` | 避免越界且覆盖任意长度 | R-001 | 自然结束 | `OpForeachSignSimt` |
| 空 Tensor | `count==0` | 不取 Tensor data ptr、不发 VF_CALL | R-001 | continue | `Process<T>` |

### POD/ABI

`ForeachSignTilingData` 是显式标准字段 struct：两个 `int32_t` 后接 256 个 `int64_t`。按源码类型的自然 ABI，offset 为 0/4/8，总大小 `8+256*8=2056` B、alignment 8 B；entry 以同一 struct 的注册宏解析。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSignTilingData` | `tensorNum` | `int32_t`/32 | offset 0；struct sizeof 2056，alignof 8 | Tensor 个数 | `tiling->tensorNum=tensorNum` | `Process` Tensor 循环上界 | `[0,256]`。 |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | block 数 | `tiling->needCoreNum=needCoreNum` | Device 未读取；launch 另由 `SetBlockDim` 消费同值 | inactive consumer 字段，必须保持 ABI。 |
| 同上 | `tensorElements` | `int64_t[256]`/每项 64 | offset 8；末尾 offset 2048 | 元素/Tensor | `tiling->tensorElements[i]=shape.GetShapeSize()` | `Process: count=...` | 活动 `[0,tensorNum)`；其余经 memset 为 0 且不读。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 统一 SIMT | schMode 0–5 | `op_host/arch35/foreach_sign_tiling.cpp::ForeachSignTilingFunc` | `ForeachSignTilingData`、blockDim/localMemory | `op_kernel/arch35/foreach_sign.cpp::foreach_sign<schMode>` | `foreach_sign_simt.h::Process<T>/OpForeachSignSimt<T>` | Host CMake ascend950→arch35；Kernel CMake 指定 arch35 source |

## Kernel 实现

### 公共执行流程

唯一 K0 解析 POD 后由编译期 `schMode` 选择 T。`Process<T>` 构造输入/输出 `ListTensorDesc`；各 block 都运行同一个 Tensor 循环，空 Tensor 直接跳过，非空 Tensor 取对应 GM 指针并调用 512-thread VF。VF 中每个线程直接从 GM 读一个 T 到寄存器，做两次比较和一次整数减法，再 cast 回 T 写 GM；通过 grid-stride 继续覆盖后续元素。无算子内 workspace 边、无 UB staging、无 atomic/barrier。

### Route R-001：统一 SIMT grid-stride

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `op_host/foreach_sign_def.cpp` | `ops::ForeachSign` | 六 dtype、ND、ascend950 | 950 Host |
| Tiling | `op_host/arch35/foreach_sign_tiling.cpp` | `ForeachSignTilingFunc` | 0–256 Tensor；dtype if/else；max shape 分核 | key 0–5、blockDim、POD |
| Router | `op_kernel/arch35/foreach_sign.cpp` | `foreach_sign<schMode>` | 六个 `if constexpr` | 对应 `Process<T>` |
| Body | `op_kernel/arch35/foreach_sign_simt.h` | `Process/OpForeachSignSimt` | count 0 continue；`idx<count` | GM 直接 sign |

#### Device 数据流

##### 函数级调用链

```text
foreach_sign<schMode>(x,y,workspace,tiling)
  -> GET_TILING_DATA_WITH_STRUCT(ForeachSignTilingData)
  -> NsForeachSign::Process<T>(x,y,&tilingData)
     -> ListTensorDesc xList/yList
     -> for i in [0,tensorNum):
        -> GetDataPtr<T>(i)
        -> Simt::VF_CALL<OpForeachSignSimt<T>>(Dim3(512),xPtr,yPtr,count)
           -> grid-stride direct GM load/compare/subtract/store
```

##### 源码等价伪码

```text
# 伪码
for tensor i = 0 .. tensorNum-1:             # 每个 block 执行同一循环
  count = tensorElements[i]
  if count == 0: continue
  xPtr = xList.GetDataPtr<T>(i)
  yPtr = yList.GetDataPtr<T>(i)
  VF_CALL threadsPerBlock=512:
    idx = blockIdx*512 + threadIdx
    while idx < count:
      val = xPtr[idx]
      signVal = (T(0)<val) - (val<T(0))
      yPtr[idx] = T(signVal)
      idx += 512*blockDim
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x TensorList descriptor | `x` GM descriptor | `tensorNum` address entries | ListTensorDesc | caller/framework | `GetDataPtr<T>(i)` | caller至 stream 完成 | 描述格式由 AscendC interface 定义。 |
| input GM | `xPtr+idx` | 1 T/iteration | T/ND | caller | thread register `val` | caller | 无显式对齐要求；逐元素。 |
| thread register | 当前 thread | `val:T`, `signVal:int` | scalar | GM load/compare | GM store | 单线程单迭代 | 下一 stride 迭代覆盖。寄存器字节数固定仓编译产物不可见。 |
| output GM | `yPtr+idx` | 1 T/iteration | T/ND | SIMT thread | caller | caller | `idx<count` 防越界。 |
| workspace | Kernel 参数 | 16 MiB caller allocation | opaque GM | first stage/caller | 当前 body 无 consumer | caller | 不别名算法数据。 |

##### 分层内存布局

N/A：R-001 不分配 AIV `TQue/TBuf`、SIMT shared/dynamic UB 或 NDDMA buffer；数据路径为 GM→线程寄存器→GM。Host 仍设置 `localMemorySize=U-128 KiB`，但固定 body 无显式 consumer。

##### 字节账与复用顺序

```text
AIV Queue/TBuf UB_total = 0 B
SIMT dynamic/shared UB = 0 B（源码无分配）
launch localMemorySize = uint32(U - 128 KiB)
register_bytes = 固定仓不可见；生命周期为单线程单迭代
workspace_total = 16 MiB（当前 K0 不读写）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar `val/signVal` | 当前 idx 的 GM load/比较 | 当前 idx 的 output store | 下一 grid-stride idx | 单线程程序顺序 | store 完成后才进入下一迭代。 |
| GM output element | 唯一映射 thread | caller | 无算子内覆盖 | idx 序列按 `tid+k*gridStride` 互不相交 | blockDim、threadNum 对所有线程一致。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| per-thread compute | SIMT 程序顺序 | GM load | compare/sub/store | 当前 element | 比较在 T 上执行，减法为 int，结果 cast 回 T。 |
| 跨线程 | N/A，无 barrier/atomic | 各 thread | 各自独占 output idx | output GM | grid-stride residue class 保证不重叠。 |
| tail | `idx<count` | thread index | load/store guard | Tensor 尾部 | 无 padding lane、无越界访问。 |
| Tensor 间 | `Process` for-loop + VF_CALL | Tensor i VF | Tensor i+1 VF_CALL | TensorList 描述/GM | 精确 VF_CALL 完成语义由 AscendC SIMT primitive 提供；body 无显式 barrier。 |
| 浮点特殊值 | 两次 `<` 比较 | val | `signVal` | scalar | ±0→0；NaN→0；±Inf→±1。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_sign/op_host/arch35/foreach_sign_tiling.cpp` | `GetPlatformInfo/ForeachSignTilingFunc` | 核数、local memory、workspace、POD、key |
| Kernel entry/router | `foreach_sign/op_kernel/arch35/foreach_sign.cpp` | `foreach_sign<schMode>` | 六 key→T |
| Device body/helper | `foreach_sign/op_kernel/arch35/foreach_sign_simt.h` | `Process<T>/OpForeachSignSimt<T>` | ListTensor 迭代、512 threads、grid-stride、tail |

## 算子特定优化

### Trick T-001：Device 侧遍历 TensorList，并用统一 grid-stride VF 覆盖每个 Tensor

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：950 arch35；`count>0`；Host 以最大 Tensor 元素数计算 `blockDim`。
- Golden 来源：[`foreach/foreach_sign/op_kernel/arch35/foreach_sign_simt.h:45-52`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_kernel/arch35/foreach_sign_simt.h#L45-L52)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t idx = static_cast<uint64_t>(
         Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
     idx < static_cast<uint64_t>(elemCount);
     idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
    T val = xPtr[idx];
    int signVal = (T(0) < val) - (val < T(0));
    yPtr[idx] = static_cast<T>(signVal);
}
```

#### 机制与设计原因

Host 只启动一个 Kernel；`Process` 在 Device 侧逐项解析 TensorList，同一个 512-thread VF 以 `blockIdx*threadNum+threadIdx` 为起点、`threadNum*blockNum` 为步长覆盖任意长度。这样无需为 TensorList 每一项建立独立 Host launch，也无需把不同 Tensor 搬到连续 workspace。比较差公式对六 dtype 共用一个模板 body；dtype key 只决定 T。

#### 迁移边界

- 适用：TensorList 各项独立、逐元素、没有跨 Tensor 归约或状态的轻量计算。
- 前提：所有列表项 dtype 一致；ListTensorDesc 地址可在 Device 解析；各输出元素只写一次；最大 Tensor 能用同一 grid 合理覆盖。
- 失效：各 Tensor 需要不同 dtype/属性、跨 Tensor 同步、共享中间状态或 per-Tensor 独立 workspace 时，必须改 launch graph/参数设计。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：以最小 1024 元素/核控制小 Tensor 的 blockDim

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：所有 shape；当 `ceil(M/C)<1024` 时最小阈值生效，随后按 32 元素对齐。
- Golden 来源：[`foreach/foreach_sign/op_host/arch35/foreach_sign_tiling.cpp:84-88`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sign/op_host/arch35/foreach_sign_tiling.cpp#L84-L88)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t perCoreElements = std::max(MIN_PER_CORE_ELEMENTS, Ops::Base::CeilDiv(maxTensorElements, maxCoreNum));
perCoreElements = Ops::Base::CeilAlign(perCoreElements, ALIGN_SIZE);
int32_t needCoreNum = static_cast<int32_t>(
    std::min(maxCoreNum, Ops::Base::CeilDiv(maxTensorElements, perCoreElements)));
needCoreNum = std::max(1, needCoreNum);
```

#### 机制与设计原因

以最大 Tensor 决定全列表共享 grid，且每个 active block 目标至少覆盖约 1024 元素；小 Tensor 因此不启动全部平台核，大 Tensor则最多使用所有 AIV cores。32 元素对齐稳定每核估算，真正 tail 仍由 SIMT `idx<count` 处理。源码能证明减少小问题的 active block 数，但未提供性能百分比。

#### 迁移边界

- 适用：所有列表项复用同一 grid、每元素成本近似一致的 SIMT unary 算子。
- 前提：最大 Tensor 是合理的 grid 决策代表；阈值和 32 元素对齐经目标平台验证。
- 失效：各 Tensor 计算成本差异大或需要按总工作量并行时，应重新设计 producer，不能直接套用常量。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 统一 SIMT | 设计选择依据 |
|---|---|---|
| 命中条件/优先级 | 六 dtype key 0–5；无 shape 分支 | dtype 只改变模板 T，执行结构一致。 |
| 模板与 dtype | `Process<T>/OpForeachSignSimt<T>` | 比较差公式对整数/浮点通用。 |
| 循环与分核 | 最大 Tensor 决定 blockDim；每 block 遍历列表；线程 grid-stride | 单 launch 复用 grid。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅线程标量寄存器；无 Queue/dynamic UB/window | GM 直接访问。 |
| 内存总占用与复用 | 显式 UB 0；workspace 16 MiB 未消费 | tools/runtime 预留不等于算法中间量。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `VF_CALL(Dim3(512))` + GM scalar load/store | 唯一执行族 SIMT。 |
| 同步、精度与 tail | 无 barrier/atomic；T 比较，int 差，cast 回 T；`idx<count` | 每 thread 独占 idx。 |
| fallback/失败边界 | dtype 非六种或资源信息非法则 Host 失败；unknown template 无 body | 正常入口不产生未知 key。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，由无后缀卡描述 | 支持，由无后缀卡描述 | 支持；公开表、OpDef、Host/Kernel 构建一致。 |
| Host/Launcher | arch22 公共 Tiling；总元素分核；32 B workspace | 同 A2 | arch35 独立 Tiling；最大 Tensor 分核；16 MiB workspace；local memory `U-128KiB`。 |
| Kernel/资源 | AIV Vector Queue/chunk/cast specialization | 同 A2 | SIMT 512 threads；GM→register→GM，无 Queue。 |

## 限制与失败边界

- `tensorNum` 由 950 Tiling 显式限制为 0–256；POD 数组也固定 256。单个 shape size 和后续 index 采用 `int64_t/uint64_t`。
- Tiling 只从第一个非空列表项（准确地说 `tensorNum>0` 时第 0 项）选择 dtype，未逐项验证列表 dtype 或 output shape；安全复用必须保留公开 ACLNN/OpDef 的一致性校验。
- `SetLocalMemorySize(uint32_t(ubSize-128KiB))` 未显式检查 `ubSize>=128KiB`；固定平台资源应满足，否则有下溢风险。
- `needCoreNum` POD 字段未被 Device body消费；blockDim 必须与 Host 计算值同步设置，不能只改 POD。
- Kernel entry 没有 unknown `schMode` 的显式失败分支；正常 Host 只生成 0–5。
- generated ACLNN bridge/executor、ListTensorDesc 物理 ABI、VF_CALL 内部 completion 机制、寄存器实际字节数和 16 MiB tools workspace 的 runtime 内部用途固定仓不可见。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sign/docs/aclnnForeachSign.md` | 两段式原型、参数/错误表 | API、平台、dtype/rank/连续性 |
| 入口实现 | `foreach/foreach_sign/op_host/foreach_sign_def.cpp` | `ops::ForeachSign` | OpDef、AutoContiguous、ascend950 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` | shape/dtype 输出推导 |
| Host/Tiling/Launcher | `foreach/foreach_sign/op_host/arch35/foreach_sign_tiling.cpp` | `GetPlatformInfo/ForeachSignTilingFunc` | 校验、分核、key、POD、workspace、local memory |
| Host/Tiling/Launcher | `foreach/foreach_sign/op_kernel/arch35/foreach_sign_tiling_data.h` | `ForeachSignTilingData` | 显式 POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_sign/op_kernel/arch35/foreach_sign_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 六个 schMode 实例 |
| Kernel entry/body | `foreach/foreach_sign/op_kernel/arch35/foreach_sign.cpp` | `foreach_sign<schMode>` | POD 解析与 dtype router |
| Kernel entry/body | `foreach/foreach_sign/op_kernel/arch35/foreach_sign_simt.h` | `Process/OpForeachSignSimt` | ListTensor、VF、线程映射、特殊值、tail |
| 构建接线 | `foreach/foreach_sign/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_sign/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_sign.cpp` | ascend950 Kernel source |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2056 B POD 提供 Tensor 数与逐项元素数 | 两个 int32 顺序、256 个 int64、inactive `needCoreNum` 仍须保留 | `ForeachSignTilingData` |
| 调用顺序 | 单 K0；Device 逐 Tensor VF_CALL | 空 Tensor 必须在 GetDataPtr 前 continue；六 key/T 对应 | `foreach_sign`、`Process<T>` |
| workspace/内存 | 算法无 UB staging；workspace 不消费 | 16 MiB API返回、local memory launch 值与 GM owner/lifetime | `ForeachSignTilingFunc` |
| 同步 | 每线程程序序；无跨线程冲突 | gridStride 必须等于 threadNum*blockNum，输出索引唯一 | `OpForeachSignSimt` |
| 精度与尾块 | T 比较、int 差、cast回T；NaN→0；idx guard | 不得用无界 load/store替代 tail guard | `OpForeachSignSimt` |
| 失败边界 | Host 拒绝 >256、非法 dtype/核数/UB=0 | 不得绕过列表 dtype 一致性契约或 UB-128KiB 隐含下界 | `GetPlatformInfo/ForeachSignTilingFunc` |