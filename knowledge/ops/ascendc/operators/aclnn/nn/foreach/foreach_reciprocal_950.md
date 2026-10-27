---
type: Operator
title: "ForeachReciprocal 950 ACLNN 算子知识"
description: "说明 aclnnForeachReciprocal 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/docs/aclnnForeachReciprocal.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_host/foreach_reciprocal_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_host/config/ascend950/foreach_reciprocal_binary.json
    role: implementation
created_at: "2026-08-25T15:59:54Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachReciprocal 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachReciprocalGetWorkspaceSize` + `aclnnForeachReciprocal`，公开 ACLNN、两段式、无编号版本 |
| 计算语义 | 对 TensorList 中每个有效元素执行 `y_i[j]=1/x_i[j]` |
| 输入输出 | 动态 TensorList；ND；FP16/FP32/BF16；输入输出与列表内 dtype 一致 |
| 执行类型 | 单节点 `SIMT`；每个 AIV block 内固定 512 threads；没有可达 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 FP16、R-002 FP32、R-003 BF16，均为同一 grid-stride body 的 dtype 实例 |
| 核心资源 | GM TensorList descriptor 与数据；每线程标量；launch local memory=`UB-128 KiB`；workspace=16 MiB，但 body 不读 workspace |
| 来源闭包 | `source_closed`：公开 API 文档→生成式 ACLNN/OpDef 边界→arch35 Tiling/POD/key→SIMT entry/body→ascend950 构建与 binary config 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入 `x=[x_0,...,x_(n-1)]`，输出同位置列表 `y`。对第 `i` 个张量的每个 storage 元素 `j`，`ComputeReciprocal<T>` 先执行 `float fval=static_cast<float>(x_i[j])`，再算 `1.0f/fval`，最后 `static_cast<T>` 写回。因此 FP16/BF16 都以 FP32 中间精度做除法，FP32 Route 的两次类型转换为同类型。源码不对零、无穷或 NaN 做显式分支。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND、0–8 维 | FP16/FP32/BF16，列表内一致 | 支持空 Tensor；可非连续输入经 `AutoContiguous()` 进入框架连续化边界 |
| `out` | TensorList；每项 ND、0–8 维 | 与 `x` 一致且列表内一致 | 不支持非连续输出；同位置输出 storage 元素数不得小于输入；图模式公共 infer-shape 将输出 shape/dtype设为输入 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 950 Tiling 为 ForeachReciprocal 节点请求 16 MiB `ASCENDC_TOOLS_WORKSPACE`；body 不读取 workspace 参数 |
| `executor` | 执行器输出/输入 | `aclOpExecutor**` | 第一段创建、第二段消费；内部所有权和附加连续化 workspace 固定仓不可见 |
| `stream` | 第二段输入 | `aclrtStream` | 指定 executor 内节点的异步执行顺序 |

### 入口与版本

固定仓只公开无编号的两段式 `aclnnForeachReciprocal`，故它是 950 实际支持的最新公开版本。API 文档明确支持 Ascend 950PR/950DT；OpDef `AddConfig("ascend950")`、Host CMake 的 `ascend950→arch35`、Kernel CMake 的 ascend950 source 选择和 binary config 的三 dtype 项互相闭合。ACLNN inner bridge 没有落盘，公开错误语义、`AutoContiguous()` 和 executor 组装作为生成框架边界，不推测其源码细节。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口检查 `x/out` 非空、dtype 受支持且输入输出一致、列表内 dtype 一致、维数不超过 8，以及输出容量。OpDef 为输入和输出声明 `AutoContiguous()`。进入 `ForeachReciprocalTilingFunc` 后，`GetPlatformInfo` 获取 AIV core 数与 UB 并要求二者非零；读取动态输入实例数并在转换为 `int32_t` 后要求 `0<=tensorNum<=256`。非空列表从第一个输入描述取 dtype，随后逐项读取 storage shape 元素数并求 `maxTensorElements`。950 Tiling 本身不读取输出描述，也不复查列表内 dtype；这些属于前述 API/OpDef/公共 infer-shape 边界。

### 参数、Workspace 与生命周期

令 `M=max_i numel(x_i)`、`Cmax=GetCoreNumAiv()`：`perCoreElements=align_up(max(1024,ceil(M/Cmax)),32)`，`needCoreNum=max(1,min(Cmax,ceil(M/perCoreElements)))`。其中 `perCoreElements` 只用于决定 grid，不写入 POD；Device 直接通过 `Simt::GetBlockNum()` 取得 launch 的 `needCoreNum`。Host 写 `tensorNum`、`needCoreNum` 和每个 Tensor 的元素数，设置 `blockDim=needCoreNum`、local memory=`uint32_t(UB-128 KiB)`、workspace[0]=16 MiB。POD、descriptor、输入输出和 workspace 由 executor/runtime 保持到 stream 上 Kernel 完成。

### 启动图与失败传播

单节点图：executor 在给定 stream 上启动模板 entry `foreach_reciprocal<schMode>(xList,yList,workspace,tiling)`；grid 为 `needCoreNum` 个 AIV block。每个 block 的 `Process<T>` 按 TensorList 顺序发起一次 `VF_CALL`（空 Tensor 跳过），每次 VF 固定 512 threads；所有 blocks/threads 以 grid stride 共同覆盖该 Tensor。无跨 Kernel 边和用户中间数据；16 MiB workspace 是节点级 runtime/tools 预留，entry/body 不解引用。平台、descriptor、shape、POD、workspace 指针、dtype 选择任一检查失败会返回 `GRAPH_FAILED` 并阻断 launch；body 没有错误回传。

## Route 与参数设计

### 分核与资源计算

Host 以列表中最大 Tensor 决定统一 block 数，避免为每个 Tensor 保存分核表。对任一非空 Tensor `i`，线程初始索引 `idx=blockIdx*512+threadIdx`，步长 `stride=512*blockNum`，循环至 `idx<tensorElements[i]`。较小 Tensor 上部分 block/thread 的初始 `idx` 已越界，循环自然为空；最大 Tensor 约获得不小于 1024 元素/core 的 Host 目标（最后由 32 元素对齐和上界取整修正）。没有 tile、former/tail core 或 UB 搬运；元素级 `idx<count` 就是 tail guard。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 SIMT | `schMode=0` / `Process<half>` | 非空列表首 Tensor dtype=`DT_FLOAT16` | 检查 FP32 |
| 2 | R-002 FP32 SIMT | `schMode=1` / `Process<float>` | 首 Tensor dtype=`DT_FLOAT` | 检查 BF16 |
| 3 | R-003 BF16 SIMT | `schMode=2` / `Process<bfloat16_t>` | 首 Tensor dtype=`DT_BF16` | 其它 dtype `GRAPH_FAILED`；空列表用默认 FP32 key 但 body 无循环 |

```text
tensorNum==0 -> xDtype defaults FP32 -> schMode 1 -> Process<float> -> zero Tensor loops
else dtype==FP16 -> schMode 0 -> Process<half>
else dtype==FP32 -> schMode 1 -> Process<float>
else dtype==BF16 -> schMode 2 -> Process<bfloat16_t>
else -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 SIMT | schMode 0 | 950；N≤256；FP16；公开校验通过 | 单 SIMT 节点；每 Tensor 一次内部 VF call | `Process<half>`→`OpForeachReciprocalSimt<half>`→`ComputeReciprocal<half>` | GM 标量 load/store；线程私有 float 中间量；无显式 UB/barrier/atomic | grid-stride `idx<count`；空 Tensor skip；无 fallback |
| R-002 FP32 SIMT | schMode 1 | 950；N=0 或非空 FP32；其余同上 | 同上 | `Process<float>`→SIMT body | 同上；static_cast<float> 为同类型 | 同一 tail；无 fallback |
| R-003 BF16 SIMT | schMode 2 | 950；N≤256；BF16；公开校验通过 | 同上 | `Process<bfloat16_t>`→SIMT body | GM BF16、线程私有 FP32 除法、GM BF16 | 同一 tail；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 宽度 | `0<=static_cast<int32_t>(GetInstanceNum())<=256` | POD 仅有 `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachReciprocalTilingFunc` |
| 每核最低元素目标 | `perCore=max(1024,ceil(M/Cmax))` | 控制 block 数，避免最大 Tensor 上启动过多 blocks | 全部 | 非 fallback，只改变 grid | `ForeachReciprocalTilingFunc` |
| 32 元素 grid 规划对齐 | `perCore=ceil_align(perCore,32)` | 仅参与 core 数计算；Device 不做块 DMA | 全部 | 非 fallback | 同上 |
| DCache/local memory | `localMemory=uint32_t(UB-128KiB)` | Host 为 arch35 launch 保留 128 KiB DCache | 全部 | 源码只检查 `UB!=0`，未检查 `UB>=128KiB` 或 uint32 窄化 | `GetPlatformInfo`、`ForeachReciprocalTilingFunc` |
| SIMT threads | `THREAD_NUM=512` | VF launch bound 与 grid-stride初始/步长都依赖 512 | 全部 | 无其它线程配置 | `OpForeachReciprocalSimt`、`Process` |
| 空 Tensor | `count==0` | 不取得 data pointer、不发起 VF | 全部 | continue | `Process` |
| dtype 一致性 | Tiling 只读取第 0 项 dtype | entry 对整个列表按同一 `T` 解引用 | 全部 | 必须由公开 API/OpDef 阻断不一致列表 | `ForeachReciprocalTilingFunc`、`Process<T>` |

### POD/ABI

`ForeachReciprocalTilingData` 是自然布局普通结构：两个 32-bit 字段后接 256 个 64-bit 元素数，故 `alignof=8`、`sizeof=2056 B`；Host 与 Device 包含同一头文件。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachReciprocalTilingData` | `tensorNum` | `int32_t`/32 | offset 0；结构 alignof 8、sizeof 2056 | Tensor 个数 | `tensorNum=static_cast<int32_t>(GetInstanceNum())` | `Process` 的 Tensor 循环上界 | 0..256 |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | core/block 数 | Host 分核公式 | Device body 未直接读取；硬件 blockNum 来自 `SetBlockDim` | ≥1；POD 字段为冗余镜像 |
| 同上 | `tensorElements` | `int64_t[256]`/每项 64 | offset 8；数组 2048 B | 元素/Tensor | 每项 `GetStorageShape().GetShapeSize()` | `Process` 读取 `tensorElements[i]` | `[N,256)` 经 `memset_s` 为 0 且 inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | schMode 0 | `ForeachReciprocalTilingFunc` | `ForeachReciprocalTilingData`、blockDim、localMemory、16 MiB WS | `foreach_reciprocal<0>` | `Process<half>`→`OpForeachReciprocalSimt<half>`→`ComputeReciprocal<half>` | ascend950 Host/Kernel CMake + binary JSON |
| R-002 FP32 | schMode 1 | 同上 | 同上 | `foreach_reciprocal<1>` | `Process<float>`→SIMT→`ComputeReciprocal<float>` | 同上 |
| R-003 BF16 | schMode 2 | 同上 | 同上 | `foreach_reciprocal<2>` | `Process<bfloat16_t>`→SIMT→`ComputeReciprocal<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

三条 Route 共用一个单 Kernel graph。模板 entry 用编译期 `schMode` 选择 `T`；`Process<T>` 构造输入/输出 `ListTensorDesc`，依次读取每个 Tensor 的 `count`，空 Tensor 直接跳过，非空 Tensor 取得 typed GM pointer 并发起 512-thread VF。SIMT 线程按 `blockIdx*threadNum+threadIdx` 起步、以 `threadNum*blockNum` 递增，GM 标量读→FP32 除法→GM 标量写。不同线程的索引同余类不重叠，无 atomic/barrier；不同 Tensor 的 VF call 按单个 `Process` 的程序序发起，且地址不重叠。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `ForeachReciprocalTilingFunc` | `xDtype==DT_FLOAT16` | `GET_TPL_TILING_KEY(mode0)` |
| Router | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<schMode>` | `schMode==0` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_reciprocal<0>(xList,yList,workspace,tiling)
  -> Process<half>
    -> for each non-empty Tensor: VF_CALL<OpForeachReciprocalSimt<half>>(threads=512)
      -> ComputeReciprocal<half> -> GM store
```

##### 源码等价伪码

```text
# 伪码
for tensor i in [0,N):
  count=elements[i]; if count==0: continue
  xPtr=xList.data<half>(i); yPtr=yList.data<half>(i)
  for idx=blockIdx*512+threadIdx; idx<count; idx+=512*blockNum:
    f=float(xPtr[idx]); yPtr[idx]=half(1.0f/f)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input descriptor/data | `ListTensorDesc::GetDataPtr<half>(i)+idx` | 1 element/thread iteration | FP16 ND storage | caller/framework | SIMT thread | stream 完成前 | 无 tile/padding |
| thread scalar | 编译器寄存器位置固定仓不可见 | `val/fval/result` 单迭代 | half→float→half | 当前线程 | 当前线程/store | 单循环迭代 | 下一 idx 覆盖 |
| GM output | `GetDataPtr<half>(i)+idx` | 1 element/thread iteration | FP16 ND storage | SIMT thread | downstream | caller | `(block,thread,iteration)` 唯一写者 |
| launch local memory | `SetLocalMemorySize(uint32(UB-128KiB))` | `UB-128KiB` | runtime local-memory resource | Host/runtime | body 无显式地址消费 | launch | 不是可复用的用户 buffer |
| workspace | 参数未读取 | 16 MiB | opaque tools reserve | runtime | 无算子 body consumer | executor | 无 slice 数据协议 |

##### 分层内存布局

N/A：SIMT body 未创建 AIV `TQue/TBuf`，`VF_CALL` 未传 dynamic UB；真实数据路径是 GM 标量 load/store。Host 仍设置 launch local-memory size 为 `UB-128 KiB`，但固定 body 没有该空间的地址布局。

##### 字节账与复用顺序

```text
UB_total = 0（无显式 AIV Queue/TBuf）
dynamic_ub = N/A（VF_CALL 仅给 threads=512 与函数参数）
launch_local_memory = uint32_t(UB-128*1024)
workspace_total = 16*1024*1024 B（body 消费 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程 `val/fval/result` | 当前 GM load/算术 | 当前 GM store | 下一 grid-stride 迭代 | SIMT 单线程程序序 | 不跨线程共享；输出索引唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→compute→GM | 单线程程序序 | scalar load | 除法与 store | 当前 idx | half→float→`1.0f/f`→half；`idx<count` 保护 tail |
| 跨线程/跨 block | N/A | 唯一索引划分 | N/A | 输出元素 | 无 barrier/atomic；stride=512*blockNum |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `ForeachReciprocalTilingFunc` | key、grid、POD、workspace/local memory |
| Kernel entry/router | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<0>` | mode 0→half |
| Device body/helper | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h` | `Process<half>`、`OpForeachReciprocalSimt<half>`、`ComputeReciprocal<half>` | Tensor 循环、grid stride 与 FP32 中间除法 |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `ForeachReciprocalTilingFunc` | `tensorNum==0` 默认或 `xDtype==DT_FLOAT` | mode 1 |
| Router | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<schMode>` | `schMode==1` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_reciprocal<1> -> Process<float>
  -> per non-empty Tensor VF_CALL<OpForeachReciprocalSimt<float>>(threads=512)
    -> ComputeReciprocal<float> -> GM store
```

##### 源码等价伪码

```text
for each Tensor and each idx assigned by idx=blockIdx*512+threadIdx+k*(512*blockNum):
  if idx<count: y[i][idx]=1.0f/x[i][idx]
# N==0 时 Tensor 循环为空
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `ListTensorDesc::GetDataPtr<float>(i)+idx` | 1 FP32/迭代 | FP32 ND storage | caller/SIMT | SIMT/downstream | stream 完成前 | 唯一 idx；无 padding |
| thread scalar | 固定仓不可见的寄存器位置 | 一个 FP32 输入/结果 | float | 当前线程 | 当前线程 | 单迭代 | 下一 idx 覆盖 |
| launch local memory/workspace | 同 R-001 | `UB-128KiB` / 16 MiB | runtime resource | Host/runtime | body 不读取 | launch/executor | 无用户数据布局 |

##### 分层内存布局

N/A：与 R-001 相同，无显式 AIV UB、dynamic UB 或中间 workspace 数据。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A
launch_local_memory=uint32_t(UB-128KiB)
workspace_total=16MiB（body 消费0）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 scalar | GM load/除法 | GM store | 下一 idx | 单线程程序序 | 无共享、无地址冲突 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar pipeline | 程序序 | FP32 load | `1.0f/x`、FP32 store | 当前 idx | 全程 FP32；`idx<count` 保护尾部 |
| 并发 | N/A | grid-stride 唯一映射 | N/A | 输出 | 无 atomic/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `ForeachReciprocalTilingFunc` | 空列表默认 key、FP32 key 与资源 |
| Kernel entry/router | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<1>` | mode 1→float |
| Device body/helper | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h` | `Process<float>`、`ComputeReciprocal<float>` | FP32 grid-stride 除法 |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `ForeachReciprocalTilingFunc` | `xDtype==DT_BF16` | mode 2 |
| Router | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<schMode>` | `schMode==2` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_reciprocal<2> -> Process<bfloat16_t>
  -> per non-empty Tensor VF_CALL<OpForeachReciprocalSimt<bfloat16_t>>(threads=512)
    -> ComputeReciprocal<bfloat16_t> -> GM store
```

##### 源码等价伪码

```text
for each assigned idx with idx<count:
  f=static_cast<float>(x_bf16[i][idx])
  r=1.0f/f
  y_bf16[i][idx]=static_cast<bfloat16_t>(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `ListTensorDesc::GetDataPtr<bfloat16_t>(i)+idx` | 1 BF16/迭代 | BF16 ND storage | caller/SIMT | SIMT/downstream | stream 完成前 | 唯一 idx |
| thread scalar | 固定仓不可见的寄存器位置 | BF16 input/result + FP32 intermediates | bf16/float | 当前线程 | 当前线程/store | 单迭代 | 下一 idx 覆盖 |
| launch local memory/workspace | 同 R-001 | `UB-128KiB` / 16 MiB | runtime resource | Host/runtime | body 不读取 | launch/executor | 无用户数据布局 |

##### 分层内存布局

N/A：无显式 AIV UB/TQue/TBuf 或 dynamic UB；FP32 中间量为线程私有标量，寄存器字节数固定仓不可见。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A
launch_local_memory=uint32_t(UB-128KiB)
workspace_total=16MiB（body 消费0）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread BF16/FP32 scalars | load/cast/div | cast/store | 下一 idx | 单线程程序序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32→BF16 | C++ `static_cast` + FP32 `/` | BF16 load | BF16 store | 当前 idx | 除法为 FP32；降精度 round mode 的底层实现固定仓不可见 |
| 并发/tail | grid-stride guard | block/thread 索引 | scalar body | 输出范围 | `idx<count`；无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `ForeachReciprocalTilingFunc` | BF16 key、grid、POD、resource |
| Kernel entry/router | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<2>` | mode 2→bfloat16_t |
| Device body/helper | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h` | `Process<bfloat16_t>`、`ComputeReciprocal<bfloat16_t>` | FP32 中间除法与 element tail |

## 算子特定优化

### Trick T-001：按 Tensor 复用同一 grid-stride launch 形态

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：`tensorNum<=256`，每个非空 Tensor 都能通过 `ListTensorDesc` 直接取得 GM 指针。
- Golden 来源：[`foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h:72-82`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h#L72-L82)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t i = 0; i < tilingData->tensorNum; i++) {
    int64_t count = tilingData->tensorElements[i];
    if (count == 0) {
        continue;
    }
    __gm__ T* xPtr = xList.GetDataPtr<T>(i);
    __gm__ T* yPtr = yList.GetDataPtr<T>(i);
    Simt::VF_CALL<OpForeachReciprocalSimt<T>>(
        Simt::Dim3(THREAD_NUM), xPtr, yPtr, count);
}
```

#### 机制与设计原因

Host 只用最大 Tensor 决定统一 `blockDim`，POD 保存各 Tensor 元素数而不保存拼接前缀和、每核 Tensor 边界或 tile 参数。Device 对每个非空 Tensor 复用同一组 blocks 和固定 512 threads，以 grid stride 直接覆盖本 Tensor；较小 Tensor 由 `idx<count` 自然裁掉多余线程。这样避免了跨 Tensor 拼接索引恢复、UB staging 和中间 workspace 数据，但保留了逐 Tensor VF call 的程序序开销。

#### 迁移边界

- 适用：TensorList 各项相互独立、逐元素、无跨 Tensor 归约或顺序依赖的 SIMT 算子。
- 前提：所有列表项 dtype 一致，descriptor 在整个 Kernel 内有效；`blockDim` 必须能由 `Simt::GetBlockNum()` 正确观察；每线程输出地址唯一。
- 失效：需要跨 Tensor 融合、共享状态、atomic 汇聚、不同 dtype 或每 Tensor 不同 launch 配置时，必须引入额外 selector/POD/同步路线；固定仓无 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | mode 0 | mode 1；空列表默认 | mode 2 | Host `if/else if` dtype 顺序 |
| 模板与 dtype | `Process<half>` | `Process<float>` | `Process<bfloat16_t>` | entry 编译期模板消除运行时 dtype 分支 |
| 循环与分核 | Tensor→grid-stride | 同 | 同 | 最大 Tensor 选择统一 blockDim |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无显式 UB；half/float 线程标量 | 无显式 UB；float 标量 | 无显式 UB；bf16/float 标量 | 全部直接 GM 标量路径 |
| 内存总占用与复用 | local memory launch reserve；16 MiB WS 不读 | 同 | 同 | 资源配置完全相同 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM load/store + SIMT VF | 同 | 同 | 无 Vector/NDDMA/通信节点 |
| 同步、精度与 tail | 程序序；FP32 中间；idx guard | 程序序；FP32；idx guard | 程序序；FP32 中间；idx guard | 无共享写，不需 barrier/atomic |
| fallback/失败边界 | 无 | 空列表零循环 | 无 | 非支持 dtype 在 Tiling 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；见无后缀卡 | 公开支持；见无后缀卡 | 公开支持 Ascend 950PR/950DT |
| Host/Launcher | arch22 公共 Tiling；跨列表拼接后按 32 B block 分核；≤50 Tensor；32 B WS | 同 A2 | arch35 专用 Tiling；按最大 Tensor 选 grid；≤256 Tensor；16 MiB WS |
| Kernel/资源 | AIV Vector、UB Queue、Div/repeat/tail DMA | 同 A2 | 512-thread SIMT、GM 直访、FP32 标量除法、无显式 UB Queue |

## 限制与失败边界

- Tensor 数上限来自 950 POD 为 256；源码先把 `size_t` 实例数窄化为 `int32_t` 再检查，超大值的转换语义没有单独保护，调用侧应先限制 `N<=256`。
- Host 只从第一个 Tensor 取 dtype，Device 随后以同一模板类型解释全列表；列表内 dtype 一致性必须由公开 API/OpDef 保证，不能绕过 Host 入口直接混用。
- `tensorElements`、`maxTensorElements` 依赖 storage shape size；负数/乘法溢出没有本算子专门检查。`UB-128 KiB` 也只检查 `UB!=0`，没有显式下溢及转 `uint32_t` 窄化保护。
- 16 MiB workspace 与 `SetLocalMemorySize` 的 runtime 内部消费固定仓不可见；只能确认算子 body 不读 workspace，也不显式寻址 local-memory buffer。
- ACLNN inner bridge、executor 编排、连续化 workspace slice 和 BF16 `static_cast` 的底层舍入模式固定仓不可见；公开错误语义存在，但不可外推其逐行实现。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_reciprocal/docs/aclnnForeachReciprocal.md` | 两段式 API、产品/错误/约束表 | 公开 API 与 950 支持 |
| 入口声明/约束 | `foreach/foreach_reciprocal/README.md` | 产品支持、公式 | 逻辑语义与平台 |
| 入口实现 | `foreach/foreach_reciprocal/op_host/foreach_reciprocal_def.cpp` | `ForeachReciprocal` | dtype/layout、AutoContiguous、ascend950 config |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`、`IMPL_OP_INFERSHAPE(ForeachReciprocal)` | TensorList 数量与输出 shape/dtype |
| Host/Tiling/Launcher | `foreach/foreach_reciprocal/op_host/arch35/foreach_reciprocal_tiling.cpp` | `GetPlatformInfo`、`ForeachReciprocalTilingFunc` | 校验、grid、key、POD、workspace/local memory |
| Host/Tiling/Launcher | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_tiling_data.h` | `ForeachReciprocalTilingData` | POD ABI 与 256 上限 |
| Host/Tiling/Launcher | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个 schMode 模板值 |
| Kernel entry/body | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal.cpp` | `foreach_reciprocal<schMode>` | 编译期 router |
| Kernel entry/body | `foreach/foreach_reciprocal/op_kernel/arch35/foreach_reciprocal_simt.h` | `Process`、`OpForeachReciprocalSimt`、`ComputeReciprocal` | Tensor 循环、VF、grid stride、精度 |
| 构建接线 | `foreach/foreach_reciprocal/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`/`SUPPORT_TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_reciprocal/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/...` | ascend950 Kernel 源接线 |
| 构建接线 | `foreach/foreach_reciprocal/op_host/config/ascend950/foreach_reciprocal_binary.json` | `op_list` | FP16/FP32/BF16 binary 接线 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2056 B POD 传 Tensor 数、grid 镜像和逐 Tensor 元素数 | 字段 offset 0/4/8、256 项宽度、int64 元素单位 | `ForeachReciprocalTilingData` |
| 调用顺序 | entry→Process→按 Tensor VF call→线程 grid stride | 空 Tensor 必须先 skip，再取指针；三 key 与 T 一致 | `foreach_reciprocal`、`Process<T>` |
| workspace/内存 | body 直接 GM；workspace 只作 16 MiB runtime reserve | 不得把 workspace 当已初始化中间数据；保留 128 KiB local-memory扣减协议 | `ForeachReciprocalTilingFunc` |
| 同步 | 唯一索引与线程程序序足够 | 引入共享写/跨 Tensor依赖时必须新增 barrier/atomic/节点边 | `OpForeachReciprocalSimt` |
| 精度与尾块 | 三 dtype 都以 FP32 除法；`idx<count` 截尾 | FP16/BF16 升降精度顺序和 512*blockNum 步长 | `ComputeReciprocal` |
| 失败边界 | 平台、N、dtype、POD/workspace 获取失败关闭 | 禁止绕过列表 dtype 一致性；无 Device fallback | `ForeachReciprocalTilingFunc` |