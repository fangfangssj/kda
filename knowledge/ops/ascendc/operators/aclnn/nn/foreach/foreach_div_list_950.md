---
type: Operator
title: "ForeachDivList 950 ACLNN 算子知识"
description: "说明 aclnnForeachDivList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/docs/aclnnForeachDivList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_host/foreach_div_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_host/config/ascend950/foreach_div_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/arch35/foreach_div_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:39:39Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachDivList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachDivListGetWorkspaceSize` + `aclnnForeachDivList`，公开 ACLNN，无带版本号的新入口。 |
| 计算语义 | 两个等长 TensorList 对应项逐元素相除：`y_i[j]=x1_i[j]/x2_i[j]`。 |
| 输入输出 | 三个动态 TensorList，ND，FP16/FP32/BF16；`x1/x2` 可非连续并经 `AutoContiguous`，输出必须连续。 |
| 执行类型 | 单 Device Kernel；其实际计算节点为 SIMT `VF_CALL`，R-001 原生 FP16/FP32，R-002 BF16→FP32→BF16。无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001：`schMode` 0/1；R-002：`schMode` 2。 |
| 核心资源 | 每对 Tensor 由 512 threads/block 的 SIMT grid-stride 循环直接访问 GM；用户 workspace 保存 count 与 3 路地址元数据；无 AIV TQue/TBuf、无 shared/dynamic UB 数据缓冲。 |
| 来源闭包 | `source_closed`：公开接口/约束→生成式 ACLNN 框架边界→OpDef/推形→arch35 Tiling/POD→模板化 entry→workspace producer/consumer→SIMT body→950 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x1=[x1_0,...,x1_(n-1)]`、`x2=[x2_0,...,x2_(n-1)]`，则 `y=[y_0,...,y_(n-1)]`，且对 `j∈[0,numel(x1_i))` 有 `y_i[j]=x1_i[j]/x2_i[j]`。公共接口要求同位置输入 shape 相同、dtype 一致；Kernel 对每项只处理 Tiling 从 `x1_i` 得到的 `elementCounts[i]`。公开文档声明默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；每项 0–8 维，ND | FP16/FP32/BF16 | 支持空 Tensor，列表内 dtype 一致；非连续 Tensor 由 `AutoContiguous()` 处理。950 Tiling 以该列表产生 tensorCount、元素数与 key。 |
| `x2` | TensorList；同位置 shape 与 `x1` 一致，ND | 与 `x1` 相同 | 列表项数应与 `x1` 一致；允许非连续输入并自动连续化。 |
| `out` | TensorList，ND | 与 `x1` 相同 | 公开接口要求输出连续；公共推形逐项复制 `x1` shape/dtype。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 第一段返回 `Wuser+Wsys`；`Wuser=align_up(8+32n,128)`，`Wsys=GetLibApiWorkSpaceSize()`。 |
| `executor` | 执行器输出 | `aclOpExecutor**` | 保存前处理、Tiling 与单 Kernel 节点；生成 bridge 固定仓不可见。 |
| `workspace`,`stream` | Device 地址、执行流 | 指针/handle | 第二段在 stream 启动 Kernel；调用方持有 workspace 至 stream 完成。 |

### 入口与版本

固定仓只公开无版本后缀的 `aclnnForeachDivList` 两段式入口，故其为 950 的最新公开版本。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef 添加 `ascend950`，Host CMake 将 `ascend950` 映射到 `arch35`，Kernel CMake 显式选用 `arch35/foreach_div_list.cpp`，binary config 包含三种 dtype，平台无发布冲突。公开 ACLNN 头与 inner bridge 为生成式框架边界，固定仓不可见；算子自有 Host、POD、Kernel 与构建链路已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开第一段接口定义空指针、dtype 不支持/不一致、shape/TensorList dtype/维数不合法的错误。OpDef 限定三种 dtype、ND、动态 TensorList，并为三路标注 `AutoContiguous()`。
2. `InferShape4ForeachCommon` 要求 `x1` 实例数等于输出数，逐项复制 `x1` shape；`InferDataType4ForeachCommon` 逐项复制 dtype。
3. arch35 `ForeachDivListTilingFunc` 先从 compileInfo 读取 `coreNum/ubSize`，无有效 compileInfo 时回退平台查询；`ubSize<=128 KiB` 直接失败，否则扣除 128 KiB DCache 预留。
4. Tiling 从动态输入 `x1` 读取 `tensorNum`，限制为 256；逐项读取 storage numel，填入 `elementCounts`，累计 `totalElements`、`maxTensorElements`，并仅从第 0 项取得 dtype。
5. 950 Tiling 本身没有复查 `x2/out` 的项数、shape、dtype，也没有显式检查 TilingData/workspace-size 指针；这些约束依赖公开 ACLNN/OpDef/推形框架边界。移植为 direct launch 时必须补齐，不能只调用该 Tiling 函数。

### 参数、Workspace 与生命周期

令 `n=tensorCount`、`Mmax=max_i numel(x1_i)`、`Cplat=GetCoreNumAiv()`、`U=平台 UB 字节数`。Tiling 产生：

- `needCoreNum = 1`，若 `Mmax>=1024` 则改为 `min(Cplat,ceil(Mmax/1024))`；同时 `blockDim=needCoreNum`。
- `localMemorySize=U-128 KiB`，通过 `SetLocalMemorySize(uint32_t(...))` 写入 launch；当前 SIMT body 不分配 shared/dynamic UB 数据结构。
- `Wuser=align_up(8+n*8*4,128)=align_up(8+32n,128)`，`Wtotal=Wuser+Wsys`。`GetUserWorkspace(workspace)` 从总 workspace 取得用户区起点；系统区布局由框架管理，固定仓不可见。

用户区由 K0 自身 `PopulateWorkspace` 写入，并在同一 K0 的后续 `ProcessDiv`/`ProcessDivBf16` 读取。调用方拥有总 workspace；用户元数据从 K0 开始写入到所有 SIMT VF 调用完成前有效，之后可随 stream 完成而复用或释放。

### 启动图与失败传播

```text
aclnnForeachDivListGetWorkspaceSize
  -> [生成式 ACLNN bridge：校验/AutoContiguous/executor，固定仓不可见]
  -> ForeachDivListTilingFunc
     -> blockDim=needCoreNum, localMemorySize=U-128KiB
     -> schMode/key={0,1,2}, workspace=Wuser+Wsys
aclnnForeachDivList(workspace, workspaceSize, executor, stream)
  -> node K0: foreach_div_list<schMode><<<needCoreNum, stream>>>
     -> GetUserWorkspace
     -> PopulateWorkspace(x1,x2,y,userWorkspace,POD)
     -> for each nonempty tensor: VF_CALL<dtype body>(threadsPerBlock=512,...)
```

这是一个 Device Kernel 节点；`PopulateWorkspace` 和多个按 Tensor 顺序发生的 `VF_CALL` 是 K0 内阶段，不是多个 Host launch。每个外层 block 都执行相同的 metadata 写入，随后自身按程序顺序读取；源码无跨 block barrier，但所有 block 对同一元数据写入相同值。若 Tiling 获取平台信息、UB、shape 或 local-memory 设置失败，则返回 `GRAPH_FAILED` 并阻止 K0；第二段提交失败通过 `aclnnStatus` 传播。

## Route 与参数设计

### 分核与资源计算

`needCoreNum` 只由最大 Tensor 的元素数决定：小于 1024 元素时固定 1 block；否则约每 1024 个最大 Tensor 元素增加一个 block，最大不超过 AIV core 数。每个 `VF_CALL` 使用 `THREAD_NUM=512`。SIMT 线程 `(b,t)` 的首元素索引为 `b*512+t`，步长为 `512*GetBlockNum()`，并以 `index<elementCount` 保护尾部，因此同一 grid 对列表中每个非空 Tensor 重新使用相同映射。

没有 tile/UB copy 循环：线程直接从 `x1_gm[index]`、`x2_gm[index]` 读 GM 并写 `y_gm[index]`。空 Tensor 在 outer loop 被 `if (elemCount==0) continue` 跳过。Host 设置的 `totalElements`、`needCoreNum`、`maxTensorElements` 除 launch 参数外不在 Device POD body 中读取。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 原生 SIMT 除法 | `schMode=0` FP32；`schMode=1` FP16 | `GetTilingKeyByDtype(DT_FLOAT)=0`，`DT_FLOAT16=1`；entry 以编译期 `if constexpr` 匹配 | `schMode=2` 去 R-002；其它值无 body |
| 2 | R-002 BF16 升精度 SIMT 除法 | `schMode=2` | `GetTilingKeyByDtype(DT_BF16)=2` | 非支持 dtype 按 Host 函数会默认为 0，必须由 API/OpDef 先拒绝 |

```text
dtype == FLOAT32  -> schMode 0 -> R-001/float
else dtype == FLOAT16 -> schMode 1 -> R-001/half
else dtype == BFLOAT16 -> schMode 2 -> R-002
else -> Host helper returns 0；公开入口必须先拒绝，否则会误入 float 实例
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 原生 SIMT 除法 | 0/1 | FP32/FP16；公开接口的列表/shape/dtype/≤8维约束；950 | 单 K0；K0 内每 Tensor 一次 `VF_CALL` | `foreach_div_list<0/1>`→`ProcessDiv<float/half>`→`ForeachDivListComputeSimt<T>` | workspace 元数据；线程直接 GM load/div/store；无 shared UB、barrier、atomic | grid-stride 的 `index<elementCount`；空 Tensor 跳过 |
| R-002 BF16 升精度 SIMT 除法 | 2 | BF16 与同样接口约束；950 | 单 K0，同上 | `foreach_div_list<2>`→`ProcessDivBf16`→`ForeachDivListBf16ComputeSimt` | 每线程 BF16 load→FP32 标量→BF16 store；无共享缓冲 | 同上；`__float2bfloat16_rn` 降精度 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| DCache 预留 | `ubSize>128*1024` | local memory launch 只使用扣除 DCache 后的容量 | 全部 | `GRAPH_FAILED` | `ForeachDivListTilingFunc` |
| Tensor 数 | `static_cast<int32_t>(tensorNum)<=256` | POD `elementCounts[256]` 与 workspace metadata | 全部 | `GRAPH_FAILED` | `ForeachDivListTilingFunc` |
| 核数阈值 | `Mmax<1024→1`；否则 `min(Cplat,ceil(Mmax/1024))` | SIMT block grid 以最大 Tensor 规模选并行度 | 全部 | 无其它 Route | `SINGLE_CORE_MIN_ELEMENTS`、`needCoreNum` |
| 线程数 | 固定 `THREAD_NUM=512` | `LAUNCH_BOUND(512)` 与 `VF_CALL(Dim3(512))` 一致 | 全部 | 无 fallback | `foreach_div_list_simt.h::THREAD_NUM` |
| dtype key | 仅第 0 个 x1 dtype 产生 0/1/2 | entry 是编译期模板实例 | 全部 | 非支持 dtype helper 默认 0；依赖前置校验 | `GetTilingKeyByDtype` |
| `maxTensorElements` 窄化 | 每项 `int64 numel` 转为 `int32_t` | 核数公式使用 32-bit 最大元素数 | 全部 | 固定仓无溢出 fallback | `ForeachDivListTilingFunc` |

### POD/ABI

Host `struct ForeachDivListTilingData` 与 Device `class ForeachDivListTilingData` 字段顺序、类型完全相同；无 packing 指令。按这些明确声明的自然 C++ 布局，`alignof=8`、`sizeof=2072`，offset 依次为 0、8、16、20、24。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachDivListTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；其后 4 B padding | block/core 数 | `needCoreNum` 窄化赋值；同时传给 `SetBlockDim` | POD 字段本身未读取；实际 grid 由 launch blockDim 消费 | ≥1，≤平台 core 数 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素/全列表 | `Σ numel(x1_i)` | Device 未读取（inactive） | 累加溢出未检查 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 16 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `PopulateWorkspace`、`ProcessDiv*` 的循环上界 | 0–256 |
| 同上 | `maxTensorElements` | `int32_t`/32 | offset 20 | 元素/最大 Tensor | `max(numel)` 经 int32 窄化 | Device 未读取（inactive） | 大于 INT32_MAX 时固定仓无 guard |
| 同上 | `elementCounts[256]` | `int64_t[256]` | offset 24；2048 B；结构总大小 2072 B | 元素/每 Tensor | 循环 `tilingData->elementCounts[i]=numel` | `PopulateWorkspace` 写用户 workspace | 仅前 `tensorCount` 项有效；可为 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 原生 SIMT | schMode 0/1 | `op_host/arch35/foreach_div_list_tiling.cpp::ForeachDivListTilingFunc` | Host/Device `ForeachDivListTilingData`；blockDim/local memory/workspace | `op_kernel/arch35/foreach_div_list.cpp::foreach_div_list<0/1>` | `foreach_div_list_simt.h::PopulateWorkspace/ProcessDiv/ForeachDivListComputeSimt` | `op_kernel/CMakeLists.txt` 将 ascend950 指向 arch35；950 binary config 三实例 |
| R-002 BF16 SIMT | schMode 2 | 同上 | 同上 | `foreach_div_list<2>` | `PopulateWorkspace/ProcessDivBf16/ForeachDivListBf16ComputeSimt` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD，通过 `GetUserWorkspace` 越过框架系统 workspace。`PopulateWorkspace` 先把 Tensor 数、每项元素数及 x1/x2/y 的真实 64-bit GM 地址整理成连续元数据。其地址提取协议与 A2/A3 一致：TensorList 描述首个 `uint64_t` 是到地址数组的字节偏移，右移 3 转成 8 B word offset，再按 Tensor 索引取地址。随后所有 block 以相同 Tensor 顺序调用 `VF_CALL`；每次 VF 内线程以 grid-stride 处理当前 Tensor，Tensor 之间在同一外层 block 的程序顺序中串行。

### Route R-001：原生 FP16/FP32 SIMT 除法

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `foreach/foreach_div_list/docs/aclnnForeachDivList.md` | 参数/约束 | FP16 或 FP32、列表约束成立 | R-001 |
| Host/Tiling | `foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.cpp` | `GetTilingKeyByDtype` | FLOAT→0，FLOAT16→1 | schMode 0/1 |
| Router | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list.cpp` | `foreach_div_list<schMode>` | `if constexpr schMode==0/1` | `ProcessDiv<float/half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_div_list<0 or 1>(x1,x2,y,workspace,tiling)
  -> GET_TILING_DATA_WITH_STRUCT(ForeachDivListTilingData)
  -> GetUserWorkspace(workspace)
  -> PopulateWorkspace
     -> GetListTensorAddrValue(x1/x2/y,i) for i=0..tensorCount-1
  -> ProcessDiv<T>
     -> for each nonempty Tensor i
        -> VF_CALL<ForeachDivListComputeSimt<T>>(Dim3(512), elemCount, x1_i, x2_i, y_i)
           -> per thread grid-stride: y[index]=x1[index]/x2[index]
```

##### 源码等价伪码

```text
# 伪码；K0 的每个 outer block 执行
ws[0] = n
for i in [0,n):
    ws[1+i]     = elementCounts[i]
    ws[1+n+i]   = tensor_addr(x1,i)
    ws[1+2n+i]  = tensor_addr(x2,i)
    ws[1+3n+i]  = tensor_addr(y,i)

for i in [0,n):
    count = ws[1+i]
    if count == 0: continue
    VF_CALL threadsPerBlock=512:
      for index = blockIdx*512 + threadIdx; index<count; index += 512*blockNum:
          y_i[index] = x1_i[index] / x2_i[index]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList 描述 GM | `base + (header>>3) + i` | 每项 64-bit 地址 | runtime layout | ACLNN runtime | `GetListTensorAddrValue` | 输入/executor 生命周期 | 8 B address |
| user ws count | byte offset 0 | 8 B | int64 | `PopulateWorkspace` | 当前 K0 调试/布局；Process 直接用 POD tensorCount | 调用方 workspace；K0 | 8 B |
| user ws elementCounts | byte offset 8 | `8n` B | int64[n] | `PopulateWorkspace` | `ProcessDiv` | 同上 | 紧密 8 B |
| user ws x1 addresses | `8+8n` | `8n` B | uint64 address[n]（以 int64 存） | `PopulateWorkspace` | `ProcessDiv` | 同上 | 紧密 8 B |
| user ws x2 addresses | `8+16n` | `8n` B | 同上 | 同上 | 同上 | 同上 | 同上 |
| user ws y addresses | `8+24n` | `8n` B | 同上 | 同上 | 同上 | 同上 | 同上 |
| user ws padding | `8+32n` | `Wuser-(8+32n)` B | 未使用 | 无 | 无 | 调用方 | 总区 128 B 对齐 |
| x1/x2/y GM element | `tensorAddr + index*sizeof(T)` | 1 element/thread/iteration | FP16/FP32 ND flatten | 调用方/VF | VF/调用方 | K0 | index guard 防尾越界 |
| local memory | launch size `U-128KiB` | 未由 body 切片 | N/A | Host `SetLocalMemorySize` | 当前 body 无显式 consumer | K0 | 128 KiB 留给 DCache |

##### 分层内存布局

```text
system workspace [0, Wsys)                       # framework-owned，布局固定仓不可见
user workspace   [Wsys, Wsys+Wuser):
  +0             : tensorCount (8 B)
  +8             : elementCounts[n] (8n B)
  +8+8n          : x1Addrs[n] (8n B)
  +8+16n         : x2Addrs[n] (8n B)
  +8+24n         : yAddrs[n] (8n B)
  +8+32n..Wuser  : alignment padding
SIMT shared/dynamic UB data buffer: N/A；VF 直接访问 GM
```

##### 字节账与复用顺序

```text
UB_total = 0 B explicit TQue/TBuf/shared allocation
dynamic_ub = localMemorySize=U-128KiB（launch reservation；当前 body 无显式数据缓冲 consumer）
workspace_user = align_up(8+32n,128)
workspace_total = workspace_user + GetLibApiWorkSpaceSize()
=> n<=256；最大 raw metadata=8200 B，向上对齐为 8320 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| user ws metadata | 每个 outer block 的 `PopulateWorkspace` | 同 block 后续 `ProcessDiv` | 本 Kernel 完成前其它 block 也写相同值；下个算子可在 stream 完成后覆盖 | 本线程块程序顺序；无跨 block barrier | 所有 block 必须写完全相同的 POD/地址值；不得改为 block 相关数据 |
| SIMT 标量寄存器 | 当前 thread load | 同 thread 除法/store | 下一 grid-stride iteration | 线程程序顺序 | `index<count` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| metadata | 同一 outer block 程序顺序 | `PopulateWorkspace` | `ProcessDiv` | user ws | 无显式跨 block barrier；相同值并发写 |
| Tensor 顺序 | `for i` + `VF_CALL` | 当前 Tensor VF | 下一 Tensor VF | 每项地址/count | 固定仓无显式 event/barrier，调用语义按函数顺序 |
| 元素计算 | SIMT grid-stride | x1/x2 GM load | y GM store | 该线程负责的 index | FP16/FP32 原 dtype除法；`index<count` 同时处理任意 tail |
| 线程冲突 | 无 atomic/barrier | 每线程唯一 index 序列 | N/A | y 的不相交元素 | 步长 512×blockNum，映射互斥 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.cpp` | `ForeachDivListTilingFunc` | core/grid、local memory、key、workspace 与 count producer |
| Kernel entry/router | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list.cpp` | `foreach_div_list<0/1>` | POD 解析、user ws、模板分发 |
| Device body/helper | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_simt.h` | `PopulateWorkspace`、`ProcessDiv<T>`、`ForeachDivListComputeSimt<T>` | 元数据布局、VF 参数、线程地址/步长/tail |

### Route R-002：BF16 升精度 SIMT 除法

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `foreach/foreach_div_list/docs/aclnnForeachDivList.md` | 参数表 | BF16 且公开列表约束成立 | R-002 |
| Host/Tiling | `foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.cpp` | `GetTilingKeyByDtype` | `DT_BF16→2` | schMode 2 |
| Router | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list.cpp` | `foreach_div_list<2>` | `if constexpr schMode==2` | `ProcessDivBf16` |

#### Device 数据流

##### 函数级调用链

```text
foreach_div_list<2>
  -> PopulateWorkspace（与 R-001 相同）
  -> ProcessDivBf16
     -> for each nonempty Tensor i
        -> VF_CALL<ForeachDivListBf16ComputeSimt>(Dim3(512), elemCount, x1_i, x2_i, y_i)
           -> per thread: BF16 loads -> __bfloat162float x2 -> FP32 div -> __float2bfloat16_rn -> GM store
```

##### 源码等价伪码

```text
# metadata 和 grid-stride index 与 R-001 相同
for each index assigned to this SIMT thread where index<elementCount:
    bx1 = x1[index]
    bx2 = x2[index]
    fx1 = __bfloat162float(bx1)
    fx2 = __bfloat162float(bx2)
    fresult = fx1 / fx2
    y[index] = __float2bfloat16_rn(fresult)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| user ws metadata | 与 R-001 相同 | `8+32n` raw，128 B 对齐 | int64/address arrays | `PopulateWorkspace` | `ProcessDivBf16` | K0 | 不按 dtype 改变 |
| x1/x2 GM | `addr + index*2` | 1 BF16/thread/iteration | BF16/ND flatten | 调用方 | SIMT thread | K0只读 | index guard |
| thread FP32 temporaries | 寄存器生命周期 | `fx1`,`fx2`,`fresult` 各一个标量；物理寄存器字节数固定仓不可见 | FP32 | convert/div | convert/store | 单 thread/单 iteration | 不共享、不物化到 workspace |
| y GM | `addr + index*2` | 1 BF16/thread/iteration | BF16/ND flatten | `__float2bfloat16_rn` | 调用方 | K0写 | 唯一线程写每元素 |
| shared/dynamic UB | N/A | 0 B explicit | N/A | N/A | N/A | N/A | localMemory launch reservation不被 body 切片 |

##### 分层内存布局

```text
workspace layout 与 R-001 相同；不增加 BF16 临时 slice。
每线程：BF16 x1/x2 GM -> FP32 scalar fx1/fx2/fresult -> BF16 y GM。
SIMT shared/dynamic UB data buffer: N/A。
```

##### 字节账与复用顺序

```text
UB_total = 0 B explicit shared/Queue allocation
dynamic_ub = U-128KiB launch reservation，body 无显式 consumer
workspace_user = align_up(8+32n,128)
workspace_total = workspace_user + Wsys
register_bytes = 固定仓不可见；仅能证明 fx1/fx2/fresult 的单 iteration 生命周期
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| user ws metadata | 同 R-001 | `ProcessDivBf16` | stream 完成后 | 同 block 程序顺序 | 所有 block 写相同数据 |
| `fresult` 标量 | FP32 `/` | `__float2bfloat16_rn` | 下一 grid-stride iteration | 单线程程序顺序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16 load/convert | `__bfloat162float` | GM load | FP32 `/` | thread-local scalar | 精确展开到 FP32 表示 |
| FP32 divide | C++ `/` in SIMT VF | fx1/fx2 | BF16 conversion | fresult | NaN/Inf 遵循目标编译器/硬件 FP 语义；注释期望 IEEE 754，但固定仓无更细开关 |
| BF16 store | `__float2bfloat16_rn` | fresult | y GM | 单元素 | round-to-nearest；tie 细节属 intrinsic 边界 |
| tail | `index<elementCount` | grid mapping | load/store | 最后不足整 grid 的元素 | 无 padding lane、无额外 copy |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.cpp` | `GetTilingKeyByDtype`、`ForeachDivListTilingFunc` | BF16 key、grid/workspace |
| Kernel entry/router | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list.cpp` | `foreach_div_list<2>` | schMode 2→BF16 body |
| Device body/helper | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_simt.h` | `ProcessDivBf16`、`ForeachDivListBf16ComputeSimt` | per-Tensor VF_CALL、grid-stride、FP32 中间与 RN cast |

## 算子特定优化

### Trick T-001：固定 512 线程的 grid-stride 复用同一 grid 处理任意 Tensor 长度

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：当前 Tensor `elementCount>0`；Host 已由最大 Tensor 选择 `needCoreNum`。
- Golden 来源：[`foreach_div_list_simt.h:81-94`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_simt.h#L81-L94)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void ForeachDivListComputeSimt(
    int64_t elementCount,
    __gm__ T* x1, __gm__ T* x2, __gm__ T* y)
{
    for (uint64_t index = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * THREAD_NUM + AscendC::Simt::GetThreadIdx());
         index < static_cast<uint64_t>(elementCount);
         index += static_cast<uint64_t>(THREAD_NUM * AscendC::Simt::GetBlockNum())) {
        // IEEE 754: 0/0=NaN, non-zero/0=Inf, NaN propagation
        y[index] = x1[index] / x2[index];
    }
}
```

#### 机制与设计原因

Host 不为每个 Tensor 重算 blockDim，也不物化一个拼接数据区；同一个 `needCoreNum×512` grid 在每次 `VF_CALL` 中通过 `index += 512*blockNum` 覆盖当前 Tensor。`index<elementCount` 同时承担任意长度尾保护。源码可证明该机制避免 Tensor 数据拼接与 UB tile 搬运，并使大 Tensor 可由全部 block/线程并行处理；没有真机性能百分比结论。

#### 迁移边界

- 适用：TensorList 中每项独立、逐元素、可直接 GM 访问的 SIMT 算子。
- 前提：不同线程的输出索引互斥；`VF_CALL` 的 blockNum 与 Host `blockDim` 一致；所有 Tensor 可共用按最大项选出的 grid。
- 失效：需要相邻元素共享数据、block 间归约，或每个 Tensor 需要不同 shared-memory 布局时必须增加同步/独立 launch 策略。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：workspace 仅物化元数据，SIMT 直接读写真实 Tensor GM

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：`tensorCount≤256`，用户 workspace 至少为 `align_up(8+32n,128)`。
- Golden 来源：[`foreach_div_list_simt.h:64-76`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_simt.h#L64-L76)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
ws[0] = static_cast<int64_t>(tensorCount);

__gm__ int64_t* elementCounts = ws + 1;
__gm__ int64_t* x1Addrs = ws + 1 + tensorCount;
__gm__ int64_t* x2Addrs = ws + 1 + 2 * tensorCount;
__gm__ int64_t* yAddrs  = ws + 1 + 3 * tensorCount;

for (int32_t i = 0; i < tensorCount; i++) {
    elementCounts[i] = tilingData->elementCounts[i];
    x1Addrs[i] = static_cast<int64_t>(GetListTensorAddrValue(x1, i));
    x2Addrs[i] = static_cast<int64_t>(GetListTensorAddrValue(x2, i));
    yAddrs[i]  = static_cast<int64_t>(GetListTensorAddrValue(y, i));
}
```

#### 机制与设计原因

用户 workspace 的增长量仅为每 Tensor 四个 64-bit 项加头部与对齐；真实 Tensor 数据不复制到 workspace 或 UB。外层 Process 读取紧密的 count/address 数组，VF 直接访问原始 GM。源码因此可证明避免了按总元素数增长的中间数据物化。workspace 参数来自 Host 的 `ComputeWorkspaceSize`，四个 slice 的 offset 与 Device 完全对应。

#### 迁移边界

- 适用：runtime TensorList 使用相同“首字节偏移+地址数组”描述协议，且 Kernel 可直接访问每项 GM 的算子。
- 前提：workspace owner 覆盖整个 K0；元数据写后再读；所有 block 写入值相同；地址为 64 bit。
- 失效：TensorList 描述 ABI 改变、需要搬移/重排真实数据，或元数据包含 block 私有值时必须重设布局与同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 原生 FP16/FP32 SIMT | R-002 BF16 升精度 SIMT | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | schMode 0/1 | schMode 2 | 首个 x1 dtype 决定 |
| 模板与 dtype | `ProcessDiv<float/half>` + 模板 VF | 专用 `ProcessDivBf16` + 专用 VF | BF16 需要显式转换 intrinsic |
| 循环与分核 | 每 Tensor 一个 VF_CALL；512-thread grid-stride | 完全相同 | shape 路线共享 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/shared；线程原 dtype 标量 | 无共享缓冲；线程 FP32 标量 | 临时值保持 thread-local |
| 内存总占用与复用 | 用户 ws `align128(8+32n)` | 相同，不增加数据 workspace | 两 Route 只存元数据 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT 直接 GM load `/` store | BF16 convert + FP32 `/` + RN convert | 均非 Cube |
| 同步、精度与 tail | 无 atomic/barrier；原 dtype；index guard | 无 atomic/barrier；FP32 中间；同 guard | 精度路线不同，不能合并 |
| fallback/失败边界 | 未知/无效 dtype 可能错误映射 key 0 | 无 fallback | 依赖 API/OpDef 前置 dtype 校验 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持；公开文档、ascend950 OpDef/config/CMake 一致 |
| Host/Launcher | arch22 按总元素 32 B block 分核，最多 50 core/Tensor | 与 A2 共用 | arch35 按最大 Tensor 每 1024 元素选 blockDim，最多 256 Tensor |
| Kernel/资源 | AIV Vector Queue/tile | 与 A2 共用 | 单 Kernel 内 SIMT VF_CALL；metadata workspace；无 AIV Queue |

## 限制与失败边界

- 公开接口要求两输入列表等长、同位置 shape 相同、三路与列表内 dtype 一致、每项维数≤8、输出连续；950 arch35 Tiling 本身未重做这些检查，direct 复用不能省略框架校验。
- `tensorCount` 最大 256。检查先把 `uint64_t tensorNum` 转成 `int32_t` 再比较，极端超大列表的窄化行为没有额外 guard。
- 每项 `int64_t numel` 写入 `int32_t maxTensorElements`；大于 INT32_MAX 的 Tensor 可能破坏 blockDim 估计，固定仓无 fallback。
- `GetTilingKeyByDtype` 对未知 dtype 默认返回 0，即 FP32 Route；安全性依赖 ACLNN/OpDef 在 Tiling 前拒绝不支持 dtype。
- `PopulateWorkspace` 由每个 outer block 对同一地址并发写相同元数据，源码无跨 block barrier。当前随后读取只依赖本 block 的程序顺序；移植时若改为 block 私有/不同值，必须新增明确同步或改成单 producer。
- Tiling 设置 `localMemorySize=U-128 KiB`，但当前 body 无显式 dynamic UB consumer；`SetLocalMemorySize` 与 `VF_CALL` 的底层映射属 AscendC 框架边界。
- ACLNN 生成 bridge、executor 具体生命周期、`GetUserWorkspace` 的系统区 offset、VF_CALL 的底层调度/完成协议与 SIMT intrinsic 的异常细节固定仓不可见。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_div_list/docs/aclnnForeachDivList.md` | 两段式 `aclnnForeachDivList*` | 原型、参数、错误、确定性 |
| 入口声明/平台 | `foreach/foreach_div_list/README.md` | 产品支持表、功能说明 | Ascend 950 支持与公式/dtype |
| Host/OpDef | `foreach/foreach_div_list/op_host/foreach_div_list_def.cpp` | `ops::ForeachDivList` | dtype/layout/AutoContiguous/ascend950 注册 |
| Host/推形 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`、`InferDataType4ForeachCommon`、ForeachDivList 注册 | 输出项数、shape 与 dtype 推导 |
| Host/Tiling | `foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.cpp` | `ForeachDivListTilingFunc`、`TilingParseForForeachDivList` | platform、key、blockDim、local memory、workspace、POD producer |
| POD/ABI | `foreach/foreach_div_list/op_host/arch35/foreach_div_list_tiling.h` | 两个 struct | Host POD 与 compileInfo 布局 |
| 构建接线 | `foreach/foreach_div_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR` | ascend950 Host 映射到 arch35 |
| Binary config | `foreach/foreach_div_list/op_host/config/ascend950/foreach_div_list_binary.json` | `op_list` | 950 的 FP16/FP32/BF16 二进制实例 |
| Kernel entry/body | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list.cpp` | `foreach_div_list<schMode>` | entry、POD、workspace、router |
| Kernel body | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_simt.h` | metadata/Process/VF symbols | workspace、线程映射、精度、tail |
| POD/ABI | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_tiling_data.h` | Device `ForeachDivListTilingData` | Device 字段消费布局 |
| Kernel router | `foreach/foreach_div_list/op_kernel/arch35/foreach_div_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 0/1/2 实例集合 |
| 构建接线 | `foreach/foreach_div_list/op_kernel/CMakeLists.txt` | ascend950 `add_kernel_sources` | 950 显式选择 arch35 Kernel，`AUTO_SYNC false` |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2072 B POD；仅 tensorCount/counts 为当前 Device 活跃字段 | Host/Device 字段顺序与位宽一致；数组 256；inactive 字段仍占 ABI | 两份 `ForeachDivListTilingData` |
| 调用顺序 | K0 内 PopulateWorkspace→逐 Tensor VF_CALL | metadata 必须先于本 block 的读取；非空 Tensor 才 VF_CALL | entry、`ProcessDiv*` |
| workspace/内存 | user ws 四个 8-byte 数组段，128 B 对齐；真实数据不物化 | `GetUserWorkspace` 基址、offset、owner/lifetime 不得改变 | `ComputeWorkspaceSize`、`PopulateWorkspace` |
| 同步 | 单线程程序顺序 + index 互斥；无 barrier/atomic | 并发 metadata 写只能保持 identical-write 条件 | `PopulateWorkspace`、SIMT VF |
| 精度与尾块 | FP16/FP32 原生，BF16 使用 FP32 中间和 RN conversion；grid guard tail | 512 threads、stride=`512*blockNum`、`index<count` | 两个 `ComputeSimt` |
| 失败边界 | platform/UB/Tensor数/local-memory失败返回 `GRAPH_FAILED` | 不支持 dtype、列表一致性与超大 numel 必须由前置层防住 | `ForeachDivListTilingFunc`、`GetTilingKeyByDtype` |