---
type: Operator
title: "ForeachLerpList 950 ACLNN 算子知识"
description: "说明 aclnnForeachLerpList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/docs/aclnnForeachLerpList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_host/foreach_lerp_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_host/arch35/foreach_lerp_list_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_host/arch35/foreach_lerp_list_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_host/config/ascend950/foreach_lerp_list_binary.json
    role: implementation
created_at: "2026-08-25T15:45:14Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachLerpList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachLerpListGetWorkspaceSize` + `aclnnForeachLerpList`，公开两段式 ACLNN；固定仓未见更高版本后缀。 |
| 计算语义 | 拼接全部 Tensor 的线性域，对每元素计算 lerp；`abs(weight)<0.5` 取 `x1+w*(x2-x1)`，否则取 `x2+(w-1)*(x2-x1)`。 |
| 输入输出 | 三个输入和一个输出均为 ND TensorList，FP16/FP32/BF16；对应 shape/dtype 一致，输入可非连续、输出必须连续。 |
| 执行类型 | 单 Kernel；K1 为 SIMT VF，按总元素数选择 1024-thread/int32 或 512-thread/int64 子路线；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001/002 FP16 int32/int64；R-003/004 FP32 int32/int64；R-005/006 BF16 int32/int64。 |
| 核心资源 | GM TensorList 指针表、2072 B GM Tiling POD、线程局部 FP32 标量；无 AIV Queue/TBuf、shared/dynamic UB 数组、atomic 或 workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→自动生成 bridge 边界→OpDef/InferShape→arch35 Tiling/显式 POD/key→templated entry→全部 SIMT IDX/dtype body→ascend950 构建和 binary config 已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设四个列表各有 `n` 个对应 Tensor，将 `x1` 的每个 storage shape 元素数累计为 `cum[t]`。Device 对拼接平坦索引 `k∈[0,totalElements)` 线性查找首个 `k<cum[t]`，得到 `local=k-(t?cum[t-1]:0)`，读取四个列表第 `t` 个 Tensor。所有 dtype 都升为线程局部 FP32：先算 `diff=x2-x1`、`path1=x1+w*diff`、`path2=x2+(w-1)*diff`，按 `abs(w)<0.5` 选择，再 cast 回 T。两条 path 代数等价；第二条对应 `x2-(1-w)*(x2-x1)`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；ND；每 Tensor rank 0–8 | FP16、FP32、BF16 | 非空且不支持空 Tensor；列表内 dtype 一致；可非连续，OpDef `AutoContiguous()`。 |
| `x2` | 与 `x1` 等长且对应 shape 相同 | 与 `x1` 相同 | 非空；输入可非连续。arch35 Tiling 不自行读取其实例数/shape，依赖 API/框架校验。 |
| `weight` | 与 `x1` 等长且对应 shape 相同 | 与 `x1` 相同 | 非空；Tiling 显式校验实例数等于 `x1`；weight 不限 `[0,1]`。 |
| `out` | 与 `x1` 对应的 ND TensorList | 与 `x1` 相同 | 非空、必须连续；公开 API 允许 shape size≥输入，图侧 InferShape 复制输入 shape；K1写输入元素数。 |
| `workspaceSize`/`executor` | 第一段输出 | `uint64_t*`/`aclOpExecutor**` | arch35 算子节点 workspace 为 0；框架 Contiguous 可能增加 executor 总 workspace，固定生成层不可见。 |
| `workspace`/`stream` | 第二段输入 | opaque/stream | entry 保留 workspace 参数但 body 不读；所有 GM 对象需活到 stream 完成。 |

### 入口与版本

实际公开入口是无后缀的 `aclnnForeachLerpListGetWorkspaceSize`/`aclnnForeachLerpList`。公开 README/API 明确支持 Ascend 950PR/950DT；OpDef 添加 `ascend950`，Host CMake 映射 `ascend950→arch35`，Kernel CMake 选择 arch35 entry，ascend950 binary config 含三种 dtype。因此 950 公开支持与实现接线一致。自动生成 ACLNN inner bridge 不在固定仓，作为框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开第一段检查空指针、FP16/FP32/BF16、四组 dtype 与列表内部 dtype、对应 shape、rank≤8，并公开对应错误码。
2. OpDef 为所有动态端口声明 ND、三种 dtype、`AutoContiguous()`；输入连续化属于生成 bridge，输出非连续由公开约束排除。
3. `InferShape4ForeachCommon` 要求 `x1` 实例数等于输出数并逐项复制 shape/dtype。
4. `ForeachLerpListTilingFunc` 取得正的 AIV core 数和 UB；读取 `x1` 实例数并以 `static_cast<int32_t>(tensorNum)>256` 拒绝常规超限，显式要求 weight 实例数等于 `x1`，清零256项累计数组，遍历 `x1` shape求累计元素及首项 dtype。获取 platform/context/POD/local-memory/workspace 失败返回 `GRAPH_FAILED`。
5. 固定 arch35 Tiling 未显式读取 `x2`/`out` 实例数、对应 shape、全列表 dtype；这些条件来自公开 API/OpDef/InferShape 边界，不能把 Tiling 单独当作完整验证器。

### 参数、Workspace 与生命周期

令 AIV 核数 `C>0`、总元素数 `E`。若报告 UB `U≤128KiB`，Host 把工作值改为 `256KiB`；随后 `usableUb=U_work-128KiB` 并调用 `SetLocalMemorySize(usableUb)`。SIMT body 没有显式 shared/dynamic UB 对象，因此这是 launch local-memory 配额而非可在源码中逐区计账的 buffer。Host 计算 `perCore=max(1024,ceil(E/C))` 再向上32 elements对齐，`blockDim=needCore=min(C,max(1,ceil(E/perCore)))`。算子 workspace slice 为0；POD、四个 TensorList 指针表及各实际 Tensor 由 executor/caller持有到 K1完成。

### 启动图与失败传播

```text
aclnnForeachLerpListGetWorkspaceSize
  -> 自动生成 executor/Contiguous（固定仓框架边界）
  -> ForeachLerpListTilingFunc
       -> blockDim=needCoreNum, localMemory=usableUb, workspace=0, dtype key 0/1/2
aclnnForeachLerpList(..., stream)
  -> K1: foreach_lerp_list<schMode><<<needCoreNum,stream>>>(x1,x2,w,y,0,tiling)
       -> Process<T>
          -> totalElements==0: return
          -> E<=INT32_MAX: VF_CALL<Op...<T,int32_t>>(Dim3(1024))
          -> else: VF_CALL<Op...<T,int64_t>>(Dim3(512))
```

单节点图没有中间 workspace/GM 边。K1 直接读三路 GM 并写 y GM；各 SIMT thread 拥有互斥的 grid-stride 索引，无 atomic/barrier。Tiling失败阻断launch；unsupported dtype 的 `GetTilingKeyByDtype` 默认返回0，因此实际安全边界依赖公开 OpDef/API只允许三种 dtype。生成 bridge 的launch返回传播固定仓不可见。

## Route 与参数设计

### 分核与资源计算

`perCore=align_up(max(1024,ceil(E/C)),32)`，`B=min(C,max(1,ceil(E/perCore)))`，Host以 `B` 设置 outer `blockDim`。Device不读取 POD 的 `needCoreNum/perCoreElements`，而由 `GetBlockNum()` 观察实际 blockDim。对 `E≤INT32_MAX`，每 block 1024 threads，`globalTid=blockIdx*1024+threadIdx`、`stride=1024*B`；对更大 E，每 block 512 threads，索引和stride用int64。每个有效flatIdx线性扫描最多`tensorCount`个累计端点，再访问对应Tensor。E=0仍设置B≥1，但`Process`在VF_CALL前返回。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16/int32 | mode0；`T=half,IDX=int32_t` | dtype FP16，且 `0<E≤INT32_MAX` | 同dtype检查大索引 R-002 |
| 2 | R-002 FP16/int64 | mode0；`T=half,IDX=int64_t` | dtype FP16，且 `E>INT32_MAX` | 无 fallback |
| 3 | R-003 FP32/int32 | mode1；`T=float,IDX=int32_t` | dtype FP32，且 `0<E≤INT32_MAX` | R-004 |
| 4 | R-004 FP32/int64 | mode1；`T=float,IDX=int64_t` | dtype FP32，且 `E>INT32_MAX` | 无 fallback |
| 5 | R-005 BF16/int32 | mode2；`T=bfloat16_t,IDX=int32_t` | dtype BF16，且 `0<E≤INT32_MAX` | R-006 |
| 6 | R-006 BF16/int64 | mode2；`T=bfloat16_t,IDX=int64_t` | dtype BF16，且 `E>INT32_MAX` | 无 fallback |

```text
dtype: FP16->mode0/half, FP32->mode1/float, BF16->mode2/bfloat16_t
then E==0 -> Process return（不形成Device计算Route）
else E<=INT32_MAX -> IDX=int32_t, threads=1024
else -> IDX=int64_t, threads=512
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | mode0/int32 | FP16、950、合法列表、`0<E≤INT32_MAX` | K1 SIMT VF，B×1024 | `Process<half>`→`Op<half,int32_t>` | GM scalar→FP32 thread scalars→GM；无同步 | `flatIdx<E`；无fallback |
| R-002 | mode0/int64 | FP16、950、合法列表、`E>INT32_MAX` | K1，B×512 | `Op<half,int64_t>` | 同上，64-bit index | 同上 |
| R-003 | mode1/int32 | FP32、其余同R-001 | K1，B×1024 | `Op<float,int32_t>` | FP32标量全程 | 同上 |
| R-004 | mode1/int64 | FP32、`E>INT32_MAX` | K1，B×512 | `Op<float,int64_t>` | FP32+64-bit index | 同上 |
| R-005 | mode2/int32 | BF16、`0<E≤INT32_MAX` | K1，B×1024 | `Op<bfloat16_t,int32_t>` | BF16升FP32后降回 | 同上 |
| R-006 | mode2/int64 | BF16、`E>INT32_MAX` | K1，B×512 | `Op<bfloat16_t,int64_t>` | 同上，64-bit index | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 每核最少1024 | `perCore=max(1024,ceil(E/C))`后align32 | 控制outer block数，避免极小域占满全部核 | 全部 | B最少1 | `ForeachLerpListTilingFunc` |
| int32/int64阈值 | `E<=INT32_MAX` | int32每VF 1024 threads；int64为控制索引成本将thread数减为512 | 六Route | 越界切换对应int64 Route | `THREAD_NUM_VF`、`Process` |
| Tensor上限 | `static_cast<int32_t>(tensorNum)>256` | POD累计数组固定256项 | 全部 | 常规>256返回失败；极大uint64先窄化的行为不可依赖 | `MAX_TENSOR_NUM...`、TilingFunc |
| weight实例数 | `weightTensorNum==tensorNum` | flatIdx映射后对四表使用同一tensorId | 全部 | `GRAPH_FAILED` | `ForeachLerpListTilingFunc` |
| x2/out一致性 | arch35 Tiling无显式检查 | K1同一tensorId/localIdx访问四表 | 全部 | 依赖API/框架；Tiling无fallback | API文档、`OpForeachLerpListSimtKernel` |
| local memory | `usableUb=U_work-128KiB`，U≤128KiB时U_work=256KiB | 预留DCache；固定body无显式local数组 | 全部 | SetLocalMemorySize失败→GRAPH_FAILED | `DCACHE_SIZE/GetPlatformInfoFallback` |
| 空域 | `E==0` | 避免VF_CALL与Tensor地址访问 | 全部 | 直接return，输出无写入 | `Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachLerpListTilingDataHost`/Device同布局类 | `needCoreNum` | int32/32 | offset0；struct align8 | blocks | `static_cast<int32_t>(needCoreNum)` | body不读；context blockDim消费同值 | C正数，B≤C |
| 同上 | `tensorCount` | int32/32 | offset4 | Tensor count | `static_cast<int32_t>(tensorNum)` | `Process`→VF arg→tensor scan | 活动累计数组项数 |
| 同上 | `totalElements` | int64/64 | offset8 | elements | storage numel累加 | `Process`选IDX并作为VF loop上界 | 累加溢出未检查 |
| 同上 | `perCoreElements` | int64/64 | offset16 | elements/core | align32(max(1024,ceil(E/C))) | body不读 | 仅Host生成blockDim的审计字段 |
| 同上 | `cumulativeOffsets[256]` | int64[256] | offset24，extent2048 B；sizeof2072，align8 | inclusive-prefix end/exclusive flat boundary | `total+=numel; cumulative[i]=total`；其余清0 | VF线性查找`flatIdx<cum[t]` | 仅前tensorCount项活动 |
| `ForeachLerpListCompileInfo` | `coreNum` | int64/64 | offset0；sizeof16，align8 | AIV cores | TilingParse从平台写 | Tiling fallback读 | 必须>0 |
| `ForeachLerpListCompileInfo` | `ubSize` | int64/64 | offset8 | bytes | TilingParse从平台UB写 | Tiling fallback/local quota | 必须>0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/002 FP16 | mode0 + E阈值 | `arch35/foreach_lerp_list_tiling.cpp::ForeachLerpListTilingFunc` | explicit Host/Device POD | `arch35/foreach_lerp_list.cpp::foreach_lerp_list<0>` | `Process<half>`→`Op<half,int32/int64>`→address/lerp helper | ascend950 arch35 CMake+config |
| R-003/004 FP32 | mode1 + E阈值 | 同上 | 同布局 | `foreach_lerp_list<1>` | `Process<float>`→两个IDX实例 | 同上 |
| R-005/006 BF16 | mode2 + E阈值 | 同上 | 同布局 | `foreach_lerp_list<2>` | `Process<bfloat16_t>`→两个IDX实例 | 同上 |

## Kernel 实现

### 公共执行流程

entry 由 binary template `schMode` 绑定 `DTYPE_X1`，解析显式 POD 后调用 `Process<T>`。Process从GM POD读取E、tensorCount、累计数组；空域直接返回，否则按E阈值选择IDX_T与threads。VF body的TensorList协议为：首`uint64_t`是指针表相对列表首址的byte offset，右移3转索引后取第tensorId个64-bit Tensor地址。每个线程以`threadNum*blockNum`为stride遍历互斥flatIdx，线性扫描累计端点，标量load三输入、FP32计算并cast/store；没有shared memory、barrier、atomic或跨线程交接。

### Route R-001：FP16/int32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/foreach_lerp_list_tiling.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | mode0 |
| Router | `op_kernel/arch35/foreach_lerp_list.cpp` | `foreach_lerp_list<0>` | binary `DTYPE_X1=half` | `Process<half>` |
| Body | `op_kernel/arch35/foreach_lerp_list_simt.h` | `Process` | `0<E<=INT32_MAX` | `IDX_T=int32_t`, threads1024 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_list<0> -> Process<half>
  -> VF_CALL<OpForeachLerpListSimtKernel<half,int32_t>>(Dim3(1024))
    -> scan cumulativeOffsets -> SimtGetTensorAddr<half> x4
    -> cast x1/x2/w to float -> LerpCompute<float> -> cast half -> y GM
```

##### 源码等价伪码

```text
for flat = blockIdx*1024+threadIdx; flat<E; flat += 1024*blockNum:
  t = first index with flat < cum[t]
  local = flat - (t?cum[t-1]:0)
  a,b,w = float(x1[t][local]),float(x2[t][local]),float(weight[t][local])
  diff=b-a; p1=a+w*diff; p2=b+(w-1)*diff
  y[t][local] = half(abs(w)<0.5 ? p1 : p2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Tiling GM | offsets 8/4/24 | 2072 B | POD | Host | Process/VF | executor | read-only |
| x1/x2/w/y GM | ptrTable[t][local] | E total, perTensor cum差 | FP16/ND | caller/thread | thread/caller | stream | scalar 2B访问 |
| thread scalars | compiler registers不可见 | a,b,w,diff,p1,p2,result | FP32/int32 | loads/arithmetic | store | one iteration | 下一flat复用 |
| local/workspace | launch quota/none | usableUb/0B | opaque | Host | body无显式consumer | K1 | N/A |

##### 分层内存布局

N/A：SIMT body 未声明 AIV `TQue/TBuf` 或 shared/dynamic UB 数组；实际数据路径是 GM scalar load→线程局部 FP32 标量→GM scalar store。Host设置的local-memory配额无固定body显式地址。

##### 字节账与复用顺序

```text
UB_total = N/A（无InitBuffer）
dynamic_ub_launch_quota = usableUb；explicit body allocation = 0 B
workspace_total = 0 B（算子节点）
register_bytes = 固定编译产物不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-thread scalars | 当前flat load/compute | 当前flat store | 下一flat | 单线程程序顺序 | grid-stride索引互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 索引分配 | grid-stride | block/thread ids | loop guard | y元素所有权 | int32；`flat<E`保护tail |
| 计算 | 单线程程序顺序 | FP16 loads/cast | GM store | thread scalars | FP32双公式，cast回half的round细节由工具链定义 |
| 线程间 | N/A | disjoint flat | N/A | GM元素 | 无barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_lerp_list/op_host/arch35/foreach_lerp_list_tiling.cpp` | `ForeachLerpListTilingFunc` | B、POD、mode0、workspace/local |
| Kernel entry/router | `foreach_lerp_list/op_kernel/arch35/foreach_lerp_list.cpp` | `foreach_lerp_list<0>` | T=half |
| Device body/helper | `foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_simt.h` | `Process<half>`、`Op<half,int32_t>`、`LerpCompute` | VF geometry、mapping、FP32计算 |

### Route R-002：FP16/int64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | arch35 Tiling/entry | mode0/`Process<half>` | dtype FP16 | half body |
| Body | `foreach_lerp_list_simt.h` | `Process` else | `E>INT32_MAX` | `IDX_T=int64_t`, threads512 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_list<0> -> Process<half>
  -> VF_CALL<Op<half,int64_t>>(Dim3(512))
  -> same address scan/FP32 LerpCompute/store as R-001
```

##### 源码等价伪码

```text
与R-001相同，global flat/local/prevCum/stride改为int64，threadNum=512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/POD/thread scalars | 同R-001 | E>INT32_MAX | FP16/FP32/int64 | 同R-001 | 同R-001 | 同R-001 | 64-bit索引 |

##### 分层内存布局

N/A：同R-001，无显式UB/shared。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic_ub quota=usableUb且显式使用0B；workspace=0B；register bytes固定仓不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalars | current flat | current store | next flat | program order | 512*B grid-stride互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mapping/compute | int64 grid-stride/program order | ids/loads | store | y element | E>INT32_MAX；FP32 compute→half |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | E/POD/blockDim | E超过int32仍保持Host int64 |
| Kernel entry/router | arch35 entry | `foreach_lerp_list<0>` | half |
| Device body/helper | arch35 simt header | `Op<half,int64_t>` | 512线程与64-bit索引 |

### Route R-003：FP32/int32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | arch35 Tiling | `GetTilingKeyByDtype(DT_FLOAT)` | FP32 | mode1 |
| Router/Body | arch35 entry/simt | `Process<float>`且`E<=INT32_MAX` | 非空 | `Op<float,int32_t>`,1024线程 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_list<1> -> Process<float> -> VF_CALL<Op<float,int32_t>>(1024)
  -> GM float loads -> LerpCompute<float> -> GM float store
```

##### 源码等价伪码

```text
使用R-001的flat/tensor映射；a/b/w本身为float，计算两条FP32 path后直接store float。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/w/y GM | ptrTable[t][local] | per cum | FP32/ND | caller/thread | thread/caller | stream | scalar4B |
| thread scalars/POD | 同R-001 | one iteration/2072B | FP32/int32/POD | loads/Host | arithmetic/VF | iteration/K1 | no shared |

##### 分层内存布局

N/A：GM→线程FP32标量→GM，无显式UB/shared。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic quota=usableUb(explicit 0B)；workspace=0B；register bytes不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 scalars | loads/LerpCompute | store | next flat | program order | disjoint flat |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all | grid-stride + program order | ids/FP32 ops | store | one GM element | FP32，无cast，tail由flat<E |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | mode1/E | FP32 key与geometry |
| Kernel entry/router | arch35 entry | `foreach_lerp_list<1>` | T=float |
| Device body/helper | arch35 simt | `Op<float,int32_t>` | 1024线程直接FP32 |

### Route R-004：FP32/int64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | arch35 Tiling/entry | mode1/float | dtype FP32 | `Process<float>` |
| Body | arch35 simt | `E>INT32_MAX` | 大域 | `Op<float,int64_t>`,512线程 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_list<1> -> Process<float> -> VF_CALL<Op<float,int64_t>>(512) -> same lerp/store
```

##### 源码等价伪码

```text
R-003映射的64-bit/512-thread版本。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/POD/scalars | 同R-003 | E>INT32_MAX | FP32/int64 | same | same | same | 64-bit index |

##### 分层内存布局

N/A：无显式UB/shared。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic quota=usableUb(explicit 0B)；workspace=0B；register不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalars | current flat | store | next flat | program order | disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all | int64 grid-stride | ids/ops | store | y | FP32；flat<E |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | mode1/E | big-domain parameters |
| Kernel entry/router | arch35 entry | mode1 | float |
| Device body/helper | arch35 simt | `Op<float,int64_t>` | 512-thread int64 |

### Route R-005：BF16/int32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | arch35 Tiling | `GetTilingKeyByDtype(DT_BF16)` | BF16 | mode2 |
| Router/Body | arch35 entry/simt | `Process<bfloat16_t>`且E≤INT32_MAX | 非空 | `Op<bfloat16_t,int32_t>`,1024 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_list<2> -> Process<bfloat16_t>
  -> VF_CALL<Op<bfloat16_t,int32_t>>(1024)
  -> BF16 loads -> FP32 LerpCompute -> BF16 store
```

##### 源码等价伪码

```text
与R-001相同，仅存储类型为BF16；所有a/b/w/path为FP32线程标量。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | ptrTable[t][local] | per cum | BF16/ND | caller/thread | thread/caller | stream | scalar2B |
| scalars/POD | iteration/offsets | FP32 scalars/2072B | FP32/int32/POD | load/Host | compute/VF | iteration/K1 | no shared |

##### 分层内存布局

N/A：GM BF16→线程FP32→GM BF16。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic quota=usableUb(explicit0B)；workspace=0B；register不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 scalars | cast/load/compute | BF16 cast/store | next flat | program order | disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promotion/demotion | scalar static_cast | BF16 load/FP32 result | Lerp/store | thread | cast round模式固定源码未另写；flat<E |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | mode2/E | BF16 key |
| Kernel entry/router | arch35 entry | mode2 | bfloat16_t |
| Device body/helper | arch35 simt | `Op<bfloat16_t,int32_t>` | 1024-thread FP32 compute |

### Route R-006：BF16/int64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | arch35 Tiling/entry | mode2/BF16 | dtype BF16 | `Process<bfloat16_t>` |
| Body | arch35 simt | E>INT32_MAX | 大域 | `Op<bfloat16_t,int64_t>`,512 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_list<2> -> Process<bfloat16_t> -> VF_CALL<Op<bfloat16_t,int64_t>>(512)
  -> same scan/FP32 lerp/BF16 store
```

##### 源码等价伪码

```text
R-005的64-bit/512-thread版本。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/POD/scalars | 同R-005 | E>INT32_MAX | BF16/FP32/int64 | same | same | same | 64-bit index |

##### 分层内存布局

N/A：无显式UB/shared。

##### 字节账与复用顺序

```text
UB_total=N/A；dynamic quota=usableUb(explicit0B)；workspace=0B；register不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalars | current flat | store | next flat | program order | disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all | int64 grid-stride/program order | ids/BF16 loads | store | y | FP32 compute→BF16；flat<E |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | mode2/E | big BF16 domain |
| Kernel entry/router | arch35 entry | mode2 | BF16 |
| Device body/helper | arch35 simt | `Op<bfloat16_t,int64_t>` | 512-thread int64/FP32 compute |

## 算子特定优化

### Trick T-001：以权重幅值切换等价公式稳定端点附近计算

#### 触发与源码依据

- 适用 Route：R-001–R-006。
- 触发条件：每个有效元素；`abs(weight)<0.5` 选 path1，否则选 path2。
- Golden 来源：[`foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_simt.h:62-68`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_simt.h#L62-L68)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__simt_callee__ inline CalcT LerpCompute(CalcT startVal, CalcT endVal, CalcT weightVal)
{
    CalcT oneCalc = static_cast<CalcT>(1.0);
    CalcT diffVal = endVal - startVal;
    CalcT path1 = startVal + weightVal * diffVal;
    CalcT path2 = endVal + (weightVal - oneCalc) * diffVal;
    return (FabsCalc<CalcT>(weightVal) < oneCalc * static_cast<CalcT>(0.5)) ? path1 : path2;
}
```

#### 机制与设计原因

每线程只保留标量 `diff/path1/path2`，不物化GM/UB中间 Tensor。weight靠近0时path1从x1出发；靠近1时path2从x2出发，从源码可证明其目的是用权重幅值决定数值顺序，同时保持两式代数等价。三种存储dtype都使用`CalcT=float`，因此FP16/BF16只在边界cast；索引Route不改变数值顺序。

#### 迁移边界

- 适用：逐元素lerp、每线程独占输出元素的SIMT实现。
- 前提：所有输入可升为共同CalcT；阈值、严格小于关系与两条path保持一致。
- 失效：若需要与只算单一路径的实现逐bit一致，不能直接迁移；若换更宽/更窄CalcT需重新验证round行为。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把条件改为只判断`weight<0.5`（会改变负weight），也不得交换path与分支方向。

## 模板与实现对比

| 对比维度 | R-001/003/005 int32 | R-002/004/006 int64 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `0<E≤INT32_MAX` | `E>INT32_MAX` | 索引可表示范围 |
| 模板与 dtype | half/float/bfloat16 × int32 | 同三dtype × int64 | mode选存储T，E选IDX_T |
| 循环与分核 | B blocks×1024 threads | B×512 | `THREAD_NUM_VF=sizeof(IDX)==4?1024:512` |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅线程标量；local quota无显式使用 | 同，索引更宽 | 无shared/通信 |
| 内存总占用与复用 | workspace0，UB N/A | 同 | 单线程标量覆盖 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar GM load/store | 同 | grid-stride |
| 同步、精度与 tail | 无barrier/atomic；所有dtype CalcT=float；flat<E | 同 | 互斥索引 |
| fallback/失败边界 | E=0 return | 无更宽索引fallback | tensor/POD/总和上限 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持；公开950表+ascend950接线 |
| Host/Launcher | arch22公共Foreach Tiling | 同A2 | arch35专用Tiling，显式2072B POD |
| Kernel/资源 | AIV Vector Queue/TBuf | 同A2 | SIMT VF，GM标量/线程FP32，无AIV Queue |

## 限制与失败边界

- 公开接口仅支持ND FP16/FP32/BF16，三输入列表等长且对应shape相同，四组dtype一致，rank≤8，输出必须连续且不支持空Tensor。
- arch35 Tiling只显式检查x1 tensor数上限与weight实例数；x2/out实例数、对应shape和所有dtype一致性依赖ACLNN/OpDef/InferShape框架边界，复用Tiling时必须补齐等价前置条件。
- `totalElements` int64累加无显式溢出检查；`tensorNum`在与256比较前窄化为int32；这些超大输入行为不可依赖。
- `needCoreNum/perCoreElements`写入POD但Device不读；实际geometry以context blockDim及SIMT `GetBlockNum()`为准。
- entry 的模板形参 `schMode` 不在函数体内分支；实际存储类型来自每个 binary specialization 的 `DTYPE_X1`。复用构建接线时必须保持 Host mode、binary config dtype 与 `DTYPE_X1` 一致。
- 自动生成ACLNN bridge、Contiguous workspace公式、launch返回传播、编译后寄存器字节数固定仓不可见；本次未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_lerp_list/docs/aclnnForeachLerpList.md` | 两段式原型/产品表/错误码 | 公开API/950支持/约束 |
| 入口声明/约束 | `foreach/foreach_lerp_list/README.md` | 产品支持/公式 | 语义与950支持 |
| 入口实现 | `foreach/foreach_lerp_list/op_host/foreach_lerp_list_def.cpp` | `ops::ForeachLerpList` | dtype/ND/AutoContiguous/ascend950 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | ForeachLerpList InferShape注册 | 输出shape/dtype |
| Host/Tiling/Launcher | `foreach/foreach_lerp_list/op_host/arch35/foreach_lerp_list_tiling.cpp` | `ForeachLerpListTilingFunc/TilingParse` | 校验、geometry、POD/key/local/workspace |
| Host/Tiling/Launcher | `foreach/foreach_lerp_list/op_host/arch35/foreach_lerp_list_tiling.h` | Host POD/compileInfo | ABI字段 |
| Host/Tiling/Launcher | `foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_tiling_data.h` | Device POD | Host/Device同布局 |
| Host/Tiling/Launcher | `foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_tiling_key.h` | schMode 0/1/2 | dtype模板key |
| Kernel entry/body | `foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list.cpp` | templated `foreach_lerp_list` | entry与POD解析 |
| Kernel entry/body | `foreach/foreach_lerp_list/op_kernel/arch35/foreach_lerp_list_simt.h` | `Process/Op.../LerpCompute/SimtGetTensorAddr` | 全部SIMT body/helpers |
| 构建接线 | `foreach/foreach_lerp_list/op_host/CMakeLists.txt` | ascend950→arch35 | Host选择 |
| 构建接线 | `foreach/foreach_lerp_list/op_kernel/CMakeLists.txt` | arch35 kernel source | Device选择 |
| 构建接线 | `foreach/foreach_lerp_list/op_host/config/ascend950/foreach_lerp_list_binary.json` | 三dtype op_list | 950 binary接线 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2072B五字段布局、累计offset | offset/位宽/256项/exclusive累计边界 | Host/Device Tiling structs |
| 调用顺序 | entry→Process→IDX分支→VF_CALL→scan/load/compute/store | 空域return必须在VF_CALL前 | `Process` |
| workspace/内存 | 算子workspace0；无显式UB/shared | local quota与GM/POD lifetime | TilingFunc/VF body |
| 同步 | grid-stride互斥，无atomic/barrier | block/thread到flatIdx公式 | `OpForeachLerpListSimtKernel` |
| 精度与尾块 | CalcT=float、双公式、abs阈值；flat<E | strict `<0.5`、path对应、dtype cast边界 | `LerpCompute`/VF loop |
| 失败边界 | E阈值切IDX、tensor≤256常规检查、weight等长 | x2/out条件需由外层保证；大数窄化/溢出不可依赖 | TilingFunc/Process |