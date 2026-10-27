---
type: Operator
title: "ForeachMaximumList 950 ACLNN 算子知识"
description: "说明 aclnnForeachMaximumList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/docs/aclnnForeachMaximumList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_host/foreach_maximum_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_host/arch35/foreach_maximum_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_host/config/ascend950/foreach_maximum_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:45:54Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMaximumList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMaximumListGetWorkspaceSize` + `aclnnForeachMaximumList`，公开 ACLNN 基础版本 |
| 计算语义 | 两个等长 TensorList 的同位置 Tensor 逐元素 maximum；浮点 SIMT body 显式传播任一输入 NaN |
| 输入输出 | ND TensorList；FLOAT32/FLOAT16/BFLOAT16/INT32；Tensor 数 ≤256；逐位置输入 shape/dtype 相同，输出容量足够 |
| 执行类型 | 单 Kernel；外层 AIV block 调用 SIMT VF，核心计算节点执行族为 `SIMT`，无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 FP32；R-002 FP16；R-003 BF16；R-004 INT32（四个 schMode specialization） |
| 核心资源 | TensorList GM 描述符；每个外层 block 对相交 Tensor 段发起 512-thread VF；无 TQue/TBuf/算子 workspace；local-memory budget=`UB-128 KiB` |
| 来源闭包 | source_closed：公开 ACLNN 文档→生成式 ACLNN/连续化框架边界→950 OpDef/config→arch35 Tiling/POD/template selector→Kernel specialization→SIMT VF body→CMake 接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x1=[x1_0,...,x1_(n-1)]` 和 `x2=[x2_0,...,x2_(n-1)]`，计算 `y_i[j]=maximum(x1_i[j],x2_i[j])`。FP32/FP16/BF16 body 使用 `(isnan(a) || a>b) ? a : b` 的等价形式：`a` 为 NaN 时返回 `a`；`b` 为 NaN 且 `a` 非 NaN 时比较为 false、返回 `b`，所以任一输入 NaN 都传播。INT32 使用 `(a>b)?a:b`。输出大于输入的尾部未被 Kernel 写入。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；ND、0–8 维 | FLOAT32/FLOAT16/BFLOAT16/INT32 | 非空；支持零元素 Tensor；列表内 dtype 一致；可非连续，OpDef `AutoContiguous`；950 Tiling 明确限制实例数 ≤256 |
| `x2` | 与 `x1` 等长且逐位置 shape 一致；ND | 与 `x1` 相同 | 非空；可非连续，生成式连续化是框架边界 |
| `out` | 与输入对应；size ≥输入；ND | 与 `x1` 相同 | 非空；公开 ACLNN 不支持非连续输出；多余尾部不写 |
| `workspaceSize/executor` | 第一段输出 | `uint64_t*`/`aclOpExecutor**` | 950 算子节点 workspace 为 0；生成式连续化可能使 ACLNN 总 workspace 非零 |
| `workspace/workspaceSize/executor/stream` | 第二段参数 | 标准 ACLNN 类型 | 在指定 stream 提交单 Kernel 任务；调用者维持资源生命周期 |

### 入口与版本

固定 Commit 只公开无版本后缀入口，故它是 950 实际最新公开版本。README 与 ACLNN 文档明确支持 Ascend 950PR/950DT；OpDef 添加 `ascend950` config，Host CMake 将其映射到 `arch35`，950 binary config 列出四种 dtype，Kernel CMake 显式选择 `arch35/foreach_maximum_list.cpp`。没有“文档不支持、实现接线支持”的冲突。

生成式 ACLNN inner bridge、输入 Contiguous 和 executor launch wrapper不在固定仓；按框架边界处理。算子自有 950 Tiling、POD producer、template key、Kernel entry 与 SIMT body 均可追溯。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 ACLNN 第一段校验空指针、dtype 范围/一致性、列表内 dtype、逐位置 shape 和 8 维上限；失败返回文档列出的 ACLNN 参数或 Tiling 错误。
2. OpDef 声明四 dtype、ND、动态 rank/shape、`AutoContiguous`，并为 ascend950 设置动态编译与精度降级标志。输入非连续化的具体任务由生成式框架拥有。
3. `ForeachMaximumListTilingFunc` 取得 AIV core 数与 UB 大小，二者任一为 0 则失败；从动态输入 0 的实例信息取 `tensorNum`，超过 256 失败。
4. Tiling 遍历 `x1` storage shape，写 `tensorSizes[i]` 并累加 `totalElements`；dtype 从 `x1[0]` descriptor 取得。该函数本身不再次读取 x2/out shape/dtype，依赖入口/OpDef 框架的已完成校验。
5. 按 dtype 选择 schMode template key；不支持 dtype 返回 `GRAPH_FAILED`。Tiling/POD 任意空指针或 `memset_s` 失败都阻断 launch。

### 参数、Workspace 与生命周期

设 `N=sum tensorSizes[i]`、AIV core 数 `C`、平台 UB 字节数 `U`：

- 若 `N=0`：`needCoreNum=1`、`perCoreElements=lastCoreElements=0`、`blockDim=1`。
- 若 `N>0`：`perCoreElements=ceilAlign(min(65536,ceil(N/C)),32)`；`needCoreNum=ceil(N/perCoreElements)`；`lastCoreElements=N-perCoreElements*(needCoreNum-1)`；`blockDim=needCoreNum`。
- `workspaces[0]=0`。输入连续化若需要空间，由不可见 ACLNN executor 另行计入总 workspace。
- `SetLocalMemorySize(U-128 KiB)` 为 Kernel 设置 local-memory budget；源码在做无符号减法前未检查 `U>=128 KiB`。
- 每个 block 的扁平覆盖区间 `[blockIdx*perCoreElements, min(start+perCoreElements,N))`，最后一核显式以 `N` 收口；POD 的 `lastCoreElements` 被 Host 写入但 Device body未读取。
- TensorList 描述符、数据和 executor/workspace 必须保持到 stream 任务完成；单 Kernel 内没有跨节点 workspace owner/lifetime。

### 启动图与失败传播

```text
aclnnForeachMaximumListGetWorkspaceSize
  -> ACLNN 参数检查 / x1,x2 AutoContiguous（框架边界）
  -> ForeachMaximumListTilingFunc
     -> ForeachMaximumListTilingData + schMode key + blockDim + localMemorySize
aclnnForeachMaximumList(..., stream)
  -> [K-001] foreach_maximum_list<schMode><<<needCoreNum, stream>>>
       -> Process<T,schMode>
          -> 对每个与本核扁平区间相交的 Tensor 段：VF_CALL<对应 SIMT 函数>(Dim3(512), cnt,...)
       -> GM y
```

| 节点/边 | producer | consumer | offset/extent、dtype/layout | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| ACLNN→K-001 | executor/Tiling | specialized entry | x1/x2/y TensorList descriptors、POD、blockDim | 同一 stream；调用者持有对象 | API/Tiling 失败阻断 K-001 |
| 外层 block→VF | `Process` | dtype 对应 SIMT VF | 当前 Tensor 交集 `[localOff,localOff+cnt)`；T/flatten ND | 同一 Kernel/block 的直接调用；指针只在 VF 调用期间使用 | 无独立节点错误返回 |
| VF→y GM | SIMT thread | 用户/后续 stream task | `yList[t]+localOff+idx` | Kernel 完成及 stream 顺序后可见；GM 由用户拥有 | Kernel runtime 错误走 ACLNN launch 边界，生成实现不可见 |
| workspace | Tiling | 无 | 0 B | N/A | N/A |

## Route 与参数设计

### 分核与资源计算

950 先按所有 Tensor 总元素形成扁平域分核，但 Device 不构造扁平 GM 临时区。每个 block 遍历 `tensorSizes[0:tensorNum)`，通过累计 `[tStart,tEnd)` 与核区间求交 `[oS,oE)`；交集非空才从三个 `ListTensorDesc` 取得该 Tensor 地址并加 `localOff=oS-tStart`。每次交集调用固定 512 个 SIMT thread，线程 `idx=threadIdx; idx<cnt; idx+=threadNum`，因此尾段由循环条件直接保护。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | schMode 0 | `dataType==DT_FLOAT` | 下一 dtype 分支 |
| 2 | R-002 FP16 SIMT | schMode 1 | `dataType==DT_FLOAT16` | 下一 dtype 分支 |
| 3 | R-003 BF16 SIMT | schMode 2 | `dataType==DT_BF16` | 下一 dtype 分支 |
| 4 | R-004 INT32 SIMT | schMode 3 | `dataType==DT_INT32` | 其它 dtype Tiling 失败 |

```text
DT_FLOAT   -> GET_TPL_TILING_KEY(schMode=0) -> entry<0> -> Process<float,0>
DT_FLOAT16 -> GET_TPL_TILING_KEY(schMode=1) -> entry<1> -> Process<half,1>
DT_BF16    -> GET_TPL_TILING_KEY(schMode=2) -> entry<2> -> Process<bfloat16_t,2>
DT_INT32   -> GET_TPL_TILING_KEY(schMode=3) -> entry<3> -> Process<int32_t,3>
otherwise  -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | 950、FLOAT32、tensorNum≤256、入口约束成立 | 单 K-001；每交集 VF 512 threads | `Process<float,0>`→`ForeachMaximumListSimtFp32<float>` | 直接 GM load/compare/store；无 UB Queue/barrier/atomic | thread-stride `idx<cnt`；无 fallback |
| R-002 FP16 | schMode 1 | 950、FLOAT16、同上 | 同上 | `Process<half,1>`→`ForeachMaximumListSimtFp16` | 直接 GM；`__hisnan` | 同上 |
| R-003 BF16 | schMode 2 | 950、BF16、同上 | 同上 | `Process<bfloat16_t,2>`→`ForeachMaximumListSimtBf16` | 每线程 BF16→FP32 标量转换后比较，原 BF16 值写回 | 同上 |
| R-004 INT32 | schMode 3 | 950、INT32、同上 | 同上 | `Process<int32_t,3>`→`ForeachMaximumListSimtInt32` | 直接整数比较，无 NaN | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数 | `tensorNum<=256` | POD 固定 `tensorSizes[256]` | 全部 | `GRAPH_FAILED` | `ForeachMaximumListTilingFunc` |
| 平台资源非零 | `coreNumAiv!=0 && ubSize!=0` | blockDim 和 local-memory budget producer | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| 每核目标上限 | `perCore=min(65536,ceil(N/C))` 后 32 元素对齐 | 限制外层 block 扁平覆盖范围 | 全部 | 无 Route 切换；可能产生多于 C 个 block | `ForeachMaximumListTilingFunc` |
| DCache 预留 | `SetLocalMemorySize((uint32_t)(U-128KiB))` | local memory 留出 128 KiB | 全部 | 固定源码无 `U<128KiB` guard | 同上 |
| 空总域 | `N==0` | 避免除法并保留合法 blockDim | 全部 | blockDim=1，Device `coreStart>=coreEnd` 立即返回 | Host Tiling + `Process` |
| 固定 VF 宽度 | `THREAD_NUM=512`、`VF_CALL(Dim3(512),cnt,...)` | SIMT 每线程 stride=512 | 全部 | 短段由空闲线程自然跳过，无 fallback | `foreach_maximum_list_simt.h` |
| dtype specialization | 四路显式 if/else | NaN/转换语义不同，不能合并 body | 各 Route | 其它 dtype Tiling 失败 | Tiling + entry |

### POD/ABI

`ForeachMaximumListTilingData` 是普通标准布局式声明，但固定仓没有 `static_assert(sizeof/offsetof)` 或编译产物；因此不猜测编译器 padding，只记录声明顺序和字段宽度。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMaximumListTilingData` | `needCoreNum` | `int32_t`/32 | 第 1 字段；精确 offset/sizeof 固定仓不可见 | core | 空域置 1；否则 `ceil(N/perCore)` | `Process` 判断最后一核 | >0；与 blockDim 相同 |
| 同上 | `totalElements` | `int64_t`/64 | 第 2 字段；padding 不可见 | 元素 | 累加所有 x1 storage numel | `Process` 最后一核 `coreEnd=N` | 固定源码无加法溢出检查 |
| 同上 | `perCoreElements` | `int64_t`/64 | 第 3 字段 | 元素/core | `ceilAlign(min(65536,ceil(N/C)),32)` 或 0 | `Process` 计算 coreStart/coreEnd | 非空时 32 元素对齐 |
| 同上 | `lastCoreElements` | `int64_t`/64 | 第 4 字段 | 元素 | `N-perCore*(needCoreNum-1)` 或 0 | 无 Device consumer | inactive/冗余字段，不能假定 Kernel读取 |
| 同上 | `tensorNum` | `int32_t`/32 | 第 5 字段 | Tensor 数 | `static_cast<int32_t>(tensorNum)` | `Process` Tensor 循环上界 | Host 先限制 ≤256 |
| 同上 | `tensorSizes[256]` | `int64_t[256]` | 第 6 字段 | 每项元素数 | 逐项 storage shape size | `Process` 累计 Tensor 区间 | 未用项由 memset 置 0且不读取 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | `op_host/arch35/...tiling_arch35.cpp::ForeachMaximumListTilingFunc` | `ForeachMaximumListTilingData` | `op_kernel/arch35/foreach_maximum_list.cpp::foreach_maximum_list<0>` | `foreach_maximum_list_simt.h::Process<float,0>/ForeachMaximumListSimtFp32` | Host/Kernel CMake + ascend950 binary config |
| R-002 FP16 | schMode 1 | 同上 | 同上 | `foreach_maximum_list<1>` | `Process<half,1>/ForeachMaximumListSimtFp16` | 同上 |
| R-003 BF16 | schMode 2 | 同上 | 同上 | `foreach_maximum_list<2>` | `Process<bfloat16_t,2>/ForeachMaximumListSimtBf16` | 同上 |
| R-004 INT32 | schMode 3 | 同上 | 同上 | `foreach_maximum_list<3>` | `Process<int32_t,3>/ForeachMaximumListSimtInt32` | 同上 |

## Kernel 实现

### 公共执行流程

四路共享单 Kernel 外层 `Process<T,schMode>`：根据 blockIdx 得到扁平区间，空区间立即返回；构造三个 `ListTensorDesc`；线性扫描 Tensor sizes，与核区间求交；每个非空交集重绑定三个 GM 指针并发起一次 512-thread SIMT VF。没有 Kernel 间边、workspace、Queue、barrier 或 atomic；不同外层 block 的扁平区间不重叠，所以输出无写冲突。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_maximum_list_def.cpp` | `ForeachMaximumList` | ascend950 + FLOAT | arch35 Tiling |
| Host/Tiling | `foreach_maximum_list_tiling_arch35.cpp` | `ForeachMaximumListTilingFunc` | `dataType==ge::DT_FLOAT` | schMode 0 |
| Router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<schMode>` | `schMode==0` | `Process<float,0>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_maximum_list<0> -> Process<float,0>
  -> per overlapping Tensor segment
     -> VF_CALL<ForeachMaximumListSimtFp32<float>>(Dim3(512),cnt,x1P,x2P,yP)
        -> thread-stride GM load -> isnan(a)||a>b -> GM store
```

##### 源码等价伪码

```text
# 伪码
[coreStart,coreEnd) = block flattened interval
for t in [0,tensorNum):
  overlap = intersect([coreStart,coreEnd), [prefix[t],prefix[t]+tensorSizes[t]))
  if nonempty:
    p1,p2,py = list[t] + overlap.begin-prefix[t]
    parallel_for thread=0..511, idx=thread; idx<cnt; idx+=512:
      a=p1[idx]; b=p2[idx]; py[idx]=(isnan(a)||a>b)?a:b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 GM | `ListTensorDesc.GetDataPtr<float>(t)+localOff+idx` | `cnt` | FP32/ND flatten | 用户 | SIMT thread | 用户 Tensor | 标量直接 load；无显式 DMA/alignment |
| y GM | `GetDataPtr<float>(t)+localOff+idx` | `cnt` | FP32 | SIMT thread | 用户 | 用户 output | 每 idx 唯一 writer |
| 寄存器 | 每线程 `a,b` | 2×FP32 + 比较临时；精确寄存器字节不可见 | FP32 | GM load | compare/store | 单线程单循环迭代 | 下一 idx 覆盖 |
| AIV UB/TQue/TBuf | N/A | 未分配 | N/A | N/A | N/A | N/A | 实际替代路径为 SIMT 直接 GM |
| dynamic UB | 固定源码未提供显式 `dynUBufSize` | N/A；Host 仅设 localMemory=`U-128KiB` | N/A | Host | runtime | Kernel 生命周期 | body 未使用 shared/dynamic UB |
| workspace | N/A | 0 B | N/A | N/A | N/A | N/A | 无交接 |

##### 分层内存布局

N/A：该 SIMT Route 不创建 AIV UB Queue、SIMT shared/dynamic UB 或中间 workspace；数据从 GM 标量加载到线程寄存器并直接写回 GM。

##### 字节账与复用顺序

```text
UB_total = 0 B（源码无 InitBuffer/TQue/TBuf）
dynamic_ub = N/A（固定源码无显式 producer；body 不使用）
local_memory_budget = U - 128 KiB（Host SetLocalMemorySize）
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程寄存器 `a,b` | GM load | isnan/compare/选择 | 下一 `idx+=512` | 顺序线程语义 | store 在下一迭代前完成；不同线程 idx 不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT 映射 | thread-stride loop | VF launch | thread body | 唯一 idx | `idx<cnt` 保护尾段；无 barrier |
| FP32 maximum | `isnan(float(a)) \|\| a>b` | 两 GM load | GM store | 标量结果 | FP32 比较；任一 NaN 传播 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_list_tiling_arch35.cpp` | `ForeachMaximumListTilingFunc` | schMode 0、blockDim、POD、0 workspace |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<0>` | `Process<float,0>` |
| Device body/helper | `op_kernel/arch35/foreach_maximum_list_simt.h` | `Process<float,0>`、`ForeachMaximumListSimtFp32` | Tensor overlap、512 threads、NaN maximum |

### Route R-002：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_maximum_list_tiling_arch35.cpp` | `ForeachMaximumListTilingFunc` | `dataType==DT_FLOAT16` | schMode 1 |
| Router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<1>` | `schMode==1` | `Process<half,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_maximum_list<1> -> Process<half,1>
  -> VF_CALL<ForeachMaximumListSimtFp16>(Dim3(512),cnt,x1P,x2P,yP)
     -> thread-stride half GM load -> __hisnan(a)||a>b -> half GM store
```

##### 源码等价伪码

```text
# 伪码；外层 overlap 同 R-001
for idx=threadIdx; idx<cnt; idx+=512:
  half a=x1[idx], b=x2[idx]
  y[idx]=(__hisnan(a)||a>b)?a:b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | 同 R-001，类型 half | `cnt` | FP16/ND flatten | 用户/SIMT | SIMT/用户 | 用户 Tensor | 标量直接 GM；唯一 idx writer |
| 寄存器 | 每线程 a,b | 2×half，精确分配不可见 | FP16 | load | compare/store | 单迭代 | 下一 idx 覆盖 |
| AIV UB/dynamic UB/workspace | N/A | 0 B 可见分配；workspace 0 | N/A | N/A | N/A | Kernel | direct GM 路线 |

##### 分层内存布局

N/A：无 AIV UB Queue、shared/dynamic UB 或中间 buffer；直接 GM→线程寄存器→GM。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; local_memory_budget=U-128 KiB; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程 a,b | GM load | `__hisnan`/比较 | 下一 idx | 顺序线程语义 | store 完成后迭代 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| FP16 maximum | `__hisnan(a)\|\|a>b` | GM load | store | 标量 | FP16 比较；任一 NaN 传播；`idx<cnt` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_list_tiling_arch35.cpp` | dtype branch | schMode 1 |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<1>` | FP16 Process |
| Device body/helper | `op_kernel/arch35/foreach_maximum_list_simt.h` | `ForeachMaximumListSimtFp16` | 512-thread loop与 NaN 语义 |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_maximum_list_tiling_arch35.cpp` | dtype branch | `dataType==DT_BF16` | schMode 2 |
| Router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<2>` | `schMode==2` | `Process<bfloat16_t,2>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_maximum_list<2> -> Process<bfloat16_t,2>
  -> VF_CALL<ForeachMaximumListSimtBf16>(Dim3(512),cnt,...)
     -> BF16 GM load -> __bfloat162float(a/b) -> isnan(fa)||fa>fb -> 写回所选原 BF16 值
```

##### 源码等价伪码

```text
# 伪码
for idx=threadIdx; idx<cnt; idx+=512:
  bf16 a=x1[idx], b=x2[idx]
  fp32 fa=convert(a), fb=convert(b)
  y[idx]=(isnan(fa)||fa>fb)?a:b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | 同公共路线 | `cnt` | BF16/ND flatten | 用户/SIMT | SIMT/用户 | 用户 | 直接标量 GM |
| 寄存器 | a,b,fa,fb | 2×BF16+2×FP32，物理寄存器字节不可见 | mixed | load/convert | compare/store | 单迭代 | 输出写原 BF16，无降精度 Cast |
| UB/workspace | N/A | 0 B visible | N/A | N/A | N/A | Kernel | direct GM |

##### 分层内存布局

N/A：仅线程寄存器临时值，无源码可见 UB/shared/workspace。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; local_memory_budget=U-128 KiB; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| fa/fb 寄存器 | BF16 conversion | isnan/compare | 下一 idx | 线程顺序 | store 所选 a/b 后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16 compare | `__bfloat162float` + FP32 isnan/`>` | BF16 load | select/store | 每线程标量 | FP32 仅用于判定，输出保留原 BF16 bit value；`idx<cnt` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_list_tiling_arch35.cpp` | BF16 branch | schMode 2 |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<2>` | BF16 Process |
| Device body/helper | `op_kernel/arch35/foreach_maximum_list_simt.h` | `ForeachMaximumListSimtBf16` | FP32 判定、原 BF16 选择写回 |

### Route R-004：INT32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_maximum_list_tiling_arch35.cpp` | dtype branch | `dataType==DT_INT32` | schMode 3 |
| Router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<3>` | `schMode==3` | `Process<int32_t,3>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_maximum_list<3> -> Process<int32_t,3>
  -> VF_CALL<ForeachMaximumListSimtInt32>(Dim3(512),cnt,...)
     -> int32 GM load -> a>b -> int32 GM store
```

##### 源码等价伪码

```text
# 伪码
for idx=threadIdx; idx<cnt; idx+=512:
  int32 a=x1[idx], b=x2[idx]
  y[idx]=(a>b)?a:b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | 同公共路线 | `cnt` | INT32/ND flatten | 用户/SIMT | SIMT/用户 | 用户 | 直接标量 GM；唯一 idx writer |
| 寄存器 | a,b | 2×INT32，物理字节不可见 | INT32 | load | compare/store | 单迭代 | 下一 idx 覆盖 |
| UB/workspace | N/A | 0 B visible | N/A | N/A | N/A | Kernel | direct GM |

##### 分层内存布局

N/A：无 AIV UB Queue、SIMT shared/dynamic UB 或 workspace。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; local_memory_budget=U-128 KiB; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程 a,b | GM load | compare/store | 下一 idx | 线程顺序 | 每 idx 唯一 writer |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| INT32 maximum | 标量 `a>b` | GM load | store | 每线程标量 | 无 cast/round；`idx<cnt` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_list_tiling_arch35.cpp` | INT32 branch | schMode 3 |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<3>` | INT32 Process |
| Device body/helper | `op_kernel/arch35/foreach_maximum_list_simt.h` | `ForeachMaximumListSimtInt32` | 512-thread integer maximum |

## 算子特定优化

### Trick T-001：核区间与 Tensor 区间求交，直接访问离散 TensorList GM

#### 触发与源码依据

- 适用 Route：R-001–R-004
- 触发条件：每个外层 block；仅对 `[coreStart,coreEnd)` 与当前 Tensor 扁平区间的非空交集发起 VF。
- Golden 来源：[`foreach_maximum_list_simt.h:115-132`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_simt.h#L115-L132)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t oS = (coreStart > tStart) ? coreStart : tStart;
int64_t oE = (coreEnd < tEnd) ? coreEnd : tEnd;

if (oS < oE) {
    int64_t localOff = oS - tStart;
    int64_t cnt = oE - oS;
    __gm__ T* x1P = x1List.GetDataPtr<T>(t) + localOff;
```

#### 机制与设计原因

Tiling 只传每个 Tensor 的长度，不创建拼接 workspace。Device 用 prefix sum 把规则的扁平核区间映射回离散列表地址，只对交集段调用 VF；这既允许一个 Tensor 跨核，也允许一个核覆盖多个小 Tensor，并避免将整个 TensorList 先物化为连续大 Tensor。

#### 迁移边界

- 适用：TensorList 各项独立、逐元素、无跨项状态的 SIMT 算子。
- 前提：列表描述符能在 Device 解析真实 GM 地址；POD 长度数组与列表项数一致。
- 失效：需要跨 Tensor 邻域/归约或必须连续 DMA 的算法不能直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：BF16 仅升精度判定，直接写回原始候选值

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：schMode 2 / BF16。
- Golden 来源：[`foreach_maximum_list_simt.h:68-76`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_simt.h#L68-L76)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bfloat16_t a = x1[idx];
bfloat16_t b = x2[idx];
// bfloat16 -> float for NaN check and comparison
float fa = __bfloat162float(a);
float fb = __bfloat162float(b);
y[idx] = (isnan(fa) || fa > fb) ? a : b;
```

#### 机制与设计原因

BF16 转 FP32 只服务于 NaN 检查和大小比较，输出直接选择原始 `a` 或 `b`，不执行 FP32→BF16 的二次舍入，也不需要 UB cast buffer。这保持所选候选的 BF16 表示并缩短数据路径。

#### 迁移边界

- 适用：输出严格等于某一输入候选、升精度只用于谓词判定的 selection 算子。
- 前提：算法没有融合算术，结果无需新数值；NaN tie 规则与本实现一致。
- 失效：输出由算术生成或需 canonical NaN 时必须显式生成/转换结果。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | R-004 INT32 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | DT_FLOAT→0 | DT_FLOAT16→1 | DT_BF16→2 | DT_INT32→3 | dtype 决定唯一 specialization |
| 模板与 dtype | `Process<float,0>` | `<half,1>` | `<bfloat16_t,2>` | `<int32_t,3>` | 标量 intrinsic 不同 |
| 循环与分核 | colspan：四路共享扁平核区间→Tensor 交集→512-thread stride 循环 | 同左 | 同左 | 同左 | 公共 Process |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程 FP32 临时 | half 临时 | BF16+FP32 判定临时 | INT32 临时 | 均无 UB Queue/shared/workspace |
| 内存总占用与复用 | 源码可见 UB 0 | 0 | 0 | 0 | 直接 GM |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + VF_CALL | 同 | BF16 conversion + VF_CALL | scalar GM + VF_CALL | 全部 SIMT |
| 同步、精度与 tail | FP32 NaN maximum | FP16 `__hisnan` | FP32 判定、原 BF16 输出 | INT32 compare | 无 barrier；`idx<cnt` tail |
| fallback/失败边界 | 无 | 无 | 无 | 无 | 其它 dtype Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 Ascend 950PR/950DT |
| Host/Launcher | arch22 公共 foreach Tiling | arch22 公共 foreach Tiling | arch35 专用 Tiling；Tensor 上限 256、0 workspace |
| Kernel/资源 | AIV Queue/Vector，最多 50 Tensor/核 | 同 A2 | SIMT VF 512 threads，直接 GM，无 Queue |

## 限制与失败边界

- 950 `tensorSizes` 固定 256 项，Host 对 `tensorNum>256` 明确失败；A2/A3 是独立的 50 项 POD 路线。
- Tiling 从 x1 取得 tensorNum/shape/dtype，算子自有 950 Tiling 不重复校验 x2/out；这些约束依赖公开 ACLNN/OpDef 框架边界。若绕过公开入口，不得假定安全。
- `totalElements` 是 `int64_t` 累加，固定源码无溢出检查；`SetLocalMemorySize(static_cast<uint32_t>(ubSize-128KiB))` 在减法前也无下界检查。
- `perCoreElements` 的 65536 cap 可使 `needCoreNum` 大于物理 AIV core 数；固定源码只证明 blockDim 取该值，不提供调度性能或上界解释。
- `lastCoreElements` 被写入 POD 但 Device 不读取；不要让派生实现依赖该字段控制 tail。
- 浮点 NaN 传播顺序固定为优先保留 x1 NaN，否则比较并可能选 x2；相等非 NaN 时返回 x2。INT32 相等也返回 x2（数值相同）。
- 公开 ACLNN 生成实现、输入 Contiguous workspace 公式、显式 dynUBufSize 与 launch wrapper 在固定仓不可见，不推测补齐。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_maximum_list/docs/aclnnForeachMaximumList.md` | 两段式 ACLNN 原型 | 公开 API、dtype/shape/error/platform |
| 平台声明 | `foreach/foreach_maximum_list/README.md` | 产品支持表 | Ascend 950 公开支持 |
| OpDef/入口边界 | `foreach/foreach_maximum_list/op_host/foreach_maximum_list_def.cpp` | `ForeachMaximumList` | ascend950 config、dtype、ND、AutoContiguous |
| Host/Tiling | `foreach/foreach_maximum_list/op_host/arch35/foreach_maximum_list_tiling_arch35.cpp` | `GetPlatformInfo`、`ForeachMaximumListTilingFunc` | 检查、分核、workspace、local memory、key、POD producer |
| POD | `foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_tiling_data.h` | `ForeachMaximumListTilingData` | 字段 ABI 声明 |
| template selector | `foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 四种 schMode |
| Kernel entry/body | `foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list.cpp` | `foreach_maximum_list<schMode>` | specialization router |
| Kernel body | `foreach/foreach_maximum_list/op_kernel/arch35/foreach_maximum_list_simt.h` | `Process`、四个 VF | overlap、thread mapping、精度/NaN/tail |
| binary 接线 | `foreach/foreach_maximum_list/op_host/config/ascend950/foreach_maximum_list_binary.json` | 四个 op_list 条目 | 950 四 dtype binary |
| Host 构建 | `foreach/foreach_maximum_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 |
| Kernel 构建 | `foreach/foreach_maximum_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950→arch35 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 用总元素、per-core 粒度和 256 长度表重建离散列表区间 | 字段顺序/位宽、256 上限；`lastCoreElements` inactive | `ForeachMaximumListTilingData` |
| 调用顺序 | 单 Kernel，外层 Tensor overlap 后调用 VF | 每交集独立 512-thread VF；三列表同 index/localOff | `Process<T,schMode>` |
| workspace/内存 | 0 workspace、直接 GM | localMemory=`U-128KiB`；不得虚构 Queue/shared buffer | Tiling + SIMT body |
| 同步 | 无 barrier/atomic，靠不重叠 idx 与 Kernel/stream 边界 | 核区间和线程 stride 必须保持无重叠 | `Process`、VF loops |
| 精度与尾块 | 浮点 NaN propagation；BF16 FP32 predicate；INT32 direct compare | `(isnan(a)\|\|a>b)?a:b` 顺序、`idx<cnt` | 四个 VF body |
| 失败边界 | resource/tensorNum/dtype 失败阻断 launch | x2/out 校验仍依赖 ACLNN 框架；N 累加无溢出检查 | `GetPlatformInfo`、Tiling dtype branch |