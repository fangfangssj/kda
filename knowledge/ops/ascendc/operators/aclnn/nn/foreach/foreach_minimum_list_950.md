---
type: Operator
title: "ForeachMinimumList 950 ACLNN 算子知识"
description: "说明 aclnnForeachMinimumList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/docs/aclnnForeachMinimumList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_host/foreach_minimum_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_host/arch35/foreach_minimum_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_host/config/ascend950/foreach_minimum_list_binary.json
    role: implementation
created_at: "2026-08-25T15:46:43Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMinimumList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMinimumListGetWorkspaceSize` + `aclnnForeachMinimumList`，公开两段式 ACLNN；无更高版本后缀。 |
| 计算语义 | 对两个等长 TensorList 的同位置 Tensor 逐元素计算最小值；浮点路线显式传播任一输入 NaN。 |
| 输入输出 | ND TensorList，FP32/FP16/BF16/INT32；同位置 shape 和 dtype 一致；输入允许框架连续化，输出必须连续；最多 256 个 Tensor。 |
| 执行类型 | 单 Device 节点；AIV 外层按串联元素域分块，每个重叠 Tensor 片段用 512-thread SIMT VF 计算，执行族记为 SIMT。 |
| Kernel 路线 | R-001 FP32、R-002 FP16、R-003 BF16、R-004 INT32；四个 `schMode` 编译期实例。 |
| 核心资源 | TensorList GM 地址表、逐线程标量寄存器；无显式 TQue/TBuf、无算子 workspace；Host 为 local memory 保留 128 KiB DCache 后设置上限。 |
| 来源闭包 | source_closed：公开 ACLNN→框架 bridge 边界→950 OpDef→arch35 Tiling/POD/template key→kernel entry→四个 SIMT body→950 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设列表长度 `n`，`S_i=numel(x1_i)=numel(x2_i)`：

```text
for i in [0,n), e in [0,S_i): y_i[e] = minimum(x1_i[e], x2_i[e])
```

FP32/FP16 的实现是 `(isnan(a) || a < b) ? a : b`；因此 `a` 为 NaN 时返回 `a`，`b` 为 NaN 且 `a` 非 NaN 时比较为 false、返回 `b`，均传播 NaN。BF16 将 `a/b` 各自转为 FP32 仅用于 `isnan` 和 `<`，最终直接选择原始 BF16 `a` 或 `b`，不重新舍入。INT32 使用 `(a<b)?a:b`。相等值均选择第二输入；这对浮点 signed-zero 的位模式选择有影响，复用时应保留条件顺序。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；成员为 ND，rank 0–8 | FP32/FP16/BF16/INT32；列表内一致 | 支持空 Tensor；同位置 shape 与 x2 相等；`AutoContiguous()`。 |
| `x2` | TensorList；同位置 shape 与 x1 相同 | 与 x1 一致 | 支持空 Tensor；列表长度与 x1 相同；`AutoContiguous()`。 |
| `out` | TensorList、ND；公开文档说 shape size≥x1，公共推形复制 x1 | 与 x1 一致 | 输出不支持非连续；kernel 只写 x1 的有效元素数。 |
| `workspaceSize` / `executor` | 第一段输出 | `uint64_t*` / `aclOpExecutor**` | 返回 workspace 与封装计算流程的 executor。950 Tiling 将算子 workspace 设为 0。 |
| `workspace` / `stream` | 第二段输入 | Device pointer / `aclrtStream` | 必须与第一段结果配套；在 stream 异步提交。 |

### 入口与版本

固定仓只公开无版本后缀 `aclnnForeachMinimumList`，故为 950 当前实际支持的最新公开版本。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef 添加 `ascend950`，host CMake 把 ascend950 映射到 `arch35`，kernel CMake 单独选择 `arch35/foreach_minimum_list.cpp`，binary config 同时列出四种 dtype，平台证据无冲突。

自动生成 ACLNN inner bridge/header 未入固定仓，作为框架边界；公开 API 原型、错误语义、workspace/executor/stream 生命周期由文档给出。算子自有的 selector、POD、entry、SIMT body 与构建已全部追踪。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段检查三 TensorList 指针以及输出参数；空指针报 `ACLNN_ERR_PARAM_NULLPTR(161001)`。
2. 框架/OpDef 限定 FP32/FP16/BF16/INT32、ND 和三列表 dtype 一致；非法 dtype 报 `ACLNN_ERR_PARAM_INVALID(161002)`。x1/x2 进入 `AutoContiguous` 框架边界；输出必须连续。
3. 公共 infer-shape 要求输出实例数等于 x1 实例数并复制 x1 shape/dtype。固定仓的 arch35 Tiling 还要求输入0实例信息和第0个动态输入 desc 存在。
4. `ForeachMinimumListTilingFunc` 读取 AIV 核数、UB 大小、x1 实例数与首 Tensor dtype，拒绝 AIV 核数/UB 为0、任一指针为空、`tensorNum>256`、未知 dtype。
5. 对 `i<tensorNum` 读取 x1 storage shape 的 numel，写 `tensorSizes[i]` 并累加 `totalElements`。x2/out 的逐项 shape/dtype 约束由公开 API/OpDef/框架校验边界承担；arch35 Tiling 本身不重复扫描它们。

### 参数、Workspace 与生命周期

令 `N=sum tensorSizes[i]`、平台 AIV 核数 `C`：

```text
if N == 0:
    needCoreNum=1; perCoreElements=0; lastCoreElements=0; blockDim=1
else:
    perCoreElements = ceilAlign(min(65536, ceil(N/C)), 32)  # 元素
    needCoreNum = ceil(N/perCoreElements)
    lastCoreElements = N - perCoreElements*(needCoreNum-1)
    blockDim = needCoreNum
workspace[0] = 0
localMemorySize = UBplatform - 128KiB
```

`perCoreElements` 的 cap 在 32 元素对齐之前应用，所以最大写入值为 65536（其本身已对齐）。POD 和 TensorList 地址表由 executor/框架传入 K0，生命周期至少覆盖 stream 完成。`lastCoreElements` 由 Host 生产但 Device 不读取；最后核直接以 `totalElements` 截断 `coreEnd`。`SetLocalMemorySize` 的值属于 launch 本地内存上限；四个 SIMT body 没有显式动态 UB/shared buffer 分配，真实底层运行时保留方式固定仓不可见。

### 启动图与失败传播

```text
aclnnForeachMinimumListGetWorkspaceSize
  -> generated ACLNN bridge（固定仓不可见）
  -> ForeachMinimumList OpDef / infer / AutoContiguous
  -> ForeachMinimumListTilingFunc
     -> dtype schMode + ForeachMinimumListTilingData + blockDim + localMemorySize
  -> executor
aclnnForeachMinimumList(..., executor, stream)
  -> K0 foreach_minimum_list<schMode><<<needCoreNum, stream>>>
     -> per overlap: VF_CALL<dtype body>(Dim3(512), cnt, x1P, x2P, yP)
```

| 节点/边 | producer / consumer | 交接介质 | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| Host→K0 | arch35 Tiling / templated entry | POD、编译期 key、blockDim；三个 ListTensorDesc GM 对象 | executor 在调用 stream 上提交 | executor 持有参数；调用者持有 Tensor storage | Tiling 任一失败返回 `GRAPH_FAILED`，阻止有效 launch。 |
| K0 outer→VF | `Process` / SIMT body | 当前 Tensor overlap 的三个 GM pointer + `cnt` | `VF_CALL` 的节点内调用顺序；各 overlap 地址互不重叠 | 指针只在当前 Process 调用有效 | 没有独立第二 kernel，VF 不是 workspace 边。 |
| K0→out | SIMT threads / 调用者 | y_i GM 有效前缀 | thread 唯一索引写；stream 对后续节点提供顺序 | caller owner | 单节点无下游私有 cleanup。 |
| workspace | Tiling / framework | 0 字节 | N/A | N/A | 无算子 workspace 失败点。 |

## Route 与参数设计

### 分核与资源计算

Host 把列表成员视为长度 N 的串联域。AIV block `b` 覆盖 `[b*P,min((b+1)*P,N))`，其中 `P=perCoreElements`；Device 每次线性扫描 Tensor 边界 `[tStart,tEnd)`，求交集 `[oS,oE)`。有交集才从三个 ListTensorDesc 取第 t 个地址并加 `localOff=oS-tStart`，以 `cnt=oE-oS` 启动一次 512-thread VF。VF thread `q` 处理 `q, q+threadNum, ... <cnt`；所以 tail 由线程循环条件直接保护，无 padding、atomic、barrier或跨线程共享。

资源账：显式 UB/TQue/TBuf/workspace 均为 0。每线程至少有两项输入标量；BF16 另有两个 FP32 比较临时值；编译器最终寄存器字节数固定仓不可见。Host 读取 UB 只为设置 `localMemorySize=UB-128KiB`，kernel 源码没有显式消费该容量。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `schMode=0` / `GET_TPL_TILING_KEY(FLOAT)` | `dataType==DT_FLOAT`，在 dtype if/else 首项 | 否则继续 FP16。 |
| 2 | R-002 FP16 SIMT | `schMode=1` / key(FLOAT16) | `dataType==DT_FLOAT16` | 否则继续 BF16。 |
| 3 | R-003 BF16 SIMT | `schMode=2` / key(BF16) | `dataType==DT_BF16` | 否则继续 INT32。 |
| 4 | R-004 INT32 SIMT | `schMode=3` / key(INT32) | `dataType==DT_INT32` | 其他 dtype `GRAPH_FAILED`。 |

```text
DT_FLOAT   -> tpl schMode 0 -> Process<float,0>      -> ForeachMinimumListSimtFp32<float>
DT_FLOAT16 -> tpl schMode 1 -> Process<half,1>       -> ForeachMinimumListSimtFp16
DT_BF16    -> tpl schMode 2 -> Process<bfloat16_t,2> -> ForeachMinimumListSimtBf16
DT_INT32   -> tpl schMode 3 -> Process<int32_t,3>    -> ForeachMinimumListSimtInt32
otherwise  -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode0 | FP32、tensorNum≤256、公开 shape/dtype 约束、950 | K0 SIMT；每 overlap 一个 VF_CALL(512) | `Process<float,0>`→`SimtFp32<float>` | GM scalar load/store；无 shared/atomic/barrier | thread stride + `idx<cnt`；无 fallback。 |
| R-002 FP16 | schMode1 | FP16，其余同上 | 同一 K0 拓扑 | `Process<half,1>`→`SimtFp16` | 同上 | 同上。 |
| R-003 BF16 | schMode2 | BF16，其余同上 | 同一 K0 拓扑 | `Process<bfloat16_t,2>`→`SimtBf16` | 每线程 BF16 load + FP32 compare temporaries | 同上。 |
| R-004 INT32 | schMode3 | INT32，其余同上 | 同一 K0 拓扑 | `Process<int32_t,3>`→`SimtInt32` | GM scalar load/store，无浮点 helper | 同上；其他 dtype Host 失败。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| tensorNum | `tensorNum<=256` | POD `tensorSizes[256]` | 全部 | `GRAPH_FAILED` | `ForeachMinimumListTilingFunc` |
| 平台资源 | `GetCoreNumAiv()!=0 && UB!=0` | blockDim 与 local memory 上限需要平台值 | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| 每核域 | `P=ceilAlign(min(65536,ceil(N/C)),32)` | 控制每个 AIV block 的串联元素范围 | 全部 | 无备用 Route | `ForeachMinimumListTilingFunc` |
| DCache 保留 | `SetLocalMemorySize(uint32_t(UB-128KiB))` | Host 明确保留 128KiB DCache | 全部 | 若 UB≤128KiB，unsigned 减法边界未显式防护 | 同上 |
| thread 数 | `THREAD_NUM=512`、`LAUNCH_BOUND(512)`、`Dim3(512)` | VF 每个 overlap 固定 512 threads | 全部 | 无可变 threads fallback | `foreach_minimum_list_simt.h` |
| 空域 | `N==0 -> blockDim=1,P=0`；Device `coreStart>=coreEnd` | 避免任何 GM 地址获取和 VF_CALL | 全部 | 安静返回 | Tiling + `Process` |
| dtype | 严格 if/else 四种 dtype | 每种 dtype 对应独立编译实例/数学 helper | 全部 | 未知 dtype `GRAPH_FAILED` | Tiling key producer + entry router |

### POD/ABI

字段由普通 C++ struct 明确给出；字段顺序和数组容量可闭合。固定仓未给出目标编译器的 ABI dump，故精确 `sizeof/alignof/offsetof` 不推测。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMinimumListTilingData` | `needCoreNum` | int32/32 | 第1字段；精确 offset/总大小固定仓不可见 | AIV block 数 | zero:1；else `ceil(N/P)` | `Process` 判断最后 block | 必须与 blockDim 一致。 |
| 同上 | `totalElements` | int64/64 | 第2字段；精确 offset固定仓不可见 | 串联元素 | 累加所有 `tensorSizes` | 最后核 `coreEnd` 截断 | 累加溢出未显式检查。 |
| 同上 | `perCoreElements` | int64/64 | 第3字段 | 元素/block | `ceilAlign(min(65536,ceil(N/C)),32)` | `coreStart/coreEnd` | 空域为0。 |
| 同上 | `lastCoreElements` | int64/64 | 第4字段 | 元素/最后 block | `N-P*(needCoreNum-1)` | Device 未读取 | inactive consumer；保留 ABI。 |
| 同上 | `tensorNum` | int32/32 | 第5字段 | Tensor 数 | `static_cast<int32_t>(tensorNum)` | Tensor 扫描循环上界 | 0..256；实际读取首 desc 要求至少一个实例描述。 |
| 同上 | `tensorSizes` | int64[256] | 第6字段 | 元素/各 Tensor | x1 storage shape numel | 计算 `tStart/tEnd` 与 overlap | 未使用尾项先由 `memset_s` 清零。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode0 | `op_host/arch35/...tiling_arch35.cpp::ForeachMinimumListTilingFunc` | `ForeachMinimumListTilingData` | `arch35/foreach_minimum_list.cpp::foreach_minimum_list<0>` | `foreach_minimum_list_simt.h::Process<float,0>/SimtFp32` | `op_kernel/CMakeLists.txt` ascend950 arch35 source |
| R-002 FP16 | schMode1 | 同上 | 同一 POD | entry `<1>` | `Process<half,1>/SimtFp16` | 同上 |
| R-003 BF16 | schMode2 | 同上 | 同一 POD | entry `<2>` | `Process<bfloat16_t,2>/SimtBf16` | 同上 |
| R-004 INT32 | schMode3 | 同上 | 同一 POD | entry `<3>` | `Process<int32_t,3>/SimtInt32` | 同上 |

## Kernel 实现

### 公共执行流程

模板 entry 解析普通 struct POD，并在编译期 `if constexpr` 选择唯一 `Process<T,schMode>`。`Process` 先计算 block 串联域；最后 block 用 N 截断，空域立即返回。随后构造三份 `ListTensorDesc`，用前缀 `tStart` 扫描每个 Tensor 的 `[tStart,tEnd)`，只对与本 block 有交集的片段取得真实 GM pointer。每个片段同步调用一个 `VF_CALL(Dim3(512),cnt,...)`；线程索引在片段内 grid-stride 循环。没有跨片段 workspace，也没有 atomic/barrier/Queue。

### Route R-001：FP32 NaN 传播 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `docs/aclnnForeachMinimumList.md`、`op_host/foreach_minimum_list_def.cpp` | dtype table / OpDef | FP32 + 950 | 允许。 |
| Tiling | `op_host/arch35/foreach_minimum_list_tiling_arch35.cpp` | `ForeachMinimumListTilingFunc` | `dataType==DT_FLOAT` | `GET_TPL_TILING_KEY(FLOAT)`。 |
| Router | `op_kernel/arch35/foreach_minimum_list.cpp` | `foreach_minimum_list<0>` | `schMode==0` | `Process<float,0>`。 |

#### Device 数据流

##### 函数级调用链

```text
ForeachMinimumListTilingFunc -> foreach_minimum_list<0>
  -> Process<float,0>
     -> ListTensorDesc::GetDataPtr<float>(t)+localOff
     -> VF_CALL<ForeachMinimumListSimtFp32<float>>(Dim3(512),cnt,x1P,x2P,yP)
```

##### 源码等价伪码

```text
# 伪码
[coreStart,coreEnd)=block串联域
for t in [0,tensorNum):
  [oS,oE)=intersection([coreStart,coreEnd),[tStart,tStart+tensorSizes[t]))
  if oS<oE:
    ptrs = list[t] + (oS-tStart); cnt=oE-oS
    for idx=threadIdx; idx<cnt; idx+=threadNum:
      a=x1[idx]; b=x2[idx]; y[idx]=(isnan(float(a)) || a<b) ? a : b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 GM | `ListTensorDesc[t]+oS-tStart+idx` | cnt FP32 | FP32/ND | caller | SIMT load | caller to stream completion | scalar thread-stride access。 |
| y GM | 同偏移 | cnt FP32 | FP32/ND | SIMT store | caller | caller | 每 idx 唯一 writer。 |
| thread registers | thread-local `a,b` | 2×FP32；编译器最终字节不可见 | scalar | GM load | compare/store | 当前迭代 | 下一 idx 覆盖。 |
| UB/workspace | N/A | 显式 0 | N/A | N/A | N/A | N/A | Host localMemorySize 不被 body 显式使用。 |

##### 分层内存布局

N/A：该 SIMT 路线没有 AIV `InitBuffer`、TQue/TBuf 或显式 dynamic UB；真实布局是 GM TensorList + 每线程标量寄存器。

##### 字节账与复用顺序

```text
UB_total = 0（源码显式分配）
dynamic_ub = N/A（Host 仅 SetLocalMemorySize(UB-128KiB)，未给 VF_CALL 动态 buffer）
workspace_total = 0
thread-local logical values = a,b；最终 register bytes 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local a/b | GM load | isnan/compare/选择 | 同线程下一 `idx+=threadNum` | 线程程序顺序 | 不跨线程共享。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| overlap launch | `VF_CALL Dim3(512)` | outer Process | VF completion（调用语义） | 当前 GM 片段 | cnt 可小于512。 |
| per thread | grid-stride + `idx<cnt` | GM loads | store | 唯一 y[idx] | FP32；显式 NaN 传播；相等选 b。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_minimum_list_tiling_arch35.cpp` | `ForeachMinimumListTilingFunc` | schMode0、分核、POD、workspace0。 |
| Kernel entry/router | `op_kernel/arch35/foreach_minimum_list.cpp` | `foreach_minimum_list<0>` | FP32 body 映射。 |
| Device body/helper | `op_kernel/arch35/foreach_minimum_list_simt.h` | `Process<float,0>`、`ForeachMinimumListSimtFp32` | overlap + 512-thread NaN-aware compare。 |

### Route R-002：FP16 NaN 传播 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | API doc / `ForeachMinimumList` | dtype table | FP16+950 | 允许。 |
| Tiling | arch35 Tiling | dtype selector | `DT_FLOAT16` | schMode1。 |
| Router | arch35 entry | `foreach_minimum_list<1>` | `schMode==1` | `Process<half,1>`。 |

#### Device 数据流

##### 函数级调用链

```text
ForeachMinimumListTilingFunc -> foreach_minimum_list<1>
  -> Process<half,1> -> VF_CALL<ForeachMinimumListSimtFp16>(Dim3(512),cnt,...)
```

##### 源码等价伪码

```text
# 伪码；block/Tensor overlap 与 R-001 相同
for idx=threadIdx; idx<cnt; idx+=threadNum:
  half a=x1[idx], b=x2[idx]
  y[idx]=(__hisnan(a) || a<b) ? a : b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `list[t]+localOff+idx` | cnt FP16 | FP16/ND | caller/SIMT | SIMT/caller | stream | 唯一 idx 访问。 |
| registers | a,b | 2×FP16 logical | scalar | load | `__hisnan`/compare/store | iteration | next idx reuse。 |
| UB/workspace | N/A | 0 | N/A | N/A | N/A | N/A | 无显式分配。 |

##### 分层内存布局

N/A：GM + 每线程 FP16 标量寄存器，无 AIV UB buffer。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0
register bytes=固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| a/b thread locals | loads | `__hisnan`、`<`、store | next idx | thread order | no sharing。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| per thread | `__hisnan` + compare | FP16 loads | store | y[idx] | FP16 不升精度；idx<cnt；相等选 b。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | `ForeachMinimumListTilingFunc` | schMode1。 |
| Kernel entry/router | arch35 entry | `foreach_minimum_list<1>` | `Process<half,1>`。 |
| Device body/helper | arch35 SIMT header | `ForeachMinimumListSimtFp16` | `__hisnan(a)\|\|a<b`、512 threads。 |

### Route R-003：BF16 选择式 FP32 比较 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | API doc / OpDef | dtype table | BF16+950 | 允许。 |
| Tiling | arch35 Tiling | dtype selector | `DT_BF16` | schMode2。 |
| Router | arch35 entry | `foreach_minimum_list<2>` | `schMode==2` | `Process<bfloat16_t,2>`。 |

#### Device 数据流

##### 函数级调用链

```text
ForeachMinimumListTilingFunc -> foreach_minimum_list<2>
  -> Process<bfloat16_t,2> -> VF_CALL<ForeachMinimumListSimtBf16>(Dim3(512),cnt,...)
```

##### 源码等价伪码

```text
# 伪码；block/Tensor overlap 与 R-001 相同
for idx=threadIdx; idx<cnt; idx+=threadNum:
  bf16 a=x1[idx], b=x2[idx]
  float fa=__bfloat162float(a), fb=__bfloat162float(b)
  y[idx]=(isnan(fa) || fa<fb) ? a : b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `list[t]+localOff+idx` | cnt BF16 | BF16/ND | caller/SIMT | SIMT/caller | stream | output selects original bits。 |
| registers | a,b,fa,fb | 2 BF16 + 2 FP32 logical | scalar | load/conversion | compare/store | iteration | next idx reuse；实际 bytes不可见。 |
| UB/workspace | N/A | 0 | N/A | N/A | N/A | N/A | 无显式分配。 |

##### 分层内存布局

N/A：GM + 每线程 BF16/FP32 标量寄存器，无 shared/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0
logical thread temporaries=a,b,fa,fb; physical register bytes固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | GM load/conversion | isnan/compare/select | next idx | thread order | output must choose a/b, not convert fa/fb back。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compare | BF16→FP32 conversion | a/b | isnan and `<` | fa/fb | compare in FP32；idx<cnt。 |
| select/store | ternary select | compare | y store | original BF16 a/b | 不发生降精度 round；相等选 b。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | `ForeachMinimumListTilingFunc` | schMode2。 |
| Kernel entry/router | arch35 entry | `foreach_minimum_list<2>` | BF16 body。 |
| Device body/helper | arch35 SIMT header | `ForeachMinimumListSimtBf16` | FP32 check/compare，原 BF16 select。 |

### Route R-004：INT32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | API doc / OpDef | dtype table | INT32+950 | 允许。 |
| Tiling | arch35 Tiling | dtype selector | `DT_INT32` | schMode3。 |
| Router | arch35 entry | `foreach_minimum_list<3>` | `schMode==3` | `Process<int32_t,3>`。 |

#### Device 数据流

##### 函数级调用链

```text
ForeachMinimumListTilingFunc -> foreach_minimum_list<3>
  -> Process<int32_t,3> -> VF_CALL<ForeachMinimumListSimtInt32>(Dim3(512),cnt,...)
```

##### 源码等价伪码

```text
# 伪码；block/Tensor overlap 与 R-001 相同
for idx=threadIdx; idx<cnt; idx+=threadNum:
  int32 a=x1[idx], b=x2[idx]
  y[idx]=(a<b) ? a : b
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `list[t]+localOff+idx` | cnt INT32 | INT32/ND | caller/SIMT | SIMT/caller | stream | scalar access，唯一 writer。 |
| registers | a,b | 2×INT32 logical | scalar | load | compare/store | iteration | next idx reuse。 |
| UB/workspace | N/A | 0 | N/A | N/A | N/A | N/A | 无显式分配。 |

##### 分层内存布局

N/A：GM + 每线程 INT32 标量寄存器，无 AIV UB buffer。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0; register bytes固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| a/b thread locals | loads | compare/store | next idx | thread order | no sharing。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| per thread | integer `<` + select | loads | store | y[idx] | 精确 INT32；idx<cnt；相等选 b（数值等价）。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | `ForeachMinimumListTilingFunc` | schMode3。 |
| Kernel entry/router | arch35 entry | `foreach_minimum_list<3>` | INT32 body。 |
| Device body/helper | arch35 SIMT header | `ForeachMinimumListSimtInt32` | 512-thread integer minimum。 |

## 算子特定优化

### Trick T-001：串联域分块与逐 Tensor overlap，免除拼接 workspace

#### 触发与源码依据

- 适用 Route：R-001–R-004。
- 触发条件：Tiling 已产生 `perCoreElements`、`tensorSizes[]`，当前 block 与某 Tensor 区间相交。
- Golden 来源：[`foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_simt.h:115-132`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_simt.h#L115-L132)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t tStart = 0;
for (int32_t t = 0; t < tilingData->tensorNum; t++) {
    int64_t tSize = tilingData->tensorSizes[t];
    int64_t tEnd = tStart + tSize;
    int64_t oS = (coreStart > tStart) ? coreStart : tStart;
    int64_t oE = (coreEnd < tEnd) ? coreEnd : tEnd;
    if (oS < oE) {
        int64_t localOff = oS - tStart;
        int64_t cnt = oE - oS;
```

#### 机制与设计原因

Host 只下发每个 Tensor 的长度，不生成前缀数组或拼接 buffer；Device 线性维护 `tStart` 并求当前 block 与 Tensor 的交集。一个 block 可以跨多个小 Tensor，一个大 Tensor也可跨 block，而每个交集直接引用原始 Tensor GM。该机制把负载均衡域从“Tensor 个数”改为“总元素数”，同时 workspace 保持 0。

#### 迁移边界

- 适用：TensorList 各成员执行相同无跨元素状态的 pointwise 操作。
- 前提：`tensorSizes` 与地址表顺序严格一致，成员数≤256，所有 block 的区间不重叠且覆盖 `[0,N)`。
- 失效：需要跨 Tensor 归约/状态或成员 layout 不可直接寻址时，必须引入额外协议，不能只复用 overlap 循环。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；可由源码证明不能把 `localOff=oS-tStart` 错写成 `oS-coreStart`，后者在 block 跨 Tensor 时会指向错误成员偏移。

### Trick T-002：固定 512-thread grid-stride VF 统一处理短片段与长片段

#### 触发与源码依据

- 适用 Route：R-001–R-004。
- 触发条件：某 Tensor overlap `cnt>0`。
- Golden 来源：[`foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_simt.h:33-47`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_simt.h#L33-L47)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr uint32_t THREAD_NUM = 512;
for (int64_t idx = static_cast<int64_t>(AscendC::Simt::GetThreadIdx());
     idx < count;
     idx += static_cast<int64_t>(AscendC::Simt::GetThreadNum())) {
    T a = x1[idx];
    T b = x2[idx];
    y[idx] = (isnan(static_cast<float>(a)) || a < b) ? a : b;
}
```

#### 机制与设计原因

`LAUNCH_BOUND(512)`、`VF_CALL Dim3(512)` 与循环步长 `GetThreadNum()` 保持一致。短片段由 `idx<count` 屏蔽多余线程，长片段由同一线程隔 512 元素继续处理，不需要单独 tail kernel、padding 或 atomic。FP32 条件还把 NaN 传播合入一次选择。

#### 迁移边界

- 适用：每个输出只依赖同索引输入、线程间无共享状态的 pointwise body。
- 前提：每个 overlap 由一个 VF 调用独占，thread 数与 launch-bound/Dim3/循环步长一致。
- 失效：需要邻域、归约、共享缓存或写冲突时，必须增加 barrier/shared/atomic 设计，不能沿用无同步假设。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；删除 `idx<count` 会越过当前 Tensor 片段，尤其当 `cnt<512` 时直接越界。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | R-004 INT32 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | DT_FLOAT/0 | DT_FLOAT16/1 | DT_BF16/2 | DT_INT32/3 | Host if/else 顺序。 |
| 模板与 dtype | `float,0` | `half,1` | `bfloat16_t,2` | `int32_t,3` | 每 dtype 独立编译 body。 |
| 循环与分核 | colspan：相同串联域 block + Tensor overlap + 512-thread grid-stride | 同左 | 同左 | 同左 | shape 与资源策略不依赖 dtype。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | a,b registers | a,b registers | a,b,fa,fb registers | a,b registers | 均无 Queue/shared/dynamic UB。 |
| 内存总占用与复用 | workspace/显式UB 0 | 同 | 同 | 同 | 直接 GM 标量访问。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar GM | 同 | BF16 convert helper | integer SIMT | 不含 NDDMA/通信。 |
| 同步、精度与 tail | isnan FP32；equal→b | `__hisnan`；equal→b | FP32 check，选择原 BF16 | 精确 int；equal→b | `idx<cnt` 统一 tail。 |
| fallback/失败边界 | 无 | 无 | 无 | 无；其后 dtype失败 | 不存在跨 dtype fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持；公开 950PR/950DT。 |
| Host/Launcher | arch22 公共 Foreach Tiling | 与 A2 共用 | arch35 专用 Tiling；tensorNum≤256、P≤65536 对齐32元素、workspace0。 |
| Kernel/资源 | AIV_VECTOR Queue/UB | 同 A2 | SIMT VF、512 threads、直接 GM；无显式 Queue。 |

## 限制与失败边界

- `tensorNum>256` 明确失败。`tensorNum==0` 时 Host 在读取 `GetDynamicInputDesc(0,0)` 已可能失败；公开“支持空 Tensor”不等价于承诺空 TensorList。非空列表中总元素 N=0 有显式 blockDim1/no-op 路径。
- `totalElements` 使用 int64 累加但没有溢出检查；极端 shape 总量不得使加法溢出。`tensorNum` 从 uint64 窄化为 int32 前已有≤256 guard。
- `localMemorySize=uint32_t(ubSize-128KiB)` 未检查 UB 是否大于 DCache reserve，也未检查向 uint32 窄化；实际 950 平台值由运行时保证，跨平台复用必须补足同等资源前置条件。
- `lastCoreElements` 是写入但未消费的 ABI 字段；Device 以 `totalElements` 截断最后 block。删字段会破坏 ABI，不能因未读而移除。
- FP32/FP16/BF16 都显式传播 NaN且相等时选择 b；BF16 只升精度比较、不重新量化。替换为通用 `min` intrinsic 前必须证明 NaN和 signed-zero 位级等价。
- `ListTensorDesc` 元数据格式、`VF_CALL` 内部完成/调度语义、物理寄存器占用、自动生成 ACLNN bridge 源码固定仓不可见；本卡不推测。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_minimum_list/docs/aclnnForeachMinimumList.md` | API 原型、产品/参数/错误表 | 公开 ACLNN、950 支持、语义与错误。 |
| 入口声明/约束 | `foreach/foreach_minimum_list/README.md` | 产品支持、参数说明 | Ascend950 与 dtype/layout 证据。 |
| 入口实现 | `foreach/foreach_minimum_list/op_host/foreach_minimum_list_def.cpp` | `ForeachMinimumList` | ascend950 OpDef、四 dtype、AutoContiguous。 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`、`InferDataType4ForeachCommon` | 输出数量检查与 shape/dtype 复制。 |
| Host/Tiling/Launcher | `foreach/foreach_minimum_list/op_host/arch35/foreach_minimum_list_tiling_arch35.cpp` | `ForeachMinimumListTilingFunc` | 校验、POD、分核、key、workspace/local memory。 |
| Host/Tiling/Launcher | `foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_tiling_data.h` | `ForeachMinimumListTilingData` | POD ABI 字段。 |
| Host/Tiling/Launcher | `foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 四 schMode 模板选择。 |
| Kernel entry/body | `foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list.cpp` | `foreach_minimum_list<schMode>` | 模板 router。 |
| Kernel entry/body | `foreach/foreach_minimum_list/op_kernel/arch35/foreach_minimum_list_simt.h` | `Process`、四个 SIMT body | overlap、thread mapping、精度和 tail。 |
| 构建接线 | `foreach/foreach_minimum_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950 到 arch35 Host Tiling 的映射。 |
| 构建接线 | `foreach/foreach_minimum_list/op_kernel/CMakeLists.txt` | ascend950 `add_kernel_sources` | 950 活动 arch35 源与 auto_sync false。 |
| 构建接线 | `foreach/foreach_minimum_list/op_host/config/ascend950/foreach_minimum_list_binary.json` | `op_list` | 950 四 dtype binary 接线。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | N/P/core 数与 256 长度表足以在 Device 求 overlap | 字段顺序/位宽、256容量、`lastCoreElements` 保留 | `ForeachMinimumListTilingData` |
| 调用顺序 | block 域→Tensor 扫描→有交集才 VF_CALL | `tStart=tEnd` 更新必须在每次迭代末尾；地址 offset 用 `oS-tStart` | `Process` |
| workspace/内存 | workspace0，直接 GM；local memory reserve 128KiB | 不得虚构跨节点 workspace；跨平台须验证 UB reserve/narrowing | Tiling `WS_SYS_SIZE/DCACHE_SIZE` |
| 同步 | 每输出元素单线程独占，无 barrier/atomic | 只能用于无写冲突 pointwise；VF_CALL completion 细节是框架边界 | 四个 SIMT body |
| 精度与尾块 | 浮点 NaN传播、equal选b；BF16 FP32比较原值选择；`idx<cnt` | 条件顺序、BF16不重编码、512 thread stride | `SimtFp32/Fp16/Bf16/Int32` |
| 失败边界 | 平台资源、指针、tensorNum、dtype错误直接 Host失败；空域 no-op | tensorNum≤256，未知 dtype无 fallback，空 list不在可证支持范围 | `GetPlatformInfo`、Tiling selector、`Process` |

- [__hisnan](../../../../apis/simt_api/math_functions/half_type/half_compare_functions/_hisnan.md)
