---
type: Operator
title: "OneHot 950 ACLNN 算子知识"
description: "说明 aclnnOneHot 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/docs/aclnnOneHot.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_api/aclnn_one_hot.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_api/aclnn_one_hot.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_api/one_hot.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/op_api/aclnn_check.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/one_hot_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/one_hot_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/arch35/one_hot_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/arch35/one_hot_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/arch35/one_hot_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_kernel/one_hot_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_kernel/arch35/one_hot.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_kernel/arch35/one_hot_mix.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/config/ascend950/one_hot_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T11:36:45Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# OneHot 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnOneHotGetWorkspaceSize` + `aclnnOneHot`；固定仓只公开这一版本族。 |
| 计算语义 | 在 `self` 的 `axis` 位置插入长度 `numClasses` 的类别轴；类别等于索引时写 `onValue[0]`，否则写 `offValue[0]`。越界或负索引整条类别轴均为 `offValue[0]`。 |
| 输入输出 | `self`: 1–7D ND，950 支持 `uint8/int32/int64`；`onValue/offValue/out`: ND 且同 dtype，支持 `float16/float32/int8/uint8/int32/int64`；输出比输入多一维。 |
| 执行类型 | 单 Kernel。R-001 为 SIMT；R-002 为同一 `KERNEL_TYPE_MIX_AIV_1_0` 节点内的 `AIV_VECTOR(Duplicate + DataCopyPad) -> SyncAll -> SIMT`。任务类型明确为 1 个 AIV、0 个 AIC，无 Cube/AIC。 |
| Kernel 路线 | R-001 `key=1000` 直接 SIMT 全量写；R-002 `key=1001` 先由 AIV/UB 批量铺 `offValue`，再由 SIMT 稀疏覆写 `onValue`。 |
| 核心资源 | GM 输入/输出、R-002 单个 `TBuf offValueUb`、SIMT 线程寄存器；算子 Tiling 声明 16 MiB system workspace，Kernel 不使用 user workspace 数据。 |
| 来源闭包 | `source_closed`：公开 API、L0 launcher、InferShape/OpDef、Tiling/POD、两个 key 的 entry/body、950 binary 与构建接线均闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `r = rank(self)`，规范化插入轴 `a = (axis == -1 ? r : axis)`，`P = product(self.shape[0:a])`，`S = product(self.shape[a:r])`，`D = numClasses`。输入线性位置 `q = i*S + j` 对应输出：

```text
out[i, c, j] = onValue[0],  当 0 <= self[q] < D 且 c == self[q]
out[i, c, j] = offValue[0], 其它情况
```

输出 shape 是在 `self.shape` 的位置 `a` 插入 `D`。API 在 `self` 为空或 `D==0` 时不添加 OneHot Kernel；输出由调用方给出的空 shape 表示。Kernel 路线仅覆盖非空且 `D>0` 的执行域。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 1–7D，ND；可非连续，API 先 `Contiguous` 再 `ReFormat(ND)` | `uint8/int32/int64` | 索引；`self[q] < 0` 或 `self[q] >= D` 时该位置没有 `onValue`。 |
| `numClasses` | Host 标量 | API 原型为 `int` | 非空 `self` 时 `>=0`，空 `self` 时 `>0`；`0` 触发无 Kernel 空输出。 |
| `onValue` | 1–8D，ND；只读取第一个元素 | 与 `out` 相同 | 950 支持 `float16/float32/int8/uint8/int32/int64`。 |
| `offValue` | 1–8D，ND；只读取第一个元素 | 与 `out` 相同 | 950 支持类型同 `onValue`。 |
| `axis` | Host 标量 | `int64_t` | `[-1, rank(self)]`；默认语义为 `-1`，即末尾插入。 |
| `out` | 1–8D，ND；可非连续，最后经 `ViewCopy` 提交 | 与 `onValue/offValue` 相同 | shape 必须精确等于插入类别轴后的 shape。 |

### 入口与版本

`aclnnOneHotGetWorkspaceSize` 生成执行器与 workspace 大小，`aclnnOneHot` 通过 `CommonOpExecutorRun` 在给定 stream 上执行。固定 Commit `165d41f02a43a62324b434cb6bda7c533b14da3e` 的公开文档明确支持 Ascend 950PR/950DT；`OneHot` OpDef 只显式 `AddConfig("ascend950", ...)`，并把 `opFile.value` 接到 `one_hot_apt`。950 API 校验通过 `IsRegBase()`（`DAV_3510`）选择扩展 dtype 集合。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段 API 的顺序不可交换：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize/executor`，`CheckNotNull` 校验四个 tensor 指针。
2. `CheckDtypeValid` 在 950 接受 `self={uint8,int32,int64}`、值/输出六种 dtype，并强制 `onValue/offValue/out` dtype 相同。
3. `CheckMaxDimension` 限制 `self<=7D`，其余 tensor `<=8D`。
4. `CheckMaxNumClasses`、`CheckAxisRange`、`CheckShapeValid` 依次校验 `D`、轴和精确输出 shape。
5. 创建 `UniqueExecutor`。`self` 为空或 `D==0` 时设置 `workspaceSize=0`、释放 executor 给调用方并直接成功，执行器中没有 OneHot 节点。
6. 非空路径执行 `Contiguous(self)`，再 `ReFormat(..., FORMAT_ND)`；把 `numClasses` 分配成 scalar tensor，`self=uint8` 时 depth tensor 强制为 `int32`，否则与 `self` dtype 相同。
7. `l0op::OneHot` 分配内部连续输出、执行 InferShape、检查 AICore dtype 后添加 `OneHot` launcher；随后 `ViewCopy(oneHotOut, out)` 处理可能非连续的用户输出。

`Contiguous/ReFormat/ViewCopy` 的 Kernel 细节属于框架边界；OneHot 自有 selector、POD 和 Device body 均已继续追踪。

### 参数、Workspace 与生命周期

`OneHotTilingBase::GetPlatformInfo/TilingPrepare4OneHotForAscendc` 从 950 平台取得 AIV 核数 `C` 与 UB 字节数 `U`。Tiling 固定写 `workspaceSize_=16*1024*1024`，并把它写到算子 workspace slot 0；entry 先 `SetSysWorkspace(workspace)` 再 `GetUserWorkspace(workspace)`，但两个 body 都不读写 `userWs`。L2 返回的总 workspace 还包含框架可能为 `Contiguous/ReFormat/ViewCopy` 产生的空间；其合并与切片公式在固定仓的框架实现中不可见，不能把 API 返回值简化成恒定 16 MiB。

executor 从第一段接口转移给调用者，在第二段调用期间持有 launcher、临时 tensor 与 workspace 规划；stream 传给 `CommonOpExecutorRun`。内部 OneHot 输出在 `ViewCopy` 消费前有效，用户 `out` 由调用者持有。

### 启动图与失败传播

```text
aclnnOneHotGetWorkspaceSize
  -> [框架边界] Contiguous(self) -> ReFormat(ND)
  -> OneHot launcher: one_hot(x, depth, on_value, off_value, internalOut, workspace, tiling)
  -> [框架边界] ViewCopy(internalOut, userOut)
aclnnOneHot(workspace, workspaceSize, executor, stream)
  -> CommonOpExecutorRun：按执行器顺序在同一 stream 提交上述节点
```

| 节点/边 | entry/介质 | grid/POD/extent | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|
| OneHot Kernel | `one_hot` | `blockDim=max(usedCoreNum, offValueUsedCoreNum)`；`OneHotTilingData`；输入 `N=P*S`，输出 `N*D` | executor 持有 tiling/workspace；内部输出由 executor 持有至 ViewCopy | entry 的 `workspace==nullptr` 或 `GetUserWorkspace==nullptr` 直接 return，固定源码未向 Host 回传 Device 内部早退状态。 |
| OneHot→ViewCopy | 内部 GM tensor | shape 为插入 `D` 后的 ND 输出，dtype 与值 tensor 相同 | 同一 executor/stream 的先后任务保证 producer 完成后消费；框架精确 event 机制不可见 | L0 添加 launcher 或 ViewCopy 返回空指针时第一段 API 失败，不生成可执行 executor。 |

R-001/R-002 都是一个 OneHot Device 节点，不存在算子自有的跨 Kernel workspace edge。

## Route 与参数设计

### 分核与资源计算

记 `N=P*S`、AIV 核数 `C`、`MIN_FACTOR=1024`：

```text
若 N <= 1024:
  usedCoreNum=1, perCoreCalSize=N, hasTail=0
若 1024 < N <= C*1024:
  usedCoreNum=ceil(N/1024), perCoreCalSize=1024, hasTail=(N%1024!=0)
若 N > C*1024:
  usedCoreNum=C, perCoreCalSize=floor(N/C), hasTail=(N%C!=0)
```

最后一个输入核在 `hasTail` 时取 `N%1024`（部分核模式）或 `perCoreCalSize+N%C`（全核模式）。R-002 还对 `T=N*D` 个输出元素分核：若 `ceil(T/64)<C`，每核 64 元素、核数 `ceil(T/64)`；否则用 `C` 核、每核 `floor(T/C)`，尾核加余数。Tiling 令 `ubSize=U-58*1024`，R-002 以该字节数初始化唯一 TBuf，并设置相同 local memory size 与 batch schedule mode 1。

`N` 与 `T` 的 Host 中间量为 `uint64_t`，但 `depth_` 成员是 `int32_t`；公开 API 的 `numClasses` 本身是 `int`。`usedCoreNum/offValueUsedCoreNum` 写入 `int16_t`，平台核数必须能安全窄化到该字段；固定源码没有额外溢出 guard。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-002 UB 批量初始化 + SIMT 稀疏覆写 | key `1001`；`OneHotMix<DTYPE_X,DTYPE_DEPTH,DTYPE_Y>` | `depth_ > 3 && uint64(N)*uint64(depth_) > 1024` | 条件为假进入 R-001。 |
| 2 | R-001 直接 SIMT 全量写 | key `1000`；`OneHot<DTYPE_X,DTYPE_DEPTH,DTYPE_Y>` | 上述条件为假，即 `depth_<=3 \|\| N*depth_<=1024` | 无下一 Kernel Route；API 空域在 launcher 前返回。 |

```text
若 self.empty || numClasses==0 -> API 成功短路，无 Kernel
否则 MergeDims 得 N=P*S
  if depth>3 && N*depth>1024:
    key=1001 -> OneHotMix
  else:
    key=1000 -> OneHot
unknown key -> entry 不执行 body，直接结束
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 直接 SIMT | 1000 | 950 的 36 个 binary dtype 组合；非空、`D>0`；`D<=3 \|\| N*D<=1024` | 单 `one_hot` 节点，SIMT | `OneHot<T1,T2,T3>::Process`；`D==1` 调 `SimtComputeDepth`，否则按每核 offset 选 `SimtCompute` 或 `SimtComputeFor64` | 无 AIV UB；线程直接 GM load/store；无显式 block barrier | 尾核修正 `coreFactor`；`startOffset>INT32_MAX` 用 64 位除法 fallback。 |
| R-002 UB+SIMT | 1001 | 950 的 36 个 binary dtype 组合；非空；`D>3 && N*D>1024` | 单 `one_hot` 节点内 AIV Vector/DMA→全核同步→SIMT | `OneHotMix<T1,T2,T3>::InitOffValueAndCopyOut` + `SimtComputeMix/For64Mix` | `TBuf offValueUb=U-58KiB`；`Duplicate` 后 V→MTE3 event；`SyncAll<true>` 保护 dense-fill→sparse-patch | 两套尾核：输出初始化尾核和输入索引尾核；`endOffset>INT32_MAX` 用 64 位 fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| UB 初始化阈值 | `D>3 && N*D>1024` | 只有足够大的输出才物化 `offValueUb` 并使用批量 Vector/DMA 初始化 | R-002 | R-001 | `OneHotTilingBase::DoOpTiling` |
| SIMT dcache 预留 | `localMemorySize=U-58KiB` | R-002 同时使用 SIMT 与 AIV UB，Host 从平台 UB 扣除固定 dcache | R-002 | 固定源码无 fallback；`U<=58KiB` 未显式拒绝 | `DoOpTiling`, `PostTiling` |
| 32/64 位地址分支 | R-001 `startOffset>INT32_MAX`；R-002 `endOffset>INT32_MAX` | 32 位 fast-division body 的 `gmIdx/coreFactor` 为 `int32_t` | 两 Route | 调对应 `For64` body | `OneHot::Process`, `OneHotMix::Process` |
| `D==1` 特化 | `depth_==1` | 输出与输入一一对应，不需要类别轴循环或二维 thread block | R-001 | R-001 普通 SIMT 子路径 | `OneHot::Process` |
| 空/零类别 | `self->IsEmpty() \|\| numClasses==0` | 避免 Tiling 中 `factor1`/输出域为零并跳过无效 launch | 两 Route 前 | API 成功短路 | `aclnnOneHotGetWorkspaceSize` |
| workspace 指针 | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 初始化 system workspace 后才进入 body | 两 Route | Device entry 直接 return | `one_hot` |

### POD/ABI

`OneHotTilingData` 由 `BEGIN_TILING_DATA_DEF` 宏生成；字段源码类型与声明顺序可见，但生成后的 `sizeof/alignof/offsetof` 未落在固定仓，ABI 布局列据此明确标为不可见，不能猜测 padding。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `OneHotTilingData` | `hasTail` | `int16_t/16` | 固定仓不可见 | bool flag | `GetTilingParam`: 输入域有余数 | 两 body 的最后输入核修正 | 仅 0/1。 |
| 同上 | `usedCoreNum` | `int16_t/16` | 固定仓不可见 | core | `GetTilingParam`: 1、`ceil(N/1024)` 或 `C` | 两 body 过滤 block 与判断最后核 | 从 `uint32_t` 窄化。 |
| 同上 | `offValueHasTail` | `int16_t/16` | 固定仓不可见 | bool flag | `CalUbTilingData`: `T%64!=0` 或 `T%C!=0` | `OneHotMix::Init` 修正最后初始化核 | R-001 inactive。 |
| 同上 | `offValueUsedCoreNum` | `int16_t/16` | 固定仓不可见 | core | R-002 为 `ceil(T/64)` 或 `C`；R-001 置 0 | `Process` 决定 dense-fill block，`Init` 判断尾核 | R-001 为 0。 |
| 同上 | `perCoreCalSize` | `int64_t/64` | 固定仓不可见 | input elements/core | `GetTilingParam` | 两 body 的 `blockFactor_` | 最后核可能额外处理 remainder。 |
| 同上 | `offValuePerCoreCalSize` | `int64_t/64` | 固定仓不可见 | output elements/core | `CalUbTilingData` | `OneHotMix::Init/InitOffValueAndCopyOut` | R-001 未读取。 |
| 同上 | `offValueCalTailSize` | `int64_t/64` | 固定仓不可见 | 未定义 | 固定源码未赋值 | 固定 Kernel 未读取 | inactive/保留字段。 |
| 同上 | `tailCalSize` | `int64_t/64` | 固定仓不可见 | 未定义 | 固定源码未赋值 | 固定 Kernel 未读取 | inactive/保留字段。 |
| 同上 | `prefixDim` | `int64_t/64` | 固定仓不可见 | elements-product | `MergeDims`: `P` | 两类 `Init` 与输出地址公式 | 非空 API 路径中正数。 |
| 同上 | `suffixDim` | `int64_t/64` | 固定仓不可见 | elements-product | `MergeDims`: `S` | fast division 与输出 stride | 非空路径中正数。 |
| 同上 | `coreNum` | `int64_t/64` | 固定仓不可见 | AIV cores | `GetTilingParam`: `C` | 两 body 的尾核模式判断 | 来自平台。 |
| 同上 | `ubSize` | `int64_t/64` | 固定仓不可见 | bytes | `DoOpTiling`: `U-58KiB` | `OneHotMix::Init` 的 TBuf 大小/容量 | R-001 不分配该 UB。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 直接 SIMT | 1000 | `one_hot_tiling_arch35.cpp::DoOpTiling/GetTilingParam/GetTilingKey/PostTiling` | `OneHotTilingData` | `one_hot_apt.cpp::one_hot` → `OneHot<T1,T2,T3>` | `arch35/one_hot.h::Init/Process/SimtCompute*` | `one_hot_def.cpp`、`one_hot_binary.json`、OneHot `CMakeLists.txt`、全局 `ascendc_config.json` |
| R-002 UB+SIMT | 1001 | 同上，另含 `CalUbTilingData` | 同一 POD，启用 `offValue*` 与 `ubSize` | `one_hot_apt.cpp::one_hot` → `OneHotMix<T1,T2,T3>` | `arch35/one_hot_mix.h::Init/InitOffValueAndCopyOut/CopyOut/Process/SimtCompute*Mix` | 同上 |

## Kernel 实现

### 公共执行流程

两个 Route 共用一个 `extern "C" __global__ __aicore__ one_hot` entry，参数顺序固定为 `x, depth, on_value, off_value, y, workspace, tiling`。entry 声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)`，表示单 AIV、零 AIC；设置 system workspace、解析 `OneHotTilingData` 后按 1000/1001 实例化。未知 key 没有 `else`，不写输出。

两类 `Init` 都从 GM 读取 `depth[0]/on_value[0]/off_value[0]`，绑定 `N` 个索引与 `N*D` 个输出，计算 `factor1=S`、`factorOut1=S*D`、`factorOut2=S`。32 位路径预计算除数 `S` 的 `shift_` 与 32 位 magic `m_`；64 位 fallback 直接使用 `/` 和 `%`。

### Route R-001：直接 SIMT 全量写

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/one_hot/op_api/aclnn_one_hot.cpp` | `aclnnOneHotGetWorkspaceSize` | 非空且 `numClasses!=0` | 建立 OneHot launcher。 |
| Tiling | `math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp` | `DoOpTiling/GetTilingKey` | `!(depth_>3 && N*depth_>1024)` | key 1000。 |
| Router | `math/one_hot/op_kernel/one_hot_apt.cpp` | `one_hot` | `TILING_KEY_IS(1000)` | `OneHot<DTYPE_X,DTYPE_DEPTH,DTYPE_Y>`。 |
| Device | `math/one_hot/op_kernel/arch35/one_hot.h` | `OneHot::Process` | `D==1` / `D!=1`，及 `startOffset>INT32_MAX` | 三个 SIMT 子 body。 |

#### Device 数据流

##### 函数级调用链

```text
OneHot launcher -> one_hot(x, depth, on, off, internalOut, workspace, tiling)
  -> OneHot<T1,T2,T3>::Init
  -> OneHot<T1,T2,T3>::Process
     -> D==1: asc_vf_call<SimtComputeDepth>
     -> D!=1 && startOffset<=INT32_MAX: asc_vf_call<SimtCompute>
     -> D!=1 && startOffset>INT32_MAX: asc_vf_call<SimtComputeFor64>
internalOut -> executor-ordered ViewCopy -> user out
```

##### 源码等价伪码

```text
# 伪码；block = GetBlockIdx()
if block>=usedCoreNum || perCoreCalSize==0: return
coreFactor = perCoreCalSize
if hasTail && block>0 && block==usedCoreNum-1:
  coreFactor = (N<=C*1024) ? N%1024 : perCoreCalSize+N%C
start = block*perCoreCalSize

if D==1:
  launch threads=(1024)
  for q=start+threadIdx.x; q<start+coreFactor; q+=1024:
    y[q] = (0<=x[q]<1) ? on : off
else if start>INT32_MAX:
  launch threads=(dimX,dimY)
  for local=threadIdx.x; local<coreFactor; local+=dimX:
    q=start+local; i=q/S; j=q%S
    for c in [0,D): y[i*S*D+c*S+j]=off
    if 0<=x[q]<D: y[i*S*D+x[q]*S+j]=on
else:
  dimX=max(1,1024/coreFactor); dimY=max(1,1024/dimX)
  将 dimX/dimY 中较大者向下规整到 32 的倍数（不足 32 时为 1）
  for local=threadIdx.y; local<coreFactor; local+=dimY:
    q=start+local; i=fast_div(q,S,m,shift); j=q-i*S
    for c=threadIdx.x; c<D; c+=dimX: y[i*S*D+c*S+j]=off
    if 0<=x[q]<D: y[i*S*D+x[q]*S+j]=on
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `q=block*factor+local` | `N` | `T1/ND` | API contiguous/reformat | SIMT load | executor 输入，Kernel 全程 | 标量 GM load，无 DMA 对齐前提。 |
| GM `depth/on/off` | 0 | 各读取 1 元素 | `T2/T3` | API/scalar inputs | `Init` | Kernel Init 读取后转为标量成员 | 输入张量其余元素 inactive。 |
| GM `y` | `i*(S*D)+c*S+j` | `N*D` | `T3/ND` | SIMT | ViewCopy | 内部输出至后继任务 | 每个类别元素最终写一次；合法索引位置先 off 后 on。 |
| SIMT 寄存器 | `q,i,j,dstOffset,m,shift` | 每线程标量，固定仓无寄存器字节报告 | 整数/T3 | SIMT body | 同一线程 | thread lifetime | 无 shared/dynamic UB。 |
| system workspace | framework-defined base | Tiling slot 16 MiB | bytes | executor | `SetSysWorkspace` | launch lifetime | user slice 未被 body 使用。 |

##### 分层内存布局

N/A：R-001 不调用 `InitBuffer`，没有 AIV `TQue/TBuf` 或 SIMT dynamic UB；数据直接在 GM 与线程寄存器之间流动。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = N/A（asc_vf_call 未传动态 UB）
workspace_kernel_slot = 16*1024*1024 bytes（system workspace）
workspace_total = executor 合并 Contiguous/ReFormat/OneHot/ViewCopy 后的值，固定框架实现不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM `y[q,c]` 合法索引 lane | 对应线程先写 `off` | 同一输入位置的 on-patch | 写成 `on` 后不再覆盖 | 同一线程/同一 SIMT 调用的程序顺序；二维路径中类别 owner 最后也执行 on-patch | `0<=x[q]<D`；不同 q 的地址互斥。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| block 过滤 | 条件 return | Host `blockDim` | `Process` | 空闲 block | `block>=usedCoreNum` 不写。 |
| SIMT 写 | 无显式 barrier/atomic | GM scalar load | GM scalar store | 每个 `(q,c)` 地址 | 不做算术/cast；保持 `T3` 位型语义。 |
| 输入尾核 | `hasTail` 分支 | Tiling | 最后 used block | `[0,N)` | 只循环真实 `coreFactor`，不写 padding。 |
| 32 位 fallback | `unlikely(startOffset>INT32_MAX)` | per-core offset | `SimtComputeFor64` | 大线性索引 | 使用 int64 `/,%`，避免 `int32_t gmIdx` 窄化。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp` | `GetTilingParam/DoOpTiling/GetTilingKey/PostTiling` | 分核、key 1000 与 blockDim。 |
| Kernel entry/router | `math/one_hot/op_kernel/one_hot_apt.cpp` | `one_hot`, `OneHot<DTYPE_X,DTYPE_DEPTH,DTYPE_Y>` | key→body；1 AIV/0 AIC task。 |
| Device body/helper | `math/one_hot/op_kernel/arch35/one_hot.h` | `Init/Process/SimtComputeDepth/SimtCompute/SimtComputeFor64` | GM 地址、线程映射、fast division、tail 与 64 位 fallback。 |

### Route R-002：UB 批量初始化 + SIMT 稀疏覆写

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/one_hot/op_api/aclnn_one_hot.cpp` | `aclnnOneHotGetWorkspaceSize` | 非空且 `D>0` | 建立 launcher。 |
| Tiling | `math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp` | `DoOpTiling/CalUbTilingData/GetTilingKey/PostTiling` | `D>3 && N*D>1024` | key 1001，local memory `U-58KiB`，schedule mode 1。 |
| Router | `math/one_hot/op_kernel/one_hot_apt.cpp` | `one_hot` | `TILING_KEY_IS(1001)` | `OneHotMix<DTYPE_X,DTYPE_DEPTH,DTYPE_Y>`。 |
| Device | `math/one_hot/op_kernel/arch35/one_hot_mix.h` | `OneHotMix::Process` | dense-fill blocks → `SyncAll<true>` → used input blocks | AIV/DMA + SIMT。 |

#### Device 数据流

##### 函数级调用链

```text
OneHot launcher -> one_hot(...)
  -> OneHotMix<T1,T2,T3>::Init
     -> TPipe::InitBuffer(offValueUb, U-58KiB)
  -> OneHotMix<T1,T2,T3>::Process
     -> active fill block: InitOffValueAndCopyOut
        -> Duplicate(offValueTensor, offValue, initSize)
        -> SetFlag/WaitFlag<V_MTE3>
        -> CopyOut -> DataCopyPad(y[offset], offValueTensor, copyLength*sizeof(T3))
     -> SyncAll<true>()
     -> active input block: asc_vf_call<SimtComputeMix 或 SimtComputeFor64Mix>
internalOut -> executor-ordered ViewCopy -> user out
```

##### 源码等价伪码

```text
# 伪码；block=GetBlockIdx(), T=N*D, cap=(U-58KiB)/sizeof(T3)
actual = offValuePerCoreCalSize
if block==offValueUsedCoreNum-1 && offValueHasTail:
  actual = (offValueUsedCoreNum>=C) ? offValuePerCoreCalSize+T%C : T%64
loops = (cap>=actual) ? 1 : ceil(actual/cap)
single = min(cap,actual)
last = (cap>=actual) ? actual : actual%cap
InitBuffer(offValueUb,U-58KiB)

if block<offValueUsedCoreNum:
  local = offValueUb.Get<T3>()
  Duplicate(local,off,max(single,last))
  V_MTE3 event
  for k in [0,loops-1): DataCopyPad(y[block*perOff+k*single],local,single*sizeof(T3))
  DataCopyPad(y[block*perOff+(loops-1)*single],local,last*sizeof(T3))
SyncAll<true>()
if block>=usedCoreNum: return

修正输入尾核 coreFactor；start=block*perCoreCalSize；end=start+coreFactor
launch 1024 threads:
  for local=threadIdx.x; local<coreFactor; local+=1024:
    q=start+local
    i,j = (end>INT32_MAX) ? (q/S,q%S) : fast_divmod(q,S,m,shift)
    if 0<=x[q]<D: y[i*S*D+x[q]*S+j]=on
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `q=block*perCore+local` | `N` | `T1/ND` | API contiguous/reformat | SIMT patch | Kernel 全程 | 标量 load。 |
| GM `y` dense slice | `block*offValuePerCore + loop*single` | 每 fill block 为 `actual`，全局 `T` | `T3/ND` | `DataCopyPad` | SIMT patch/ViewCopy | 内部输出 | block slices 不重叠；最后 copy 使用真实字节数。 |
| `TBuf offValueUb` | UB base，单子区 | `ubSize=U-58KiB` bytes；元素容量 `floor(ubSize/sizeof(T3))` | `T3` flat | `Duplicate` | 所有本核 `DataCopyPad` | TPipe，Kernel lifetime | 同一内容跨 loop 复用；无 double buffer。 |
| SIMT 寄存器 | `q,i,j,dstOffset,m,shift` | 每线程标量 | integers/T3 | SIMT body | GM store | thread lifetime | 无 shared/dynamic UB。 |
| system workspace | framework-defined base | 16 MiB slot | bytes | executor | `SetSysWorkspace` | launch lifetime | user workspace 地址传入 Init 但 body 不使用。 |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  offValueUb["offValueUb · TBuf depth=1 · (platform_UB - 58 KiB) bytes"]
```

##### 字节账与复用顺序

```text
offValueUb_bytes = ubSize = platform_UB_bytes - 58*1024
UB_total = offValueUb_bytes
UB_total <= PostTiling 设置的 localMemorySize = platform_UB_bytes - 58*1024
dynamic_ub = N/A（SIMT asc_vf_call 未另传动态 UB）
workspace_kernel_slot = 16*1024*1024 bytes（system workspace）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 全部 `offValueUb` | `Duplicate` 一次 | 本核多次 `DataCopyPad` | Kernel 结束，无再次覆盖 | `V_MTE3 SetFlag/WaitFlag` 保证 Vector 写完成后 MTE3 读 | `initSize<=floor(ubSize/sizeof(T3))`。 |
| GM `y` | fill blocks 写 `off` | SIMT 读取索引后覆写合法类别位置；ViewCopy 最终消费 | on-patch 后不再由 fill 覆盖 | `SyncAll<true>` 位于所有 fill 与所有 patch 之间；batch schedule mode 1 | `blockDim=max(fill cores,input cores)`，所有参与 barrier 的 block 同时启动。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| UB fill→DMA | `SetFlag/WaitFlag<HardEvent::V_MTE3>` | `Duplicate` | 首个 `DataCopyPad` | `offValueUb` | `T3` 原 dtype，无 cast。 |
| dense→patch | `SyncAll<true>` + schedule mode 1 | 所有 fill block | 所有 SIMT patch block | 整个 GM `y[0:T)` | 防止迟到的 off-fill 覆盖 on-patch。 |
| fill tail | `offValueHasTail` 与 loop tail | Host POD | 最后 fill block/copy | `[0,T)` | copyLength 以元素计，传给 `DataCopyPad` 时乘 `sizeof(T3)` 成字节。 |
| input tail | `hasTail` | Host POD | 最后 input block | `[0,N)` | 仅处理真实索引数。 |
| 64 位 fallback | `endOffset>INT32_MAX` | Device per-block 范围 | `SimtComputeFor64Mix` | 大 q | int64 `/,%`；普通路径用 magic division。 |

当 `actual>cap` 且 `actual%cap==0` 时，源码令 `last=0`，循环只执行 `loops-1` 次满块 copy，再执行一次 0 字节 copy；固定源码没有额外整除修正。该边界可能导致最后一个满 UB chunk 未被 dense-fill，属于源码可见限制，不能在复用实现中擅自改写公式。

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp` | `CalUbTilingData/DoOpTiling/PostTiling` | dense-fill 分核、UB 扣减、key 1001、batch schedule。 |
| Kernel entry/router | `math/one_hot/op_kernel/one_hot_apt.cpp` | `one_hot`, `OneHotMix<DTYPE_X,DTYPE_DEPTH,DTYPE_Y>` | key→混合 AIV/SIMT body；仍为 0 AIC。 |
| Device body/helper | `math/one_hot/op_kernel/arch35/one_hot_mix.h` | `Init/CopyOut/InitOffValueAndCopyOut/Process/SimtComputeMix/SimtComputeFor64Mix` | UB、event、全核同步、DMA 参数、SIMT patch 与 tail。 |

## 算子特定优化

### Trick T-001：大输出先批量铺 offValue 再稀疏覆写 onValue

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`depth_ > 3 && inputTotalDimNum_ * uint64(depth_) > 1024`
- Golden 来源：[`math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp:143-164`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp#L143-L164)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (depth_ > DEPTH_THRESHHOLD && inputTotalDimNum_ * static_cast<uint64_t>(depth_) > TOTALSIZE_THRESHHOLD) {
    if (CalUbTilingData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "CalUbTiling failed.");
        return ge::GRAPH_FAILED;
    }
    isUsedUbInit_ = true;
} else {
    tilingData_.set_offValueUsedCoreNum(0);
    isUsedUbInit_ = false;
}
```

#### 机制与设计原因

直接路线对每个索引遍历全部 `D` 类别并逐元素写 off。R-002 把输出看成连续 `T=N*D`，先在 UB 用一次 `Duplicate(T3 offValue, initSize)` 生成模板，再以 `DataCopyPad` 多次复制到 GM；全核 `SyncAll` 后，SIMT 对每个合法索引只写一个 on 位置。该机制把大输出的 off 初始化从 SIMT 标量类别循环改成 Vector 生成 + MTE3 批量搬运，并复用同一 UB 内容。

#### 迁移边界

- 适用：输出绝大多数元素为同一默认值、每行只有一个稀疏例外的编码/掩码类算子。
- 前提：950、`D>3`、`N*D>1024`、可用 UB 大于 58 KiB，且能提供全 block barrier/batch schedule。
- 失效：小输出改用 R-001；若没有跨 block 同步，dense-fill 与 patch 可能互相覆盖。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 `SyncAll<true>` 删除或在它之前开始 on-patch，会允许晚到的 `DataCopyPad(offValue)` 覆盖正确的 `onValue`；固定源码以同步与 schedule mode 1 明确阻止此顺序错误。

### Trick T-002：为 32 位线性索引预计算 magic division

#### 触发与源码依据

- 适用 Route：R-001、R-002 的普通地址子路径
- 触发条件：R-001 `startOffset<=INT32_MAX`；R-002 `endOffset<=INT32_MAX`
- Golden 来源：[`math/one_hot/op_kernel/arch35/one_hot.h:101-106`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/one_hot/op_kernel/arch35/one_hot.h#L101-L106)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
auto lz = BIT64 - clz(factor1);
auto bc = bcnt1(factor1);
shift_ = (bc != 1) ? lz : lz - 1;
uint64_t magic = ((ONE << BIT32) * ((ONE << shift_) - factor1)) / factor1 + 1;
m_ = static_cast<uint32_t>(magic);
```

#### 机制与设计原因

`factor1=S` 来自 POD `suffixDim`。普通 SIMT body 用 `__umulhi(gmIdx,m_)`、加法和右移得到 `i=floor(q/S)`，再以 `j=q-i*S` 得余数，避免每线程在热点循环执行整数除法。幂为 2 的 `S` 用 `bcnt1==1` 调整 shift。超过 int32 范围时不继续使用该变换，而落到 int64 `/,%` body。

#### 迁移边界

- 适用：正整数固定除数、32 位非负线性索引的 flatten/unflatten。
- 前提：`S>0`；本算子由非空 API 路径和维度合并保证；索引范围满足对应 Route 的 int32 guard。
- 失效：大 offset 必须使用 `SimtComputeFor64*`，不得将 `gmIdx` 强制窄化。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

无视 offset guard 对大张量继续调用 32 位 body 会窄化 `gmIdx/coreFactor`；固定源码提供 `For64` fallback，构成可验证反例。

## 模板与实现对比

| 对比维度 | R-001 直接 SIMT | R-002 UB+SIMT | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `D<=3 \|\| N*D<=1024`，key 1000 | `D>3 && N*D>1024`，key 1001 | 输出规模与类别数共同决定批量初始化是否启用。 |
| 模板与 dtype | `OneHot<T1,T2,T3>`，36 组 950 binary dtype 组合 | `OneHotMix<T1,T2,T3>`，同 36 组 | dtype 不改变 key，只实例化模板类型。 |
| 循环与分核 | 输入 q 分核；每 q 全量写 D 个元素 | 输出 T 分核 fill；同步后输入 q 分核 sparse patch | 大 D 时分离 dense 与 sparse 阶段。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 UB；SIMT 标量寄存器 | 一个 `TBuf` 占 `U-58KiB`；SIMT 标量寄存器 | R-002 为批量 DMA 物化默认值模板。 |
| 内存总占用与复用 | UB 0 | UB=`U-58KiB`，同一 off 模板跨 copy loop 复用 | localMemorySize 与分配严格相等。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM 标量 load/store，`asc_vf_call` | `Duplicate`、`DataCopyPad`、`asc_vf_call` | 均无 NDDMA/通信/Cube。 |
| 同步、精度与 tail | 无显式 barrier；输入尾核；64 位 fallback | V→MTE3 event、`SyncAll<true>`；输出/输入双 tail；64 位 fallback | dense-fill 必须先于 patch 全局可见。 |
| fallback/失败边界 | unknown key 不执行；workspace null 早退 | 同上；另有 UB 扣减与整除 tail 边界 | 固定源码没有第三条 Kernel fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 API 文档写支持，但固定仓的该实现未注册/构建 `ascend910b`，source closure 不足 | 公开 API 文档写支持，但固定仓的该实现未注册/构建 `ascend910_93`，source closure 不足 | 公开文档、`AddConfig("ascend950")`、binary config 与 Kernel 接线一致支持。 |
| Host/Launcher | 固定仓不可见 | 固定仓不可见 | `IsRegBase(DAV_3510)` 启用 950 dtype；arch35 Tiling 两 key。 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | 单 `KERNEL_TYPE_MIX_AIV_1_0`，SIMT 或 AIV+SIMT；UB 扣 58 KiB。 |

## 限制与失败边界

- API 仅接受精确 dtype、rank、axis 和输出 shape；950 `self<=7D`，值与输出 `<=8D`。`numClasses` 为 C `int`，不会形成大于 `INT_MAX` 的公开输入。
- Tiling 的 `usedCoreNum/offValueUsedCoreNum` 是 `int16_t`，固定源码未见显式窄化 guard；`ubSize=U-58KiB` 也未检查下溢。
- R-002 在 `actual>cap && actual%cap==0` 时形成 0 字节末 copy，源码未见整除修正；复用实现必须验证该边界后再决定是否保持或修复。
- entry 对 null workspace/user workspace 直接 return，unknown key 也静默不写；固定 Device 源码未提供向 Host 传播该早退的状态通道。
- `OneHotTilingData` 宏生成后的精确 ABI offsets、框架 `Contiguous/ReFormat/ViewCopy` 的 Kernel/workspace 切片与 executor stream/event 实现固定仓不可见。
- A2/A3 的公开支持与本目录缺失相应 `ascend910b/ascend910_93` 实现接线之间无法闭合，因此未生成无后缀正式卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/one_hot/docs/aclnnOneHot.md` | `aclnnOneHotGetWorkspaceSize/aclnnOneHot` | 平台、dtype、rank、shape、错误与两段式语义。 |
| 入口声明/约束 | `math/one_hot/op_api/aclnn_one_hot.h` | 两个 ACLNN 原型 | C ABI 参数与公开符号。 |
| 入口实现 | `math/one_hot/op_api/aclnn_one_hot.cpp` | `CheckParams/aclnnOneHotGetWorkspaceSize/aclnnOneHot` | 校验、短路、Contiguous/ReFormat/OneHot/ViewCopy、executor。 |
| 入口实现 | `math/one_hot/op_api/one_hot.cpp` | `IsAiCoreSupport/l0op::OneHot` | 950 dtype selector、InferShape 与 launcher 添加。 |
| 入口实现 | `common/inc/op_api/aclnn_check.h` | `IsRegBase` | `DAV_3510` 的 950/RegBase 判定。 |
| Host/Tiling/Launcher | `math/one_hot/op_host/one_hot_infershape.cpp` | `InferShape4OneHot` | axis 插入与 depth 数据依赖。 |
| Host/Tiling/Launcher | `math/one_hot/op_host/one_hot_def.cpp` | `ops::OneHot` | dtype tuple、950 注册与 `one_hot_apt` 接线。 |
| Host/Tiling/Launcher | `math/one_hot/op_host/arch35/one_hot_tiling.h` | `OneHotCompileInfo` | AIV 核数、UB compile info。 |
| Host/Tiling/Launcher | `math/one_hot/op_host/arch35/one_hot_tiling.cpp` | `OneHotTiling/TilingPrepare4OneHot` | 平台资源 producer 与 Tiling 注册。 |
| Host/Tiling/Launcher | `math/one_hot/op_host/arch35/one_hot_tiling_arch35.h` | `OneHotTilingData/OneHotTilingBase` | POD 字段、常量与 selector 类。 |
| Host/Tiling/Launcher | `math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp` | `MergeDims/GetTilingParam/CalUbTilingData/DoOpTiling/PostTiling` | 两 key、分核、UB、workspace、blockDim。 |
| Kernel entry/body | `math/one_hot/op_kernel/one_hot_apt.cpp` | `one_hot` | task type、workspace guard、key router。 |
| Kernel entry/body | `math/one_hot/op_kernel/arch35/one_hot.h` | `OneHot/SimtCompute*` | R-001 完整 body。 |
| Kernel entry/body | `math/one_hot/op_kernel/arch35/one_hot_mix.h` | `OneHotMix/SimtCompute*Mix` | R-002 完整 body、UB、DMA 与同步。 |
| 构建接线 | `math/one_hot/op_host/config/ascend950/one_hot_binary.json` | `op_type=OneHot` 的 36 个条目 | 950 dtype tuple binaries。 |
| 构建接线 | `math/one_hot/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | one_hot 模块构建入口。 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `name=OneHot` | 950 编译选项与 `auto_sync=false`。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | `P/S`、两套分核、UB 与 tail 字段可直接复用 | 字段声明顺序/位宽；两个 inactive 保留字段；宏生成 offsets 不可猜 | `OneHotTilingData` |
| 调用顺序 | API 前处理→OneHot→ViewCopy；R-002 fill→barrier→patch | 空域必须在 launcher 前短路；patch 不得先于全局 fill 完成 | `aclnnOneHotGetWorkspaceSize`, `OneHotMix::Process` |
| workspace/内存 | 16 MiB system workspace；R-002 单 TBuf | executor 总 workspace 不等于简单 16 MiB；UB 必须扣 58 KiB | `GetWorkspaceSize`, `DoOpTiling`, `OneHotMix::Init` |
| 同步 | V→MTE3 event 与 `SyncAll<true>` | `auto_sync=false`，不能依赖构建系统自动补同步；batch schedule mode 1 必须配套 | `InitOffValueAndCopyOut`, `Process`, `PostTiling` |
| 精度与尾块 | 值以 T3 直接读写；输入/输出双 tail；大 offset 64 位 fallback | 不添加数值 cast；copyLength 的元素→字节换算；合法索引范围判断 | `SimtCompute*`, `CopyOut` |
| 失败边界 | 参数失败阻断 executor；unknown key/null workspace 不写输出 | 不把 Device 静默 early-return 宣称为成功计算 | `CheckParams`, `one_hot` |
