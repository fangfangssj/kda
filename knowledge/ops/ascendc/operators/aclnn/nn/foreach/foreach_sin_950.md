---
type: Operator
title: "ForeachSin 950 ACLNN 算子知识"
description: "说明 aclnnForeachSin 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/docs/aclnnForeachSin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/foreach_sin_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/arch35/foreach_sin_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/arch35/foreach_sin_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/arch35/foreach_sin_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/arch35/foreach_sin.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/arch35/foreach_sin_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/config/ascend950/foreach_sin_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:20:33Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSin 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSinGetWorkspaceSize` + `aclnnForeachSin`，公开 ACLNN 两段式接口；固定仓未提供生成后的 C 头和 inner bridge。 |
| 计算语义 | 对 TensorList 中每个张量逐元素计算 `y_i[j]=sin(x_i[j])`。 |
| 输入输出 | 动态 TensorList；ND；FLOAT16/FLOAT32/BFLOAT16；输入与输出 dtype一致。 |
| 执行类型 | 单 Kernel、单 SIMT 节点；每个 VF block 512 threads，无 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 FP16 SIMT；R-002 FP32 SIMT；R-003 BF16 SIMT。 |
| 核心资源 | TensorList GM 描述符、每线程标量寄存器；无显式 AIV UB Queue/TBuf、shared子区或 atomic；Host把 SIMT local memory设为 `ubSize-128KiB` 并上报16MiB工具workspace。 |
| 来源闭包 | source_closed：公开 API 文档→950 OpDef/config→arch35 Tiling/POD/template key→SIMT entry/body→950 构建接线闭合；生成式 ACLNN bridge 与 `sinf` 库内部为明确框架/库边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x=[x_0,...,x_{n-1}]` 与 `y=[y_0,...,y_{n-1}]`。950 Kernel 对每个张量取得独立 GM 数据指针，并由所有 SIMT blocks/threads以 grid-stride 覆盖 `[0, tensorElements[i])`，执行 `float f=static_cast<float>(x_i[j]); y_i[j]=static_cast<T>(sinf(f))`。FP16/BF16 因此升到 FP32 计算再降回存储 dtype；FP32 路线保持 FP32。公开值域保证为 `[-65504,65504]`，接口默认确定性。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 0–8维、ND；可非连续，OpDef启用 `AutoContiguous()` | FLOAT16/FLOAT32/BFLOAT16，列表内一致 | 非空指针；支持零元素 Tensor。950 Tiling最多256个实例。 |
| `out` | TensorList；每个 Tensor 0–8维、ND；公开接口不支持非连续输出 | 与 `x` 一致，列表内一致 | 对应输出 shape size不小于输入；只写 `[0,inputElements)`。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | arch35 Tiling上报固定 `16*1024*1024` 字节 `ASCENDC_TOOLS_WORKSPACE`。 |
| `executor` | 执行器输出 | `aclOpExecutor**` | 第一段创建，第二段消费；生成实现固定仓不可见。 |
| `workspace`、`stream` | Device地址、执行流 | 指针/handle | workspace由调用者持有；Kernel签名接收但 body不访问。 |

### 入口与版本

固定 Commit 只公开 `aclnnForeachSin` 版本族。公开 README/API 支持 Ascend 950PR/950DT；OpDef注册 `ascend950`，binary config列出 FP16/FP32/BF16，Host CMake映射到 arch35，Kernel CMake显式选 `arch35/foreach_sin.cpp`，无发布冲突。生成式 ACLNN声明/inner bridge固定仓不可见，作为框架边界；算子拥有的 Host selector、POD、模板实例与SIMT body均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口约束 `x/out` 非空、dtype合法且一致、每项不超过8维、输出容量满足输入，输入可非连续而输出必须连续。OpDef的 input/output均为动态ND并 `AutoContiguous()`；公共 infer要求输出实例数等于输入实例数并逐项传播shape/dtype。arch35 Tiling再检查平台信息、AIV核数和UB非零、`tensorNum∈[0,256]`；若 `tensorNum>0`，从第0项取得 selector dtype并读取所有输入storage元素数。arch35 Tiling自身未重复验证列表内其余dtype和输出shape，这些属于公开ACLNN/OpDef前置边界。

### 参数、Workspace 与生命周期

Host读取 `maxCoreNum=GetCoreNumAiv()` 与 `ubSize`。令 `M=max_i tensorElements[i]`，计算 `perCoreElements=align_up(max(1024,ceil(M/maxCoreNum)),32)`，`needCoreNum=max(1,min(maxCoreNum,ceil(M/perCoreElements)))`。Host把 `tensorNum`、`needCoreNum`、每项元素数写POD，设置dtype模板key、`blockDim=needCoreNum`、`localMemorySize=ubSize-128KiB`，并上报16MiB workspace。Device未读 `needCoreNum` 和workspace，而通过SIMT runtime读取实际block数；workspace所有权从第一段返回持续到stream完成。

### 启动图与失败传播

单节点图：executor在用户stream启动实例化的 `foreach_sin<schMode>(xListDesc,yListDesc,workspace,tiling)`，grid blocks=`needCoreNum`；entry解析 `ForeachSinTilingData`后按编译期 `schMode` 调用一种 `Process<T>`。`Process`对TensorList顺序执行一次 `VF_CALL<OpForeachSinSimt<T>>(Dim3(512),...)`；每个Tensor彼此无数据依赖或workspace边。Host取平台信息、POD、dtype、workspace指针失败即返回 `GRAPH_FAILED` 并阻断launch。模板只实例化0/1/2三种模式；非法模式没有body分支。stream ordering和executor错误码映射固定仓不可见。

## Route 与参数设计

### 分核与资源计算

分核只以列表中最大Tensor的元素数 `M` 为依据，不按Tensor总元素数求和。`perCoreElements`至少1024元素且向上对齐32元素；`needCoreNum`至少1且不超过AIV核数。对任一Tensor `i`，thread全局起点为 `blockIdx*512+threadIdx`，步长为 `512*blockNum`；循环上界是POD `tensorElements[i]`。因此较小Tensor自然由少量线程命中，较大Tensor在grid stride中多轮覆盖。Host把launch local memory设为 `ubSize-128KiB`；固定body未显式划分或访问其shared/dynamic子区。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 SIMT | `GET_TPL_TILING_KEY(0)`；`foreach_sin<0>` | `xDtype==DT_FLOAT16` | 否则继续FLOAT |
| 2 | R-002 FP32 SIMT | `GET_TPL_TILING_KEY(1)`；`foreach_sin<1>` | `xDtype==DT_FLOAT` | 否则继续BF16 |
| 3 | R-003 BF16 SIMT | `GET_TPL_TILING_KEY(2)`；`foreach_sin<2>` | `xDtype==DT_BF16` | 其它dtype `GRAPH_FAILED` |

```text
if tensorNum>0: xDtype = input[0].dtype else xDtype = FLOAT
if xDtype==FLOAT16 -> schMode 0 -> R-001
else if xDtype==FLOAT -> schMode 1 -> R-002
else if xDtype==BF16 -> schMode 2 -> R-003
else -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 SIMT | mode 0 | FP16、tensorNum 0..256、950 | 单SIMT `foreach_sin<0>` | `Process<half>`→`OpForeachSinSimt<half>` | 直接GM，512 threads/block，无barrier/atomic | `idx<count` grid-stride guard；无fallback |
| R-002 FP32 SIMT | mode 1 | FP32、同上 | 单SIMT `foreach_sin<1>` | `Process<float>`→`OpForeachSinSimt<float>` | 同上 | 同上 |
| R-003 BF16 SIMT | mode 2 | BF16、同上 | 单SIMT `foreach_sin<2>` | `Process<bfloat16_t>`→`OpForeachSinSimt<bfloat16_t>` | 同上 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor数 | `0<=tensorNum<=256` | POD数组固定256 | 全部 | `GRAPH_FAILED` | `ForeachSinTilingFunc` |
| Tensor数窄化 | `GetInstanceNum()`先 `static_cast<int32_t>` 再检查 | POD循环和字段均为32位计数 | 全部 | 窄化后落在范围外则失败；窄化前超大值语义固定仓未防护 | `ForeachSinTilingFunc` |
| 平台资源 | `coreNum>0 && ubSize>0` | 必须形成非零grid并设置local memory | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| DCache保留 | `localMemorySize=uint32(ubSize-128KiB)` | 为DCache扣除128KiB | 全部 | 固定仓无 `ubSize>=128KiB` guard | `DCACHE_SIZE/SetLocalMemorySize` |
| 最小每核域 | `perCoreElements>=1024`且align 32 elements | 限制小Tensor启用的block数 | 全部 | 少用block，无备用Route | `MIN_PER_CORE_ELEMENTS` |
| 线程块 | `THREAD_NUM=512`，`LAUNCH_BOUND(512)` | VF启动与grid stride步长一致 | 全部 | 无其它threads模板 | `OpForeachSinSimt` |
| 空Tensor | `count==0` | 避免取得GM数据指针和VF_CALL | 全部 | continue | `Process` |
| dtype | 只接受FP16/FP32/BF16 | 三个编译期模板实例 | 全部 | Host失败/非法mode无body | `ForeachSinTilingFunc/foreach_sin` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSinTilingData` | `tensorNum` | `int32_t`/32 | 首字段；固定仓未显式给出整体 `sizeof/alignof` | Tensor个数 | `tiling->tensorNum=tensorNum` | `Process` 的Tensor循环上界 | 0..256 |
| `ForeachSinTilingData` | `needCoreNum` | `int32_t`/32 | 紧随 `tensorNum`；精确offsetof固定仓未显式给出 | blocks | `tiling->needCoreNum=needCoreNum` | Device未读取；launch消费的是`SetBlockDim` | 初始化至少1，POD字段inactive |
| `ForeachSinTilingData` | `tensorElements[256]` | `int64_t[256]` | 紧随两个int32；精确sizeof/alignof/offsetof固定仓未显式给出 | 元素/张量 | `shape.GetShapeSize()` | `Process: count=tensorElements[i]` | 仅前`tensorNum`项有效，其余memset为0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | mode0 | `ForeachSinTilingFunc` | `ForeachSinTilingData`、blockDim/localMemory | `foreach_sin<0>` | `Process<half>/OpForeachSinSimt<half>/ComputeSin` | 950 Host/Kernel CMake + ascend950 binary |
| R-002 | mode1 | 同上 | 同上 | `foreach_sin<1>` | `Process<float>/OpForeachSinSimt<float>/ComputeSin` | 同上 |
| R-003 | mode2 | 同上 | 同上 | `foreach_sin<2>` | `Process<bfloat16_t>/OpForeachSinSimt<bfloat16_t>/ComputeSin` | 同上 |

## Kernel 实现

### 公共执行流程

entry通过 `REGISTER_TILING_DEFAULT` 和 `GET_TILING_DATA_WITH_STRUCT` 取得POD；编译期 `schMode`选择dtype模板。`Process<T>`在每个block内构造输入/输出 `ListTensorDesc`，依次遍历Tensor；零元素项直接跳过，否则取第i项GM数据地址并发起512线程VF。VF中每线程从全局thread id出发，以整个grid线程总数为步长处理多个元素。每个索引只有一个thread写，没有atomic、barrier、shared/dynamic UB或跨Tensor workspace交接。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | `ForeachSinTilingFunc` | `xDtype==DT_FLOAT16` | mode0 |
| Router | `foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<schMode>` | `schMode==TILING_KEY_FLOAT16(0)` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sin<0>(x,y,workspace,tiling)
  -> NsForeachSin::Process<half>
    -> ListTensorDesc::GetDataPtr<half>(i)
    -> VF_CALL<OpForeachSinSimt<half>>(Dim3(512), xPtr, yPtr, count)
      -> ComputeSin<half> -> sinf(float) -> half
```

##### 源码等价伪码

```text
# 源码等价伪码
for tensor i in [0,tensorNum):
  count=tensorElements[i]; if count==0: continue
  xPtr=list_x[i]; yPtr=list_y[i]
  parallel 512 threads/block:
    idx=blockIdx*512+threadIdx
    while idx<count:
      yPtr[idx]=half(sinf(float(xPtr[idx])))
      idx += 512*blockNum
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| ListTensor desc GM | `x/y` entry地址 | `tensorNum`项 | descriptor | ACLNN框架 | `ListTensorDesc` | caller/executor | ABI固定仓不可见 |
| input/output GM | `GetDataPtr(i)+idx` | `count`元素 | FP16/ND linear | caller/thread | thread/caller | caller，stream完成前有效 | 每线程标量访问 |
| thread scalar | 线程私有 | 1个T + FP32中间 | half/float | load/cast/sinf | cast/store | 单次循环迭代 | 寄存器字节数固定仓不可见 |
| workspace | entry参数 | 16MiB上报 | bytes | caller | 无consumer | caller | 未访问 |

##### 分层内存布局

N/A：该SIMT节点不调用 `InitBuffer`，没有显式AIV UB Queue/TBuf或body可见的shared子区；实际路径是GM标量load→线程私有FP32计算→GM标量store。launch仍按Host设置获得 `ubSize-128KiB` local memory。

##### 字节账与复用顺序

```text
AIV_UB_queue_total = 0
dynamic_ub/localMemorySize = ubSize - 128KiB（Host设置；固定body无显式子分配）
workspace_total = 16MiB（Kernel未访问）
register_bytes = 固定仓不可见，仅可证明val/fval/result为单迭代线程私有
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有标量 | 当前idx load/sinf | 当前idx store | 下一grid-stride idx | 程序顺序 | 每线程独占；无跨线程别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| grid-stride | 无barrier/atomic | thread load | 同thread sinf/store | 唯一idx | `idx<count`保护tail；FP16升FP32再降FP16 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | `ForeachSinTilingFunc` | mode0、blockDim、POD、workspace |
| Kernel entry/router | `foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<0>` | mode0→half |
| Device body/helper | `foreach_sin/op_kernel/arch35/foreach_sin_simt.h` | `Process<half>/OpForeachSinSimt<half>` | 512线程grid-stride与FP32 sinf |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | `ForeachSinTilingFunc` | `xDtype==DT_FLOAT` | mode1 |
| Router | `foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<schMode>` | `schMode==TILING_KEY_FLOAT(1)` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sin<1> -> Process<float>
  -> VF_CALL<OpForeachSinSimt<float>>(512)
    -> ComputeSin<float>: float -> sinf -> float
```

##### 源码等价伪码

```text
for each nonempty tensor:
  for idx=blockIdx*512+threadIdx; idx<count; idx+=blockNum*512:
    y[idx]=sinf(x[idx])  # static_cast<float>保持float
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `list[i]+idx` | count | FP32/ND linear | caller/thread | thread/caller | caller | 标量load/store |
| thread scalar | 私有 | val/fval/result | FP32 | load/sinf | sinf/store | 当前迭代 | 无跨线程共享 |
| workspace | entry参数 | 16MiB | bytes | caller | N/A | caller | 未访问 |

##### 分层内存布局

N/A：直接GM SIMT路径，无显式UB/shared子区；Host仍设置 `localMemorySize=ubSize-128KiB`。

##### 字节账与复用顺序

```text
AIV_UB_queue_total=0; dynamic_ub/localMemorySize=ubSize-128KiB（无显式子分配）
workspace_total=16MiB（未访问）
register_bytes=固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | 当前idx | 当前idx store | 下一idx | thread程序顺序 | grid-stride索引唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT compute | 无barrier/atomic | GM load | sinf/store | 每线程idx | FP32；`idx<count` tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | dtype selector | FLOAT→mode1 |
| Kernel entry/router | `foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<1>` | mode1→float |
| Device body/helper | `foreach_sin/op_kernel/arch35/foreach_sin_simt.h` | `ComputeSin<float>` | FP32 sinf与直接写回 |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | `ForeachSinTilingFunc` | `xDtype==DT_BF16` | mode2 |
| Router | `foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<schMode>` | `schMode==TILING_KEY_BF16(2)` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sin<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachSinSimt<bfloat16_t>>(512)
    -> ComputeSin<bfloat16_t>: BF16 -> float -> sinf -> BF16
```

##### 源码等价伪码

```text
for each nonempty tensor:
  for idx=blockIdx*512+threadIdx; idx<count; idx+=blockNum*512:
    f=float(x[idx]); result=sinf(f); y[idx]=bfloat16_t(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `list[i]+idx` | count | BF16/ND linear | caller/thread | thread/caller | caller | 标量load/store |
| thread scalar | 私有 | BF16 val + FP32 fval/result | mixed | load/cast/sinf | cast/store | 当前迭代 | 无共享 |
| workspace | entry参数 | 16MiB | bytes | caller | N/A | caller | 未访问 |

##### 分层内存布局

N/A：直接GM SIMT路径，无显式UB/shared子区；Host仍设置 `localMemorySize=ubSize-128KiB`。

##### 字节账与复用顺序

```text
AIV_UB_queue_total=0; dynamic_ub/localMemorySize=ubSize-128KiB（无显式子分配）
workspace_total=16MiB（未访问）
register_bytes=固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | 当前idx | 当前idx store | 下一idx | thread程序顺序 | idx唯一；cast结束后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT compute | 无barrier/atomic | BF16 load | FP32 sinf/BF16 store | 每线程idx | FP32中间；降BF16 round细节固定仓不可见；`idx<count` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | dtype selector | BF16→mode2 |
| Kernel entry/router | `foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<2>` | mode2→bfloat16_t |
| Device body/helper | `foreach_sin/op_kernel/arch35/foreach_sin_simt.h` | `ComputeSin<bfloat16_t>` | FP32中间计算与BF16写回 |

## 算子特定优化

### Trick T-001：以最大Tensor定grid并让所有Tensor复用grid-stride

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：任一非空Tensor进入 `OpForeachSinSimt<T>`；grid由列表最大Tensor的Host分核结果决定。
- Golden 来源：[`foreach/foreach_sin/op_kernel/arch35/foreach_sin_simt.h:53-59`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/arch35/foreach_sin_simt.h#L53-L59)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t idx = static_cast<uint64_t>(
         Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
     idx < static_cast<uint64_t>(elemCount);
     idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
    T val = xPtr[idx];
    yPtr[idx] = ComputeSin(val);
}
```

#### 机制与设计原因

Host无需为每个Tensor产生独立核区POD，只以最大Tensor选择统一block数；Device为每项重新使用同一grid，grid-stride让线程覆盖任意元素数，`idx<elemCount`同时承担tail保护。该设计消除显式tile、UB DMA和每Tensor分核数组，代价是小Tensor可能只有部分线程有效；固定仓没有性能测量，不能推断收益比例。

#### 迁移边界

- 适用：TensorList各项独立、逐元素、无跨元素状态的SIMT unary计算。
- 前提：每个输出索引只由一个thread写；所有Tensor可以共用同一grid；列表循环和VF调用顺序不要求跨Tensor同步。
- 失效：归约、邻域访问、跨Tensor依赖或需要shared缓存时必须引入对应同步/资源路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16统一FP32 opmath

#### 触发与源码依据

- 适用 Route：R-001、R-003；R-002也复用同一函数但cast为恒等类型。
- 触发条件：`ComputeSin<T>`被每个有效idx调用。
- Golden 来源：[`foreach/foreach_sin/op_kernel/arch35/foreach_sin_simt.h:37-43`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/arch35/foreach_sin_simt.h#L37-L43)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__simt_callee__ inline T ComputeSin(T val)
{
    float fval = static_cast<float>(val);
    float result = sinf(fval);
    return static_cast<T>(result);
}
```

#### 机制与设计原因

存储dtype由模板决定，但计算统一走FP32 `sinf`；低精度路线避免以FP16/BF16直接近似Sin。所有中间量线程私有，无需UB cast buffer或barrier。降精度由`static_cast<T>`完成，具体硬件round mode固定仓未显式给出。

#### 迁移边界

- 适用：可逐元素升精度、FP32数学库可用、最终可回写低精度的SIMT unary算子。
- 前提：接受目标平台 `static_cast<T>` 的数值语义，输入有效域满足公开精度约束。
- 失效：要求特定round mode、异常值策略或更高精度时必须替换转换/数学实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | mode0 | mode1 | mode2 | 第0输入dtype决策树 |
| 模板与 dtype | `Process<half>` | `Process<float>` | `Process<bfloat16_t>` | 三个编译期实例 |
| 循环与分核 | 统一512-thread grid-stride | 同左 | 同左 | dtype不改变调度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程私有half+float标量 | float标量 | BF16+float标量 | 无显式UB/shared |
| 内存总占用与复用 | 寄存器字节不可见 | 同左 | 同左 | 每迭代复用标量 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM scalar + `sinf` | 同左 | 同左 | SIMT |
| 同步、精度与 tail | 无barrier；FP32中间→FP16 | FP32 | FP32中间→BF16 | `idx<count`统一tail |
| fallback/失败边界 | 无fallback | 无fallback | 无fallback | unsupported dtype Host失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持 |
| Host/Launcher | arch22公共foreach Tiling | arch22公共foreach Tiling | arch35专用Tiling、最多256项、16MiB workspace |
| Kernel/资源 | AIV Vector Queue | AIV Vector Queue | 直接GM的SIMT、512 threads/block |

## 限制与失败边界

- 公开输入值域保证为 `[-65504,65504]`；超出不保证精度。输出非连续Tensor不支持。
- 950 Tiling只从第0输入实例取dtype；列表内dtype一致、输出shape/dtype等约束依赖ACLNN/OpDef前置层，arch35函数内不重复检查。
- `tensorNum`最大256；`GetInstanceNum()`在检查前窄化成`int32_t`，固定仓没有窄化前范围guard。空列表把dtype默认成FLOAT、blockDim设为1，Device循环不发VF调用。
- `SetLocalMemorySize(static_cast<uint32_t>(ubSize-128KiB))`没有 `ubSize>=128KiB` 的显式guard；仅检查UB非零。
- `maxTensorElements`、`CeilDiv`与`CeilAlign`使用`int64_t`但没有显式算术溢出检查。
- 上报16MiB workspace但Kernel不访问；其工具用途和executor内部切片固定仓不可见。生成式ACLNN bridge、ListTensor描述符ABI、template key编码细节、SIMT VF内部调度、寄存器字节数和低精度`static_cast` round mode固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sin/docs/aclnnForeachSin.md` | `aclnnForeachSin*` | API、950平台、dtype/shape/错误与精度 |
| 入口实现 | `foreach/foreach_sin/op_host/foreach_sin_def.cpp` | `ForeachSin` | ascend950、ND、AutoContiguous |
| Host infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon/InferDataType4ForeachCommon` | 输出实例、shape、dtype传播 |
| Host/Tiling | `foreach/foreach_sin/op_host/arch35/foreach_sin_tiling.cpp` | `ForeachSinTilingFunc` | 平台、分核、key、POD、workspace/local memory |
| POD | `foreach/foreach_sin/op_kernel/arch35/foreach_sin_tiling_data.h` | `ForeachSinTilingData` | 三字段POD |
| template key | `foreach/foreach_sin/op_kernel/arch35/foreach_sin_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | mode 0/1/2实例集合 |
| Kernel entry/body | `foreach/foreach_sin/op_kernel/arch35/foreach_sin.cpp` | `foreach_sin<schMode>` | mode→dtype router |
| Kernel body | `foreach/foreach_sin/op_kernel/arch35/foreach_sin_simt.h` | `Process/OpForeachSinSimt/ComputeSin` | Tensor循环、512-thread grid-stride、FP32 sinf |
| 构建接线 | `foreach/foreach_sin/op_host/config/ascend950/foreach_sin_binary.json` | `ForeachSin op_list` | 950三dtype binary实例 |
| 构建接线 | `foreach/foreach_sin/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_sin/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_sin.cpp` | 950 Kernel选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | `tensorNum/needCoreNum/tensorElements[256]` | 字段类型/顺序、数组上限、元素单位；`needCoreNum` Device inactive | `ForeachSinTilingData` |
| 调用顺序 | entry→Process→逐Tensor VF_CALL→grid-stride | 零Tensor continue、Tensor循环顺序、512 threads | `Process/OpForeachSinSimt` |
| workspace/内存 | 直接GM，无显式UB；16MiB workspace未访问 | TensorList owner、stream lifetime、DCache扣减 | `ForeachSinTilingFunc` |
| 同步 | 无barrier/atomic，线程独占idx | `idx`起点/步长必须保持全grid唯一 | `OpForeachSinSimt` |
| 精度与尾块 | FP32 `sinf`统一opmath，`idx<count`tail | 低精度升降顺序与有效域 | `ComputeSin` |
| 失败边界 | Host平台/tensorNum/dtype失败阻断launch | 非法mode不得视为fallback | `GetPlatformInfo/ForeachSinTilingFunc/foreach_sin` |