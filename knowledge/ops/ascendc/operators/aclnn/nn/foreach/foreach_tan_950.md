---
type: Operator
title: "ForeachTan 950 ACLNN 算子知识"
description: "说明 aclnnForeachTan 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/docs/aclnnForeachTan.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/foreach_tan_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/arch35/foreach_tan_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/arch35/foreach_tan_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/arch35/foreach_tan_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/arch35/foreach_tan.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/arch35/foreach_tan_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/config/ascend950/foreach_tan_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:22:21Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachTan 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachTanGetWorkspaceSize` + `aclnnForeachTan`，公开 ACLNN 两段式接口。 |
| 计算语义 | 对 TensorList 的每个有效元素计算 `tan`；FP16/BF16 显式转 FP32 调用 `tanf` 后转回。 |
| 输入输出 | ND TensorList，FP16/FP32/BF16；列表内 dtype 一致，输出对应项容量不少于输入。 |
| 执行类型 | 单 Kernel、单个 SIMT 节点；每 block 512 线程，grid-stride 直接读写 GM；不存在 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16→FP32 SIMT；R-003 BF16→FP32 SIMT。 |
| 核心资源 | ListTensor GM 描述符、直接 GM load/store、SIMT thread/grid；显式 UB Queue/TBuf 为 N/A；固定 16 MiB AscendC tools workspace。 |
| 来源闭包 | `source_closed`：公开 API→950 OpDef/config→arch35 Tiling/POD/key→模板 entry→SIMT VF body 与构建接线闭合；生成 ACLNN bridge、`VF_CALL` 调度器内部和 math intrinsic 实现属于框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入 `x=[x_0,...,x_(n-1)]`，输出 `y`。对每个列表成员 `i` 和 `0<=j<numel(x_i)`，`y_i[j]=tan(x_i[j])`。950 FP32 路线直接执行 `tanf(float)`；FP16/BF16 路线严格按 source 顺序执行低精度→FP32、`tanf`、FP32→低精度。输出容量可大于输入，但多出元素不被写入。公开值域精度约束为 `[-65504,65504]`，默认确定性。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，每项维数 0–8 | FP16/FP32/BF16 | 输入 TensorList 指针非空；成员 dtype 一致；支持空 Tensor；输入非连续由 `AutoContiguous` 框架边界处理。 |
| `out` | ND，每项维数 0–8，容量≥对应输入 | 与 x 相同 | 输出 TensorList 指针非空且输出须连续；Kernel 只写输入有效范围。 |
| `workspaceSize`/`executor` | Host 输出 | `uint64_t*`/executor | 第一段返回；包含框架计算流程。 |
| `workspace`/`workspaceSize`/`stream` | Device 地址/字节/流 | N/A | 第二段提交；算子 Tiling 报告 16 MiB workspace。 |

### 入口与版本

固定仓公开文档仅给无版本后缀的 `aclnnForeachTan`，950 产品 `Ascend 950PR/950DT` 明确支持。OpDef 的 `ascend950` 配置支持三种 dtype、动态 rank/shape，并指向 `foreach_tan`；binary config 为三种 dtype 各列一个 ND 动态 binary。未出现更高公开 API 版本。生成的 ACLNN header/inner bridge 不在固定仓，executor 组装和 Contiguous 临时量细节按框架边界处理。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口校验空指针、支持 dtype、输入输出 dtype、shape 容量、列表成员 dtype 与维数。arch35 `ForeachTanTilingFunc` 再读取平台 AIV 核数和 UB，取得动态输入实例数 `tensorNum`，显式限制 `0<=tensorNum<=256`，逐项读取 storage shape 元素数；首项决定 dtype，dtype 不是 FP32/FP16/BF16 时返回 `GRAPH_FAILED`。输出侧深度校验的生成 ACLNN bridge 实现固定仓不可见，公开错误语义仍作为入口约束。

### 参数、Workspace 与生命周期

设 950 AIV 核数为 `C`，平台 UB 为 `U`。Host 写 `workspace[0]=16*1024*1024` 字节，设置 `localMemorySize=U-128*1024`，并以 `needCoreNum` 作为 blockDim。Kernel 参数保留 workspace 但 body 不读；16 MiB 被注释为 AscendC tools workspace，具体内部消费者固定仓不可见。ACLNN caller 持有 workspace 至 stream 完成，executor 至第二段提交完成；框架 Contiguous 可能带来的额外合并空间与 offset 固定仓不可见。

### 启动图与失败传播

单节点图：ACLNN executor/可能的 Contiguous→`ForeachTanTilingFunc`→`foreach_tan<schMode>(xList,yList,workspace,tiling)`。节点 grid 为 `needCoreNum` blocks，每 block 的 VF 线程维度为 512；POD 为 `ForeachTanTilingData`。没有 Kernel 间边。平台信息、描述符、shape、POD、workspace 数组或 dtype 任一检查失败均阻断 launch。模板 key 只从 0/1/2 产生；编译期 entry 对其它 `schMode` 无 body 分支。

## Route 与参数设计

### 分核与资源计算

令 `M=max_i numel(x_i)`，平台最大 AIV 核数为 `C`。Host 计算 `perCore=max(1024,ceil(M/C))`，再向上对齐到 32 个元素；`needCoreNum=max(1,min(C,ceil(M/perCore)))`。因此所有 Tensor 共用按最大成员选出的 grid。对任一成员元素 `idx`，线程初值为 `blockIdx*512+threadIdx`，循环步长为 `512*blockNum`，直到 `idx>=numel(x_i)`；成员为 0 元素时在 `Process` 中直接跳过。

Host 读取 `U` 仅用于 `SetLocalMemorySize(U-128KiB)`；SIMT body 没有 shared/dynamic UB、Queue 或 TBuf 分配。`needCoreNum` 同时写 POD，但 body 不读取该字段，实际 block 数由 launch grid 提供。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `schMode=0` | `xDtype==DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16→FP32 SIMT | `schMode=1` | `xDtype==DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BF16→FP32 SIMT | `schMode=2` | `xDtype==DT_BF16` | 其它 dtype `GRAPH_FAILED` |

```text
if xDtype == FP32: GET_TPL_TILING_KEY(0) -> foreach_tan<0> -> R-001
else if xDtype == FP16: GET_TPL_TILING_KEY(1) -> foreach_tan<1> -> R-002
else if xDtype == BF16: GET_TPL_TILING_KEY(2) -> foreach_tan<2> -> R-003
else: GRAPH_FAILED, no launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | 950、FP32、tensorNum≤256、公开 shape/dtype 约束 | 单 SIMT Kernel | `Process<float>`→`VF_CALL<OpForeachTanSimt<float>>` | 直接 GM；无显式 UB/barrier/atomic | grid-stride `idx<count` 保护 |
| R-002 FP16 | schMode 1 | 同上但 FP16 | 单 SIMT Kernel | `Process<half>`→`OpForeachTanHalfSimt` | 每线程寄存器中 FP16→FP32→FP16 | 同上；空 Tensor 跳过 |
| R-003 BF16 | schMode 2 | 同上但 BF16 | 单 SIMT Kernel | `Process<bfloat16_t>`→`OpForeachTanBf16Simt` | 每线程寄存器中 BF16→FP32→BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 上限 | `0<=tensorNum<=256` | POD 的 `tensorElements[256]` 固定容量 | 全部 | `GRAPH_FAILED` | `ForeachTanTilingFunc`/`FOREACH_TAN_MAX_TENSOR_NUM` |
| 最少每核 1024 元素 | `perCore=max(1024,ceil(M/C))` | 避免按最大 Tensor 为很小工作量启动过多 block | 全部 | 自动减少 blockDim，最少 1 | `MIN_PER_CORE_ELEMENTS` |
| 32 元素对齐 | `perCore=CeilAlign(perCore,32)` | block 数按对齐工作粒度反推 | 全部 | 合并到相邻较少 block | `ALIGN_SIZE`/`ForeachTanTilingFunc` |
| 512 threads | 编译常量 `THREAD_NUM=512` 与 `LAUNCH_BOUND(512)` | VF thread 映射和 grid-stride 以该值为块内宽度 | 全部 | 固定源码无其它线程数 Route | `foreach_tan_simt.h` |
| 空 Tensor | `count==0` | 避免取数据指针并调用 VF | 全部 | `continue` 到下一成员 | `NsForeachTan::Process` |
| UB 减 128KiB | `SetLocalMemorySize(U-DCACHE_SIZE)`，且先检查 U>0 | 为 DCache 保留 128KiB | 全部 | 源码未显式检查 `U>=128KiB` 的无符号下溢 | `GetPlatformInfo`/`ForeachTanTilingFunc` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachTanTilingData` | `tensorNum` | `int32_t`/32 | offset 0；结构 align 8 | Tensor 数 | `tiling->tensorNum=tensorNum` | `Process` 的成员循环上界 | [0,256] |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | block 数 | `tiling->needCoreNum=needCoreNum` | body 不读取；launch 另由 `SetBlockDim` 消费同值 | ≥1 且≤平台 AIV 核数 |
| 同上 | `tensorElements[256]` | `int64_t[256]` | offset 8；结构总 `sizeof=2056`，align 8 | 元素 | 每项 `shape.GetShapeSize()` | `Process`: `count=tensorElements[i]`；VF 循环上界 | inactive 槽 memset 为 0 且不读取 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | `foreach_tan_tiling_arch35.cpp::ForeachTanTilingFunc` | `ForeachTanTilingData` | `foreach_tan<0>` | `Process<float>`/`OpForeachTanSimt<float>` | `op_kernel/CMakeLists.txt` ascend950 + binary config FP32 |
| R-002 FP16 | 1 | 同上 | 同上 | `foreach_tan<1>` | `Process<half>`/`OpForeachTanHalfSimt` | 同上 FP16 binary |
| R-003 BF16 | 2 | 同上 | 同上 | `foreach_tan<2>` | `Process<bfloat16_t>`/`OpForeachTanBf16Simt` | 同上 BF16 binary |

## Kernel 实现

### 公共执行流程

模板 entry 解析 POD，编译期 `schMode` 选择 `Process<T>`。每个 block 都构造 x/y `ListTensorDesc`，按 `tensorNum` 顺序遍历列表；空项跳过，非空项解析 T 类型的 GM 指针并调用 512-thread `VF_CALL`。VF 内各线程使用全 grid 的 block/thread 索引形成唯一首元素与固定步长，直接 load、计算、store。没有显式 Queue、shared memory、barrier、event 或 atomic，也没有跨 Kernel workspace 交接。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachTan.md` | 两段式接口 | FP32 且公开约束成立 | executor |
| Host/Tiling | `op_host/arch35/foreach_tan_tiling_arch35.cpp` | `ForeachTanTilingFunc` | `xDtype==DT_FLOAT` | `GET_TPL_TILING_KEY(0)` |
| Router | `op_kernel/arch35/foreach_tan.cpp` | `foreach_tan<schMode>` | `schMode==0` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachTanTilingFunc -> foreach_tan<0>(x,y,workspace,tiling)
  -> NsForeachTan::Process<float>
    -> for each non-empty tensor: VF_CALL<OpForeachTanSimt<float>>(Dim3(512),xPtr,yPtr,count)
      -> per thread grid-stride loop -> tanf -> GM store
```

##### 源码等价伪码

```text
# 源码等价伪码
for tensor i in [0,tensorNum):
  count=tensorElements[i]; if count==0: continue
  xPtr=xList.GetDataPtr<float>(i); yPtr=yList.GetDataPtr<float>(i)
  parallel 512 threads per block:
    idx=blockIdx*threadNum+threadIdx
    while idx<count:
      yPtr[idx]=tanf(xPtr[idx])
      idx += threadNum*blockNum
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `xList.GetDataPtr<float>(i)+idx` | count | FP32/ND | caller/framework | thread load | caller/framework，stream 完成前 | 单元素访问；idx<count |
| thread registers | thread-local `val` | 1 FP32 | FP32 | GM load | `tanf` | 当前迭代 | 无跨线程共享 |
| y GM | `yList.GetDataPtr<float>(i)+idx` | count | FP32/ND | `tanf` store | caller | caller | 每 idx 唯一写者 |
| tools workspace | 固定仓未见 body offset | 16 MiB | opaque | ACLNN caller | AscendC tools（具体 consumer 不可见） | stream | Kernel source 不显式读取参数 |

##### 分层内存布局

N/A：该 SIMT body 直接 GM load/store，仅使用线程局部标量；未声明 dynamic/shared UB、Queue 或 TBuf。Host 仍设置 `localMemorySize=U-128KiB`，其内部布局固定仓不可见。

##### 字节账与复用顺序

```text
UB_total = N/A（无显式 AIV Queue/TBuf）
dynamic_ub = 固定仓未见 VF_CALL 的 dynUB 参数；Host localMemorySize=U-131072 bytes
workspace_total = 16777216 bytes（tools workspace，body 参数未读）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `val` | 当前 idx 的 GM load | 当前 `tanf` | 下一 grid-stride idx | 单线程程序顺序 | store 完成计算后才迭代 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT 迭代 | grid-stride + `idx<count` | thread index | load/store | GM 有效域 | FP32 `tanf`；无 padding lane |
| 跨线程 | N/A | 每线程 | N/A | 不同 idx | 无共享写、无需 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_tan_tiling_arch35.cpp` | `ForeachTanTilingFunc` | grid、key、POD、workspace/local memory |
| Kernel entry/router | `op_kernel/arch35/foreach_tan.cpp` | `foreach_tan<0>` | FP32 body mapping |
| Device body/helper | `op_kernel/arch35/foreach_tan_simt.h` | `Process<float>`/`OpForeachTanSimt<float>` | 512-thread grid-stride `tanf` |

### Route R-002：FP16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachTan.md` | 第一段接口 | FP16 | executor |
| Host/Tiling | `ForeachTanTilingFunc` | dtype branch | `DT_FLOAT16` | schMode 1 |
| Router | `foreach_tan<schMode>` | compile-time branch | `schMode==1` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachTanTilingFunc -> foreach_tan<1>
  -> Process<half> -> VF_CALL<OpForeachTanHalfSimt>(Dim3(512),...)
    -> __half2float -> tanf -> __float2half -> GM store
```

##### 源码等价伪码

```text
# 源码等价伪码
for each non-empty tensor and each thread's idx<count:
  half val=xPtr[idx]
  float val_f32=__half2float(val)
  float result_f32=tanf(val_f32)
  yPtr[idx]=__float2half(result_f32)
  idx += 512*blockNum
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | list ptr + idx | count | FP16/ND | caller/thread | thread/caller | stream | idx guard |
| thread registers | `val`, `val_f32`, `result_f32` | 1 FP16 + 2 FP32 标量 | scalar | load/conversion/tanf | next step/store | 当前迭代 | 固定仓不提供寄存器字节分配 |
| tools workspace | opaque | 16 MiB | opaque | caller | tools boundary | stream | body 不读 |

##### 分层内存布局

N/A：无显式 UB/shared buffer；数值中间量是线程局部标量，Host local memory 的实际使用固定仓不可见。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=固定仓不可见
workspace_total=16777216 bytes
thread-local lifetime: half load -> FP32 input -> FP32 result -> half store
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local scalars | 当前 idx | convert/tanf/store | 下一 idx | 单线程程序顺序 | 无跨线程别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| conversion | `__half2float` | FP16 load | `tanf` | local scalar | FP32 计算输入 |
| down conversion | `__float2half` | FP32 `tanf` | GM store | local scalar | 固定源码未标显式 rounding 参数 |
| tail | `idx<count` | thread index | memory ops | GM bounds | 无无效 lane 写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tan_tiling_arch35.cpp` | FP16 branch | schMode 1 |
| Kernel entry/router | `foreach_tan.cpp` | `foreach_tan<1>` | `Process<half>` |
| Device body/helper | `foreach_tan_simt.h` | `OpForeachTanHalfSimt` | FP16→FP32→FP16 顺序与 grid-stride |

### Route R-003：BF16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachTan.md` | 第一段接口 | BF16 | executor |
| Host/Tiling | `ForeachTanTilingFunc` | dtype branch | `DT_BF16` | schMode 2 |
| Router | `foreach_tan<schMode>` | compile-time branch | `schMode==2` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachTanTilingFunc -> foreach_tan<2>
  -> Process<bfloat16_t> -> VF_CALL<OpForeachTanBf16Simt>(Dim3(512),...)
    -> __bfloat162float -> tanf -> __float2bfloat16 -> GM store
```

##### 源码等价伪码

```text
# 源码等价伪码
for each non-empty tensor and each thread's idx<count:
  bfloat16 val=xPtr[idx]
  float val_f32=__bfloat162float(val)
  float result_f32=tanf(val_f32)
  yPtr[idx]=__float2bfloat16(result_f32)
  idx += 512*blockNum
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | list ptr + idx | count | BF16/ND | caller/thread | thread/caller | stream | idx guard |
| thread registers | BF16 `val` + two FP32 intermediates | 单元素 | scalar | GM/conversion/tanf | conversion/store | 当前迭代 | 无共享 |
| tools workspace | opaque | 16 MiB | opaque | caller | tools boundary | stream | body 不读 |

##### 分层内存布局

N/A：无显式 AIV UB、dynamic shared memory、Queue 或 TBuf；仅线程局部标量。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=固定仓不可见
workspace_total=16777216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local scalars | 当前 idx | convert/tanf/store | 下一 idx | 程序顺序 | 每线程独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| conversion | `__bfloat162float` | BF16 load | `tanf` | local scalar | FP32 中间计算 |
| down conversion | `__float2bfloat16` | FP32 result | GM store | local scalar | rounding 细节在 intrinsic 边界不可见 |
| tail/empty | `idx<count` 与 `count==0 continue` | POD | VF/memory | GM bounds/list ptr | 空项不取指针，尾元素逐个保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tan_tiling_arch35.cpp` | BF16 branch | schMode 2 |
| Kernel entry/router | `foreach_tan.cpp` | `foreach_tan<2>` | BF16 Process |
| Device body/helper | `foreach_tan_simt.h` | `OpForeachTanBf16Simt` | BF16→FP32→BF16 与 bounds |

## 算子特定优化

### Trick T-001：一个 grid 复用到整个 TensorList 的逐项 grid-stride 执行

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：`tensorNum<=256`，Host 根据最大成员产生一个 `needCoreNum`，Kernel 对每个非空成员使用相同 block grid 和 512-thread VF。
- Golden 来源：[`foreach_tan_simt.h:89-110`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/arch35/foreach_tan_simt.h#L89-L110)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t i = 0; i < tilingData->tensorNum; i++) {
    int64_t count = tilingData->tensorElements[i];
    if (count == 0) {
        continue;
    }
    __gm__ T* xPtr = xList.GetDataPtr<T>(i);
    __gm__ T* yPtr = yList.GetDataPtr<T>(i);
```

#### 机制与设计原因

Host 只生成一次 launch grid，并用最大 Tensor 控制 blockDim；Device 在同一 entry 中遍历列表，空项不解析数据指针，非空项通过 VF_CALL 让所有 block/thread 以 `idx += threadNum*blockNum` 覆盖该成员。相比把每个列表成员物化为独立 Host launch，这一实现从源码结构上复用同一启动节点和 POD，且无需列表中间 workspace；没有真机数据支撑具体性能幅度。

#### 迁移边界

- 适用：列表成员共享 dtype/逐元素语义、彼此无依赖且可共用 grid 的 SIMT 一元算子。
- 前提：POD 能容纳全部成员元素数；每个线程写唯一 idx；小成员可接受由最大成员决定的 grid。
- 失效：超过 256 项时 Host 失败；需要成员间同步、不同 dtype 或 per-member workspace 时必须改设计。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | Host dtype 决策树 |
| 模板与 dtype | `Process<float>` | `Process<half>` | `Process<bfloat16_t>` | binary config 三 dtype |
| 循环与分核 | 512-thread grid-stride | 同左 | 同左 | grid 由最大 Tensor 决定 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | FP32 local scalar | FP16 + FP32 locals | BF16 + FP32 locals | 均无显式 Queue/shared UB |
| 内存总占用与复用 | 无显式 UB | 无显式 UB | 无显式 UB | tools workspace 均 16 MiB |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM load/store + `tanf` | 加 half conversions | 加 bf16 conversions | 均为 SIMT，无 NDDMA/通信 |
| 同步、精度与 tail | FP32，无 barrier，idx guard | FP32 tan 后转 half | FP32 tan 后转 bf16 | 每 idx 唯一写者 |
| fallback/失败边界 | 未知 schMode 无 body | 同左 | 同左 | 不支持 dtype Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持 |
| Host/Launcher | arch22 公共 Tiling/POD | arch22 公共 Tiling/POD | arch35 专用最大 Tensor 分核与 256 项 POD |
| Kernel/资源 | AIV Queue + AscendC Tan | 同 A2 | SIMT 512 threads，直接 GM，16 MiB tools workspace |

## 限制与失败边界

- 950 公开支持 FP16/FP32/BF16、ND、维数≤8；输出必须连续且容量不少于输入。非连续输入的实际 Contiguous 节点、临时 workspace 与失败传播在生成 ACLNN bridge 中，固定仓不可见。
- TensorList 显式上限 256；`tensorElements` 与 `maxTensorElements` 是 `int64_t`，但 shape size 的溢出检查固定仓未见。
- `SetLocalMemorySize(static_cast<uint32_t>(ubSize-128KiB))` 只检查 `ubSize!=0`，未显式检查 `ubSize>=128KiB`；正常 950 平台是否由框架保证该前提固定仓不可见。
- `VF_CALL` 的内部 cooperative group、warp 映射、dynUB 使用与失败语义，以及 `tanf`/低精度 conversion intrinsic 的微实现和 rounding 细节固定仓不可见；可见 body 没有 barrier、atomic 或 shared buffer。
- workspace 固定为 16 MiB，但 entry/body 不读取参数；tools 的真实消费 offset 与生命周期细节固定仓不可见。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_tan/docs/aclnnForeachTan.md` | API 原型/产品表 | 公开语义、950 支持、错误和值域 |
| 入口实现 | `foreach/foreach_tan/op_host/foreach_tan_def.cpp` | `ForeachTan` | 950 dtype/layout/动态 shape config |
| Host/Tiling/Launcher | `foreach/foreach_tan/op_host/arch35/foreach_tan_tiling_arch35.cpp` | `ForeachTanTilingFunc` | 校验、分核、workspace、POD、key |
| Host/Tiling/Launcher | `foreach/foreach_tan/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`/`SUPPORT_TILING_DIR` | ascend950→arch35 Host 接线 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `IMPL_OP_INFERSHAPE(ForeachTan)` | 公共图侧 shape/dtype 推导注册；ACLNN 使用显式 out |
| Host/Tiling/Launcher | `foreach/foreach_tan/op_kernel/arch35/foreach_tan_tiling_data.h` | `ForeachTanTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_tan/op_kernel/arch35/foreach_tan_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 实例集合 |
| Kernel entry/body | `foreach/foreach_tan/op_kernel/arch35/foreach_tan.cpp` | `foreach_tan<schMode>` | 编译期 router |
| Kernel entry/body | `foreach/foreach_tan/op_kernel/arch35/foreach_tan_simt.h` | `Process`/三个 VF | ListTensor、线程映射、计算、tail |
| 构建接线 | `foreach/foreach_tan/op_host/config/ascend950/foreach_tan_binary.json` | `op_list` | 三 dtype binary |
| 构建接线 | `foreach/foreach_tan/op_kernel/CMakeLists.txt` | ascend950 `add_kernel_sources` | arch35 entry active 接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | `tensorNum/needCoreNum/tensorElements[256]` | 字段顺序、宽度、256 上限；inactive 槽不可读取 | `ForeachTanTilingData` |
| 调用顺序 | 单 entry→逐 Tensor→VF grid-stride | 空项先跳过；ListTensor dtype 与模板一致 | `foreach_tan<schMode>`/`Process<T>` |
| workspace/内存 | 16 MiB tools workspace；无显式 UB | localMemorySize 保留 128KiB DCache；不要臆造 body buffer | `ForeachTanTilingFunc` |
| 同步 | 每 idx 唯一线程、无共享依赖 | 引入共享/atomic 后不能沿用“无需 barrier”结论 | 三个 VF body |
| 精度与尾块 | FP32 `tanf`；FP16/BF16 升精度 | conversion 顺序与 `idx<count` guard 不得改变 | `OpForeachTan*Simt` |
| 失败边界 | dtype/平台/POD 检查失败不 launch | >256、非法 dtype、空平台信息必须保持失败关闭 | `ForeachTanTilingFunc` |

- [ListTensorDesc](../../../../apis/simd_api/basic_api/aux_data_structures/list_tensor_desc.md)
- [__bfloat162float](../../../../apis/simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162float.md)
- [__float2bfloat16](../../../../apis/simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16.md)
- [__float2half](../../../../apis/simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half.md)
- [__half2float](../../../../apis/simt_api/math_functions/half_type/half_precision_conversion_functions/_half2float.md)
- [数学函数tanf](../../../../apis/simt_api/math_functions/float_math_functions/tanf.md)
