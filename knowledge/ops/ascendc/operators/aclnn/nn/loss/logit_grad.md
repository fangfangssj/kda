---
type: Operator
title: "LogitGrad A2/A3 ACLNN 算子知识"
description: "说明 aclnnLogitGrad 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/docs/aclnnLogitGrad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_graph/logit_grad_proto.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/config/ascend910b/logit_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/config/ascend910_93/logit_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
created_at: "2026-08-25T17:14:22Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# LogitGrad A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnLogitGradGetWorkspaceSize` / `aclnnLogitGrad`；固定仓只公开这一版本族。 |
| 计算语义 | 对每个元素先以 FP32 计算 `q = dy / (x * (1 - x))`；`eps < 0` 时只在 `0 <= x <= 1` 选择 `q`，否则选择 NaN；`eps >= 0` 时只在 `eps <= x <= 1-eps` 选择 `q`，否则选择 0。 |
| 输入输出 | `x`、`dy`、`out(dx)` 均为 ND，shape 相同，支持 0D～8D、空 Tensor，以及 BF16/FP16/FP32；公开接口支持非连续 Tensor。 |
| 执行类型 | 单个 `AIV_VECTOR` Kernel 节点；`key=1/2/3` 分别实例化 `LogitGradND<half/float/bfloat16_t>`，无可达 Cube/AIC 或 `MIX_CV` 节点。 |
| Kernel 路线 | R-001 FP16 升精度路线；R-002 FP32 原生路线；R-003 BF16 升精度路线。 |
| 核心资源 | 输入/输出 GM；每核一个 192 KiB `VECCALC TBuf`，物理上分成两个 96 KiB ping-pong 半区；无 `TQue`，Kernel 不读取 workspace。 |
| 来源闭包 | `source_closed`：公开 API 文档、OpDef 生成入口接线、InferShape、Tiling/POD、三条 key router、实际 Vector body、平台注册和 CMake 构建均已闭合；自动生成 ACLNN bridge 是明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `N = numel(x)`，对 `i in [0,N)`：

```text
denom_i = x_i * (1 - x_i)
q_i     = dy_i / denom_i
if eps < 0:
    dx_i = q_i  when 0 <= x_i <= 1, else NaN
else:
    dx_i = q_i  when eps <= x_i <= 1-eps, else 0
```

Device 对三种输入 dtype 都用 FP32 执行 `1-x`、乘、除、双边比较和选择；FP16/BF16 输入先升为 FP32，结果再降回原 dtype。公开文档另称 `eps > 1` 输出 NaN、`eps == 1` 输出 Inf，但固定 Kernel 的可见谓词是同时满足 `x >= eps && x <= 1-eps`，不满足即在 `eps >= 0` 分支选择 0；该边界存在文档与实现不一致，复用时应以目标 CANN 的实测契约确认，本文不推测消解。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 0D～8D，ND | BF16、FP16、FP32 | 概率域输入；支持空 Tensor 和非连续 Tensor，OpDef 标记 `AutoContiguous()`。 |
| `dy` | 与 `x` 相同，ND | 与 `x` 相同 | 上游梯度；支持空 Tensor 和非连续 Tensor。 |
| `eps` | 标量属性 | ACLNN 参数为 `double`；OpDef/Tiling/POD 为 `float` | 默认/建议值 `-1.0`；由生成式 API bridge 到 float 属性的具体窄化语句固定仓不可见。 |
| `out` / `dx` | 与 `x` 相同，ND | 与 `x` 相同 | 输出梯度；支持非连续 Tensor，OpDef 标记 `AutoContiguous()`。 |

### 入口与版本

固定仓公开 `aclnnLogitGradGetWorkspaceSize(x, dy, eps, out, workspaceSize, executor)` 与一次性执行接口 `aclnnLogitGrad(workspace, workspaceSize, executor, stream)`，没有 `V2` 或其它公开 LogitGrad 版本。第一段完成公开参数检查并建立 executor，第二段在给定 stream 上执行。仓内没有这两个符号的手写 `.cpp/.h`；`op_host/CMakeLists.txt` 以 `OPTYPE logit_grad ACLNNTYPE aclnn` 把 `logit_grad_def.cpp` 接入 ACLNN OpDef 目标，`cmake/func.cmake::add_modules_sources` 再将 `*_def*.cpp` 加入 `${OPHOST_NAME}_opdef_aclnn_obj`，因此生成式 inner bridge、executor 壳与 Contiguous/ViewCopy 属于明确框架边界，而不是补写的手工入口。

公开 README/API 产品表均明确支持 Atlas A2 与 A3；`LogitGrad` OpDef 注册 `ascend910b`、`ascend910_93`，两个 binary config 都包含 BF16/FP16/FP32，Kernel CMake 也对这两个 compute unit 启用同一源码，因此平台证据无冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

公开契约要求 `x/dy/out` 非空，三者 dtype、format、shape 一致且 dtype 属于 BF16/FP16/FP32；文档明确空 Tensor 合法。固定仓不可见生成 bridge 中逐项检查的精确调用顺序和返回宏，只能确认文档列出空指针错误 `161001` 与非法 dtype 错误 `161002`。OpDef 对三路 Tensor 都声明 ND 和 `AutoContiguous()`；非连续输入/输出的具体 Contiguous/ViewCopy 节点、其临时 workspace 与失败传播由框架生成，固定仓不可见。

图/算子 Host 链路中，`InferShape4LogitGrad` 将 `x` 的完整 shape 复制给 `dx`，`InferDataType4LogitGrad` 将 `x` dtype 复制给 `dx`。Tiling 随后按顺序检查 input tensor、attrs、input desc；任一为空返回 `GRAPH_FAILED`，不支持 dtype 也返回失败。对 `GetInputShape(0)`、`GetFloat(0)`、workspace 数组与 raw tiling buffer 没有额外显式空检查。

### 参数、Workspace 与生命周期

`RunBigKernelTiling` 读取 origin shape 的 `GetShapeSize()` 得到元素数 `N`，读取 float 属性 `epsilon`，读取平台 AIV 核数 `C = GetCoreNumAiv()`，计算：

```text
needCoreNum = min(max(ceil(N / 8192), 1), C)
workspace[0] = 16 * 1024 * 1024 bytes
blockDim = needCoreNum
```

外部调用者按第一段返回的总 workspace 大小持有 Device 内存，executor 在第二段执行后不可重复使用。算子 Tiling 固定声明 16 MiB workspace；但 `logit_grad` entry 将 `userWs` 设为 `nullptr`，`LogitGradND::Init` 的 workspace 形参也未读取，因此这 16 MiB 不形成 Kernel 内部数据边。框架为非连续 Tensor 引入的额外 workspace 切片/offset/owner 固定仓不可见，不能与该 16 MiB 做源码外合并。

### 启动图与失败传播

```text
aclnnLogitGradGetWorkspaceSize                     # 生成 bridge，固定仓不可见
  -> LogitGrad OpDef / AutoContiguous              # 可能产生框架临时张量
  -> InferShape4LogitGrad / InferDataType4LogitGrad
  -> TilingLogitGradTiling -> RunBigKernelTiling
  -> executor
aclnnLogitGrad(workspace, workspaceSize, executor, stream)
  -> [K-001] logit_grad<<<blockDim=needCoreNum, stream>>>(x, dy, dx, workspace, tiling)
       -> exactly one LogitGradND<T>::Process()
```

| 节点/边 | entry/介质 | 参数与范围 | ordering、owner/lifetime 与失败传播 |
|---|---|---|---|
| Host→K-001 | `logit_grad` | `x,dy,dx,workspace,tiling`；`blockDim=needCoreNum`；POD 为 `LogitGradTilingData` | 生成 executor 在用户 stream 上提交；具体 launcher 符号固定仓不可见。第一段失败则没有可执行 executor；第二段 launch 失败返回 ACLNN 状态。 |
| K-001 输入/输出 | 用户或框架 GM Tensor | 每路逻辑 extent 为 `N * sizeof(T)`，ND contiguous 视图 | 单 Kernel，无跨 Kernel workspace 边；输入在 Kernel 执行期间只读，`dx` 由 Kernel 写。 |
| workspace | caller-owned GM | Tiling 声明 16 MiB | 生存期覆盖第二段调用；Kernel body 不解引用，固定仓无 producer→consumer 数据交接。 |

## Route 与参数设计

### 分核与资源计算

Host 以 8192 元素为基本 tile，使 `needCoreNum` 至多为平台 AIV 核数且空域仍启动 1 核。Device 再令 `totalTimes=ceil(N/8192)`、`base=floor(totalTimes/needCoreNum)`、`loopRemain=totalTimes mod needCoreNum`；前 `loopRemain` 个核处理 `base+1` 个 tile，其余核处理 `base` 个 tile。对前组，起点是 `(base+1)*blockIdx*8192`；对后组，源码等价式为 `base*blockIdx*8192 + N mod (8192*needCoreNum)`。若 `remain=N mod 8192` 非零，尾 tile 由 `lastCoreNum=(loopRemain==0 ? needCoreNum-1 : loopRemain-1)` 处理，`calNum=remain`；空域 `N=0` 时唯一核的循环次数为 0。

每核固定分配 192 KiB UB，不由 Host UB 查询动态决定。每个 ping-pong 半区 96 KiB，恰容纳两个 32 KiB FP32 数据区和一个 32 KiB scratch/mask 复用区；因此源码必要条件是平台可提供至少 192 KiB UB，固定 Host selector 未显式查询或拒绝更小 UB。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 升精度 | `key=1` / `LogitGradND<half>` | `dataType == ge::DT_FLOAT16` | 否则检查 FP32。 |
| 2 | R-002 FP32 原生 | `key=2` / `LogitGradND<float>` | 前项未命中且 `dataType == ge::DT_FLOAT` | 否则检查 BF16。 |
| 3 | R-003 BF16 升精度 | `key=3` / `LogitGradND<bfloat16_t>` | 前两项未命中且 `dataType == ge::DT_BF16` | 其它 dtype：Tiling 返回 `GRAPH_FAILED`。 |

```text
if dtype == FLOAT16: key=1
else if dtype == FLOAT: key=2
else if dtype == BF16: key=3
else: GRAPH_FAILED
SetTilingKey(key)
router under (__CCE_AICORE__ == 220 || __CCE_AICORE__ == 310):
    key 1 -> LogitGradND<half>
    key 2 -> LogitGradND<float>
    key 3 -> LogitGradND<bfloat16_t>
    unknown key -> no body/no output in this entry
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 升精度 | 1 | FP16、ND、公开 shape 0D～8D；A2/A3 config 可达 | 单节点 K-001 `logit_grad` | `LogitGradND<half>::Init/Process/CopyInAndCast/ComputeStepOne/ComputeStepTwo/CastAndCopyOut` | 192 KiB TBuf；FP16 DMA 到半区后 Cast FP32；EVENT0/1 ping-pong | `calNum` 控制字节 tail；无其它 fallback。 |
| R-002 FP32 原生 | 2 | FP32、ND、公开 shape 0D～8D；A2/A3 config 可达 | 同上 | `LogitGradND<float>` 同一方法集 | 192 KiB TBuf；DMA 直接进入 FP32 view，无输入/输出 Cast | 同上。 |
| R-003 BF16 升精度 | 3 | BF16、ND、公开 shape 0D～8D；A2/A3 config 可达 | 同上 | `LogitGradND<bfloat16_t>` 同一方法集 | 与 FP16 同布局；输出以 `CAST_RINT` 降精度 | 同上。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 固定 tile | `MAX_ELEMENT_NUM_EACH_CORE == PP_ELEMENT_NUM == 8192` | 每个 FP32 逻辑数据区恰为 32 KiB；96 KiB 半区承载 `x/dy/scratch` | 全部 | 固定仓没有更小 tile Route。 | `RunBigKernelTiling`、`LogitGradND::Process` |
| 固定 UB | `InitBuffer(ubTBuf, 192*1024)` | 两个 96 KiB ping-pong 半区 | 全部 | 无 Host fallback；平台 UB 不足时的行为固定仓不可见。 | `LogitGradND::Init` |
| compare 对齐 | `tmpDataCount=ceil(calNum/64)*64` | FP32 compare 以 64 元素 repeat 对齐；Select 只提交 `calNum` | 全部 | 尾部用向上取整 mask，不切 Route。 | `ComputeStepTwo` |
| DMA 长度窄化 | `blockLen=uint32_t(calNum*sizeof(T))` | `DataCopyExtParams.blockLen` 为 32 位字节数；这里上界仅 32768 B | 全部 | 由固定 tile 保证，无 fallback。 | `CopyInAndCast`、`CastAndCopyOut` |
| 编译架构 | `__CCE_AICORE__ == 220 \|\| == 310` | router 只在两种 AICore 编译值下实例化 Vector body | 全部 | 其它值 entry 为空。 | `logit_grad` |
| 空域 | `N==0 -> needCoreNum=1,totalTimes=0` | 仍满足 blockDim 非零，Process 只建立/回收 event | 全部 | 正常无输出结束。 | `GetNeedCoreNum`、`Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LogitGradTilingData` | `elementNum` | `int64_t` / 64 bit | 宏生成 `sizeof/alignof/offsetof` 固定仓不可见；声明顺序第 1 | 元素 | `RunBigKernelTiling: set_elementNum(inputShape.GetShapeSize())` | `Init -> elementNum`；`Process` 计算 tile/core/offset/tail | 允许 0；shape 乘积溢出策略固定仓不可见。 |
| `LogitGradTilingData` | `eps` | `float` / 32 bit | 宏生成布局固定仓不可见；声明顺序第 2 | 数值标量 | `RunBigKernelTiling: set_eps(*(attrs->GetFloat(0)))` | `Init -> eps/epslion/selectValue` | ACLNN `double` 到该 float 的桥接窄化代码固定仓不可见。 |
| `LogitGradTilingData` | `needCoreNum` | `uint64_t` / 64 bit | 宏生成布局固定仓不可见；声明顺序第 3 | AIV core 数 | `set_needCoreNum(min(max(ceil(N/8192),1),GetCoreNumAiv()))` | `Init -> needCoreNumber`；`Process` 分核 | 正常至少 1；若平台报告 0，源码会返回 0 并形成 `blockDim=0`，无显式保护。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | `logit_grad_tiling.cpp::RunBigKernelTiling` | `LogitGradTilingData` | `logit_grad.cpp::logit_grad -> LogitGradND<half>` | `logit_grad.h` 全部五个方法 | OpDef `ascend910b/ascend910_93`；Kernel CMake 同名 compute units；两个 binary JSON 的 FP16 项。 |
| R-002 | key 2 | 同上 | 同上 | `logit_grad -> LogitGradND<float>` | 同上 | 两个 binary JSON 的 FP32 项。 |
| R-003 | key 3 | 同上 | 同上 | `logit_grad -> LogitGradND<bfloat16_t>` | 同上 | 两个 binary JSON 的 BF16 项。 |

## Kernel 实现

### 公共执行流程

K-001 是唯一 Device 节点。entry 用 `GET_TILING_DATA` 解析 POD，workspace 被替换为未使用的 `nullptr`，然后按 key 构造一个 `LogitGradND<T>`。每核 `Init` 绑定三路 GM、将 eps 归一为比较上下界与 false 分支常量、分配 192 KiB TBuf；`Process` 计算本核 tile 范围，对两个 96 KiB 半区交替执行 `CopyInAndCast → ComputeStepOne → ComputeStepTwo → CastAndCopyOut`。没有跨核归约、Atomic、workspace 边或第二 Kernel。

### Route R-001：FP16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `logit_grad_def.cpp` | `ops::LogitGrad` | FP16 是第 2 个允许 dtype；ND；AutoContiguous | 生成 ACLNN OpDef 路径。 |
| Tiling | `logit_grad_tiling.cpp` | `RunBigKernelTiling` | `dataType == ge::DT_FLOAT16` | `tilingKey=1`。 |
| Router | `logit_grad.cpp` | `logit_grad` | 架构 guard 命中且 `TILING_KEY_IS(1)` | `LogitGradND<half>`。 |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> logit_grad(x,dy,dx,workspace,tiling)
  -> LogitGradND<half>::Init
  -> LogitGradND<half>::Process
     -> CopyInAndCast       # FP16 GM -> UB tail views -> FP32 views
     -> ComputeStepOne      # 1-x, x*(1-x), dy/denom
     -> ComputeStepTwo      # GE/LE masks, And, Select
     -> CastAndCopyOut      # FP32 -> FP16(CAST_NONE) -> GM
```

##### 源码等价伪码

```text
# 伪码；baseOffset/loopNum 按“分核与资源计算”
init EVENT0, EVENT1 as MTE3_MTE2-ready
for i in [0, loopNum):
    half = i & 1; event = half ? EVENT1 : EVENT0
    count = (this is global last partial tile) ? N % 8192 : 8192
    off = coreStart + i*8192
    wait MTE3_MTE2(event)
    DataCopyPad(xTmp@half+16KiB, xGM+off, {1,count*2,0,0,0}, noPad)
    DataCopyPad(dyTmp@half+48KiB,dyGM+off,{1,count*2,0,0,0}, noPad)
    signal+wait MTE2_V(event)
    Cast(xFp32@half+0, xTmp, CAST_NONE, count); V barrier
    Cast(dyFp32@half+32KiB,dyTmp,CAST_NONE,count); V barrier
    tmp = -x; tmp += 1; tmp *= x; dyFp32 /= tmp   # barrier after each
    rounded = ceil(count/64)*64
    m0 = (xFp32 >= epslion, rounded); m1 = (xFp32 <= 1-epslion, rounded)
    m0 &= m1; Select(xFp32,m0,dyFp32,selectValue,count)
    Cast(xHalf@half+0,xFp32,CAST_NONE,count); signal+wait V_MTE3(event)
    DataCopyPad(dxGM+off,xHalf,{1,count*2,0,0,0}); signal MTE3_MTE2(event)
wait EVENT0 and EVENT1 MTE3_MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/dy/dx` | `coreStart+i*8192` | 有效 `count`，各 `count*2` B | FP16/ND | caller/framework、K-001 | K-001、caller | 调用期间 | `DataCopyPad` 单 block，stride 0。 |
| `ubTBuf` | 0 | 192 KiB | raw bytes | `InitBuffer` | 全部方法 | 每核整个 Kernel | 两个 96 KiB 半区。 |
| 半区 x | `halfBase+[0,32KiB)` | FP32 8192；原始 FP16 view 位于 `[16,32)KiB` | alias FP32/FP16 | DMA→Cast | Compute/输出 Cast | 本 tile | 扩展 Cast 的 src/dst 同属该区。 |
| 半区 dy | `halfBase+[32,64)KiB` | FP32 8192；原始 FP16 view 位于 `[48,64)KiB` | alias FP32/FP16 | DMA→Cast | Div | 本 tile | 同上。 |
| 半区 scratch/mask | `halfBase+[64,96)KiB`；mask 起点 64/80 KiB | FP32 scratch 32 KiB；两个 mask view 在其内 | FP32/uint8/uint16 alias | StepOne 后由 StepTwo 覆盖 | StepOne/StepTwo/Select | 本 tile | 生命周期错开，不额外分配。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  h0x["half0 x/fp16-tail alias · 32KiB"]:1
  h0dy["half0 dy/fp16-tail alias · 32KiB"]:1
  h0s["half0 scratch/masks alias · 32KiB"]:1
  h1x["half1 x/fp16-tail alias · 32KiB"]:1
  h1dy["half1 dy/fp16-tail alias · 32KiB"]:1
  h1s["half1 scratch/masks alias · 32KiB"]:1
```

##### 字节账与复用顺序

```text
one_half = 32KiB(x FP32) + 32KiB(dy FP32) + 32KiB(scratch/masks alias) = 96KiB
UB_total = 2 * one_half = 192KiB
UB_total <= 平台可用 UB                         # Host 未显式查询，属必要条件
dynamic_ub = N/A（AIV TBuf 固定分配）
workspace_total = 16MiB（Tiling 声明，Kernel 未读取）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 当前半区 x/dy | MTE2 DMA、Cast | Vector Compute、MTE3 output | 相隔一次迭代后复用 | 同 event 的 `MTE3_MTE2` | 对应输出 DMA 已完成。 |
| `[64,96)KiB` scratch/masks | StepOne FP32 scratch | Div 完成后不再读 scratch | StepTwo 写 compare masks | `PIPE_V` barrier | StepOne 最后一条 Div 已消费 denominator。 |
| x FP32 区 | 输入 Cast | 比较、Mul、Select | 输出 Cast 原地缩窄 | 每个 Vector op 后 `PIPE_V` | Select 已完成后才能缩窄。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 半区复用 | `MTE3_MTE2 EVENT0/1` | 上次 CopyOut | 下次 CopyIn | 同一 96 KiB 半区 | 初始预置 flag；末尾等待两个 event。 |
| DMA→Vector | `MTE2_V` | 两路 `DataCopyPad` | 输入 Cast | 原始 FP16 views | `blockLen=count*2` B。 |
| Vector 链 | `PipeBarrier<PIPE_V>` | 每条 Cast/Muls/Adds/Mul/Div/Compare/And/Select | 下一条 Vector 指令 | alias buffer 和数值依赖 | 全部 FP32；compare 长度向上对齐 64，Select 仅 `count`。 |
| Vector→DMA | `V_MTE3` | 输出 Cast | `DataCopyPad` | FP16 输出 view | FP32→FP16 为 `CAST_NONE`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key 1、core、workspace、POD、blockDim。 |
| Kernel entry/router | `loss/logit_grad/op_kernel/logit_grad.cpp` | `logit_grad` / `LogitGradND<half>` | key 1 到 body 映射，workspace 不传入 body。 |
| Device body/helper | `loss/logit_grad/op_kernel/logit_grad.h` | `LogitGradND<half>` 全部方法 | FP16 DMA、FP32 升精度、ping-pong、mask、tail 和 CopyOut。 |

### Route R-002：FP32 原生

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `logit_grad_def.cpp` | `ops::LogitGrad` | FP32、ND、AutoContiguous | 生成 ACLNN OpDef 路径。 |
| Tiling | `logit_grad_tiling.cpp` | `RunBigKernelTiling` | FP16 未命中且 `dataType == ge::DT_FLOAT` | `tilingKey=2`。 |
| Router | `logit_grad.cpp` | `logit_grad` | `TILING_KEY_IS(2)` | `LogitGradND<float>`。 |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> logit_grad
  -> LogitGradND<float>::Init -> Process
     -> CopyInAndCast       # FP32 GM 直接到 x/dy FP32 views，无 Cast
     -> ComputeStepOne -> ComputeStepTwo
     -> CastAndCopyOut      # 无降精度 Cast，FP32 直接回 GM
```

##### 源码等价伪码

```text
# 分核、event、count/off 与 R-001 相同
DataCopyPad(xFp32@half+0, xGM+off, {1,count*4,0,0,0}, noPad)
DataCopyPad(dyFp32@half+32KiB,dyGM+off,{1,count*4,0,0,0},noPad)
MTE2_V(event)
tmp=-x; tmp+=1; tmp*=x; dy/=tmp
m0=(x>=epslion,ceil(count/64)*64); m1=(x<=1-epslion,same); m0&=m1
Select(x,m0,dy,selectValue,count)
V_MTE3(event); DataCopyPad(dxGM+off,xFp32,{1,count*4,0,0,0})
MTE3_MTE2(event)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `coreStart+i*8192` | `count*4` B/路 | FP32/ND | caller/K-001 | K-001/caller | 调用期间 | 单 block、stride 0。 |
| 每个半区 x | `halfBase+0` | 32 KiB | FP32 | DMA | Compute/CopyOut | 本 tile | 无 Cast 临时 view。 |
| 每个半区 dy | `halfBase+32KiB` | 32 KiB | FP32 | DMA | Div | 本 tile | Div 原地覆盖 dy。 |
| scratch/mask | `halfBase+64KiB`，mask2 起点 `+80KiB` | 32 KiB alias 区 | FP32/uint8/uint16 | StepOne/StepTwo | Div/Select | 本 tile | scratch 与 masks 分阶段复用。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  h0x["half0 x FP32 · 32KiB"]:1
  h0dy["half0 dy FP32 · 32KiB"]:1
  h0s["half0 scratch/masks · 32KiB"]:1
  h1x["half1 x FP32 · 32KiB"]:1
  h1dy["half1 dy FP32 · 32KiB"]:1
  h1s["half1 scratch/masks · 32KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * (32KiB + 32KiB + 32KiB alias) = 192KiB
dynamic_ub = N/A
workspace_total = 16MiB declared, 0 bytes consumed by visible Kernel
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dy FP32 | MTE2 | Div | 下次同半区 DMA | `MTE3_MTE2`（等待本 tile 输出） | Div 已完成且输出不再依赖原 dy。 |
| scratch/masks | Muls/Adds/Mul | Div | CompareScalar/And | Vector barrier | denominator 已被 Div 消费。 |
| x FP32 | MTE2 | arithmetic/compare | Select 结果原地覆盖 | Vector barrier | 最后一次 compare 已读取 x。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/半区 | `MTE3_MTE2`、`MTE2_V`、`V_MTE3` | MTE3/MTE2/V | 下一流水段 | 两个半区 | 与 R-001 相同。 |
| 计算 | `PIPE_V` barriers | Vector op | 下一 Vector op | 数据依赖/alias | 输入到输出保持 FP32，无 Cast round。 |
| tail | `count` 与 `ceil(count/64)*64` | Process | DMA/Compare/Select | 最后 tile | DMA/Select 只覆盖有效 count；mask 计算可触及半区内对齐 padding。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key 2。 |
| Kernel entry/router | `loss/logit_grad/op_kernel/logit_grad.cpp` | `LogitGradND<float>` | key 2 实例。 |
| Device body/helper | `loss/logit_grad/op_kernel/logit_grad.h` | `CopyInAndCast` / `CastAndCopyOut` | `if constexpr` 条件不命中，省去两端 Cast。 |

### Route R-003：BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `logit_grad_def.cpp` | `ops::LogitGrad` | BF16、ND、AutoContiguous | 生成 ACLNN OpDef 路径。 |
| Tiling | `logit_grad_tiling.cpp` | `RunBigKernelTiling` | 前两项未命中且 `dataType == ge::DT_BF16` | `tilingKey=3`。 |
| Router | `logit_grad.cpp` | `logit_grad` | `TILING_KEY_IS(3)` | `LogitGradND<bfloat16_t>`。 |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> logit_grad
  -> LogitGradND<bfloat16_t>::Init -> Process
     -> CopyInAndCast       # BF16 -> FP32 CAST_NONE
     -> ComputeStepOne -> ComputeStepTwo
     -> CastAndCopyOut      # FP32 -> BF16 CAST_RINT
```

##### 源码等价伪码

```text
# 地址、count、两路 DMA、FP32 算术与 mask 同 R-001，T=bfloat16_t
Cast(xFp32,xBf16Tmp,CAST_NONE,count)
Cast(dyFp32,dyBf16Tmp,CAST_NONE,count)
q=dyFp32/(xFp32*(1-xFp32)); mask=(x>=lo)&&(x<=hi); outFp32=Select(mask,q,selectValue)
Cast(xBf16Out,outFp32,CAST_RINT,count)
DataCopyPad(dxGM+off,xBf16Out,{1,count*2,0,0,0})
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `coreStart+i*8192` | `count*2` B/路 | BF16/ND | caller/K-001 | K-001/caller | 调用期间 | 单 block、stride 0。 |
| 两个 96 KiB 半区 | 与 R-001 完全相同 | 每半区 96 KiB | BF16 raw tail + FP32 + masks alias | DMA/Cast/Vector | Vector/MTE3 | 交替 tile | 原始 BF16 位于 FP32 区后半段。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  h0x["half0 x/bf16-tail alias · 32KiB"]:1
  h0dy["half0 dy/bf16-tail alias · 32KiB"]:1
  h0s["half0 scratch/masks · 32KiB"]:1
  h1x["half1 x/bf16-tail alias · 32KiB"]:1
  h1dy["half1 dy/bf16-tail alias · 32KiB"]:1
  h1s["half1 scratch/masks · 32KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * 96KiB = 192KiB
dynamic_ub = N/A
workspace_total = 16MiB declared, visible Kernel consumes 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| raw BF16 tail views | MTE2 | FP32 Cast | FP32 expansion/后续输出 Cast | `MTE2_V` + `PIPE_V` | expansion Cast 完成后 raw input 无需保留。 |
| scratch/masks | StepOne | Div | StepTwo | `PIPE_V` | 同 R-001。 |
| ping-pong half | 当前迭代 | MTE3 | 下下次迭代 MTE2 | per-half `MTE3_MTE2` | 同一 event 不能错配。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | `MTE2_V` + Cast | BF16 DMA | FP32 Vector | raw/expanded alias | 升精度用 `CAST_NONE`。 |
| 计算 | `PIPE_V` | FP32 Vector | 下一 op | 所有中间量 | FP32 除法与比较。 |
| 输出 | `V_MTE3` + Cast | FP32 Select | BF16 DMA | 输出 alias | 降精度明确用 `CAST_RINT`。 |
| tail | `count` / rounded compare count | Process | APIs | 尾 tile | 与 R-001 相同。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key 3。 |
| Kernel entry/router | `loss/logit_grad/op_kernel/logit_grad.cpp` | `LogitGradND<bfloat16_t>` | key 3 实例。 |
| Device body/helper | `loss/logit_grad/op_kernel/logit_grad.h` | `CopyInAndCast` / `CastAndCopyOut` | BF16→FP32 `CAST_NONE`，FP32→BF16 `CAST_RINT`。 |

## 算子特定优化

### Trick T-001：双半区手工事件流水复用固定 192 KiB UB

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：所有非空 tile 循环；`pingPongFlag` 在 0/1 间切换，分别绑定 EVENT0/EVENT1。
- Golden 来源：[`loss/logit_grad/op_kernel/logit_grad.h:134-153`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.h#L134-L153)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
pingPongFlag = 0;
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
for (int64_t i = 0; i < loopNum; i++) {
    int64_t localOffset = i * PP_ELEMENT_NUM;

    // 最后一轮的最后一个核处理余数
    if (remain > 0 && i == loopNum - 1 && blockIdx == lastCoreNum) {
        calNum = remain;
    }
    eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
    CopyInAndCast(eachCoreStartOffset + localOffset, calNum);

    ComputeStepOne(calNum);

    ComputeStepTwo(calNum);

    CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);

    pingPongFlag = 1 - pingPongFlag;
}
```

#### 机制与设计原因

一个 TBuf 被静态切成两个 96 KiB 半区；迭代 `i` 的半区和 event 由同一 flag 选择。`CopyOut` 最后发出该半区的 `MTE3_MTE2`，两次迭代后复用前必须等待，因此下一 tile 的 MTE2 不会覆盖尚未写回的结果。`AUTO_SYNC false` 使这些显式 event/barrier 成为实现协议的一部分。该机制允许 MTE3 与另一个半区的后续阶段形成流水，同时把 UB 上界固定在 192 KiB；固定仓未给出性能百分比。

#### 迁移边界

- 适用：逐 tile 独立、每个 tile 的 GM→UB→Vector→GM 数据流。
- 前提：两个半区各能容纳 96 KiB，EVENT0/1 与 half 选择严格一致，输出 DMA 完成后才复用。
- 失效：若增加跨 tile 状态、扩大任一半区或改变 pipeline，必须重做 UB 账和 event 图；固定仓没有单缓冲 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码协议本身表明，删除 `MTE3_MTE2` 等待会使下一次同半区 CopyIn 可能覆盖未完成 CopyOut 的数据。

### Trick T-002：半精度原始输入嵌入 FP32 区尾部并分阶段原地复用

#### 触发与源码依据

- 适用 Route：R-001、R-003。
- 触发条件：`T` 为 `half` 或 `bfloat16_t`。
- Golden 来源：[`loss/logit_grad/op_kernel/logit_grad.h:169-193`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.h#L169-L193)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
    int64_t elementByte = PP_ELEMENT_NUM * sizeof(T);
    x1Tmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                           tmpTensor[elementByte].ReinterpretCast<T>();
    x2Tmp = pingPongFlag ?
                tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
    DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);
    DataCopyPad(x2Tmp, dyGm[inputOffset], dataCopyParams, padParams);

} else {
    DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
    DataCopyPad(x2Tensor, dyGm[inputOffset], dataCopyParams, padParams);
}
```

#### 机制与设计原因

每个 8192 元素半精度输入只占 16 KiB，源码把它放到对应 32 KiB FP32 目标区的后半段；Cast 完成后，同一物理区直接作为 FP32 view 参与计算。StepOne 的 32 KiB scratch 随后又被 StepTwo 的两个 mask view 覆盖。逻辑 view 总容量大于物理 96 KiB，但生命周期不重叠，所以不能把各 view 独立相加；这一布局避免为两路 raw 半精度输入和 compare mask 另建 TBuf。

#### 迁移边界

- 适用：输入宽度小于内部计算宽度、Cast 与后续 view 的覆盖顺序可由 Vector barrier 证明的逐元素算子。
- 前提：8192 元素、源码相同 offset、Cast 支持这种重叠布局，并维持 `MTE2_V`/`PIPE_V` 顺序。
- 失效：FP32 Route 不需要 raw tail view；改变 tile、dtype 字节数或 Cast 方向必须重算 offset，不能照搬。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若把 raw input 放在 FP32 区起点而仍执行扩展 Cast，源/目的覆盖关系会变化，不能由当前源码的布局证明安全。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | dtype FP16，key 1 | dtype FP32，key 2 | dtype BF16，key 3 | Host 单函数 `if/else if` 顺序。 |
| 模板与 dtype | `LogitGradND<half>` | `LogitGradND<float>` | `LogitGradND<bfloat16_t>` | router 与 binary config 一致。 |
| 循环与分核 | colspan：三者相同，8192 元素/tile、former/tail core | 同左 | 同左 | dtype 不改变 tile 元素数。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 192 KiB TBuf；半精度 tail view | 192 KiB TBuf；直接 FP32 view | 192 KiB TBuf；BF16 tail view | 无 TQue/dynamic UB/通信。寄存器占用固定仓不可见。 |
| 内存总占用与复用 | raw/FP32/mask alias | FP32/scratch/mask alias | 同 FP16 | 物理总量均 192 KiB。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad+Cast+Vector` | `DataCopyPad+Vector` | `DataCopyPad+Cast+Vector` | 只有 AIV Vector。 |
| 同步、精度与 tail | FP32 compute，输出 `CAST_NONE` | FP32 compute，无 Cast | FP32 compute，输出 `CAST_RINT` | compare 对齐 64，输出只写有效 count。 |
| fallback/失败边界 | 无 dtype 内 fallback | 无 | 无 | 非支持 dtype Host 失败；未知 key entry 无输出。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` | 公开支持；`ascend910_93` | 公开支持，但由 `_950` 卡覆盖。 |
| Host/Launcher | 共用同一 OpDef/Infer/Tiling；核数来自 `GetCoreNumAiv()` | 同左 | 固定源码同一 Host，平台注册为 `ascend950`。 |
| Kernel/资源 | 同一 Kernel，CMake 第一组 compute units；编译 guard 可达 | 同左 | 同一 body，CMake 独立 `ascend950` 条目；固定 192 KiB TBuf。 |

## 限制与失败边界

- 公开 API 生成 bridge、executor launcher、非连续 Tensor 的实际 Contiguous/ViewCopy 节点及其 workspace 切片固定仓不可见；本文只把它们作为框架边界，不补造符号。
- 公开 `eps` 是 double，而 OpDef/Tiling/POD/Device 只保留 float；具体窄化代码与 double 边界处理固定仓不可见。
- 文档所述 `eps > 1` / `eps == 1` 特殊输出与可见 Kernel 的双边 mask/false=0 逻辑不一致；本次未执行数值验证。
- Tiling 无动态 UB 检查，Kernel 固定申请 192 KiB；不存在更小 UB、SIMT 或其它 fallback。
- Tiling 对 input tensor、attrs、input desc 和 dtype 有显式失败；shape、attr pointer、workspace/raw buffer 的部分子对象没有显式检查。unknown key 或非 220/310 编译值在 Kernel entry 内没有错误写回，只是不执行 body。
- `elementNum` 为 int64，`dataCount*sizeof(T)` 窄化为 uint32；固定 tile 将单次长度限制为 32768 B，但 origin shape 乘积本身的溢出处理固定仓不可见。
- 算子 Tiling 声明 16 MiB workspace，当前 Device body 不使用；不要据此推断 ACLNN 总 workspace 恒为 16 MiB，因为框架连续化可能追加临时区。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/logit_grad/docs/aclnnLogitGrad.md` | `aclnnLogitGradGetWorkspaceSize`、`aclnnLogitGrad` | 两段式原型、公开 Tensor/平台/错误契约。 |
| 入口声明/约束 | `loss/logit_grad/README.md` | `LogitGrad` | 数学语义、dtype、平台和边界说明。 |
| 图入口 | `loss/logit_grad/op_graph/logit_grad_proto.h` | `REG_OP(LogitGrad)` | 图语义、输入输出与默认 eps。 |
| OpDef/入口接线 | `loss/logit_grad/op_host/logit_grad_def.cpp` | `ops::LogitGrad`、`OP_ADD` | dtype/format/AutoContiguous 和三平台。 |
| shape/dtype 推导 | `loss/logit_grad/op_host/logit_grad_infershape.cpp` | `InferShape4LogitGrad`、`InferDataType4LogitGrad` | dx 复制 x 的 shape/dtype。 |
| POD | `loss/logit_grad/op_host/logit_grad_tiling.h` | `LogitGradTilingData` | 三个字段的声明顺序与类型。 |
| Host/Tiling | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key、core、workspace、POD、blockDim。 |
| Kernel entry | `loss/logit_grad/op_kernel/logit_grad.cpp` | `logit_grad` | 三 key router 与架构 guard。 |
| Kernel body | `loss/logit_grad/op_kernel/logit_grad.h` | `LogitGradND<T>` | 完整 Vector 数据流、UB、同步、精度和 tail。 |
| Host 构建 | `loss/logit_grad/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN OpDef 接线。 |
| Kernel 构建 | `loss/logit_grad/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3/950 kernel compute units 与 `AUTO_SYNC false`。 |
| A2 binary | `loss/logit_grad/op_host/config/ascend910b/logit_grad_binary.json` | `LogitGrad` op_list | 三 dtype binary 实例。 |
| A3 binary | `loss/logit_grad/op_host/config/ascend910_93/logit_grad_binary.json` | `LogitGrad` op_list | 三 dtype binary 实例。 |
| 公共构建依赖 | `cmake/func.cmake` | `add_modules_sources`、`add_kernel_sources` | OpDef ACLNN 目标和按 compute unit 选择 Kernel。 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | `elementNum/eps/needCoreNum` 的 producer→consumer | 字段声明顺序、64/32/64 位、元素/标量/core 单位；宏布局不可猜测 | `LogitGradTilingData` |
| 调用顺序 | 单 Kernel `Init→Process→CopyIn→StepOne→StepTwo→CopyOut` | key 与模板实例、每 tile 顺序、单 stream launch | `logit_grad`、`LogitGradND<T>` |
| workspace/内存 | 16 MiB 声明但 body 不用；192 KiB 两半区 | halfBase、x/dy/scratch offsets、alias 生命周期 | `RunBigKernelTiling`、`Init` |
| 同步 | EVENT0/1 与 ping-pong half 一一对应 | `MTE3_MTE2`、`MTE2_V`、`V_MTE3` 和 Vector barrier 不得省略/错序 | `Process`、`CopyInAndCast`、`CastAndCopyOut` |
| 精度与尾块 | FP32 内部计算、compare 向上对齐 64、Select/DMA 只用 count | FP16 `CAST_NONE`、BF16 `CAST_RINT`、尾核公式 | `ComputeStepTwo`、`CastAndCopyOut` |
| 失败边界 | 不支持 dtype Host 失败，空域循环 0 次 | unknown key/非目标编译值无 body；无 UB fallback | `RunBigKernelTiling`、`logit_grad`、`Process` |