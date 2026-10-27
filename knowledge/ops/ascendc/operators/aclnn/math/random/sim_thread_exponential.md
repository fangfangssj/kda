---
type: Operator
title: "SimThreadExponential A2/A3 ACLNN 算子知识"
description: "说明 aclnnSimThreadExponential 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 AIV_VECTOR 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/docs/aclnnSimThreadExponential.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_host/sim_thread_exponential_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_kernel/sim_thread_exponential.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_kernel/sim_thread_exponential.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:23:17Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# SimThreadExponential A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnSimThreadExponentialGetWorkspaceSize` + `aclnnSimThreadExponential`，公开 ACLNN 两段式接口 |
| 计算语义 | 用 Philox4x32-10 产生 `u`，写入 `x = -log(u) / lambda`；`selfRef` 是原地语义的输入/输出张量 |
| 输入输出 | `selfRef`：ND，最多 8 维，FP16/BF16/FP32；`count/seed/offset` 为 INT64，`lambda` 在 ACLNN 原型为 DOUBLE、进入 Op 属性后为 FLOAT |
| 执行类型 | 单 Kernel：AIV_VECTOR；没有可达 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | R-001 FP32 直写；R-002 FP16/BF16 FP32 计算后降精度写回 |
| 核心资源 | GM 输出、5 个 VECCALC `TBuf`、Philox 临时张量；用户 workspace 被申请但 Kernel body 不读取 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档/自动生成桥边界 → OpDef → arch32 Tiling/POD → dtype key router → AIV body → CMake 平台接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对每个线性位置生成 Philox4x32-10 的 32 位结果，先变换为 `u = r * 2^-32 + 2^-33`，再将等于区间上端的值回绕到 0。指数变换在 FP32 中执行：非零保护、`Log(u)`、接近 1 时用 `-epsilon/2` 替代对数结果，最后计算 `-log(u)/lambda`。FP16/BF16 路线在最后一步以 `CAST_RINT` 降精度，FP32 路线直接写回。

公开文档写作 `u ∈ (0, 1]`；Device 代码实际先构造半格偏移的 `(0,1)` 值，并含 `u == 1` 回绕及近 1 保护。实现事实以固定源码为准。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，文档要求维数 `<= 8`；输出 shape/dtype 与输入相同 | FP16/BF16/FP32 | OpDef 对输入启用 `AutoContiguous()`；Kernel 不读取原值，只通过输出参数覆盖结果 |
| `count` | 标量属性 | INT64 | Tiling 要求 `count >= 0` 且等于 `selfRef` storage shape 元素积；零元素继续进入分块公式会出现除零边界，见限制章节 |
| `lambda` / Op 属性 `lambd` | 标量属性 | ACLNN DOUBLE，OpDef/Tiling FLOAT | 公开 ACLNN 要求 `> 0`；arch32 Tiling 自身只拒绝 `== 0`，不得绕过 ACLNN 校验 |
| `seed` | 标量属性 | INT64 | 低/高 32 位写入 `key0/key1`，保持补码位模式 |
| `offset` | 标量属性 | INT64 | 转为 `uint64_t`，`state = offset & 3`，`offset_t = offset / 4`；arch32 未声明必须是 4 的倍数 |

### 入口与版本

固定仓公开的最新入口族是无版本后缀的 `aclnnSimThreadExponentialGetWorkspaceSize` 与 `aclnnSimThreadExponential`。第一段接口完成文档化校验、创建 executor 并返回 workspace 大小；第二段在调用者提供的 stream 上执行。自动生成 ACLNN inner bridge 与 executor 壳在本算子目录不可见，作为框架边界；算子拥有的 OpDef、Tiling producer、POD、router 与 Kernel body 均已追到真实实现。

README/API 产品表明确支持 Atlas A2 与 A3；`CMakeLists.txt` 将 `ascend910b`、`ascend910_93` 都映射到 `arch32`，OpDef 也注册两平台。固定仓未出现更高版本 ACLNN 符号。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 文档声明第一段拒绝空 `selfRef`、不支持 dtype、超过 8 维以及 `lambda <= 0`；非连续张量允许，OpDef 的 `AutoContiguous()` 是连续化框架边界。
2. OpDef 将 `self` 注册为同 dtype/ND 的输入与输出，将属性按 `count/lambd/seed/offset` 顺序交给 Host。
3. `SetAttrParams` 读取属性，固定 `start=0`、`end=1`，将 `lambda` 收窄为 float；其内部仅再次拒绝 `lambda == 0`。
4. `GetInputTensorInfo` 计算 storage shape 元素积，要求与 `count` 相等，再检查 FP16/BF16/FP32。
5. `GetPlatformInfo` 取得 AIV core 数与 UB 字节数；后者仅记录，当前 Host 不用它筛 Route。
6. `Tiling4Block` 生成分核、Philox key/counter 偏移与循环字段；任何失败返回 `GRAPH_FAILED` 并阻断启动。

### 参数、Workspace 与生命周期

`usrWorkspaceSize` 初值为 1，乘以所有 storage 维度后等于 `count`，再计算 `ceil(count / (6*256)) * 4` 字节。最终 workspace 为该用户区加 `GetLibApiWorkSpaceSize()`。框架在第一段返回总大小，调用者持有到第二段 stream 工作完成；Kernel entry 调用 `GetUserWorkspace(workspace)`，但 `SimThreadExponential::Init` 的 `workspace` 形参没有任何读取，因此用户区对当前 body 是未消费的保留空间，lib API 区仍由框架拥有。

POD 由 `SaveToBuffer` 序列化，blockDim 为 `useCoreNum`。输入与输出对象的 owner 是调用方/ACLNN executor；每个 AIV core 只绑定输出 GM 的本核 base，UB 由本核 `TPipe` 在 entry 生命周期内管理。

### 启动图与失败传播

单节点图：

```text
ACLNN GetWorkspaceSize
  -> generated executor / AutoContiguous framework boundary
  -> Tiling4SimThreadExponential::DoTiling
     [blockDim=useCoreNum, key∈{1,2,3}, POD=SimThreadExponentialTilingData]
  -> sim_thread_exponential(self, self_ref, workspace, tiling)
     -> SimThreadExponential<T>::Init
     -> SimThreadExponential<T>::Process
  -> self_ref GM
```

节点输入为 `self/self_ref/workspace/tiling`；实际 body 忽略 `self` 与用户 workspace，输出写 `self_ref`。没有 Kernel 间 workspace 边、event 或 atomic。Tiling 任一步失败时不产生有效 launch 参数；router 对未知 key 没有 `else` 报错，只会静默不写输出。

## Route 与参数设计

### 分核与资源计算

常量 `B=256` 元素、每次 handle 含 `H=6` 个 batch、Philox unroll `U=4`。令 `C=count`，平台 AIV 数为 `A`：

```text
default_block_num = (2048 / 256) * 78 = 624
batchNumTotal = min(ceil(C / 256), 624)
stepNum = 256 * batchNumTotal * 4                 # 元素
stepBlock = batchNumTotal * 4                     # 256-element block
roundedSizeNum = ceil(C / stepNum) * stepNum      # 元素
roundedSizeBlock = roundedSizeNum / 256            # block
useCoreNum = ceil(batchNumTotal / ceil(batchNumTotal / A))
batchNumPerCore = ceil(batchNumTotal / useCoreNum)
batchNumTailCore = batchNumTotal - (useCoreNum-1)*batchNumPerCore
handleNumLoop = floor(batchNumPerCore / 6)
handleNumTail = batchNumPerCore % 6
```

Host 没有用读到的 `ubSize` 检查 178176 B 固定 UB 配置；资源前提由 Kernel 固定分配隐式给出。所有乘法中 `count/numel` 为 int64，但多个 batch/rounded 字段收窄到 uint32，固定源码没有显式溢出检查。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 直写 | key 3，`SimThreadExponential<float>` | `selfDType == DT_FLOAT`，`GetDataTypeKey` 令 `dataSizeType=3` | dtype 不匹配则检查 key 1 |
| 2 | R-002 低精度写回 | key 1/2，`SimThreadExponential<half/bfloat16_t>` | FP16→1，BF16→2；router 按 1 后 2 判断 | 其他 dtype 在 Tiling 失败；未知 key 在 entry 静默结束 |

```text
dtype == FLOAT16  -> key=1 -> SimThreadExponential<half>
dtype == BFLOAT16 -> key=2 -> SimThreadExponential<bfloat16_t>
dtype == FLOAT32  -> key=3 -> SimThreadExponential<float>
otherwise         -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 直写 | 3 | FP32、ND、count 等于元素数、ACLNN lambda>0、A2/A3 arch32 | 单 AIV Kernel | `SimThreadExponential<float>::Process/genRandomData` | 178176 B TBuf；全程 PIPE_V/PIPE_ALL barrier；FP32 GM 直写 | former/tail core；以 256 元素整块写，未知 key 无输出 |
| R-002 低精度写回 | 1/2 | FP16 或 BF16，其余条件同 R-001 | 单 AIV Kernel | `SimThreadExponential<half/bfloat16_t>` | 同一 UB 图；FP32 变换后 `CAST_RINT` 到目标 dtype | 与 R-001 相同，另含降精度 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype key | FP16/BF16/FP32 | router 的三种模板实例 | 全部 | Tiling `GRAPH_FAILED` | `GetDataTypeKey`, `sim_thread_exponential` |
| 非空隐式边界 | `batchNumTotal > 0` 才能安全计算 `stepNum` 分母 | 零 count 令 `stepNum=0`，后续 rounded 公式除零 | 全部 | 固定源码未提供安全 fallback | `Tiling4Block` |
| UB 下界 | 每核实际固定分配 178176 B | 5 个 TBuf 必须同时容纳 Philox/比较/转换临时区 | 全部 | Host 未检查，固定源码无 fallback | `SimThreadExponential::Init` |
| handle 宽度 | `batchNumPerCore = 6*q+r` | 每次向量化 6×256 个 counter，四路 unroll | 全部 | `r>0` 进入 tail handle | `Process` |
| 输出类型 | `T1==float` 编译期分支 | FP32 直接复用转换结果；低精度需 cast buffer | R-001/R-002 | 由 dtype key 静态选择 | `genRandomData` |
| 尾部写保护 | 每个 stripe 仅检查 `stripe_start <= numel` | 搬运单位仍为 `batchNumCurHandle*256` 整块 | 全部 | 固定源码没有逐元素 masked copy fallback | `genRandomData` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 的生成宏展开在固定算子目录不可见，因此 `sizeof/alignof/offsetof` 不作推测；字段声明顺序、源码类型和 producer/consumer 已闭合。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `SimThreadExponentialTilingData` | `key0` | uint32/32 | 宏生成布局不可见 | bits | `uint32(seed)` | Philox key low | 截断低 32 位 |
| 同上 | `key1` | uint32/32 | 同上 | bits | `uint32(seed >> 32)` | Philox key high | 算术右移后截断 |
| 同上 | `offset_t_low` | uint32/32 | 同上 | counter | `uint32(offset/4)` | `setnlo1nhi1` | offset 先转 uint64 |
| 同上 | `offset_t_high` | uint32/32 | 同上 | counter | `uint32((offset/4)>>32)` | `setnlo1nhi1` | 高 32 位 |
| 同上 | `useCoreNum` | uint32/32 | 同上 | AIV core | 分核公式 | `Init` former/tail 判定 | 必须非零 |
| 同上 | `batchNumPerCore` | uint32/32 | 同上 | 256-element batch | ceil 分配 | 本核 base/handle 循环 | 非尾核份额 |
| 同上 | `batchNumTailCore` | uint32/32 | 同上 | batch | 余数公式 | 最后一核份额 | 仅 tail core |
| 同上 | `batchNumTotal` | uint32/32 | 同上 | batch | `min(ceil(count/256),624)` | 四 stripe 地址步长 | 最大 624 |
| 同上 | `numel` | int64/64 | 同上 | element | `count` | 输出边界条件 | 无收窄 |
| 同上 | `stepBlock` | uint32/32 | 同上 | 256-element block | `batchNumTotal*4` | 外层线性 block 步长 | count=0 时为0 |
| 同上 | `roundedSizeBlock` | uint32/32 | 同上 | block | `ceil(numel/stepNum)*stepNum/256` | 外层循环上界 | Host 无溢出保护 |
| 同上 | `range` | float/32 | 同上 | value | `end-start` | 仅解析为成员；变换重新算 range | 固定为1 |
| 同上 | `handleNumLoop` | uint32/32 | 同上 | handle | `batchNumPerCore/6` | `Process` 主循环 | 每 handle 6 batch |
| 同上 | `handleNumTail` | uint32/32 | 同上 | batch | `batchNumPerCore%6` | `Process` tail | 0 时 inactive |
| 同上 | `state` | uint64/64 | 同上 | lane | `offset & 3` | `setXTensor` 的 2/3/else 分支 | state 0/1 共用 else |
| 同上 | `start` | float/32 | 同上 | value | 固定 0 | `uniform_func` | 全 Route active |
| 同上 | `end` | float/32 | 同上 | value | 固定 1 | `uniform_func` | 全 Route active |
| 同上 | `lambda` | float/32 | 同上 | rate | ACLNN double→Op float | `Duplicate` 后 `Div` | 公开接口要求 >0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 直写 | key 3 | `op_host/arch32/sim_thread_exponential_tiling_arch32.cpp::DoTiling/GetDataTypeKey` | `SimThreadExponentialTilingData` | `op_kernel/sim_thread_exponential.cpp::sim_thread_exponential` | `op_kernel/sim_thread_exponential.h::SimThreadExponential<float>` | `CMakeLists.txt::SUPPORT_TILING_DIR` arch32 |
| R-002 低精度写回 | key 1/2 | 同上 | 同上 | 同上 | `SimThreadExponential<half/bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD，创建本核 `TPipe`，按 key 实例化模板。`Init` 计算本核 batch/output base，配置五个 TBuf。`Process` 以 6-batch handle 循环；每个 handle 初始化 Philox counter，执行 10 轮 Philox，按 `stepBlock` 跨步生成四路结果，将 uint32 映射到 uniform FP32，做指数变换并写 GM。各向量依赖由显式 `PipeBarrier<PIPE_V/PIPE_ALL>` 串联，没有 Queue、atomic 或跨核同步。

### Route R-001：FP32 直写

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnSimThreadExponential.md` | 两段式原型 | selfRef FP32/ND，lambda>0 | 进入 executor |
| Host | `op_host/arch32/sim_thread_exponential_tiling_arch32.cpp` | `GetDataTypeKey` | `DT_FLOAT` | `dataSizeType=3` |
| Tiling | 同上 | `SetTilingKey` | 无额外 selector | key 3 |
| Router | `op_kernel/sim_thread_exponential.cpp` | `sim_thread_exponential` | `TILING_KEY_IS(3)` | `SimThreadExponential<float>` |

#### Device 数据流

##### 函数级调用链

```text
sim_thread_exponential(..., tiling)
  -> GET_TILING_DATA
  -> SimThreadExponential<float>::Init
     -> ParseTilingData + InitBuffer + SetGlobalBuffer
  -> Process
     -> genRandomData
        -> setnlo0nhi0 -> setnlo1nhi1
        -> rand_Philox4x32_10 (10 rounds via multiply_uint64)
        -> setCtrTensor -> setXTensor
        -> uint32_to_float_func -> uniform_func
        -> Compare/Select -> Log -> Muls(-1) -> Div(lambda)
        -> DataCopy FP32 UB→GM
```

##### 源码等价伪码

```text
for handle in [0, handleNumLoop) plus optional tail:
  threadIdxBlock = core_batch_base + handle*6
  construct 4-word counters for active_batch*256 lanes; add offset/4 and state
  philox_output = Philox4x32_10(counter, seed_low, seed_high)
  for linearBlock = threadIdxBlock; linearBlock < roundedSizeBlock; linearBlock += stepBlock:
    advance counter; run Philox4x32-10
    x_u32 = select four Philox lanes according to state
    u_fp32 = uint32_to_float(x_u32) * 2^-32 + 2^-33
    if u == 1: u = 0
    if u == 0: u = 1
    logv = log(u)
    if not (u < 1-eps/2): logv = -eps/2
    out_fp32 = -logv / lambda
    for stripe s in 0..3:
      if linearBlock*256 + s*batchNumTotal*256 <= numel:
        copy active_batch*256 FP32 elements to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `self_ref` | core base=`batchNumPerCore*core*256`; stripe adds `s*batchNumTotal*256` | `count` logical elements | FP32/ND | Device transform | caller | caller-owned，stream 完成后可读 | 以 256 元素块写 |
| `ctrTensorBuf` | UB 独立区 | `6*4*256*4=24576 B` | uint32 | counter builders | Philox | TPipe/kernel | 单缓冲 |
| `outputBuf` | UB 独立区 | 24576 B | uint32/FP32 reinterpret | Philox | counter advance/transform | TPipe/kernel | 原地多阶段复用 |
| `ctrTensorTmpAndTmpBuf` | 子区 0/24576 B | `2*24576=49152 B` | uint32/FP32 | Philox/counter copy | conversion/transform | TPipe/kernel | 两个 24576 B alias 子区 |
| `largeTmpTensorBuf` | offset 0 多种视图 | 49152 B | uint32 | compare/multiply/xTensor | Philox/conversion | TPipe/kernel | helper 间覆盖复用 |
| `tmpTensorBuf` | 五个 6144 B 子区 | `5*6*256*4=30720 B` | uint32/float/mask | helpers | helpers/Div denominator | TPipe/kernel | `tmp0..tmp4` 与后续转换视图复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ctr["ctrTensorBuf · 24576 B"]:1
  out["outputBuf · 24576 B"]:1
  pair["ctrTmp + tmp · 49152 B"]:1
  large["largeTmp · 49152 B"]:1
  small["tmp0..tmp4 · 30720 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 24576 + 24576 + 49152 + 49152 + 30720 = 178176 B
UB_total <= platform_UB 是隐式运行前提；Host 读取 UB 但未检查该不等式
dynamic_ub = N/A（AIV TPipe 固定分配）
workspace_total = ceil(count/1536)*4 + GetLibApiWorkSpaceSize()
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `outputBuf` | Philox round | `DataCopy(ctrTensorTmp, output)` | 下一 counter/Philox round | `PIPE_V` barrier | round 消费完成后覆盖 |
| `ctrTensorTmp` | Philox result copy | uniform/log/div | 下一 linearBlock | `PIPE_ALL`/`PIPE_V` | GM copy 完成后进入下一轮 |
| `largeTmpTensorBuf` | compare/multiply helpers | helper 内部立即消费 | `xTensor` zero/write | barriers 串行 | helper 临时值不跨调用存活 |
| `tmpTensorBuf` | `tmp0..tmp4` Philox helpers | counter/Philox | `u32Tof32Tensor`、mask、lambda vector | `PIPE_ALL` before reinterpret | Philox 阶段结束后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| counter/Philox | `PipeBarrier<PIPE_V>` | Duplicate/Add/Mul/Xor | 下一向量指令 | 同一 TBuf | uint32 位运算，10 rounds |
| 转换 | `PIPE_ALL` + `PIPE_V` | selected uint32 | cast/scale/uniform | alias TBuf | FP32 uniform |
| 指数变换 | Compare/Select/Log/Div | uniform FP32 | GM copy | `ctrTensorTmp` | eps/2 近 1 保护，FP32 运算 |
| 输出 | `DataCopy` + `PIPE_ALL` | FP32 result | 下一迭代 | GM stripe | 整块 256 元素；条件仅保护起点 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch32/sim_thread_exponential_tiling_arch32.cpp` | `Tiling4Block/SetTilingData` | 分核、offset、POD、workspace |
| Kernel entry/router | `op_kernel/sim_thread_exponential.cpp` | `sim_thread_exponential`, key 3 | 路由到 float 模板 |
| Device body/helper | `op_kernel/sim_thread_exponential.h` | `SimThreadExponential<float>::genRandomData` | Philox、FP32 变换、直写与 tail 条件 |

### Route R-002：FP16/BF16 低精度写回

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnSimThreadExponential.md` | 两段式原型 | selfRef FP16 或 BF16 | 进入 executor |
| Host | `op_host/arch32/sim_thread_exponential_tiling_arch32.cpp` | `GetDataTypeKey` | FP16→1；BF16→2 | 写 dtype key |
| Tiling | 同上 | `SetTilingKey` | 无额外 selector | key 1/2 |
| Router | `op_kernel/sim_thread_exponential.cpp` | `sim_thread_exponential` | key 1 优先于 key 2 的 `else if` | half/BF16 模板 |

#### Device 数据流

##### 函数级调用链

```text
sim_thread_exponential
  -> SimThreadExponential<half|bfloat16_t>::Init/Process
  -> genRandomData
     -> shared Philox + FP32 uniform/exponential pipeline
     -> Cast<T1,float>(xTensor, ctrTensorTmp, CAST_RINT)
     -> DataCopy T1 UB→GM
```

##### 源码等价伪码

```text
execute the same counter, Philox4x32-10, uniform and FP32 -log(u)/lambda pipeline as R-001
low = cast_rint<T1>(fp32_result)
for each of four stripes whose start satisfies start <= numel:
  copy active_batch*256 T1 elements from xTensor stripe to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `self_ref` | 同 R-001 | `count` logical elements | FP16/BF16 ND | low-precision copy | caller | caller-owned | 256 元素块 |
| 五个 TBuf | 同 R-001 | 合计 178176 B | uint32/FP32/T1 reinterpret | Philox/transform | cast/copy | TPipe/kernel | 固定同图 |
| `largeTmpTensorBuf` 的 `xTensor` | offset 0 | 24576 B 有效视图 | T1 packed over uint32 storage | `Cast<T1,float>` | GM copy | 迭代内 | 复用 Philox large temp |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ctr["ctrTensorBuf · 24576 B"]:1
  out["outputBuf · 24576 B"]:1
  pair["ctrTmp + tmp · 49152 B"]:1
  large["largeTmp/xTensor · 49152 B"]:1
  small["tmp0..tmp4 · 30720 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 178176 B
dynamic_ub = N/A
workspace_total = ceil(count/1536)*4 + GetLibApiWorkSpaceSize()
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `largeTmpTensorBuf/xTensor` | Philox helpers，随后 `Cast` | four GM copies | 下一 linearBlock helper | `PIPE_ALL` | 所有 stripe copy 完成后覆盖 |
| `ctrTensorTmp` | conversion/transform | `Cast<T1,float>` | 下一轮 | `PIPE_ALL` | cast 完成后可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| shared pipeline | 同 R-001 | Philox/uniform | exponent | TBuf aliases | FP32 |
| 降精度 | `Cast<T1,float>(CAST_RINT)` + `PIPE_ALL` | FP32 exponent | GM copy | `xTensor` | FP16/BF16 舍入 |
| 输出 | `DataCopy` + `PIPE_ALL` | low result | 下一迭代 | GM stripe | 仍以 256 元素整块输出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch32/sim_thread_exponential_tiling_arch32.cpp` | `GetDataTypeKey` | FP16/BF16 key |
| Kernel entry/router | `op_kernel/sim_thread_exponential.cpp` | key 1/2 branches | half/BF16 实例 |
| Device body/helper | `op_kernel/sim_thread_exponential.h` | `genRandomData`, `T1!=float` branch | FP32 计算、CAST_RINT、低精度写回 |

## 算子特定优化

### Trick T-001：一次 Philox 结果以四路 stride 写回

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：所有 arch32 dtype；`unroll_factor=4`，`stepBlock=batchNumTotal*4`
- Golden 来源：[`op_kernel/sim_thread_exponential.h:295-320`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_kernel/sim_thread_exponential.h#L295-L320)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for(uint32_t linearIndexBlock = threadIdx; linearIndexBlock < roundedSizeBlock; linearIndexBlock += stepBlock) {
    uint32_t linearIndexBlockCurCore = linearIndexBlock - blockIdxoffset;
    AscendC::DataCopy(tmp, output, batchNumCurHandle * unroll_factor * blockSize);

    setCtrTensor(ctrTensor, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, largeTmpTensorBuf, batchNumCurHandle * blockSize);
    AscendC::DataCopy(output, ctrTensor, batchNumCurHandle * unroll_factor * blockSize);
```

#### 机制与设计原因

Philox4x32 每个 counter 产生 4 个 uint32。代码把四路结果保持在连续 UB 中，并按 `batchNumTotal*256` 的 GM stride 写到四个输出区，而不是为每个输出重新执行一次随机算法；外层 `stepBlock=batchNumTotal*4` 与这四路展开严格配套。这减少重复 counter/round 计算并扩大单 handle 的向量工作量。

#### 迁移边界

- 适用：counter-based PRNG 且每 counter 固定产生四个独立结果的逐元素变换。
- 前提：四路 counter 映射、offset state、GM stripe 地址和 tail 规则必须整体迁移。
- 失效：改变 PRNG 输出宽度或只消费部分 lane 时必须重算 `stepBlock`、counter 跳步与写回布局。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得只复制四路写回而保留单路 counter 步进。

### Trick T-002：Philox 临时区在指数变换阶段原地复用

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：完成当前轮 Philox 后进入 uint32→uniform→exponential 阶段
- Golden 来源：[`op_kernel/sim_thread_exponential.h:317-345`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_kernel/sim_thread_exponential.h#L317-L345)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::LocalTensor<uint32_t> xTensor = largeTmpTensorBuf.GetWithOffset<uint32_t>(UINT_DOUBLE * batchNumCurHandle * unroll_factor * blockSize, static_cast<uint32_t>(0));
AscendC::Duplicate<uint32_t>(xTensor, zeroVal, batchNumCurHandle * unroll_factor * blockSize);
AscendC::PipeBarrier<PIPE_V>();
setXTensor(xTensor, tmp, ctrTensorTmp, batchNumCurHandle * blockSize);

AscendC::PipeBarrier<PIPE_ALL>();
ctrTensorTmp = ctrTensorTmpAndTmpBuf.GetWithOffset<uint32_t>(UINT_DOUBLE* batchNumCurHandle * unroll_factor * blockSize, static_cast<uint32_t>(0));
AscendC::LocalTensor<uint32_t> u32Tof32Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumCurHandle * unroll_factor * blockSize, static_cast<uint32_t>(0));
```

#### 机制与设计原因

`largeTmpTensorBuf` 先服务 64 位乘法/比较 helper，Philox 消费结束后被重新解释为 `xTensor`；`tmpTensorBuf` 的五个 Philox 子区随后复用为转换临时值、mask 与 lambda vector。`PIPE_ALL` 划定复用边界，使固定 178176 B UB 同时覆盖随机算法和后处理，而无需把两阶段峰值相加。

#### 迁移边界

- 适用：前后两个阶段严格串行、前阶段临时值不跨阶段存活的 AIV pipeline。
- 前提：保留 barrier，复用视图容量必须覆盖最大 active handle。
- 失效：若 Philox 与指数变换流水重叠或异步消费临时区，则必须拆分 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；删除阶段间 `PIPE_ALL` 会失去源码现有的覆盖顺序证明。

## 模板与实现对比

| 对比维度 | R-001 FP32 直写 | R-002 FP16/BF16 低精度写回 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 3 | key 1/2 | dtype 唯一决定 |
| 模板与 dtype | `T1=float` | `T1=half/bfloat16_t` | Host key 与 entry 实例一致 |
| 循环与分核 | 相同 | 相同 | 随机序列映射与 dtype 无关 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 5 TBuf，178176 B | 相同 | FP32 中间管线共用 |
| 内存总占用与复用 | `ctrTensorTmp` 直接输出 | `largeTmp/xTensor` 承接 cast | 输出 dtype 不同 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Vector Philox + FP32 DataCopy | 同管线 + Cast | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | FP32，整块 tail | FP32→CAST_RINT，整块 tail | 编译期分支 |
| fallback/失败边界 | 未知 key 静默无输出 | 同左 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；`ascend910b` | 支持；`ascend910_93` | 支持但由独立 `_950` 卡覆盖 |
| Host/Launcher | arch32 共用 | arch32 共用 | arch35 `RandomTilingArch35` |
| Kernel/资源 | AIV_VECTOR、固定 178176 B UB | 同左 | SIMT VF、无 AIV TBuf |

## 限制与失败边界

- 公开 ACLNN 要求 `lambda > 0`，arch32 Tiling 自身只拒绝 `lambda == 0`；复用 Host/Tiling 时必须保留公开入口校验，不能把内部检查误写成完整约束。
- API 文档只在 950 小节明列 `count == selfRef.numel`，但 arch32 Tiling 也强制该等式；A2/A3 实际 Host 行为更严格。
- `count==0` 能通过 `count>=0` 和 shape-product 检查，却令 `Tiling4Block` 的 `stepNum` 为 0，后续 rounded 公式存在除零；固定源码没有空域 fallback。
- 输出 tail 只比较每个 stripe 起始位置与 `numel`，随后整块复制 `batchNumCurHandle*256` 元素；固定源码没有逐元素 mask，非 256 对齐容量的安全性无法从该目录证明。
- `state=offset&3`，Device 只为 state 2、3 提供专门 lane 拼接，state 0 与 1 共用 else；固定源码未提供 state 1 的独立分支或反例。
- Host 读取 UB 容量但不与固定 178176 B 配置比较；平台 UB 小于该值时没有 Host fallback。
- 用户 workspace 字节区被计算并传入，但当前 Kernel body 不消费；不得把它当作跨节点中间结果。
- 自动生成 ACLNN bridge、框架连续化和 lib API workspace 内部不可见；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `random/sim_thread_exponential/docs/aclnnSimThreadExponential.md` | ACLNN 两段式原型 | 公开性、参数与平台 |
| Op 注册 | `random/sim_thread_exponential/op_host/sim_thread_exponential_def.cpp` | `SimThreadExponential` | dtype/ND/AutoContiguous/平台 |
| Host/Tiling/POD | `random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.h` | `SimThreadExponentialTilingData` | POD 字段 |
| Host/Tiling/Launcher | `random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp` | `DoTiling/Tiling4Block` | 校验、分核、workspace、key |
| Kernel entry | `random/sim_thread_exponential/op_kernel/sim_thread_exponential.cpp` | `sim_thread_exponential` | dtype router |
| Kernel body | `random/sim_thread_exponential/op_kernel/sim_thread_exponential.h` | `SimThreadExponential<T1>` | AIV pipeline、UB、Philox、tail |
| 构建接线 | `random/sim_thread_exponential/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3→arch32 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 17 字段的 producer/consumer 映射 | 字段顺序、位宽、元素/块单位 | `SimThreadExponentialTilingData`, `ParseTilingData` |
| 调用顺序 | Init→handle→Philox→uniform→log/div→copy | 10-round key 增量与四路展开顺序 | `Process/genRandomData` |
| workspace/内存 | 固定 178176 B UB 与阶段复用 | 复用边界、用户 workspace 未消费 | `Init`, `DoTiling` |
| 同步 | Vector/alias 依赖用 PIPE_V/PIPE_ALL | buffer 覆盖前 barrier | `genRandomData` |
| 精度与尾块 | FP32 变换，低精度 CAST_RINT | epsilon 保护、整块写回条件 | `genRandomData` |
| 失败边界 | Host dtype/count/lambda/核数错误阻断 launch | 未知 key、空域、UB 与尾块缺少 fallback | `DoTiling`, entry router |