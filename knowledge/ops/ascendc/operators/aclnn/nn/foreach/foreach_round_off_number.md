---
type: Operator
title: "ForeachRoundOffNumber A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachRoundOffNumberV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/docs/aclnnForeachRoundOffNumberV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/op_api/aclnn_foreach_round_off_number_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/op_api/foreach_round_off_number_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/arch22/foreach_round_off_number_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/foreach_round_off_number_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/config/ascend910b/foreach_round_off_number_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/config/ascend910_93/foreach_round_off_number_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/foreach_round_off_number.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/foreach_round_off_number.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:59:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachRoundOffNumber A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachRoundOffNumberV2GetWorkspaceSize` / `aclnnForeachRoundOffNumberV2`，公开 ACLNN V2 两段式接口 |
| 计算语义 | 对列表中每个张量逐元素按 `roundMode` 舍入；V2 将标量转换成一个 `DT_INT8` Tensor 后启动 `ForeachRoundOffNumber` |
| 输入输出 | `x[]`/`out[]`：同 dtype 的 `FLOAT32/FLOAT16/BFLOAT16`、ND、单 Tensor 0–8 维；`roundMode`：`INT8/INT64` 标量 |
| 执行类型 | 单 Kernel，`AIV_VECTOR` |
| Kernel 路线 | R-001 FP32 直算；R-002 FP16/BF16 升精度分段计算 |
| 核心资源 | GM TensorList 描述符、单深度 VECIN/VECOUT Queue；FP16/BF16 另有单深度 FP32 Queue；算子 Kernel 不读 user workspace |
| 来源闭包 | `source_closed`：公开 V2 ACLNN → executor/L0 launcher → arch22 common tiling/POD → dtype router → `ForeachRoundOffNumberND<T>` → CMake/config 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设输入列表为 `x=[x_0,...,x_{n-1}]`，输出列表为 `y`，则每个有效元素满足 `y_i[j]=round(x_i[j], roundMode)`。A2/A3 Device 将模式 `0..6` 映射到同名 `AscendC::RoundMode`；源码还为值 `7` 实现 `frac(x)=x-trunc(x)`，其它值回落 `CAST_NONE`。对 FP16/BF16，先升到 FP32、在 FP32 中舍入，再以 `CAST_RINT` 降回原 dtype。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；各 Tensor ND、0–8 维 | FP32/FP16/BF16，列表内一致 | API 非空；空列表由 ACLNN 直接成功返回；每个非连续输入经 executor `Contiguous` |
| `roundMode` | `aclScalar`，单值 | INT8/INT64 | executor 总是 `ConvertToTensor(..., DT_INT8)`；公开值 1–6 分别为 rint/floor/ceil/round/trunc/odd |
| `out` | TensorList；与对应输入 shape 相等（ACLNN 层） | 与 `x` 一致 | API 非空；输出不做 ViewCopy，调用方提供连续输出 |

### 入口与版本

固定仓同时公开 `aclnnForeachRoundOffNumber` 与 `aclnnForeachRoundOffNumberV2`；V2 文档明确说明相较旧接口仅把 `roundMode` 从 `aclTensor` 改为 `aclScalar`。按同架构取最新公开版本，本卡采用 V2。A2/A3 产品支持由 V2 文档、`ForeachRoundOffNumber` OpDef 的 `ascend910b/ascend910_93` 注册、Host CMake 的 arch22 映射和 Kernel CMake 共同支撑。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecForeachRoundOffNumberV2GetWorkspaceSize` 依次创建 executor，检查 `x/roundMode/out` 非空、平台属于 DAV_2201 或 regbase、列表 dtype 一致且输入输出 dtype 相同、输入维数不超过 8、对应输入输出 shape 相等、storage format 非私有格式。空 `x` 列表直接令 workspace 为 0 并返回。其余输入逐 Tensor 调用 `l0op::Contiguous`，把标量以 `DT_INT8` 转 Tensor，再由 `l0op::ForeachRoundOffNumberV2` 添加一个 `ForeachRoundOffNumber` AICore launcher；任一步失败均不提交后续节点。

### 参数、Workspace 与生命周期

第一段返回 executor 汇总的 workspace；它可能包含输入 `Contiguous` 的框架 workspace，并包含 Tiling 设置的 32 字节 `WORK_SPACE_SIZE`。算子 Kernel entry 明确把 `userWS` 设为 `nullptr`，因此这 32 字节不被算子 body 读写。executor 持有连续化临时 Tensor、标量 Tensor、launcher 与其生命周期；第二段 `CommonOpExecutorRun` 在调用方 stream 上按 executor 顺序执行。

### 启动图与失败传播

```text
aclnnForeachRoundOffNumberV2GetWorkspaceSize
  -> [每个非连续 x_i] Contiguous（框架边界，可选）
  -> ConvertToTensor(roundMode, DT_INT8)
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachRoundOffNumber)
       -> Tiling4ForeachRoundOffNumberTiling
       -> foreach_round_off_number<<<blockDim=needCoreNum>>>
```

核心语义只有一个 Device 节点。输入边是 executor 所持连续 `x` TensorList、INT8 标量 Tensor 和调用方 `out` TensorList；POD 随 launcher 传入。所有节点处于同一 executor/stream 顺序，前处理或 launcher 添加失败立即返回错误，不启动 Kernel。固定仓未展开 `Contiguous` 内部 Kernel 与其 workspace 切片，属于允许的框架边界。

## Route 与参数设计

### 分核与资源计算

令 `s=sizeof(T)`、`e=32/s`、`N=sum_i numel(x_i)`，Host 取 `needCoreNum=min(AIV核数, ceil(N/e), 50)`，且空域最少为 1。把 `ceil(N/e)` 个 32B block 尽量均分给各核；POD 为每核记录起止 Tensor 索引及首尾元素偏移。令 `U=平台UB字节数`、`D=ForeachCommonTilingData::GetDataSize()`、`C=floor((U-D)/2)`、`P=floor(256/(256+2s)*C)`。FP32 令 `B=floor(P/32)*32`；FP16/BF16 令 `B=floor((P/10)/64)*64`。`B` 即 `inputsTensorUbSize`。workspace 固定写 32B。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 直算 | key 2，`ForeachRoundOffNumberND<float>` | `dataType == ge::DT_FLOAT` | key 1/4 或无 body |
| 2 | R-002 FP16/BF16 升精度 | key 1 → `half`；key 4 → `bfloat16_t` | `DT_FLOAT16`；或 `DT_BF16` 且编译满足 `__CCE_AICORE__>=220` 且 NPU arch 非 3003/3113 | 未知 key 无输出；BF16 编译 guard 不满足时无 body |

```text
dtype FP32 -> key 2 -> ND<float>
dtype FP16 -> key 1 -> ND<half>
dtype BF16 -> key 4 -> [编译 guard] ND<bfloat16_t>
其它 dtype -> ACLNN/OpDef 拒绝；未知 key -> entry 不执行 body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 直算 | 2 | FP32，A2/A3，ACLNN 校验通过 | 单 AIV Kernel | `ForeachRoundOffNumberND<float>` + `InnerComputer<float>` | input/output Queue 各 `B`，Queue 与 V pipe barrier | 整 tile `DataCopy`，尾 tile `DataCopyPad` |
| R-002 FP16/BF16 升精度 | 1/4 | FP16；或支持 BF16 编译 guard 的 A2/A3 | 单 AIV Kernel | `ForeachRoundOffNumberND<half/bfloat16_t>` + 通用 `InnerComputer<T>` | input/output Queue 各 `9B`，FP32 Queue `2B`；逐 cast barrier | tile 内按 `B/4` 元素再切；GM tail 用 `DataCopyPad` |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32B 分核 | `elementsPerBlock=32/s` | 核边界以 DMA block 粒度均分 | 全部 | 空域仍启动 1 核 | `ForeachCommonTiling::GetNeedCoreNum/AssignDataToEachCore` |
| 最大核/列表 | `needCoreNum<=50`，POD 最多 50 Tensor | POD 固定数组边界 | 全部 | 超出固定数组可表达边界；公开 API 未单独声明 50 上限 | `MAX_TENSOR_CONT/MAX_CORE_CONT` |
| UB 预算 | 上述 `B` 公式，FP16/BF16 额外 `/10` 与 64B 下对齐 | 实际 Queue 总量 FP32=`2B`，FP16/BF16=`20B` | 全部 | 无备用 Route |
| BF16 编译 | key 4 且 `__CCE_AICORE__>=220`，非 arch 3003/3113 | BF16 body 仅在该 guard 下编译 | R-002 | entry 无 body |
| 空列表 | `x->Size()==0` | 避免构造/启动空 TensorList Kernel | 全部 | ACLNN 成功、workspace=0 |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 宏生成布局；固定仓未展开 `offsetof` | 字节 | `DivideUbMemory9` 公式 | `ParseTilingData`；Queue 容量与 tile 上界 | 32/64B 下对齐 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 宏生成布局 | 元素 | `Init` 逐 Tensor `GetShapeSize` | `Process` 确定各 Tensor 尾位置 | 每项可为 0 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 宏生成布局 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | 仅活动核读取 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 宏生成布局 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | 仅活动核读取 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 宏生成布局 | 元素 | `AssignDataToEachCore` | `Process` 的 `cursorStart` | 首 Tensor 核内偏移 |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 宏生成布局 | 元素 | `AssignDataToEachCore` | `Process` 的 `cursorEnd` | 尾 Tensor 核内偏移 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | key 2 | arch22 `Tiling4ForeachRoundOffNumberTiling` → common tiling | `ForeachCommonTilingData` | `foreach_round_off_number` key 2 | `ForeachRoundOffNumberND<float>` | Kernel CMake A2/A3 units |
| R-002 FP16/BF16 | key 1/4 | 同上 | 同上 | `foreach_round_off_number` key 1/4 | `ForeachRoundOffNumberND<half/bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 从 TensorList 描述符解析每个 Tensor 的 GM 地址；每核根据 POD 覆盖一个跨 Tensor 的连续元素区间。对每个相交 Tensor，以 `maxDataCount` 分 tile，`CopyIn` 入 VECIN，计算后经 VECOUT 写回对应输出位置。单深度 Queue 的 `EnQue/DeQue/FreeTensor` 管理 GM↔Vector 流水，Kernel 不使用 workspace。

### Route R-001：FP32 直算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_round_off_number.cpp` | `foreach_round_off_number` | `TILING_KEY_IS(2)` | `ForeachRoundOffNumberND<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_round_off_number(x, roundMode, y, workspace, tiling)
  -> ForeachRoundOffNumberND<float>::Init
  -> ForeachRoundOffNumberND<float>::Process
    -> SingleTensorProcess
      -> CopyIn
      -> ComputeAndCopyOut
        -> InnerComputer<float>::Compute
```

##### 源码等价伪码

```text
block = GetBlockIdx()
[tensorStart, tensorEnd, firstOffset, lastOffset] = POD[block]
mode = int8(roundMode[0])
for tensor k in inclusive range:
  lo = (k == tensorStart ? firstOffset : 0)
  hi = (k == tensorEnd ? lastOffset : numel[k]-1)
  bind xGM/yGM at tensor[k] + lo
  for each full B/4-element tile, then optional tail:
    GM -> inputQueue
    if mode == 7: out = x - trunc(x)
    else: out = Cast<float>(x, ConvertRoundMode(mode))
    outputQueue -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `tensorAddr[k]+lo+tile*B/4` | full `B/4` 元素或 tail | FP32/ND | caller/Contiguous | `CopyIn` | caller/executor | full tile `DataCopy`；tail 按有效字节 |
| `dataQueue` | Queue 私有 | `B` 字节，depth 1 | FP32 | `CopyIn` | `ComputeAndCopyOut` | 当前 tile | Free 后复用 |
| `outQueue` | Queue 私有 | `B` 字节，depth 1 | FP32 | Compute | `CopyOut` | 当前 tile | Free 后复用 |
| y GM | 同 x 的逻辑位置 | 有效元素数 | FP32/ND | `CopyOut` | caller | caller | tail 不写无效元素 |
| workspace | N/A | body 为 0 | N/A | N/A | N/A | executor 仍可保留 32B | `userWS=nullptr` |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · B bytes"]:1
  outQ["outQueue · depth=1 · B bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = B + B = 2B
UB_total <= U - POD_reserved（由 Host 的 C/P 预算保证）
dynamic_ub = N/A
workspace_total(kernel body) = 0；Tiling 声明 32B
=> maxDataCount = B / 4 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` | `CopyIn` | `InnerComputer<float>` | 下一 tile `AllocTensor` | `EnQue→DeQue→FreeTensor` | depth 1 必须先释放 |
| `outQueue` | `InnerComputer<float>` | GM `DataCopy` | 下一 tile | `EnQue→DeQue→FreeTensor` | 写回完成后释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V | Queue | `DataCopy/DataCopyPad` | `DeQue` | input tile | tail 仅搬 `dataCount*4` 字节 |
| frac | `PipeBarrier<PIPE_V>` | trunc Cast | `Sub` | `outLocal` | FP32 原精度 |
| V→GM | Queue | Compute `EnQue` | CopyOut `DeQue` | output tile | tail 仅写有效字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling/DivideUbMemory9` | key、blockDim、B、32B workspace |
| Kernel entry/router | `op_kernel/foreach_round_off_number.cpp` | `foreach_round_off_number` key 2 | 路由到 FP32 模板 |
| Device body/helper | `op_kernel/foreach_round_off_number.h` | `ForeachRoundOffNumberND<float>`、`InnerComputer<float>` | 分核、tile、舍入、tail |

### Route R-002：FP16/BF16 升精度分段计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | FP16→1，BF16→4 | 对应 key |
| Router | `op_kernel/foreach_round_off_number.cpp` | `foreach_round_off_number` | key 1；key 4 + BF16 编译 guard | `ND<half/bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_round_off_number
  -> ForeachRoundOffNumberND<half|bfloat16_t>::Init
  -> Process -> SingleTensorProcess
    -> CopyIn
    -> ComputeAndCopyOut
      -> InnerComputer<T>::Compute
        -> ComputerPerCast
```

##### 源码等价伪码

```text
tileCapacity = 9B / sizeof(T)
castCapacity = B / sizeof(float)
for each GM tile:
  GM -> dataQueue
  for each castCapacity chunk and final chunk:
    f0 = Cast<float>(T_input, CAST_NONE)
    if mode == 7: f1 = f0 - Cast<float>(f0, CAST_TRUNC)
    else: f1 = Cast<float>(f0, ConvertRoundMode(mode))
    T_out = Cast<T>(f1, CAST_RINT)
  outQueue -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | TensorList 地址 + core/tile 元素偏移 | tile `<=9B/2` 元素 | FP16/BF16 ND | caller/Kernel | Kernel/caller | caller | tail 有效字节搬运 |
| `dataQueue` | Queue 私有 | `9B` 字节，depth 1 | T | `CopyIn` | cast | 当前 tile | Free 后复用 |
| `outQueue` | Queue 私有 | `9B` 字节，depth 1 | T | final cast | `CopyOut` | 当前 tile | Free 后复用 |
| `float32Queue` 子区 0 | 起点 | `B` 字节 | FP32 | T→FP32 cast | round/sub | Kernel 全生命周期 | 与子区1同一 Queue 不重叠 |
| `float32Queue` 子区 1 | `B` 字节偏移 | `B` 字节 | FP32 | round/trunc/sub | FP32→T cast | Kernel 全生命周期 | 下一 cast chunk 覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=1 · 9B"]:1
  f0["float32Queue[0] · B"]:1
  f1["float32Queue[B/4] · B"]:1
  outQ["outQueue · depth=1 · 9B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 9B + 9B + 2B = 20B
UB_total <= U - POD_reserved（Host 以 /10 衰减并对齐 B）
dynamic_ub = N/A
workspace_total(kernel body) = 0；Tiling 声明 32B
=> maxDataCount=9B/2，maxCastDataCount=B/4 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 子区0 | 升精度 Cast | round/trunc/Sub | 下一 chunk Cast | 连续 `PIPE_V` barrier | 当前计算完成 |
| FP32 子区1 | round/trunc/Sub | 降精度 Cast | 下一 chunk | `PIPE_V` barrier | 不得在降精度前覆盖 |
| Queue | Copy/Compute | 下一阶段 | 下一 tile | Queue 生命周期 | depth 1 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| T→FP32 | `Cast(...CAST_NONE)` + barrier | input Queue | round | FP32 子区0 | 升精度 |
| round | `Cast` 或 `trunc+Sub` + barrier | 子区0 | 降精度 | 子区1 | 模式来自 int8 GM 标量 |
| FP32→T | `Cast(...CAST_RINT)` + barrier | 子区1 | output Queue | T output | 最终 ties-to-even 降精度 |
| GM tail | `DataCopyPad` | GM/UB | 下一阶段 | 有效 tail | blockLen=`dataCount*sizeof(T)` 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory9` | FP16/BF16 的 `/10` 与 64B 对齐 |
| Kernel entry/router | `op_kernel/foreach_round_off_number.cpp` | key 1/key 4 branch | dtype 与模板映射、BF16 guard |
| Device body/helper | `op_kernel/foreach_round_off_number.h` | `InnerComputer<T>::ComputerPerCast` | 双 FP32 子区、barrier、cast 顺序 |

## 算子特定优化

### Trick T-001：FP16/BF16 大搬运 tile 内再切 FP32 舍入片段

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`T` 为 FP16/BF16，Host 已按 `/10` 预算 `B`
- Golden 来源：[`foreach_round_off_number.h:207-217`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/foreach_round_off_number.h#L207-L217)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
#if __CCE_AICORE__ >= 220
    if (std::is_same<T, bfloat16_t>::value) {
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
        LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
        float32Queue.EnQue(float32Tensor);
        maxCastDataCount = inputsTensorUbSize / sizeof(float);
    }
```

#### 机制与设计原因

GM tile 容量扩大为 `9B`，但 FP32 临时只保留两个各 `B` 的片段；`InnerComputer<T>` 以 `maxCastDataCount=B/4` 分段复用两片 FP32 空间。这样不必为整个 `9B/2` 元素 tile 同时物化 FP32 输入与输出，源码可证明降低了临时 FP32 容量并保持较大的 GM tile。

#### 迁移边界

- 适用：需要 FP32 中间计算、输入输出仍为 16 位的逐元素算子。
- 前提：每个分段之间有 V pipe barrier，最终降精度完成后才能覆盖 FP32 子区。
- 失效：若计算跨分段依赖、需要更多 FP32 中间量或异步重叠，必须重算 buffer 比例和同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16/BF16 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 2 | key 1/4 | dtype 唯一决定 |
| 模板与 dtype | `ND<float>` | `ND<half/bfloat16_t>` | 16 位需 FP32 中间 |
| 循环与分核 | 核→Tensor→tile | 核→Tensor→GM tile→cast chunk | 临时容量不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 两个 B Queue | 两个 9B Queue + 一个 2B FP32 Queue | UB 总量 2B vs 20B |
| 内存总占用与复用 | 2B | 20B，FP32 双子区跨 chunk 复用 | Host 对 16 位 B 做 `/10` |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Cast/Sub | DataCopy(Pad)+三段 Cast/Sub | 均为 AIV Vector |
| 同步、精度与 tail | FP32 原精度 | FP32 舍入后 CAST_RINT 降精度 | tail 均按有效字节 |
| fallback/失败边界 | 未知 key 无 body | BF16 guard 不满足无 body | 无备用 Kernel |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 V2 | 支持 V2 | 支持，但见独立 `_950` 卡 |
| Host/Launcher | arch22 common tiling | 与 A2 共用 arch22 | arch35 自定义 tiling |
| Kernel/资源 | AIV Queue/UB | 与 A2 共用 | SIMT，无 AIV Queue |

## 限制与失败边界

- V2 对 `INT64` scalar 校验通过后仍转换为 `DT_INT8` Tensor；超出 INT8 可表示范围的转换细则在固定仓框架边界不可见。
- ACLNN 实现的内部 dtype 列表还包含 `DT_INT32`，但公开文档、OpDef、binary config 与 Kernel router 均未发布 INT32 路线；它不属于本卡公开可达 Route，绕过公开约束时会在下游失败或无 body。
- 公开文档只定义 1–6 与“其它值”语义；A2/A3 body 另把值 7 实现为小数部分，这与公开“其它值”描述不一致，复用时必须按实际 body 处理。
- API CheckShape 要求对应输入输出 shape 相等；common Tiling 的更宽松“输出元素数不少于输入”分支不改变 ACLNN 可达域。
- Tiling 固定数组最多表达 50 个 Tensor/50 个核；公开 V2 文档未声明 TensorList 数量上限。
- 固定仓未展开 executor/Contiguous 的内部 workspace offset 与实现；本次知识提取未执行运行验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_round_off_number/docs/aclnnForeachRoundOffNumberV2.md` | V2 原型/产品支持 | API、平台、dtype/shape |
| 入口实现 | `foreach/foreach_round_off_number/op_host/op_api/aclnn_foreach_round_off_number_v2.cpp` | `ExecForeachRoundOffNumberV2GetWorkspaceSize` | 校验、Contiguous、scalar 转 Tensor、executor |
| L0 launcher | `foreach/foreach_round_off_number/op_host/op_api/foreach_round_off_number_v2.cpp` | `ForeachRoundOffNumberV2` | 单 AICore launcher |
| Host/Tiling | `foreach/foreach_round_off_number/op_host/arch22/foreach_round_off_number_tiling_arch22.cpp` | `Tiling4ForeachRoundOffNumberTiling` | common tiling 入口 |
| 平台注册 | `foreach/foreach_round_off_number/op_host/foreach_round_off_number_def.cpp` | `ForeachRoundOffNumber` | A2/A3 OpDef 与 dtype |
| Host 构建 | `foreach/foreach_round_off_number/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3→arch22 |
| binary config | `foreach/foreach_round_off_number/op_host/config/ascend910b/foreach_round_off_number_binary.json` | `op_list` | A2 dtype 实例 |
| binary config | `foreach/foreach_round_off_number/op_host/config/ascend910_93/foreach_round_off_number_binary.json` | `op_list` | A3 dtype 实例 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling` 等 | 分核、UB、POD、workspace |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | ABI 字段 |
| Kernel entry | `foreach/foreach_round_off_number/op_kernel/foreach_round_off_number.cpp` | `foreach_round_off_number` | key router |
| Kernel body | `foreach/foreach_round_off_number/op_kernel/foreach_round_off_number.h` | `ForeachRoundOffNumberND/InnerComputer` | Vector 管线、UB、同步、tail |
| 构建接线 | `foreach/foreach_round_off_number/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 使用默认 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | common foreach 分核 POD | 六字段顺序、数组上限与单位 | `ForeachCommonTilingData` |
| 调用顺序 | Contiguous→scalar Tensor→单 Kernel | 前序失败必须阻断 launcher | `ExecForeachRoundOffNumberV2GetWorkspaceSize` |
| workspace/内存 | Kernel body 零 workspace；Queue 预算公式 | 16 位 Queue 的 `9B+9B+2B` | `DivideUbMemory9/Init` |
| 同步 | Queue 生命周期与 V barrier | FP32 子区覆盖前必须完成消费者 | `ComputerPerCast` |
| 精度与尾块 | 16 位升 FP32 舍入再降精度 | 降精度固定 `CAST_RINT`；tail 按有效字节 | `InnerComputer<T>` |
| 失败边界 | dtype/shape/format/编译 guard | 未知 key 与 BF16 guard 无 fallback | `CheckParams/foreach_round_off_number` |