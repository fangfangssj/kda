---
type: Operator
title: "AmpUpdateScale 950 ACLNN 算子知识"
description: "说明 aclnnAmpUpdateScale 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/docs/aclnnAmpUpdateScale.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/amp_update_scale_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/amp_update_scale_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/amp_update_scale_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_kernel/amp_update_scale.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_kernel/amp_update_scale.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/config/ascend950/amp_update_scale_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T10:04:08Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# AmpUpdateScale 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAmpUpdateScaleGetWorkspaceSize` + `aclnnAmpUpdateScale`，固定仓唯一公开版本，ACLNN |
| 计算语义 | foundInf 非零时回退 scale；成功计数达到 interval 时在有限值保护下增长 scale；更新 tracker |
| 输入输出 | 三个 `[1]` ND 输入、两个 `[1]` ND 输出；scale 族同为 FLOAT/FLOAT16/BFLOAT16，tracker 为 INT32 |
| 执行类型 | 单节点 `amp_update_scale`，AIV_VECTOR；BF16 使用 Vector `Cast`；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 FP32，R-002 FP16，R-003 BF16 |
| 核心资源 | FP32/FP16：5×32 B TBuf；BF16：6×32 B TBuf；无 TQue，body 不读 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→OpDef/Tiling/POD→dtype router→三个 body→ascend950 config 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

Device 将 `currentScale[0]` 与 `foundInf[0]` 统一为 FP32。若 `foundInf != 0`，输出 `scale*backoffFactor, tracker=0`；否则令 `successful=tracker+1`，当其等于 `growthInterval` 时计算 `newScale=scale*growthFactor`，仅在 FP32 指数域不全为 1 时接受该值，但无论是否有限都清零 tracker；未达 interval 时 scale 不变、tracker 写 `successful`。最后将 scale 转回输入 dtype。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `currentScale` | `[1]`/ND | FLOAT/FLOAT16/BFLOAT16 | 必选；非连续 Tensor 不支持 |
| `growthTracker` | `[1]`/ND | INT32 | 必选，成功步数 |
| `foundInf` | `[1]`/ND | 与 currentScale 相同 | 必选；非零优先回退 |
| `growthFactor` | 属性标量 | API double；Host/POD float | 必选，无可见范围/默认值 |
| `backoffFactor` | 属性标量 | API double；Host/POD float | 必选，无可见范围/默认值 |
| `growthInterval` | 属性标量 | API int64_t；POD int32_t | `[1,INT32_MAX]` 后窄化 |
| `updatedScale` | `[1]`/ND | 与 currentScale 相同 | 必选输出 |
| `updatedGrowthTracker` | `[1]`/ND | INT32 | 必选输出 |

### 入口与版本

固定仓只有该两段式公开 API，无 V2/V3。公开产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，其 binary config 同时列出 FP32/FP16/BF16 三种匹配实例，平台证据无冲突。自动生成的 ACLNN bridge/executor 未公开，作为框架边界；算子自有 OpDef、Tiling、POD、router/body 和构建配置均可见。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段文档规定五个 Tensor 非空、scale dtype 属于三种、foundInf/updatedScale 与 currentScale 同 dtype、growthTracker 为 INT32、所有输入输出均 `[1]`。可见 `AmpUpdateScaleTiling::Init` 再取得三个属性，拒绝空指针，校验 `growthIntervalRaw` 后窄化，并逐个检查三个输入 storage shape 的 dimNum 和 size 都为 1。输出 shape/dtype 的实际 bridge 校验代码固定仓不可见。接口不支持非连续 Tensor，没有算子自有 Contiguous/ViewCopy。

### 参数、Workspace 与生命周期

Host 把两个 float 属性和已校验的 int32 interval 写入 POD，固定 `blockDim=1`，不计算 tile/loop/workspace。第一段所返回 workspace 大小的框架公式固定仓不可见；调用者负责内存生命期。kernel 参数含 `workspace` 但 body 不读取，故算子自有 Device 节点使用 0 B workspace。executor 在第二段随调用者 stream 提交。

### 启动图与失败传播

```text
aclnnAmpUpdateScaleGetWorkspaceSize
  -> TilingAmpUpdateScale -> Init -> RunKernelTiling
  -> K-001 amp_update_scale(blockDim=1, AmpUpdateScaleTilingData)
       GM inputs -> AIV/Vector body -> GM outputs
```

只有一个 Device 节点，无跨节点 workspace/同步边。非法属性、interval 或三个输入 shape 令 Tiling 返回 `GRAPH_FAILED` 并阻断 launch。未知 key 的 entry 没有 else，静默不写输出；公开校验和 ascend950 config 使其不可达。executor 提交失败的传播细节固定仓不可见。

## Route 与参数设计

### 分核与资源计算

数据域固定 1 元素，`blockDim=1`，Device 无 block/tile/loop。每个 TBuf 固定 32 B，逻辑访问只到元素 0。FP32/FP16 总 UB 为 `5*32=160 B`；BF16 增加 `castBuf_`，为 `6*32=192 B`。没有基于 950 UB 容量的 selector，平台可用 UB 上限固定仓不可见。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 | 0/`AmpUpdateScale<float>` | `dtype==DT_FLOAT` | R-002 检查 |
| 2 | R-002 FP16 | 1/`AmpUpdateScale<half>` | 前项未命中且 `dtype==DT_FLOAT16` | R-003 检查 |
| 3 | R-003 BF16 | 2/`AmpUpdateScale<bfloat16_t>` | 前两项未命中且 `dtype==DT_BF16` | 内部绕过时 key 保持初值 0；公开 API 拒绝 |

```text
FLOAT -> key0; else FLOAT16 -> key1; else BF16 -> key2; else -> initial key0
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | FLOAT、三个输入 `[1]`、ascend950 | 单 K-001 | `AmpUpdateScale<float>::Compute` | 160 B；hard event | 4 B DataCopyPad，无循环 tail |
| R-002 FP16 | 1 | FLOAT16、三个输入 `[1]`、ascend950 | 单 K-001 | `AmpUpdateScale<half>::Compute` | 160 B；FP32 中间 | 2/4 B 搬运，无循环 tail |
| R-003 BF16 | 2 | BF16、三个输入 `[1]`、ascend950 | 单 K-001 | `AmpUpdateScale<bfloat16_t>::Compute` | 192 B；Cast/PIPE_V/event | 2/4 B 搬运，无循环 tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype | 三路 `if/else if` | 模板决定宽度/cast/UB | 全部 | 公开层报错 | `RunKernelTiling` |
| 标量 shape | `dimNum==1 && size==1` | body 只访问 index 0 | 全部 | `GRAPH_FAILED` | `Init` |
| interval | `[1,INT32_MAX]` | int64_t→int32_t ABI | 全部 | `GRAPH_FAILED` | `Init` |
| BF16 buffer | `if constexpr BF16` | Vector Cast 需 FP32 local | R-003 | 其他模板不分配 | `AmpUpdateScale::Init` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AmpUpdateScaleTilingData` | `growthFactor` | float/32 | 声明第 1 字段；精确 offset/sizeof/alignof 固定仓不可见 | multiplier | `set_growthFactor` | entry→`growthFactor_` | API double→float bridge 细节不可见 |
| 同上 | `backoffFactor` | float/32 | 声明第 2 字段；精确 offset 不可见 | multiplier | `set_backoffFactor` | entry→`backoffFactor_` | 无范围校验 |
| 同上 | `growthInterval` | int32_t/32 | 声明第 3 字段；精确 offset 不可见 | step | `set_growthInterval` | entry→`growthInterval_` | 先校验再窄化 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | `amp_update_scale_tiling.cpp::RunKernelTiling` | `AmpUpdateScaleTilingData` | entry key0 | `AmpUpdateScale<float>` | ascend950 json + op_host CMake |
| R-002 FP16 | 1 | 同上 | 同上 | entry key1 | `AmpUpdateScale<half>` | 同上 |
| R-003 BF16 | 2 | 同上 | 同上 | entry key2 | `AmpUpdateScale<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 创建 TPipe、解析 POD、按 key 实例化模板。`Init` 绑定五个 extent=1 的 GlobalTensor，分配基础 TBuf；`Process→Compute` 严格调用 `LoadInputData→ComputeScaleUpdate→StoreOutputData`。每次 `DataCopyPad` 的 blockCount=1、blockLen=`sizeof(T)` 或 4 B、两端 stride=0。所有乘法、分支和有限性判断均在 FP32 标量成员执行；无循环、Atomic、跨核通信或 workspace。

### Route R-001：FP32 标量直通

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | DT_FLOAT | key0/blockDim1 |
| Router | `op_kernel/amp_update_scale.cpp` | `amp_update_scale` | `TILING_KEY_IS(0)` | float body |

#### Device 数据流

##### 函数级调用链

```text
amp_update_scale -> AmpUpdateScale<float>::Init -> Process -> Compute
  -> LoadInputData -> ComputeScaleUpdate -> StoreOutputData
```

##### 源码等价伪码

```text
copy s,t,f (各4B) GM->UB; wait MTE2_V/V_S; scalar-read
if f: backoff/reset; else if t+1==interval: grow-if-finite/reset; else increment
scalar-write outputs; wait S_V/V_MTE3; copy each 4B UB->GM; wait MTE3_MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | base+0 | 1 element | float/int32 ND | caller/MTE3 | MTE2/caller | caller | 有效 4 B |
| 五个 TBuf | 独立 UB | 各32 B | float/int32 | MTE2/scalar | scalar/MTE3 | TPipe/kernel | 无 alias |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  cs["currentScale 32B"] fi["foundInf 32B"] gt["tracker 32B"] us["updatedScale 32B"] ug["updatedTracker 32B"]
```

##### 字节账与复用顺序

```text
UB_total=160 B; dynamic_ub=N/A; workspace_total=body 0 B（ACLNN 框架值不可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | 独占 TBuf | 对应阶段 | kernel 结束 | hard event | 单次标量 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input | MTE2_V/V_S | MTE2 | scalar | input UB | FP32，4 B |
| compute/output | S_V/V_MTE3/MTE3_MTE2 | scalar/Vector | MTE3/end | output UB/GM | FP32，指数域 finite |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | key0/POD/blockDim |
| Kernel entry/router | `op_kernel/amp_update_scale.cpp` | key0 | float instance |
| Device body/helper | `op_kernel/amp_update_scale.h` | `AmpUpdateScale<float>` | 4B copy、FP32 update、events |

### Route R-002：FP16 升精度标量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | DT_FLOAT16 | key1 |
| Router | `op_kernel/amp_update_scale.cpp` | entry | key1 | half body |

#### Device 数据流

##### 函数级调用链

```text
entry -> AmpUpdateScale<half>::Compute -> Load(static_cast<float>)
  -> FP32 Update -> Store(static_cast<half>)
```

##### 源码等价伪码

```text
copy 2B s/f + 4B t; cast half->float; FP32 update; cast float->half; copy 2B scale + 4B tracker
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+0 | 1 element | half/int32 ND | caller/MTE3 | MTE2/caller | caller | 2 B/4 B |
| 五个 TBuf | 独立 | 各32 B | half/int32 | MTE2/scalar | scalar/MTE3 | TPipe | 无 alias |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  cs["currentScale 32B"] fi["foundInf 32B"] gt["tracker 32B"] us["updatedScale 32B"] ug["updatedTracker 32B"]
```

##### 字节账与复用顺序

```text
UB_total=160 B; dynamic_ub=N/A; workspace body=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | 独占 TBuf | 对应阶段 | end | hard event | 单标量 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input | MTE2_V/V_S | copy | scalar | UB | half→FP32 |
| compute/output | S_V/V_MTE3 | scalar | MTE3 | output | FP32 math→half；显式 round mode 未命名 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | key1 |
| Kernel entry/router | `op_kernel/amp_update_scale.cpp` | key1 | half instance |
| Device body/helper | `op_kernel/amp_update_scale.h` | half branches | scalar casts and FP32 math |

### Route R-003：BF16 Vector Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | DT_BF16 | key2 |
| Router | `op_kernel/amp_update_scale.cpp` | entry | key2 | bfloat16_t body |

#### Device 数据流

##### 函数级调用链

```text
entry -> AmpUpdateScale<bfloat16_t>::Init(+castBuf) -> Compute
  -> Load(Cast BF16->FP32 twice) -> FP32 Update -> Store(Cast FP32->BF16 RINT)
```

##### 源码等价伪码

```text
copy inputs; Cast s to shared castBuf and scalar-read; overwrite with Cast f and scalar-read
FP32 update; scalar-write result to castBuf; Cast_RINT to BF16 output; copy outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+0 | 1 element | BF16/int32 ND | caller/MTE3 | MTE2/caller | caller | 2 B/4 B |
| 五个基础 TBuf | 独立 | 各32 B | BF16/int32 | MTE2/scalar | Vector/scalar/MTE3 | TPipe | 无 alias |
| castBuf | 独立 | 32 B | float | Cast/scalar | scalar/Cast | TPipe | s→f→output 复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  cs["currentScale 32B"] fi["foundInf 32B"] gt["tracker 32B"] cast["castBuf 32B"] us["updatedScale 32B"] ug["updatedTracker 32B"]
```

##### 字节账与复用顺序

```text
UB_total=192 B; dynamic_ub=N/A; workspace body=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castBuf[0] | Cast(s) | V_S scalar read | Cast(f) | PIPE_V+V_S | s 已进成员 |
| castBuf[0] | Cast(f) | V_S scalar read | SetValue(output s) | PIPE_V+V_S | f 已进成员 |
| castBuf[0] | SetValue | Cast_RINT | end | S_V | Cast 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input cast | CAST_NONE/PIPE_V/V_S | MTE2/Vector | scalar | castBuf | count 1 |
| output cast | S_V/CAST_RINT/V_MTE3 | scalar/Vector | MTE3 | castBuf/output | FP32→BF16 RINT，2 B tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | key2 |
| Kernel entry/router | `op_kernel/amp_update_scale.cpp` | key2 | BF16 instance |
| Device body/helper | `op_kernel/amp_update_scale.h` | BF16 Init/Load/Store | conditional buffer, casts, reuse/events |

## 算子特定优化

### Trick T-001：按 dtype 编译期消除转换缓冲区

#### 触发与源码依据

- 适用 Route：R-003 分配，R-001/R-002 消除。
- 触发条件：`IsSameType<T, bfloat16_t>::value`。
- Golden 来源：[`math/amp_update_scale/op_kernel/amp_update_scale.h:49-58`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_kernel/amp_update_scale.h#L49-L58)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
pipe_->InitBuffer(currentScaleBuf_, BlockBytes);
pipe_->InitBuffer(growthTrackerBuf_, BlockBytes);
pipe_->InitBuffer(foundInfBuf_, BlockBytes);

pipe_->InitBuffer(updatedScaleBuf_, BlockBytes);
pipe_->InitBuffer(updatedGrowthTrackerBuf_, BlockBytes);

if constexpr (IsSameType<T, bfloat16_t>::value) {
    pipe_->InitBuffer(castBuf_, BlockBytes);
}
```

#### 机制与设计原因

FP32 不转换、FP16 使用标量 cast，因此编译期不分配 castBuf，UB 为 160 B；BF16 才分配 32 B 并通过顺序同步复用输入 s、输入 f、输出 s，避免三个转换临时量分别物化。

#### 迁移边界

- 适用：转换需求由编译期 dtype 决定且临时值生命周期不重叠。
- 前提：覆盖前保持 PIPE_V/V_S/S_V 同步。
- 失效：并发转换或流水重叠时需要独立 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | FLOAT/key0 | FP16/key1 | BF16/key2 | Host 顺序 |
| 模板与 dtype | float | half | bfloat16_t | 950 config |
| 循环与分核 | 1 block，无循环 | 同左 | 同左 | 标量 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 5 TBuf | 5 TBuf | 6 TBuf | BF16 Cast |
| 内存总占用与复用 | 160 B | 160 B | 192 B/castBuf reuse | conditional allocation |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad | DataCopyPad/scalar cast | DataCopyPad/Vector Cast | dtype |
| 同步、精度与 tail | FP32 | FP32 intermediate | FP32 + RINT | 1-element padded DMA |
| fallback/失败边界 | unknown internal key0 | none | none | public dtype guard |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持，Ascend 950PR/950DT |
| Host/Launcher | 同源码 | 同源码 | `ascend950` OpDef/config，Host 逻辑同源码 |
| Kernel/资源 | 同 body | 同 body | 三 dtype binary；同一 AIV body 与资源公式 |

## 限制与失败边界

- 仅 `[1]` ND 标量且不支持非连续 Tensor；可见 Tiling 只复查三个输入 shape，输出校验实现位于生成 bridge 边界。
- interval 限 `[1,INT32_MAX]`；输入 tracker 的 `+1` 为 int32_t，固定仓无独立溢出检查。
- growth/backoff factor 无可见范围或有限性校验；只有增长结果做 FP32 Inf/NaN 保护。
- FP16 降精度显式 round mode不可见；BF16 明确 `CAST_RINT`。
- ACLNN bridge/executor、精确 workspace size、Tiling 宏的 offset/sizeof/alignof、950 UB 上限固定仓不可见。
- unknown key 无错误分支，不能作为可复用 fallback。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/amp_update_scale/docs/aclnnAmpUpdateScale.md` | 两段式 API/产品表 | 公开入口、950 支持、约束 |
| 入口实现 | `math/amp_update_scale/op_host/amp_update_scale_def.cpp` | `ops::AmpUpdateScale` | 三 dtype、ascend950 注册 |
| Host/Tiling/Launcher | `math/amp_update_scale/op_host/amp_update_scale_tiling.h` | POD | 三字段 |
| Host/Tiling/Launcher | `math/amp_update_scale/op_host/amp_update_scale_tiling.cpp` | `Init/RunKernelTiling` | 校验、key、blockDim、POD |
| Kernel entry/body | `math/amp_update_scale/op_kernel/amp_update_scale.cpp` | `amp_update_scale` | router |
| Kernel entry/body | `math/amp_update_scale/op_kernel/amp_update_scale.h` | `AmpUpdateScale<T>` | 完整 Device body |
| 构建接线 | `math/amp_update_scale/op_host/config/ascend950/amp_update_scale_binary.json` | op_list | 950 三实例 |
| 构建接线 | `math/amp_update_scale/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN 模块接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 两 float + 一 int32 属性 | 字段顺序/位宽，interval 先校验 | `AmpUpdateScaleTilingData` |
| 调用顺序 | Load→Update→Store | foundInf 优先，增长命中清零 tracker | `ComputeScaleUpdate` |
| workspace/内存 | 5×32 B，BF16 加 castBuf | castBuf 生命周期；workspace body 未读 | `Init/Load/Store` |
| 同步 | hard events + PIPE_V | 覆盖/读取顺序 | `Sync*`/BF16 branches |
| 精度与尾块 | FP32 math、BF16 RINT | bit-level finite、单元素有效字节 | `IsFinite/StoreOutputData` |
| 失败边界 | Host 拒绝非法输入 | 不依赖 unknown key | `Init/RunKernelTiling/entry` |