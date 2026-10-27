---
type: Operator
title: "AmpUpdateScale A2/A3 ACLNN 算子知识"
description: "说明 aclnnAmpUpdateScale 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/config/ascend910b/amp_update_scale_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/config/ascend910_93/amp_update_scale_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/amp_update_scale/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T10:04:08Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# AmpUpdateScale A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAmpUpdateScaleGetWorkspaceSize` + `aclnnAmpUpdateScale`，固定仓唯一公开版本，入口类型 ACLNN |
| 计算语义 | 发现 Inf/NaN 时回退 scale；达到 growth interval 时在有限值保护下增长 scale；同步更新 tracker |
| 输入输出 | 三个 `[1]` ND 输入与两个 `[1]` ND 输出；scale/foundInf/updatedScale 为同一 `FLOAT/FLOAT16/BFLOAT16`，tracker 为 `INT32` |
| 执行类型 | 单 Kernel：`amp_update_scale`，AIV_VECTOR；BF16 实例使用 Vector `Cast`，无 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 FP32，R-002 FP16，R-003 BF16 |
| 核心资源 | 5 个 32 B `TBuf`；BF16 额外 1 个 32 B `castBuf_`；无 `TQue`，kernel 不读 `workspace` |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→OpDef/Host Tiling→POD→dtype router→三个模板实例→body→A2/A3 binary config 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `s=currentScale[0]`、`t=growthTracker[0]`、`f=foundInf[0]`。Device 先把 `s/f` 转为 FP32，再按下列顺序执行：若 `f != 0`，则 `s'=s*backoffFactor,t'=0`；否则令 `u=t+1`，若 `u==growthInterval`，计算 `n=s*growthFactor`，仅当 `n` 的 FP32 指数域不是全 1 时写 `s'=n`，并令 `t'=0`；其余情况 `s'=s,t'=u`。输出 scale 再转换到入口 dtype。`foundInf` 分支优先于增长分支。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `currentScale` | `[1]` / ND | FLOAT/FLOAT16/BFLOAT16 | 必选；当前 loss scale；非连续 Tensor 不支持 |
| `growthTracker` | `[1]` / ND | INT32 | 必选；连续成功步数 |
| `foundInf` | `[1]` / ND | 与 `currentScale` 相同 | 必选；零为正常，任意非零值进入回退分支 |
| `growthFactor` | 标量属性 | API 为 `double`，OpDef/Tiling 为 float | 必选；固定仓未声明数值范围或默认值 |
| `backoffFactor` | 标量属性 | API 为 `double`，OpDef/Tiling 为 float | 必选；固定仓未声明数值范围或默认值 |
| `growthInterval` | 标量属性 | API 为 int64_t，POD 为 int32_t | 必选；Host 先校验 `[1, INT32_MAX]` 再窄化 |
| `updatedScale` | `[1]` / ND | 与 `currentScale` 相同 | 必选输出；更新后的 scale |
| `updatedGrowthTracker` | `[1]` / ND | INT32 | 必选输出；更新后的 tracker |

### 入口与版本

固定仓只公开 `aclnnAmpUpdateScale` 两段式接口，没有同语义 V2/V3。第一段返回 `workspaceSize` 和 executor，第二段在给定 stream 上执行。ACLNN 自动生成 bridge/executor 实现未随固定仓公开，作为框架边界；公开文档、`AmpUpdateScale` OpDef、Tiling 注册及 kernel 均可见。文档明确支持 Atlas A2/A3，OpDef 分别注册 `ascend910b`、`ascend910_93`，两平台 binary config 都列出相同三个 dtype 实例，无发布冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

公开 ACLNN 第一段校验五个 Tensor 非空、`currentScale` dtype 属于三种支持类型、`foundInf/updatedScale` 与其同 dtype、`growthTracker` 为 INT32；文档还约束所有输入输出为 `[1]`。可见 Tiling `Init` 重新取得三个属性，拒绝空属性；对 `growthIntervalRaw` 做 `[1,INT32_MAX]` 校验；逐一检查三个输入 storage shape 的维数为 1 且元素数为 1。输出 shape 检查和 ACLNN dtype 匹配实现位于未公开的生成 bridge 边界。接口明确不支持非连续 Tensor，因此没有算子自有 `Contiguous/ViewCopy` 节点。

### 参数、Workspace 与生命周期

`growthFactor/backoffFactor` 从属性 float 指针读取；`growthInterval` 先以 int64_t 读取并安全窄化为 int32_t。Tiling 固定 `blockDim=1`，不计算 tile/loop/workspace。公开第一段的精确 `workspaceSize` 生成公式在固定仓不可见；调用者拥有该内存直到第二段完成。Device entry 接收 `workspace` 参数但从未读取，算子自有语义节点没有 workspace slice。executor 由第一段创建并携带 launch，第二段在调用者 stream 上提交。

### 启动图与失败传播

```text
aclnnAmpUpdateScaleGetWorkspaceSize (生成 bridge，固定仓不可见)
  -> AmpUpdateScale OpDef / TilingAmpUpdateScale
  -> 单节点 K-001: amp_update_scale(blockDim=1, POD=AmpUpdateScaleTilingData)
       inputs: current_scale, growth_tracker, found_inf
       outputs: updated_scale, updated_growth_tracker
       workspace: 参数存在但 body 未读
```

仅一个 Device 节点，无跨 Kernel 边。Host Tiling 任一空属性、非法 interval 或非法输入 shape 返回 `GRAPH_FAILED`，阻断 launch。公开 ACLNN 参数错误在第一段返回相应 ACLNN 错误。entry 遇未知 key 没有 `else`、不写输出且无 Device 错误上报；公开 dtype/配置使 key 仅为 0/1/2。第二段的 stream 提交错误传播由固定仓不可见的 ACLNN executor 框架处理。

## Route 与参数设计

### 分核与资源计算

数据域恒为一个标量，Host 固定 `blockDim=1`；Device 没有 `GetBlockIdx`、tile 或循环。每个输入/输出 `TBuf` 的物理容量均为 32 B，逻辑有效 extent 为一个元素。FP32/FP16 的 `UB_total=5*32=160 B`；BF16 条件增加 32 B `castBuf_`，`UB_total=192 B`。固定仓未暴露平台可用 UB 数值，但该算子不按 UB 做 Route 选择。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 | key 0 / `AmpUpdateScale<float>` | `currentScaleDtype == ge::DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16 | key 1 / `AmpUpdateScale<half>` | 前项未命中且 `currentScaleDtype == ge::DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BF16 | key 2 / `AmpUpdateScale<bfloat16_t>` | 前两项未命中且 `currentScaleDtype == ge::DT_BF16` | Tiling 层保留默认 key 0；公开入口先拒绝该 dtype |

```text
if dtype == FLOAT: key=0
else if dtype == FLOAT16: key=1
else if dtype == BF16: key=2
else: key 保持 0（仅内部绕过公开校验时可见）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | FLOAT、三个输入 `[1]`、A2/A3 config | 单节点 K-001 | `AmpUpdateScale<float>::Compute` | 160 B TBuf；MTE2→V→S→V→MTE3 | `DataCopyPad` 搬 4 B；无循环 tail |
| R-002 FP16 | 1 | FLOAT16、三个输入 `[1]`、A2/A3 config | 单节点 K-001 | `AmpUpdateScale<half>::Compute` | 160 B；标量升 FP32、再降 FP16 | `DataCopyPad` 搬 2 B；无循环 tail |
| R-003 BF16 | 2 | BFLOAT16、三个输入 `[1]`、A2/A3 config | 单节点 K-001 | `AmpUpdateScale<bfloat16_t>::Compute` | 192 B；复用 castBuf，Vector Cast + barrier/event | `DataCopyPad` 搬 2 B；无循环 tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype guard | 三路显式 `if/else if` | 模板决定 GM 元素宽度、cast 和 UB | 全部 | 公开 ACLNN 报参数无效；内部绕过会错误落 key 0 | `AmpUpdateScaleTiling::RunKernelTiling` |
| 标量 shape | `dimNum==1 && shapeSize==1`，逐三个输入 | body 只绑定并访问元素 0 | 全部 | `GRAPH_FAILED` | `AmpUpdateScaleTiling::Init` |
| interval 窄化 | `1 <= raw <= INT32_MAX` | POD 与 Device 成员均为 int32_t | 全部 | `GRAPH_FAILED` | `AmpUpdateScaleTiling::Init` |
| BF16 临时区 | `if constexpr IsSameType<T,bfloat16_t>` | BF16 标量路径依赖 FP32 Vector Cast | R-003 | 其他 dtype 编译期不分配 | `AmpUpdateScale::Init` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AmpUpdateScaleTilingData` | `growthFactor` | float/32 | 声明第 1 字段；宏展开后的 offsetof/sizeof/alignof 固定仓不可见 | 无量纲乘数 | `set_growthFactor(growthFactor)` | entry 传入 `Init`→`growthFactor_` | API double 到 Host float 的 bridge 转换细节不可见 |
| `AmpUpdateScaleTilingData` | `backoffFactor` | float/32 | 声明第 2 字段；精确 offset 固定仓不可见 | 无量纲乘数 | `set_backoffFactor(backoffFactor)` | entry→`backoffFactor_` | 任意 float 值，固定仓无范围校验 |
| `AmpUpdateScaleTilingData` | `growthInterval` | int32_t/32 | 声明第 3 字段；精确 offset 固定仓不可见 | step | `set_growthInterval(static_cast<int32_t>(raw))` | entry→`growthInterval_` | raw 必须在 `[1,INT32_MAX]` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | `amp_update_scale_tiling.cpp::RunKernelTiling` | `AmpUpdateScaleTilingData` | `amp_update_scale.cpp::amp_update_scale` key 0 | `amp_update_scale.h::AmpUpdateScale<float>` | A2/A3 binary json + `op_host/CMakeLists.txt` |
| R-002 FP16 | 1 | 同上 | 同上 | entry key 1 | `AmpUpdateScale<half>` | 同上 |
| R-003 BF16 | 2 | 同上 | 同上 | entry key 2 | `AmpUpdateScale<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 创建一个 `TPipe`，解析 POD，按 key 构造唯一模板实例并调用 `Init→Process→Compute`。`Init` 将五个 GM tensor 各绑定到 extent 1，保存属性，分配 32 B TBuf。`Compute` 顺序固定为 `LoadInputData→ComputeScaleUpdate→StoreOutputData`。输入使用 blockCount=1、blockLen=`sizeof(T)` 或 4 B、stride=0 的 `DataCopyPad`；计算在 FP32 标量上完成；输出再用同样的有效字节数写回。无循环、无跨核归约、无 Atomic、无 workspace 交接。

### Route R-001：FP32 标量直通

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | `dtype==DT_FLOAT` | key 0，blockDim 1 |
| Router | `op_kernel/amp_update_scale.cpp` | `amp_update_scale` | `TILING_KEY_IS(0)` | `AmpUpdateScale<float>` |

#### Device 数据流

##### 函数级调用链

```text
amp_update_scale(..., tiling) -> AmpUpdateScale<float>::Init -> Process
  -> Compute -> LoadInputData -> ComputeScaleUpdate -> StoreOutputData
```

##### 源码等价伪码

```text
copy 4 B currentScale, 4 B tracker, 4 B foundInf GM->UB; wait MTE2_V
scalar-read s, f, t
if f != 0: s *= backoff; t=0
else if (t+1)==interval: n=s*growth; if finite(n): s=n; t=0
else: t=t+1
scalar-write s,t to UB; wait S_V then V_MTE3
copy 4 B s and 4 B t UB->GM; wait MTE3_MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三个输入 GM | base+0 | 1 element | float,float,int32 / ND | caller | `LoadInputData` | caller | 有效 4 B，DataCopyPad |
| `currentScaleBuf_/foundInfBuf_/growthTrackerBuf_` | 各独立 UB 区 | 各 32 B | 对应 dtype | MTE2 | scalar read | TPipe，整个 kernel | 无 alias |
| 两个输出 TBuf | 各独立 UB 区 | 各 32 B | float/int32 | scalar write | MTE3 | TPipe | 无 alias |
| 两个输出 GM | base+0 | 1 element | float/int32 / ND | MTE3 | caller | caller | 各有效 4 B |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  cs["currentScaleBuf · 32B"] fi["foundInfBuf · 32B"] gt["growthTrackerBuf · 32B"] us["updatedScaleBuf · 32B"] ug["updatedTrackerBuf · 32B"]
```

##### 字节账与复用顺序

```text
UB_total = 5 * 32 B = 160 B
dynamic_ub = N/A
workspace_total = 算子 body 使用 0 B；ACLNN 框架返回值固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | 五个 TBuf 均独占 | 对应阶段 | kernel 结束 | hard event 保序 | 单次执行、无循环 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | `MTE2_V` + `V_S` | DataCopyPad | scalar GetValue | 三输入 UB | FP32 原值；有效 4 B |
| 计算 | 标量顺序 | scalar read | scalar write | 成员变量 | FP32 乘法；指数域全 1 判 Inf/NaN |
| 输出 | `S_V`、`V_MTE3`、`MTE3_MTE2` | SetValue | DataCopyPad/结束 | 两输出 UB/GM | FP32；有效 4 B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/amp_update_scale_tiling.cpp` | `Init/RunKernelTiling` | 校验、key 0、blockDim 1、POD |
| Kernel entry/router | `op_kernel/amp_update_scale.cpp` | `amp_update_scale` key 0 | 映射 float 模板 |
| Device body/helper | `op_kernel/amp_update_scale.h` | `AmpUpdateScale<float>::Compute` | 4 B 搬运、FP32 标量计算、hard event |

### Route R-002：FP16 升精度标量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | `dtype==DT_FLOAT16` | key 1 |
| Router | `op_kernel/amp_update_scale.cpp` | `amp_update_scale` | `TILING_KEY_IS(1)` | `AmpUpdateScale<half>` |

#### Device 数据流

##### 函数级调用链

```text
amp_update_scale -> AmpUpdateScale<half>::Init -> Process -> Compute
  -> LoadInputData(static_cast<float>) -> ComputeScaleUpdate -> StoreOutputData(static_cast<half>)
```

##### 源码等价伪码

```text
copy 2 B s, 4 B t, 2 B f GM->UB; synchronize; scalar-cast s,f half->float
execute the same ordered FP32 update; scalar-cast result float->half
copy 2 B updatedScale and 4 B updatedTracker UB->GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| scale/foundInf GM | base+0 | 1 element | half / ND | caller | MTE2 | caller | 各有效 2 B |
| tracker GM | base+0 | 1 element | int32 / ND | caller | MTE2 | caller | 4 B |
| 五个 TBuf | 各独立 | 各 32 B | half/int32 | MTE2/scalar | scalar/MTE3 | TPipe | 无 alias |
| 输出 GM | base+0 | 1 element | half/int32 | MTE3 | caller | caller | 2 B + 4 B |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  cs["currentScaleBuf · 32B"] fi["foundInfBuf · 32B"] gt["growthTrackerBuf · 32B"] us["updatedScaleBuf · 32B"] ug["updatedTrackerBuf · 32B"]
```

##### 字节账与复用顺序

```text
UB_total = 160 B; dynamic_ub = N/A
workspace_total = body 0 B；ACLNN 框架精确值固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | 独占 TBuf | 对应阶段 | kernel 结束 | hard event | 单标量 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | MTE2_V、V_S | 2/4 B copy | scalar read | 输入 TBuf | half→float C++ cast |
| 计算 | 标量 | FP32 成员 | output cast | scale/tracker | 乘法与有限检查均 FP32 |
| 输出 | S_V、V_MTE3、MTE3_MTE2 | scalar SetValue | MTE3 | 输出 | float→half cast；舍入模式固定仓未显式命名 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | key 1 |
| Kernel entry/router | `op_kernel/amp_update_scale.cpp` | key 1 branch | half 实例 |
| Device body/helper | `op_kernel/amp_update_scale.h` | `LoadInputData/StoreOutputData` half 分支 | FP32 升精度计算与回写 |

### Route R-003：BF16 Vector Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | `dtype==DT_BF16` | key 2 |
| Router | `op_kernel/amp_update_scale.cpp` | `amp_update_scale` | `TILING_KEY_IS(2)` | `AmpUpdateScale<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
amp_update_scale -> AmpUpdateScale<bfloat16_t>::Init(+castBuf) -> Compute
  -> LoadInputData(Cast BF16->FP32 twice) -> ComputeScaleUpdate
  -> StoreOutputData(Cast FP32->BF16, CAST_RINT) -> GM
```

##### 源码等价伪码

```text
copy BF16 s/f and int32 t to UB; wait MTE2_V
Cast castBuf[0] = FP32(s), PIPE_V, wait V_S, scalar-read; reuse castBuf for FP32(f)
perform ordered FP32 update
scalar-write FP32 s to castBuf; wait S_V; Cast updatedScale = BF16_RINT(castBuf)
scalar-write tracker; wait S_V/V_MTE3; copy 2 B + 4 B to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 输入/输出 GM | base+0 | 各 1 element | BF16/int32 ND | caller/MTE3 | MTE2/caller | caller | 2 B 或 4 B |
| 五个基础 TBuf | 各独立 | 各 32 B | BF16/int32 | MTE2/scalar | Vector/scalar/MTE3 | TPipe | 无 alias |
| `castBuf_` | 独立 UB 区 | 32 B | float | Cast/scalar | scalar/Cast | TPipe，整个 kernel | 依次复用 s、f、输出 s |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  cs["currentScale · 32B"] fi["foundInf · 32B"] gt["tracker · 32B"] cast["castBuf · 32B"] us["updatedScale · 32B"] ug["updatedTracker · 32B"]
```

##### 字节账与复用顺序

```text
UB_total = 6 * 32 B = 192 B
dynamic_ub = N/A
workspace_total = body 0 B；ACLNN 框架精确值固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `castBuf_[0]` | Cast(currentScale) | V_S 后 scalar GetValue | Cast(foundInf) | PIPE_V + V_S | 前值已读入成员 |
| `castBuf_[0]` | Cast(foundInf) | V_S 后 scalar GetValue | scalar SetValue(updatedScale) | PIPE_V + V_S | foundInf 已读入成员 |
| `castBuf_[0]` | scalar SetValue | Cast to BF16 output | kernel 结束 | S_V | Cast 完成后才进入 MTE3 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16 输入转换 | Cast `CAST_NONE` + PIPE_V + V_S | MTE2/Vector | scalar read | castBuf | 仅 mask/count 1 |
| 计算 | FP32 标量 | scalar | output conversion | 成员 | FP32 乘法和 bit-level finite |
| BF16 输出转换 | S_V + Cast `CAST_RINT` | scalar | Vector | castBuf/output TBuf | RINT，count 1 |
| 输出 | V_MTE3 + MTE3_MTE2 | Vector/scalar | MTE3/结束 | output UB/GM | 有效 2 B scale、4 B tracker |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/amp_update_scale_tiling.cpp` | `RunKernelTiling` | key 2 |
| Kernel entry/router | `op_kernel/amp_update_scale.cpp` | key 2 branch | BF16 实例 |
| Device body/helper | `op_kernel/amp_update_scale.h` | `Init/LoadInputData/StoreOutputData` | 条件 castBuf、两次升精度、RINT 降精度、复用同步 |

## 算子特定优化

### Trick T-001：按 dtype 编译期消除转换缓冲区

#### 触发与源码依据

- 适用 Route：R-003 分配；R-001/R-002 编译期消除。
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

只有 BF16 需要 Vector `Cast` 才分配 32 B `castBuf_`；FP32 直接读写，FP16 使用标量 cast，编译期分支令两条路线的 UB 从 192 B 降为 160 B。BF16 的同一物理缓冲又按 `currentScale→foundInf→updatedScale` 顺序复用，各次覆盖前通过 PIPE_V/V_S 或 S_V 建立可见性，避免为三个转换值分别物化缓冲。

#### 迁移边界

- 适用：只有部分 dtype 需要显式 Vector 转换的单标量或小张量控制算子。
- 前提：模板 dtype 编译期已知；各转换值生命周期不重叠；事件/pipe 顺序仍成立。
- 失效：转换并发、流水重叠或前值尚未消费时必须拆分缓冲或增加严格同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | dtype FLOAT/key0 | FLOAT16/key1 | BF16/key2 | Host dtype 决策树 |
| 模板与 dtype | `float` | `half` | `bfloat16_t` | binary config 三实例 |
| 循环与分核 | blockDim 1，无循环 | 同左 | 同左 | 标量 `[1]` |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 5 TBuf | 5 TBuf | 6 TBuf | BF16 需要 castBuf |
| 内存总占用与复用 | 160 B | 160 B | 192 B，castBuf 三阶段复用 | 编译期条件分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 4 B DataCopyPad | 2/4 B DataCopyPad | DataCopyPad + Cast | dtype 转换能力 |
| 同步、精度与 tail | FP32；hard event | FP32 中间；标量 cast | FP32 中间；CAST_RINT | 均无循环 tail |
| fallback/失败边界 | unknown dtype 误落 key0 仅内部可见 | 无 Device fallback | 无 Device fallback | 公开 ACLNN 先拒绝非法 dtype |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；`ascend910b` | 支持；`ascend910_93` | 支持；详见 `_950` 卡 |
| Host/Launcher | 共用同一 OpDef/Tiling，blockDim 1 | 同左 | 固定源码同一 Host |
| Kernel/资源 | 三 dtype config；同一 AIV body | 三 dtype config；同一 AIV body | 独立 ascend950 config，源码 body 相同 |

## 限制与失败边界

- 仅支持 `[1]` ND 标量；公开接口不支持非连续 Tensor。Host Tiling 可见代码只复查三个输入 shape，输出 shape 校验位于固定仓不可见的 ACLNN bridge。
- `growthInterval` 必须为 `[1,2147483647]`。`growthTracker_+1` 使用 int32_t，固定仓未对输入 tracker 范围或有符号溢出单独校验。
- `growthFactor/backoffFactor` 没有可见范围或有限性校验；回退乘法也不做 `IsFinite` 保护。有限性检查只作用于增长结果，并把 FP32 Inf 和 NaN 都判为非有限。
- FP16 降精度的显式 round mode 固定仓不可见；BF16 明确使用 `CAST_RINT`。
- ACLNN 生成实现、executor 的精确 workspace size/launch API、Tiling data 宏展开后的精确 ABI offset/sizeof/alignof，以及平台 UB 容量固定仓不可见。
- unknown key 没有 Device 错误分支；公开链路通过 dtype 校验和配置保证只生成 0/1/2。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/amp_update_scale/docs/aclnnAmpUpdateScale.md` | 两段式原型、产品表 | 公开性、语义、参数和错误 |
| 入口实现 | `math/amp_update_scale/op_host/amp_update_scale_def.cpp` | `ops::AmpUpdateScale` | dtype/format/属性与平台注册 |
| Host/Tiling/Launcher | `math/amp_update_scale/op_host/amp_update_scale_tiling.h` | `AmpUpdateScaleTilingData` | POD 字段与注册 |
| Host/Tiling/Launcher | `math/amp_update_scale/op_host/amp_update_scale_tiling.cpp` | `Init/RunKernelTiling` | shape/interval 校验、key、blockDim、序列化 |
| Kernel entry/body | `math/amp_update_scale/op_kernel/amp_update_scale.cpp` | `amp_update_scale` | key→模板实例 |
| Kernel entry/body | `math/amp_update_scale/op_kernel/amp_update_scale.h` | `AmpUpdateScale<T>` | 完整搬运、计算、同步与资源 |
| 构建接线 | `math/amp_update_scale/op_host/config/ascend910b/amp_update_scale_binary.json` | `AmpUpdateScale` op_list | A2 三 dtype binary |
| 构建接线 | `math/amp_update_scale/op_host/config/ascend910_93/amp_update_scale_binary.json` | `AmpUpdateScale` op_list | A3 三 dtype binary |
| 构建接线 | `math/amp_update_scale/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN OpDef/Tiling 模块接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三个属性直接送入 Device | 字段声明顺序和 32 位类型；interval 先校验再窄化 | `AmpUpdateScaleTilingData` |
| 调用顺序 | Load→Update→Store | foundInf 分支优先；增长命中后 tracker 恒清零 | `Compute/ComputeScaleUpdate` |
| workspace/内存 | 五个 32 B 基础 TBuf，BF16 加一个 | BF16 castBuf 覆盖前必须完成消费；workspace body 未读 | `Init/LoadInputData/StoreOutputData` |
| 同步 | MTE2_V、V_S/S_V、V_MTE3、MTE3_MTE2 | 不得在 producer 完成前读取或复用 | `SyncM2toV/SyncVtoM3/SyncM3toM2` |
| 精度与尾块 | FP32 中间计算，BF16 RINT 回写 | 有限性按 FP32 指数域；DataCopyPad 仅提交一个元素 | `IsFinite/LoadInputData/StoreOutputData` |
| 失败边界 | 公开 dtype/shape/interval 校验 | 不可依赖 unknown key 行为；非法 interval 不得窄化 | `Init/RunKernelTiling/amp_update_scale` |