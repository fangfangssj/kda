---
type: Operator
title: "Logit 950 ACLNN 算子知识"
description: "说明 aclnnLogit 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/docs/aclnnLogit.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/logit_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/logit_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/logit_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/config/ascend950/logit_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:13:26Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Logit 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnLogitGetWorkspaceSize` + `aclnnLogit`，公开 ACLNN、固定仓无更高版本 |
| 计算语义 | `eps>=0` 时把非 NaN 输入截断到 `[eps,1-eps]`，再以 FP32 计算 `ln(z/(1-z))`；`eps<0` 时不截断 |
| 输入输出 | ND，0～8 维，FP16/FP32/BF16；输出同 shape/dtype/layout；支持空 Tensor 和非连续 Tensor |
| 执行类型 | 单 `logit` Kernel，`AIV_VECTOR`；无 Cube/AIC、MIX_CV、SIMT、NDDMA、通信节点 |
| Kernel 路线 | R-001 key 1 FP16、R-002 key 2 FP32、R-003 key 3 BF16 |
| 核心资源 | GM 输入输出、固定 192 KiB VECCALC TBuf（两个 96 KiB 半区）；Tiling 声明 16 MiB workspace，Kernel 不消费 |
| 来源闭包 | `source_closed`：公开 950 支持 → `ascend950` OpDef/binary/CMake → Tiling/POD → router → `LogitND<T>` 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

`eps>=0` 时，Device 先构造 `x==x`、`x>=eps`、`x<=1-eps` 三个 mask，依次完成上界、下界和 NaN 选择，得到 `z`；随后原地执行 `x2=-z+1`、`x1=z/x2`、`x1=ln(x1)`。`eps<0` 跳过 clamp。FP16/BF16 输入先升为 FP32，所有算术均为 FP32。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `input` | ND，0～8 维，可为空/非连续 | BF16/FP16/FP32 | 输入概率；OpDef `AutoContiguous()`，连续化实体属于框架边界 |
| `eps` | 标量 | ACLNN `double`；内部属性/POD `float` | 默认/建议 `-1.0`；内部使用 FP32，非负启用 clamp |
| `out` | 同输入 | 同输入 | 文档要求 shape/dtype/format 与输入一致；空域不写 GM |

### 入口与版本

固定 Commit 只公开无版本后缀的两段式 `aclnnLogit`。文档明确 Ascend 950PR/950DT 支持；`Logit` OpDef 注册 `ascend950`，binary config 发布三种 dtype，Kernel CMake 编译 `ascend950`，entry 外层允许 `__CCE_AICORE__==310`。生成式 ACLNN C 实现和 executor 壳未入仓，按 Skill 允许的框架边界标注。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段按公开文档检查空指针和 dtype，OpDef 把 input/output 限为 BF16/FP16/FP32、ND，并启用自动连续化。Tiling 依次检查输入 Tensor、attrs、输入描述，读取属性、dtype 和 origin shape；其它 dtype 返回 `GRAPH_FAILED`。输出一致性和非连续 Tensor 的具体生成代码固定仓不可见。

### 参数、Workspace 与生命周期

令 `N` 为输入元素数，`C=max(1,min(GetCoreNumAiv(),ceil(N/8192)))`。Host 将 `elementNum=N`、`eps`、`needCoreNum=C` 序列化，令 `blockDim=C`，并报告固定 `16*1024*1024` 字节 workspace。executor/workspace 由调用者保留至第二段提交；Kernel 内把 workspace 替换成 null，`LogitND::Init` 也不读它。

### 启动图与失败传播

唯一 Device 节点是 `logit(input,output,workspace,tiling)`，在调用 stream 上以 C 个 AIV block 启动，无跨 Kernel workspace 边。Tiling 失败时不启动；unknown key 或编译 guard 排除 key 时 entry 无 body、无输出提交。框架 stream 提交和错误码转换符号固定仓不可见。

## Route 与参数设计

### 分核与资源计算

每个逻辑 tile 为 8192 元素。`totalTimes=ceil(N/8192)`，前 `totalTimes%C` 个 block 多一轮；最后一个 former block 处理不足 8192 的尾 tile。每轮使用一个 96 KiB 半区，两轮 parity 交替；每 block 固定分配 192 KiB。`N=0` 时 Host 仍令 C=1，而 Device `loopNum=0`，只初始化/收尾两个 event。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 升精度 | key 1/`LogitND<half>` | `dtype==DT_FLOAT16` | 检查 FP32 |
| 2 | R-002 FP32 原生 | key 2/`LogitND<float>` | 前项不命中且 `dtype==DT_FLOAT` | 检查 BF16 |
| 3 | R-003 BF16 升精度 | key 3/`LogitND<bfloat16_t>` | 前两项不命中且 `dtype==DT_BF16`；key 3 分支受非 `__NPU_ARCH__==3003` guard | Host 失败或 router 无写出 |

```text
FP16 -> key1 -> LogitND<half>
FP32 -> key2 -> LogitND<float>
BF16 -> key3 -> LogitND<bfloat16_t>
其它 dtype -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | FP16、ND、950 | 单 `logit` | `LogitND<half>` 四阶段 helper | 16-bit staging→FP32；MTE/V event | 精确 n*2 B tail |
| R-002 FP32 | 2 | FP32、ND、950 | 单 `logit` | `LogitND<float>` | FP32 直接搬入/算术/搬出 | 精确 n*4 B tail |
| R-003 BF16 | 3 | BF16、ND、950，且 router guard 保留实例 | 单 `logit` | `LogitND<bfloat16_t>` | BF16→FP32，写回 RINT | guard/unknown key 无 body |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| tile/core | `C=min(AIV cores,ceil(N/8192))`，0 修正为 1 | 固定 UB 每轮容纳 8192 个 FP32 工作元素 | 全部 | 每核增加 loop | `GetNeedCoreNum` |
| Compare 64 粒度 | `eps>=0` 且 `roundUp(n,64)` | Compare mask 的 FP32 repeat 粒度 | 全部 | Select 和写出仍只用 n | `ComputeStepOne` |
| 950 entry guard | `__CCE_AICORE__==310` 或同一外层允许条件 | 只有编译支持目标实例化 body | 全部 | entry 空结束 | `logit` |
| BF16 guard | key 3 且非 `__NPU_ARCH__==3003` | BF16 模板条件编译 | R-003 | 无 BF16 body | `logit` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LogitTilingData` | `elementNum` | `int64_t`/64 | 第 1 字段；宏生成的 offset/总大小固定仓不可见 | 元素 | `set_elementNum(inputShapeSize)` | `Init`/`Process` | shape 乘积溢出保护不可见 |
| `LogitTilingData` | `eps` | `float`/32 | 第 2 字段；精确 offset 固定仓不可见 | 无量纲 | `set_eps(epsilon)` | clamp 条件和上下界 | ACLNN double→float 位置不可见 |
| `LogitTilingData` | `needCoreNum` | `uint64_t`/64 | 第 3 字段；精确 offset 固定仓不可见 | AIV core | `set_needCoreNum(C)` | 分核、loop、last core | `[1,GetCoreNumAiv()]` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | `logit_tiling.cpp::RunBigKernelTiling` | `LogitTilingData` | `logit.cpp::logit` | `logit.h::LogitND<half>` | `ascend950/logit_binary.json`、Kernel CMake |
| R-002 | key 2 | 同上 | 同上 | 同上 | `LogitND<float>` | 同上 |
| R-003 | key 3 | 同上 | 同上 | 同上 | `LogitND<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

`Init` 绑定 input/output GM、读取 POD、取得 block index并初始化 192 KiB TBuf。`Process` 计算连续 core 区间和 loop/tail，给两个半区各预置 `MTE3_MTE2` flag；每轮等待本半区可覆盖，搬入并升精度，可选 clamp，计算 logit，降精度并异步搬出，最后等待两套 MTE3 完成。

### Route R-001：FP16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | `DT_FLOAT16` | key 1 |
| Router | `loss/logit/op_kernel/logit.cpp` | `logit` | `TILING_KEY_IS(1)` | `LogitND<half>` |

#### Device 数据流

##### 函数级调用链

```text
logit -> LogitND<half>::Init -> Process
  -> CopyInAndCast -> ComputeStepOne(eps>=0) -> ComputeStepTwo -> CastAndCopyOut
```

##### 源码等价伪码

```text
for tile in 本核连续区间:
  n=8192，只有全局最后尾 tile 改为 N%8192
  wait 当前半区 MTE3->MTE2; DataCopyPad n*2 B 到 FP16 staging; MTE2->V
  Cast FP16->FP32 x
  if eps>=0: Compare/CompareScalar 三个 mask并 Select 上界、下界、NaN
  x2=-x+1; x=x/x2; x=Ln(x)
  Cast FP32->FP16; V->MTE3; DataCopyPad n*2 B 到输出; 标记半区可在未来复用
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `coreStart+i*8192` | n | FP16 ND | caller/MTE3 | MTE2/caller | caller | `DataCopyPad`, byte len=n*2 |
| 半区 | `pingPongFlag*98304` | 98304 B | raw UB | TPipe | 全阶段 | block lifetime | 两半区交替 |
| x1/staging | 半区+0 / +16384 | 32768/16384 B | FP32/FP16 | Cast/MTE2 | compute/Cast | tile | staging 与 FP32 目的区重叠 |
| x2/masks | +32768/+65536/+81920 | 32768 B + mask 区 | FP32/bit | Vector | Vector | tile | mask3 复用 x2 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  pingA["ping x1/staging · 32 KiB"]:1
  pingB["ping x2/mask3 · 32 KiB"]:1
  pingC["ping mask1/2 + spare · 32 KiB"]:1
  pong["pong · 同构 96 KiB"]:3
```

##### 字节账与复用顺序

```text
UB_total=192*1024=196608 B；half-region=98304 B
dynamic_ub=N/A；workspace_total=16 MiB（无 Kernel consumer）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 当前半区 | MTE2/Vector | Vector/MTE3 | 同 parity 下轮 | event 0/1 `MTE3_MTE2` | WaitFlag 后才能重用 |
| x2 区 | mask3/Muls/Adds | Select/Div | 同 tile 后继 op | `PIPE_V` barrier | 不跨 barrier 覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入 | MTE3→MTE2、MTE2→V | 上轮 MTE3/本轮 MTE2 | CopyIn/Vector | 半区 | n*2 B，随后 FP32 |
| 算术 | V barrier | 每个 Cast/Select/算术 | 后继 Vector | x1/x2/mask | Compare roundUp(n,64)，其它只处理 n |
| 搬出 | V→MTE3 | FP16 `CAST_NONE` | MTE3 | x1 | 只写 n |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit/op_host/logit_tiling.cpp` | `GetNeedCoreNum`/`RunBigKernelTiling` | 分核、key/POD/workspace |
| Kernel entry/router | `loss/logit/op_kernel/logit.cpp` | `LogitND<half>` | key 1 body |
| Device body/helper | `loss/logit/op_kernel/logit.h` | `Process` 与四 helper | 搬运、FP32 算术、event、tail |

### Route R-002：FP32 原生 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | `DT_FLOAT` | key 2 |
| Router | `loss/logit/op_kernel/logit.cpp` | `logit` | key 2 | `LogitND<float>` |

#### Device 数据流

##### 函数级调用链

```text
logit -> LogitND<float>::Init -> Process
  -> CopyInAndCast(无 Cast) -> 可选 clamp -> FP32 Muls/Adds/Div/Ln -> CastAndCopyOut(无 Cast)
```

##### 源码等价伪码

```text
沿用 R-001 的 core/tile/event；直接搬 n*4 B 至 x1；
可选三 mask clamp；原地计算 1-x、除法、Ln；直接搬 n*4 B 回 GM。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `coreStart+i*8192` | n | FP32 ND | caller/MTE3 | MTE2/caller | caller | byte len=n*4 |
| x1/x2 | 半区+0/+32768 | 各 32768 B | FP32 | MTE2/Vector | Vector/MTE3 | tile | x1 原地；x2 兼 mask3 |
| mask1/2 | +65536/+81920 | 至少 ceil(n/8) B | bit | Compare | Select | tile | 同半区固定 offset |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  pingX1["ping x1 · 32 KiB"]:1
  pingX2["ping x2/mask3 · 32 KiB"]:1
  pingM["ping masks + spare · 32 KiB"]:1
  pong["pong · 同构 96 KiB"]:3
```

##### 字节账与复用顺序

```text
UB_total=196608 B；dynamic_ub=N/A；workspace_total=16 MiB（Kernel 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 | MTE2/Select/Div/Ln | 后继 Vector/MTE3 | 当前半区下轮 | V barrier + MTE event | 保持原地 op 顺序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | MTE event + V barrier | MTE2/Vector | Vector/MTE3 | 半区 | 全 FP32；搬运和算术仅 n，mask Compare 补 64 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | key 2 |
| Kernel entry/router | `loss/logit/op_kernel/logit.cpp` | `LogitND<float>` | FP32 实例 |
| Device body/helper | `loss/logit/op_kernel/logit.h` | `CopyInAndCast`/`CastAndCopyOut` | 编译期跳过 cast |

### Route R-003：BF16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | `DT_BF16` | key 3 |
| Router | `loss/logit/op_kernel/logit.cpp` | `logit` | key 3 且 guard 保留 | `LogitND<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
logit -> LogitND<bfloat16_t>::Init -> Process
  -> BF16 CopyIn/Cast -> 可选 clamp -> FP32 logit -> BF16 CAST_RINT/CopyOut
```

##### 源码等价伪码

```text
沿用 R-001 的地址、分核、mask 与事件；只把输入/输出 dtype 换成 BF16，
输入 CAST_NONE 至 FP32，输出以 CAST_RINT 从 FP32 降为 BF16。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `coreStart+i*8192` | n | BF16 ND | caller/MTE3 | MTE2/caller | caller | byte len=n*2 |
| staging/FP32/masks | 同 R-001 | 96 KiB/半区 | BF16/FP32/bit | MTE2/Vector | Vector/MTE3 | tile | staging 与目的区重叠、mask3 复用 x2 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  pingA["ping FP32 x1/BF16 staging · 32 KiB"]:1
  pingB["ping FP32 x2/mask3 · 32 KiB"]:1
  pingC["ping masks + spare · 32 KiB"]:1
  pong["pong · 同构 96 KiB"]:3
```

##### 字节账与复用顺序

```text
UB_total=196608 B；dynamic_ub=N/A；workspace_total=16 MiB（Kernel 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| staging/x1 | MTE2/Cast | Cast/Vector/MTE3 | 同 parity 下轮 | MTE/V event 和 barrier | offset、round mode、顺序不可改变 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入/算术 | MTE2→V、V barrier | BF16 CopyIn/Cast | FP32 compute | x1 | n 元素升 FP32 |
| 输出 | V→MTE3 | `CAST_RINT` | DataCopyPad | x1 | n 元素降 BF16、n*2 B 写出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | key 3 |
| Kernel entry/router | `loss/logit/op_kernel/logit.cpp` | `LogitND<bfloat16_t>` | BF16 实例与 guard |
| Device body/helper | `loss/logit/op_kernel/logit.h` | cast helpers/compute helpers | FP32 算术、BF16 RINT、tail |

## 算子特定优化

### Trick T-001：固定 TBuf 内双半区与双事件交替复用

#### 触发与源码依据

- 适用 Route：R-001～R-003
- 触发条件：任意非空 tile，`pingPongFlag` 交替
- Golden 来源：[`loss/logit/op_kernel/logit.h:118-139`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.h#L118-L139)

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
    if (eps >= 0) {
        ComputeStepOne(calNum);
    }
    ComputeStepTwo(calNum);
    CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);
    pingPongFlag = 1 - pingPongFlag;
}
WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
```

#### 机制与设计原因

一块 192 KiB TBuf 以固定 96 KiB offset 切成两套工作区；两套 `MTE3_MTE2` event 将每个半区的最后读取与下一次写入连接起来。这样 tile 状态不需要额外 GM/workspace 物化，同时保持输出 DMA 与未来 tile 的覆盖安全。

#### 迁移边界

- 适用：无跨 tile 依赖的逐元素 AIV pipeline。
- 前提：每 tile 至多 8192 元素、两套 event 与半区 offset 成对。
- 失效：扩大 tile 或增加 buffer 必须重算 192 KiB 字节账；新增异步阶段必须扩展同步边。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Compare 补齐、Select 限长的尾块处理

#### 触发与源码依据

- 适用 Route：R-001～R-003
- 触发条件：`eps>=0`，尤其 `dataCount%64!=0`
- Golden 来源：[`loss/logit/op_kernel/logit.h:187-205`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.h#L187-L205)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
float hi = static_cast<float>(1.0) - eps;
auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;
Compare(selMaskThree, x1TensorFp32, x1TensorFp32, CMPMODE::EQ, tmpDataCount);
PipeBarrier<PIPE_V>();

CompareScalar(selMaskOne, x1TensorFp32, (float)eps, CMPMODE::GE, tmpDataCount);
PipeBarrier<PIPE_V>();

CompareScalar(selMaskTwo, x1TensorFp32, (float)hi, CMPMODE::LE, tmpDataCount);
PipeBarrier<PIPE_V>();

Select(x1TensorFp32, selMaskTwo, x1TensorFp32, (float)hi, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
PipeBarrier<PIPE_V>();

Select(x1TensorFp32, selMaskOne, x1TensorFp32, (float)eps, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
PipeBarrier<PIPE_V>();

Select(x1TensorFp32, selMaskThree, x1TensorFp32, (float)nanValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

Compare 的工作长度上取整到 64 元素，而 Select、后续算术和 GM 写出仍使用真实 `dataCount`。它满足 mask primitive 的 repeat 粒度，同时阻止补齐 lane 成为有效输出。

#### 迁移边界

- 适用：mask 生成需要固定 repeat、结果操作支持精确有效长度的尾 tile。
- 前提：补齐区仍落在已分配的 8192 元素 UB 内，所有有外部副作用的步骤只用 `dataCount`。
- 失效：若后续算子对补齐 lane 做归约或写 GM，必须显式清零/屏蔽。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 3 + guard | dtype 决策顺序 |
| 模板与 dtype | `half` | `float` | `bfloat16_t` | 编译期消除 cast 分支 |
| 循环与分核 | 8192/tile | 同左 | 同左 | 共用 Host/POD |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 固定 192 KiB TBuf | 同左 | 同左 | 无 Queue/dynamic UB/window |
| 内存总占用与复用 | FP16 staging+FP32 | FP32 direct | BF16 staging+FP32 | 两个 96 KiB 半区 |
| 搬运/Vector/SIMT/NDDMA/通信 API | CopyPad+Cast+Vector | CopyPad+Vector | CopyPad+Cast+Vector | 均为 AIV_VECTOR |
| 同步、精度与 tail | FP32 core、FP16 NONE | 全 FP32 | FP32 core、BF16 RINT | Compare 64 对齐，n 有效 |
| fallback/失败边界 | 无 Device fallback | 同左 | guard 不成立无 body | unknown key 无写出 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持 |
| Host/Launcher | `ascend910b` config | `ascend910_93` config | `ascend950` config；共用 Tiling/POD |
| Kernel/资源 | 共用 body | 共用 body | CMake 单独列 `ascend950`，entry 外层含 `__CCE_AICORE__==310`；仍固定使用 192 KiB |

## 限制与失败边界

- 固定仓没有生成后的 ACLNN C 实现、executor/stream 提交、`AutoContiguous` workspace 聚合；这些只作为框架边界，不推测具体函数。
- `LogitTilingData` 宏展开后的精确 ABI offset/size、Vector primitive 寄存器占用及重叠 Cast 的底层契约固定仓不可见。
- 固定仓未给出 CMake compute unit 到所有预定义宏组合的展开表；正文只保留源码可见的 `ascend950` 构建接线和 entry guard。
- Kernel 不使用已声明的 16 MiB workspace；unknown key/guard 排除时 entry 没有显式错误返回或 fallback。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/logit/docs/aclnnLogit.md` | 两段式 ACLNN | 950 支持、接口、dtype/shape/错误语义 |
| 入口实现 | `loss/logit/op_host/logit_def.cpp` | `Logit`/`OP_ADD` | OpDef、AutoContiguous、ascend950 注册 |
| Host/Tiling/Launcher | `loss/logit/op_host/logit_tiling.h` | `LogitTilingData` | POD |
| Host/Tiling/Launcher | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | selector、分核、workspace、blockDim |
| Kernel entry/body | `loss/logit/op_kernel/logit.cpp` | `logit` | 950 guard、key router |
| Kernel entry/body | `loss/logit/op_kernel/logit.h` | `LogitND<T>` | 实际 AIV body、UB、同步、精度/tail |
| 构建接线 | `loss/logit/op_host/config/ascend950/logit_binary.json` | `op_list` | 950 三 dtype binary |
| 构建接线 | `loss/logit/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 Kernel target |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | element count、eps、core count 三字段 | 类型、顺序、单位 | `LogitTilingData` |
| 调用顺序 | CopyIn→clamp→logit→CopyOut | `eps<0` 必须跳过 clamp；NaN mask 顺序 | `Process`/`ComputeStepOne` |
| workspace/内存 | 192 KiB TBuf、8192/tile | 半区 98304 B、staging/mask offsets | `Init`/helpers |
| 同步 | event 0/1 + V barriers | 同半区覆盖必须等待 MTE3 | `Process`/helpers |
| 精度与尾块 | FP32 core；BF16 RINT | Compare 64 补齐不等于有效长度 | `ComputeStepOne`/`CastAndCopyOut` |
| 失败边界 | Host dtype 失败 | unknown key 无 fallback/无写出 | `RunBigKernelTiling`/`logit` |