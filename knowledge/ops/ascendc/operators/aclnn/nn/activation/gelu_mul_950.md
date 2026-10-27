---
type: Operator
title: "GeluMul 950 ACLNN 算子知识"
description: "说明 aclnnGeluMul 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/docs/aclnnGeluMul.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/gelu_mul_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/gelu_mul_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/gelu_mul_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/gelu_mul_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_kernel/gelu_mul.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_kernel/gelu_mul.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/config/ascend950/gelu_mul_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T14:59:51Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GeluMul 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnGeluMulGetWorkspaceSize` / `aclnnGeluMul`，公开 ACLNN 两段式接口 |
| 计算语义 | 尾轴拆成 `x1/x2`，输出 `GELU(x1, approximate) * x2` |
| 输入输出 | ND、2～8 维、尾轴 `2d`；FP16/FP32/BF16；输出尾轴 `d` |
| 执行类型 | 单 Kernel `gelu_mul`：`AIV_VECTOR`；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001～R-003：none×FP16/FP32/BF16；R-004～R-006：tanh×FP16/FP32/BF16 |
| 核心资源 | GM；192 KiB `TBuf<VECCALC>`，两个 96 KiB ping-pong bank；无 TQue；声明 16 MiB workspace、body 不访问 |
| 来源闭包 | `source_closed`：公开 ACLNN→OpDef/InferShape→Tiling/POD→key router→`GeluMulND<T>`→binary config/CMake |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令尾轴为 `2d`，`x1=input[..., :d]`、`x2=input[..., d:]`。`none` 语义是 `0.5*x1*(1+erf(x1/sqrt(2)))*x2`；`tanh` 语义是 `0.5*x1*(1+tanh(sqrt(2/pi)*(x1+0.044715*x1^3)))*x2`。Device 的 tanh 路线用等价的 `x1/(1+exp(-sqrt(8/pi)*(x1+0.044715*x1^3)))`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `input` | ND，2～8 维 | FP16/FP32/BF16 | 尾轴非 0、偶数、`<=1024`；前导维乘积 `<=200000`；`AutoContiguous()` |
| `approximateOptional` | 字符串 | `char*` | `nullptr` 默认 `none`；只允许 `none/tanh` |
| `out` | 前导维同输入，尾轴减半 | 同输入 | 非空指针；`AutoContiguous()` |
| `workspaceSize/executor` | 第一段输出 | `uint64_t*` / `aclOpExecutor**` | 保存第二段的资源与执行流程 |
| `workspace/stream` | Device 地址/stream | 指针 | 第二段按返回大小提供 workspace |

公开文档称支持空 Tensor；Tiling 拒绝尾轴为 0，但允许其他维为 0 且尾轴非 0，此时 `batchSize=0`、Device 循环 0 次。实现复用应保留这个精确边界。

### 入口与版本

实际入口是无版本后缀的公开 `aclnnGeluMul`。仓内没有手写 ACLNN inner bridge；`op_host/CMakeLists.txt` 用 `ACLNNTYPE aclnn` 接线 `GeluMul` OpDef，生成层作为框架边界。公开表支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，其 binary config 列出三种 dtype。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN/OpDef 检查指针、dtype、shape、输出一致性与属性，并对非连续 Tensor 执行框架自动连续化。
2. `InferShape4GeluMul` 复制 shape 后把尾轴除 2；`InferDataType4GeluMul` 复制 dtype。
3. `RunBigKernelTiling`：`none→mode=0,P=6144`，`tanh→mode=1,P=8192`，否则失败。
4. `FillTilingKey`：FP16→key1、FP32→key2、BF16→key3；相应 32 B 元素数为 16/8/16。
5. 读取 origin shape，检查 rank≥2、尾轴非 0/偶数/≤1024、`batchSize≤200000`；失败时阻断启动。

### 参数、Workspace 与生命周期

Tiling 无条件返回 `16*1024*1024` 字节 workspace。Kernel entry 虽接收它，却把 `userWs=nullptr` 传给 `Init`，且 `Init` 不读取 workspace：当前 body 消费 0 B，但调用方仍须遵守 ACLNN 返回的 16 MiB ABI。executor、POD 和框架连续化对象保持到指定 stream 上任务完成。

### 启动图与失败传播

```text
aclnnGeluMulGetWorkspaceSize
  -> GeluMul OpDef / AutoContiguous framework boundary
  -> InferShape4GeluMul + TilingGeluMulTiling
  -> workspace=16 MiB, blockDim=needCoreNum, key=1/2/3, POD
aclnnGeluMul(..., stream)
  -> gelu_mul(input, output, workspace, tiling)
     -> GeluMulND<T>::Init -> Process -> SmallTailProcess
```

只有一个 Device 节点，无跨 Kernel workspace 边。调用方拥有 GM/workspace；同 stream 提供框架前后处理与 Kernel 的顺序。Tiling 或 launch 失败阻断任务；未知 key 无 fallback、无输出。

## Route 与参数设计

### 分核与资源计算

设 `d=lastDimSize/2`，`P=PPMaxCalNum`，`B=32/sizeof(T)`，`A=ceil(d/B)*B`。

```text
P_none = 192 KiB / 8 / 4 B = 6144 elements
P_tanh = 192 KiB / 6 / 4 B = 8192 elements
n = floor(P/A)
needCoreNum = min(coreNumAiv, max(1, ceil(batchSize/n)))
eachCoreNum = floor(batchSize/needCoreNum) + (blockIdx < batchSize%needCoreNum)
loopNum = ceil(eachCoreNum/n)
```

公开约束给出 `d<=512<P`，所以 `BigTailProcess` 不可达。每核处理连续行，每轮 `tmpCalNum<=n`；Vector 长度为 `tmpCalNum*A`，GM 每行只写 `d` 个有效元素。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 ERF-FP16 | 1/`GeluMulND<half>` | `none && FP16` | 下一行 |
| 2 | R-002 ERF-FP32 | 2/`GeluMulND<float>` | `none && FP32` | 下一行 |
| 3 | R-003 ERF-BF16 | 3/`GeluMulND<bfloat16_t>` | `none && BF16` | tanh 路线/失败 |
| 4 | R-004 TANH-FP16 | 1/`GeluMulND<half>` | `tanh && FP16` | 下一行 |
| 5 | R-005 TANH-FP32 | 2/`GeluMulND<float>` | `tanh && FP32` | 下一行 |
| 6 | R-006 TANH-BF16 | 3/`GeluMulND<bfloat16_t>` | `tanh && BF16` | 失败 |

```text
attribute selector -> mode/P; dtype selector -> key/B; shape/resource checks
  -> key router -> GeluMulND<T>::SmallTailProcess
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | key1/mode0 | FP16、合法 shape、950 | 单 AIV | half+polynomial ERF | 2×96 KiB；窄类型别名；event0/1 | A=align16(d)，无 fallback |
| R-002 | key2/mode0 | FP32、合法 shape、950 | 单 AIV | float+polynomial ERF | 2×96 KiB；无 Cast | A=align8(d) |
| R-003 | key3/mode0 | BF16、合法 shape、950 | 单 AIV | bf16+polynomial ERF | 同 R-001，输出 RINT | A=align16(d) |
| R-004 | key1/mode1 | FP16、合法 shape、950 | 单 AIV | half+exp/div tanh | 每 bank 3 个 FP32 区 | 同 R-001 |
| R-005 | key2/mode1 | FP32、合法 shape、950 | 单 AIV | float+exp/div tanh | 无 Cast | 同 R-002 |
| R-006 | key3/mode1 | BF16、合法 shape、950 | 单 AIV | bf16+exp/div tanh | 输出 RINT | 同 R-003 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 小尾轴 | `lastDimSize<=1024 && even && !=0` | 保证 `d<P` | 全部 | Host 失败 | `ShapeCheck/RunBigKernelTiling` |
| UB 模式预算 | none 8 区、tanh 6 区 | 两个 96 KiB bank | 全部 | 非法属性失败 | `RunBigKernelTiling` |
| 32 B 行对齐 | `A=ceil(d/B)*B` | `DataCopyPad` 与 Vector 域 | 全部 | padding，无新 Route | `GetNeedCoreNum/SmallTailProcess` |
| dtype guard | key 1/2/3 | 模板、Cast、舍入 | 全部 | unknown key 无 body | `FillTilingKey/gelu_mul` |
| core 上限 | `needCoreNum<=coreNumAiv` | 避免超出物理 AIV 核 | 全部 | 截断 | `GetNeedCoreNum` |

### POD/ABI

POD 宏展开在固定目录外，精确 `sizeof/alignof/offsetof` 不可见；字段顺序、类型和两端消费闭合。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `GeluMulTilingData` | `lastDimSize` | int32/32 | 宏展开不可见 | 元素 | `set_lastDimSize` | `Init`→`d` | 非0、偶数、≤1024 |
| 同上 | `batchSize` | int32/32 | 同上 | 行 | `inputShapeSize/lastDimSize` | core/loop | ≤200000，可0 |
| 同上 | `approximateMode` | int32/32 | 同上 | 枚举 | none→0，tanh→1 | `Compute` | 0/1 |
| 同上 | `PPMaxCalNum` | int32/32 | 同上 | FP32 元素 | 6144/8192 | Process/offset | >0 |
| 同上 | `needCoreNum` | uint32/32 | 同上 | AIV core | `GetNeedCoreNum` | core guard/分行 | 1～coreNumAiv |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-004 | key1 | `gelu_mul_tiling.cpp::GeluMulTiling` | `GeluMulTilingData` | `gelu_mul`→half | `gelu_mul.h::GeluMulND` | CMake + 950 config |
| R-002/R-005 | key2 | 同上 | 同上 | `gelu_mul`→float | 同上 | 同上 |
| R-003/R-006 | key3 | 同上 | 同上 | `gelu_mul`→bf16 | 同上 | 同上 |

## Kernel 实现

### 公共执行流程

`gelu_mul` 解析 POD 并按 key 实例化模板。`Init` 绑定 GM、读取 POD、分配单个 192 KiB TBuf。`SmallTailProcess` 为每核划连续行；每轮选择 ping/pong bank，等待该 bank 上一次 MTE3，分别搬入多行 x1/x2，窄类型先升 FP32，执行 GELU 与乘法，再按需降精度并只搬出每行 d 个元素。

### Route R-001：ERF-FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/gelu_mul_tiling.cpp` | `RunBigKernelTiling/FillTilingKey` | none+FP16 | mode0、P6144、key1 |
| Router | `op_kernel/gelu_mul.cpp` | `gelu_mul` | key1 | `GeluMulND<half>` |

#### Device 数据流

##### 函数级调用链

```text
Init -> SmallTailProcess -> CopyIn(FP16) -> Cast(FP32) -> ComputeGeluErf
  -> Mul(x2) -> Cast(FP16) -> CopyOut
```

##### 源码等价伪码

```text
for group<=n: wait bank; DMA rows x1/x2; cast tmpCalNum*A lanes;
clamp x; polynomial(x^2); exp; divide; multiply x2; cast; DMA d lanes/row; release bank
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x1/x2 | row*2d / +d | d/row | FP16 ND | caller | CopyIn | stream | pad to A |
| bank x1/x2 | bank*96KiB + 0/4P | 各 4P=24576B | FP32 | Cast | GELU/Mul | iteration | FP16 staging aliases upper 2P |
| bank temp/xSq | +8P/+12P | 各 24576B | FP32 | Vector | Vector | iteration | no cross-bank alias |
| GM out | row*d | d/row | FP16 | CopyOut | caller | caller | padding not written |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  x1["x1 FP32 · 4P\nFP16 staging aliases upper 2P"]
  x2["x2 FP32 · 4P\nFP16 staging aliases upper 2P"]
  tmp["temp · 4P"]
  sq["xSquared · 4P"]
```

该布局在 bank0/bank1 各一份。

##### 字节账与复用顺序

```text
UB_total = 2*4*6144*4 = 196608 B = 192 KiB
dynamic_ub = N/A
workspace_total = 16 MiB declared; 0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1/x2 高半区 | DMA staging | Cast | FP32 扩展目标 | MTE2_V+PIPE_V | staging 位于高地址 |
| ping/pong bank | Vector/MTE3 | CopyOut | 后续同 bank CopyIn | MTE3_MTE2 event0/1 | 末尾等待两 event |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | MTE3_MTE2、MTE2_V | MTE3/DMA | DMA/Vector | bank | pad to A |
| Compute | PIPE_V barrier | Vector op | next op | FP32 tensors | clamp[-13.15,5.751] |
| CopyOut | V_MTE3 | Cast | DMA | x1 | CAST_NONE，只写d |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/gelu_mul_tiling.cpp` | `GeluMulTiling` | P/core/POD |
| Kernel entry/router | `op_kernel/gelu_mul.cpp` | key1 | half body |
| Device body/helper | `op_kernel/gelu_mul.h` | `SmallTailProcess/ComputeGeluErf` | DMA/ERF/tail |

### Route R-002：ERF-FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `gelu_mul_tiling.cpp` / `gelu_mul.cpp` | selector/router | none+FP32/key2 | P6144、`GeluMulND<float>` |

#### Device 数据流

##### 函数级调用链

```text
SmallTailProcess -> direct FP32 CopyIn -> ComputeGeluErf -> Mul -> direct CopyOut
```

##### 源码等价伪码

```text
同 R-001；B=8、A=align8(d)，省略输入和输出 Cast。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| bank x1/x2/temp/xSq | 0/4P/8P/12P | 各24576B | FP32 | DMA/Vector | Vector/DMA | bank iteration | 无 staging |

##### 分层内存布局

同 R-001 的 4 区/bank，无窄类型别名。

##### 字节账与复用顺序

```text
UB_total=196608 B; dynamic_ub=N/A; workspace=16 MiB declared/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ping/pong bank | DMA/Vector | MTE3 | next same bank | event0/1 | same as R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pipeline | events+barriers | DMA/Vector | next stage | bank | all FP32，only d to GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/gelu_mul_tiling.cpp` | key2 | B=8 |
| Kernel entry/router | `op_kernel/gelu_mul.cpp` | float instance | key2 mapping |
| Device body/helper | `op_kernel/gelu_mul.h` | CopyIn/Compute/CopyOut | no Cast |

### Route R-003：ERF-BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `gelu_mul_tiling.cpp` / `gelu_mul.cpp` | selector/router | none+BF16/key3 | P6144、bf16 body |

#### Device 数据流

##### 函数级调用链

```text
同 R-001；输入 BF16->FP32，输出 FP32->BF16 CAST_RINT。
```

##### 源码等价伪码

```text
执行 R-001 padded ERF 管线，仅边界 dtype/round 不同。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| two banks | same R-001 | 96KiB/bank | BF16 staging+FP32 | DMA/Cast | Vector/DMA | ping-pong | upper-half alias |

##### 分层内存布局

同 R-001，staging dtype 为 BF16。

##### 字节账与复用顺序

```text
UB_total=196608 B; dynamic_ub=N/A; workspace=16 MiB declared/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| alias/banks | same R-001 | same | same | events/barriers | output RINT |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast pipeline | events/barriers | DMA/Vector | next stage | bank | BF16↔FP32，out RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/gelu_mul_tiling.cpp` | key3 | BF16 selector |
| Kernel entry/router | `op_kernel/gelu_mul.cpp` | bf16 instance | key3 mapping |
| Device body/helper | `op_kernel/gelu_mul.h` | `CopyOut` | CAST_RINT |

### Route R-004：TANH-FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `gelu_mul_tiling.cpp` / `gelu_mul.cpp` | selector/router | tanh+FP16/key1 | P8192、half body |

#### Device 数据流

##### 函数级调用链

```text
CopyIn -> Cast -> ComputeGeluTanh(Mul,Mul,Muls,Add,Muls,Exp,Adds,Div)
  -> Mul(x2) -> Cast -> CopyOut
```

##### 源码等价伪码

```text
t=x*x; t=x*t; t=.044715*t; t=x+t; t=-1.5957691*t;
t=exp(t); t=1+t; x=x/t; x=x*x2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| bank x1/x2/temp | 0/4P/8P | each32768B | FP32+FP16 staging | DMA/Cast/Vector | Vector/DMA | iteration | staging aliases upper 2P |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  x1["x1 FP32 · 4P\nFP16 staging upper 2P"]
  x2["x2 FP32 · 4P\nFP16 staging upper 2P"]
  tmp["temp · 4P"]
```

该布局在两个 bank 各一份。

##### 字节账与复用顺序

```text
UB_total=2*3*8192*4=196608 B; dynamic_ub=N/A
workspace=16 MiB declared/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| staging alias | DMA | Cast | FP32 expansion | MTE2_V/PIPE_V | high-address source |
| temp | each tanh op | next op | next in-place op | PIPE_V | source order unchanged |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tanh pipeline | events+barriers | DMA/Vector | next stage | bank/temp | FP32 compute，FP16 boundary，only d out |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/gelu_mul_tiling.cpp` | mode1/P8192 | tanh resource |
| Kernel entry/router | `op_kernel/gelu_mul.cpp` | key1 | half instance |
| Device body/helper | `op_kernel/gelu_mul.h` | `ComputeGeluTanh` | exp/div identity |

### Route R-005：TANH-FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `gelu_mul_tiling.cpp` / `gelu_mul.cpp` | selector/router | tanh+FP32/key2 | P8192、float body |

#### Device 数据流

##### 函数级调用链

```text
direct FP32 CopyIn -> ComputeGeluTanh -> Mul -> direct FP32 CopyOut
```

##### 源码等价伪码

```text
同 R-004；B=8、A=align8(d)，省略 Cast。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| bank x1/x2/temp | 0/4P/8P | each32768B | FP32 | DMA/Vector | Vector/DMA | iteration | no staging |

##### 分层内存布局

同 R-004 的 3 区/bank，无窄类型别名。

##### 字节账与复用顺序

```text
UB_total=196608 B; dynamic_ub=N/A; workspace=16 MiB declared/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| banks/temp | DMA/Vector | Vector/MTE3 | next iteration/op | events/barriers | source order |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pipeline | events+barriers | DMA/Vector | next stage | bank | all FP32，only d out |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/gelu_mul_tiling.cpp` | mode1/key2 | FP32 tanh |
| Kernel entry/router | `op_kernel/gelu_mul.cpp` | float instance | key2 |
| Device body/helper | `op_kernel/gelu_mul.h` | `ComputeGeluTanh` | no Cast |

### Route R-006：TANH-BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `gelu_mul_tiling.cpp` / `gelu_mul.cpp` | selector/router | tanh+BF16/key3 | P8192、bf16 body |

#### Device 数据流

##### 函数级调用链

```text
同 R-004；输入 BF16->FP32，输出 FP32->BF16 CAST_RINT。
```

##### 源码等价伪码

```text
执行 R-004 exp/div pipeline，仅边界 dtype/round 不同。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| two banks | same R-004 | 96KiB/bank | BF16 staging+FP32 | DMA/Cast | Vector/DMA | ping-pong | upper-half alias |

##### 分层内存布局

同 R-004，staging dtype 为 BF16。

##### 字节账与复用顺序

```text
UB_total=196608 B; dynamic_ub=N/A; workspace=16 MiB declared/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| alias/temp/banks | same R-004 | same | same | events/barriers | output RINT |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pipeline | events+barriers | DMA/Cast/Vector | next stage | bank | BF16↔FP32，RINT，only d out |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/gelu_mul_tiling.cpp` | mode1/key3 | BF16 tanh |
| Kernel entry/router | `op_kernel/gelu_mul.cpp` | bf16 instance | key3 |
| Device body/helper | `op_kernel/gelu_mul.h` | `ComputeGeluTanh/CopyOut` | RINT |

## 算子特定优化

### Trick T-001：按 GELU 模式重分同一 192 KiB UB

#### 触发与源码依据

- 适用 Route：R-001～R-006
- 触发条件：`approximate==none/tanh`
- Golden 来源：[`op_host/gelu_mul_tiling.cpp:115-125`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_host/gelu_mul_tiling.cpp#L115-L125)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (approximate == "none") {
    approximateMode = 0;
    PPMaxCalNum = UB_SIZE / ERF_BUF_NUM / static_cast<int32_t>(sizeof(float));
} else if (approximate == "tanh") {
    approximateMode = 1;
    PPMaxCalNum = UB_SIZE / TANH_BUF_NUM / static_cast<int32_t>(sizeof(float));
}
```

#### 机制与设计原因

ERF 每 bank 需要 x1/x2/temp/xSquared 四个 FP32 区，tanh 只需前三个；两个 bank 对应 8/6 区。Host 因此选择 6144/8192，使两条算法都恰好占 192 KiB，tanh 每轮可覆盖更多行。

#### 迁移边界

- 适用：不同算法分支拥有不同临时 tensor 数的融合逐元素算子。
- 前提：Host 区数、Device offset、dtype 字节数完全一致。
- 失效：增加临时区或改变 compute dtype 后必须重算 P。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓没有代码反例；字节账证明 ERF 若误用 P=8192，每 bank 四区需 128 KiB，会跨越 96 KiB bank。

### Trick T-002：窄类型 staging 与 FP32 目标高半区别名

#### 触发与源码依据

- 适用 Route：R-001/R-003/R-004/R-006
- 触发条件：`T` 为 half 或 bfloat16_t
- Golden 来源：[`op_kernel/gelu_mul.h:180-190`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_mul/op_kernel/gelu_mul.h#L180-L190)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
    int32_t elementByte = PPMaxCalNum * sizeof(T);
    x1Tmp = pingPongFlag ?
            tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
            tmpTensor[elementByte].ReinterpretCast<T>();
```

#### 机制与设计原因

2 B staging 放在对应 4 B FP32 目标区的高半段，Cast 后目标扩展覆盖整个区，不再额外分配 staging。MTE2_V 与 PIPE_V 保证读取源后才继续覆盖/消费。

#### 迁移边界

- 适用：2 B→4 B 扩展且 primitive 支持该重叠方向。
- 前提：源起点保持 `P*sizeof(T)` 的高地址布局。
- 失效：字节比、Cast 方向或并发模型变化时需重新证明。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；低地址 staging 可能被扩展目标覆盖，不能直接替换。

## 模板与实现对比

| 对比维度 | R-001/R-002/R-003 ERF | R-004/R-005/R-006 TANH | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | mode0，key1/2/3 | mode1，key1/2/3 | 属性定算法、dtype 定模板 |
| 模板与 dtype | half/float/bf16 | 同左 | 窄类型边界，FP32 计算 |
| 循环与分核 | P=6144 | P=8192 | 8 区 vs 6 区 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 192 KiB TBuf，无 TQue | 同左 | 2×96 KiB |
| 内存总占用与复用 | 4 区/bank | 3 区/bank | 均 192 KiB |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+polynomial/Exp/Div | DataCopyPad+Mul/Muls/Add/Exp/Div | AIV_VECTOR |
| 同步、精度与 tail | events/barriers，FP32，dtype cast | 同左 | padding 不写 GM |
| fallback/失败边界 | 无公开可达 fallback | 同左 | Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀 A2/A3 卡 | 支持，见无后缀 A2/A3 卡 | 支持，`ascend950` |
| Host/Launcher | 同源实现 | 同源实现 | 本卡：共用 Tiling/POD，独立 binary config |
| Kernel/资源 | 见无后缀卡 | 见无后缀卡 | key1/2/3，192 KiB TBuf |

## 限制与失败边界

- 尾轴非 0、偶数、≤1024；前导维乘积≤200000；公开 rank 2～8。
- 非法属性无 fallback；未知 key 静默不执行 body，依赖 Host 校验。
- `BigTailProcess` 在公开有效域不可达，不是 ACLNN 大尾轴 fallback。
- workspace 返回 16 MiB，body 消费 0 B；调用方仍遵守返回值。
- POD 宏展开与自动生成 ACLNN bridge 是固定目录外框架边界，未推测其 ABI offset。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnGeluMul.md` | ACLNN 两段接口 | 公开平台与约束 |
| 入口实现 | `op_host/gelu_mul_def.cpp` | `ops::GeluMul` | dtype/format/平台注册 |
| 入口实现 | `op_host/gelu_mul_infershape.cpp` | InferShape/InferDataType | 输出元数据 |
| Host/Tiling/Launcher | `op_host/gelu_mul_tiling.h` | `GeluMulTilingData` | POD |
| Host/Tiling/Launcher | `op_host/gelu_mul_tiling.cpp` | `GeluMulTiling` | key/core/UB/workspace |
| Kernel entry/body | `op_kernel/gelu_mul.cpp` | `gelu_mul` | router |
| Kernel entry/body | `op_kernel/gelu_mul.h` | `GeluMulND<T>` | AIV body |
| 构建接线 | `op_host/config/ascend950/gelu_mul_binary.json` | `op_list` | 950 binaries |
| 构建接线 | `op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN 接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 五字段传递 | 顺序/位宽/单位 | `GeluMulTilingData/Init` |
| 调用顺序 | CopyIn→Cast→GELU→Mul→Cast→CopyOut | event/barrier 顺序 | `SmallTailProcess/Compute` |
| workspace/内存 | 2×96 KiB bank | P/offset/alias/owner | `Init/CopyIn/Compute` |
| 同步 | event0/1 ping-pong | MTE3 完成后复用 | `SmallTailProcess` |
| 精度与尾块 | FP32 compute | BF16 RINT；only d to GM | `Compute/CopyOut` |
| 失败边界 | Host rejects invalid inputs | unknown key/zero last dim | `RunBigKernelTiling/gelu_mul` |