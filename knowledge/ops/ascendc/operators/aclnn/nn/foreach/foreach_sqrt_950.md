---
type: Operator
title: "ForeachSqrt 950 ACLNN 算子知识"
description: "说明 aclnnForeachSqrt 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/docs/aclnnForeachSqrt.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_host/foreach_sqrt_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_graph/foreach_sqrt_proto.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_kernel/arch35/foreach_sqrt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_kernel/arch35/foreach_sqrt_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_host/config/ascend950/foreach_sqrt_binary.json
    role: implementation
created_at: "2026-08-25T16:21:34Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSqrt 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSqrtGetWorkspaceSize` + `aclnnForeachSqrt`，无版本后缀的公开两段式 ACLNN。 |
| 计算语义 | 对 TensorList 中每个输入元素计算 `y_i[j]=sqrt(x_i[j])`。 |
| 输入输出 | 动态 TensorList，FP16/FP32/BF16、ND；950 对应输入输出 shape size 相同，最大 50 个 Tensor；公开输出不支持非连续。 |
| 执行类型 | 单 AIV_VECTOR RegBase Kernel：`ForeachSqrtRegbase` → `ForeachRegbaseUnary` → AscendC `Cast/Sqrt`；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP16→FP32 Sqrt→FP16；R-002 FP32 direct Sqrt；R-003 BF16→FP32 Sqrt→BF16；key 10001/10002/10004。 |
| 核心资源 | GM `ListTensorDesc`；VECIN/VECOUT Queue 各 depth2；FP16/BF16 另有一个 FP32 `TBuf<VECCALC>`；Tiling 预留 32-byte workspace，Kernel 参数未使用。 |
| 来源闭包 | `source_closed`：公开 ACLNN → OpDef/生成 bridge 边界 → regbase selector/Tiling/POD → arch35 key router → RegBase Unary/body → ascend950 构建与 binary config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`，每个有效元素满足 `y_i[j]=sqrt(x_i[j])`。950 的 FP32 Route 直接执行 `Sqrt(outLocal,inLocal,count)`；FP16 与 BF16 Route 先以 `RoundMode::CAST_NONE` 转 FP32，在同一 FP32 TBuf 原地执行 `Sqrt`，再以 `RoundMode::CAST_RINT` 转回输出 dtype。Kernel 每项只写输入 storage numel 范围。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每 Tensor 0～8 维，ND | FP16/FP32/BF16，列表内一致 | 非空指针；支持零元素 Tensor；950 列表含 1～50 个 Tensor；OpDef `AutoContiguous()`。 |
| `out` | TensorList；每 Tensor 0～8 维，ND | 与 `x` 相同，列表内一致 | 非空指针；公开 950 契约要求与对应输入 shape size 相同；输出不支持非连续。 |
| `workspaceSize` | scalar | `uint64_t*` | 第一段输出 executor 总需求；算子 Tiling 固定写 32 bytes。 |
| `executor` | N/A | `aclOpExecutor**` | 第一段产生、第二段消费。 |
| `workspace/stream` | Device 地址/stream | pointer | 第二段消费；单 Kernel 接收 workspace 参数但 RegBase body 不读。 |

### 入口与版本

固定仓仅公开无后缀 `aclnnForeachSqrt`，没有更新的同语义入口。手写 ACLNN bridge/header 不在仓内，`op_host/CMakeLists.txt` 的 `ACLNNTYPE aclnn` 是生成边界。README/ACLNN 产品表明确支持 Ascend 950PR/950DT；OpDef 注册 ascend950，Kernel CMake 将 ascend950 精确接到 `arch35/foreach_sqrt.cpp`，binary JSON 列出三种 dtype，证据无冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开第一段检查 `x/out` 指针、FP16/FP32/BF16、输入输出及列表内 dtype 一致性、0～8 维和 950 对应 shape size 相同；最大 Tensor 数 50。
2. OpDef 对动态输入输出配置 ND 与 `AutoContiguous`；公开契约仍明确输出非连续不支持。
3. `Tiling4ForeachSqrtTiling` 进入 registry。priority 10000 membase 模板先尝试但 `IsCapable=!IsRegbaseSocVersion` 在 950 为 false并返回 `GRAPH_PARAM_INVALID`；随后 priority 30000 的 `ForeachRegbaseTilingUnary` 命中 regbase。
4. Tiling 依框架顺序执行 `GetShapeAttrsInfo → GetPlatformInfo → IsCapable → DoOpTiling → GetWorkspaceSize → PostTiling → GetTilingKey`。任何 descriptor、shape、dtype、output、workspace/POD 指针失败均返回失败并阻断 launch。

### 参数、Workspace 与生命周期

记 `N=sum_i numel(x_i)`、`B=4096/sizeof(T)` 个元素、平台 AIV core 数 `C`：

```text
numBlocks = min(C,80,ceil(N/B))
if N==0: numBlocks=1, interval=(tensor0,offset0)..(tensor0,offset-1)
else: 按 ceil(N/B) 个 4096-byte block 均分到 numBlocks
blockDim = numBlocks
```

`AssignDataToEachCore` 产生每 core 的 Tensor 起止与 inclusive offset。FP32 每 core tile `S=floor_align((U/16)*4,32)/4` 元素；FP16/BF16 为 `S=floor_align((U/12)*2,32)/2` 元素，其中 12 bytes/element 来自两个 depth2 原 dtype Queue 与一个 FP32 cast buffer。Tiling 预留 32 bytes workspace，但 `ForeachRegbaseUnary::Init` 不引用 workspace；输入/输出 ListTensor descriptor、POD 和 executor 必须存活到 stream 完成。

### 启动图与失败传播

```text
aclnnForeachSqrtGetWorkspaceSize
  -> framework-generated ForeachSqrt bridge / AutoContiguous boundary
    -> Tiling4ForeachSqrtTiling
      -> TilingRegistry -> ForeachRegbaseTilingUnary
        -> one foreach_sqrt(inputs,outputs,workspace,tiling), blockDim=numBlocks
          -> ForeachSqrtImpl<T>
            -> ForeachSqrtRegbase<T,ForeachSoloTilingDataRegbase>::Init/Process
```

单节点，无 Kernel 间 workspace edge。GM 输入/输出是 `ListTensorDesc` 描述的 TensorList；Tiling slice offset 0、extent 32 bytes、owner 为 executor/caller，Kernel 不读。entry 对未知 key 没有 else 错误上报，只会销毁 `TPipe` 后返回，因此 Host dtype/key 校验不可省略。

## Route 与参数设计

### 分核与资源计算

记 `S=inputsTensorUbSize`（elements）、`D=sizeof(T)`、平台 UB 为 `U`：

```text
FP32: ubBytesPerElement=4*D=16; S=floor_align((U/16)*4,32)/4; UB_total=4*S*4=16S
FP16/BF16: ubBytesPerElement=4*D+4=12; S=floor_align((U/12)*2,32)/2; UB_total=4*S*2+4S=12S
```

每个 in/out Queue 均 depth2、单 slot `S*D` bytes；FP16/BF16 的 `castBuf_` 为 `S*sizeof(float)=4S` bytes。Tiling 先以 int64 计算 S，写入 POD 的 uint32 字段时没有显式范围检查；平台 UB 规模使正常值远低于 uint32 上限，但迁移时不得忽略这个窄化边界。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 cast | 10001；`ForeachSqrtRegbase<half,...>` | regbase selector 命中且 `dataType_==DT_FLOAT16` | FP32/BF16；非法 dtype key 0 |
| 2 | R-002 FP32 direct | 10002；`ForeachSqrtRegbase<float,...>` | regbase selector 命中且 `dataType_==DT_FLOAT` | BF16；非法 dtype key 0 |
| 3 | R-003 BF16 cast | 10004；`ForeachSqrtRegbase<bfloat16_t,...>` | regbase selector 命中且 `dataType_==DT_BF16` | 非支持 dtype，无 Device fallback |

```text
regbase SoC
  -> priority 10000 membase: GRAPH_PARAM_INVALID
  -> priority 30000 ForeachRegbaseTilingUnary
       FLOAT16  -> key 10001 -> half cast route
       FLOAT32  -> key 10002 -> float direct route
       BFLOAT16 -> key 10004 -> BF16 cast route
       otherwise -> key 0 -> no body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 10001 | 950、FP16、1～50 Tensor、合法 shape/output | 单 AIV Kernel | `ForeachSqrtRegbase<half>` → Cast/Sqrt/Cast | in/out Queue depth2 + FP32 TBuf；Queue 同步，Vector 程序序 | 所有 tile 都用 exact-byte `DataCopyPad` |
| R-002 FP32 | 10002 | 950、FP32、同上 | 单 AIV Kernel | `ForeachSqrtRegbase<float>` → direct Sqrt | in/out Queue depth2，无 cast TBuf | `ceil(dataCount/S)`，末 tile exact count |
| R-003 BF16 | 10004 | 950、BF16、同上 | 单 AIV Kernel | `ForeachSqrtRegbase<bfloat16_t>` → Cast/Sqrt/Cast | 与 R-001 同布局 | 同 R-001，无独立 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| RegBase selector | `IsRegbaseSocVersion(context_)` | 950 使用 arch35 RegBase entry/POD | 全部 | membase 不可用；registry 最终失败 | `ForeachRegbaseTiling::IsCapable` |
| Tensor 数 | `1<=totalTensorCount_<=50` | POD Tensor 数组固定 50；ListTensor route | 全部 | `GRAPH_FAILED` | `ForeachRegbaseTiling::GetShapeAttrsInfo` |
| Core 数 | `numBlocks=min(AIV,80,ceil(N/B))` | POD core 数组固定 80 | 全部 | core 数截到 80；空域强制 1 | `DoOpTiling` |
| 4096-byte 分核粒度 | `B=4096/sizeof(T)` | core interval 按 4096-byte block 分配 | 全部 | 末 core/末 Tensor保留精确 offset | `DoOpTiling/AssignDataToEachCore` |
| UB divisor | FP32 `16`；FP16/BF16 `12` bytes/element | Queue depth2×2，加可选 FP32 TBuf | 全部 | dtype size≤0 时失败；资源不足无 fallback | `ForeachRegbaseTilingUnaryScalar::DoOpTiling` |
| 最小 UB/tile | 对齐后必须有 `S>0`，但源码未显式检查 | `CEIL_DIV(dataCount,maxDataCount)` 以 S 为除数 | 全部 | 固定仓无 fallback；平台 UB 必须保证 | unary-scalar `DoOpTiling` / `SingleTensorProcess` |
| DMA tail | 每 tile `blockLen=dataCount*sizeof(T)` | `DataCopyPad` 支持非 32-byte 有效长度 | 全部 | 同 Route exact-byte tail | `ForeachRegbaseUnary::CopyIn/CopyOut` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | uint32/32 | 第1字段；最终 `sizeof/alignof/offsetof` 固定仓不可见 | elements/Queue slot | unary-scalar `DoOpTiling` 的 S | `ParseTilingData`→`maxDataCount`/InitBuffer | Host int64→uint32 无显式上界检查 |
| 同上 | `tensorDataCountList` | int64[50] | 第2字段；400-byte payload；最终 offset 固定仓不可见 | elements/Tensor | `GetShapeAttrsInfo` | `Process` 默认 cursorEnd | 前 totalTensorCount 项有效 |
| 同上 | `tensorStartList` | uint16[80] | 第3字段；160-byte payload；最终 offset 固定仓不可见 | Tensor index/core | `AssignDataToEachCore` | `ParseTilingData` | blockIdx<numBlocks≤80 |
| 同上 | `tensorEndList` | uint16[80] | 第4字段；160-byte payload；最终 offset 固定仓不可见 | Tensor index/core | 同上 | `ParseTilingData` | inclusive |
| 同上 | `tensorStartOffsetList` | int64[80] | 第5字段；640-byte payload；最终 offset 固定仓不可见 | elements | 同上 | 首 Tensor cursorStart | inclusive |
| 同上 | `tensorEndOffsetList` | int64[80] | 第6字段；640-byte payload；最终 offset 固定仓不可见 | elements | 同上 | 尾 Tensor cursorEnd | inclusive；空域为 -1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10001/regbase | `ForeachRegbaseTilingUnary` | `ForeachSoloTilingDataRegbase` | arch35 `foreach_sqrt` key 10001 | `ForeachSqrtRegbase<half>` + `ForeachRegbaseUnary` | CMake ascend950 + binary JSON |
| R-002 | 10002/regbase | 同上 | 同上 | key 10002 | float specialization | 同上 |
| R-003 | 10004/regbase | 同上 | 同上 | key 10004 | BF16 specialization | 同上 |

## Kernel 实现

### 公共执行流程

entry 创建一个 `TPipe`，按 key 调 `ForeachSqrtImpl<T>`，解析 `ForeachSoloTilingDataRegbase`，初始化 `ForeachSqrtRegbase`，执行后销毁 pipe。基类用 `ListTensorDesc` 解码输入/输出地址；每 core 按 POD interval 逐 Tensor，按 `maxDataCount=S` 切 tile。每 tile执行 `CopyIn(DataCopyPad) → DeQue/Alloc → pred.Compute → Free input/EnQue output → CopyOut(DataCopyPad)`。Queue 的生命周期操作承担 MTE/Vector 阶段衔接；源码没有显式 event、barrier、atomic 或 workspace 访问。

### Route R-001：FP16 升精度 Sqrt

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host selector | `foreach_regbase_tiling.h` | `ForeachRegbaseTiling::IsCapable` | `IsRegbaseSocVersion(context_)` | regbase Route |
| Host key | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT16` | 10001 |
| Router | `foreach_sqrt/op_kernel/arch35/foreach_sqrt.cpp` | `foreach_sqrt` | `TILING_KEY_IS(FOREACH_TILING_KEY_HALF)` | `ForeachSqrtImpl<half>` |
| Body | `foreach_sqrt_regbase.h` | `Compute<half>` | `IsSameType<T,half>` | Cast→Sqrt→Cast |

#### Device 数据流

##### 函数级调用链

```text
foreach_sqrt key10001
  -> ForeachSqrtImpl<half>
    -> ForeachSqrtRegbase<half,ForeachSoloTilingDataRegbase>::Init
      -> ForeachRegbaseUnary::Init + castBuf_.InitBuffer
    -> ForeachRegbaseUnary::Process/SingleTensorProcess
      -> CopyIn -> base Compute -> derived Compute<half> -> CopyOut
```

##### 源码等价伪码

```text
core = GetBlockIdx(); parse tensor interval
for tensor i in [start,end]:
  begin/end = POD offsets or full tensor bounds
  bind in/out ListTensor address + begin
  for tile k in [0,ceil(count/S)):
    valid = last ? count-(q-1)*S : S
    DataCopyPad x[k*S] -> inLocal, blockCount=1, blockLen=valid*2 bytes
    castLocal[0:valid] = Cast<half,float>(inLocal) CAST_NONE
    Sqrt(castLocal,castLocal,valid)
    outLocal = Cast<float,half>(castLocal) CAST_RINT
    DataCopyPad outLocal -> y[k*S], blockLen=valid*2 bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM ListTensor x/y | `GetDataPtr<T>(i)+cursorStart+k*S` | valid elements | FP16 ND | framework/caller | base CopyIn/CopyOut | stream | exact-byte DataCopyPad |
| `inQueue` | UB | depth2 × `2S` bytes | FP16 VECIN | CopyIn | Cast | TPipe/core | double buffer |
| `outQueue` | UB | depth2 × `2S` bytes | FP16 VECOUT | cast-back | CopyOut | TPipe/core | double buffer |
| `castBuf_` | UB | `4S` bytes | FP32 VECCALC | first Cast | Sqrt/second Cast | TPipe/core | one TBuf reused every tile |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue FP16 · depth=2 · 4S bytes"]:1
  cast["castBuf FP32 · 4S bytes"]:2
  outQ["outQueue FP16 · depth=2 · 4S bytes"]:1
```

##### 字节账与复用顺序

```text
S=floor_align((U/12)*2,32)/2 elements
UB_total=2*S*2 + 2*S*2 + S*4 = 12S <= U
dynamic_ub=N/A
workspace_total=32 bytes reserved; body-used bytes=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | CopyIn | DeQue→Cast | `FreeTensor` 后 | Queue lifecycle | first Cast 已消费 |
| castBuf | Cast NONE / in-place Sqrt | Cast RINT | 下一 tile first Cast | 同一 Vector 程序序 | cast-back 已完成 |
| outQueue slot | Cast RINT | DeQue→CopyOut | `FreeTensor` 后 | Queue lifecycle | GM copy 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→Vector | `EnQue/DeQue` | CopyIn | Compute | inQueue slot | exact valid elements，无 pad 参与 Sqrt |
| Vector chain | 同一线程程序序 | Cast NONE | Sqrt→Cast RINT | castBuf/outLocal | FP16→FP32→FP16，RINT |
| Vector→MTE3 | `EnQue/DeQue` | Compute | CopyOut | outQueue slot | blockLen=`valid*2` bytes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | unary-scalar `DoOpTiling` | 12 bytes/element、S、POD、blockDim |
| Kernel entry/router | `arch35/foreach_sqrt.cpp` | key 10001/`ForeachSqrtImpl<half>` | key→body |
| Device body/helper | `foreach_sqrt_regbase.h` + `foreach_regbase_unary.h` | `Init/Compute` + base loop | FP32 cast buffer、Queue、DMA tail |

### Route R-002：FP32 原类型 Sqrt

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host key | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT` | 10002 |
| Router | `arch35/foreach_sqrt.cpp` | `FOREACH_TILING_KEY_FLOAT` | true | `ForeachSqrtImpl<float>` |
| Body | `foreach_sqrt_regbase.h` | `Compute<float>` | cast constexpr 为 false | direct Sqrt |

#### Device 数据流

##### 函数级调用链

```text
foreach_sqrt key10002 -> ForeachSqrtImpl<float>
  -> ForeachSqrtRegbase<float>::Init -> ForeachRegbaseUnary::Init
  -> Process -> CopyIn -> Compute -> Sqrt(outLocal,inLocal,count) -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 相同的 core/Tensor/tile 循环；S 个 FP32 元素/slot；
每 tile 以 blockLen=valid*4 bytes 搬入，Sqrt(outLocal,inLocal,valid)，再 exact-byte 搬出。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `GetDataPtr<float>(i)+begin+k*S` | valid×4 bytes | FP32 ND | caller/CopyIn | CopyIn/caller | stream | exact-byte DMA |
| `inQueue` | UB | depth2×4S | FP32 VECIN | CopyIn | Sqrt | core | double buffer |
| `outQueue` | UB | depth2×4S | FP32 VECOUT | Sqrt | CopyOut | core | double buffer |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue FP32 · depth=2 · 8S bytes"]:1
  outQ["outQueue FP32 · depth=2 · 8S bytes"]:1
```

##### 字节账与复用顺序

```text
S=floor_align((U/16)*4,32)/4 elements
UB_total=2*S*4 + 2*S*4 = 16S <= U
dynamic_ub=N/A; workspace_total=32 bytes reserved, body uses 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| in/out Queue slots | CopyIn/Sqrt | Sqrt/CopyOut | `FreeTensor` 后 | Queue lifecycle | producer 已完成、consumer 已释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue pipeline | EnQue/DeQue/Free | MTE2/Vector | Vector/MTE3 | in/out slots | FP32 direct；末 tile exact valid count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | unary-scalar `DoOpTiling` | FP32 divisor 16、S |
| Kernel entry/router | `arch35/foreach_sqrt.cpp` | key 10002 | float body |
| Device body/helper | `foreach_sqrt_regbase.h` | `Compute<float>` | direct Sqrt |

### Route R-003：BF16 升精度 Sqrt

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host key | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_BF16` | 10004 |
| Router | `arch35/foreach_sqrt.cpp` | `FOREACH_TILING_KEY_BF16` | true | `ForeachSqrtImpl<bfloat16_t>` |
| Body | `foreach_sqrt_regbase.h` | `Compute<bfloat16_t>` | BF16 constexpr | Cast→Sqrt→Cast |

#### Device 数据流

##### 函数级调用链

```text
foreach_sqrt key10004 -> ForeachSqrtImpl<bfloat16_t>
  -> ForeachSqrtRegbase<bfloat16_t>::Init (base queues + FP32 castBuf)
  -> Process -> CopyIn -> Cast NONE -> Sqrt<float> -> Cast RINT -> CopyOut
```

##### 源码等价伪码

```text
按 R-001 的 interval 与 tile 循环；T=BF16：
castLocal = Cast<BF16,float>(inLocal, CAST_NONE, valid)
Sqrt(castLocal,castLocal,valid)
outLocal = Cast<float,BF16>(castLocal, CAST_RINT, valid)
DataCopyPad exact valid*2 bytes to output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `GetDataPtr<bfloat16_t>(i)+begin+k*S` | valid×2 bytes | BF16 ND | caller/CopyIn | CopyIn/caller | stream | exact-byte DMA |
| in/out Queue | UB | 各 depth2×2S | BF16 | CopyIn/cast-back | Cast/CopyOut | core | double buffer |
| `castBuf_` | UB | 4S bytes | FP32 VECCALC | Cast NONE | Sqrt/Cast RINT | core | 原地 Sqrt、逐 tile复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue BF16 · depth=2 · 4S bytes"]:1
  cast["castBuf FP32 · 4S bytes"]:2
  outQ["outQueue BF16 · depth=2 · 4S bytes"]:1
```

##### 字节账与复用顺序

```text
S=floor_align((U/12)*2,32)/2 elements
UB_total=4S+4S+4S=12S <= U
dynamic_ub=N/A; workspace_total=32 bytes reserved, body uses 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castBuf | BF16→FP32 Cast / Sqrt | FP32→BF16 Cast | 下一 tile | Vector 程序序 | cast-back 完成 |
| Queue slots | CopyIn/cast-back | Cast/CopyOut | Free 后 | Queue lifecycle | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue/Vector | Queue + 程序序 | CopyIn/Cast/Sqrt | 后继阶段 | Queue/castBuf | BF16→FP32、FP32 Sqrt、RINT→BF16；valid count 控制 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | unary-scalar `DoOpTiling` | BF16 divisor 12、S |
| Kernel entry/router | `arch35/foreach_sqrt.cpp` | key 10004 | BF16 body |
| Device body/helper | `foreach_sqrt_regbase.h` + `foreach_regbase_unary.h` | derived `Compute` + base CopyIn/Out | FP32 cast pipeline 与 exact tail |

## 算子特定优化

### Trick T-001：FP16/BF16 共享单个 FP32 TBuf 并原地执行 Sqrt

#### 触发与源码依据

- 适用 Route：R-001、R-003。
- 触发条件：950 regbase，`T` 为 `half` 或 `bfloat16_t`，key 10001/10004。
- Golden 来源：[`foreach_sqrt_regbase.h:39-47`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sqrt/op_kernel/arch35/foreach_sqrt_regbase.h#L39-L47)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Cast(this->castLocal_, inLocal, RoundMode::CAST_NONE, dataCount);
Sqrt(this->castLocal_, this->castLocal_, dataCount);
Cast(outLocal, this->castLocal_, RoundMode::CAST_RINT, dataCount);
```

#### 机制与设计原因

低精度 Route 只分配一个 `S*sizeof(float)` 的 `castBuf_`。第一步把输入升到 FP32，Sqrt 以同一 `castLocal_` 作为 src/dst 原地覆盖，最后直接降精度写到 outQueue；因此不需要第二个 FP32 结果 buffer。Host 的 12 bytes/element 预算精确覆盖 in/out 两个 depth2 低精度 Queue（8 bytes/element）和这一 FP32 TBuf（4 bytes/element）。

#### 迁移边界

- 适用：可在单一 FP32 scratch 上原地执行、最终只需一个低精度输出的逐元素 unary op。
- 前提：Cast/Sqrt/Cast 的程序序不能被打乱，S 必须按 12 bytes/element 预算且 32-byte 向下对齐，输出 Queue 必须在 scratch 下次覆盖前完成 cast-back。
- 失效：FP32 使用 R-002 且不分配 castBuf；需要同时保留原 FP32 输入和结果的算法不能套用单 TBuf 账。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；实现复用时若额外分配第二个 FP32 buffer而不重算 Host divisor，会突破源码证明的 UB 上界。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 10001 | key 10002 | key 10004 | regbase 模板命中后按 dtype |
| 模板与 dtype | `ForeachSqrtRegbase<half>` | `<float>` | `<bfloat16_t>` | 低精度共用 constexpr cast 分支 |
| 循环与分核 | 4096-byte block；S-element tile | 同 | 同 | 公共 RegBase Unary |
| Queue/TBuf/寄存器/dynamic UB/通信 window | in/out depth2 + FP32 TBuf | in/out depth2 | in/out depth2 + FP32 TBuf | dtype 决定条件分配 |
| 内存总占用与复用 | 12S，cast TBuf 原地 | 16S，无 scratch | 12S，cast TBuf 原地 | Host divisor 与 InitBuffer 对应 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + Cast/Sqrt/Cast | DataCopyPad + Sqrt | 同 R-001 | 全部 AIV_VECTOR |
| 同步、精度与 tail | Queue；FP32 Sqrt；RINT | Queue；FP32 direct | Queue；FP32 Sqrt；RINT | valid count 控制 exact tail |
| fallback/失败边界 | 无 Device fallback | 无 Device fallback | 无 Device fallback | 非法 key 由 Host 阻断 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持；公开 950PR/950DT |
| Host/Launcher | non-regbase membase priority 10000 | 同 A2 | regbase priority 30000、4096-byte 分核、80-core POD |
| Kernel/资源 | implicit-output 原地 Queue；仅 BF16 升精度 | 同 A2 | arch35 独立 in/out Queue；FP16/BF16 都用 FP32 TBuf |

## 限制与失败边界

- 950 仅支持 FP16/FP32/BF16、ND、1～50 个 Tensor、每 Tensor 0～8 维；对应输出 shape size 按公开契约必须等于输入。负数输入特殊值由 AscendC `Sqrt` 决定，固定仓无额外承诺。
- Host Tiling 实现的 `CheckOutput` 只拒绝输出 storage numel 小于输入，而公开 950 文档要求相等；工程调用必须服从更严格的公开契约，不能利用 Host 实现的宽松条件扩张接口域。
- `totalDataCount_` 为 int64 和，shape 乘积/总和的极端溢出保护固定仓不可见；S 从 int64 写到 uint32 POD 也无显式上界检查。
- Host 未显式检查 32-byte 对齐后的 S 是否为 0；如果移植到 UB 过小的平台，Device 的 `CEIL_DIV(dataCount,S)` 会失去合法除数。950 固定平台配置必须作为资源前提保留。
- unknown key 无 Device fallback；`TPipe::Destroy` 后直接结束且不写输出。Tiling/OpDef 的 dtype、shape 和平台 guard 不可省略。
- 手写 ACLNN bridge、executor 内部前处理节点、`AutoContiguous` 的 workspace 切片及 32-byte 保留 workspace 的框架用途在固定仓不可见。
- `ForeachSoloTilingDataRegbase` 的字段声明与 2004-byte payload 可见，但生成宏来自固定仓外部依赖；最终 `sizeof/alignof/offsetof` 固定仓不可见。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sqrt/docs/aclnnForeachSqrt.md` | 两段式原型/产品表 | 950 API、错误与 shape/Tensor 数约束 |
| 入口声明/约束 | `foreach/foreach_sqrt/README.md` | 产品支持/语义 | 950 支持交叉证据 |
| 入口实现 | `foreach/foreach_sqrt/op_host/foreach_sqrt_def.cpp` | `ops::ForeachSqrt` | dtype/ND/AutoContiguous/ascend950 |
| 入口声明/约束 | `foreach/foreach_sqrt/op_graph/foreach_sqrt_proto.h` | `REG_OP(ForeachSqrt)` | 动态列表 dtype 与 50 项边界 |
| 入口实现 | `foreach/foreach_sqrt/op_host/CMakeLists.txt` | `ACLNNTYPE aclnn` | 生成 bridge 边界 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachSqrtTiling`/template registrations | registry 与优先级 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachSoloTilingDataRegbase`/membase macro | 950 POD 与互斥 selector |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_regbase_tiling.h` | `ForeachRegbaseTilingUnary` | regbase `IsCapable` 与继承关系 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | base/unary Tiling methods | 校验、分核、UB、key、workspace |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 顺序与 `GRAPH_PARAM_INVALID` fallthrough |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | Host Tiling 阶段顺序与失败传播 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_util.h` | `IsRegbaseSocVersion` | RegBase 架构集合判定 |
| Kernel entry/body | `foreach/foreach_sqrt/op_kernel/arch35/foreach_sqrt.cpp` | `foreach_sqrt`/`ForeachSqrtImpl` | key router、POD parse、TPipe lifetime |
| Kernel entry/body | `foreach/foreach_sqrt/op_kernel/arch35/foreach_sqrt_regbase.h` | `ForeachSqrtRegbase::Init/Compute` | cast buffer与三 dtype计算 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_common.h` | key macros | 10001/10002/10004 ABI |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h` | `ForeachRegbaseUnary` | ListTensor、Queue、循环、DMA tail |
| 构建接线 | `foreach/foreach_sqrt/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_sqrt.cpp` | ascend950 绑定 arch35 body |
| 构建接线 | `foreach/foreach_sqrt/op_host/config/ascend950/foreach_sqrt_binary.json` | `op_list` | 三 dtype binary config |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段 RegBase POD | `inputsTensorUbSize` 是 uint32 elements；Tensor 50、core 80；宏生成的最终 offset/sizeof 不可外推 | `ForeachSoloTilingDataRegbase` |
| 调用顺序 | registry→RegBase Tiling→单 arch35 entry→Impl/Base/derived | membase 先拒绝、regbase 后命中；key 10001/2/4 | `TilingRegistry`/registrations |
| workspace/内存 | workspace body-used 0；两个 depth2 Queue；低精度条件分配 FP32 TBuf | divisor 16/12 与 InitBuffer 字节账必须一致 | unary-scalar `DoOpTiling`/`ForeachSqrtRegbase::Init` |
| 同步 | Queue lifecycle + Vector 程序序 | Free/覆盖点不可提前；无源码 event 可外推 | `ForeachRegbaseUnary::SingleTensorProcess` |
| 精度与尾块 | FP16/BF16 CAST_NONE→FP32 Sqrt→CAST_RINT；FP32 direct | `dataCount` 同时控制 Vector 与 exact-byte DMA | `ForeachSqrtRegbase::Compute`/`CopyIn/CopyOut` |
| 失败边界 | Host 任一校验失败阻断；Device unknown key 无 body | 950 公开 shape 相等约束严于 Host `>=` 实现 | `CheckOutput`/`foreach_sqrt` |