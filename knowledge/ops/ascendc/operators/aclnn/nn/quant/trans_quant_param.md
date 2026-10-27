---
type: Operator
title: "TransQuantParam A2/A3 ACLNN 算子知识"
description: "说明 aclnnTransQuantParamV3 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/docs/aclnnTransQuantParamV3.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/op_api/aclnn_trans_quant_param_v3.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/op_api/aclnn_trans_quant_param_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/trans_quant_param_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/trans_quant_param_v2_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/config/ascend910b/trans_quant_param_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/config/ascend910_93/trans_quant_param_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/build.sh
    role: implementation
created_at: "2026-08-25T18:33:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# TransQuantParam A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnTransQuantParamV3GetWorkspaceSize` + `aclnnTransQuantParamV3`，公开 ACLNN 两段式接口；V3 是 A2/A3 在固定仓中实际支持的最新公开版本。 |
| 计算语义 | 将 FP32 `scale` 编码为 FP19 位段并置 bit 46；可选 FP32 `offset` 以 RINT 取整、夹到 `[-256,255]`，再把低 9 bit 放入输出 bit 37–45。 |
| 输入输出 | `scale` FP32 ND，可选 `offset` FP32 ND，`roundMode` 为 0/1，`out` 公开允许 UINT64/INT64 ND；API 在调用 inner 前把输出内部 dtype 统一设为 UINT64。 |
| 执行类型 | 单个 `AIV_VECTOR` Kernel；`ASCEND_IS_AIC` 分支立即返回，无可达 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点。 |
| Kernel 路线 | R-001：无 key 的单一 body，内部按 `scaleLength`/`offsetLength` 是 0、1 或大于 1，以及 `roundMode` 动态分支。 |
| 核心资源 | GM 三路张量；`resUb_` 65536 B，`offsetUb_` 主循环 65536 B；标量 offset 预处理还会先对同一 TBuf 执行一次 64 B `InitBuffer`；无算子自有跨 Kernel workspace。 |
| 来源闭包 | `source_closed`：公开 V3 经 ACLNN inner 生成框架边界进入 `TransQuantParamV2`，算子拥有的 API 校验、OpDef、Tiling/POD、单 Kernel entry/body、A2/A3 binary config 与 AscendC 编译清单均闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `sbits=bitcast<uint32>(scale)`。`roundMode=0` 时：

```text
scalePart = (sbits & 0xFFFFE000) | (1ULL << 46)
```

`roundMode=1` 时先从 FP32 位段取 sign(1)、exponent(8)和高 10 位 mantissa：零指数直接仅保留 sign，全 1 指数保留高 10 位 mantissa，普通数用被截掉部分的 round bit、sticky bits 和当前最低位实现 ties-to-even；mantissa 进位可推高 exponent，指数溢出时形成 exponent=255、mantissa=0。得到 19 bit 后左移 13 位，再置 bit 46。

若 `offset` 存在，对每个有效元素执行 `oi=clamp(RINT(offset),-256,255)`，并以 `scalePart | ((uint64(oi)&0x1FF)<<37)` 产生输出。`scaleLength==1` 或 `offsetLength==1` 时对输出域广播；输出元素数为 `max(scaleLength,offsetLength)`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `scale` | 1D `(t,)` 或 2D `(1,n)` / ND | FP32 | 必选、非空，各维为正；A2/A3 API 对 2D 要求首维为 1。公开契约要求无 NaN/Inf。 |
| `offset` | 可选 1D `(t,)` 或 2D `(1,n)` / ND | FP32 | 存在时非空，2D 首维必须为 1；当 scale/offset 都是 1D 且长度都大于 1 时，API 要求两者长度相等。公开契约要求无 NaN/Inf。 |
| `roundMode` | 标量 | int64 | 只允许 0 或 1；0 为 V2 兼容截断，1 为手写 FP19 ties-to-even 舍入。 |
| `out` | 与公开 shape 契约匹配 / ND | UINT64 或 INT64 | 必选。API 允许两种 dtype，但在 inner 调用前以 `SetDataType(DT_UINT64)` 统一内部解释；Kernel 以 `uint64_t*` 写位模式。 |

### 入口与版本

公开版本族包含 Host 数组接口 `aclnnTransQuantParam`、两段式 V2 和 V3。V3 相比 V2 增加 `roundMode`；`aclnnTransQuantParamV3GetWorkspaceSize` 校验后调用 `aclnnInnerTransQuantParamV2GetWorkspaceSize(...,roundMode,...)`，第二段 V3 同样复用 `aclnnInnerTransQuantParamV2`。因而 A2/A3 本卡只提取最新公开 V3，但追踪到真实 `TransQuantParamV2` Host/Kernel body。

V3 文档明确支持 Atlas A2/A3；OpDef 注册 `ascend910b`/`ascend910_93`，两个 binary JSON 均提供完整 I/O 与 `round_mode`，全局 `ascendc_config.json` 也把 `TransQuantParamV2` 编译到这两个 compute unit。`build.sh::SOC_TO_ARCH` 将二者均映射为 2201。

## Host/Launcher 执行流程

### 参数校验与前处理

V3 第一段按顺序执行 `CheckNotNull -> CheckDtypeValid -> CheckFormat -> CheckShape`：`scale/out` 不得为空；scale/offset 必须 FP32，out 必须 UINT64 或 INT64；存储格式均为 ND；A2/A3 的 scale 只接受正长的 1D 或 `(1,n)`，offset 同样只接受正长 1D 或 `(1,n)`；`roundMode` 只能为 0/1。通过后把 `out` 的 executor 内部 dtype 改为 UINT64，进入自动生成的 ACLNN inner 边界。入口没有可见 `Contiguous/ViewCopy`，文档对 scale/offset 标记不支持非连续 Tensor。

Tiling 再检查 POD 容量、scale shape、attrs 和 platform/workspace 指针。它从 storage shape 将元素数窄化为 `uint32_t`，从 attr 0 生成 `roundMode`，然后写 POD。

### 参数、Workspace 与生命周期

Tiling 设 `blockDim=1`，无 TilingKey。`workspace[0]=PlatformAscendC(platformInfo).GetLibApiWorkSpaceSize()`；entry 要求传入 workspace 和 `GetUserWorkspace(workspace)` 均非空，但 body 不读写 user workspace，所以算子自有中间 slice 为 0，非零大小属于 AscendC 库/executor 保留区。调用者必须使用第一段返回的总 workspace，并保持 workspace、executor、输入输出在第二段 stream 任务完成前有效。

### 启动图与失败传播

```text
aclnnTransQuantParamV3GetWorkspaceSize(scale, offset, roundMode, out)
  -> CheckParams; outX->SetDataType(UINT64)
  -> aclnnInnerTransQuantParamV2GetWorkspaceSize(...)  # 生成框架边界
  -> TransQuantParamV2::TilingFunc
     -> POD={scaleLength,offsetLength,roundMode}, blockDim=1, workspace=Wlib
aclnnTransQuantParamV3(workspace, workspaceSize, executor, stream)
  -> aclnnInnerTransQuantParamV2(...)
  -> K-001 trans_quant_param_v2<<<1, stream>>>
     -> TransQuantParamV2::Init -> Process -> GM out
```

K-001 输入 scale/offset GM、输出 y GM、library workspace 和由 3 个 `uint32_t` 字段组成的 POD；宏展开后的精确 ABI 字节数在固定仓不可见。没有 Kernel 间边。API 或 Tiling 校验失败会阻断 launch；entry 发现 workspace/user workspace 为空时静默返回且不写输出；AIC 上下文或 `blockIdx>0` 也不执行 body。

## Route 与参数设计

### 分核与资源计算

Host 固定一核。令 `L=max(scaleLength,offsetLength)`、`E=65536/sizeof(uint64)=8192`：`loops=floor(L/E)`，`tailLength=L-loops*E`。每个完整 tile 处理 8192 个输出并搬出 65536 B；tail 只提交 `tailLength*8` B。`offsetUb_` 主分配量为 `2*align32(E*4)=65536` B，`resUb_` 为 65536 B，主循环 UB 合计 131072 B。`offsetLength==1` 时还先调用 `InitBuffer(offsetUb_,2*align32(4))=64 B`；同一 TBuf 重新 `InitBuffer` 后旧分配是否退还由外部 `TPipe` 实现决定，固定仓不可见，保守的本次 Process 分配请求总量为 131136 B。Host 没有读 UB 大小或执行 `IsCapable` 检查。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 单核 FP19 编码 | 无 key / `TransQuantParamV2` | 所有通过 API 与 Tiling 校验的 A2/A3 V3 调用；`roundMode`、长度 0/1/>1 只改变 body 内分支，不改变 POD 布局、启动图或内存上界 | 校验失败，无 fallback |

```text
if API/Tiling validation fails: no launch
else: blockDim=1, no tiling key -> R-001
  roundMode==0: truncate FP32 low 13 bits
  roundMode==1: scalar RNE FP32->FP19 bits
  scaleLength==1: preload+broadcast; else per-output GM scalar load
  offsetLength==0: no offset
  offsetLength==1: preload/vector-convert+broadcast
  offsetLength>1: tiled DMA+vector-convert
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 无 key | A2/A3，V3 API 校验通过，FP32 scale/可选 offset，`roundMode in {0,1}` | 单 K-001 AIV_VECTOR，`blockDim=1` | `TransQuantParamV2::Process` 及 `CalQuantPreScale*`/`CalOffset*` | 标量 scale 位运算 + offset DMA/Vector Cast/Maxs/Mins；`PIPE_ALL`和 MTE2_V/V_MTE3/MTE3_S 事件 | 8192 元素切片；tail 以 DataCopyPad 按字节提交；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 公开平台与构建 | V3 文档支持 A2/A3；OpDef/config/全局编译清单均含 910b/910_93 | 确保 Host 与 Kernel binary 同时可达 | R-001 | 其他平台不归入本卡 | `TransQuantParamV2::TransQuantParamV2`；`ascendc_config.json` |
| UB 上界 | Host 无检查 | 主循环需 131072 B；标量 offset 路径有额外 64 B 初始化请求 | R-001 | 固定仓无 fallback/报错 | `Process`；`CalOffsetValueShapeOne` |
| 长度窄化 | `GetShapeSize()` 赋给 `uint32_t` | POD 和 Device 地址/循环均为 32 bit | R-001 | 溢出时截断，固定仓无上界校验 | `TilingFunc` |
| shape 配对 | 只对“两者均 1D 且长度>1”检查相等；未检查 out shape | body 按 `max(lengths)` 写输出，非 1 的较短输入不会广播 | R-001 | 公开 shape 契约之外可越界，无 Device guard | `CheckShape`；`Process` |
| 8192 切片 | `E=65536/8` 固定 | 使 `resUb_` 恒为 64 KiB，完整片使用一次连续 DMA | R-001 | 最后进入 tail | `Process` |
| AIV-only | `ASCEND_IS_AIC` 或 `blockIdx>0` 返回 | body 使用 Vector Cast/Maxs/Mins，仅 block 0 产生结果 | R-001 | 不写 out | `Init`；`Process` |
| 平台搬出 | A2/A3 映射 arch 2201，命中 `__CCE_AICORE__>200` | tail 用 `DataCopyPad(Global,Local,{count,byteLen,...})`，不进入 rollback fallback | R-001 | 其他 arch 才进 `SafeDataCopy` | `DataCopyGlobal`；`SOC_TO_ARCH` |

### POD/ABI

`TransQuantParamV2TilingData` 按声明顺序仅含 3 个 `uint32_t`。固定仓未包含 `tilingdata_base.h` 宏展开，因此精确 `sizeof/alignof/offsetof` 不可见；Host `SetDataPtr`/字段 setter 与 Device `GET_TILING_DATA` 使用同一注册结构，字段顺序不得改变。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `TransQuantParamV2TilingData` | `scaleLength` | uint32/32 | 第 1 字段；精确 offset 不可见 | 元素 | `uint32_t(scaleStorageShape.GetShapeSize())` | `Init`、`Process`、`CalQuantPreScale*` | 必须 >0；int64→uint32 窄化无 guard |
| 同上 | `offsetLength` | uint32/32 | 第 2 字段；精确 offset 不可见 | 元素 | offset 缺失为 0，否则 `uint32_t(offsetStorageShape.GetShapeSize())` | `Process`、`CalOffsetValueShapeOne`、`CalOffset` | 0/1/>1 分别为缺失/广播/逐元素 |
| 同上 | `roundMode` | uint32/32 | 第 3 字段；精确 offset 不可见 | 枚举 0/1 | attr 0 缺失时 0，否则 `uint32_t(*attrs->GetInt(0))` | `CalQuantPreScaleValue` | Host/API 与 Tiling 均检查 0/1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 无 key；单 body | `trans_quant_param_v2_tiling.cpp::TilingFunc` | `TransQuantParamV2TilingData` 三字段 | `trans_quant_param_v2.cpp::trans_quant_param_v2` | `trans_quant_param_v2.h::TransQuantParamV2` 全方法 | A2/A3 binary JSON + `ascendc_config.json::TransQuantParamV2` + `op_host/CMakeLists.txt` |

## Kernel 实现

### 公共执行流程

唯一 entry 检查 workspace，解析 POD，在栈上创建 `TPipe` 和 `TransQuantParamV2`。`Init` 绑定三个 GM tensor；`Process` 先预处理可广播的单 offset 和单 scale，再按 8192 输出切片。scale 的位编码用标量 `GetValue/SetValue` 循环；offset>1 先 GM→UB，再原地以 Vector `Cast(CAST_RINT)->Maxs(-256)->Mins(255)` 处理，最后以标量循环 OR 到 result UB。完整 tile 用对齐 `DataCopy`，tail 在 A2/A3 用 `DataCopyPad`。

### Route R-001：单核 FP19 编码

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_trans_quant_param_v2.cpp` | `aclnnTransQuantParamV3GetWorkspaceSize` | `CheckParams(...,roundMode,...)==ACLNN_SUCCESS` | inner `TransQuantParamV2` |
| Tiling | `op_host/trans_quant_param_v2_tiling.cpp` | `TilingFunc` | POD 容量足够，shape/attr/platform/workspace 有效 | `blockDim=1`，无 key |
| Entry | `op_kernel/trans_quant_param_v2.cpp` | `trans_quant_param_v2` | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | `TransQuantParamV2::Init/Process` |
| Body | `op_kernel/trans_quant_param_v2.h` | `Process` | `!ASCEND_IS_AIC && GetBlockIdx()==0` | 唯一实现体 |

#### Device 数据流

##### 函数级调用链

```text
trans_quant_param_v2(scale, offset, y, workspace, tiling)
  -> TransQuantParamV2::Init
  -> Process
     -> CalOffsetValueShapeOne
        -> DataCopyLocal -> Cast(CAST_RINT) -> Maxs -> Mins -> GetValue
     -> CalScaleValueShapeOne -> CalQuantPreScaleValue
     -> [full tiles]
        -> CalQuantPreScale -> CalQuantPreScaleValue
        -> [broadcast scalar offset OR | CalOffset]
           -> DataCopyLocal -> SetOffsetValue(Cast/Maxs/Mins/scalar OR)
        -> DataCopy GM y
     -> [tail]
        -> same compute -> DataCopyGlobal -> DataCopyPad
```

##### 源码等价伪码

```text
# 伪码；L=max(scaleLength,offsetLength), E=8192
if offsetLength==1:
  DMA 1 FP32 offset -> offsetUb
  oi = Mins(Maxs(CastRint(offset), -256), 255)[0]
  broadcastOffsetBits = (uint64(oi)&0x1ff)<<37
if scaleLength==1:
  broadcastScaleBits = EncodeScale(bitcast_u32(scale[0]), roundMode)

allocate offsetUb[65536 B], resUb[65536 B]
for tile in 0 .. floor(L/E)-1:
  for i in 0 .. E-1:
    s = broadcastScaleBits if scaleLength==1
        else EncodeScale(bitcast_u32(scale[tile*E+i]), roundMode)
    res[i] = s
    if offsetLength==1: res[i] |= broadcastOffsetBits
  if offsetLength>1:
    DMA offset[tile*E : +E] -> offsetUb
    in-place CastRint -> Maxs(-256) -> Mins(255)
    for i: res[i] |= (uint64(offsetInt[i])&0x1ff)<<37
  event V_MTE3; DMA 65536 B res -> y[tile*E]
  event MTE3_S before scalar reuse

tail=L%E
for i in 0 .. tail-1: repeat same scale/broadcast work
if offsetLength>1: CalOffset(tail, floor(L/E)*E, res)
if tail!=0: event V_MTE3; DataCopyPad tail*8 B -> y[floor(L/E)*E]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM scale | `gmIdx=tile*8192+i`；广播时 0 | `scaleLength` | FP32 ND | caller | `CalQuantPreScale*` | ACLNN 调用期 | 标量 `GetValue`；长度 1 只读一次 |
| GM offset | `gmOffset=tile*8192`；广播时 0 | `offsetLength` | FP32 ND | caller | `CalOffset*` | ACLNN 调用期 | >1 按 tile DMA；1 只 DMA 一次 |
| GM y | `tile*8192` | `L=max(lengths)` | UINT64 bit pattern ND | CopyOut | caller | ACLNN 调用期 | full 65536 B；tail 精确 `tail*8` B |
| `resUb_` | 0 | 65536 B = 8192 UINT64 | UINT64 | scale 编码/offset OR | GM CopyOut | K-001 Process | 每 tile 整块复用；有效 tail 才外写 |
| `offsetUb_` 主分配 | 0 | 65536 B = `2*8192*4` | FP32/INT32 alias | offset DMA/Cast | Maxs/Mins/scalar OR | K-001 Process | `Get<float>(alignedLength)` 与 `Get<int32_t>(alignedLength)` 均从 0 取，原地覆盖；后 32768 B 未被显式寻址 |
| `offsetUb_` 标量预处理 | 0 | 64 B = `2*align32(4)` | FP32/INT32 alias | 1 元素 DMA/Cast | `offetInt9Bit_` | 主分配之前 | 同一 TBuf 之后再 `InitBuffer(65536)` |
| scalar members | 寄存器/标量存储 | 2×uint64 | bit fields | `Cal*ShapeOne` | 每元素循环 | K-001 | 只在对应 length==1 时有效 |
| library workspace | `GetUserWorkspace(workspace)` | `Wlib=GetLibApiWorkSpaceSize()` | 外部库布局 | ACLNN/executor | entry 仅判空 | 第二段 stream 任务 | body 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  off["offsetUb_ · 65536 B · FP32→INT32 in-place"]:2
  res["resUb_ · 65536 B · UINT64"]:2
```

`offsetLength==1` 时，上图之前还对 `offsetUb_` 请求 64 B 标量预处理分配；固定仓未展开同一 TBuf 再次 `InitBuffer` 的 allocator 回收语义，不把这 64 B 臆测为已释放。

##### 字节账与复用顺序

```text
UB_main = offsetUb(2 * align32(8192*4)) + resUb(65536)
        = 65536 + 65536 = 131072 B
offsetLength==1: prior InitBuffer request = 2 * align32(1*4) = 64 B
conservative Process allocation requests = 131136 B
dynamic_ub = N/A
workspace_total = Wlib; operator-owned workspace slice = 0
=> Host 没有 UB IsCapable，平台可用 UB 必须覆盖实际 TPipe 分配语义。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `resUb_[0:8192]` | scale 标量编码，后续 offset OR | full/tail CopyOut | 下一 tile scale 编码 | full tile 后 `MTE3_S` event；tail 前 `PIPE_ALL` | CopyOut 完成后才可改写 |
| `offsetUb_[0:alignedLength]` | GM offset FP32 DMA | Vector Cast | 同 tile Cast 原地改写为 INT32 | `MTE2_V` event | DMA 完成后 Cast |
| 同上 INT32 | Cast/Maxs/Mins | scalar `GetValue` + OR | 下一 tile DMA | `PIPE_V`/`PIPE_ALL` | 所有 offset 元素被 OR 后才重载 |
| 后半 `offsetUb_` | 无显式写入 | 无 | 下一 Init/Process | N/A | 分配量为前半的 2 倍，但两个 typed view 均从 offset 0 开始 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| offset GM→V | `MTE2_V` event | `DataCopyPad` | `Cast` 前 `WaitFlag` | `offsetUb_` | FP32 以 `CAST_RINT` 转 INT32，ties-to-even |
| offset Vector 链 | `PipeBarrier<PIPE_V/ALL>` | Cast，Maxs，Mins | 下一 Vector/标量 GetValue | 原地 INT32 | 夹到 [-256,255]，再保留二补码低 9 bit |
| scale 标量链 | `PIPE_ALL` | 广播标量预处理 | 主循环 | scalar members | mode 0 截低 13 bit；mode 1 手写 RNE，零指数与特殊指数单独分支 |
| result V/标量→MTE3 | `V_MTE3` event | `SetValue`/offset Vector 链 | GM CopyOut | `resUb_` | full tile 8192；tail 只写 `tailLength` |
| full tile CopyOut→标量复用 | `MTE3_S` event | 65536 B `DataCopy` | 下一 tile 标量写 | `resUb_` | full tile 无 padding |
| tail CopyOut | A2/A3 `DataCopyPad` | `resUb_` | GM y | 最后 `tail*8` B | `tail==0` 时不启动；未输出 UB lane 不外泄 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `quant/trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp` | `TilingFunc` | 三 POD 字段、库 workspace、blockDim=1，无 key |
| Kernel entry/router | `quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp` | `trans_quant_param_v2` | workspace 判空、POD 解析、唯一 body |
| Device body/helper | `quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h` | `Process`、`CalQuantPreScaleValue`、`CalOffset*`、`DataCopyGlobal` | 8192 切片、FP19 位舍入、offset Vector 处理、A2/A3 tail DataCopyPad |

## 算子特定优化

### Trick T-001：长度 1 的 scale/offset 预编码后广播

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：`scaleLength_==1` 或 `offsetLength_==1`
- Golden 来源：[`quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h:226-270`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h#L226-270)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (scaleLength_ == 1) {
    resTensor.SetValue(idx, scalequantPre_);
    return;
}
```

#### 机制与设计原因

`CalScaleValueShapeOne` 只读一次 GM 并调用一次 `CalQuantPreScaleValue`，之后每个输出只写入已编码的 `scalequantPre_`。单 offset 同样只 DMA/Cast/Clamp 一次，把 9 bit 结果保存在 `offetInt9Bit_`。这两个分支减少广播参数的重复 GM 访问和重复转换；输出仍按 L 逐元素写入。

#### 迁移边界

- 适用：长度 1 标量张量对更大输出域广播的位编码算子。
- 前提：广播值在整个 launch 中不变；scale/offset 编码不依赖输出索引。
- 失效：长度大于 1 时回到逐元素 scale 读取或分 tile offset DMA。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：固定 64 KiB 切片与精确 tail 提交

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：所有有效调用；`tailLength!=0` 时启用 tail CopyOut
- Golden 来源：[`quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h:139-181`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h#L139-181)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t eachLength = SPLIT_SIZE / sizeof(uint64_t);
uint32_t loops = Max<uint32_t>(scaleLength_, offsetLength_) / eachLength;
uint32_t tailLength = Max<uint32_t>(scaleLength_, offsetLength_) - loops * eachLength;
```

#### 机制与设计原因

`SPLIT_SIZE=65536` 同时限定 result UB 和完整 GM CopyOut，使每个满 tile 为 8192 个 UINT64。最后一片不把 padding lane 写回 GM，而把 `tailLength*8` 作为 DataCopyPad 字节长度，避免尾部越界与无效元素物化。

#### 迁移边界

- 适用：单核线性输出且平台支持 GM 方向 DataCopyPad 的算子。
- 前提：64 KiB result UB 与 offset UB 可同时分配；A2/A3 命中 `__CCE_AICORE__>200`。
- 失效：不支持 DataCopyPad 的旧 arch 进入 `SafeDataCopy` rollback 路径，不能直接复用本卡的 tail 搬运结论。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

`SafeDataCopy` 的旧 arch 分支明确注释，当有效字节小于一个 32 B block 时仍会按整 block 搬运并有“内存踩踏”风险；A2/A3 不命中此分支，迁移到旧 arch 时不得忽略该边界。

### Trick T-003：offset FP32/INT32 同址原地转换

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：`offsetLength>0`
- Golden 来源：[`quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h:328-339`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h#L328-339)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<float> offsetFp32_ = offsetUb_.Get<float>(alignedLength);
LocalTensor<int32_t> offsetInt32_ = offsetUb_.Get<int32_t>(alignedLength);
```

#### 机制与设计原因

两个 typed view 都从 `offsetUb_` offset 0 开始，FP32 与 INT32 又等宽，因此 `Cast` 把 DMA 进来的 FP32 就地覆盖为 INT32，后续 Maxs/Mins 继续原地执行，不需第二个可寻址的中间区。源码仍保守地分配 2 倍字节；优化事实是实际访问的原地覆盖，不将未访问的后半区计为别名节省。

#### 迁移边界

- 适用：输入/输出等宽且 Vector Cast 支持原地重解释的中间转换。
- 前提：必须保持 `MTE2_V` 等待和后续 Vector barrier；Cast 后不再需要 FP32 原值。
- 失效：当转换前后位宽不同、API 不支持原地或原值仍需保留时，必须拆分缓冲区。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 单核 FP19 编码 | 设计选择依据 |
|---|---|---|
| 命中条件/优先级 | 所有通过校验的 A2/A3 V3 调用 | 无 Host selector 或 key |
| 模板与 dtype | 非模板 `TransQuantParamV2`；FP32→UINT64 bit pattern | out INT64 仅保留相同 64 bit 图样 |
| 循环与分核 | 单核，8192 元素/tile，标量 scale 编码 | 固定 64 KiB result tile |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2 个 TBuf + 2 个广播标量；无 Queue/dynamic UB/window | offset 在 TBuf 中 Vector 处理 |
| 内存总占用与复用 | 主 UB 131072 B；单 offset 路径有额外 64 B InitBuffer 请求 | result 和 offset 每 tile 复用 |
| 搬运/Vector/SIMT/NDDMA/通信 API | offset DataCopyPad + Cast/Maxs/Mins；result DataCopy/DataCopyPad | scale 是标量位运算，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | PIPE_ALL/PIPE_V + MTE2_V/V_MTE3/MTE3_S；mode 0 truncate、mode 1 RNE；tail 精确字节搬出 | 保护原地 offset 与 res UB 复用 |
| fallback/失败边界 | 无 fallback；workspace 空、AIC、非 0 block 直接无输出 | Host 必须保持校验/启动契约 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 V3，`source_closed` | 支持 V3，`source_closed` | 公开文档/OpDef/binary JSON 声称支持，但全局 AscendC 编译清单缺 `ascend950`，本次不发布 950 卡 |
| Host/Launcher | `IsRegbase()==false`，2D scale 首维必须 1；blockDim=1 | 与 A2 相同 | API 允许 scale `(g,n)`，InferShape 还可在 1D offset 更长时取 offset shape；构建闭包不足 |
| Kernel/资源 | arch 2201，tail 命中 `__CCE_AICORE__>200` DataCopyPad | 与 A2 相同 | 源文件有 `__NPU_ARCH__==3510` DataCopyPad 分支，但无活动 global compile unit 证据 |

## 限制与失败边界

- API 不检查 out shape，且只在 scale/offset 都为 1D、长度均大于 1 时检查相等；Kernel 按 `max(scaleLength,offsetLength)` 写出，所以必须由调用者严格满足公开 shape 契约，否则可能读/写越界。
- storage shape size 直接窄化为 `uint32_t`，无溢出 guard；复用时不得扩展到超过 32 bit 长度域。
- Host 没有 UB 能力检查；body 主分配 131072 B，`offsetLength==1` 还会在此前对同一 TBuf 发出 64 B `InitBuffer`，重新初始化的底层回收语义在固定仓不可见。
- entry 将 workspace 和 user workspace 为空视为静默无输出；必须传入第一段返回的 workspace。
- `SetCopyParams(DataCopyParams params)` 按值传参，其对 `blockCount/srcStride/dstStride` 的赋值不会回写调用者；当前路径依赖外部 AscendC `DataCopyParams` 默认构造值，该类定义不在固定仓中，迁移时应显式按值构造 `{1,byteLen,0,0}`。
- 950 的公开支持与活动构建证据冲突：`ascend950` binary JSON 存在，但 `scripts/kernel/binary_config/ascendc_config.json::TransQuantParamV2` 不含 `ascend950`，因此本次不生成 `trans_quant_param_950.md`。
- 固定仓未展开 ACLNN inner executor 壳、`TPipe` allocator 和 AscendC primitive 内部，它们作为明确框架/原语边界；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/trans_quant_param_v2/docs/aclnnTransQuantParamV3.md` | `aclnnTransQuantParamV3*` 文档 | 公开 API、语义、shape/dtype、A2/A3/950 支持表 |
| 入口声明/约束 | `quant/trans_quant_param_v2/op_host/op_api/aclnn_trans_quant_param_v3.h` | `aclnnTransQuantParamV3GetWorkspaceSize`、`aclnnTransQuantParamV3` | C ABI 声明 |
| 入口实现 | `quant/trans_quant_param_v2/op_host/op_api/aclnn_trans_quant_param_v2.cpp` | `CheckParams`、`aclnnTransQuantParamV3*` | 校验顺序、V3→V2 inner 复用、out dtype 统一 |
| Host/Tiling/Launcher | `quant/trans_quant_param_v2/op_host/trans_quant_param_v2_def.cpp` | `ops::TransQuantParamV2` | I/O/attr 与平台 OpDef 注册 |
| Host/Tiling/Launcher | `quant/trans_quant_param_v2/op_host/trans_quant_param_v2_infershape.cpp` | `InferShape4TransQuantParamV2`、`IsDavidSupported` | shape 推导与 950 差异 |
| Host/Tiling/Launcher | `quant/trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.h` | `TransQuantParamV2TilingData` | POD 三字段 |
| Host/Tiling/Launcher | `quant/trans_quant_param_v2/op_host/trans_quant_param_v2_tiling.cpp` | `TilingFunc` | 长度/roundMode/workspace/blockDim producer |
| Kernel entry/body | `quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.cpp` | `trans_quant_param_v2` | workspace guard、entry、唯一 body launch |
| Kernel entry/body | `quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h` | `TransQuantParamV2`、`SafeDataCopy` | 全部 Device 循环、位编码、Vector、UB、同步、tail |
| 构建接线 | `quant/trans_quant_param_v2/op_host/config/ascend910b/trans_quant_param_v2_binary.json` | `TransQuantParamV2` binary | A2 binary I/O/attr |
| 构建接线 | `quant/trans_quant_param_v2/op_host/config/ascend910_93/trans_quant_param_v2_binary.json` | `TransQuantParamV2` binary | A3 binary I/O/attr |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `TransQuantParamV2` | 活动 compute unit 列表；包含 910b/910_93，缺 ascend950 |
| 构建接线 | `quant/trans_quant_param_v2/op_host/CMakeLists.txt` | `add_modules_sources` | `ACLNNTYPE aclnn_inner` 生成框架接入 |
| 构建接线 | `build.sh` | `SOC_TO_ARCH` | 910b/910_93→2201，ascend950→3510 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三个 uint32 长度/模式字段 | 顺序、位宽、元素单位；不得以 int64 shape size 直接替代 | `TransQuantParamV2TilingData` |
| 调用顺序 | 单 offset→单 scale→主 tile→tail | 标量预处理必须在主 `InitBuffer`/循环前完成 | `Process` |
| workspace/内存 | 64 KiB result + 64 KiB offset 主 UB；workspace 仅库保留 | 必须保留 8192 tile、typed view offset=0 和库 workspace 判空 | `Process`；`CalOffset`；entry |
| 同步 | offset MTE2_V/Vector barriers，result V_MTE3/MTE3_S | 不得在 DMA/Vector/CopyOut 完成前覆盖同一 TBuf | `SetOffsetValue`；`Process` |
| 精度与尾块 | mode 0 位截断，mode 1 ties-to-even，offset CAST_RINT+clamp，tail 精确字节搬出 | FP19 特殊指数分支、round/sticky/odd 判定、[-256,255] 和 bit37 偏移不得改 | `CalQuantPreScaleValue`；`SetOffsetValue`；`DataCopyGlobal` |
| 失败边界 | API/Tiling 失败阻断 launch；entry guard 静默无输出 | 必须保持公开 shape 契约、workspace 非空和 blockDim=1 | `CheckParams`；`TilingFunc`；`trans_quant_param_v2` |
