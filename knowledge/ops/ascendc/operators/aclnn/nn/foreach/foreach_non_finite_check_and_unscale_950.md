---
type: Operator
title: "ForeachNonFiniteCheckAndUnscale 950 ACLNN 算子知识"
description: "说明 aclnnForeachNonFiniteCheckAndUnscale 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/docs/aclnnForeachNonFiniteCheckAndUnscale.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_base_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/arch35/foreach_non_finite_check_and_unscale.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/arch35/foreach_non_finite_check_and_unscale_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_host/config/ascend950/foreach_non_finite_check_and_unscale_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
created_at: "2026-08-25T15:47:27Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachNonFiniteCheckAndUnscale 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize` + `aclnnForeachNonFiniteCheckAndUnscale`，公开两段式 ACLNN，无额外版本后缀 |
| 计算语义 | 检查所有 `scaledGrads` 原值中的 Inf/NaN；命中时把 `foundInf[0]` 置 `1.0f`，否则保持；所有元素原地乘 `invScale[0]` |
| 输入输出 | `scaledGrads`：1～256 个连续 ND Tensor，FP32/FP16/BF16、dtype 一致；`foundInf`/`invScale`：单元素 FP32 ND |
| 执行类型 | 单 Kernel `foreach_non_finite_check_and_unscale`；arch35 `AIV_VECTOR`，在 `__VEC_SCOPE__` 内使用 MicroAPI 寄存器 |
| Kernel 路线 | R-001 arch35 RegBase 融合检测与反缩放，key 100；dtype 由 binary specialization `DTYPE_SCALED_GRADS` 给出 |
| 核心资源 | TensorList/原地 GM，三个 depth-2 Queue、两个 32 B TBuf、FP32 RegTensor，32 B user workspace（body 不读写） |
| 来源闭包 | `source_closed`：公开 ACLNN 契约经生成 bridge 边界闭合到 regbase Tiling、POD、arch35 entry/body、公共 load/store helper 与 Ascend 950 构建接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对列表内每个原值 `x`：

```text
coreMax = max(coreMax, x)
coreMin = min(coreMin, x)
scaledGrads[i][j] = cast_T(float(x) * invScale[0])
core 完成全部 tile 后：
  if exponent_bits(abs(coreMax)) == 0xff or exponent_bits(abs(coreMin)) == 0xff:
    foundInf[0] = 1.0f; clean+invalidate that cache line
```

`LoadOneTensorForDtypeT<T>` 对 FP16/BF16 以 `DIST_UNPACK_B16` 加 `ZEROING` cast 升为 FP32；`StoreOneTensorForDtypeT<T>` 对二者用 `NO_SAT + CAST_RINT + ZEROING` 降回原 dtype。FP32 specialization 直接寄存器 load/store。检测和乘法使用同一个寄存器 `x` 的缩放前值。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `scaledGrads` | TensorList；每 Tensor rank 0～8，ND | FP32/FP16/BF16 | 公开 ACLNN：列表长度 1～256、Tensor 非空、dtype 一致、原地参数不支持非连续；输出写回原对象 |
| `foundInf` | 单元素 rank 0～8，ND | FP32 | sticky flag：仅检测到非有限值时写 1；不支持空或非连续 |
| `invScale` | 单元素 rank 0～8，ND | FP32 | 只读标量；文档允许非连续，OpDef 标记 `AutoContiguous()` |
| `workspaceSize`/`executor` | Host 输出 | `uint64_t*`/`aclOpExecutor**` | 第一段返回 32 B user workspace 需求和执行器 |
| `workspace`/`stream` | Device 地址/运行时流 | `void*`/`aclrtStream` | 第二段提交单 Kernel；workspace 地址必须非空 |

### 入口与版本

固定仓的最新公开版本就是无后缀 ACLNN。README 和 API 文档的产品表均明确列出 Ascend 950PR/950DT 支持，OpDef 注册 `ascend950`，binary JSON 包含三种 dtype，Kernel CMake 将 arch35 source 仅接到 `ascend950`，因此 950 发布证据无冲突。固定仓未包含手写 ACLNN `op_api` 实现；`ACLNNTYPE aclnn` 与 OpDef 驱动的生成 inner bridge 被标为框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段执行公开契约的空指针、dtype、列表长度与一致性、标量元素数检查。
2. OpDef 声明动态 `scaled_grads` 和两个必需 FP32 输入，均 ND/`AutoContiguous()`；原地参数是否可接受非连续以公开接口表为准。
3. Tiling registry 先尝试 priority 10000 membase；其 `GetShapeAttrsInfo` 是空成功，`GetPlatformInfo` 后 `IsCapable` 在 regbase NPU 返回 false，即 `GRAPH_PARAM_INVALID`，再尝试 priority 30000 regbase。
4. regbase `GetShapeAttrsInfo` 计算列表长度，要求 `(0,256]`、dtype 一致且属于 FP32/FP16/BF16，并在 Host 层显式要求 `found_inf` 与 `inv_scale` storage 元素数均为 1、dtype 均为 FP32。
5. 公开 ACLNN 拒绝空 Tensor；regbase Tiling 本身未拒绝单 Tensor 空域，并含 `totalDataCount_==0` 的一核特例。正式 Route 只覆盖公开可达的非空输入，不把内部空域能力扩大为 ACLNN 支持。

### 参数、Workspace 与生命周期

设 `b=GetUbBlockSize(context_)`（Device 固定 `BYTE_BLOCK=32`），`e=b/sizeof(T)`，`n_i` 为实际元素数，`a_i=ceil(n_i/e)*e`，`A=sum(a_i)`：

- `needCoreNum=min(A/e, platformCoreNum,80)`；公开非空域下至少为 1。
- 用按 32 B 对齐后的虚拟扁平域按商/余数分核，再以五组 POD 数组表达每核跨 Tensor 区间。
- `SetBlockDim(needCoreNum)`，`workspace[0]=32 B`。entry 取得 `userWS` 后传给 body，但 body 不绑定或访问该参数。
- `scaledGrads` TensorList 指针表、原地 Tensor、两个标量与 workspace 均由 ACLNN 调用方/runtime 持有，至少存活到 stream 上此 Kernel 完成。

### 启动图与失败传播

```text
aclnn...GetWorkspaceSize
  -> generated bridge -> Host node
  -> TilingRegistry priority 10000 (reject regbase)
  -> priority 30000 ForeachNonFiniteCheckAndUnscaleRegbaseTiling
  -> key=100, blockDim, POD, workspaceSize=32 B, executor
aclnn...(workspace, executor, stream)
  -> [stream] arch35::foreach_non_finite_check_and_unscale<<<blockDim>>>
     -> NDRegbase<DTYPE_SCALED_GRADS>::Process
```

只有 Device 节点 N1，无跨 Kernel workspace 边。N1 从各 Tensor GM 读取并原地回写；core 局部累计 min/max，结束时可能写 `foundInf[0]`。Host/Tiling 任何 `GRAPH_FAILED` 都阻断 launch；entry 的 workspace guard 或非 key 100 则直接结束，固定仓没有 Device 错误传播。

## Route 与参数设计

### 分核与资源计算

设平台 UB 为 `U`，RegBase POD 序列化大小 `P`，两个固定 TBuf 共 64 B，双缓冲折算区：

```text
C = floor(((U - P - 64)/2)/b)*b
predictS = floor(256*sizeof(T) / (2*256*sizeof(T)+b) * C)
S = floor(predictS/b)*b
srcCount = S/sizeof(T)
R = ceil(ceil(2*srcCount/64)/(b/4))*(b/4)*4 bytes
guard: 2*S + R <= C
```

Device 分配总量是 `copyIn 2S + copyOut 2S + tempVal 2R + max 32 + min 32`；Host guard 乘 2 后正好覆盖 depth-2 Queue。`maxDataCount=S/sizeof(T)`，每 tile 实际元素数尾部取余，再对齐到 `32/sizeof(T)`；MicroAPI 每次处理 `VL=platform::GetVRegSize()/4` 个 FP32 lane，`repeatTimes=ceil(alignedCount/VL)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 arch35 RegBase | priority 30000；key 100；`NDRegbase<DTYPE_SCALED_GRADS>` | priority 10000 因 regbase 返回 `GRAPH_PARAM_INVALID`；当前 NPU arch 是 DAV_3510 或 DAV_5102；公开输入校验、Tiling 校验、UB guard 均通过 | 参数/资源失败终止；unknown key 无 body |

```text
priority 10000 membase:
  IsRegbaseSocVersion == true -> GRAPH_PARAM_INVALID
priority 30000 regbase:
  IsRegbaseSocVersion == false -> GRAPH_PARAM_INVALID
  else validate -> key 100 -> R-001
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 arch35 RegBase | key 100 | 950/regbase；1～256 个公开非空连续 ND Tensor；统一 FP32/FP16/BF16；标量 FP32 单元素；UB guard | 单 AIV_VECTOR N1 | `ForeachNonFiniteCheckAndUnscaleNDRegbase<DTYPE_SCALED_GRADS>`；MicroAPI helper 按 T 特化 | 3 个 depth-2 Queue + 2 个 32 B TBuf；Queue、V→S、cache line clean/invalidate | GM tail 用 Ext/Pad API；寄存器尾 mask；无其他 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| regbase arch | NPU arch ∈ `{DAV_3510,DAV_5102}` | 选择 arch35 MicroAPI body | R-001 | 对非 regbase 走 membase；950 不适用 | `IsRegbaseNpuArch/IsCapable` |
| core 上限 | `needCoreNum=min(GetNeedCoreNum(coreNum),80)` | POD per-core 数组长度 80 | R-001 | 继续以 80 核内分区 | `RunBigKernelTiling` |
| UB 上界 | `2*S+R<=C`，且 `U!=0` | 覆盖双缓冲 Queue 和固定 64 B TBuf | R-001 | `GRAPH_FAILED` | `DivideUbMemory` |
| Tensor/标量 | 列表 `(0,256]`，dtype 一致；标量 size=1/FP32 | POD 与 scalar GM load 约束 | R-001 | `GRAPH_FAILED` | `GetShapeAttrsInfo/CheckParams` |
| DMA tail | `valid % (32/sizeof(T))` | 未对齐 GM 传输需 Ext/Pad | R-001 | Route 内尾分支 | `CopyIn/CopyOut` |
| MicroAPI tail | `maskReg=UpdateMask<float>(sreg)` | 最后一 VL 只处理有效/32 B 对齐元素 | R-001 | Route 内 predication | `Compute` |
| workspace | 两层地址均非空 | entry 统一要求 user workspace | R-001 | 直接返回 | arch35 entry lines 24-30 |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 宏定义来自固定仓外框架，所以精确 `sizeof/alignof/offsetof` 不可见；字段声明顺序、类型、数组长度和 Host/Device 同名结构可闭合。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachNonFiniteCheckAndUnscaleRegbaseTilingData` | `scaledGradsUbSize` | `uint32_t`/32 | 第 1 字段；精确 offset/sizeof 不可见 | 字节/slot | `set_scaledGradsUbSize(S)` | `ParseTilingData`; Queue 容量/`maxDataCount_` | b 字节向下对齐 |
| 同上 | `reduceTempValUbSize` | `uint32_t`/32 | 第 2 字段 | 字节/slot | `set_reduceTempValUbSize(R)` | `InitBuffer(tempValQueue_,2,R)` | body 未 Alloc 但物理 Queue 仍分配 |
| 同上 | `tensorDataCountList` | `int64_t[256]` | 第 3 字段 | 元素 | storage shape size | `Process` | 未激活尾项不读 |
| 同上 | `tensorStartList` | `uint16_t[80]` | 第 4 字段 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData[blockIdx_]` | 活跃 core 范围 |
| 同上 | `tensorEndList` | `uint16_t[80]` | 第 5 字段 | Tensor 索引 | 同上 | 同上 | 含端点 |
| 同上 | `tensorStartOffsetList` | `int64_t[80]` | 第 6 字段 | 元素 | 同上 | 首 Tensor 起点 | 对齐虚拟域 |
| 同上 | `tensorEndOffsetList` | `int64_t[80]` | 第 7 字段 | 元素 | 同上 | 末 Tensor上限 | 含端点，再受实际长度裁剪 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 arch35 RegBase | priority 30000/key 100 | `..._regbase_tiling.cpp::GetShapeAttrsInfo/RunBigKernelTiling` | `...RegbaseTilingData` | `op_kernel/arch35/...cpp::foreach_non_finite_check_and_unscale` | `..._regbase.h::NDRegbase<T>` + common `load_store_utils.h/platform.h` | ascend950 binary JSON；Kernel CMake arch35 target |

## Kernel 实现

### 公共执行流程

本卡只有 R-001。entry 先检查 workspace，key 100 时用显式 RegBase POD 解析宏，再以 binary 编译期 `DTYPE_SCALED_GRADS` 实例化 body。每个 core 从 TensorList GM 指针表解析实际 Tensor 地址，遍历自己覆盖的首尾 Tensor 和 tile。单 tile 执行 GM→UB、MicroAPI 寄存器 load/FP32 min/max/乘法/store、UB→GM；两个 32 B TBuf 的第 0 个 FP32 值跨 tile 保存该 core 的累计 min/max，最后只做一次 scalar 检测和可能的全局 flag 写。

### Route R-001：arch35 RegBase

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Registry | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 升序；`GRAPH_PARAM_INVALID` 才继续 | 10000→30000 |
| Host selector | `op_host/..._regbase_tiling.cpp` | `IsCapable` | `IsRegbaseSocVersion(context_)` | 接受 R-001 |
| Host key | 同上 | `GetTilingKeyVal` | 无 dtype 分支 | 100 |
| Router | `op_kernel/arch35/...cpp` | entry | `TILING_KEY_IS(100)` | `NDRegbase<DTYPE_SCALED_GRADS>` |

#### Device 数据流

##### 函数级调用链

```text
arch35::foreach_non_finite_check_and_unscale(..., key=100)
  -> NDRegbase<T>::Init -> ParseTilingData
  -> NDRegbase<T>::Process
     -> Duplicate(maxLocal/minLocal, 0)
     -> GetTensorAddr -> SingleTensorProcess
        -> CopyIn -> Compute(__VEC_SCOPE__) -> CopyOut
     -> V_S flag/wait -> IsNonFinite -> foundInf.SetValue -> DataCacheCleanAndInvalid
```

##### 源码等价伪码

```text
maxLocal[0:8]=0; minLocal[0:8]=0
for tensor i in [tensorStart_, tensorEnd_]:
  start/end = POD 区间与实际 n_i-1 的交集
  GM = tensor_ptr[i] + start
  for tile k:
    valid = min(maxDataCount_, remaining)
    aligned = ceil(valid/(32/sizeof(T)))*(32/sizeof(T))
    CopyIn(valid): 若尾块，DataCopyPad fill=0
    repeats = ceil(aligned/(VRegBytes/4)); sreg=aligned
    regMin=0; regMax=0
    for 非末 VL:
      mask=UpdateMask<float>(sreg)
      x=LoadOneTensorForDtypeT<T>(UB, offset, mask)  # T->FP32
      regMax=max(regMax,x); regMin=min(regMin,x)
      y=x*invScale; StoreOneTensorForDtypeT<T>(UBout,y,mask,offset)
    最末 VL:
      mask=UpdateMask<float>(sreg); load x
      x=Adds(x,0,mask)  # inactive lane 归零
      max/min 用 full mask 合入；乘法/写回仍用 tail mask
    reduce regMax/regMin；与 maxLocal[0]/minLocal[0] 合并并写回 TBuf 首元素
    CopyOut(valid)
V_S wait
if IsNonFinite(maxLocal[0]) || IsNonFinite(minLocal[0]):
  foundInf[0]=1; clean+invalidate single cache line
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList GM | `dataAddr + ((*dataAddr)>>3) + i` | ≤256 指针 | `uint64_t` table | ACLNN runtime | `GetTensorAddr` | 调用方/launch | 8 B 指针 |
| `scaledGradsGM_` | `tensorPtr[i]+cursorStart+k*maxDataCount_` | `valid` T 元素 | T/ND | 调用方/前次写回 | CopyIn/CopyOut | 原地 GM | 尾部 Ext/Pad |
| `copyInQueue_` | Queue | `2*S` | T | GM DMA | MicroAPI Compute | Alloc→EnQue→DeQue→Free | depth 2 |
| `copyOutQueue_` | Queue | `2*S` | T | MicroAPI store | GM DMA | 同上 | depth 2 |
| `tempValQueue_` | Queue | `2*R` | 未消费的 FP32 预留 | `InitBuffer` 仅分配 | 无 | `TPipe` 全 launch | depth 2；固定 body 未 Alloc/DeQue |
| `maxBuf_` | TBuf | 32 B | FP32 | Duplicate/每 tile 合并 | 最终 scalar 检测 | 全 launch | 只使用首 FP32 作跨 tile状态 |
| `minBuf_` | TBuf | 32 B | FP32 | 同上 | 同上 | 全 launch | 同上 |
| RegTensor | 寄存器作用域 | `x,y,min,max,lastMin,lastMax`；物理字节固定仓不可见 | FP32 | MicroAPI load/compute | store/reduce | 单次 `__VEC_SCOPE__` | 无 UB 别名 |
| workspace | `[0,32)` | 32 B | 未解释 | Host/runtime | body 不消费 | executor launch | entry 只校验地址 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["copyInQueue · depth=2 · 2*S"]:1
  outQ["copyOutQueue · depth=2 · 2*S"]:1
  tempQ["tempValQueue · depth=2 · 2*R（未消费）"]:2
  maxB["maxBuf · 32 B"]:1
  minB["minBuf · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*S + 2*S + 2*R + 32 + 32 = 4*S + 2*R + 64
C = floor(((U-P-64)/2)/b)*b
Host guard 2*S+R <= C
=> UB_total <= U-P
dynamic_ub = N/A
workspace_total = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `copyInQueue_` slot | `CopyIn` | `Compute` | 后续 tile | Queue EnQue/DeQue/Free | Compute 已读完 |
| `copyOutQueue_` slot | `Compute` | `CopyOut` | 后续 tile | Queue EnQue/DeQue/Free | DMA 已提交实际长度 |
| `maxBuf_[0]` | 初始 Duplicate；tile Compute | 下一 tile Compute；最终 Process | 每 tile末覆盖为累计 max | MicroAPI DataCopy + 最终 V→S | 必须与旧值 max 合并后覆盖 |
| `minBuf_[0]` | 同上 | 同上 | 同上 | 同上 | 必须与旧值 min 合并 |
| `tempValQueue_` | 无 | 无 | 无 | N/A | 不得从 UB 总量扣除，InitBuffer 已物理分配 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopy` 或 `DataCopyPad` + Queue | MTE2 | `Compute` DeQue | input slot | Ext `blockLen=valid*sizeof(T)` B；pad value 0 |
| 寄存器 load | `LoadOneTensorForDtypeT` | UB | FP32 `x` | 当前 VL | FP16/BF16 unpack + zeroing cast；FP32 direct |
| 尾 VL | `UpdateMask`；末 VL `Adds(x,x,0,mask)` | mask/sreg | full-mask min/max | 无效 lane | inactive lane 归零后才能用 `pregMain` 做归约 |
| 累计 min/max | Reg Reduce + first-element DataCopy | Vector | 下一 tile或 scalar | 两个 TBuf 首元素 | 始终 FP32；每 core 累计 |
| 反缩放 | `Muls(y,x,invScale,mask)` | FP32 x | dtype store | 输出 VL | 低精度以 `CAST_RINT` 返回；只存 active lane |
| Vector→Scalar | `SetFlag/WaitFlag<V_S>` | 所有 tile Vector | `GetValue(0)` | 累计 min/max | core 尾只检查一次 |
| found flag 可见性 | `SetValue` + `DataCacheCleanAndInvalid<SINGLE_CACHE_LINE>` | Scalar | 外部/其他可见方 | `foundInf` cache line | 唯一写值 1.0；不清零 |
| UB→GM | Ext `DataCopyPad`/`DataCopy` + Queue | MTE3 | FreeTensor | output slot | 只写 `valid` 元素，padding 不越界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/..._regbase_tiling.cpp` | `GetShapeAttrsInfo/AssignDataToEachCore/DivideUbMemory` | 完整校验、80-core、S/R、workspace |
| Kernel entry/router | `op_kernel/arch35/...cpp` | key 100 entry | 显式 POD 到 dtype specialization |
| Device body/helper | `op_kernel/arch35/..._regbase.h` | `NDRegbase<T>::Process/Compute` | 循环、MicroAPI、跨 tile归约、tail、cache flush |
| Device dtype helper | `common/inc/op_kernel/load_store_utils.h` | `LoadOneTensorForDtypeT/StoreOneTensorForDtypeT` | unpack/pack 与 cast trait |

## 算子特定优化

### Trick T-001：单次寄存器 load 同时完成检测统计与反缩放

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：每个 `__VEC_SCOPE__` 的 active VL；dtype specialization 为 FP32/FP16/BF16
- Golden 来源：[`foreach_non_finite_check_and_unscale_regbase.h:221-237`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/arch35/foreach_non_finite_check_and_unscale_regbase.h#L221-L237)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint16_t i = 0; i < repeatTimes - 1; i++) {
    maskReg = UpdateMask<float>(sreg);
    ops::LoadOneTensorForDtypeT<T>(inUbAddr, x, maskReg, i * dataCountPerLoop);
    Max(max, max, x, maskReg);
    Min(min, min, x, maskReg);
    Muls(y, x, invScaleVal, maskReg);
    ops::StoreOneTensorForDtypeT<T>(outUbAddr, y, maskReg, i * dataCountPerLoop);
}
```

#### 机制与设计原因

同一个 FP32 RegTensor `x` 一次 load 后同时进入 max/min 统计与 `Muls`，`y` 直接写 output UB。这样检测与反缩放共享 GM→UB 和 UB→寄存器数据，不物化另一份“检测输入”。FP16/BF16 的升精度发生在 load helper 内，降精度发生在 store helper 内，避免 membase Route 的 FP32 calc Queue。

#### 迁移边界

- 适用：一个输入同时供轻量归约统计与逐元素输出。
- 前提：两条计算都能在同一 FP32 寄存器表示上完成，tail 使用同一 mask。
- 失效：需要跨算子保存完整 FP32 中间结果时，不能只靠寄存器融合。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：两枚 32 B TBuf 保留跨 tile 归约状态，core 尾只做一次标量写

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：一个 core 覆盖一个或多个 Tensor/tile
- Golden 来源：[`foreach_non_finite_check_and_unscale_regbase.h:238-245`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/arch35/foreach_non_finite_check_and_unscale_regbase.h#L238-L245)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
DataCopy(lastMin, (__local_mem__ float*)(minUbAddr));
DataCopy(lastMax, (__local_mem__ float*)(maxUbAddr));
ReduceMax(max, max, pregMain);
ReduceMin(min, min, pregMain);
Max(lastMax, lastMax, max, pregMain);
Min(lastMin, lastMin, min, pregMain);
DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)minUbAddr), lastMin, pregMerge);
DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)maxUbAddr), lastMax, pregMerge);
```

#### 机制与设计原因

每个 tile 只把归约后的首元素合并进 `maxBuf_`/`minBuf_`，不为每 tile 写 GM，也不分配 workspace 归约数组；直到 core 完成全部数据才 V→S 并位检测，至多写一次 `foundInf`。32 B TBuf 是实际 UB 成本，Host 在 `U-P-64` 后计算 tile。

#### 迁移边界

- 适用：归约结果可结合且最终只需 sticky flag。
- 前提：初值 0 与 min/max 组合仍能保留所有非有限值检测信息；尾部无效 lane 必须可靠归零。
- 失效：要求精确有限值 min/max 输出时，零初值会改变全正/全负数据的数值结果；本算子只做非有限位检测。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-003：显式 cache line clean/invalidate 发布 sticky flag

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：core 累计 max 或 min 的 FP32 指数域为 `0xff`
- Golden 来源：[`foreach_non_finite_check_and_unscale_regbase.h:138-144`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_non_finite_check_and_unscale/op_kernel/arch35/foreach_non_finite_check_and_unscale_regbase.h#L138-L144)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventIDVToS);
WaitFlag<HardEvent::V_S>(eventIDVToS);
if (IsNonFinite(maxLocal_.GetValue(0)) || IsNonFinite(minLocal_.GetValue(0))) {
    foundInfGM_.SetValue(0, 1.0);
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(foundInfGM_);
}
```

#### 机制与设计原因

V→S 保证标量读取发生在 Vector 累计之后；写 sticky flag 后，显式 clean/invalidate 单 cache line，使 arch35 scalar cache 中的写值被发布。CMake 对 ascend950 编译还指定 `--cce-no-dcache-flush`，因此此处是算子拥有的定点可见性动作。

#### 迁移边界

- 适用：arch35 scalar GM 写经过 data cache，构建关闭自动 flush。
- 前提：目标对象位于可由单 cache line 操作覆盖的 FP32 标量；多写者只写相同值。
- 失效：若写多个 cache line 或协议要求 atomic/read-modify-write，需重新设计同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 arch35 RegBase | R-002 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | regbase，priority 30000/key 100 | N/A | 本平台只有一个 Host 可达 Route |
| 模板与 dtype | `NDRegbase<DTYPE_SCALED_GRADS>` | N/A | binary 按 FP32/FP16/BF16 specialize |
| 循环与分核 | 最多 80 core；Tensor→tile→VL | N/A | POD 数组与寄存器宽度 |
| Queue/TBuf/寄存器/dynamic UB | 3 个 depth-2 Queue、2 个 32 B TBuf、FP32 RegTensor | N/A | 寄存器融合与跨 tile state |
| 内存总占用与复用 | `4S+2R+64` | N/A | Host guard 与 InitBuffer 一致 |
| 搬运/Vector API | Ext/Pad DMA；MicroAPI load/store；Max/Min/Muls/Reduce | N/A | dtype helper 封装 pack/unpack |
| 同步、精度与 tail | Queue、V→S、cache flush；FP32 compute；predicate tail | N/A | sticky flag 与尾 lane 安全 |
| fallback/失败边界 | Tiling 失败或 entry 直接返回 | N/A | 无第二 Route |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | membase priority 10000 | membase priority 10000 | membase 拒绝后 regbase priority 30000，最多 80 core |
| Kernel/资源 | 传统 Vector，低精度使用 FP32 calc Queue | 同 A2 | arch35 MicroAPI；寄存器内升降精度；2×32 B 累计 TBuf；显式 cache flush |

## 限制与失败边界

- 正式 ACLNN Route 要求 1～256 个非空、连续的 `scaledGrads` Tensor，dtype 统一且为 FP32/FP16/BF16；两个标量必须是单元素 FP32。
- regbase Tiling 源码允许 `totalDataCount_==0` 并配置一核，但这与公开 ACLNN“不支持空 Tensor”冲突；公开入口会先拒绝，故不发布为空域 Route。
- `tempValQueue_` 被 `InitBuffer` 分配但固定 body 从不 Alloc/DeQue；字节账必须保留这部分，不可按逻辑未使用扣除。
- 多 core 对同一 `foundInf[0]` 非 atomic 写 `1.0f`；源码通过幂等同值写及逐 core cache clean/invalidate工作，固定仓没有更强 atomic 顺序证明。
- `totalDataCountAligned_` 是 `int64_t`，但 `GetNeedCoreNum` 构造 `uint32_t tempCoreNum`，无显式超大列表元素总数溢出 guard。
- `U-P-64` 在转 `uint32_t` 前使用无符号表达式，固定仓没有显式验证 `U>P+64`；正常平台资源由配置保证，异常资源不应推测为支持。
- workspace 固定 32 B 但 body 不使用；仍不能传空，因为 entry 会直接返回。
- 固定仓不可见：生成 ACLNN bridge、executor stream enqueue 细节、TilingData 宏精确 ABI、MicroAPI primitive 内部实现；均不外推。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/.../docs/aclnnForeachNonFiniteCheckAndUnscale.md` | 两段式 ACLNN | 参数、错误、950 公开支持 |
| 平台/语义 | `foreach/.../README.md` | 产品表/公式 | 950 支持与核心语义 |
| Host OpDef | `foreach/.../op_host/foreach_non_finite_check_and_unscale_def.cpp` | `ops::ForeachNonFiniteCheckAndUnscale` | ascend950 注册、dtype/format |
| Host 入口 | `foreach/.../op_host/foreach_non_finite_check_and_unscale_base_tiling.cpp` | `Tiling4ForeachNonFiniteCheckAndUnscale` | registry 入口 |
| POD | `foreach/.../op_host/foreach_non_finite_check_and_unscale_tiling.h` | RegBase TilingData | 80-core ABI 字段 |
| Host/Tiling | `foreach/.../op_host/foreach_non_finite_check_and_unscale_regbase_tiling.cpp` | RegBase tiling class | selector、校验、分核、UB、key |
| Kernel entry | `foreach/.../op_kernel/arch35/foreach_non_finite_check_and_unscale.cpp` | entry/key 100 | POD/router/workspace guard |
| Kernel body | `foreach/.../op_kernel/arch35/foreach_non_finite_check_and_unscale_regbase.h` | `NDRegbase<T>` | 完整 arch35 body |
| dtype helper | `common/inc/op_kernel/load_store_utils.h` | Load/Store helper | FP16/BF16/FP32 寄存器转换 |
| 平台 helper | `common/inc/op_kernel/platform.h` | `GetVRegSize` | VL 字节数 producer |
| binary 接线 | `foreach/.../op_host/config/ascend950/foreach_non_finite_check_and_unscale_binary.json` | dtype 条目 | 三种 950 binary |
| 构建接线 | `foreach/.../op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 source 与 no-dcache-flush |
| selector 框架 | `common/inc/op_host/tiling_templates_registry.h` | `DoTilingImpl` | priority 顺序和 fallthrough |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | S/R、256 个元素数、80-core 四数组 | 字段顺序/位宽/单位；Host/Device 必须使用 RegBase struct | `...RegbaseTilingData` |
| 调用顺序 | CopyIn→寄存器融合 Compute→CopyOut→core尾检测 | 检测读取缩放前 x；flag 写不能早于 V→S | `SingleTensorProcess/Process` |
| workspace/内存 | 32 B workspace；`4S+2R+64` UB | 未使用 temp Queue 仍占物理 UB；64 B 固定 TBuf先扣除 | `DivideUbMemory/Init` |
| 同步 | Queue、V→S、单 cache line clean/invalidate | 不得省略显式发布或提前覆盖 TBuf | `Process/Compute` |
| 精度与尾块 | T→FP32→T；`CAST_RINT`；mask+尾 DMA | 最末 VL inactive lane 清零；GM 只写 valid | load/store helper；`Compute/CopyOut` |
| 失败边界 | priority fallthrough、Tiling hard fail、entry guard | 不把内部空域分支宣传成 ACLNN 支持 | selector/Tiling/entry |