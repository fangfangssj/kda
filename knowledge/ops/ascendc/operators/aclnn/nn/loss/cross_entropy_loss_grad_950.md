---
type: Operator
title: "CrossEntropyLossGrad 950 ACLNN 算子知识"
description: "说明 aclnnCrossEntropyLossGrad 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/docs/aclnnCrossEntropyLossGrad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_host/cross_entropy_loss_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_host/cross_entropy_loss_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_host/cross_entropy_loss_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_full_load.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_weight_none.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_weight_not_none.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_host/config/ascend950/cross_entropy_loss_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:14:45Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# CrossEntropyLossGrad 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCrossEntropyLossGradGetWorkspaceSize` + `aclnnCrossEntropyLossGrad`，公开两段式 ACLNN，无版本后缀 |
| 计算语义 | CrossEntropyLoss 对 `logProb` 的梯度；正式公开域为 `target=int64`、`labelSmoothing=0`、zloss 不生效 |
| 输入输出 | `logProb/out:[N,C]`，`target:[N]`，`gradLoss:[N]`（none）或标量（mean/sum），可选 `weight:[C] fp32`；ND |
| 执行类型 | 单节点 `AIV_VECTOR`；entry 标记 `KERNEL_TYPE_MIX_AIV_1_0`，只有 AIV，没有 Cube/AIC 或 `MIX_CV` |
| Kernel 路线 | R-001 空张量；R-002 分块无权重；R-003 分块有权重；R-004 full-load 无权重；R-005 full-load 有权重；每个非空路线含 none/mean/sum 子路线 |
| 核心资源 | GM；depth=1 TQue/TBuf；MicroAPI RegTensor；mean 用每核 fp32 workspace；full-load 把多行完整 C 片同时驻留 UB |
| 来源闭包 | source_closed：公开 ACLNN/OpDef → regbase selector/POD/key → entry → split/full-load bodies与直接 platform helper → ascend950 binary/CMake 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开 `labelSmoothing=0` 时，令 `m_i=1[target_i!=ignoreIndex]`，`w_i=1` 或 `weight[target_i]`：

```text
none: q_i = gradLoss[i] * m_i * w_i
sum:  q_i = gradLoss * m_i * w_i
mean: q_i = gradLoss * m_i * w_i / sum_k(m_k*w_k)
out[i,j] = exp(logProb[i,j])*q_i - 1[j=target_i]*q_i
```

950 split body 用 MicroAPI 对 C tile 做 exp/mul，并用 Scalar 修正 target lane；full-load body先对 `Ntile×C` 批量 exp/mul，再以 `DataCopyGather/DataCopyScatter` 修改每行 target。公开文档仍展示 label smoothing/zloss 完整数学定义，但明确平滑只支持 0、zloss 不支持，因此 `labelS=1` 及 zloss 相关源码不是正式公开 Route。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradLoss` | none `[N]`；mean/sum 0D（Host 也接受 `[1]`） | fp16/bf16/fp32 | 与 `logProb/out` 同 dtype |
| `logProb` | `[N,C]`，ND | fp16/bf16/fp32 | 必须 2D，`C>0`；AutoContiguous |
| `target` | `[N]`，ND | int64 | 公开 ACLNN 仅 int64；target 为类索引或 ignoreIndex |
| `weightOptional` | `[C]`，ND | fp32 | 缺省为全 1 |
| `gradZlossOptional` | none `[N]`，其余 0D | 同主 dtype | 当前不支持/Device 不读取 |
| `lseForZlossOptional` | `[N]` | 同主 dtype | 当前不支持/Device 不读取 |
| `reductionOptional` | attr | string | none/mean/sum，默认 mean；非法值 Host 失败 |
| `ignoreIndex` | attr | int64 | 默认 -100；950 编译 key 仅在值 `>=0` 时置 `isIgnore=1` |
| `labelSmoothing` | attr | double→float | Host 接受 `[0,1]`，但公开 ACLNN 只支持 0 |
| `lseSquareScaleForZloss` | attr | double→float | 保留且不生效 |
| `out` | `[N,C]`，ND | 同 logProb | shape/dtype 由 logProb 推导 |

### 入口与版本

公开 API 无版本后缀。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef 为 ascend950 配置 bf16/fp16/fp32，并额外登记 int32 target 组合，但 ACLNN 文档只公开 int64，所以本卡正式路线只覆盖 int64 binary。`Tiling4CrossEntropyLossGrad` 通过 `IsRegbaseSocVersion` 选择 `CrossEntropyLossGradRegbaseTiling::RunTiling`。手写 ACLNN bridge 不在固定仓，`ACLNNTYPE aclnn` 生成部分作为框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

OpDef 对所有输入启用 `AutoContiguous`。regbase Host 顺序为：读取 UB/AIV/VReg/blockSize → 校验 reduction、ignoreIndex、labelSmoothing → 校验 shape → 校验主 dtype 一致、weight fp32、target int32/int64 → 检查空 tensor → 分核 → 先尝试 full-load 容量模型，失败才进行 C 分块。公开调用额外受 API 文档的 int64 target 与 `labelSmoothing=0` 限制。

### 参数、Workspace 与生命周期

`totalCoreNum=P`，`vectorLength=GetVRegSize()`，`blockSize=GetUbBlockSize()`。非空时用户区 `usrSize=P*4*blockSize`，再加固定 `16MiB` 系统 workspace；空 tensor 只返回 `16MiB`。mean 且需要按 mask/weight 求分母时，每核写 `workspace[coreIndex]` 一个 fp32，`SyncAll` 后读取 `[0,usedCoreNum)`。Kernel 源码绑定的是传入 `workspace` 首地址；entry 虽计算 `GetUserWorkspace(workspace)`，该局部变量未被 body 使用。

### 启动图与失败传播

```text
ACLNN GetWorkspaceSize
 -> AutoContiguous/generated bridge（框架边界）
 -> Tiling4CrossEntropyLossGrad
    -> IsRegbaseSocVersion=true
    -> CrossEntropyLossGradRegbaseTiling::RunTiling
       -> GetPlatform -> GetTilingAttr -> GetTilingInput -> CheckDtype
       -> empty ? schId=2 : InitSplitInfo -> (FullLoad ? schId=1 : CoreSplit schId=0)
 -> executor/workspaceSize
ACLNN execute(stream)
 -> cross_entropy_loss_grad<schId,reduction,isWeight,labelS,isIgnore>
 -> x_grad GM（单 Kernel）
```

空路线 `blockDim=1`，entry 的 constexpr 分支不匹配 schId 0/1 而直接 return；因为输出元素数为 0，无 GM 写入。所有 Host `OP_CHECK` 失败均阻断 launch。不存在 Kernel 间 workspace 边；mean workspace 是单节点内部全核交接。

## Route 与参数设计

### 分核与资源计算

分核与 A2/A3 同构：`frontCoreNum=N%P?N%P:P`，`tailCoreNum=N<=P?0:P-frontCoreNum`，`frontRowNum=ceil(N/P)`，`tailRowNum=floor(N/P)`，`blockDim=front+tail`。

full-load 首先令 `A=align_up(C,blockSize/d)`，估算常驻 C 缓冲：`2*A*d + A*4` 每行（input/output/tmp）；再计 target、ignore、targetCast、none grad、weight gather/queue、mean 两个标量 TBuf与全核 TBuf。求出至少 1 行可驻留时：`schId=1`，`alignNCNum=Ntile*A`，每次同时计算 `Ntile×C`。

否则 `schId=0`：只为不超过 `min(frontRowNum,VRegBytes)` 的 N 元数据留 UB，余量除以 `bufferNum`（无 weight 为 12；有 weight 为 16），C tile 按 128 个 fp32 元素向下对齐，得到 `alignColLoopNum`、主循环/尾块。公开 `labelSmoothing=0` 时不会使用二分缓存字段。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 empty | `(schId=2,0,0,0,0)` | `logProb.storageShape.GetShapeSize()==0` | 非空进入后续 |
| 2 | R-004/R-005 full-load | `(1,reduction,isWeight,0,isIgnore)` | `CELossGradInnerCoreFullLoad()` 返回 true；优先于 split | 容量不足进入 R-002/R-003 |
| 3 | R-002 split no-weight | `(0,reduction,0,0,isIgnore)` | full-load false 且 weight null | R-003 |
| 4 | R-003 split weighted | `(0,reduction,1,0,isIgnore)` | full-load false 且 weight 非 null | 资源/参数失败 |

```text
if empty: key=(2,0,0,0,0)
else:
  reduction = none?0 : mean?1 : sum?2
  isWeight = weight!=null
  labelS = (labelSmoothing>0)  # 正式公开域固定 0
  isIgnore = (ignoreIndex>=0)
  schId = FullLoadFitsUB()?1:0
  key=GET_TPL_TILING_KEY(schId,reduction,isWeight,labelS,isIgnore)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 empty | 2,0,0,0,0 | 950，元素数 0 | 单 entry 空返回 | entry `else{return;}` | 无 AIV UB/用户 workspace | 无输出元素 |
| R-002 split no-weight | 0,r,0,0,i | 950、int64、非空、full-load 不可容纳、无 weight | 单 AIV | `WeightNone<T,int64_t>::Process` | C tile；mean+ignore 才跨核归约 | MicroAPI mask；C tail |
| R-003 split weighted | 0,r,1,0,i | 同上，有 weight | 单 AIV | `WeightNotNone<T,int64_t>::Process` | targetWeight；mean 总是跨核归约 | C tail |
| R-004 full-load no-weight | 1,r,0,0,i | 950、full-load fits、无 weight | 单 AIV | `CELossGradFullLoad<T,int64_t,r,0,0,i>` | Ntile×C UB；mean+ignore 归约 | N 尾组、C VReg tail |
| R-005 full-load weighted | 1,r,1,0,i | 950、full-load fits、有 weight | 单 AIV | `CELossGradFullLoad<T,int64_t,r,1,0,i>` | weight 全 C 驻 UB；mean 归约 | N 尾组、C VReg tail |

其中 `r∈{0:none,1:mean,2:sum}`，`i∈{0,1}`；reduction 令资源/计算不同，正文逐路线明确条件缓冲和分支，不把它视为新执行族。

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| full-load 容量 | `finalTotal<=ubNSize && finalN>=1` | 同时驻留 `finalN×A` input/output/fp32 tmp 与 N 元数据 | R-004/R-005 | R-002/R-003 | `CELossGradInnerCoreFullLoad` |
| split 128-element C align | `A=floor(min(candidate,C),128)` | C tile 以 512B(fp32) 粒度；MicroAPI/VReg 循环稳定 | R-002/R-003 | A=0 无后续 fallback | `CELossGradCoreSplitInfo` |
| mean global denominator | `reduction==1 && (weight \|\| isIgnore)` | 仅这些条件下分母不是简单 N | 非空 mean | 无 weight 且 negative ignore 用 `rowVal` | `Process/TargetWeightReduceSum*` |
| public target dtype | ACLNN docs int64 | 950 binary/OpDef 还有 int32，但不属于入口公开域 | 全部正式路线 | 不发布 int32 route | API 文档 + binary JSON |
| negative ignore | `isIgnore=(ignoreIndex>=0)` | constexpr mask/gather 只为非负 ignore 编译 | 950 全部 | 负 ignore 当作无 ignore；target 实值必须仍在 `[0,C)` | `GetTilingAttr` / full-load assert |
| VReg tail | `tailNum=C%vfLen` 或 `N%vfLen` | `UpdateMask` 保护寄存器 lane | full-load/split | tail=0 跳过 | MicroAPI loops |

### POD/ABI

`CrossEntropyLossGradRegbaseTilingData` 是普通标准布局字段序列：按源码类型自然 8 字节对齐推得 `sizeof=224`、`alignof=8`；`labelSmoothing` 后有 4 字节 padding。Host 以 `GetTilingData<...>()` 取得同一 struct，Device 用 `REGISTER_TILING_DEFAULT/GET_TILING_DATA_WITH_STRUCT` 解析。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CrossEntropyLossGradRegbaseTilingData` | `rowVal` | int64/64 | off 0 | row | logProb dim0 | 分母/分核 | N |
| 同上 | `colVal` | int64/64 | off 8 | element/row | logProb dim1 | C loops/address | C>0 nonempty |
| 同上 | `reduction` | int64/64 | off 16 | enum | GetTilingAttr | base/runtime | 0/1/2 |
| 同上 | `ignoreIndex` | int64/64 | off 24 | label | attr | compare/assert | 默认 -100 |
| 同上 | `labelSmoothing` | float/32 | off 32，后 pad 4 | ratio | attr | label factors | 公开 0 |
| 同上 | `usedCoreNum` | int64/64 | off 40 | core | InitSplitInfo | workspace reduce | blockDim |
| 同上 | `frontCoreNum` | int64/64 | off 48 | core | InitSplitInfo | Init offsets | `[1,P]` |
| 同上 | `tailCoreNum` | int64/64 | off 56 | core | InitSplitInfo | 间接 | N≤P 为0 |
| 同上 | `frontRowNum` | int64/64 | off 64 | row/core | ceil(N/P) | Init | former |
| 同上 | `tailRowNum` | int64/64 | off 72 | row/core | floor(N/P) | Init | tail |
| 同上 | `alignColLoopNum` | int64/64 | off 80 | element | full/split sizing | C/N×C layout | A |
| 同上 | `colLoop` | int64/64 | off 88 | loop | split C/A | split body | full-load 值 0 |
| 同上 | `colLoopNumTail` | int64/64 | off 96 | element | C%A | split tail/base capacity | 0 可跳过 |
| 同上 | `alignNCNum` | int64/64 | off 104 | element | full `Ntile*A`；split A | UB queues/tmp | 非零 |
| 同上 | `targetSize` | int64/64 | off 112 | byte | align N*targetDtype | target queue | block aligned |
| 同上 | `targetCastSize` | int64/64 | off 120 | byte | align N*4 | targetCast/sum | block aligned |
| 同上 | `smoothSize` | int64/64 | off 128 | byte | labelS conditional | smooth TBuf | 公开为 0 |
| 同上 | `gradLossSize` | int64/64 | off 136 | byte | none conditional | grad queue | non-none 0 |
| 同上 | `ignoreSize` | int64/64 | off 144 | byte | align N*4 | ignore TBuf | base 始终分配 |
| 同上 | `targetWeightSize` | int64/64 | off 152 | byte | weight conditional | targetWeight | no weight 0 |
| 同上 | `tBuf2Size` | int64/64 | off 160 | byte | mean=blockSize | two scalar TBuf | non-mean 0 |
| 同上 | `tBuf3Size` | int64/64 | off 168 | byte | mean align(max(P,Ntile)*4) | all-core vector | non-mean 0 |
| 同上 | `kTimes` | int64/64 | off 176 | stage count | smooth+weight split | smooth reduction | 公开 inactive |
| 同上 | `cOnceNum` | int64/64 | off 184 | element | split | smooth helper | 公开 inactive |
| 同上 | `cOnceNumTail` | int64/64 | off 192 | element | split | smooth tail | 公开 inactive |
| 同上 | `kTimesTail` | int64/64 | off 200 | stage count | split | smooth helper | 公开 inactive |
| 同上 | `cOnceNumTailAlign` | int64/64 | off 208 | element | split align | smooth helper | 公开 inactive |
| 同上 | `cacheStart` | int64/64 | off 216 | fp32 element | `log2(kTimes)*VReg/4` | smooth cache | 公开 inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | sch2 | `DoEmptyTensorTiling` | zero/default | arch35 entry return | N/A，无元素 | ascend950 JSON/CMake |
| R-002 | sch0,w0,label0 | `CELossGradCoreSplitInfo` | regbase POD | entry→WeightNone | base + weight_none + platform | 同上 |
| R-003 | sch0,w1,label0 | 同上 | 同上 | entry→WeightNotNone | base + weight_not_none + platform | 同上 |
| R-004 | sch1,w0,label0 | `CELossGradInnerCoreFullLoad` | 同上 | entry→FullLoad模板 | base + full_load + platform | 同上 |
| R-005 | sch1,w1,label0 | 同上 | 同上 | entry→FullLoad模板 | base + full_load + platform | 同上 |

## Kernel 实现

### 公共执行流程

entry 固定为单 AIV Kernel。`Base::Init` 解析 POD、按 former/tail 核设置 `logOffset/targetOffset/nPeerCoreNum`，绑定 GM 与 workspace。16 位输入通过 `LoadRegTensor` 的 unpack+Cast 升至 fp32；输出用 RINT Cast+Pack。ignore mask 通过 MicroAPI compare/select；mean 分母按条件使用 workspace+`SyncAll`。所有 `UpdateMask` 以剩余元素更新，防止 tail lane 写出。

### Route R-001：空张量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/cross_entropy_loss_grad_tiling_arch35.cpp` | `RunTiling/DoEmptyTensorTiling` | `GetShapeSize()==0` | sch2，blockDim=1，workspace=16MiB |
| Router | `op_kernel/arch35/cross_entropy_loss_grad.cpp` | constexpr chain | schId 非 0/1 | return |

#### Device 数据流

##### 函数级调用链

```text
cross_entropy_loss_grad<2,0,0,0,0> -> else -> return
```

##### 源码等价伪码

```text
if schId not 0 and not 1: return  # output has zero elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| system workspace | 0 | 16MiB | opaque | ACLNN | framework | caller/executor | Device body 不访问 |

##### 分层内存布局

N/A：entry 直接返回，不创建 AIV UB buffer；替代路径是零元素空操作。

##### 字节账与复用顺序

```text
UB_total=0；dynamic_ub=N/A；workspace_total=16MiB system workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | entry return | 输出元素数为0 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| empty | N/A | N/A | N/A | N/A | 无计算/写出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/cross_entropy_loss_grad_tiling_arch35.cpp` | `DoEmptyTensorTiling` | sch2/block1 |
| Kernel entry/router | `op_kernel/arch35/cross_entropy_loss_grad.cpp` | final else | return |
| Device body/helper | `op_kernel/arch35/cross_entropy_loss_grad.cpp` | entry | 无 body |

### Route R-002：分块无权重

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_arch35.cpp` | `RunTiling` | nonempty；`FullLoad()==false`；weight null | sch0,isWeight0 |
| Router | `arch35/cross_entropy_loss_grad.cpp` | `schId==0&&isWeight==0` | template key | `WeightNone<T,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WeightNone::Init -> Base::Init/InitUB -> WeightNone::Process
 -> [mean+ignore] WeightReduceSum/CopyOutWsAndReduce
 -> ProcessPerCore: ComputeIgnoreMask(when ignore>=0) -> GradReductionNone/MeanSum
 -> each row: CopyInLog -> ComputeLogPre -> cast/copy -> CopyOutLog
```

##### 源码等价伪码

```text
for N chunk of VRegBytes/4 rows:
  optional mask=target!=ignore
  q = none?grad[row] : (mean?grad/reduceFactor:grad); q*=mask when active
  for each row and C tile:
    tmp=exp(fp32(logProb))*q; tmp[target-tileBase]-=q
    cast_rint_or_copy valid calcLen -> out
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| log/out queues | `row*C+tile*A` | `B*d` each，`B=max(A,align_up(Ctail,64),256 for tmp only)` | T/ND | CopyIn/Compute | Compute/CopyOut | TPipe per tile | depth1 |
| target/grad queues | `targetOffset+nChunk` | targetSize / none gradLossSize | int64/T | GM DMA | mask/Grad | per N chunk | depth1 |
| ignore/targetCast | 0 | ignoreSize+targetCastSize | fp32 | mask/Grad | ComputeLogPre | Kernel | VReg chunk |
| fp32Buf4 | 0 | `max(A,align_up(Ctail,64),256)*4` | fp32 | exp/mul/target sub | output Cast/Copy | per tile | 原地复用 |
| mean buffers/workspace | core index | `2*tBuf2Size+tBuf3Size` UB；P fp32 GM | fp32 | ReduceSum | factor | launch | 仅 mean+ignore 实际跨核 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  io["log inQ + outQ · 2*L*d"]:2
  row["target + optional grad + ignore + targetCast"]:2
  tmp["fp32Buf4 · B*4"]:2
  mean["mean: blockBuf2 + blockBuf3 + weightMask"]:2
```

##### 字节账与复用顺序

```text
UB_total=2*L*d+targetSize+(none?gradLossSize:0)+ignoreSize+targetCastSize+B*4
         +(mean?2*tBuf2Size+tBuf3Size:0)
L=max(alignNCNum,align_up(Ctail,64)); B=max(A,align_up(Ctail,64),256)
workspace_total=P*4*blockSize+16MiB；dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| fp32Buf4 | ComputeLogPre | Cast/Copy | next C tile | queue ordering | previous tile copied out |
| workspace core slot | local ReduceSum | all cores | launch end | V→MTE3 event + SyncAll + MTE2→V | mean+ignore |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar/vector | S_V/V_S events | Scalar target/grad | MicroAPI/Scalar update | target/q/tile | target lane update 可见 |
| mean | V_MTE3→SyncAll→MTE2_V | local sum | global factor | workspace | fp32 |
| C loop | Queue + `UpdateMask` | DMA/reg compute | CopyOut | tile | 仅 valid lanes/bytes |
| cast | unpack/Cast; RINT/Pack | fp16/bf16 | fp32 compute/T output | RegTensor | 单次输出降精度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/cross_entropy_loss_grad_tiling_arch35.cpp` | `CELossGradCoreSplitInfo` | sch0 C tile/POD |
| Kernel entry/router | `op_kernel/arch35/cross_entropy_loss_grad.cpp` | sch0/w0 | WeightNone |
| Device body/helper | `arch35/cross_entropy_loss_grad_weight_none.h` | `ProcessPerCore/ComputeLog` | split no-weight |

### Route R-003：分块有权重

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_arch35.cpp` | split sizing | full-load false，weight nonnull | sch0,w1，bufferNum=16 |
| Router | `arch35/cross_entropy_loss_grad.cpp` | sch0/w1 | template key | `WeightNotNone<T,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
WeightNotNone::Process -> [mean] TargetWeightReduceSum
 -> ProcessPerCore: TargetWeight(scalar gather) -> optional ignore mask
 -> GradReductionNone or GradReductionMeanSumWeight
 -> per row/tile ComputeLogPre -> Cast/Copy -> CopyOut
```

##### 源码等价伪码

```text
targetWeight[row]=weight[target[row]]; if active ignore, multiply mask
mean: denom=sum_all_cores(targetWeight); q=grad/denom*targetWeight
none/sum: q=(grad[row] or grad)*mask*targetWeight
for C tiles: exp(logProb)*q; subtract q at target; masked tail write
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weight GM/targetWeight | target index | scalar per N；UB targetWeightSize | fp32 | scalar gather | Grad/mean | N chunk | weight values remain GM |
| inQueWeight | C tile offset | `B*4` | fp32 | public label0 不使用 C weight tile | inactive | allocated by class | label smoothing only consumer |
| base/fp32Buf4 | 同 R-002 | 同 R-002 | mixed | common | common | tile | depth1 |
| mean workspace | core index | P fp32 | fp32 | weighted ReduceSum | all cores | launch | SyncAll |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  io["log/out queues · 2*L*d"]:1
  weightQ["inQueWeight · B*4"]:1
  row["target/grad/ignore/targetCast/targetWeight"]:1
  tmp["fp32Buf4 · B*4"]:1
  mean["mean reduce buffers"]:4
```

##### 字节账与复用顺序

```text
UB_total(R-003)=UB_base(R-002)+B*4(inQueWeight)+targetWeightSize
workspace_total=P*4*blockSize+16MiB；mean 使用首 P 个 fp32
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| targetWeight | Scalar gather / optional Mul | Grad/denominator | next N chunk | S_V/Pipe events | target in `[0,C)` |
| fp32Buf4/queues | Compute/DMA | output | next tile | queue lifecycle | valid calcLen |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| gather→Vector | `PipeS2V` | Scalar target weight writes | Mul/Grad | targetWeight | fp32 |
| weighted mean | V_MTE3/SyncAll/MTE2_V | per-core sum | factor | workspace | 0 denominator无 guard |
| C output | MicroAPI mask + queue | Compute | CopyOut | tail | fp32 compute/RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_arch35.cpp` | split weight branch | bufferNum/targetWeight |
| Kernel entry/router | `arch35/cross_entropy_loss_grad.cpp` | sch0/w1 | weighted split body |
| Device body/helper | `arch35/cross_entropy_loss_grad_weight_not_none.h` | `TargetWeight/ProcessPerCore` | gather/weighted q |

### Route R-004：Full-load 无权重

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_arch35.cpp` | `CELossGradInnerCoreFullLoad` | finalTotal≤ubNSize 且 finalN≥1，weight null | sch1,w0 |
| Router | `arch35/cross_entropy_loss_grad.cpp` | sch1 | template full-load | `CELossGradFullLoad<T,int64,r,0,0,i>` |

#### Device 数据流

##### 函数级调用链

```text
FullLoad::Init -> Base::Init/InitUB -> allocate tmp alignNC*4
 -> Process -> [mean+ignore] TargetWeightReduceSumFullLoad
 -> ProcessPerCore: optional ComputeIgnoreAndTarget -> GradReduction
 -> ComputerEachBatch: DMA Ntile×C -> ComputeLogSoftmax -> DMA target
    -> ComputePredictGradLoss(gather/scatter) -> cast/copy -> Ntile×C GM
```

##### 源码等价伪码

```text
for N group of peerLoopNum=alignNC/A:
  q[group]=grad vector/scalar; apply mean factor and optional mask
  tmp[Ngroup,C]=exp(fp32(logProb))*broadcast(q)
  idx=row*A+target[row]; tmp[idx]-=q[row] via gather/scatter
  cast/copy only Ngroup*C valid bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| log/out queues | `peerN*C` | `alignNCNum*d` each | T/[Ntile,A] | one 2D DMA | MicroAPI/GM | per N group | depth1，GM blockLen=C*d |
| tmpBuf | `[row*A+c]` | `alignNCNum*4` | fp32/[Ntile,A] | exp*Q | target scatter/output | per N group | 原地 target 修正 |
| target/grad/ignore/q | N group offset | POD byte sizes | int64/T/fp32 | DMA/MicroAPI | Grad/scatter | per group | block aligned |
| mean reduce buffers | core slot | `2*tBuf2+tBuf3` UB + P fp32 GM | fp32 | ReduceSum | factor | launch | only mean+isIgnore |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in["log inQ · alignNC*d"]:1
  out["xGrad outQ · alignNC*d"]:1
  tmp["tmpBuf · alignNC*4"]:2
  target["target + optional grad"]:1
  iq["ignore + targetCast"]:1
  mean["mean reduce buffers"]:2
```

##### 字节账与复用顺序

```text
UB_total=2*alignNC*d+alignNC*4+targetSize+(none?gradLossSize:0)+ignoreSize+targetCastSize
         +(mean?2*tBuf2Size+tBuf3Size:0)
workspace_total=P*4*blockSize+16MiB；dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf | ComputeLogSoftmax | gather/scatter then output Cast | next N group | function order/queue | scatter completes before output |
| in/out queue | 2D DMA/Compute | Compute/2D DMA | next N group | EnQue/DeQue/FreeTensor | depth1 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| N×C compute | RegTensor + `UpdateMask` | log/q | tmp | VReg lanes | C tail zero/merge mask |
| target patch | gather/scatter | tmp/q/target | output cast | one lane/row | int64 cast/pack to int32 index |
| precision | unpack→fp32 exp/mul/sub→RINT pack | T | T | tmp | fp32 intermediate |
| mean | SyncAll protocol | partial sums | all cores | workspace | only nontrivial denominator |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_arch35.cpp` | `CELossGradInnerCoreFullLoad` | Ntile/alignNC/sch1 |
| Kernel entry/router | `arch35/cross_entropy_loss_grad.cpp` | sch1 template | FullLoad parameters |
| Device body/helper | `arch35/cross_entropy_loss_grad_full_load.h` | `ComputeLogSoftmax/ComputePredictGradLoss` | N×C vector + scatter |

### Route R-005：Full-load 有权重

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_arch35.cpp` | full-load estimator | fits 且 weight nonnull | sch1,w1，计入 weight C queue/targetWeight |
| Router | `arch35/cross_entropy_loss_grad.cpp` | sch1/w1 | template | `FullLoad<T,int64,r,1,0,i>` |

#### Device 数据流

##### 函数级调用链

```text
FullLoad::Process -> CopyInWeight(all C once) -> [mean] TargetWeightReduceSumFullLoad
 -> ComputeIgnoreAndTarget(DataCopyGather weight[target]) -> GradReduction(weighted)
 -> ComputerEachBatch -> ComputeLogSoftmax -> ComputePredictGradLoss -> CopyOut
 -> FreeTensor(weightLocal)
```

##### 源码等价伪码

```text
weightUB=DMA(weight[0:C]) once per core
for each N group: targetWeight=gather(weightUB,target); optional mask
q=grad*(mean?1/sum(targetWeight):1)*targetWeight
tmp=exp(logProb)*q; scatter(tmp[target]-q); cast/copy Ngroup*C
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQueWeight | 0 | `A*4` | fp32/[C padded A] | one DMA | gather | whole Process | depth1，只搬一次 |
| targetWeightBuf | N group | targetWeightSize | fp32 | DataCopyGather | Grad/Reduce | per group | mask 可原地作用 |
| full-load base/tmp | 同 R-004 | 同 R-004 | mixed | common | common | N group | 同上 |
| mean workspace | core | P fp32 | fp32 | weighted partial | all cores | launch | SyncAll |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in["log inQ · alignNC*d"]:1
  out["outQ · alignNC*d"]:1
  tmp["tmpBuf · alignNC*4"]:2
  weight["weight inQ · A*4"]:1
  targetW["targetWeight"]:1
  row["target/grad/ignore/q"]:1
  mean["mean buffers"]:1
```

##### 字节账与复用顺序

```text
UB_total(R-005)=UB_total(R-004)+A*4(inQueWeight)+targetWeightSize
workspace_total=P*4*blockSize+16MiB；mean 使用 P 个 fp32 slot
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| weight queue slot | CopyInWeight | all N groups gather | Process end | DeQue 后延长 LocalTensor 生命周期，末尾 Free | depth1 不再 Alloc 同 queue |
| tmpBuf | exp/mul | scatter/output | next N group | ordered Vector ops | same as R-004 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| weight gather | DataCopyGather + masks | weightUB/target | targetWeight | VReg/UB | int64 target cast to int32 gather index |
| mean | V_MTE3/SyncAll/MTE2_V | weighted partial | factor | workspace | fp32 |
| N×C | same as R-004 | same | output | tmp/queues | C/N VReg tails |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_arch35.cpp` | full-load weight sizing | queBuffer/targetWeight |
| Kernel entry/router | `arch35/cross_entropy_loss_grad.cpp` | sch1,w1 | weighted full-load |
| Device body/helper | `arch35/cross_entropy_loss_grad_full_load.h` | `Process/ComputeIgnoreAndTarget/GradReduction` | one-time weight load/gather |

## 算子特定优化

### Trick T-001：UB 可容纳时把多行完整 C 维合并为 full-load

#### 触发与源码依据

- 适用 Route：R-004、R-005
- 触发条件：Host 容量模型得到 `finalTotal<=ubNSize && finalN>=1`。
- Golden 来源：[`loss/cross_entropy_loss_grad/op_host/cross_entropy_loss_grad_tiling_arch35.cpp:186-236`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_host/cross_entropy_loss_grad_tiling_arch35.cpp#L186-L236)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t finalN = std::min(theoreticalMaxN, allowedN);
int64_t finalNAlign = Ops::Base::CeilAlign(finalN, static_cast<int64_t>(blockSize / dtypeSize));
int64_t finalTotal = ncBufferNum * finalN + tmpABuffer * finalNAlign;
if (finalTotal > ubNSize) {
    finalN = ubNSize / (ncBufferNum + tmpABuffer);
}
if (finalTotal > ubNSize || finalN < 1) {
    return false;
}
alignNCNum = finalN * alignColLoopNum;
ceLossGradTilingKey.schId = static_cast<uint32_t>(CROSS_ENTROPY_LOSS_GRAD_MODE_1);
```

#### 机制与设计原因

Host 把 input/output/fp32 tmp 的每行 C 成本和 target/grad/mask 等每行 N 成本分开估算，选择能驻留 UB 的最大 `finalN`。Device 随后一次 2D DMA 搬 `Ntile×C`，MicroAPI 广播每行 q 并用 gather/scatter 修正 target，避免 split 路线对每一行重复建立 C tile 管线。

#### 迁移边界

- 适用：行独立且每行有一个稀疏索引修正的 N×C 梯度。
- 前提：完整 C 及至少一行元数据能装入 UB；dtype/target dtype 与容量模型一致。
- 失效：`finalN<1` 或复核仍超 UB 时必须转 sch0 split。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Full-load 以 gather/scatter 完成每行 one-hot 修正

#### 触发与源码依据

- 适用 Route：R-004、R-005
- 触发条件：schId=1，完成 `exp(logProb)*q` 后。
- Golden 来源：[`loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_full_load.h:322-356`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_full_load.h#L322-L356)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
MicroAPI::Arange(regIndex1, 0);
MicroAPI::Duplicate(regIndex0, totalCLenAlign, regAllB32);
MicroAPI::Mul(regIndex1, regIndex0, regIndex1, regAllB32);
MicroAPI::Add(regTargetB32, regTargetB32, regIndex1, regAllB32);
MicroAPI::DataCopyGather(regGather, xGradTmpAddr + loop * vfLen * totalCLenAlign,
                         (MicroAPI::RegTensor<uint32_t>&)regTargetB32, regAllGather);
MicroAPI::Sub(regGather1, regGather, regNllLossGrad, regAllB32);
MicroAPI::DataCopyScatter(xGradTmpAddr + loop * vfLen * totalCLenAlign, regGather1,
                          (MicroAPI::RegTensor<uint32_t>&)regTargetB32, regAllGather);
```

#### 机制与设计原因

`Arange*alignColLoopNum + target` 直接形成各行目标元素在二维 UB 临时矩阵的扁平索引；一次 gather 取出多个行的目标 lane，向量减 q 后 scatter 回原位。它把逐行 Scalar `GetValue/SetValue` 改成跨行 RegTensor 操作，同时复用已经物化的 fp32 tmp，不新增 one-hot Tensor。

#### 迁移边界

- 适用：每行一个合法索引、行 stride 固定的 UB resident 矩阵。
- 前提：target 可安全转为 int32 索引且在 `[0,C)`；tail mask 与 int64 Pack 必须保留。
- 失效：C/索引超 int32 或 target 非法时不能复用此 scatter 方案。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-003：mean 只在分母非平凡时做全核归约

#### 触发与源码依据

- 适用 Route：R-002～R-005 的 mean 子路线。
- 触发条件：full-load 为 `reduction==mean && (isWeight || (isIgnore && !isWeight))`；split 无权重仅 `ignoreIndex>=0`，有权重总归约。
- Golden 来源：[`loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_full_load.h:428-442`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_full_load.h#L428-L442)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (reductionKey == REDUCTION_MEAN && (isWeight == 1 || (isIgnore == 1 && isWeight == 0))) {
  TargetWeightReduceSumFullLoad();
}
for (uint64_t n = 0; n < this->nLoopTimes; n++) {
  ProcessPerCore(n * peerLoopNum, peerLoopNum);
}
```

#### 机制与设计原因

无 weight 且无非负 ignore 时，mean 分母恒为 N，直接使用 `rowVal`，不触发 workspace 写读与 `SyncAll`。只有 weight 或有效 ignore 令分母依赖数据时才执行全核标量归约，源码可证明减少了不必要的同步和 GM 标量交接。

#### 迁移边界

- 适用：能由 shape 常量证明归约分母的 mean 类算子。
- 前提：无数据依赖的 mask/weight；若 ignore 或 weight 生效必须走归约。
- 失效：任何样本权重/有效性依赖输入数据时不可用常量 N 替代。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-002/R-003 split | R-004/R-005 full-load | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | full-load 容量失败后 fallback | 至少一行完整 C 可驻 UB，优先 | Host 容量模型 |
| 模板与 dtype | WeightNone/WeightNotNone，reduction runtime | FullLoad 的 reduction/weight/ignore constexpr | 消除 full-load 内部分支 |
| 循环与分核 | N chunk→逐行→C tile | N group→整块 N×C | UB 容量 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | C tile queues+fp32 tmp | N×C queues+tmp；可选全 C weight | 均无 dynamic UB/通信 |
| 内存总占用与复用 | 较小，fp32Buf4 每 tile 覆盖 | 较大，tmp 原地 scatter 后直接输出 | 容量换管线粒度 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 逐行 DataCopyPad + MicroAPI + Scalar target patch | 2D DataCopyPad + MicroAPI gather/scatter | 都是 AIV_VECTOR |
| 同步、精度与 tail | Queue/event；C tail mask | Queue/event；N/C tail mask | 16 位均 fp32 中间+RINT |
| fallback/失败边界 | A=0 无进一步 route | 容量不足回 split | empty 单独 R-001 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持 Ascend 950PR/950DT |
| Host/Launcher | classic 单层 Tiling | 同 A2 | regbase，full-load 优先 + split fallback + empty key |
| Kernel/资源 | Vector API，int64 target | 同 A2 | MicroAPI RegTensor/VReg，int64 public binary；2D full-load/gather-scatter |

## 限制与失败边界

- 正式公开域仅 target int64、`labelSmoothing=0`；ascend950 OpDef/binary 中的 int32 target和源码 labelS=1组合不是本 ACLNN 卡的公开路线。
- 950 `isIgnore` 仅由 `ignoreIndex>=0` 置位；负 ignoreIndex 被当作“无 ignore”。若 target 实值等于负 ignoreIndex，full-load 的 target range/assert 与公开文档允许 ignore 值存在边界差异，复用时必须避免将负值送入 gather/scatter。
- target int64 在 full-load 中转 int32 作为 UB gather/scatter 索引；固定仓没有 C 超 int32 范围的显式 Host guard。
- split 路线若 UB 余量导致 `alignColLoopNum=0`，Host 没有后续 fallback；mean 分母为 0 时 Device 没有显式 guard。
- 自动生成 ACLNN bridge/executor、16MiB 系统 workspace 内部用途固定仓不可见；本次未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnCrossEntropyLossGrad.md` | ACLNN prototypes/支持表 | 950公开性、int64、label0 |
| 入口实现 | `op_host/cross_entropy_loss_grad_def.cpp` | `CrossEntropyLossGrad` | ascend950 OpDef/AutoContiguous |
| Host/Tiling/Launcher | `op_host/cross_entropy_loss_grad_tiling.cpp` | `Tiling4CrossEntropyLossGrad` | regbase selector |
| Host/Tiling/Launcher | `op_host/cross_entropy_loss_grad_tiling_arch35.cpp` | `RunTiling` | validation、sch、UB/workspace/key |
| Host/Tiling/Launcher | `op_kernel/arch35/cross_entropy_loss_grad_tiling_data.h` | regbase POD | ABI fields |
| Host/Tiling/Launcher | `op_kernel/arch35/cross_entropy_loss_grad_tiling_key.h` | template key macros | key axes/allowed selections |
| Kernel entry/body | `op_kernel/arch35/cross_entropy_loss_grad.cpp` | entry | AIV type与router |
| Kernel entry/body | `op_kernel/arch35/cross_entropy_loss_grad_base.h` | Base | 公共 UB/MicroAPI/sync/precision |
| Kernel entry/body | `op_kernel/arch35/cross_entropy_loss_grad_full_load.h` | FullLoad | N×C/gather-scatter |
| Kernel entry/body | `op_kernel/arch35/cross_entropy_loss_grad_weight_none.h` | WeightNone | split 无权重 |
| Kernel entry/body | `op_kernel/arch35/cross_entropy_loss_grad_weight_not_none.h` | WeightNotNone | split 有权重 |
| Kernel entry/body | `common/inc/op_kernel/platform.h` | `GetVRegSize/GetUbBlockSize` | VReg/block helper |
| 构建接线 | `op_host/config/ascend950/cross_entropy_loss_grad_binary.json` | `op_list` | 950 binary dtype组合 |
| 构建接线 | `op_kernel/CMakeLists.txt` | ascend950 `add_kernel_sources` | arch35 source/compile option |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | plain struct 224B/align8、同 struct Host/Device | 字段顺序、float 后 padding、offset | `CrossEntropyLossGradRegbaseTilingData` |
| 调用顺序 | full-load 优先，失败 split；empty 先返回 | selector 优先级不能倒置 | `RunTiling` |
| workspace/内存 | mean 每核 fp32 slot；full-load N×C tmp | SyncAll 前不可读；UB 按实际 depth/条件缓冲 | `CopyOutWsAndReduce/InitUB` |
| 同步 | S/V/MTE event、Queue、必要时 SyncAll | Scalar target patch与DMA/Vector可见性 | `PipeS2V/PipeV2S/PipeV2M` |
| 精度与尾块 | 16 位升 fp32、RINT Pack；UpdateMask | int64 target pack、tail mask、有效 GM blockLen | `LoadRegTensor/CopyOutRegTensor` |
| 失败边界 | 参数/Tiling失败阻断 launch；full-load容量回split | label/int32非公开、negative ignore、A=0不应推测 fallback | `GetTilingAttr/CheckDtype/CELossGradInnerCoreFullLoad` |