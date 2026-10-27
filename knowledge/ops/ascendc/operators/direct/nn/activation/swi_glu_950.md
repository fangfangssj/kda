---
type: Operator
title: "SwiGlu 950 Direct Invoke 算子知识"
description: "说明 cann_bench::swi_glu 在 950 上的语义、Host/tiling/POD、三类实现 family、九个可达叶、UB 流水、尾块与精度边界。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swi_glu/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swi_glu/op_host/swi_glu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swi_glu/op_host/config/ascend950/swi_glu_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swi_glu/op_host/swi_glu_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swi_glu/op_kernel/arch35/swi_glu_single.hpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swi_glu/op_kernel/arch35/glu_tiling_kernel.hpp
    role: implementation
created_at: "2026-08-24T19:33:26Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# SwiGlu 950 Direct Invoke 算子知识

## 算子契约

| 项目 | 契约 |
|---|---|
| 入口 | Python `cann_bench.swi_glu` → dispatcher `cann_bench::swi_glu` → `swi_glu_npu` → dtype 专用 `launch_swi_glu_*`。这是 Direct Invoke，不是 ACLNN 两段式入口。 |
| 语义 | 沿 `dim` 等分为 A/B，`Y=A/(1+exp(-A))*B`。A 是该轴前半，B 是后半。 |
| 输入 | rank 2～8；标准 contiguous strided Tensor；FP32、FP16 或 BF16；split 轴长度为正偶数。 |
| 输出 | 同 rank、dtype、device，split 轴减半；`torch::empty(outputShape,input.options())` 新分配连续存储，不是 view，也不继承输入 strides。 |
| 执行 | 每次只提交一个 AIV-only Kernel；无 L1/L0/Cube、Device workspace、跨 Kernel 中间边或 dynamic UB 参数。 |
| 可达 family | P0 普通 DB 主干；P1 BF16 `colLen==1` compact；P2 条件性普通 SB。九个 active 叶均由这三类 body/template 覆盖。 |

### 双半与 flatten 语义

归一化切分轴为 `d`，令：

~~~text
prefix = prod(shape[0:d])
suffix = prod(shape[d+1:rank])
rowLen = prefix
colLen = (shape[d]/2)*suffix = prod(shape[d:rank])/2

input_view  = [rowLen, 2*colLen]
output_view = [rowLen, colLen]
A[row,j] = input[row, j]
B[row,j] = input[row, colLen+j]
Y[row,j] = A[row,j] / (1 + exp(0-A[row,j])) * B[row,j]
~~~

“前半/后半”是每个折叠行内的两个连续段，不是整个 Tensor 扁平地址的全局前后半。若 `colLen==1`，由 split 轴正偶数可推出 `shape[d]==2 && suffix==1`，所以 GM 物理顺序必为 `[A0,B0,A1,B1,...]`；compact family 的 pair 搬运只在这个域成立。

### 校验、空域与 checked arithmetic

`CheckAndFlatten` 检查 rank、strided layout、contiguous、dtype、dim 范围和 split 轴正偶数。Meta 与 PrivateUse1 共用这套 shape 逻辑；Meta 只分配 meta output，不启动 Device。split 轴之外的零维可通过前置 shape 检查，但随后 `CalcSwiGluTiling` 会因 `rowLen==0 || colLen==0` 拒绝 launch。

当前源码的 shape 乘法不是 checked arithmetic，必须区分已有检查与复用时应补的安全契约：

| 量 | 实际 producer/检查 | 安全关系与边界 |
|---|---|---|
| `rowLen` | `uint64_t` 从 1 连乘 `shape[0:d]`；无显式 overflow check | 非空可达域要求 `rowLen>0` |
| `shapeAfter/colLen` | `uint64_t` 连乘 `shape[d:rank]` 后除 2；仅先证明 split 轴为正偶数 | `2*colLen=shape[d]*suffix` |
| 元素账 | 源码未显式与 `numel()` 交叉核对 | 应保持 `input.numel=rowLen*2*colLen`、`output.numel=rowLen*colLen` |
| tile/offset | Host 和 Device 用 `uint64_t` 乘加；无独立 overflow check | 最末 A/B/Y offset 必须落在各 Tensor extent 内 |
| copy/VF 窄化 | row/col 受 4095、65535 和 UB 限制；`curCalLen` 转 `uint32_t`、repeat 转 `uint16_t` | 要求 `curCalLen<=UINT32_MAX` 且 `ceil(curCalLen/64)<=UINT16_MAX` |

新 Host 的最小加固是在每次乘法前检查 `a==0 || b<=UINT64_MAX/a`，并在 launch 前核对两条 `numel` 等式；这不是现有源码已经实现的检查。

## Host、POD 与启动图

### Host/Launcher ABI

`swi_glu_npu` 安装输入 device guard，分配 output，取得 GM 指针和当前 NPU stream，再按 dtype 进入以下 C ABI：

~~~cpp
void launch_swi_glu_{float|half|bfloat16}(
    GM_ADDR input, GM_ADDR output,
    uint64_t rowLen, uint64_t colLen, void* stream);
~~~

`dim` 已被 flatten 消解，dtype 由 launcher/entry 编码。每个 launcher 都先查询 AIV 核数、每核 UB 和当前架构，再构造 `SwiGluLaunchTiling{tiling,blockDim}`。workspace 固定为 0：

~~~text
validate/flatten -> allocate output -> dtype launcher
  -> CalcSwiGluTiling
  -> exactly one kernel<<<blockDim,nullptr,current_stream>>>(
         input, output, tiling_by_value)
~~~

`OpCommand::RunOpApi("SwiGlu",directCall)` 包装提交；`directCall` 在 void launcher 返回后固定返回 0。源码没有读取 `<<<>>>` 的异步提交状态，因此固定 0 不能证明 Device 异步成功。

### POD/ABI

`SwiGluTilingData` 按值传入 Kernel。按字段声明顺序和 8B 自然对齐推导 `sizeof=80`、`alignof=8`；该布局未由卡片内的编译态 `static_assert` 独立证明。

| 字段 | 类型/offset | Host producer | Device/launcher consumer | 可达边界 |
|---|---|---|---|---|
| `is32BAligned` | u32 / 0 | `colLen%(32/s)==0` | 普通 `SwigluSingle::Process/GetTilingAndOffset` | compact 不读 |
| `isDoubleBuffer` | u32 / 4 | DB 默认 1；完整 S 谓词命中写 0 | launcher 选择 DB/SB entry | body 不据此分支；compact 忽略 |
| `rowLen` | u64 / 8 | flatten | 普通 tiling；compact 行分区 | 必须 >0 |
| `colLen` | u64 / 16 | flatten | 普通 tiling/offset | 必须 >0；compact 已知为 1 |
| `baseRowLen` | u32 / 24 | 选中 base shape | 普通四区 tile | compact 不读 |
| `baseColLen` | u32 / 28 | 选中 base shape | 普通四区 tile | compact 不读 |
| `activateLeft` | u32 / 32 | 默认 0 | 可达 body 不读 | inactive |
| `biasIsEmpty` | u32 / 36 | 默认 0 | 可达 body 不读 | inactive |
| `quantScaleIsEmpty` | u32 / 40 | 默认 0 | 可达 body 不读 | inactive |
| `activateScaleIsEmpty` | u32 / 44 | 默认 0 | 可达 body 不读 | inactive |
| `swiColLen` | u64 / 48 | 默认 0 | 可达 body 不读 | inactive |
| `perRowLen` | u64 / 56 | 默认 0 | 可达 body 不读 | inactive |
| `modRowLen` | u64 / 64 | 默认 0 | 可达 body 不读 | inactive |
| `usedCoreNum` | u32 / 72 | 默认 0 | 可达 body 不读 | 76～79 tail padding |

实际核数来自 `<<<launch.blockDim,...>>>` 与 Device `GetBlockNum()`，不是 POD 的 `usedCoreNum`。

## Route family 与完整叶映射

### 实际 selector 顺序

令 `s=sizeof(dtype)`，并定义：

~~~text
A = (colLen % (32/s) == 0)
C = (dtype == BF16 && colLen == 1)
S = (DB.tileNumPerCore == 2
     && CalcOptTiling<1>(SB) succeeds
     && SB.tileNumPerCore == 1
     && SB.usedCoreNum >= DB.usedCoreNum)
~~~

生产链顺序必须保持：

~~~text
H0: plugin dtype dispatch
H1: calculate DB candidate; if S, replace POD/base shape with SB
H2: BF16 && colLen==1 ? compact entry
H3: otherwise isDoubleBuffer ? DB entry : SB entry
D1: ordinary body chooses aligned/nonaligned copy by is32BAligned
~~~

compact 的 H2 在普通 DB/SB entry 分发之前，但在公共 tiling 之后：若 H1 失败，compact 不会绕过 tiling；若 C 成立，H1 产生的 DB/SB body 选择被忽略，但其 `blockDim` 仍用于 compact launch。

### family 重要性

这里不对互斥 dtype 建立无源码依据的全序。family 按合法域覆盖面、selector 优先级和资源价值排序；family 内的 dtype/alignment 叶同级。

| 重要性 | family / 状态 | 完整域 | 价值与退化关系 |
|---:|---|---|---|
| P0 | F-GDB 普通 DB / active 主干 | `!C && !S`，覆盖三 dtype 与 aligned/unaligned | 默认最广域；公共二维 tile 与 depth-2 Queue。S 成立时由 Host 替换为 P2；C 成立时由 launcher 短路 P1。 |
| P1 | F-C1 BF16 compact / active 特化 | `C`，与 H1 得到的 DB/SB 值无关 | selector 在普通 entry 前短路；避免单元素行按 32B UB padding。固定 UB 前提失败时无普通 body fallback。 |
| P2 | F-GSB 普通 SB / active（强条件可达） | `!C && S`，覆盖三 dtype 与 aligned/unaligned | 只在每核 tile 从 2 降 1 且不减少用核数时替换 DB；SB 计算失败或价值谓词不成立就保持 P0。 |

三类 family 都有生产 selector 到 entry 的真实调用边，因此按 reachability 均为 active；P2 虽依赖强 shape/resource 谓词，也不是仅存在于源码外假设中的 conditional。该实现没有额外的 conditional Kernel family：SB 失败时只保留 DB，其他失败边也不启动备用 entry。未被 launcher 引用的源码 shell 在后文单列为 dormant。

### 九个 active 叶闭包矩阵

`T16` 只是表格去重记号：FP16 叶实际绑定 `half` 并进入 half entry，BF16 叶绑定 `bfloat16_t` 并进入 bfloat16 entry，不存在运行时联合类型；BF16 普通叶都显式排除 C。所有普通叶的 `blockDim=min(totalTileNum,AIVCoreNum)`，每 core 按 `tileIdx=blockIdx+k*blockDim` 轮转；workspace/L1/L0 均为 0。

| leaf（family） | 精确谓词 | selector/POD → entry/template | UB 与同步 delta | tail/precision/fallback |
|---|---|---|---|---|
| R-002（P0）FP32 aligned DB | `dtype=FP32 && A && !S` | H0 float；H1 DB；H3 → `swi_glu_float_expsub_db_aiv`；`SwigluSingle<SwigluExpSubVector<float,2>,float,float>` | 3 个 depth-2 TQue；实际 `24L` B；TQue lifecycle | 32B block DMA；四区 tail + 64-lane mask；FP32 全程；tiling/launch 失败即报错 |
| R-004（P0）FP32 unaligned DB | `dtype=FP32 && !A && !S` | 同 R-002 entry/template，D1 走 nonaligned methods | depth-2；实际 `24L` B，`L=baseRow*align_up(baseCol,8)` | byte-unit DataCopyPad；只写有效列；FP32；无跨 Route 重试 |
| R-006（P0）T16 aligned DB | `dtype in {FP16,BF16} && !C && A && !S` | H0 half/BF16；H1 DB；H3 → `swi_glu_{half\|bfloat16}_expsub_db_aiv`；`SwigluSingle<SwigluExpSubVector<T16,2>,T16,T16>` | 3 个 depth-2 TQue；实际 `12L` B | 32B block DMA；T16→FP32→`CAST_RINT` T16；mask tail；失败报错 |
| R-008（P0）T16 unaligned DB | `dtype in {FP16,BF16} && !C && !A && !S` | 同 R-006 entry/template，D1 走 nonaligned methods | depth-2；实际 `12L` B，`L=baseRow*align_up(baseCol,16)` | byte-unit padded row；有效 bytes 回写；FP32 中间值；失败报错 |
| R-001（P1）BF16 col1 compact | `dtype=BF16 && colLen==1` | H1 后 H2 → `swi_glu_bfloat16_col1_compact_aiv`；`SwiGluNarrowBf16`；只消费 `rowLen` 和 launch blockDim | inQ depth2、outQ depth1、3 TBuf；固定 135168B；TQue + 显式 PIPE_V | 连续 pair copy；6144 行 tile；BF16→FP32，`Muls+Exp`，RINT；count tail；无 Device fallback |
| R-003（P2）FP32 aligned SB | `dtype=FP32 && A && S` | H1 写 DB=0；H3 → `swi_glu_float_expsub_sb_aiv`；`SwigluSingle<SwigluExpSubVector<float,1>,float,float>` | 3 个 depth-1 TQue；实际 `12L` B | 与 R-002 相同 DMA/tail/FP32；SB 候选不成立则不进入本叶而保持 P0 |
| R-005（P2）FP32 unaligned SB | `dtype=FP32 && !A && S` | 同 R-003 entry/template，D1 nonaligned | depth-1；实际 `12L` B，padded L | 与 R-004 相同 valid-only tail/FP32；无运行时 retry |
| R-007（P2）T16 aligned SB | `dtype in {FP16,BF16} && !C && A && S` | H1 写 DB=0；H3 → `swi_glu_{half\|bfloat16}_expsub_sb_aiv`；`SwigluSingle<SwigluExpSubVector<T16,1>,T16,T16>` | 3 个 depth-1 TQue；实际 `6L` B | 与 R-006 相同 aligned tail/FP32 intermediate/RINT；S 不成立保持 P0 |
| R-009（P2）T16 unaligned SB | `dtype in {FP16,BF16} && !C && !A && S` | 同 R-007 entry/template，D1 nonaligned | depth-1；实际 `6L` B，padded L | 与 R-008 相同 valid-only tail/精度；无运行时 retry |

| family | Host/selector 与 POD 锚点 | entry/body 锚点 | 后文关键伪码 |
|---|---|---|---|
| F-GDB / F-GSB | `op_kernel/swi_glu_kernel.cpp` 的 `GluSingleTilingCalculator::CalcTiling`、三个 `launch_swi_glu_*`；`swi_glu_tilingdata.h` | 同文件 `*_expsub_{db\|sb}_aiv`；`arch35/swi_glu_single.hpp`；`swi_glu_expsub.h` | base shape/S、二维 tile offset、四区 DMA、Queue lifecycle、ExpSub VF |
| F-C1 | `launch_swi_glu_bfloat16` 的 `colLen==1` 分支；POD `rowLen` 与公共 blockDim | `swi_glu_bfloat16_col1_compact_aiv`；`swi_glu_narrow_bf16.h` | 连续行分核、pair copy/Gather、135168B UB、Muls+Exp 与 count tail |

## Tiling、分核与普通 body

### Host 资源预算与 base shape

950 上 FP32 的 `s=4`，FP16/BF16 的 `s=2`。令 `U` 为每核 UB，`N` 为 AIV 核数：

~~~text
u32   = 32/s
pack  = 512/s
A     = (colLen % u32 == 0)

host_bytes_per_element:
  DB: FP32=28, FP16/BF16=20
  SB: FP32=16, FP16/BF16=14

maxTileLen  = align_down(U/host_bytes_per_element, u32)
baseColUp   = min(colLen,maxTileLen,A ? 65535 : floor(65535/s))
baseRowUp   = min(rowLen,4095)
totalTiles  = ceil(rowLen/baseRow)*ceil(colLen/baseCol)
usedCores   = min(totalTiles,N)
~~~

Host 预算沿用官方 body 的保守 TBuf 账；生产可达 `SwigluExpSubVector` 实际只分配 inA/inB/out 三条 Queue，因此实际 UB 小于 selector 预算。不能因实际 body 无 TBuf 就擅自放大 Host tile：route 选择仍以原预算为事实。

候选 base shape 的比较顺序为：使用核数更多、总 tile 更少、base 面积更大、完整 base tile 更多。若 input stride 超过 DMA 字段上限，固定 `baseRow=1`；否则 base col 以 512B pack 为步长缩小搜索。DB 必须先成功，才考虑 S 对应的 SB 替换。

### 普通二维 tile、四区 tail 与 DMA

~~~text
rowTileNum = ceil(rowLen/baseRowLen)
colTileNum = ceil(colLen/baseColLen)
for tileIdx = GetBlockIdx(); tileIdx < rowTileNum*colTileNum;
    tileIdx += GetBlockNum():
  rowTile = tileIdx / colTileNum
  colTile = tileIdx % colTileNum
  gmRow   = rowTile*baseRowLen
  Aoff    = gmRow*(2*colLen) + colTile*baseColLen
  Boff    = Aoff + colLen
  Yoff    = gmRow*colLen + colTile*baseColLen
~~~

`tailRow=rowLen%baseRow`、`tailCol=colLen%baseCol`。每 tile 只可能落在 baseRow/baseCol、baseRow/tailCol、tailRow/baseCol、tailRow/tailCol 四区之一；不存在相应 tail 时，线性 tile 计数不会消费预计算的零长参数。A/B 使用相同 `calRow,calCol`，B 仅多固定 `colLen` 偏移。

| copy path | UB 计算长度 | input copy | output copy | tail 安全 |
|---|---|---|---|---|
| aligned | `curCalLen=calRow*calCol` | `blockCount=calRow`；`blockLen=calCol*s/32`；`srcStride=(2*colLen-calCol)*s/32`，单位 32B block | 相同 blockLen；`dstStride=(colLen-calCol)*s/32` | base/tail col 都是 32B 整倍数；VF 最后 repeat 用 mask |
| unaligned | `curCalLen=calRow*align_up(calCol,32/s)` | `DataCopyPad`；blockLen/stride 分别为 `calCol*s`、`(2*colLen-calCol)*s` bytes | 只提交 `calCol*s` bytes/row；dstStride 为 `(colLen-calCol)*s` bytes | padding lane 可能参与逐元素计算，但无归约/跨 lane 依赖，且不写回 GM |

多行 tile 的 base-col 与 tail-col stride 都要可表示；超限时 `baseRow=1`，copy 参数把单行 stride 置 0。output stride 小于同 tile 的 input stride，受同一上界覆盖。

### 普通 UB 流水与 DB/SB 语义

DB/SB 是三个 `TQue` 的 slot 深度，不是 A/B 双半，也不是双 Kernel：

~~~text
CopyIn:
  a = inA.AllocTensor -> DMA(A) -> inA.EnQue
  b = inB.AllocTensor -> DMA(B) -> inB.EnQue
Compute:
  a,b = DeQue; y = out.AllocTensor
  VF(y,a,b,curCalLen)
  FreeTensor(a,b); out.EnQue(y)
CopyOut:
  y = out.DeQue -> DMA(Y) -> out.FreeTensor
~~~

`BufferNum=2` 时各 Queue 有两个 slot，`BufferNum=1` 时各一个。源码控制流仍逐 tile 执行 CopyIn→Compute→CopyOut，没有显式预取下一 tile 的额外循环；Queue 事件负责 DMA/VF 生命周期和 slot 轮换，不能据 depth=2 推断跨 tile 数值依赖。

### 普通精度路径

~~~cpp
for each 64-lane repeat:
  mask = UpdateMask<float>(remaining)
  load A,B
  if T16: unpack and cast A,B to FP32
  denominator = ExpSub(0.0f,A,mask)
  denominator = denominator + 1.0f
  activated   = A / denominator
  Y           = activated * B
  if T16: CAST_RINT and pack to original dtype
  masked store
~~~

FP32 叶全程 FP32。FP16/BF16 普通叶都升到 FP32 计算并以 `CAST_RINT` 降回原 dtype。alignment 与 DB/SB 只改变 DMA、base shape 和 Queue depth，不改变算术顺序或舍入边界。

## P1 compact body

### selector、分核与关键伪码

compact 仍使用公共 tiling 算出的 `blockDim`，不按 6144 行 tile 数重新 launch：

~~~text
rowsPerCore = ceil(rowLen/GetBlockNum())
coreBegin   = GetBlockIdx()*rowsPerCore
coreEnd     = min(coreBegin+rowsPerCore,rowLen)
if coreBegin >= rowLen: return

for row in [coreBegin,coreEnd) step 6144:
  count = min(6144,coreEnd-row)
  copy input[row*2 : row*2+2*count] as one BF16 byte block
  GatherMask(even lanes) -> A_bf16
  GatherMask(odd lanes)  -> B_bf16
  cast A,B to FP32
  C = A/(1+exp(-A))*B
  cast_rint C to BF16
  copy count BF16 to output[row:]
~~~

core 拥有连续、不重叠的半开行区间；多余 core 由 `coreBegin>=rowLen` 返回。末块 count 同时限制 input 的 `2*count` BF16、Gather repeat、向量长度和 output 的 count BF16。

### UB、同步与精度

| UB 区 | 容量 | 生命周期 |
|---|---:|---|
| `inputQueue_` | depth 2 × `6144*2*2` B = 49152B | CopyIn EnQue；Gather 后 FreeTensor |
| `outputQueue_` | depth 1 × `6144*2` B = 12288B | RINT cast EnQue；CopyOut 后释放 |
| `aBuffer_/bBuffer_` | 各 `6144*4` B | Gather 写从 BF16 offset 6144 开始的高地址 view，barrier 后 cast 到 offset 0 的 FP32 view |
| `computeBuffer_` | `6144*4` B | Muls/Exp/Adds/Div/Mul 原地阶段复用 |
| 合计 | 135168B | 无 dynamic UB 或 workspace |

两个 Gather 后和各条 Vector 指令之间使用 `PipeBarrier<PIPE_V>`；Queue 管理 copy↔compute。compact 用 `Muls(A,-1)->Exp`，普通 body 用 `ExpSub(0,A)`；两者代数语义和 BF16→FP32→RINT BF16 边界相同，但源码没有证明两种指数序列 bitwise 一致。

普通 nonaligned BF16 单列会把每行一个真实元素扩成 16 个 BF16 UB lane；compact 将连续 pair 一次搬入再拆偶/奇 lane。该优化依赖 `colLen==1` 推导出的 pair layout，不能推广到 `colLen>1`。

## fallback、dormant 路线与不变量

### fallback/失败边

| 起点 | 条件 | 出边 |
|---|---|---|
| DB tiling | DB 计算失败 | 直接抛错；不会尝试 SB，也不会进入 compact |
| SB candidate | 计算失败、每核 tile 不是 1，或用核数下降 | 保持 P0 DB；这是 Host 选择退化，不是运行时 retry |
| BF16 compact | C 成立 | 无条件覆盖普通 DB/SB entry；固定 135168B UB 没有显式 `ubSize` guard，也无普通 body fallback |
| aligned/unaligned | `is32BAligned` | Device 确定性选择 copy 方法；DMA 失败不会切换另一方法 |
| Host 校验/平台查询 | 任一失败 | launch 前终止 |
| Device launch | 异步错误 | launcher 源码不读取状态，不形成其它 Route 出边 |

### dormant entries

生产 launcher 只调用 `*_expsub_{db|sb}_aiv` 和 compact entry。下列源码存在但不可达：

| dormant family | entry/body | 不可达原因 | 重新启用边界 |
|---|---|---|---|
| faithful shells | `swi_glu_{float\|half\|bfloat16}_{db\|sb}`；`SwigluVector` 或 `SwigluVectorBF16` | 三个 launcher 无调用边，且 shell 不设置 AIV-only task metadata | 需要显式 launcher selector，并补齐 950 direct task metadata 与独立精度/资源验证 |
| adapted non-ExpSub shells | `swi_glu_{float\|half\|bfloat16}_{db\|sb}_aiv`；官方 `SwiGLU`/BF16 body | 有 AIV marker，但生产 launcher 仍无调用边 | 需要新增生产 selector；不能与当前 ExpSub 叶混记为 active |

### Route 间不变量

- 普通八叶共享同一个二维 tile、A/B/Y offset、四区 tail、round-robin 分核和 ExpSub VF；dtype、BufferNum、copy path 是唯一模板 delta。
- aligned/unaligned 只改变 DMA 单位与 UB 行跨度；GM 有效域、算术和 output extent 不变。
- DB/SB 只改变 Host base shape 和三个 Queue 的 slot 深度；数学顺序不变。
- compact 保持数学语义和 BF16 I/O 舍入边界，但改写分核、搬运和指数指令序列，不能与普通 BF16 body 拼接。
- 所有 active 叶均为单 AIV Kernel；GM input/output 由 Tensor 持有，当前 stream 提供调用次序；没有显式 event、L1/L0 或 workspace。

## 实现技巧

### `ExpSub(0,A)` 合并取负与指数输入

适用普通八叶。核心关系是操作数顺序：

~~~cpp
Duplicate(zero,0.0f);
ExpSub(denominator,zero,srcA,mask);
Adds(denominator,denominator,1.0f,mask);
~~~

交换 zero 与 A 会得到 `exp(A)`。缺少 ExpSub primitive 时才退化为 `Muls(A,-1)` 后 `Exp`；mask、FP32 中间值和 T16 输出舍入必须保持。

### BF16 单列连续 pair 搬运

仅适用 R-001：

~~~cpp
DataCopyExtParams p{
    1, count*2*sizeof(bfloat16_t), 0, 0, 0};
DataCopyPad(inputLocal,inputGm[row*2],p,pad);
~~~

一次复制 `[A0,B0,...]`，再 Gather 偶/奇 lane，避免逐行 32B padding。若 `colLen>1`，布局是 `[A-row,B-row]` 而非相邻 pair，必须退回普通 family。

### 受限 SB 替换

~~~cpp
if (db.tileNumPerCore == 2
    && CalcOptTiling<1>(sb)
    && sb.tileNumPerCore == 1
    && sb.usedCoreNum >= db.usedCoreNum) {
  use(sb);
  isDoubleBuffer = 0;
}
~~~

SB 释放的 Queue slot 用于更大 base tile；只有它把每核两 tile 降成一 tile且不牺牲并行核数时才覆盖 DB。不能仅因 SB tile 更大就选择它。

## 平台与实现边界

| 项目 | 950 事实 | 边界 |
|---|---|---|
| 构建 | `NPU_ARCH=ascend950` 映射 `dav-3510`；算子经 `register_direct_launch_op` 接入 kernel object 与 `_C` | A2/A3 枚举存在，但本卡未闭合其 Route/资源 |
| 任务类型 | active entries 均设置 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 无可达 AIC/Cube/MIX_CV |
| Host 算术 | shape 与 tile 乘积缺少显式 overflow check | 安全复用应补 checked arithmetic 与 numel 交叉检查 |
| compact UB | 固定 135168B | selector 未显式检查 `ubSize>=135168` |
| POD | 80B/8B 为声明顺序自然对齐推导 | 缺编译态 static_assert 证据 |
| 精度 | 普通 ExpSub 与 compact Muls+Exp 都保持 FP32 中间值 | 不声称两序列 bitwise 一致 |
| 错误传播 | 同 stream 单 Kernel 提交 | 异步 Device 状态未转为 directCall 非零返回 |
| 来源范围 | 官方 HTTP source 支撑语义、950 注册和复用骨架；ExpSub/compact 是 Direct 转换实现事实 | 本卡只完成源码闭包，不外推编译、设备数值或性能结论 |

## 源码闭包索引

| 链路 | 关键符号 | 支撑内容 |
|---|---|---|
| `cann_bench/__init__.py` | `swi_glu` | Python→dispatcher |
| `op_plugin/swi_glu_plugin.cpp` | `CheckAndFlatten`、`swi_glu_npu` | schema、校验、flatten、output、dtype、stream |
| `op_kernel/swi_glu_launch.h` | `launch_swi_glu_*` | C ABI |
| `op_kernel/swi_glu_tilingdata.h` | `SwiGluTilingData` | 80B by-value POD |
| `op_kernel/swi_glu_kernel.cpp` | `GluSingleTilingCalculator`、launchers、entries | DB/SB selector、blockDim、active/dormant launch |
| `op_kernel/arch35/glu_tiling_kernel.hpp` | `SwigluSingleTilingKernel` | tile、offset、四区 copy 参数 |
| `op_kernel/arch35/swi_glu_single.hpp` | `SwigluSingle` | GM/UB copy、aligned/nonaligned router |
| `op_kernel/swi_glu_expsub.h` | `SwigluExpSubVector`、`SwiGluExpSubVF` | Queue template、Reg VF、精度/mask |
| `op_kernel/swi_glu_narrow_bf16.h` | `SwiGluNarrowBf16` | compact 分核、Gather、UB、barrier、tail |
| 算子 `CMakeLists.txt` 与顶层 `CMakeLists.txt` | `register_direct_launch_op`、`NPU_ARCH` | build wiring 与 950 arch |
| 官方 `README.md`、`swi_glu_def.cpp`、950 binary config | 产品与 dtype 注册 | 官方语义/950 映射 |
| 官方 `swi_glu_tiling.cpp`、arch35 搬运头 | tiling 与 `SwigluSingle` | 转换实现复用关系 |