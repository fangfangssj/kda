---
type: Guide
title: Tensor API 布局描述与 Copy 特化的边界
description: 将 Layout 可表示的形状和步长与具体 Copy 实现实际消费的字段分开，保持地址单位、存储跨度和同步合同一致。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/tensor-api-reference.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/mmad_compute_key_features/unit_flag.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute_tensor_api/cube_store_key_features/relu_pre.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute_tensor_api/cube_store_key_features/f32_channel_split.md
  role: constraint
created_at: '2026-09-10T14:27:23Z'
updated_at: '2026-09-14T12:42:50Z'
platforms: []
aliases:
- Tensor API routing
- CopyGM2L1Trait
- MmadTrait
- AscendC::Te::MmadParams
---

# Tensor API 布局描述与 Copy 特化的边界

Tensor API 把存储位置、元素类型或布局 Trait、Shape、Stride 与 Pattern 组合为张量视图。Copy 的具体实现还由源和目的位置、Pattern、架构及操作 Trait 等条件选择。因此，一个 Layout 能记录某个 stride，不代表所选 Copy 特化会按该 stride 搬运。

对非连续 slice、预打包数据或新增 dtype，应从实际 Copy 调用核对具体特化消费哪些 shape、offset 和 stride 字段。如果某轴没有被实现读取，不能仅通过改变 Layout 字段宣称已经支持该轴的非连续访问。

片上硬件地址构造与逻辑元素寻址可能采用不同单位。来源所述 `MakeMemPtr<L1,T>(offset)` 使用字节偏移，元素数必须先按载体转换；分形、打包低比特数据与 Scale 的布局 Trait 又有各自粒度。不能把 C++ 载体的 `sizeof` 直接当成一个逻辑低比特元素的宽度。

通过预打包适配 Copy 时，应同时更新物理地址、Pattern、shape、stride、实际存储跨度和所有下游消费者。中间文件排列正确，但消费者仍使用旧 stride，整个调用链仍然错误。Scale 的逻辑数学视图与原始文件字节顺序也必须分开；用于 Golden 的 reshape/permute 不自动成为设备输入格式。

Tensor API 的调用与资源生命周期应分别核对。视图构造不证明异步搬运已完成；调用者或上层 Blaze 组件仍需提供生产完成和复用前的同步。不能从编译通过推断目标架构实际执行了期望的 Copy。

具体 API 名称、Routing 支持表与空实现行为依赖所用版本，本卡不继承来源对 master 的完整支持断言。官方 [MakeFrameLayout](../../apis/simd_api/tensor_api/layout/make_frame_layout.md) 卡也将其标为试验接口。验证宜使用可区分每行、每 tile 和每个 Scale 块的数据，覆盖非零偏移、非连续 stride、尾块及首块之外的访问。

## 固定来源的构造与派发合同

以下补充用于核对一次 Tensor 构造、Copy 与 Mmad 调用是否采用一致的物理合同。来源自述为 DAV-3510、CANN 9.0.0-beta.2、ops-tensor master；固定的是 reference 文件版本，上游 master 没有固定提交，不能把表中组合当成所有当前版本的支持全集。接口位于 AscendC::Te，公开包含入口为 `tensor_api/tensor.h`；直接包含内部实现头可能触发 internal header 警告。

派发键包含 `(dstLocation, srcLocation, archVersion, dstPattern, srcPattern[, copyMode])`。来源的 `ArchVersion::V3510=3510`、`V2201=2201` 和由 `__NPU_ARCH__` 得到的 CURRENT_ARCH_VERSION 只是架构标识。该快照报告非 3510 路线可能落到 `*Ignore` 空实现；枚举存在、编译通过均不能证明实际搬运，更不能据此认定所有其他版本都没有实现。

| Location 标签 | 来源中的带地址空间指针 |
|---|---|
| GM / UB | `__gm__ T*` / `__ubuf__ T*` |
| L1 | `__cbuf__ T*` |
| L0A / L0B / L0C | `__ca__ T*` / `__cb__ T*` / `__cc__ T*` |
| BIAS / FIXBUF / SSBUF | `__biasbuf__ T*` / `__fbuf__ T*` / `__ssbuf__ T*` |

INVALID 是未提供该位置的标签，不是可分配内存。BIAS 对应 BT；FIXBUF 是 Fixpipe 参数所在缓冲，不能与输出 UB 混用。容量按实际平台可用资源核算，来源 L1=512KB、L0A/B=64KB、L0C=256KB、BT=4KB 只属于该快照，不能变成可跨平台硬编码的分配上限。

构造链中的接口形式如下，省去重复函数修饰符以突出模板和参数；具体约束仍以匹配版本头文件为准：

```cpp
template <typename Hardware, typename TraitOrType, typename... Args>
auto MakeMemPtr(Args... args);
template <typename Hardware, typename... Args>
auto MakeMemPtr(Args... args);
template <typename Iterator>
auto MakeMemPtr(const Iterator& iter);

template <typename Pattern, typename TraitType = LayoutTraitDefault<>, typename... Args>
decltype(auto) MakeFrameLayout(const Args&... args);
template <typename T, typename U> auto MakeLayout(const T& shape, const U& stride);
template <typename T> auto MakeLayout(const T& shape);
template <typename... Ts> auto MakeShape(const Ts&... t);
template <typename... Ts> auto MakeStride(const Ts&... t);
template <typename... Ts> auto MakeCoord(const Ts&... t);
template <typename... Ts> auto MakeTile(const Ts&... t);
template <typename Pattern, typename Trait = LayoutTraitDefault<>> struct FrameLayoutFormat;

template <typename Iterator, typename... Args>
auto MakeTensor(const Iterator& iter, const Args&... args);
```

MakeMemPtr 的三类用途是按物理位置/类型从地址或字节偏移构造、指定位置从已有指针转换、从 Iterator 推导；已有 [MakeMemPtr](../../apis/simd_api/tensor_api/tensor/make_mem_ptr.md) 卡提供试验接口边界。GM 的输入是地址；片上 MakeMemPtr<Location::L1,T>(offset) 的 offset 为字节偏移，不能把元素索引直接用作字节地址。

MakeFrameLayout 的第二模板参数可使用布局 Trait，来源也展示 `Std::Int<C0>` 显式 C0 写法。这是两种参数用法，不应把仅改变模板形参名字的示意写成两个可并存的无约束 C++ 重载。MakeTensor 在 Iterator 之后接受 Layout、(shape,stride) 或 shape。指针的元素类型和 Layout 的 Trait 应分别核对；[MakePatternLayout](../../apis/simd_api/tensor_api/layout/make_pattern_layout.md) 中 Trait 的 T 不是另一次 Tensor 元素类型存储。

| LocalTensor 操作 | 该来源中的用途及边界 |
|---|---|
| Slice(coord, shape) | 切子张量并裁剪逻辑 shape；不能单凭裁剪证明 DMA 不越界 |
| operator()(coord) 或 operator()(c0,c1,...) | 构造子 Tensor 视图 |
| `operator[](coord)` | 访问元素引用 |
| Compose(layouts...) | 用新 Layout 解释相同内存，不搬运数据 |
| Layout() / Engine() | 取得布局或底层指针引擎 |
| SetL2CacheHint(CacheMode) | 配置缓存访问提示，不建立完成同步 |

### Pattern 与 C0 的一致性

来源列出 13 个 Pattern 名称：NDExtLayoutPtn、DNExtLayoutPtn、NDLayoutPtn、DNLayoutPtn、NZLayoutPtn、ZNLayoutPtn、ZZLayoutPtn、NNLayoutPtn、ScaleANDLayoutPtn、ScaleADNLayoutPtn、ScaleBNDLayoutPtn、ScaleBDNLayoutPtn、DefaultPtn。Ext 是对应外层嵌套 shape 的标签，不能仅按 ND/DN 字样互换；GM 也可以承载已按 NZ/ZN 编码的输入。

| Trait | 来源中的默认参数 | 布局用途 |
|---|---|---|
| LayoutTraitDefault | T=uint16_t，C0=32/sizeof(T) | half/bfloat16 为 16、float 为 8、8bit 数据为 32 |
| LayoutTraitScale | T=fp8_e8m0_t，C0=2/sizeof(T) | E8M0 Scale 的两个槽 |
| LayoutTraitFP4 | T=fp4x2_e2m1_t，C0=64/sizeof(T) | 打包 FP4 的逻辑分组 |

来源将 fp4x2 的 sizeof 写成 0.5，不是合法的 C++ sizeof 值；应区分 4bit 逻辑元素与承载两个值的字节类型。默认 Trait 的 T=uint16_t 也不代表 GM 的真实数据必为 16bit。对 int8/FP8 预打包入口，GM 与 L1 两侧都要采用真实 dtype 的 C0，不能只修 L1。

来源的 FRACTAL_FIXED=16、BLOCK_CUBE=16，以及该快照 L0C 的 16 边长，不能用普通 32/sizeof(float)=8 替代。NZ 的物理形状、容量和轴变换复用[逻辑形状与物理存储](matmul_logical_shape_and_physical_storage.md)；合法基形与逻辑尾块复用[基形与当前范围](base_shapes_and_runtime_extents.md)。

### Copy/Mmad 对象与参数

```cpp
template <typename... Args> auto MakeCopy(const Args&... traits);
template <typename T, typename... Params>
void Copy(const CopyAtom<T>& atomCopy, const Params&... params);
template <typename Tp, const Tp& traits, typename T, typename... Params>
void Copy(const CopyAtom<T>& atomCopy, const Params&... params);

template <typename... Args> auto MakeMmad(const Args&... traits);
template <typename T, typename... P>
void Mmad(const MmadAtom<T>& atom, const P&... params);
template <typename Tp, const Tp& traits, typename T, typename... P>
void Mmad(const MmadAtom<T>& atom, const P&... params);
```

MakeCopy/MakeMmad 构造调用对象；来源通过 with(trait/params) 注入配置，也列出 const Tp& 编译期 trait 形式。Copy 的数据参数为 `(atom, dst, src[, fixpipeParams])`，Mmad 为 `(atom, dstL0C, A_L0A, B_L0B[, bias])`。Copy 目的在前，与 DataCopy 目的在前的用法一致，来源开头称两者“相反”是前后矛盾，不能采用。

| Copy Atom | 搬运方向 |
|---|---|
| CopyGM2L1 | GM→L1 |
| CopyL12L0A / CopyL12L0B | L1→L0A / L0B |
| CopyL12BT / CopyL12FB / CopyL12UB | L1→BIAS / FIXBUF / UB |
| CopyL0C2GM / CopyL0C2GMWith | L0C→GM，With 为带模式形式 |
| CopyL0C2UB / CopyL0C2UBWith | L0C→UB，With 为带模式形式 |
| CopyGM2UB / CopyUB2GM / CopyUB2L1 | GM→UB / UB→GM / UB→L1 |

字段默认值决定没有显式配置时的行为，不能由名称猜测：

| 类型 | 来源明确的字段与默认值 |
|---|---|
| CopyGM2L1Trait、CopyL12BTTrait、CopyL12FBTrait、CopyL12UBTrait、DataCopyTrait | 该快照为空占位 |
| LoadDataTrait；CopyL12L0ATrait / CopyL12L0BTrait 为其别名 | bool transposed=false |
| CopyL0C2GMTrait | RoundMode roundMode=DEFAULT；bool enableRelu=false；bool enableChannelSplit=false |
| CopyL0C2UBTrait | 上一行字段，加 DualDstMode dualDstCtl=DUAL_DST_DISABLE |
| FixpipeTrait | RoundMode roundMode=DEFAULT；bool enableRelu=false；bool enableChannelSplit=false；DualDstMode dualDstCtl=DUAL_DST_DISABLE |
| FixpipeParams | uint8_t unitFlag=0 |
| MmadTrait | int32_t fmOffset=0；bool kDirectionAlign=false；bool cmatrixSource=false；bool disableGemv=true；MmadType **mmadType**=NORMAL |
| MmadParams | uint16_t m=0、n=0、k=0；uint8_t unitFlag=0；bool cmatrixInitVal=false |

MmadTrait 的准确成员名是 `mmadType`。fmOffset 为来源所述 L0A 偏移配置，资料没有给出可推广的单位；kDirectionAlign 控制 K 对齐，cmatrixSource 选择 C 来源（该表为 false=L0C、true=BIAS），disableGemv=true 表示默认关闭 GEMV。该 Te::MmadParams 的 m/n/k 与其他版本的 MmadTiling 或基础 API mSize/nSize/kSize 不能按相似名字互换；构造时按真实字段顺序及类型范围填写。

| 枚举 | 该来源的取值 |
|---|---|
| CacheMode : uint8_t | CACHE_MODE_NORMAL=0，CACHE_MODE_DISABLE=4，随后 CACHE_MODE_LAST、CACHE_MODE_PERSISTENT |
| RoundMode : uint8_t | DEFAULT=0，HYBRID |
| DualDstMode : uint8_t | DUAL_DST_DISABLE=0，DUAL_DST_SPLIT_M，DUAL_DST_SPLIT_N |
| MmadType : uint8_t | NORMAL=0，MX=1 |

这里的 Te::RoundMode 与 Cast 的 RoundMode 不同。随路 [Relu](../../apis/simd_api/tensor_api/matrix_compute/cube_store_key_features/relu_pre.md) 的默认关闭、[F32 ChannelSplit](../../apis/simd_api/tensor_api/matrix_compute/cube_store_key_features/f32_channel_split.md) 的格式和 unitFlag 互斥条件已有独立卡；字段存在不等于任意组合均支持。双目的输出模式影响接收尺寸和工作归属，见 [Fixpipe UB 路由](../matmul/fixpipe_ub_mirror_and_slot_reuse_contract.md)。

### GM→L1 的具体路由

表中 dst/src 为 Pattern；除 Default 外，短名均加 LayoutPtn 后缀，例如 NDExt 表示 NDExtLayoutPtn。下列是来源明确列出的组合；dtype、对齐、stride 和容量仍由具体实现约束，不继承来源“任意 dtype”的无条件说法。

| dst | src | CopyGM2L1 选中的实现 |
|---|---|---|
| NDExt | NDExt | CopyGmToCbufAlignV2ND |
| NZ | NDExt | CopyGmToCbufMultiND2Nz |
| ZN | NDExt | CopyGmToCbufMultiND2Zn |
| NZ | DNExt | CopyGmToCbufMultiDN2Nz |
| ZN | DNExt | CopyGmToCbufMultiDN2Zn |
| NZ | NZ | CopyGmToCbufAlignV2NZ |
| ZN | ZN | CopyGmToCbufAlignV2ZN |
| ZZ | ScaleAND | CopyGmToCbufScaleAND2Zz |
| ZZ | ScaleADN | CopyGmToCbufScaleADN2Zz |
| ZZ | ZZ | CopyGmToCbufScaleAZz2Zz |
| NN | ScaleBND | CopyGmToCbufScaleBND2Nn |
| NN | ScaleBDN | CopyGmToCbufScaleBDN2Nn |
| NN | NN | CopyGmToCbufScaleBNn2Nn |

比如 L1=ZZ、GM=NDExt 不在这张快照表中，不能因为二者均为合法 Pattern 就推断可直接 Copy。预编码 NZ/ZN 输入还需要匹配 TagToTrans<NZLayoutPtn>/TagToTrans<ZNLayoutPtn> 特化；来源分别为 value=false/true，不能只更换 launcher 的 trans 标志而保留旧地址、Pattern 或文件排列。

MX ScaleB 在该快照的 E8M0、C0=2、非转置 ND 路线中，数学视图为 [scaleK,N]，raw 容器为 [ceil(scaleK/2),N,2]。令 scaleIndex=2*group+slot，则 byte offset=`group*(N*2)+n*2+slot`。Golden 的 reshape/permute 是派生数学视图，不能直接物化为新的设备输入顺序。Batch 必须另有真实 batch stride，其他 transpose、dtype、Pattern 或 C0 要重新推导；四种转置的编码背景见 [MX Scale 与 FP4 打包](../layouts/mx_block_scales_and_fp4_byte_packing.md)。

### L1→L0、其他搬运和 Mmad 路由

CopyMode 还受 transposed 与 B8/B4 类型分类影响。下表的 `_COORD` 表示来源提及的坐标变体族；源表未给出完整拼写的地方不补造枚举或类名。

| 目的 | dst/src Pattern | CopyMode 与条件 | 实现 |
|---|---|---|---|
| L0A | NZ / NZ | NORMAL 或 NORMAL_COORD；非 B8/B4、transposed=false | LoadDataL12L0ANZ2NZ3510；来源另列 WithCoord 变体 |
| L0A | NZ / ZN | TRANS 或 TRANS_COORD；transposed=true | LoadDataL12L0AZN2NZ3510 |
| L0A | NZ / ZN | TRANS_B8B4 及坐标变体；转置 int8/FP8/FP4 | LoadDataL12L0AZN2NZB8B43510 |
| L0A | ZZ / ZZ | NORMAL 及坐标变体；MX ScaleA | LoadDataL12L0MxScaleA3510 |
| L0B | ZN / ZN | NORMAL 或 NORMAL_COORD；transposed=false | LoadDataL12L0BZN2ZN3510 |
| L0B | ZN / NZ | TRANS 及坐标变体；transposed=true | LoadDataL12L0BNZ2ZN3510 |
| L0B | ZN / NZ | TRANS_B8B4 及坐标变体；转置 B4/B8 | LoadDataL12L0BNZ2ZNB8B43510 |
| L0B | NN / NN | NORMAL 及坐标变体；MX ScaleB | LoadDataL12L0MxScaleB3510 |

LoadDataTrait{transposed=true} 选择 TRANS 系列，但能否消费给定坐标和 stride 仍须检查被选中的具体实现，不能只看最终成功实例化的类型。

| 方向 | 来源中的实现 |
|---|---|
| L1→BT / FIXBUF / UB | DataCopyL12BT3510 / DataCopyL12FB3510 / DataCopyL12UB3510 |
| L0C→GM | DataCopyL0C2GM3510，另有 VectorQuant 量化变体 |
| L0C→UB | DataCopyL0C2UB3510，另有 VectorQuant 量化变体 |

源表在这些行的 Pattern 栏没有给出具体值，这不表示所有 Pattern 都合法。量化变体完整类型名也应以匹配头文件为准。

| dstLoc / fmLoc / filterLoc | biasLoc | Mmad 实现 |
|---|---|---|
| L0C / L0A / L0B | INVALID | Mmad3510 |
| L0C / L0A / L0B | L0C | MmadWithBias3510 |
| L0C / L0A / L0B | BIAS | MmadWithBias3510 |

MmadTrait::mmadType=NORMAL/MX 分别选择 MmadInstr/MmadMxInstr。BIAS 参数位置与 cmatrixSource、输出初值合同一起核对，不能只添加一个 bias 指针便推导所有 MX/普通组合可用。

### 累加、同步与诊断边界

cmatrixInitVal 控制一段计算的初值。该来源普通独立输出 tile 的首个 K 块置 true，后续 K 块置 false 继续累加；切换输出 tile 时重新建立初值。它给出的双层循环表达式 `(iter0==0 && iter1==0)` 只适用于 iter0/iter1 确为该输出的 K-L1/K-L0 索引。独立 StreamK 分段各自初始化 partial，再由对应归并路径合并；不能把“每段初始化”误解成连续累加中每次清零。

unitFlag 是计算与搬出的细粒度同步合同，来源 MmadParams/FixpipeParams 默认均为 0，FINAL_ACCUMULATION=3 用在对应终结阶段；它不自动建立任意 L0C 双缓冲，也不替代其余 SetFlag/WaitFlag。详见 [Tensor API UnitFlag](tensor_api_unit_flag_reference_contract.md)。[本地 SplitK 模板](../shmem/blaze_local_splitk_sentinels_and_atomic_output.md) 有不同初始化哨兵，不能把普通 tile 公式无条件套入。

对尾块，Slice 的逻辑裁剪必须与合法 Copy 基形、物理容量和下游 MMAD 读取范围同时满足。对双缓冲，若 L0C_SIZE 表示字节数，半区字节偏移为 L0C_SIZE/2，不再除 sizeof(T)；若 caller 改用元素计数，所有消费者单位需一起改变。NZ 的切片顺序依赖实际 specialization 消费的轴，来源“Kernel 先 M/N、Block 再 K”只是其路线，不能作为所有 stride/版本的统一规则。

Unsupported layout pattern 通常指向标签注册；data format not supported 指向具体路由组合。聚合初始化字段数量错误应核对真实 Params 定义；non-const lvalue reference 绑定错误应核对引用类型及值类别，单纯乘 sizeof(T) 不会修复该编译错误。全零结果也不能唯一归因于 unitFlag 或初值，须同时确认输入、路由确实搬运、累加过程和结果搬出。

以上保存来源的可核对接口与组合，没有把其省略上下文的整段 Matmul 示例当作已验证程序。使用时绑定真实上游 revision，选取可区分 tile/行/Scale 的输入，核对字节偏移、各级 Pattern、转置、首末 K 状态和生产消费顺序。
