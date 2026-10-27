---
type: "Runbook"
title: "TCubeTiling 双定义：kernel 侧禁宿主类型与头"
description: "共享 TilingData 里 kernel 侧只能用 AscendC::tiling::TCubeTiling 纯结构体（kernel_tiling/kernel_tiling.h）：include 宿主 adv_api/matmul/matmul_tilingdata.h 会拖入 register/tilingdata_base.h→graph/types.h 连片 expected identifier，直接写 optiling::TCubeTiling 则报 undeclared identifier；宿主侧用框架类计算后 SaveToBuffer/逐字段序列化。"
tags: ["task.implementation", "topic.compilation", "topic.tiling"]
status: stable
sources: []
created_at: '2026-09-17T04:05:46Z'
updated_at: '2026-09-17T04:05:46Z'
platforms: ["950"]
---

# TCubeTiling 双定义：kernel 侧禁宿主类型与头

## 现象

`TCubeTiling` 在 CANN 里有两个定义：宿主侧 `optiling::TCubeTiling`（`BEGIN_TILING_DATA_DEF` 宏生成、继承 `TilingDef` 的框架类）与 kernel 侧 `AscendC::tiling::TCubeTiling`（`kernel_tiling.h` 的纯结构体）。把宿主那份的类型或头泄漏进 host/kernel 共享的 TilingData 头后，kernel 编译以两种形态失败（三条独立轨迹实测）。

**形态一：共享头 include 宿主 tilingdata 头 → graph 头枚举连片报错**。MoE 分组矩阵乘算子（Ascend950PR / arch35 / CANN 9.1.0）与 GRU 算子（CANN 9.2.0）的 op_kernel 共享 TilingData 头照 host 侧写法包含宿主 tiling 头，kernel 编译（ccec）阶段报错行号全部落在 CANN 头文件里：

```text
<ASCEND_HOME>/asc/../../include/graph/types.h:82:3: error: expected identifier
<ASCEND_HOME>/asc/../../include/graph/types.h:83:3: error: expected identifier
... (types.h:82-100 连续 19 行)
fatal error: too many errors emitted, stopping now [-ferror-limit=]
```

`grep -B5` 展开完整 include 链：

```text
In file included from op_kernel/arch35/bench_no_harness_grouped_matmul.h:36:
In file included from op_kernel/arch35/bench_no_harness_grouped_matmul_tiling_data.h:19:
In file included from <ASCEND_HOME>/asc/include/adv_api/matmul/matmul_tilingdata.h:18:
In file included from <ASCEND_HOME>/asc/../../include/register/tilingdata_base.h:19:
In file included from <ASCEND_HOME>/asc/../../include/graph/ascend_string.h:17:
<ASCEND_HOME>/asc/../../include/graph/types.h:82:3: error: expected identifier
```

出错的 types.h 内容是 `enum DataType { DT_FLOAT = ::C_DT_FLOAT, ... }` 逐行 expected identifier（CANN 9.2.0 轨迹观测为 82..101 连续 20 行）。

**形态二：共享头直接写 optiling:: 类型 → 命名空间/宏依赖未定义**。Conv3d 反传 filter 算子（Ascend950PR / arch35 / CANN 9.1.0）为 `REGIST_MATMUL_OBJ` 在共享 TilingData 头里直接写 `optiling::TCubeTiling cubeTiling;` 字段、kernel 入口 include `lib/matmul_intf.h`，`bash build.sh --soc=ascend950 -j8` 只见 FATBIN 链接器报错：

```text
ld.lld: error: cannot open .../kernel_meta_<hash>/BenchNoHarnessConv3dBackpropFilter_<hash>_mix_aic_27.o: No such file or directory
ld.lld: error: cannot open .../BenchNoHarnessConv3dBackpropFilter_<hash>_mix_aic_283.o: No such file or directory
RuntimeError: Kernel Compilation Error: OpType BenchNoHarnessConv3dBackpropFilter Kernel File .../op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp!
```

用 `bash build.sh --soc=ascend950 -j1 2>&1 | grep -i "error"` 串行重跑后，才暴露真实的前端编译错误（共 7 个）：

```text
op_kernel/arch35/bench_no_harness_conv_3d_backprop_filter_tiling_data.h:79:5: error: use of undeclared identifier 'optiling'
op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp:38:5: error: use of undeclared identifier 'ASCubeObjConfig'
op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp:38:5: error: static assertion expression is not an integral constant expression
op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp:38:5: error: constexpr if condition is not a constant expression
op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp:38:5: error: default initialization of an object of const type 'const int8_t'
```

即：TilingData 头文件里的 `optiling::` 命名空间在 kernel 编译上下文不存在；kernel 入口 `REGIST_MATMUL_OBJ` 宏展开所依赖的 `ASCubeObjConfig` 类型同样未定义（`lib/matmul_intf.h` 这条 include 路径在 kernel 构建的 include 集合里解析不到，宏与依赖类型实际都没有被引入）。

## 适用场景

- **工程形态**：算子仓（ops-registry-invoke）自定义算子开发，`op_kernel/arch35/*_tiling_data.h` 同时被 host tiling 与 kernel 编译消费，需要为 Matmul 高阶 API / `REGIST_MATMUL_OBJ` 提供 `TCubeTiling` 字段。
- **API 路径**：host 侧 `matmul_tiling::MatmulApiTiling` / `matmul_tiling::MultiCoreMatmulTiling`（`asc/include/adv_api/matmul/`、`asc/include/tiling/matrix/`）、`optiling::TCubeTiling`（`matmul_tilingdata.h`，`BEGIN_TILING_DATA_DEF` 宏族，include 链经 `register/tilingdata_base.h` 绑定 GE 头）；kernel 侧 `AscendC::tiling::TCubeTiling`（`ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h` 纯结构体，头文件尾部 `using` 引入全局命名空间，部分构建框架另由 `impl/adv_api/cmake/scripts/gen_kernel_tiling_data_def.py` 构建期生成注入）；`REGIST_MATMUL_OBJ`（`asc/include/adv_api/matmul_intf.h`，宏展开为 `REGIST_CUBE_OBJ` 并使用 `ASCubeObjConfig`）。
- **验证过的平台与 CANN 版本**：Ascend 950 / 950PR（arch35 / DAV_3510），CANN 9.1.0（MoE 分组矩阵乘、Conv3d 反传 filter 两条轨迹）与 CANN 9.2.0（GRU 轨迹）。
- **适用粒度**：`operator_family`——凡在共享 TilingData 中嵌入 cube tiling 并注册 Matmul 对象的自定义算子均适用。
- **路线优先级**：`recommended_fastpath`（对"kernel 侧 TilingData 需要 cubeTiling 字段"这一需求，这是能通过编译并保住 tiling 框架能力的标准做法）。
- **明确不适用 / 未验证场景**：纯 Vector 算子（无 Matmul tiling）不涉及；kernel 侧完全不消费 TCubeTiling 字段的工程不受影响；`ASCENDC_CPU_DEBUG` 下的 UT 编译路径由 UT CMake 自带的 `REGIST_MATMUL_OBJ` stub 覆盖，与 NPU kernel 编译路径不同；其他 CANN 版本中两个 TCubeTiling 的字段布局是否仍逐字段一致未验证。

## 根因

`TCubeTiling` 在 CANN 里是两个不同的东西——宿主侧 `optiling::TCubeTiling` 是 `BEGIN_TILING_DATA_DEF` 宏生成的、继承 `TilingDef`（带虚函数、data_ptr_ 等框架成员）的框架类，其 include 链 `register/tilingdata_base.h → graph/ascend_string.h → graph/types.h` 依赖 GE 编译上下文；kernel 侧 `AscendC::tiling::TCubeTiling` 是 `kernel_tiling.h` 里 `#pragma pack(push, 8)` 的 50 个 `int32_t`（200 字节）纯结构体，且该头文件尾部有 `using AscendC::tiling::TCubeTiling;` 把它引入全局命名空间。

kernel 编译单元（ccec/bisheng，include 基路径为 ascendc 树）没有 GE 注册体系上下文：include 宿主 tilingdata 头则 `register/tilingdata_base.h → graph/types.h` 的 `DT_*` 枚举逐行 `expected identifier`（同签名的宏冲突机制证据——graph/types.h 的 DT_* 枚举成员与 kernel 侧 kernel_type.h 的 DT_* `#define` 同名冲突——见 [共享 tiling.h 宏范式 DT 宏冲突卡](tiling_h_macro_dt_conflict_plain_pod.md)）；引用 `optiling::` 类型则该命名空间不存在直接 undeclared identifier，连带 `REGIST_MATMUL_OBJ` 宏依赖的 `ASCubeObjConfig` 一并未定义。本质上：**kernel 侧代码不能 include/使用宿主侧的 tilingdata/GE 注册头与类型**，两套 TCubeTiling 是不同 include 树下的镜像，共享 TilingData 结构时 kernel 侧必须用 kernel 树的那个，宿主侧框架类只做计算器、结果序列化进纯结构体字段。

## 判别方法

**触发特征**：

1. kernel 编译阶段 `graph/types.h` 连续多行 `expected identifier` + `too many errors`，而 host 侧同一批头编译正常——查共享头的 include 链是否经 `adv_api/matmul/matmul_tilingdata.h → register/tilingdata_base.h → graph/*`（形态一）。
2. kernel 编译报 `use of undeclared identifier 'optiling'` / `'ASCubeObjConfig'`，且 `-j8` 并行构建下只看到 FATBIN `ld.lld: cannot open *_mix_aic_*.o`——先 `-j1` 重跑拿真实前端错误（形态二；`_mix_aiv` 变体的链接噪声判别见 [ld.lld cannot open _mix_aiv 卡](ld_lld_cannot_open_mix_aiv.md)）。

**可执行探针**（三条轨迹逐条跑通，路径按需替换）：

```bash
# 1) kernel 侧 matmul 接口只依赖前向 TCubeTiling 类型（不需要宿主完整定义）：
grep -rn "TCubeTiling" <repo>/asc-devkit/include/adv_api/matmul/matmul_intf.h | head -5
grep -rn "TCubeTiling\|tilingdata" <repo>/asc-devkit/impl/adv_api/detail/matmul/param/matmul_shape_tiling.h | head -5
# → matmul_shape_tiling.h:30: __aicore__ inline void SetTiling(const TCubeTiling* __restrict tiling)

# 2) kernel 侧 TCubeTiling 由构建期脚本生成（不是手写宿主定义）：
grep -rn "TCubeTiling" <repo>/asc-devkit/impl/adv_api/cmake/ | head -5
# → impl/adv_api/cmake/scripts/gen_kernel_tiling_data_def.py:33: "TCubeTiling",

# 3) 宿主侧 TCubeTiling 定义来源（会被 tilingdata_base.h 拖进 graph 头）：
grep -rn "BEGIN_TILING_DATA_DEF.*TCubeTiling" <repo>/asc-devkit/include/ <CANN>/x86_64-linux/asc/include/ | head -5
# → asc-devkit/include/adv_api/matmul/matmul_tilingdata.h:21: BEGIN_TILING_DATA_DEF(TCubeTiling)

# 4) -j1 重跑暴露被 FATBIN 链接阶段掩盖的前端错误：
bash build.sh --soc=ascend950 -j1 2>&1 | grep -i "error"
```

## 解决配方

三处配套修改（形态二轨迹 verbatim 闭环；形态一轨迹以同型 header 替换闭环）：

**1. 共享 TilingData（kernel 侧共享头）使用全局命名空间的纯结构体 `TCubeTiling`**，并注释清楚双类型约定。字段声明处 before/after：

```cpp
// before：宿主框架类直接进 kernel 侧结构体 —— kernel 编译报 optiling 未声明
    // —— TCubeTiling（供 REGIST_MATMUL_OBJ，GM/ND 配置）——
    // 仅用于使 REGIST_MATMUL_OBJ 初始化成功（taskRation="1:2" → workspace 可写）。
    // 不调用任何 Matmul compute，实际计算由 Vector Contraction 完成。
    optiling::TCubeTiling cubeTiling;
```

```cpp
// after：kernel 侧纯结构体；宿主侧用 optiling::TCubeTiling 计算后手动拷贝/序列化进来
    // —— TCubeTiling（供 REGIST_MATMUL_OBJ，GM/ND 配置）——
    // 仅用于使 REGIST_MATMUL_OBJ 初始化成功（taskRation="1:2" → workspace 可写）。
    // 不调用任何 Matmul compute，实际计算由 Vector Contraction 完成。
    // 注意：使用全局命名空间的 TCubeTiling（= AscendC::tiling::TCubeTiling，kernel 侧纯结构体）。
    // 宿主侧用 optiling::TCubeTiling（框架类）计算后，手动拷贝字段到此。
    TCubeTiling cubeTiling;
```

头文件 include 的 before/after（形态一轨迹，共享头不再含宿主 tilingdata 头）：

```cpp
// before：宿主 GE 注册头进共享头 —— kernel 编译报 graph/types.h 枚举连片错
#include "adv_api/matmul/matmul_tilingdata.h"

struct BenchNoHarnessGroupedMatmulTilingData {
    // ===== Matmul API Tiling =====
    optiling::TCubeTiling mmTiling;   // Matmul 高阶 API Tiling（M=max_m_i, N, K）
    ...
```

```cpp
// after：kernel 树头 + kernel 命名空间类型（host 编译也能解析该 plain struct 镜像）
#include "kernel_tiling/kernel_tiling.h"

struct BenchNoHarnessGroupedMatmulTilingData {
    // ===== Matmul API Tiling =====
    AscendC::tiling::TCubeTiling mmTiling;   // Matmul 高阶 API Tiling（M=max_m_i, N, K）
    ...
```

（中间态曾先改成 `#include "adv_api/kernel_tiling.h"` 仍不通，最终定位到 kernel 树的规范路径 `kernel_tiling/kernel_tiling.h`，位于 `ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h`。）

**2. 宿主侧 tiling 的两种验证过的写法**：

写法 a（形态一轨迹）：host tiling.cpp 直接使用 `AscendC::tiling::TCubeTiling`（plain struct 对 host 也可编译），按字段名赋值：

```bash
cd operators/<op>/op_host/arch35 && \
  sed -i 's/optiling::TCubeTiling/AscendC::tiling::TCubeTiling/g' <op>_tiling.cpp
```

写法 b（形态二轨迹，宿主需要框架类计算时）：host tiling 用 `optiling::TCubeTiling` 框架类承接 `MultiCoreMatmulTiling::GetTiling`，再把字段序列化进 TilingData 的纯结构体字段。逐字段 getter 拷贝（保守、字段可见；两者字段为同一组 50 个 int32_t，类型不同不能直接赋值）：

```cpp
    // TCubeTiling（供 REGIST_MATMUL_OBJ，GM/ND 配置）
    // 从 optiling::TCubeTiling（框架类）手动拷贝到 TilingData 的 TCubeTiling（纯结构体）
    // 两者字段完全一致（50 个 int32_t），但类型不同，不能直接赋值
    tilingData->cubeTiling.usedCoreNum    = cubeTilingHost.get_usedCoreNum();
    tilingData->cubeTiling.M              = cubeTilingHost.get_M();
    tilingData->cubeTiling.N              = cubeTilingHost.get_N();
    tilingData->cubeTiling.Ka             = cubeTilingHost.get_Ka();
    tilingData->cubeTiling.Kb             = cubeTilingHost.get_Kb();
    tilingData->cubeTiling.singleCoreM    = cubeTilingHost.get_singleCoreM();
    tilingData->cubeTiling.singleCoreN    = cubeTilingHost.get_singleCoreN();
    tilingData->cubeTiling.singleCoreK    = cubeTilingHost.get_singleCoreK();
    tilingData->cubeTiling.baseM          = cubeTilingHost.get_baseM();
    tilingData->cubeTiling.baseN          = cubeTilingHost.get_baseN();
    tilingData->cubeTiling.baseK          = cubeTilingHost.get_baseK();
    tilingData->cubeTiling.depthA1        = cubeTilingHost.get_depthA1();
    tilingData->cubeTiling.depthB1        = cubeTilingHost.get_depthB1();
    tilingData->cubeTiling.stepM          = cubeTilingHost.get_stepM();
    tilingData->cubeTiling.stepN          = cubeTilingHost.get_stepN();
    tilingData->cubeTiling.isBias         = cubeTilingHost.get_isBias();
    tilingData->cubeTiling.transLength    = cubeTilingHost.get_transLength();
    tilingData->cubeTiling.iterateOrder   = cubeTilingHost.get_iterateOrder();
    tilingData->cubeTiling.shareMode      = cubeTilingHost.get_shareMode();
    tilingData->cubeTiling.shareL1Size    = cubeTilingHost.get_shareL1Size();
    tilingData->cubeTiling.shareL0CSize   = cubeTilingHost.get_shareL0CSize();
    tilingData->cubeTiling.shareUbSize    = cubeTilingHost.get_shareUbSize();
```

`SaveToBuffer` 序列化写法（推荐，不易漏字段；`SaveToBuffer` 是 `TilingDef` 的公开方法，只序列化字段数据、不含类元数据）：

```cpp
    // —— 生成 TCubeTiling，序列化到 TilingData 的 cubeTiling 字段 ——
    // 使用 optiling::TCubeTiling（框架类）计算 tiling，再通过 SaveToBuffer 序列化到 TilingData。
    // optiling::TCubeTiling 是宿主侧框架类（继承 TilingDef，有虚函数和额外成员），
    // TCubeTiling（全局命名空间）是 kernel 侧纯结构体（50 个 int32_t = 200 字节）。
    // SaveToBuffer 是 TilingDef 的公开方法，将字段数据序列化到外部 buffer（仅字段数据，不含类元数据）。
    optiling::TCubeTiling cubeTilingHost;
    int64_t mmRet = matmulTiling.GetTiling(cubeTilingHost);
    OP_CHECK_IF(mmRet != 0,
                OP_LOGE(context, "MultiCoreMatmulTiling GetTiling failed, ret=%ld", mmRet),
                return ge::GRAPH_FAILED);
    // 序列化 cubeTiling 字段数据到 TilingData（SaveToBuffer 只拷贝字段数据，不含类元数据）
    cubeTilingHost.SaveToBuffer(&tilingData->cubeTiling, sizeof(tilingData->cubeTiling));
```

配套的宿主侧 GM/ND tiling 生成（供 `REGIST_MATMUL_OBJ` 消费的 cubeTiling 计算，verbatim）：

```cpp
    // —— 生成 TCubeTiling（GM/ND 配置，供 REGIST_MATMUL_OBJ 使用）——
    // 关键：REGIST_MATMUL_OBJ(GM/ND) 能使 taskRation="1:2"，workspace MTE3 可写。
    // 不调用 SetMatmulConfigParams（用默认 CFG_NORM，不用 CFG_ENVECND2NZ）。
    // 不调用任何 Matmul compute，实际计算由 Vector Contraction 完成。
    matmul_tiling::DataType matmulDt = matmul_tiling::DataType::DT_FLOAT16;
    if (dtypeX == ge::DT_BF16) {
        matmulDt = matmul_tiling::DataType::DT_BF16;
    }

    matmul_tiling::MultiCoreMatmulTiling matmulTiling(ascendcPlatform);
    matmulTiling.SetDim(static_cast<int32_t>(coreNum));
    // GM/ND 配置（不是 TSCM/NZ）：A=grad[M,K] 不转置，B=im2col[N,K] 转置存储
    matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDt, false);
    matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDt, true);
    matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_FLOAT);
    matmulTiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    matmulTiling.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    matmulTiling.SetBufferSpace(-1, -1, -1);
```

**3. kernel 头文件 include 修正**（形态二轨迹的构建框架，kernel -I 含 `asc/include/adv_api`）：直接 `#include "matmul_intf.h"`（与 `asc/include/adv_api/matmul_intf.h` 同内容），不要写 `lib/` 前缀：

```cpp
// before
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "bench_no_harness_conv_3d_backprop_filter_tiling_data.h"
#include "bench_no_harness_conv_3d_backprop_filter_tiling_key.h"
```

```cpp
// after
#include "kernel_operator.h"
#include "matmul_intf.h"
#include "bench_no_harness_conv_3d_backprop_filter_tiling_data.h"
#include "bench_no_harness_conv_3d_backprop_filter_tiling_key.h"
```

kernel 入口（`REGIST_MATMUL_OBJ` 消费 cubeTiling 的形态，verbatim）：

```cpp
template <typename D_T_X, int GROUPS_MODE>
__global__ __aicore__ void bench_no_harness_conv3d_backprop_filter(
    GM_ADDR x, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    AscendC::SetSysWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv3dBackpropFilterTilingData, tilingData, tiling);
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    NsConv3dBpFilter::Conv3dBpFilterKernel<D_T_X, GROUPS_MODE> op;
    // GM/ND 配置的 REGIST_MATMUL_OBJ（仅用于启用 taskRation="1:2" 使 workspace 可写）
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
    op.Init(x, grad, y, user, &tilingData, &tPipe);
    op.Process();
}
```

（kernel 侧 matmul 头在另一构建框架中按 ascendc 树解析为 `lib/matmul/matmul_intf.h`，见 [kernel Matmul 头两棵树卡](kernel_matmul_intf_include_tree.md)。）

## 设计决策

- **为何选"kernel 纯结构体 + 宿主序列化"而不是条件编译**：TilingData 头被 host 与 kernel 两个编译上下文共享；用 `#ifdef` 分叉两套字段布局会让两侧 TilingData 二进制布局漂移，`GET_TILING_DATA_WITH_STRUCT` 按偏移解包即错。统一用 kernel 侧 200 字节纯结构体做"线上格式"，宿主侧框架类只做计算器，是布局单一事实源的做法。
- **逐字段 getter 拷贝 vs `SaveToBuffer`**：先实现逐字段拷贝（保守、字段可见）；后续排查 cubeTiling 数据疑似污染时改用 `SaveToBuffer`——它只序列化字段数据不含类元数据，且有 plog `InitData:TilingDef::InitData, op TCubeTiling, data size 200` 佐证序列化长度就是 200 字节。两种写法在该轨迹均编译通过、UT 通过；`SaveToBuffer` 更不易漏字段，推荐后者。
- **放弃的候选**：`optiling::TCubeTiling(void* ptr)` 指针构造函数直接绑定 `&tilingData->cubeTiling` 让 `GetTiling` 原地写入——曾在中间版本尝试，后因 plog 中 cubeTiling 首字段出现 `0x800000001b` 这类异常值怀疑指针构造路径写入布局与纯结构体不一致而放弃，改用 `SaveToBuffer`（该怀疑的机制层结论未最终定论，见遗留问题）。
- **raw buffer（定长 int32 数组）承载**：GRU 轨迹曾以 `int32_t mmGiTiling[TCUBE_TILING_BUF_SIZE]` raw buffer + `memcpy_s` 隔离两侧类型；注意宏生成的宿主类真实 `sizeof`（实测 1816B，含元数据/对齐分量）远超按字段数心算的容量，raw buffer 容量必须先跑 sizeof 探针——见 [宏生成 tiling 结构容量卡](tcube_tiling_sizeof_probe_buffer_capacity.md)；本卡主路线（直接纯结构体成员 + `SaveToBuffer` 只序列化字段数据 200B）天然规避该容量坑。
- 当前方案属 `recommended_fastpath`（编译问题层面的标准解法；注意它只解决"编译与 tiling 数据交付"，不解决 `REGIST_MATMUL_OBJ` 场景下 workspace 可写性——那是另一张卡的范围）。

## 系统信息

- 工具链：CANN 9.1.0（MoE 分组矩阵乘、Conv3d 反传 filter）与 CANN 9.2.0（GRU），bisheng 编译器 + FATBIN 打包，构建命令 `bash build.sh --soc=ascend950 -j8`（排查真实错误时用 `-j1`）。
- 目标平台：ascend950 / Ascend950PR（arch35 / DAV_3510）。
- 相关头文件（事件流中实际读过）：`asc/include/tiling/matrix/matmul_tilingdata.h`（optiling::TCubeTiling，BEGIN_TILING_DATA_DEF）、`ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h`（AscendC::tiling::TCubeTiling，尾部 `using` 引入全局）、`asc/include/adv_api/matmul_intf.h`（REGIST_MATMUL_OBJ = REGIST_CUBE_OBJ，include `kfc_register_obj.h` 提供 ASCubeObjConfig）、`asc/impl/adv_api/detail/kfc/kfc_register_obj.h`、`include/register/tilingdata_base.h`（TilingDef/SaveToBuffer）。
- 相关改动文件：`op_kernel/arch35/*_tiling_data.h`、`op_kernel/arch35/*.h`（include 修正）、`op_host/arch35/*_tiling.cpp`。

## 预防

- 在 TilingData 头文件里写任何带命名空间前缀的类型前，先确认该类型在 kernel 编译单元可见：`kernel_tiling.h` 尾部的 `using AscendC::tiling::TCubeTiling;` 意味着裸 `TCubeTiling` 在两侧都可用，而 `optiling::` 前缀只在宿主可用。
- host/kernel 共享头（tiling_data.h 一类）保持"平凡类型 only"：不 include 任何宿主侧宏定义族头（`register/*`、`graph/*`、`adv_api/matmul/matmul_tilingdata.h`）。
- 看到 kernel 编译报 graph 头枚举 expected identifier 时，第一时间检查共享头是否泄漏宿主侧 tiling/graph 头；报错全在 CANN 头内且是"枚举定义本身报错"，初看像 CANN 安装损坏或头文件循环包含，实际根因在自己的 include 选择。
- include CANN matmul 接口时按 kernel 构建实际 include 路径写（如 `matmul_intf.h`），不要照抄宿主侧示例或内置算子源码的 `lib/matmul_intf.h`。
- 遇到 FATBIN `ld.lld: cannot open *_mix_aic_*.o` 先用 `-j1` 重跑拿真实前端错误，再定位源码；不要只看链接报错就去怀疑构建目录/缓存问题。

## 验证信号

来源说明：本卡结论合并三条独立轨迹——MoE 分组矩阵乘（CANN 9.1.0，形态一修复闭环）、Conv3d 反传 filter（CANN 9.1.0，形态二三处修改 + UT 闭环）、GRU（CANN 9.2.0，形态一触发 + 探针链）。

- MoE 轨迹修复前：`graph/types.h:82:3: error: expected identifier`（types.h:82-100 共 19 行）+ `fatal error: too many errors emitted`；include 链经 `asc/include/adv_api/matmul/matmul_tilingdata.h:18 → include/register/tilingdata_base.h:19 → include/graph/ascend_string.h:17`（来源平台：950；证据等级：tool_observed）。
- MoE 轨迹修复后：kernel 编译不再触达 graph/types.h，报错转为 `no template named 'Matmul'`（下一根因，见 [kernel Matmul 头两棵树卡](kernel_matmul_intf_include_tree.md)）；全量回归 `build.sh --soc=ascend950 -j8` 生成 custom_opp_ubuntu_x86_64.run 与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。
- GRU 轨迹（CANN 9.2.0 独立复现）：kernel 阶段 `graph/types.h:82..101 连续 expected identifier` + `fatal error: too many errors emitted`，同轮 host 侧已通过——错误仅在 kernel TU 出现（来源平台：950；证据等级：tool_observed）。
- GRU 轨迹探针输出：`matmul_shape_tiling.h:30` 的 kernel 侧 SetTiling 前向声明、`gen_kernel_tiling_data_def.py:33` 的 TCubeTiling 生成条目、`matmul_tilingdata.h:21` 的 BEGIN_TILING_DATA_DEF(TCubeTiling)（三条 grep 输出均为完整工具输出）（来源平台：950；证据等级：tool_observed）。
- Conv3d 轨迹反向信号（修改前）：同工程 `-j1` 编译稳定复现 `error: use of undeclared identifier 'optiling'`（tiling_data.h:79）与 `error: use of undeclared identifier 'ASCubeObjConfig'`（kernel 入口:38）共 7 个错误；`-j8` 下仅见 `ld.lld: cannot open ..._mix_aic_27.o/_mix_aic_283.o`（来源平台：950；证据等级：tool_observed）。
- Conv3d 轨迹修复后：`rm -rf build/ build_out/ && bash build.sh --soc=ascend950 -j8` 输出 `CPack: - package: ... custom_opp_ubuntu_x86_64.run ... successfully created`，Kernel 二进制生成（来源平台：950；证据等级：tool_observed）。
- Conv3d 轨迹修复后 UT：`cd tests/ut && bash run.sh` 输出 `✓ 通过的测试: - op_host`、`测试结果: PASS ✓`（10/10，含 tiling 用例）（来源平台：950；证据等级：tool_observed）。
- `SaveToBuffer` 序列化长度佐证：plog `InitData:TilingDef::InitData, op TCubeTiling, data size 200`，与 kernel 侧 50×int32_t 结构体一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只闭合"编译错误 + cubeTiling 数据交付"这一层；`REGIST_MATMUL_OBJ(GM/ND)` 场景下 workspace 可写性与运行期行为是另一独立问题——同一轨迹后续实测 `SaveToBuffer` 修好后 ST Real 仍 507035/507015，见 [纯 AIV 写 workspace 507035 卡](../runtime/pure_aiv_workspace_write_507035.md)与 [REGIST_MATMUL_OBJ arch35 MIX L1 崩溃卡](507015_regist_matmul_obj_arch35_mix_l1_crash.md)。
- `optiling::TCubeTiling` 与 kernel 侧纯结构体"字段为同一组 50 个 int32_t"是 CANN 9.1.0 的本轨迹实测（SaveToBuffer data size 200 + UT 通过）；其他轨迹观测到两类型**整体**布局不保证一致（跨命名空间 reinterpret_cast 曾致 tiling 全零、宿主宏类 sizeof 实测 1816B 含元数据）——禁止把宿主对象整体 reinterpret_cast 给 kernel 侧类型，跨侧只走字段级序列化（`SaveToBuffer`/逐字段）并校验 data size（参考 200），见 [Matmul 高阶 API 头不可见卡](matmul_highlevel_api_include_paths_ini.md)坏实践与 [宏生成 tiling 结构容量卡](tcube_tiling_sizeof_probe_buffer_capacity.md)。
- 版本升级后任一侧字段增删：逐字段拷贝会静默漏字段，`SaveToBuffer` 会因 `sizeof` 不匹配暴露问题——优先 `SaveToBuffer` 并校验 data size。
- `#pragma pack(8)` 下 int32_t 字段不额外填充（4≤8），结构体保持 200 字节；若未来字段类型变化引入 8 字节成员，布局结论需重验。
- 其他 arch（如 a2/a3）未验证，推断可迁移。
- GRU 轨迹曾提出的"raw buffer + memcpy 隔离"处置方向在该轨迹未完成运行期验证，不作为已验证配方（本卡主路线为 kernel_tiling.h 纯结构体，已由另两条轨迹闭合）。

**未决项**：

- `optiling::TCubeTiling(void* ptr)` 指针构造路径的实际写入布局（是否含 placeholder/偏移）未做机制层定论——可写最小 UT 用 `SaveToBuffer` 与指针构造各写一份，逐字节 diff。
- `AscendC::tiling::TCubeTiling` 与宿主 `optiling::TCubeTiling` 的字段级一致性（序列化布局是否逐字段相同）由 plain struct 镜像保证，未在本轨迹做字节级比对验证；跨版本迁移时按失效边界用 data size 校验兜底。
- `gen_kernel_tiling_data_def.py` 生成头的确切内容与 TCubeTiling 字段布局对齐方式未逐字段核对。
- 本轨迹中 `REGIST_MATMUL_OBJ` 恢复后 ST Real 仍 507035：`SaveToBuffer` 修的是编译与数据交付，未解决 workspace 可写性（跟踪见 [纯 AIV 写 workspace 507035 卡](../runtime/pure_aiv_workspace_write_507035.md)与 [REGIST_MATMUL_OBJ arch35 MIX L1 崩溃卡](507015_regist_matmul_obj_arch35_mix_l1_crash.md)）。
## 坏实践

- 把宿主框架类（`optiling::TCubeTiling`）直接塞进 kernel 侧 TilingData 结构体，或在共享头里 `#include "adv_api/matmul/matmul_tilingdata.h"`（或任何 `register/` 开头的宿主头）——两侧编译上下文不同，必然编译失败。
- 把 `optiling::TCubeTiling` 对象整体 `reinterpret_cast` 成 kernel 侧 `AscendC::tiling::TCubeTiling` 使用——两类型整体布局不保证一致，字段命名亦有差异（如 kernel 侧无 `K` 字段而是 `Ka`/`Kb`），错配致 tiling 数据全零、Matmul 内部循环失控。
- include 路径照抄别的仓库/示例/内置算子的 `lib/matmul_intf.h` 而不核对当前构建的 include 集合。
- 只看 `-j8` 的 FATBIN `ld.lld ... No such file or directory` 就去怀疑构建目录/缓存问题——那是前端编译失败被链接阶段掩盖（本轨迹中曾先去 find *.log、*.mk 排查构建产物，`-j1` 后才回到源码层）。
- 看到 graph 头报错就去改 CANN 头或加 `-fpermissive` 类编译开关绕过。

## 误诊

- 形态一：起初把 `expected identifier` 当作 graph/types.h 与 asc-devkit 两份同名头的选择冲突（include 顺序问题）排查；探针定位后确认为宿主宏族头泄漏进 kernel TU 的枚举名污染。
- 形态二：`-j8` 下曾先沿构建产物/缓存方向排查（find *.log、*.mk），`-j1` 重跑拿到 7 个前端错误后才回到源码层。
- 形态二中间版本曾用 `optiling::TCubeTiling(void* ptr)` 指针构造直接绑定 TilingData 字段，plog 中 cubeTiling 区域出现 `0x800000001b` 异常值，据此怀疑指针构造路径写入 208 字节（含 placeholder）污染后续字段；该机制解释未经源码定论，属假设——但据此换成 `SaveToBuffer` 后序列化行为有 plog data size 200 佐证，使用指针构造写法前应自行验证布局。
