---
type: "Runbook"
title: EleWise tiling 参数化切分与 stub 基线 UT 配方
description: 逐元素算子公共 TilingFunc 的可复用实现套路：任意 rank 展平 dim0、dtype 分发 tilingKey、elemBytes 参数化多核/UB 两级切分、分支委托公共路径不复制公式；配合 stub 哨兵 0-pass 基线 UT 先行（哨兵取 INT32_MIN 防撞合法负期望、常量校验类用例移出 UT），实现后 18/18、28/28 全绿。
tags: [task.implementation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T04:50:06Z'
updated_at: '2026-09-17T04:50:06Z'
platforms: ["950"]
---

# EleWise tiling 参数化切分与 stub 基线 UT 配方

## 现象

开发一个逐元素一元数学算子（任意 rank、FP32/FP16/BF16 三 dtype，bessel_i1e）时，需要实现公共 TilingFunc 并用 gtest 单元测试锁定其行为。做法是 UT 先行：先写 tiling 接口的 stub（不实现任何逻辑）与一套"独立 oracle"期望值用例，验证 stub 基线 0 通过；再实现真实的 `ChooseTilingKey`（dtype → tilingKey 分发）与 `ComputeTiling`（多核 + UB 两级切分，全部字段以 `elemBytes` 参数化），随后同一套 UT 全部转绿；各 dtype 分支（Branch-0 FP32 等）不重复写公式，而是以本分支常数委托公共参数化路径，补充分支专属用例后仍是全绿。整个过程在 `./build.sh <package>`（ascend950 目标，CANN 9.0.0-beta.2）内编译，UT 为 host 侧 gtest 可执行文件。

关键信号（stub 基线 → 实现后）：

```text
[  PASSED  ] 0 tests.            ← stub 阶段（哨兵未实现任何逻辑）
[  FAILED  ] 18 tests.
...
[  PASSED  ] 18 tests.           ← 公共 TilingFunc 实现后
...
[  PASSED  ] 28 tests.           ← Branch-0 委托公共路径实现后（公共 18 + Branch-0 10）
```

## 适用场景

- 工程形态：算子仓自定义算子开发（EleWise/逐元素范式），host 侧 TilingFunc 实现 + gtest TilingUT（测试先行，先红后绿）。
- API 路径：`op_host/<op>_tiling_host.h/.cpp`（自建 tiling 接口层）、`op_kernel/<op>_tiling.h`（TilingData 结构体）、`context` 提供的 `availableCoreNum`/`ubSize` 等输入、`ge::DataType` dtype 分发；UT CMake 复用 `$ENV{ASCEND_HOME_PATH}/x86_64-linux/include`。
- 验证过的平台与 CANN 版本：ascend950 构建目标包内 host 侧 UT（x86_64 编译执行），CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：operator_family——所有"任意 rank 展平成 1D 后逐元素处理"的算子（一元/二元 Elementwise）的公共 tiling 层。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：
  - reduce/matmul/带轴语义的算子（切分维度不是展平 dim0）；
  - 需要 workspace 切分或双缓冲深度调优的 kernel（本配方 UB 规划只按 bufferNum=2 一档推导，属正确性切分而非性能最优切分）；
  - device 侧实际执行未在本卡验证（UT 只锁 host 切分公式，kernel 消费正确性由后续 ST 覆盖）。

## 根因/解决配方

本卡是完整配方（task_recipe），核心是把"逐元素 tiling"做成 **dtype 参数化的单实现 + 哨兵 stub 基线 UT + 分支委托** 三件套：

1. **TilingData 结构体**（`op_kernel/<op>_tiling.h`，9 字段）：`dim0 / coreNum / blockFormer / blockNum / ubFormer / ubLoopOfFormerBlock / ubTailOfFormerBlock / ubLoopOfTailBlock / ubTailOfTailBlock`，全部以"元素数"为量纲并注释标明。
2. **stub 哨兵**：未实现阶段 `ChooseTilingKey` 返回 `INT32_MIN`（**不是 -1**），`ComputeTiling` 完全不写字段（调用方 memset 归零）——保证 stub 基线 0 通过，且哨兵不会巧合命中 oracle 的"不支持 dtype → -1"分支。
3. **独立 oracle UT**：期望值由设计文档公式手算（不调用被测函数），覆盖 dtype 分发 5 例 + 切分公式 13 例（空 Tensor、单元素、对齐、非对齐尾块、ubFormer 边界、多核大 shape 等）。
4. **真实实现**：`ChooseTilingKey` 按 `ge::DT_FLOAT→0 / DT_FLOAT16→1 / DT_BF16→2 / 其余→-1` 分发；`ComputeTiling` 以 `elemBytes`（FP32=4，FP16/BF16=2）参数化计算全部字段（完整代码见下）。
5. **分支委托**：`ComputeBranch0Tiling` 不复制公式，以 `{dim0, availableCoreNum, ubSize, /*elemBytes=*/4, /*bufferNum=*/2}` 委托 `ComputeTiling`。

stub 版（before，`op_host/<op>_tiling_host.cpp`）：

```cpp
#include "bessel_i1e_tiling_host.h"
#include <climits>

namespace optiling {

int32_t ChooseTilingKey(ge::DataType dtype)
{
    // STUB: 返回非法哨兵 INT32_MIN（非 -1），避免巧合命中 oracle 的不支持分支
    // UT 阶段基线通过数必须 = 0
    return INT32_MIN;
}

void ComputeTiling(const BesselI1eTilingInputs& inputs, BesselI1eTilingData& out)
{
    // STUB: 不填写 out 字段（调用方已 memset 归零），保持全 0
    // UT 阶段基线通过数必须 = 0
    (void)inputs;
    (void)out;
}

} // namespace optiling
```

实现版（after，终版全文即下节"完整代码"）。

## 设计决策

- **为什么哨兵用 `INT32_MIN` 而不是 -1**：oracle 的"不支持 dtype"分支返回 -1，若 stub 也返回 -1，`DT_DOUBLE→-1` 等用例会在 stub 阶段"假通过"，破坏 0-pass 基线；`INT32_MIN` 与一切合法期望值都不相等。这是"stub 返回值撞上合法负向期望"的一个实例——同域已另有 bool stub 恒 false 撞负向用例、期望全 0 时须预填哨兵垃圾值两种形态，三者同属 0-pass 基线防污染纪律。
- **为什么常量校验类用例被移出 UT**：最初 UT 里还有 3 个校验头文件常量（Chebyshev 系数表、对齐常数）的用例，它们不经过被测函数、stub 阶段就通过（实测 3 过 18 失败），使"通过数=0"的基线判据失效——移除后基线才是干净的 0/18。常量一致性此后靠单一事实源（实现头）维护。
- **为什么 UB 切分统一到字节量纲**：`ubSize` 是字节，元素宽度随 dtype 变化；统一用 `bufferDivisor = bufferNum * elemBytes` 折算成最大元素数、再用 `alignFactor = UB_ALIGN_BYTES / elemBytes` 对齐回元素域，避免"先按元素对齐再除字节"的量纲混算错误。
- **为什么空 Tensor 早返**：`dim0 == 0` 时只设 `coreNum = 1`、其余字段保持归零，避免 `CeilDiv(0, x)` 之类边界把 0 算成非 0。
- **为什么分支委托而不是每个 tilingKey 各写一份**：三 dtype 的切分公式同形，仅 `elemBytes` 不同；委托后公式只有一份，UT 也只需锁一份（分支补充分支专属边界用例即可），后续修公式不会出现分支间漂移。
- 当前方案属 recommended_fastpath；备选"每分支独立实现"被放弃（重复代码、易漂移）。

## 系统信息

- 工具链：gcc 13.3.0、cmake ≥3.16、gtest + gtest_main + pthread（`/usr/lib/x86_64-linux-gnu/libgtest*.a`）。
- CANN：`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`（UT CMake 以 `$ENV{ASCEND_HOME_PATH}/x86_64-linux/include` 解析 `graph/types.h`）。
- 目标平台：ascend950（`[ascend950] Generating <Op>_<hash> ...`、`aic-ascend950-ops-info.json`）。
- 目录布局：`operators/<pkg>/op_kernel/<op>_tiling.h`（TilingData）、`operators/<pkg>/op_host/<op>_tiling_host.h/.cpp`（接口 + 实现）、`operators/<pkg>/tests/ut/test_<op>_tiling.cpp`（公共 UT）、`operators/<pkg>/tests/ut/test_<op>_branch0_tiling.cpp`（分支 UT，无自有 main）、`operators/<pkg>/tests/ut/CMakeLists.txt`（UT 编入包构建：包级 CMake `add_subdirectory(tests/ut)` + `add_dependencies(binary <op>_tiling_ut)`；binary 聚合目标挂依赖的通用做法见同域 build.sh 接线卡）。
- UT CMake（被测 = 真实 tiling 源文件同源编译，不做 mock）：

```cmake
cmake_minimum_required(VERSION 3.16.0)

# PKG_ROOT = operators/<pkg>
get_filename_component(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

add_executable(bessel_i1e_tiling_ut
    test_bessel_i1e_tiling.cpp
    ${PKG_ROOT}/op_host/bessel_i1e_tiling_host.cpp
)

target_include_directories(bessel_i1e_tiling_ut PRIVATE
    ${PKG_ROOT}/op_host
    ${PKG_ROOT}/op_kernel
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include
)

target_link_libraries(bessel_i1e_tiling_ut PRIVATE
    gtest
    gtest_main
    pthread
)
```

## 完整代码

公共 TilingFunc 实现终版（`op_host/<op>_tiling_host.cpp`，verbatim）：

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * bessel_i1e 公共 Tiling 实现
 * §5.2 多核/UB 切分公式 + §5.5 dtype→tilingKey 判定
 */

#include "bessel_i1e_tiling_host.h"
#include <algorithm>

namespace {

inline int64_t CeilDiv(int64_t a, int64_t b)  { return (a + b - 1) / b; }
inline int64_t AlignUp(int64_t v, int64_t f)   { return CeilDiv(v, f) * f; }
inline int64_t AlignDown(int64_t v, int64_t f) { return (v / f) * f; }

} // namespace

namespace optiling {

int32_t ChooseTilingKey(ge::DataType dtype)
{
    // §5.5: DT_FLOAT → 0, DT_FLOAT16 → 1, DT_BF16 → 2, 其余 → -1
    switch (dtype) {
        case ge::DT_FLOAT:   return 0;
        case ge::DT_FLOAT16: return 1;
        case ge::DT_BF16:    return 2;
        default:             return -1;
    }
}

void ComputeTiling(const BesselI1eTilingInputs& inputs, BesselI1eTilingData& out)
{
    // §5.3 公式：以 elemBytes 参数化计算全部 TilingData 字段
    const int64_t dim0            = inputs.dim0;
    const int32_t availableCoreNum = inputs.availableCoreNum;
    const int64_t ubSize          = inputs.ubSize;
    const int32_t elemBytes       = inputs.elemBytes;
    const int32_t bufferNum       = inputs.bufferNum;

    out.dim0 = dim0;

    // 空 Tensor 早返
    if (dim0 == 0) {
        out.coreNum = 1;
        // 其余字段保持调用方 memset 归零
        return;
    }

    const int32_t elemBits = elemBytes * 8;

    // §5.2.1 多核切分
    int64_t rawCores = CeilDiv(dim0 * static_cast<int64_t>(elemBits), MIN_TILING_BITS);
    out.coreNum = static_cast<int32_t>(std::min(rawCores, static_cast<int64_t>(availableCoreNum)));
    if (out.coreNum < 1) { out.coreNum = 1; }

    int64_t perCore     = CeilDiv(dim0, out.coreNum);
    out.blockFormer     = AlignUp(perCore, ELEM_ALIGN_FACTOR);
    out.blockNum        = CeilDiv(dim0, out.blockFormer);

    // §5.2.2 UB 切分（统一字节量纲）
    int64_t bufferDivisor = static_cast<int64_t>(bufferNum) * static_cast<int64_t>(elemBytes);
    int64_t maxElemNum    = ubSize / bufferDivisor;
    int64_t alignFactor   = UB_ALIGN_BYTES / static_cast<int64_t>(elemBytes);
    out.ubFormer = AlignDown(maxElemNum, alignFactor);
    if (out.ubFormer < 1) { out.ubFormer = alignFactor; }

    // 首 block 循环
    out.ubLoopOfFormerBlock = CeilDiv(out.blockFormer, out.ubFormer);
    out.ubTailOfFormerBlock = out.blockFormer - (out.ubLoopOfFormerBlock - 1) * out.ubFormer;

    // 尾 block
    int64_t blockTail = dim0 - (out.blockNum - 1) * out.blockFormer;
    out.ubLoopOfTailBlock = CeilDiv(blockTail, out.ubFormer);
    out.ubTailOfTailBlock = blockTail - (out.ubLoopOfTailBlock - 1) * out.ubFormer;
}

} // namespace optiling
```

分支委托（Branch-0 FP32，stub → 委托公共路径）：

```cpp
void ComputeBranch0Tiling(const BesselI1eBranch0TilingInputs& inputs, BesselI1eTilingData& out)
{
    // Branch-0: FP32 原生路径（DESIGN-BRANCH-0.md §2.2）.
    // elemBytes=4, bufferNum=2, alignFactor=64 (=256/4).
    // 公式与公共 ComputeTiling 同形，委托其参数化路径，以公共常量代入计算.
    BesselI1eTilingInputs common{inputs.dim0, inputs.availableCoreNum, inputs.ubSize, 4, 2};
    ComputeTiling(common, out);
}
```

接口层常量与输入结构（`op_host/<op>_tiling_host.h`，节选）：

```cpp
#include <cstdint>
#include "graph/types.h"
#include "../op_kernel/bessel_i1e_tiling.h"

namespace optiling {

// ===== §5.2 Tiling 对齐常量 =====
constexpr int64_t ELEM_ALIGN_FACTOR = 512;
constexpr int64_t UB_ALIGN_BYTES    = 256;
constexpr int64_t MIN_TILING_BITS   = 32768;

// ===== ChooseTilingKey：dtype → tilingKey（§5.5）=====
// DT_FLOAT → 0, DT_FLOAT16 → 1, DT_BF16 → 2, 其余 → -1
int32_t ChooseTilingKey(ge::DataType dtype);

// ===== 公共 Tiling 计算（§5.2）=====
struct BesselI1eTilingInputs {
    int64_t dim0;             // 展平元素总数
    int32_t availableCoreNum; // 可用核数
    int64_t ubSize;           // UB 总字节数
    int32_t elemBytes;        // 单元素字节数（FP32=4, FP16/BF16=2）
    int32_t bufferNum;        // buffer 数（=2：1输入+1输出）
};

void ComputeTiling(const BesselI1eTilingInputs& inputs, BesselI1eTilingData& out);

} // namespace optiling
```

UT 参数化骨架（期望值由公式手算 oracle 独立给出，非被测函数产出）：

```cpp
int32_t OracleChooseTilingKey(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:   return 0;
        case ge::DT_FLOAT16: return 1;
        case ge::DT_BF16:    return 2;
        default:             return -1;
    }
}

INSTANTIATE_TEST_SUITE_P(Sec55, DtypeTilingKeyTest, testing::Values(
    DtypeTilingKeyCase{"DT_FLOAT   → 0",               ge::DT_FLOAT,   0},
    DtypeTilingKeyCase{"DT_FLOAT16 → 1",               ge::DT_FLOAT16, 1},
    DtypeTilingKeyCase{"DT_BF16    → 2",               ge::DT_BF16,    2},
    DtypeTilingKeyCase{"DT_DOUBLE  → -1(unsupported)", ge::DT_DOUBLE, -1},
    DtypeTilingKeyCase{"DT_INT32   → -1(unsupported)", ge::DT_INT32,  -1}
));
```

## 性能影响

本卡为正确性切分配方，无性能故事（未做 tiling 参数调优对比）；性能调优时的第一测量点见遗留问题。

## 预防

- 逐元素算子 tiling 先写 stub 哨兵 + 0-pass 基线 UT，再实现；基线不干净（stub 阶段有用例通过）说明要么哨兵撞了合法期望值、要么用例没有真正驱动被测函数（常量校验类用例即属后者）。
- 切分字段全部注明量纲（元素数 vs 字节），UB 容量折算必须经过 `bufferNum × elemBytes`。
- 分支 tiling 一律委托参数化公共路径，禁止复制公式。

## 验证信号

以下信号来自该算子一次开发会话的 tiling 开发全程（stub 基线 → 实现 → 分支委托），全部为同日同工程实测（ascend950 构建目标的 host 侧 gtest UT，x86_64 执行）：

- stub 基线：移除常量类用例后公共 TilingUT 18 用例 0 通过 / 18 失败（stub 返回 INT32_MIN、不填字段）（来源平台：950；证据等级：tool_observed）。
- 公共 TilingFunc 实现后：`./build.sh <package>` 退出码 0、无 "do not registe tiling struct" 告警，公共 TilingUT 18/18 全部通过（5 dtype 分发 + 13 切分公式）（来源平台：950；证据等级：tool_observed）。
- Branch-0 以 elemBytes=4/bufferNum=2 委托公共路径后：UT 28/28 全部通过（公共 18 + Branch-0 10，无回归）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 切分常量（`ELEM_ALIGN_FACTOR=512`、`UB_ALIGN_BYTES=256`、`MIN_TILING_BITS=32768`、`bufferNum=2`）来自该算子设计文档，换平台/换算子须按目标芯片对齐要求重新取值；公式骨架可迁移。
- UB 规划只考虑"1 输入 + 1 输出"两 buffer；kernel 需要更多 UB buffer（双缓冲、中间 temp）时 `bufferNum` 须相应增大，否则 UB 越界。
- 本卡只验证 host 侧切分公式（UT）；tiling 数据被 kernel 正确消费、上板精度与性能未在本卡验证（由白盒 ST 与后续联调覆盖）。
- 未验证，推断可迁移：FP16/BF16 分支（Branch-1/2）委托同一公式路径在本会话尚未实现，公式按 elemBytes 对称设计、推断可迁移但无实测。

**未决项**：

- tiling 参数（blockFormer 对齐宽度、ubFormer 上限）未做性能调优；下一测量点是在白盒 ST/性能阶段对比不同对齐档位的端到端耗时。
- Branch-1（FP16）/Branch-2（BF16）委托路径待实现后回归（推断可迁移，未验证）。
## 坏实践

- stub 返回 -1 当哨兵：与"不支持 dtype → -1"的合法期望撞值，用例假通过、基线失真（本会话显式选 `INT32_MIN` 规避）。
- 在 UT 里放"校验头文件常量"的用例：不经过被测函数，stub 阶段即通过，污染 0-pass 基线。
- 每个 tilingKey 复制一份切分公式：dtype 间公式漂移的温床。

## 误诊

- 无（本卡为正向配方；tiling UT 编译期曾另遇 `ge::DataType` 枚举重定义问题，与本卡根因无关，未单独成卡）。
