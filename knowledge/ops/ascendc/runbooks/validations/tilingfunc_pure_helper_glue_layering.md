---
type: "Runbook"
title: "Host Tiling 纯函数分层与 UT 同源编译"
description: "自定义算子 host tiling 按\"纯函数 helper（无 CANN 依赖）+ 薄 TilingFunc glue（只做 CANN API 调用）\"分层；UT 直接编译同一份 helper .cpp（same-translation-unit），TDD 红转绿不动测试，测的就是 TilingFunc 调的代码。复现增量：同一源文件须同时兼容 C++11（opbuild 宿主链）与 C++17（GTest UT）；UT 经 add_dependencies(binary) 编入包级 binary 目标随 build.sh 一并构建；新增分支的红基线用\"不填 TilingData + 哨兵 0xDEADBEEF\"获得逐字段判别力。复现增量：链接期 stub + 实现宏切换（target_compile_definitions 排除 stub，宏定义而实现缺失即链接失败）；同一 .cpp 经 npu_op_code_gen/npu_op_library/UT 三目标同源链接；异构 oracle 以 awk 核验独立性；bool 返回值 stub 须配合宏门控 report 防 0-pass 基线假通过。"
tags: [task.implementation, task.testing, topic.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# Host Tiling 纯函数分层与 UT 同源编译

## 现象

为自定义算子 BenchHarnessSoftmax（ascend950 / arch35）实现 host 侧 TilingFunc（合轴预处理、空 tensor 短路、UB 切分、多核切分、Group 判定、tilingKey 编码）时面临测试困境：TilingFunc 本体要调 CANN 上下文 API（`gert::TilingContext`、平台参数查询、`SetTilingKey` 等），这类代码在 host 单测里既难构造输入也难断言；而若为测试另写一份"测试版公式"，测试就与真实执行路径脱钩——TilingFunc 里改了公式、UT 仍然全绿，测试失去意义。

开发流程上先用 gtest + 哨兵 stub 建立了 41 例公共 tiling UT（0/41 红基线），随后按下面分层把 stub 换成真实实现，同一批 UT 41/41 转绿，后续公共层扩展到 51/51，TilingFunc glue（真实 aclnn 路径）同时落地并通过 smoke 回归。

## 适用场景

- 工程形态：CANN 自定义算子（Ascend C）host 侧 TilingFunc 开发 + gtest 单测；TDD（测试先行的红/绿基线）。
- API 路径：`gert::TilingContext`（GetInputShape/GetInputTensor/GetAttrs/GetPlatformInfo/GetTilingData/SetTilingKey/SetBlockDim/SetScheduleMode/GetWorkspaceSizes）；`op_common/op_host/util/platform_util.h` 的 `Ops::Base::GetUbBlockSize/GetCacheLineSize/GetVRegSize`；`platform_ascendc::PlatformAscendC`（GetCoreMemSize(UB)/GetCoreNumAiv）。
- 验证过的平台与 CANN 版本：ascend950（arch35），CANN 9.x（构建 9.1.0 / 运行 9.2.0）。
- 适用粒度：`platform_wide`（所有自定义算子的 host tiling 层）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用 / 未验证场景：kernel 侧（device 代码）测试不适用；tiling 与 CANN 上下文强耦合、无法抽纯函数的逻辑（如必须读 CompileInfo 的场景）未验证。

## 解决配方

分层规则一句话：**公式进纯函数，CANN 调用进 glue；UT 与算子库编同一份纯函数源文件。**

1. **纯函数 helper 层** `op_host/<op>_tiling_host.{h,cpp}`：只依赖 `<cstdint>` 等标准头，不 include 任何 CANN 头。签名示例（合轴/切分/判定）：

```cpp
int32_t HandleEmptyTensor(const int64_t* shape, int32_t rank, int32_t dimIdx,
                          int32_t& isRAxisEmpty);
void PreprocessPattern(const int64_t* shape, int32_t rank, int32_t dimIdx,
                       PatternState& ps);
int32_t ShouldUseGroup(int64_t aLoopCntTotal, int32_t coreNum,
                       int64_t rLoopCntTotal);
int32_t EncodeTilingKey(int32_t templateType, int32_t isEmptyTensor,
                        int32_t isTailR);
// ComputeAUbFactor / ComputeRUbFactor / ExpandAIfRFullyLoaded /
// ComputeFusedALoopSplit / ComputeRLoopCnt / ComputeGroupSplit / ComputeBranch0Tiling ...
```

2. **薄 glue 层** `op_host/arch35/<op>_tiling_arch35.cpp`：只做 CANN API 调用与字段装配，不含公式。关键片段（verbatim）：

```cpp
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // ─── 0) null / platform / hardware params ──────────────────────────────
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE("BenchHarnessSoftmax", "GetPlatformInfo returned nullptr");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC plat(platformInfo);
    uint64_t ubSize = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = plat.GetCoreNumAiv();
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize(context);   // 32B
    uint32_t cacheLineSize = Ops::Base::GetCacheLineSize(context); // 256B
    uint32_t vRegSize = Ops::Base::GetVRegSize(context);          // 256B
    if (ubSize == 0 || coreNum == 0 || ubBlockSize == 0 ||
        cacheLineSize == 0 || vRegSize == 0) {
        OP_LOGE("BenchHarnessSoftmax",
                "invalid hardware params: ubSize=%lu coreNum=%u ubBlockSize=%u "
                "cacheLineSize=%u vRegSize=%u",
                ubSize, coreNum, ubBlockSize, cacheLineSize, vRegSize);
        return ge::GRAPH_FAILED;
    }

    // ─── 1) input shape / dtype（GetInputShape/GetInputTensor/GetDataType）──
    // ─── 2) dim attr fold（GetAttrs()->GetAttrPointer<int64_t>(0)）─────────
    // ─── 3)~6) 空短路 / 合轴四步 / UB 切分三步 / 多核切分：全部调用纯函数 ──

    // ─── 7) Group trigger ────────────────────────────────────────────────────
    int32_t templateType = ShouldUseGroup(aLoopCntTotal, static_cast<int32_t>(coreNum),
                                          rLoopCntTotal);
    int64_t rGroupCnt = 0;
    int64_t rPerCore = 0;
    uint64_t userWs = 0;
    if (templateType) {
        ComputeGroupSplit(aLoopCntTotal, rLoopCntTotal,
                          static_cast<int32_t>(coreNum),
                          rGroupCnt, rPerCore, usedCoreNum);
        context->SetScheduleMode(1);  // SyncAll requirement
        // Group workspace: [rGroupCnt, aTotal] fp32 dense
        userWs = static_cast<uint64_t>(rGroupCnt * totalA * sizeof(float));
    }

    // ─── fill TilingData：td->axisNum/axisShape/axisStride/aUbFactor/... ──
    td->cacheBufUbSize = CACHE_BUF_SIZE;  // ⛔ fixed 16KB

    // ─── SetTilingKey / SetBlockDim / SetWorkspaceSize ────────────────────
    int32_t tilingKey = EncodeTilingKey(templateType, /*isEmptyTensor=*/0,
                                        static_cast<int32_t>(isTailR));
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum > 0 ? usedCoreNum : 1));

    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) ws[0] = static_cast<size_t>(userWs);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BenchHarnessSoftmax).Tiling(TilingFunc);
```

3. **UT 同源编译**：`tests/ut/CMakeLists.txt` 的源文件直接写 `../../op_host/<op>_tiling_host.cpp`（same-translation-unit），并把它同时登记进顶层 `host_tiling_srcs`——算子库与 UT 编译的是**同一份** helper 源文件：

```cmake
add_executable(${UT_EXE}
    test_common_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../op_host/bench_harness_softmax_tiling_host.cpp
)
target_include_directories(${UT_EXE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../op_host   # helper header + tiling_data.h
)
```

4. **TDD 红基线 stub 纪律**：stub 不能只是 no-op——no-op 在平凡输入（原始轴列恰好已是最终形态）上可能与 oracle 意外一致而假绿；stub 必须写哨兵：

```cpp
void DropSizeOneAxes(PatternState& ps) {
    // STUB: write sentinels so the result never matches the oracle (avoids
    // accidental pass on trivial inputs where the raw list is already final).
    ps.axisNum = INT32_MIN;
    for (int32_t i = 0; i < MAX_RAW_RANK; ++i) {
        ps.axisShape[i] = INT64_MIN;
        ps.axisType[i] = INT32_MIN;
    }
}
```

实现落地时只删 tests/ut 下的 stub 文件、改 UT 源文件列表指向真实实现，**测试用例与 oracle 一行不动**。

## 设计决策

- 为何纯函数分层：UT 无法构造 `gert::TilingContext`（其输入 shape/attr/平台参数由 GE 运行时注入），把公式抽成 `(shape, rank, dimIdx, 常量) → 结果` 的纯函数后，测试输入就是普通数组与整数。
- 为何同源编译而非复制代码：杜绝"测试版公式"漂移；实现改公式，UT 立刻感知。文件内注释明确写了这一约束（"kept file-static here so ComputeBranch0Tiling is self-contained and the UT exercises the same code the real TilingFunc glue calls"）。
- 为何 glue 里先校验 5 个硬件参数非 0：`GetUbBlockSize/GetCacheLineSize/GetVRegSize/GetCoreMemSize(UB)/GetCoreNumAiv` 任一返回 0 都意味着平台信息异常，直接 `GRAPH_FAILED` 早退，避免用 0 做除法/对齐。
- 备选与放弃：在 UT 里 mock TilingContext（要桩掉大量 CANN 符号，成本高且测不到真实 glue）；dlopen 独立 tiling so 的重型框架（对本卡场景过重）。
- `SetBlockDim` 头文件已标注 `@deprecated`（推荐 `SetSimdNumBlocks`），本实现仍用 SetBlockDim 以兼容当前版本；迁移时注意。

## 系统信息

- 工具链：x86_64，g++ 13.3.0；CANN 9.x（构建 9.1.0 / vendor 安装与运行 9.2.0）。
- 目标平台：ascend950（arch35）。
- 关键 CANN 头：`$ASCEND_HOME_PATH/x86_64-linux/include/op_common/op_host/util/platform_util.h`（`Ops::Base` 模板函数，形参带 context）；`exe_graph/runtime/tiling_context.h`（SetTilingKey/SetScheduleMode/SetBlockDim/GetWorkspaceSizes/GetTilingData）；`exe_graph/runtime/{storage_shape,shape,tensor,runtime_attrs,extended_kernel_context}.h`。
- 相关文件：`op_host/bench_harness_softmax_tiling_host.{h,cpp}`（纯函数）、`op_host/arch35/bench_harness_softmax_tiling_arch35.cpp`（glue）、`op_host/tiling_data.h`（TilingData 结构）、`tests/ut/test_common_tiling.cpp` / `tests/ut/test_branch0_tiling.cpp`（gtest + 独立 oracle）、`tests/ut/CMakeLists.txt`、顶层 `CMakeLists.txt`（host_tiling_srcs）。

## 预防

- 新算子 host tiling 开工即建三层文件：`<op>_tiling_host.{h,cpp}`（纯函数）+ `arch35/<op>_tiling_arch35.cpp`（glue）+ `tests/ut/`（同源编译）；先写 oracle 用例配哨兵 stub（红），再填实现（绿）。
- glue 收尾固定核对清单：5 硬件参数非 0、空 tensor 短路、SetTilingKey、SetBlockDim≥1、Group 时 SetScheduleMode(1) + workspace、TilingData 字段单位（字节 vs 元素数）与结构体注释一致。

## 验证信号

来源说明：BenchHarnessSoftmax 公共 Tiling 与 Branch-0 Tiling 实现会话（build.sh、gtest 运行、aclnn smoke 回归输出）。

- 红基线（stub 哨兵）：UT `41 tests from 9 test suites`，`[  PASSED  ] 0 tests.`（来源平台：950 工具链构建环境；证据等级：tool_observed）。
- 真实实现落地后同一批 UT `[  PASSED  ] 41 tests.`，且 aclnn smoke 回归 `3/3 cases PASSED`（fp16/fp32/bf16 n=64 dim=-1）不退化（来源平台：950；证据等级：tool_observed）。
- Branch-0 扩展（同源编译同一 helper 文件）后 `[  PASSED  ] 51 tests.`（41 公共 + 10 分支）（来源平台：950 工具链构建环境；证据等级：tool_observed）。
- 字段单位一致性核对：`tiling_data.h` 注释"固定 16KB（16384 字节）" = `tiling_host.h` `CACHE_BUF_SIZE = 16*1024` = glue `td->cacheBufUbSize = CACHE_BUF_SIZE` 三处一致（来源平台：950 工具链构建环境；证据等级：tool_observed）。

## 失效边界

- 纯函数层不覆盖 glue 本身的错误（如 attr 索引写错、SetTilingKey 未调用、平台参数误读）——glue 只能靠 aclnn ST/白盒覆盖。
- helper 若需要读 `GetCompileInfo`（解析期产物），纯函数签名需扩展输入结构，本卡未验证该场景。
- TDD 红基线依赖"stub 写哨兵"；若 stub 改回 no-op，平凡输入可能假绿。
- 其他 CANN 版本的 `Ops::Base` 工具函数签名未验证（推断可迁移）。

**未决项**：

- 无（本卡覆盖的公共层与 Branch-0 均已闭环；Group 路径的 glue 逻辑已写入但其在真实 kernel 下的行为在后续 kernel 任务验证）。
## 坏实践

- 为通过 UT 在测试文件里复刻一份公式（非独立 oracle）——公式改了测试不感知。
- oracle 函数与被测函数共用常量/结构体引用（`using optiling::MAX_RAW_RANK`）：虽只是数组容量常量，仍建议 oracle 侧本地声明（会话中已改为本地 `ORACLE_MAX_RAW_RANK = 8`，零逻辑依赖）。

## 误诊

- oracle 独立性检查若用 `grep 'Oracle…' -A 40 | grep 'optiling::'` 会假阳性——上下文窗口会覆盖到 TEST_P 体内**合法的**被测函数调用（`act = optiling::ComputeBranch0Tiling(in, act)`）。正确做法是按函数边界提取 oracle 区段（sed 行区间 / awk 到 `}  // namespace` / python 提取匿名命名空间）再 grep；会话中三种提取均确认 oracle 主体零 `optiling::` 调用。

## 增量（双标准同源编译、binary 目标挂接与逐字段红基线）

另一算子族（RoPE 类 BenchHarnessApplyRotaryPosEmb，Ascend950PR + CANN 9.2.0）以同一分层模式完成公共 Tiling + 8 个分支的逐分支 TDD，补齐三条工程细节：

**1) 同一 helper 源文件双标准编译（C++11 / C++17）**：`op_host/tiling_host.cpp` 同时编进 opbuild 宿主链（真实 TilingFunc glue 调用，`-std=c++11`）与 GTest UT（`-std=c++17`）——同源编译的前提是该文件不得使用 C++14+ 特性，否则破坏 opbuild 链。UT 的 `add_executable` 直接编 `${CMAKE_SOURCE_DIR}/op_host/<op>_tiling_host.cpp`，并 `target_compile_features(${UT_NAME} PRIVATE cxx_std_17)`。

**2) UT 编入包级 binary 目标**：`add_dependencies(binary ${UT_NAME})` 使 `./build.sh`（`make -j binary package`）单命令同时产出交付物与测试；`binary` 目标由 `npu_op_kernel_library` 创建，`add_subdirectory(tests/ut)` 必须放在其之后。构建日志同时以无 "do not registe tiling struct" 告警作旁证（tiling 结构注册链完整）。

**3) 新增分支的逐字段红基线（哨兵 0xDEADBEEF）**：分支函数 stub "不填 TilingData（调用方 memset 归零）+ 返回 0xDEADBEEF"，与主卡的"stub 写哨兵值"互补——不写字段使每个 `EXPECT_EQ(act.X, exp.X)` 都从字段级暴露差异（如 `act.tilingKey Which is: 0 / exp.tilingKey Which is: 3`），红基线自带定位信息；0xDEADBEEF 超出任何物理 coreNum，杜绝与空 Tensor oracle `blockDim=1` 或字段默认 0 巧合相等。TEST_P 体内"先调被测、后调 oracle"（act-before-exp），使"oracle 定义之后 N 行窗口内无被测函数调用"的 grep 独立性检查天然干净（与主卡误诊节的函数边界提取法互为替代）。

stub 形态（verbatim）：

```cpp
uint32_t ComputeBranch3Tiling(const Branch3Inputs& in, BenchHarnessApplyRotaryPosEmbTilingData& td)
{
    (void)in;
    (void)td;
    return 0xDEADBEEFu;  // 哨兵：与任何 oracle blockDim（≤ coreNum）必不等
}
```

逐分支增量的红绿节奏（每新增一个 tilingKey 分支重复一次）：先落分支 UT（红基线：新分支 N 例全红、前序用例全过无回归）→ 实现特化函数（公共切分复用 `ComputeTiling` 单一真值源，分支只填差异字段与 tilingKey）→ 同批 UT 全绿。

验证信号（复现增量，BenchHarnessApplyRotaryPosEmb 公共与分支 Tiling 的 TDD 会话，宿主 x86_64 gtest / 编译目标 ascend950）：

- 公共层红基线：`[  PASSED  ] 0 tests.` / `[  FAILED  ] 33 tests`（哨兵 stub）；公式实现后 `[  PASSED  ] 33 tests.`，追加分支 0 后 45/45，验收复跑 5 套件 45 tests 全过（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。
- 分支增量红绿（8 分支累计）：branch-2 红基线 14 例全 FAILED（`act.tilingKey=0 vs exp=2`）→ 实现 71/71；branch-3 红基线 14 例（`act.cosDstStrideD 0 vs 64`、`act.tilingKey 0 vs 3`）→ 85/85；branch-6 红基线 17 例（`act.cosSrcStrideDh 0 vs 64`、`act.tilingKey 0 vs 6`）→ 131/131；branch-7 增量后 148/148——每轮前序用例零回归（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。
- 编译链自洽：`build.sh` exit 0 且 UT 编入 `make -j binary package`，构建日志无 "do not registe tiling struct" 告警（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。
- oracle 独立性机械检查：限定行区间内 `optiling::` 函数调用为零（act-before-exp 排序后 grep 干净）（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。

失效边界（复现增量）：

- 双标准约束要求 helper 不用 C++14+ 特性；引入即破坏 opbuild 链（这是同源编译的代价）。
- `add_dependencies(binary ...)` 依赖包级构建布局；换独立 CMake 配置时须重新保证同源性。
- 哨兵 0xDEADBEEF 的防碰撞论证依赖"被测输出域为小非负整数（blockDim ≤ coreNum、字段为 shape/stride 值）"；输出域含大数时另选哨兵。

## 增量（宏切换链接期 stub 与生产链三目标同源链接）

另一算子（softplus_v2_grad，AscendC aclnn 工程，Ascend950PR + CANN 9.1.0）以同一"纯函数 + 薄 glue + 同源编译"分层完成公共 tiling 与 rank4 分支的测试先行 TDD，补齐三条工程细节：

**1) 链接期 stub + 实现宏切换（与"删 stub 文件改源列表"等价的替代形态）**：stub 与用例写在同一 UT 文件内，以 `#ifndef <OP>_COMMON_IMPLEMENTED` 守卫提供链接期定义（bool 恒返回 false、整型返回 -1、出参填哨兵值）；实现落地时 UT 侧只需 CMake `target_compile_definitions(<UT> PRIVATE <OP>_XXX_IMPLEMENTED)` 定义宏排除 stub。红阶段用例代码与函数签名即被完整编译校验（只延迟"判定生效"），且"宏已定义而实现缺失"会直接链接失败，形成天然保护：

```cpp
// The real tiling functions are declared below but NOT yet defined; temporary
// stubs (guarded by SOFTPLUS_V2_GRAD_COMMON_IMPLEMENTED) provide link-time
// definitions that return false / -1, so every test case FAILS → pass count = 0.
#ifndef SOFTPLUS_V2_GRAD_COMMON_IMPLEMENTED
namespace softplus_v2_grad_tiling {
bool PadAndSqueeze(const std::vector<std::vector<int64_t>>&,
                   const std::vector<std::vector<int64_t>>&,
                   std::vector<int64_t>&,
                   std::vector<std::vector<int64_t>>&,
                   std::vector<std::vector<int64_t>>&)
{
    return false;   // 出参填哨兵值，避免 UB
}
// ... 其余 stub 同型
} // namespace softplus_v2_grad_tiling
#endif
```

分层原则取自 `op_host/softplus_v2_grad_common_tiling.h` 文件头注释（verbatim）：

```cpp
// 这些函数为纯 host 侧逻辑（仅依赖 std::vector<int64_t>），无 CANN SDK 依赖，
// 故可同时编入 cust_optiling（供 TilingFunc 调用）与 TilingUT（供单测链接）。
// 函数签名与 tests/tiling_ut/tiling_ut.cpp 中的声明逐字一致。
```

**2) 生产链三目标同源链接**：同一份纯 host .cpp 同时编入 `npu_op_code_gen(SRC ...)`、`npu_op_library(cust_optiling TILING ...)` 与 UT 可执行文件——被测对象就是生产 cust_optiling 里跑的那份代码，杜绝"测试实现/生产实现"漂移：

```cmake
# 修复前：UT 只链 UT 自身源文件（stub 生效，0-pass 基线）
add_executable(tiling_ut tests/tiling_ut/tiling_ut.cpp)
target_include_directories(tiling_ut PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
set_target_properties(tiling_ut PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# 修复后（实现落地时）：同一份实现源进双目标，UT 侧定义宏排除 stub
file(GLOB host_ops_srcs
    op_host/softplus_v2_grad_host.cpp
    op_host/softplus_v2_grad_common_tiling.cpp
)
npu_op_code_gen(SRC ${host_ops_srcs} PACKAGE ${package_name} OUT_DIR ${ASCEND_AUTOGEN_PATH})
npu_op_library(cust_optiling TILING ${host_ops_srcs})

add_executable(tiling_ut
    tests/tiling_ut/tiling_ut.cpp
    op_host/softplus_v2_grad_common_tiling.cpp
)
target_include_directories(tiling_ut PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
target_compile_definitions(tiling_ut PRIVATE SOFTPLUS_V2_GRAD_COMMON_IMPLEMENTED)
```

**3) 异构 oracle + awk 独立性核验**：期望值由 `namespace oracle` 内"从头重写、变量名与代码路径刻意不同"的第二实现计算（如 `/32*32` vs `& ~31` 同一 floor-to-32 的两种写法），防 copy-paste 耦合——实现里的公式错误会被同源 oracle 原样复现。独立性用命令机械核验而非口头声明：

```bash
# 抽出 oracle 段，确认其中没有对被测命名空间的调用（输出为空 = 独立）
awk '/^namespace oracle/,/^} \/\/ namespace oracle/' tests/tiling_ut/tiling_ut.cpp \
    | grep -n 'softplus_v2_grad_tiling' ; echo "exit: 0（无输出=独立）"
```

**注意（bool 返回值 stub 的假阳性）**：bool 返回值函数写不出"与所有期望都不相等"的哨兵——只有 true/false 两个取值，负向用例期望 false 时常量 false stub 恒假 PASS（正向用例期望 true 时常量 true stub 同理）。红基线语义须配合"实现宏门控 report() 计分"（见 [TilingUT 基线假通过：stub 常量撞上负向期望](stub_constant_collides_negative_expectation.md)），与主卡及先前增量的哨兵 stub 纪律互补：哨兵适用于整型/结构体出参，计分门控适用于 bool 返回值。

验证信号（复现增量，softplus_v2_grad 公共与分支 tiling 测试先行会话，宿主侧 UT，包目标平台 ascend950 / CANN 9.1.0）：

- 公共 UT 测试先行 `Summary: 0 / 27 PASSED`、退出码 1（stub 基线）；实现落地后同一 UT `Summary: 27 / 27 PASSED`、退出码 0（来源平台：950；证据等级：tool_observed）。
- 分支 UT 同模式复刻：`tiling_ut_rank4` 测试先行 `Summary: 0 / 12 PASSED`、`EXIT_CODE=1`；实现落地后 `Summary: 12 / 12 PASSED`，同轮公共 UT 仍 27/27 无回归（来源平台：950；证据等级：tool_observed）。
- oracle 独立性：awk 抽取 oracle 段 grep 被测命名空间输出为空（来源平台：950；证据等级：tool_observed）。
- 生产链路无回归：`./build.sh softplus_v2_grad_package` 退出码 0，aclnn smoke `9/9 dtype combinations PASSED`（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 宏切换要求 stub 与实现签名逐字一致（同一头文件声明保证）；多分支并存时各分支 UT 用各自 `<BRANCH>_IMPLEMENTED` 宏，宏集合要与链接的实现源一一对应。
- 该工程链（npu_op_code_gen/npu_op_library）绑定算子仓 CMake 模板；与复现增量的 opbuild 宿主链（C++11/C++17 双标准）是两种构建形态，迁移时按目标工程链选择。
- 0-pass 基线语义依赖宏门控 report（见上）；未接门控直接套本架构会保留 bool stub 假阳性风险。

## 增量（glue 层属性按索引读取的类型映射与顺序锁）

另一独立 trace（ApplyAdamW 类算子，7 个标量属性 lr/beta1/beta2/weight_decay/epsilon/step/maximize，arch35/Ascend950PR，CANN 9.2.0 构建 + 9.1.0 复验）补 glue 层属性读取的 API 事实（与上文"dim attr fold"同层）：

**属性不是按名字读，而是按 `OP_ADD`/OpDef 中 `Attr()` 声明顺序的索引读**，且读取接口按属性声明类型固定（存取类型映射：`.Float()` 属性存储为 `float`、`.Int()` 为 `int64_t`、`.Bool()` 为 `bool`）：

```cpp
    // ---- attrs (indexed by declaration order in def.cpp) ----
    // def.cpp Attr order: 0:lr 1:beta1 2:beta2 3:weight_decay 4:epsilon 5:step 6:maximize
    // .Float attrs are stored as float; .Int as int64_t; .Bool as bool.
    double  lr          = 1.0e-3;
    double  beta1       = 0.9;
    double  beta2       = 0.999;
    double  weight_decay= 0.01;
    double  epsilon     = 1.0e-8;
    int64_t step        = 1;
    bool    maximize    = false;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* fPtr = attrs->GetFloat(0);  if (fPtr != nullptr) lr           = static_cast<double>(*fPtr);
        fPtr = attrs->GetFloat(1);                if (fPtr != nullptr) beta1        = static_cast<double>(*fPtr);
        fPtr = attrs->GetFloat(2);                if (fPtr != nullptr) beta2        = static_cast<double>(*fPtr);
        fPtr = attrs->GetFloat(3);                if (fPtr != nullptr) weight_decay = static_cast<double>(*fPtr);
        fPtr = attrs->GetFloat(4);                if (fPtr != nullptr) epsilon      = static_cast<double>(*fPtr);
        const int64_t* iPtr = attrs->GetInt(5);   if (iPtr != nullptr) step         = *iPtr;
        const bool*    bPtr = attrs->GetBool(6);  if (bPtr != nullptr) maximize      = *bPtr;
    }
```

要点：

- **glue 里用注释显式锁定声明顺序**（索引 ↔ def.cpp 的 `Attr()` 顺序一一对应），属性顺序变更时 glue 必须同步——不锁注释的"凭记忆索引"会在属性列表变动后静默读错值。
- **本地默认值兜底**：attrs 指针为空或某项为空时落到本地默认值，不解引用空指针。
- 平台参数配套（同 glue 骨架）：`PlatformAscendC plat(context->GetPlatformInfo())` → `GetCoreNumAiv()` + `GetCoreMemSize(CoreMemType::UB, ubSize)`（ascend950 代表值约 196608B，再减栈/标量保留），dtype 特化放纯计算层（不支持的 dtype 返回 false → glue 返回 GRAPH_FAILED）。
- 作用域坑：`PlatformAscendC` 在辅助函数内构造时，`GetLibApiWorkSpaceSize()` 等方法须在同一作用域调用或经出参传出——在 TilingFunc 主体直接引用会报 `'ascendcPlatform' was not declared in this scope`。

坏实践（独立 trace 增补）：按属性名找读取接口（不存在该形态）；凭记忆写索引不锁注释——属性顺序一变静默读错值；在 glue 里直接写切分公式（无法被 UT 同源覆盖）。

验证信号（独立 trace 增补，ApplyAdamW 类算子公共 Tiling 实现会话，Ascend950PR）：

- 构建：`./build.sh <Op>_package` → `SUCCESS`，构建日志 grep `do not registe tiling struct` 0 命中（来源平台：950；证据等级：tool_observed）。
- 同源 TilingUT：17/17 通过（纯计算层与上板同一份源码）（来源平台：950；证据等级：tool_observed）。
- 回归：aclnn 冒烟 3/3（fp32/fp16/bf16 两段式调用 OK）（来源平台：950；证据等级：tool_observed）。
- 复验：CANN 9.1.0 环境下重新构建 + TilingUT 28/28（含 Branch-0 追加用例）通过（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：属性按索引读取是 CANN 9.x `RuntimeAttrs` 的形态；若框架未来提供按名读取，索引方案仍向后兼容但可读性差（未验证）；属性类型映射按 `.Float/.Int/.Bool` 三类实测，ListAttr/String 形态未在本增量覆盖。