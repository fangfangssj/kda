---
type: "Runbook"
title: op_host UT 工程：faker+源码 .so+全链旧 ABI
description: 从参考模板搭自定义算子 op_host Tiling/InferShape UT 的完整工程配方：op_host 源码编成 .so 经 OpImplSpaceRegistryV2/AddSoToRegistry 动态加载、全链旧 ABI（_GLIBCXX_USE_CXX11_ABI=0 + gtest 源码重编）、whole-archive 链 librt2_registry.a/libtiling_api.a、直链 libplatform.so；TilingContextFaker 构造上下文断言 TilingKey/BlockDim/TilingData 字段与 GroupSchedule；StorageShape 构造、compileInfo 非空、initializer_list 悬空、nlohmann 整目录四个坑各有对策。复现增量：nlohmann 三条处置路线与 faker 平台信息直接设置重写。复现增量：DYNAMIC_INPUT 用例的 inputInstanceNum 构造、common 从同机已验证工程复制、nlohmann 系统路径第 4 路线与 op_common 头位置。
tags: [task.testing, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-17T03:16:34Z'
platforms: ["950"]
---

# op_host UT 工程：faker+源码 .so+全链旧 ABI

## 现象

为自定义算子（registry-invoke 算子仓，Ascend950PR/arch35 目标）搭建 op_host UT 工程：目标是把 op_host 源码（def/infershape/arch35/tiling）在不依赖 NPU 的主机上直接跑 Tiling/InferShape 逻辑回归——给定输入 shape/dtype/属性，验证 TilingKey 编码、BlockDim、TilingData 字段与 GroupSchedule 是否与设计一致，以及 InferShape 对不同布局与非法输入的处理。参考样例（ascendc-registry-invoke-template 的 add_example tests/ut/）提供 faker/executor 框架，但其 CMake 假设 aarch64 与系统级依赖。两轮独立落地（CANN 9.1.0 与 9.2.0、两个不同算子）均一次跑通：11/11（InferShape 7 + Tiling 4）、迭代后 27/27 与 37/37（含 failed-as-expected 双路径与跨 dtype 独立 TEST_F）。

## 适用场景

- 工程形态：自定义算子 op_host 层 UT（Tiling 计算逻辑、参数校验、InferShape 推导），registry-invoke 算子仓，宿主机直接运行（无需 NPU）。
- API 路径：`tiling_context_faker.h`/`tiling_case_executor.h`（gert::TilingContextPara + ExecuteTiling 返回 TilingInfo 含 tilingKey/tilingData/blockNum）、`infershape_context_faker.h`；`gert::OppSoDesc` + `gert::OpImplSpaceRegistryV2`（AddSoToRegistry + SetSpaceRegistry）；`Ops::Math::AnyValue`（属性构造）；`libplatform.so`/`libtiling_api.a`/`librt2_registry.a`（$ASCEND_HOME_PATH/lib64）。
- 验证过的平台与 CANN 版本：Ascend950（BUILD_SOC_VERSION=Ascend950），宿主 x86_64，CANN 9.1.0 与 9.2.0（与 frontmatter `platforms` 一致；UT 级验证，950 目标算子）。
- 适用粒度：platform_wide——UT 工程配方与算子无关（换算子只改 OP_NAME/TILING_SRC 与用例表）。
- 路线优先级：recommended_fastpath（ST 前先锁 host 逻辑，成本低于上板调试）。
- 明确不适用 / 未验证场景：kernel/device 侧行为（须 ST/上板）；aclnn L2 层接口（另用 ST 直连）；aarch64 主机（include/lib 路径按 ${CMAKE_SYSTEM_PROCESSOR}-linux 自适应后理论可用，未验证）；CANN 9.2.0 之外的库布局未验证。

## 设计决策

- **ABI 策略（本配方核心）**：CANN 的 libplatform.so/libtiling_api.a 是旧 ABI（`_GLIBCXX_USE_CXX11_ABI=0`）构建，宿主系统 gtest/stdlib 默认新 ABI；混链即崩。选择"全链旧 ABI"：gtest 从源码以旧 ABI 重编（BuildGoogleTest.cmake），op_host .so 与 UT 可执行文件都显式定义 `_GLIBCXX_USE_CXX11_ABI=0`——放弃给 CANN 库做 stub 的路线（更复杂且易漂移）。
- **op_host 编成 `.so` 动态加载**而不是直接链进测试：注册宏（REG_OP/IMPL_OP_OPTILING/IMPL_OP_INFERSHAPE）在 .so 加载期执行，`AddSoToRegistry()` 走与真实部署一致的注册路径；直接链接会丢注册语义；.so 也方便单独替换被测实现。
- **whole-archive 链 librt2_registry.a/libtiling_api.a**：注册入口分散在静态库各目标文件，不 whole-archive 会被链接器裁掉，出现"注册不生效/加载 .so 时注册数为 0"类静默失败。
- **直链 libplatform.so**：本机就是 CANN 安装环境，无需 stub 间接层；`-Wl,--no-as-needed` 防 as-needed 裁剪。
- **属性顺序 = 属性全局索引**：faker 的 `OpAttr` 列表顺序必须与 def.cpp 的 `.ATTR(...)` 声明顺序一致（tiling 代码 `GetListInt(0)/GetInt(1)/GetBool(2)` 按全局索引取）；InferShape 不读的属性也要占位。
- **TilingKey 期望值独立计算**：按 `template_argument.h` 的 `FastEncodeTilingKeyDirect` 编码规则在测试里复算（DATATYPE 8bit 起始位 0、BOOL 1bit），dtype 直接用 `ge::DataType` 枚举值（`DT_FLOAT16=1`、`DT_BF16=27`）。
- **TilingData 断言前先校验容量**：`ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TilingData))` 再 `reinterpret_cast`，防止 faker 容量不足读到截断数据。
- **ExecuteTiling 返回 TilingInfo 结构体**而不是硬编码 expectTilingData 字符串比对：hTile 等字段随 dtype/UB 预算动态计算，字符串比对脆弱；按字段（tilingKey/blockNum/hTile/…）分别断言并可只对确定字段精确校验。
- 放弃的候选：直接在测试里手搓 TilingContext 结构（绕过 faker）——接口面太大；mock 模式 ST——覆盖不到 tiling 内部字段。

## 完整代码

### 1. UT 入口：加载 op_host `.so` 进注册表（`test_op_host_main.cpp`）

```cpp
#include <filesystem>
#include <iostream>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

// 参考 ops-math/tests/ut/op_host/test_op_host_main.cpp
class OpHostUtEnvironment : public testing::Environment {
public:
    OpHostUtEnvironment(char** argv) : argv_(argv)
    {}

    virtual void SetUp() {
        cout << "Global Environment SetUp." << endl;

        // 获取可执行文件路径
        std::filesystem::path currDir = argv_[0];
        if (currDir.is_relative()) {
            currDir = std::filesystem::weakly_canonical(std::filesystem::current_path() / currDir);
        } else {
            currDir = std::filesystem::canonical(currDir);
        }

        // 加载 lib<op>_op_host_ut_lib.so（包含算子注册代码）
        string opHostSoPath = currDir.parent_path().string() +
                              string("/libbench_no_harness_grouped_matmul_op_host_ut_lib.so");
        cout << "Loading op_host .so from: " << opHostSoPath << endl;

        // 创建 OppSoDesc 并加载 .so
        gert::OppSoDesc oppSoDesc({ge::AscendString(opHostSoPath.c_str())}, "op_host_so");

        // 创建 SpaceRegistry 并添加 .so 中的注册信息
        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cerr << "Failed to add .so to registry." << endl;
            return;
        }

        // 设置到全局 DefaultOpImplSpaceRegistryV2
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
        cout << "OpImplSpaceRegistryV2 initialized successfully" << endl;
    }

    virtual void TearDown() {
        cout << "Global Environment TearDown" << endl;
    }

private:
    char** argv_;
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpHostUtEnvironment(argv));
    return RUN_ALL_TESTS();
}
```

### 2. shape/上下文构造辅助（覆盖两个已知坑：StorageShape 双填 + compileInfo 非空）

```cpp
// DAV_3510 (Ascend950PR) 平台参数：UB=248KB=253952B；UT 取 coreNum=32（保守值）
static constexpr uint64_t DAV3510_CORE_NUM = 32UL;
static constexpr uint64_t DAV3510_UB_SIZE = 253952UL;  // 248 KB
static constexpr uint64_t TILING_DATA_MAX_SIZE = 4096UL;

// 空的 CompileInfo 结构体（与 op_host/tiling.cpp 中定义一致，入图场景依赖）
// 传非空地址避免框架内部访问 nullptr
struct BenchNoHarnessDepthwiseConv2dCompileInfo {};
static BenchNoHarnessDepthwiseConv2dCompileInfo g_compileInfo;

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    auto& storage = shape.MutableStorageShape();
    auto& origin = shape.MutableShape();
    storage.SetDimNum(dims.size());
    origin.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        storage.SetDim(i, dims[i]);
        origin.SetDim(i, dims[i]);
    }
    return shape;
}

static void BuildTilingContextPara(const TilingTestParam& param,
                                   gert::TilingContextPara& para)
{
    gert::StorageShape xStorageShape = MakeStorageShape(param.xShape);
    gert::StorageShape wStorageShape = MakeStorageShape(param.wShape);
    gert::StorageShape bStorageShape = MakeStorageShape(param.bShape);
    gert::StorageShape yStorageShape = MakeStorageShape(param.xShape);  // 占位，Tiling 重算

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc = {
        {xStorageShape, param.xDtype, ge::FORMAT_ND},
        {wStorageShape, param.xDtype, ge::FORMAT_ND},
        {bStorageShape, param.xDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc = {
        {yStorageShape, param.xDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"kernelSize", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(param.kernelSize)},
        {"stride",     Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(param.stride)},
        {"padding",    Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(param.padding)},
        {"dilation",   Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilation)},
        {"groups",     Ops::Math::AnyValue::CreateFrom<int64_t>(param.groups)},
    };

    para.~TilingContextPara();
    new (&para) gert::TilingContextPara(
        OP_NAME, inputTensorDesc, outputTensorDesc, attrs,
        &g_compileInfo,               // compileInfo：非空空结构体地址（传 nullptr 会段错误）
        param.coreNum, param.ubSize, TILING_DATA_MAX_SIZE);
}
```

（另一轮落地的等价写法：`StorageShape` 也可用 `MutableOriginShape().AppendDim(d)` + `MutableStorageShape().AppendDim(d)` 双填；CANN 9.1.0 头里该结构无 `SetDimNum/SetDim` 时用 AppendDim 版。）

### 3. Tiling UT 核心（参数结构 + 期望值复算 + 逐字段断言）

参数结构（**注意：shape 成员必须用 `std::vector`，不能用 `std::initializer_list`**，见坏实践）：

```cpp
struct TilingTestParam {
    std::string caseName;
    std::string desc;
    std::vector<int64_t> xShape;
    ge::DataType xDtype;
    std::vector<int64_t> wShape;
    std::vector<int64_t> bShape;
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    int64_t groups;
    uint64_t coreNum;
    uint64_t ubSize;
    ge::graphStatus expectStatus;      // GRAPH_SUCCESS / GRAPH_FAILED
    int64_t expectHOut = 0;            // 仅成功场景有效
    int64_t expectWOut = 0;
    int64_t expectNcTotal = 0;
    int64_t expectBlockFactor = 0;
    int64_t expectUsedCoreNum = 0;
};
```

期望 TilingKey 独立复算（编码规则见 `template_argument.h FastEncodeTilingKeyDirect`：DATATYPE 8bit 起始位 0、BOOL 1bit；fp16(DT_FLOAT16=1)+no bias+no tw → 1）：

```cpp
static uint64_t CalcExpectTilingKey(ge::DataType dtype, bool hasBias, bool transposeWeight) {
    uint64_t dTypeX = static_cast<uint64_t>(dtype);
    uint64_t hasBiasBit = hasBias ? 1ULL : 0ULL;
    uint64_t twBit = transposeWeight ? 1ULL : 0ULL;
    return (dTypeX << 0) | (hasBiasBit << 8) | (twBit << 9);
}
```

单个用例执行与断言（终版：维度从 vector 解析、shape 用 MakeStorageShape、属性顺序对齐 def.cpp；断言覆盖 TilingKey/BlockDim/TilingData 字段/GroupSchedule）：

```cpp
static void RunTilingCase(const TilingTestParam& param) {
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

    // ---------- 构造 TilingContextPara 并执行 ----------
    gert::TilingContextPara tilingContextPara(...);   // 按 BuildTilingContextPara 组装
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ok) << "ExecuteTiling failed for case: " << param.caseName;

    // ---------- 验证 1: TilingKey ----------
    EXPECT_EQ(tilingInfo.tilingKey, param.expectTilingKey)
        << "TilingKey mismatch. case=" << param.caseName
        << " actual=" << tilingInfo.tilingKey << " expect=" << param.expectTilingKey;

    // ---------- 验证 2: BlockDim == usedCoreNum ----------
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(param.expectUsedCoreNum))
        << "BlockDim mismatch. case=" << param.caseName;

    // ---------- 验证 3: TilingData 字段 ----------
    // 断言 TilingData 容量足够容纳结构体
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(BenchNoHarnessGroupedMatmulTilingData))
        << "TilingData size too small. case=" << param.caseName;

    const auto* tiling = reinterpret_cast<const BenchNoHarnessGroupedMatmulTilingData*>(tilingInfo.tilingData.get());

    // 算子级参数
    EXPECT_EQ(tiling->E, E) << "E mismatch. case=" << param.caseName;
    EXPECT_EQ(tiling->M, M) << "M mismatch. case=" << param.caseName;
    EXPECT_EQ(tiling->N, N) << "N mismatch. case=" << param.caseName;
    EXPECT_EQ(tiling->K, K) << "K mismatch. case=" << param.caseName;
    EXPECT_EQ(tiling->splitItem, param.splitItem) << "splitItem mismatch. case=" << param.caseName;
    EXPECT_EQ(tiling->usedCoreNum, param.expectUsedCoreNum) << "usedCoreNum mismatch. case=" << param.caseName;

    // ---------- 验证 4: group_list 解析 + maxMi ----------
    std::vector<int64_t> mList, startList;
    ParseGroupList(param.groupList, mList, startList);
    int64_t expectMaxMi = 0;
    for (int64_t g = 0; g < E; g++) {
        if (mList[g] > expectMaxMi) {
            expectMaxMi = mList[g];
        }
    }
    EXPECT_EQ(tiling->maxMi, expectMaxMi) << "maxMi mismatch. case=" << param.caseName;

    // ---------- 验证 5: GroupSchedule 填充 ----------
    std::vector<int64_t> expectCoreAssign = CalcExpectCoreAssign(mList, static_cast<int64_t>(ASCEND950PR_AIC_CORE_NUM));

    for (int64_t g = 0; g < E; g++) {
        const auto& gs = tiling->groupSchedule[g];
        EXPECT_EQ(gs.coreIdx, expectCoreAssign[g])
            << "groupSchedule[" << g << "].coreIdx mismatch. case=" << param.caseName;
        EXPECT_EQ(gs.m_i, mList[g])
            << "groupSchedule[" << g << "].m_i mismatch. case=" << param.caseName;
        // xOffset = start_g * K（x: [M, K] 行主序）
        EXPECT_EQ(gs.xOffset, startList[g] * K)
            << "groupSchedule[" << g << "].xOffset mismatch. case=" << param.caseName;
        // weightOffset = g * K * N（weight: [E, K, N] 第 g 个 expert）
        EXPECT_EQ(gs.weightOffset, g * K * N)
            << "groupSchedule[" << g << "].weightOffset mismatch. case=" << param.caseName;
        // biasOffset = g * N（bias: [E, N]）
        EXPECT_EQ(gs.biasOffset, g * N)
            << "groupSchedule[" << g << "].biasOffset mismatch. case=" << param.caseName;
        // yOffset = start_g * N（y: [M, N] 行主序，split_item=2/3）
        EXPECT_EQ(gs.yOffset, startList[g] * N)
            << "groupSchedule[" << g << "].yOffset mismatch. case=" << param.caseName;
    }

    // ---------- 打印 TilingKey 用于调试 ----------
    std::cout << "[RESULT] case=" << param.caseName
              << " tilingKey=" << tilingInfo.tilingKey
              << " blockNum=" << tilingInfo.blockNum
              << " usedCoreNum=" << tiling->usedCoreNum
              << " maxMi=" << tiling->maxMi
              << std::endl;
}

TEST_P(BenchNoHarnessGroupedMatmulTilingTest, tiling_test) {
    const TilingTestParam& param = GetParam();
    RunTilingCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    BenchNoHarnessGroupedMatmulTilingTests,
    BenchNoHarnessGroupedMatmulTilingTest,
    testing::ValuesIn(BuildTestCases()));
```

（`ParseGroupList`/`CalcExpectCoreAssign` 在测试里按 tiling 的贪心均衡规则独立复算期望值；`BuildTestCases` 内 `TilingTestParam` 用 `{4, 16}` 列表初始化自动构造 `std::vector<int64_t>`。）

### 4. InferShape UT 样例（`test_<op>_infershape.cpp`）

```cpp
/*!
 * \brief 构造属性列表（顺序与 def.cpp 声明一致）
 *   index 0: group_list (ListInt) - InferShape 不读，但需占位以保持索引
 *   index 1: split_item (Int)
 *   index 2: transpose_weight (Bool)
 */
static std::vector<gert::InfershapeContextPara::OpAttr> BuildAttrs(
    int64_t splitItem, bool transposeWeight, const std::vector<int64_t>& groupList = {1}) {
    return {
        {"group_list", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(groupList)},
        {"split_item", Ops::Math::AnyValue::CreateFrom<int64_t>(splitItem)},
        {"transpose_weight", Ops::Math::AnyValue::CreateFrom<bool>(transposeWeight)},
    };
}

// 用例2: split_item=2，transpose_weight=true，weight=[E,N,K]，输出 [M, N]
TEST_F(BenchNoHarnessGroupedMatmulInfershape, split2_tw_fp16) {
    // x: [M=8, K=64], weight: [E=3, N=128, K=64]
    InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // x
            {{{3, 128, 64}, {3, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // weight [E,N,K]
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        BuildAttrs(/*splitItem=*/2, /*transposeWeight=*/true, /*groupList=*/{3, 6, 8}));

    // 预期输出 y: [M=8, N=128]（tw=true → N=dim(1)）
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 用例6: x 非 2D（1D），应返回 GRAPH_FAILED
TEST_F(BenchNoHarnessGroupedMatmulInfershape, invalid_x_1d) {
    InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // x: 1D，非法
            {{{2, 16, 32}, {2, 16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        BuildAttrs(/*splitItem=*/2, /*transposeWeight=*/false, /*groupList=*/{8, 16}));

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
```

### 5. `tests/ut/op_host/CMakeLists.txt` 关键段（旧 ABI + .so + whole-archive）

```cmake
set(OP_NAME bench_no_harness_depthwise_conv_2d)
set(UT_EXE ${OP_NAME}_op_host_ut)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend950")
endif()

set(OP_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host)
set(OP_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel)
set(TILING_SRC ${OP_HOST_DIR}/arch35/${OP_NAME}_tiling.cpp)
set(TILING_DATA_H ${OP_KERNEL_DIR}/arch35/${OP_NAME}_tiling_data.h)

set(OP_HOST_SRCS
    ${OP_HOST_DIR}/${OP_NAME}_def.cpp
    ${OP_HOST_DIR}/${OP_NAME}_infershape.cpp
    ${TILING_SRC}
)
set(UT_TEST_SRCS
    test_${OP_NAME}_infershape.cpp
    test_${OP_NAME}_tiling.cpp
)
set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_host_main.cpp)

# op_host 编成 SHARED .so：包含全部注册代码，运行期经 AddSoToRegistry() 加载
set(OP_HOST_SO ${OP_NAME}_op_host_ut_lib)
add_library(${OP_HOST_SO} SHARED ${OP_HOST_SRCS})
target_include_directories(${OP_HOST_SO} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS} ${OP_HOST_DIR} ${OP_HOST_DIR}/arch35 ${OP_KERNEL_DIR}/arch35)
target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0)          # 旧 ABI 匹配 libplatform.so / libtiling_api.a
target_link_libraries(${OP_HOST_SO} PRIVATE
    -Wl,--no-as-needed
    -Wl,--whole-archive                # 注册宏在静态库目标文件里，必须 whole-archive
        $ENV{ASCEND_HOME_PATH}/lib64/librt2_registry.a
        $ENV{ASCEND_HOME_PATH}/lib64/libtiling_api.a
    -Wl,--no-whole-archive
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so)

add_executable(${UT_EXE} ${UT_MAIN_SRC} ${UT_TEST_SRCS} ${UT_COMMON_SRCS})
target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS} ${OP_HOST_DIR} ${OP_HOST_DIR}/arch35 ${OP_KERNEL_DIR}/arch35)
target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0)          # UT 可执行文件同样旧 ABI
target_compile_options(${UT_EXE} PUBLIC -fPIE -fno-access-control)
target_link_directories(${UT_EXE} PRIVATE ${UT_COMMON_LIB_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${UT_EXE} PRIVATE
    gtest gtest_main                    # 源码重编的旧 ABI gtest
    -Wl,--no-as-needed
    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so   # 直链，不用 stub
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
    dl pthread)
# 另需设置 RPATH 指向 .so 构建输出目录与 $ASCEND_HOME_PATH/lib64
set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE)
```

根 `tests/ut/CMakeLists.txt` 负责：校验 `ASCEND_HOME_PATH`、`include(cmake/BuildGoogleTest.cmake)`（旧 ABI 编 gtest）、聚合 `common/` faker 源文件（infershape_case_executor/infershape_context_faker/tiling_case_executor/tiling_context_faker 四个 .cpp）、设置 CANN include 目录集、`add_compile_options(-fno-access-control)`、`add_subdirectory(op_host)`。架构自适应：`set(ARCH_DIR "${CMAKE_SYSTEM_PROCESSOR}-linux")`，include 路径覆盖 include、`${ARCH_DIR}/include`、`${ARCH_DIR}/include/aclnn`、`${ARCH_DIR}/asc/include`、`${ARCH_DIR}/pkg_inc` 全家。

### 6. `run.sh` 执行链

```bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true
SOC_VERSION="Ascend950"

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "错误: 未设置 ASCEND_HOME_PATH 环境变量"
    exit 1
fi
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
make -j$(nproc)

cd "${BUILD_DIR}/op_host"
if ./<op>_op_host_ut; then
    echo "[PASS] op_host 测试通过"; exit 0
else
    echo "[FAIL] op_host 测试失败"; exit 1
fi
```

## 工程搭建步骤（按序五步，每步的坑与对策）

1. **拷框架**：从参考模板复制 tests/ut/{cmake,common,op_host} 骨架；common 依赖的 nlohmann json 从 `$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include/` **整目录**复制（非单头版本，只拷 json.hpp 不够）。
2. **写顶层 CMakeLists**：`ARCH_DIR=${CMAKE_SYSTEM_PROCESSOR}-linux` 自适应；include 路径覆盖 CANN 全家；先 include BuildGoogleTest.cmake（旧 ABI gtest）。
3. **写 op_host/CMakeLists**：op_host 源码编 SHARED `.so`，`BUILD_SOC_VERSION` + `_GLIBCXX_USE_CXX11_ABI=0`，whole-archive 链 librt2_registry.a/libtiling_api.a；UT 可执行文件同旧 ABI，直链 libplatform.so 等动态库并设 RPATH。
4. **写用例**：TilingTestParam 参数化（成功/失败同表），MakeStorageShape 造 shape，BuildTilingContextPara 传非空 compileInfo 地址；断言按功能契约（互异/字段值）而非实现细节。
5. **跑**：`bash tests/ut/run.sh`（clean）或 `-c false`（增量）；单用例 `--gtest_filter='*verify_tiling_logic/0'` + `LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64`。

## 预防（四个已踩实坑的 checklist）

1. `gert::StorageShape` 无 `GetDim`/不接受单 initializer_list：取维度用 `xShape.GetStorageShape().GetDim(0)`；从 `std::vector<int64_t>` 造 shape 用 `MakeStorageShape`（Mutable 双填或 AppendDim），不要用单个 initializer_list 构造（`{x, y}` 会被当作 origin 一个维度）。
2. `TilingContextPara` 的 compileInfo 不能传 nullptr（ExecuteTiling 无声段错误）：定义静态空 CompileInfo 结构体传地址。
3. **faker 测试参数结构体不要用 `std::initializer_list` 成员存 shape**：`initializer_list` 不拥有底层数组，`cases.push_back({... {2, 16, 16} ...})` 里的临时数组在表达式结束销毁，成员变成悬空引用——tiling 读到 `E=94055876119648` 这类垃圾值（本质是指针值），`ExecuteTiling` 返回 false，且伴随 `GetSizeInBytes failed, element_count:-9223372036854775808 less than 0.` 告警。用 `std::vector<int64_t>` 成员。
4. common 框架的 `tiling_case_executor.cpp` 依赖 nlohmann json，系统无装时从 CANN python site-packages 整目录复制 nlohmann/。

## 验证信号

来源说明：两个独立算子（grouped_matmul / depthwise_conv_2d）的 op_host UT 工程搭建会话，各含初版修复后全量回归。

- 编译/配置信号：CMake 输出 "=== Google Test Build Configuration (ops-math mode) === ABI: OLD (_GLIBCXX_USE_CXX11_ABI=0) / Reason: Match CANN libraries (libplatform.so, libtiling_api.a)" 与 "Direct Linking (No Stub)"（来源平台：950；证据等级：tool_observed）
- 初版编译错误（CANN 9.1.0 轮）：`error: 'struct gert::StorageShape' has no member named 'GetDim'`（多处）；改为 `GetStorageShape().GetDim()` 后编译通过（来源平台：950；证据等级：tool_observed）
- 初版运行失败（CANN 9.1.0 轮）：Tiling 4 例 `ExecuteTiling` 返回 false，plog `group_list length 2 != E 94055876119648`（垃圾值）+ `GetSizeInBytes failed, element_count:-9223372036854775808`；参数结构体改 `std::vector` + `MakeStorageShape`（AppendDim）后通过（来源平台：950；证据等级：tool_observed）
- 终版（CANN 9.1.0 轮）：11/11 通过——`[  PASSED  ] 11 tests.`；Tiling 用例输出 `tilingKey=1 blockNum=usedCoreNum maxMi` 全部符合预期（如 `case=fp16_no_bias_no_tw_split2_small tilingKey=1 blockNum=2 usedCoreNum=2 maxMi=2`）；run.sh 全流程（清理→cmake→make→运行）复现通过（来源平台：950；证据等级：tool_observed）
- 独立复现（CANN 9.2.0 轮）：迭代一 UT 27/27（10 InferShape + 6 TilingSuccess + 11 TilingFailure failed as expected）；迭代二同一工程扩到 37/37（新增 fp32/bf16 成功用例 8 条、负向用例改测真正不支持的 int8/uint8、跨 dtype 独立 TEST_F 两条：TilingKey 互异 + UB 预算 hTile 关系 bf16(360)<fp16(466)<fp32(528)）；单用例 `--gtest_filter='*verify_tiling_logic/0'` 输出 tilingKey=1/blockNum=32/hTile=56 且 EXIT=0（来源平台：950；证据等级：tool_observed）

## 失效边界

- libplatform/libtiling_api 的旧 ABI 约定以 CANN 9.1.0/9.2.0 为准；CANN 改新 ABI 构建后 `_GLIBCXX_USE_CXX11_ABI=0` 全链策略需同步翻转（未验证）。
- 平台参数（coreNum=32、ubSize=253952）按 SoC 替换；换平台须同步改 faker 的 coreNum/ubSize 与期望核数。UT 验证的是 op_host 逻辑在给定平台参数下的行为，不覆盖 kernel 执行与真实平台的 platformInfo 差异。
- faker 框架文件来自模板算子工程的 `common/`（含 `Ops::Math::AnyValue` 等），跨工程复制时命名空间若不同需适配。
- 本工程 SOC_VERSION 仅支持 Ascend950 分支；多 SOC 算子需扩 TILING_SRC 选择逻辑（未实现）。
- 未覆盖 tiling 的 `SetWorkspaceSizes` 声明路径（该轮 workspace=0）；含用户 workspace 的算子可按本配方扩展断言 workspaces 数组（未尝试）。

**未决项**：

- tiling_context_faker 的 platformInfo 注入（SoCInfo/AICoreSpec/SetCoreNumByCoreType）对 GetCoreNumAiv 等 PlatformAscendC 接口的覆盖面未逐项验证（本算子 tiling 用到的 coreNum/ubSize 路径已验证）。
## 坏实践

- 用系统包管理器装的 gtest（新 ABI）直链 CANN 旧 ABI 库——运行期崩溃或符号解析失败。
- op_host 直接链进测试可执行文件（丢注册路径）；链 `librt2_registry.a`/`libtiling_api.a` 不加 whole-archive（注册符号被裁剪，加载 .so 时注册数为 0）。
- 测试参数结构体用 `std::initializer_list` 成员（悬空 → 垃圾维度值，见预防）。
- 在 `gert::StorageShape` 上直接调 `GetDim`（编译错）或用单个 initializer_list 构造。
- 用 expectTilingData 整串字符串硬编码比对 tiling 输出——动态字段（hTile 随 dtype 变）让用例脆断。

## 误诊

- 垃圾维度值出现时曾先怀疑 faker 的 `InputTensors`/`NodeIoNum` 索引偏移与 Tensor 构造——排查后确认传入的 StorageShape 本身就是悬空数据（initializer_list 生命周期），与 faker 无关。

## 增量（nlohmann 依赖的中间失败证据与 include 路径替代路线）

另一独立轨迹并入两条独立 trace 的 nlohmann 依赖处置细节（MlaProlog 类 / DepthwiseConv2d 类算子 UT，CANN 9.2.0），补齐本卡预防清单第 4 条的证据与替代路线：

**1. 多头版本的中间失败证据（只拷 json.hpp 不够的直接证明）**：从 `$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include/nlohmann/` 只拷 `json.hpp` 一个文件后，错误变为 `fatal error: nlohmann/adl_serializer.hpp: No such file or directory`——CANN 内置副本是**多头文件版本**（json.hpp 内部还 `#include <nlohmann/adl_serializer.hpp>`、`detail/…` 等），不是常见的 single_include 单头版。复制前先 `ls` 源目录：有 `adl_serializer.hpp`/`detail/` 即多头版，必须 `cp -r` 整目录：

```bash
# 依赖定位（先找再拷）
find "$ASCEND_HOME_PATH" -name "json.hpp" -path "*/nlohmann/*" 2>/dev/null | head -5
find /usr -name "json.hpp" -path "*/nlohmann/*" 2>/dev/null   # 确认系统级没有
# 整目录复制（只拷 json.hpp 会报 adl_serializer.hpp 缺失）
cp -r $ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include/nlohmann \
      tests/ut/common/nlohmann
```

**2. include 路径替代路线（不复制文件）**：把 CANN 内置副本目录直接加进 UT 的 include 路径（`tests/ut/CMakeLists.txt` 的 `UT_COMMON_INCLUDE_DIRS` 追加一行，verbatim）：

```cmake
    # nlohmann/json.hpp for tiling_case_executor
    $ENV{ASCEND_HOME_PATH}/python/site-packages/pypto/lib/framework/3rd/include
```

两条路线取舍：复制进 `tests/ut/common/` 使 UT 自包含、可随仓分发，不暴露 pypto 头树；加 include 路径零拷贝、始终跟随 CANN 内置版本，但 python site-packages 路径随 CANN 升级/重装漂移，且把整个 pypto framework 头树暴露进编译 include。

**3. 版本漂移坏实践**：直接 `apt install nlohmann-json3-dev` 引入与 CANN 副本不同版本的头——faker JSON 解析行为可能与 CANN 库预期不一致；优先用同源副本。

### 复现增量验证信号（nlohmann 路线）

- 修复前：`fatal error: nlohmann/json.hpp: No such file or directory`（tiling_case_executor.cpp:14）（来源平台：950；证据等级：tool_observed）。
- 中间失败：只拷 json.hpp 后错误变为 `fatal error: nlohmann/adl_serializer.hpp: No such file or directory`，证明其为多头版本（来源平台：950；证据等级：tool_observed）。
- 整目录复制后编译通过，UT 全量 27/27（InferShape 10/10 首先跑通，随后 Tiling 修复后 27/27）（来源平台：950；证据等级：tool_observed）。
- include 路径路线：`UT_COMMON_INCLUDE_DIRS` 补充后重编译 `[100%] Built target <op>_op_host_ut`，nlohmann 错误消除，InferShape 6/6 通过（来源平台：950；证据等级：tool_observed）。

## 增量（faker 平台信息直接设置重写：消除 json 依赖 + vector_core_cnt）

另一独立轨迹并入第三条处置路线（LSTM 类算子 UT，CANN 9.2.0）：不引入 nlohmann，把参考模板 `tiling_case_executor.cpp` 的 `GetPlatFormInfos`（"先拼 JSON 字符串再解析回 socInfos/aicoreSpec/intrinsics"）重写为**直接设置**——值本来就在 `TilingContextPara` 手里，JSON 序列化/反序列化往返纯属冗余；且 Python site-packages 路径作为 C++ 编译依赖不可靠。

重写实现（verbatim 取自 `tests/ut/common/tiling_case_executor.cpp` 核心部分）：

```cpp
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

// SOC 版本到架构的映射
static const std::map<std::string, std::string> SOC_TO_ARCH = {
    {"Ascend310P", "2002"},
    {"Ascend910B", "2201"},
    {"Ascend910_93", "2201"},
    {"Ascend950", "3510"},
    {"Ascend910", "1001"}
};

// SOC 小写到大写的映射
static const std::map<std::string, std::string> SOC_TO_UPPER = {
    {"ascend910b", "Ascend910B"},
    {"ascend910_93", "Ascend910_93"},
    {"ascend950", "Ascend950"},
    {"ascend310p", "Ascend310P"},
    {"ascend910", "Ascend910"},
    {"ascend310b", "Ascend310B"}
};

// DAV_3510 (Ascend950) 默认硬件参数
struct DefaultHardwareInfo {
    uint64_t ubSize;
    uint64_t l2Size;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t coreNum;
};

static DefaultHardwareInfo GetDefaultHardwareInfo()
{
    DefaultHardwareInfo info;
    std::string buildSocVersion = STR(BUILD_SOC_VERSION);
    auto it = SOC_TO_UPPER.find(buildSocVersion);
    if (it != SOC_TO_UPPER.end()) {
        buildSocVersion = it->second;
    }

    if (buildSocVersion == "Ascend950") {
        // DAV_3510 / arch35
        info.ubSize = 253952;   // 248 KB
        info.l2Size = 33554432; // 32 MB
        info.l1Size = 524288;   // 512 KB
        info.l0aSize = 65536;   // 64 KB
        info.l0bSize = 65536;   // 64 KB
        info.l0cSize = 262144;  // 256 KB
        info.coreNum = 48;
    } else {
        // 默认 (Ascend910B / DAV_2201)
        info.ubSize = 262144;   // 256 KB
        info.l2Size = 33554432; // 32 MB
        info.l1Size = 1048576;  // 1 MB
        info.l0aSize = 65536;   // 64 KB
        info.l0bSize = 65536;   // 64 KB
        info.l0cSize = 262144;  // 256 KB
        info.coreNum = 32;
    }
    return info;
}

// 直接设置平台信息 (避免 nlohmann/json 依赖)
static void SetPlatFormInfos(const gert::TilingContextPara& para,
                             std::map<std::string, std::string>& socInfos,
                             std::map<std::string, std::string>& aicoreSpec,
                             std::map<std::string, std::string>& intrinsics,
                             std::string& socVersionStr,
                             std::string& npuArchStr)
{
    // 解析 SOC 版本
    std::string buildSocVersion = STR(BUILD_SOC_VERSION);
    auto it = SOC_TO_UPPER.find(buildSocVersion);
    if (it != SOC_TO_UPPER.end()) {
        socVersionStr = it->second;
    } else {
        // 如果 BUILD_SOC_VERSION 已经是大写格式, 直接使用
        socVersionStr = buildSocVersion;
    }

    auto archIt = SOC_TO_ARCH.find(socVersionStr);
    npuArchStr = (archIt != SOC_TO_ARCH.end()) ? archIt->second : "";

    // 获取硬件信息
    auto hw = GetDefaultHardwareInfo();
    // 优先使用 TilingContextPara 传入的值
    uint64_t ubSize = (para.ubSize_ > 0) ? para.ubSize_ : hw.ubSize;
    uint64_t coreNum = (para.coreNum_ > 0) ? para.coreNum_ : hw.coreNum;

    // SoCInfo
    socInfos["core_type_list"] = "AICore";
    socInfos["ai_core_cnt"] = std::to_string(coreNum);
    socInfos["cube_core_cnt"] = std::to_string(coreNum);
    socInfos["vector_core_cnt"] = std::to_string(coreNum);  // AIV 模式需要
    socInfos["l2_size"] = std::to_string(hw.l2Size);

    // AICoreSpec
    aicoreSpec["ub_size"] = std::to_string(ubSize);
    aicoreSpec["l0_a_size"] = std::to_string(hw.l0aSize);
    aicoreSpec["l0_b_size"] = std::to_string(hw.l0bSize);
    aicoreSpec["l0_c_size"] = std::to_string(hw.l0cSize);
    aicoreSpec["l1_size"] = std::to_string(hw.l1Size);
    aicoreSpec["bt_size"] = "0";
    aicoreSpec["load3d_constraints"] = "1";
    aicoreSpec["cube_freq"] = "cube_freq";

    // Intrinsics (DAV_3510 向后兼容 DAV_2201)
    intrinsics["Intrinsic_data_move_l12ub"] = "float16";
    intrinsics["Intrinsic_data_move_l0c2ub"] = "float16";
}
```

`DO_TILING` 宏中的调用点（构建 faker 后覆盖平台信息）：

```cpp
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // ...（contextFaker 构建输入/输出/attr，与参考工程一致）...
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::string socVersionStr;
    std::string npuArchStr;
    SetPlatFormInfos(tilingContextPara, socInfos, aicoreSpec, intrinsics, socVersionStr, npuArchStr);
    std::map<std::string, std::string> socversions = {
        {"NpuArch", npuArchStr}, {"Short_SoC_version", socVersionStr}};
    auto tilingContext = contextHolder.GetContext();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);
```

两个关键修正：① 参考实现只设 `ai_core_cnt`，**纯 AIV/Vector 核算子的 tiling（`GetCoreNumAiv()`）依赖 `vector_core_cnt`，必须一并设置**；② 默认硬件表是近似值（未与芯片手册逐项核对），`TilingContextPara` 显式传入优先。直接设置法与 [自定义算子 TilingFunc 无设备 gtest 单测范式卡](tilingfunc_no_device_gtest_registry_mock_oracle.md) 的 mock 平台注入（SetPlatformRes 三张 map）同构。需要解析**外部传入的任意 JSON** compileInfo 的场景不适用本重写。

### 复现增量验证信号（直接设置重写）

- 依赖核查：`find` 确认 `nlohmann/json.hpp` 仅存在于 `$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include`，系统与 CANN 标准 include 树无（来源平台：950；证据等级：tool_observed）。
- 重写后 UT 编译通过（无 nlohmann 引用），全量 26/26 通过（InferShape 8 + Tiling 18），`bash run.sh` 清理重建复跑仍 26/26（来源平台：950；证据等级：tool_observed）。
- 注意：`SetPlatFormInfos` 设 48 核不等于 `GetCoreNumAiv()` 返回 48（实测 34，另卡）；本增量只证明直接设置法能让 tiling UT 全量运行（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 副本路径 `python/site-packages/pypto/lib/framework/3rd/include` 是 CANN 9.2.0 目录布局；其他版本/安装方式路径可能变化（用 find 定位）；属 CANN 内部打包，未来版本可能移位或移除（未验证）。
- 直接设置法的默认硬件表为该环境近似值；新 SOC 需自行扩 `SOC_TO_UPPER`/`SOC_TO_ARCH` 与默认参数分支。

## 增量（DYNAMIC_INPUT 用例的 inputInstanceNum 构造与 x86 环境事实）

另一独立轨迹并入两个独立 trace（TensorList 多张量算子 / 自适应池化类算子的 op_host UT 搭建，CANN 9.1.0、x86_64 主机、faker/executor 框架）的增量：

**1. DYNAMIC_INPUT（TensorList）用例的 TilingContextPara 构造**：`TilingContextPara` 提供带 `inputInstanceNum/outputInstanceNum` 向量的重载（每个 IR 输入/输出的实例数），faker 的 `IrInstanceNum`/`DO_TILING` 宏按它展开动态输入：

```cpp
// 3 个动态输入 (x1/x2/x3) 每个各 listLen 个实例; 1 个动态输出 (y) listLen 个实例
vector<uint32_t> inputInstanceNum = {
    static_cast<uint32_t>(listLen),
    static_cast<uint32_t>(listLen),
    static_cast<uint32_t>(listLen)
};
vector<uint32_t> outputInstanceNum = {static_cast<uint32_t>(listLen)};

return TilingContextPara(OP_NAME, inputDesc, outputDesc, attrs,
                         inputInstanceNum, outputInstanceNum,
                         &g_compileInfo, CORE_NUM, UB_SIZE, TILING_DATA_MAX_SIZE);
```

（`inputDesc/outputDesc` 按每实例一条 TensorDescription 展开，shape 用 MakeStorageShape 从 vector 构造；单张量用例仍走无 instanceNum 的重载，静态 shape 用 initializer_list 直构 `StorageShape shape = {inputShape, inputShape};`。）

**2. common faker 框架的复制来源**：优先从**同机已验证工程**整目录复制 `tests/ut/common/`（faker 与 CANN 版本的 binary 接口强耦合，复制已跑通的版本最稳，不要按模板重写），同时复制其 `cmake/BuildGoogleTest.cmake`：

```bash
mkdir -p tests/ut/{common,cmake,op_host}
SRC=<参考工程>/tests/ut/common; DST=tests/ut/common
for f in any_value.h tiling_context_faker.h tiling_context_faker.cpp \
         tiling_case_executor.h tiling_case_executor.cpp \
         infershape_context_faker.h infershape_context_faker.cpp \
         infershape_case_executor.h infershape_case_executor.cpp; do
    cp "$SRC/$f" "$DST/$f"
done
cp <参考工程>/tests/ut/cmake/BuildGoogleTest.cmake tests/ut/cmake/
```

**3. x86_64 安装布局与环境三事实**（x86_64 开发机，CANN 9.1.0）：

- `x86_64-linux` 为实目录，`include`/`lib64`/`aarch64-linux` 等为指向它的符号链接——模板里 aarch64 路径恰好因 symlink 而碰巧可用；CMake 用 `set(ASCEND_ARCH_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux)` 自适应（未实测 aarch64 主机）。
- GTest 1.14.0 须源码 old-ABI 构建：CMake 输出 `Building Google Test from source with OLD ABI ... Target ABI: _GLIBCXX_USE_CXX11_ABI=0`（Reason: System gtest uses new ABI, incompatible with libplatform.so）。
- nlohmann/json 第 4 条路线（与本卡先前增量 三条并列）：CANN 标准 include 树无 json.hpp 时，`find_path` 从系统路径取（miniconda/opensource/json 等 HINTS，找不到只 WARNING 不 FATAL——缺它时 `tiling_case_executor.cpp` 编不过但错误信息可读）：

```cmake
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS /home/<user>/miniconda3/include /usr/include /usr/local/include
    NO_DEFAULT_PATH)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
endif()
```

- op_common 头（`log/log.h`、`op_host/util/math_util.h` 等）在 `$ASCEND_HOME_PATH/x86_64-linux/include/op_common/`，include 集需覆盖该目录。

**4. 平台参数进测试常量**：UB_SIZE/CORE_NUM 按 SoC 写死（如 DAV_3510：UB=248KB=253952、32 AIV 核），期望值（如 ubFactor=15872）在用例注释里给出推导，平台变化时一望即知要改哪。

### 复现增量验证信号

来源说明：两个独立算子的 op_host UT 工程搭建与扩展会话（TensorList 多张量 / 自适应池化类）。

- TensorList trace：搭建完成首轮 `[ PASSED ] 14 tests`（含 dtype 失败校验、空 shape、标量输入边界）；OpDef DYNAMIC 化后测试改用 inputInstanceNum 构造重编译 `[ PASSED ] 14 tests`；多张量扩展后 `[ PASSED ] 21 tests`（listLen=2/4/8、不同 shape、双缓冲路径）（来源平台：950；证据等级：tool_observed）
- 池化类 trace：CMake 配置输出 `ASCEND_ARCH_DIR: x86_64-linux`、`Building Google Test from source with OLD ABI ... Target ABI: _GLIBCXX_USE_CXX11_ABI=0`、`Found nlohmann/json at: /home/<user>/miniconda3/include`；编译完成 `[ PASSED ] 16 tests.`（6 Infershape + 10 Tiling）；迭代二同一框架扩用例至 35/35（29 Tiling + 6 Infershape）（来源平台：950；证据等级：tool_observed）

### 复现增量失效边界

- inputInstanceNum 重载按 CANN 9.1.0 faker 框架验证；其他版本框架接口未验证。
- UT 通过不等于联调通过：host 侧 UT 只覆盖 tiling/infershape 逻辑，kernel 侧缺陷（如 DataCopyPad 对齐、上采样特化路径）不会被 UT 拦截。
- aarch64 开发机上同配方未实测；`BuildGoogleTest.cmake` 的 gtest 下载源可用性与离线构建未验证。
- nlohmann 系统路径因机器而异；无任何系统副本的机器需先安装或改走 先前增量 的 CANN 内置副本/include 路径/直接设置路线。

## 增量（四条新 trace 复现与两条工程决策补充）

另一独立轨迹并入四个独立 trace（GQA 注意力 / resize 双线性 / gcd / GRU 类算子的 op_host UT 搭建，CANN 9.1.0 与 9.2.0、faker/executor 框架从 add_example 或同机已验证工程复制）的同配方复现，并补两条工程决策：

**1. 源码直编 vs 已构建算子包 .so 的取舍**：本卡配方（op_host 源码编 UT 专用 .so）被明确决策保留——联调阶段算子包频繁变动，直编源码反馈环更短，且被测代码与仓库源保持单一事实源；验收/部署形态一致性要求高的场景改走"构建 run 包 → 安装 → 对安装树二次构建 gtest"闭环，见库内 [自定义算子 TilingFunc 无设备 gtest 单测范式](tilingfunc_no_device_gtest_registry_mock_oracle.md)（两种形态互补，勿混用：后者明令 gtest 只见安装树）。

**2. op_api 参数校验 UT 变体**（GRU trace，CANN 9.2.0）：同一 tests/ut 工程可加 `op_api/` 子目录测 aclnn 两段式接口的参数校验路径——`TestTensorFactory` 造 aclTensor/TensorList 实参（含 nullptr/长度/dtype 失配用例），断言 `aclnn<Op>GetWorkspaceSize` 返回 `ACLNN_ERR_PARAM_NULLPTR` 等；清理函数只 DestroyTensor。真实计算路径依赖 NPU，属 ST 范围（同思路的完整配方见库内 [自定义算子 host 侧 UT 工程配方（无 NPU）](custom_operator_host_ut_recipe_no_npu.md)）。

**3. include 树完整清单（x86_64 宿主，CANN 9.1.0 逐项 find 确认）**：`${ASCEND_HOME_PATH}/include`（根级符号链接）+ `${ARCH_DIR}/include` 及其 `include/aclnn`、`include/aclnn/opdev`、`include/base`、`include/base/context_builder`、`include/platform`、`include/op_common` + `${ARCH_DIR}/asc/include` 及其 `asc/include/tiling`、`asc/include/register`、`asc/include/adv_api` + `${ARCH_DIR}/pkg_inc` 及其 `pkg_inc/{op_common,base,exe_graph,graph}`；`/usr/include` 供系统 nlohmann（faker 的 tiling_case_executor 依赖，三个 trace 用系统 nlohmann-json3-dev 通过）。参考样例硬编码 `aarch64-linux` 路径在 x86_64 机器上要全部替换（见库内 [编译报 dlog_pub.h 缺失：模板硬编码 aarch64 路径](../compilation/template_aarch64_hardcode_x86_64_host.md)）；faker 头按裸文件名拉 context_builder 头的缺目录形态见库内 [UT 编译缺 context_builder include 路径](../compilation/ut_context_builder_include_path.md)。

**4. "UT 全绿 ≠ 算子正确"的边界实例**：GRU trace 的 op_host/op_api UT 共 31/31 全绿后，同算子随后的 ST 上板全部崩溃（kernel 507015）——UT 只覆盖 host 侧逻辑（tiling 字段/TilingKey 编码/infershape/aclnn 校验），kernel 行为必须 ST/上板（与本卡先前增量 边界互为印证，本例为 507035 具体形态）。

### 复现增量验证信号

来源说明：四个独立算子的 op_host UT 工程搭建会话（GQA 12 用例 / resize 双线性 25→47 用例 / gcd 27 用例 / GRU 31 用例），各含清理重建复跑。

- GQA trace（CANN 9.1.0）：CMake 配置成功（`ASCEND_HOME_PATH`/`ASCEND_ARCH_DIR` 正常打印，无头文件缺失）；编译成功（仅 deprecation warning）；首跑 `[ PASSED ] 12 tests.`（op_host .so 经注册链正常加载，`OpImplSpaceRegistryV2 initialized successfully`）；`run.sh` 端到端 `编译成功 → [ PASSED ] 12 tests. → 测试结果: PASS`；清空 build 目录重建复跑仍 12/12（幂等，无缓存依赖）（来源平台：950；证据等级：tool_observed）。
- resize 双线性 trace（CANN 9.1.0）：修复 StorageShape 构造与 TilingKey 断言两个问题后 25/25 通过（clean build 与增量构建均验证）；联调回归期间 UT 作为回归门禁多次重跑保持 25/25 无环境性抖动；扩展到 27 Tiling + 20 InferShape = 47 条后 47/47（来源平台：950（ascend950 host 代码，x86 宿主机执行）；证据等级：tool_observed）。
- gcd trace（CANN 9.1.0）：路径探针 `find` 确认头文件在 `x86_64-linux/include`、`libtiling_api.a`/`librt2_registry.a`/`libplatform.so` 在 `$ASCEND_HOME_PATH/lib64`、`nlohmann/json.hpp` 在 `/usr/include`；`cmake ..` 配置成功、`make -j` 编译通过（仅 CANN 头文件 `#pragma message` 弃用告警）；`[==========] 27 tests from 2 test suites ran. [ PASSED ] 27 tests.`（InferShape 13 + Tiling 14）；`bash run.sh` 清理重建后同样 27/27（来源平台：950（宿主 x86_64 工具链）；证据等级：tool_observed）。
- GRU trace（CANN 9.2.0）：CMake 配置输出确认 gtest 以旧 ABI 构建（`Reason: System gtest uses new ABI, incompatible with libplatform.so / Target ABI: _GLIBCXX_USE_CXX11_ABI=0 (old ABI)`）；op_host UT `[  PASSED  ] 14 tests.`（tiling 9 + infershape 5）；op_api UT `[  PASSED  ] 17 tests.`（参数校验路径）；run.sh 端到端汇总 PASS（31/31）；随后 ST 上板 kernel 507015 全崩（UT 边界实例，见上）（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- include 路径清单与 CANN 9.1.0/9.2.0 安装布局绑定（尤其 pkg_inc 与 aclnn/opdev 子树），版本升级需重验。
- op_api UT 只覆盖参数校验路径（nullptr/长度/dtype）；aclnn 真实计算路径依赖 NPU，属 ST 范围。
- aarch64 宿主（CMake 探测分支已就绪）与其他 ascend SOC_VERSION（换 SOC_VERSION 变量即可）未验证，推断可迁移。

### 复现增量坏实践

- 照抄参考样例的 `aarch64-linux` 路径到 x86_64 宿主——头文件全部找不到。
- 把 op_host .so 与 UT 可执行文件混成一套链接配置——.so 需要 whole-archive 静态注册库，可执行文件需要 gtest/动态运行库，二者需求不同。
- 跳过 main 里的 `OpImplSpaceRegistryV2` 注册直接跑用例——`ExecuteTiling` 找不到算子实现。
- 用 UT 全绿推断算子正确（kernel 侧必须 ST/上板，见边界实例）。
