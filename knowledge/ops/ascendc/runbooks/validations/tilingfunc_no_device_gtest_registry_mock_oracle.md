---
type: "Runbook"
title: 自定义算子 TilingFunc 无设备 gtest 单测范式
description: 对安装 run 包 op_host .so 里的 TilingFunc 做无 NPU 设备的 gtest 单测：gert 注册表解析被测函数 + mock PlatFormInfos 平台参数 + 手搓 TilingContext，期望值来自独立 oracle。增量（CANN 9.1.0）：多分支 tilingKey UT 家族（branch-agnostic 切分 + 触发条件反推用例 shape + workspace 下界断言）；branch-agnostic 实现下新分支 tiling 零实现即被独立 oracle UT 交叉验证——"测试先行预期 0 通过被推翻"时的正确判读是 oracle 独立性 + 字段级断言强度，而非通过数本身。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T01:10:00Z'
platforms: ["950"]
---

# 自定义算子 TilingFunc 无设备 gtest 单测范式

## 现象

在 ACLNN/AscendC 自定义算子仓（ops-registry-invoke 算子仓开发流程）里开发一个对标 torch THNN fused GRU cell 反向的算子时，host 侧"公共 TilingFunc"（输入 shape/dtype/attr + 平台参数 → TilingData/blockDim/tilingKey/workspace 的纯 host 计算）实现完成后需要验收：公共 TilingUT 全通过（33/33、退出码 0）、UT 的 oracle 不得调用被测 TilingFunc、实现常量公式与设计文档一致、前序 aclnn 用例回归不退化。

Tiling 是纯 host 侧计算，不跑 kernel——为了不占 NPU 设备、又能对 tiling 输出做逐字段断言，测试采用 gtest 直接加载**安装 run 包**里的 op_host .so（`libcust_opmaster_rt2.0.so`），用 mock 平台信息构造 TilingContext 调用被测函数。验收时在 `tests/tiling` 目录执行 `./run.sh`（构建 → CPack 打包 → 本地安装 → 对安装树二次构建 gtest → 执行，一条龙），输出：

```text
[setup] loading op_host .so: .../install/vendors/<Op>/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so
[setup] tiling kernel for '<Op>' resolved
...
[==========] 33 tests from 1 test suite ran.
[  PASSED  ] 33 tests.
EXIT_CODE=0
```

33/33 全通过，且日志里 grep 不到 `do not registe tiling struct` 隐藏警告与 `FAILED`。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发（自定义算子 run 包）中 host 侧 tiling 的单元测试；不适用于 kernel 直调工程的核函数测试。
- **API 路径**：`gert::OppSoDesc` / `gert::OpImplSpaceRegistryV2`（AddSoToRegistry + SetSpaceRegistry）/ `gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl(opType)->tiling` / `gert::OpTilingContextBuilder`（OpType / IONum / InputTensors / OutputTensors / CompileInfo / PlatformInfo / TilingData / Workspace / AppendAttr）/ `fe::PlatFormInfos::SetPlatformRes("SoCInfo"|"AICoreSpec"|"version")` + `SetCoreNumByCoreType` / `platform_ascendc::PlatformAscendC::GetLibApiWorkSpaceSize()` / `gert::TilingData::CreateCap`；构建侧 cmake + CPack（`--target binary` 聚合目标）。
- **验证过的平台与 CANN 版本**：Ascend 950（arch35；run 包目标 ascend950；UT 在 x86_64 host 侧以 mock 平台参数执行）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度（transfer_scope 五档取最窄）**：`operator_family`——加载/mock/调用/快照这套脚手架可迁移到任何"CPack run 包 + op_host tiling .so"形态的自定义算子；oracle 与逐字段断言必须按算子各自的 TilingData 定义重写。
- **路线优先级**：`recommended_fastpath`（上板 ST 之前先用无设备 UT 锁定 tiling 正确性；本卡非 fallback 路线，无成对相对代价可报）。
- **明确不适用 / 未验证**：kernel 执行、数值精度、性能不在本卡；真实设备上的 tiling 行为未在本卡验证（UT 用 mock 平台参数执行）；CANN 9.0.0-beta.2 之外的版本与 kernel 直调工程形态未验证。

## 解决配方

1. **构建-安装-测试闭环（run.sh）**：对算子包 cmake 配置后构建 `binary` 聚合目标并 `cpack` 出 `custom_opp_ubuntu_x86_64.run`，安装到测试目录旁的固定 `install/`；随后**删掉 build 目录重新 cmake**，以 `-DOPP_INSTALL_DIR=<install>` 对安装树二次构建 gtest（头文件与 .so 只取安装树，绝不链源码 build 目录）；运行前把 `LD_LIBRARY_PATH` 前插安装树的 `op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m)`（被测 .so 为 `libcust_opmaster_rt2.0.so`）。完整脚本见 ## 完整代码。
2. **从 gert 注册表解析被测 TilingFunc**：不走 dlsym 找符号——`gert::OppSoDesc` 描述安装的 .so → 自建 `OpImplSpaceRegistryV2` 做 `AddSoToRegistry` → `DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry` → `GetSpaceRegistry()->GetOpImpl(kOpType)->tiling` 取函数指针；SetUp 里 function-local static 保证只加载一次。
3. **mock 平台信息**：TilingFunc 的平台参数（AIV 核数 / UB 大小 / blockSize / cacheLineSize）全部运行时经 `context->GetPlatformInfo()` 派生接口获取，因此只要用 `fe::PlatFormInfos::SetPlatformRes` 注入三张 map——`SoCInfo`（ai_core_cnt / cube_core_cnt / vector_core_cnt / core_type_list / l2_size）、`AICoreSpec`（ub_size / l0_* / l1_size）、`version`（Short_SoC_version / NpuArch）——再 `SetCoreNumByCoreType("AICore")`，被测函数读到的就是 mock 值。于是**每个测试用例可独立指定 coreNum/ubSize**（示例默认 48 核 / 256KB），随核数漂移的分支判定（如 group 触发阈值 `coreNum/2`）可以在不同核数下分别验证。
4. **手搓 TilingContext**：`gert::OpTilingContextBuilder` 逐项喂 OpType / OpName / IONum / InputTensors / OutputTensors（`gert::Tensor` 用 MakeShape + StorageFormat(ND) + dtype 构造）/ CompileInfo（占位 dummy）/ PlatformInfo / TilingData（`gert::TilingData::CreateCap` 分配 + memset 清零，保证 TilingFunc 未写的字段读回 0）/ Workspace（`ContinuousVector::Create<size_t>(16)`）；attr 按 OpDef 注册序 `AppendAttr`。
5. **调用后立即快照**：`tilingFunc_(ctx)` 返回后，`ctx->GetRawTilingData()->GetData()` 指向的缓冲属于局部 `CreateCap` unique_ptr——函数返回即析构，后续堆复用（如 gtest 失败消息格式化）会踩坏悬垂指针，必须立刻 `assign` 进 fixture 自有的 `std::vector<uint8_t>` 再比较。blockDim / tilingKey / ws[0] 同步取出；sysWorkspaceSize 用**同一个 mock 平台**构造 `PlatformAscendC` 后 `GetLibApiWorkSpaceSize()` 现算，断言口径 `ws[0] == sysWs + 期望 usrWs`。
6. **期望值 = 独立 oracle**：oracle 是按设计文档手写的纯整数运算实现（常量手抄自设计文档），不 include 被测头文件、不调用被测函数、不引用 `optiling::` 命名空间；断言层把快照的 raw tiling data `reinterpret_cast` 成 TilingData 结构体逐字段 `EXPECT_EQ`，另加 tilingKey / blockDim / ws0 三件套。
7. **判别命令（oracle 独立性 + 隐藏警告自检；括号内为本次验收实测输出）**：

```bash
# (a) UT 日志无隐藏注册警告、无 FAILED（实测 grep_exit=1，即无命中）
grep -iE 'do not registe tiling struct|FAILED' /tmp/<op>_tiling_ut.log; echo "grep_exit=$?"
## (b) UT 源码无被测命名空间引用（实测仅 fixture / TEST_F 命名行命中，无 optiling:: 调用）
grep -nE 'optiling::|OPTILING_SYS|IMPL_OP_OPTILING|<Op>Tiling' tests/tiling/test_tiling_gtest.cpp
## (c) oracle 函数体 40 行窗口内无对被测实现的间接调用（实测 oracle_indirect_exit=1）
grep -E 'Oracle(Compute|ChooseTilingKey)' -A 40 tests/tiling/*.cpp | grep -E 'optiling::[A-Za-z]'; echo "oracle_indirect_exit=$?"
```

## 设计决策

- **为何测安装树而不是源码 build 目录**：run.sh 头注释明令 "Build the gtest against the INSTALLED run package (headers + .so only from the install tree, never from the source build directory)"——被测物与真实部署形态（run 包安装后的 .so）一致，避免"源码能过、装包不过"。
- **为何经 gert 注册表解析而不是 dlsym 符号**：被测 TilingFunc 在 .so 内经 OpDef `AICore().SetTiling(optiling::TilingFunc)` 绑定（被测实现文件头注释记载），注册表路径复刻了真实 tiling 分发对 .so 的加载方式；本卡来源会话实测 `GetOpImpl(kOpType)->tiling` 解析成功（`[setup] tiling kernel resolved`）。
- **为何 mock 三张平台表 + SetCoreNumByCoreType**：TilingFunc 对平台参数的全部读取都走 `GetPlatformInfo()` 派生接口（GetCoreNumAiv / GetCoreMemSize(UB) / GetUbBlockSize / GetCacheLineSize），mock 这些键即可逐用例控制平台；这同时让"平台参数运行时获取、禁止写死"的实现纪律可被 UT 检验。
- **为何快照 raw tiling data**：防 `CreateCap` 局部缓冲悬垂（代码注释原文见 ## 完整代码 InvokeTiling 段）。
- **为何 sysWs 现算而不是写常数**：`ws[0] = sysWorkspaceSize + usrWorkspaceBytes`，sys 随平台/版本变化；用同一 mock 平台查询保证断言与被测函数读到同值。
- **被放弃的候选**：对源码 build 目录链接/加载 .so 做单测（run.sh 注释显式禁止，见上）。本卡来源会话未讨论其他备选（如直接上板 ST 验证 tiling）——无证据，不臆造。

## 系统信息

- toolchain：CANN 9.0.0-beta.2（`ASCEND_HOME_PATH` 指向 `<home>/Ascend/cann-9.0.0-beta.2`，run.sh 自动 source 其 `set_env.sh`）；GCC 13.3.0；CPack External 产 `custom_opp_ubuntu_x86_64.run`（约 2960 KB）。
- 目标平台：ascend950 / arch35（mock `Short_SoC_version=Ascend950`、`NpuArch=3510`）。
- 目录布局：`operators/<Op>_package/{op_host/arch35, op_kernel/arch35, tests/tiling}`；安装树 `tests/tiling/install/vendors/<Op>/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so`。
- 相关文件（与本卡证据一一对应）：`tests/tiling/run.sh`（构建-安装-运行脚本，来源会话全文读出）、`tests/tiling/test_tiling_gtest.cpp`（harness + oracle + 33 个用例；本卡内联其 harness 核心）、`tests/tiling/CMakeLists.txt`（把 `OPP_INSTALL_DIR` / `OP_HOST_SO_ABS` 注入 gtest——来源会话未读出内容，迁移时需对照）、`op_host/arch35/<op>_tiling_arch35.cpp`（被测 TilingFunc）。

## 完整代码

`tests/tiling/run.sh`（完整，verbatim；`<Op>` 处为示例算子名，迁移时替换）：

```bash
#!/bin/bash
## =============================================================================
## ThnnFusedGruCellGrad_package/tests/tiling/run.sh
## =============================================================================
## ROLE: Build, install and test the ThnnFusedGruCellGrad tiling gtest unit test.
##   1. Build the run package (cmake + cpack -> custom_opp_ubuntu_x86_64.run)
##   2. Install the run package into a local directory
##   3. Build the gtest against the INSTALLED run package (headers + .so only
##      from the install tree, never from the source build directory)
##   4. Run the gtest executable
## =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${SCRIPT_DIR}/../.."

## Source CANN environment setup script if available.
_CANN_HOME="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
if [ -z "${_CANN_HOME}" ]; then
    echo "ERROR: ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME not set. Run: source <CANN>/set_env.sh" >&2
    exit 1
fi
SET_ENV="${_CANN_HOME}/set_env.sh"
if [ -f "${SET_ENV}" ]; then
    . "${SET_ENV}"
fi
CANN_HOME="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME}}"

## -----------------------------------------------------------------------------
## Step 1: build the run package (cmake + cpack -> custom_opp_ubuntu_x86_64.run)
## -----------------------------------------------------------------------------
BUILD_DIR="${SCRIPT_DIR}/build"
RUN_PKG_FILE="${BUILD_DIR}/custom_opp_ubuntu_x86_64.run"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${PKG_DIR}" -B "${BUILD_DIR}" -DCANN_HOME="${CANN_HOME}"
## Build the "binary" aggregate target (same flow as operators/build.sh's
## `make binary package`): it transitively builds ThnnFusedGruCellGrad_ascend950
## (host-side libraries + device kernel binaries) AND
## ascendc_kernels_ascendc_bin_ascend950_gen_ops_config, which produces the
## ascendc_kernels/binary/config/ directory that the CPack install rules
## require. Building ThnnFusedGruCellGrad_ascend950 alone is NOT enough in this
## CANN version (binary/config missing -> CPack "file INSTALL cannot find").
cmake --build "${BUILD_DIR}" --target binary -j"$(nproc)"
cpack --config "${BUILD_DIR}/CPackConfig.cmake" -B "${BUILD_DIR}"
if [ ! -f "${RUN_PKG_FILE}" ]; then
    echo "ERROR: run package not produced: ${RUN_PKG_FILE}" >&2
    exit 1
fi

## -----------------------------------------------------------------------------
## Step 2: install the run package into a FIXED directory next to run.sh
## -----------------------------------------------------------------------------
INSTALL_DIR="${SCRIPT_DIR}/install"
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
bash "${RUN_PKG_FILE}" --quiet --nox11 --install-path="${INSTALL_DIR}"

## -----------------------------------------------------------------------------
## Step 3: build the gtest against the installed run package
## -----------------------------------------------------------------------------
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCANN_HOME="${CANN_HOME}" \
    -DOPP_INSTALL_DIR="${INSTALL_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

## -----------------------------------------------------------------------------
## Step 4: run the gtest tiling unit test
## -----------------------------------------------------------------------------
OP_HOST_SO_DIR="${INSTALL_DIR}/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m)"
export LD_LIBRARY_PATH="${OP_HOST_SO_DIR}:${CANN_HOME}/lib64:${LD_LIBRARY_PATH}"
"${BUILD_DIR}/test_tiling_gtest"

```

> 注：脚本注释中"仅构建 `<OP>_ascend950` 单目标会因 binary/config 缺失报 CPack `file INSTALL cannot find`"是脚本编写会话的结论；本卡来源会话只观测到按 `--target binary` 构建成功的路径，未复现该失败面。

`tests/tiling/test_tiling_gtest.cpp` 的 harness 核心（verbatim；`kOpType` / `kOpHostSoName` / `kTilingDatumSize` / `MakeShape` 为同文件内常量与辅助函数，`<Op>TilingData` 结构体来自 op_kernel 侧共享头）。**加载与解析**：

```cpp
// LoadTilingKernel: load the installed run package op_host .so and return the
// registered TilingFunc. Called once (function-local static in SetUp).
static gert::OpImplKernelRegistry::TilingKernelFunc LoadTilingKernel()
{
#if defined(OP_HOST_SO_ABS)
    std::string soPath = OP_HOST_SO_ABS;
#else
    std::string soPath = std::string("./") + kOpHostSoName;
#endif

    fprintf(stdout, "[setup] loading op_host .so: %s\n", soPath.c_str());

    gert::OppSoDesc soDesc({ge::AscendString(soPath.c_str())}, ge::AscendString("op_host_so"));
    auto spaceReg = std::make_shared<gert::OpImplSpaceRegistryV2>();
    if (spaceReg->AddSoToRegistry(soDesc) != ge::GRAPH_SUCCESS) {
        fprintf(stderr, "[setup] FAILED to add op_host .so to registry\n");
        return nullptr;
    }
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceReg);

    auto reg = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (reg == nullptr) {
        fprintf(stderr, "[setup] FAILED: space registry is null\n");
        return nullptr;
    }
    const auto *impl = reg->GetOpImpl(kOpType);
    if (impl == nullptr || impl->tiling == nullptr) {
        fprintf(stderr, "[setup] FAILED: tiling kernel for '%s' not found\n", kOpType);
        return nullptr;
    }
    fprintf(stdout, "[setup] tiling kernel for '%s' resolved\n", kOpType);
    return impl->tiling;
}
```

**mock 平台信息（完整函数）**：

```cpp
// Mock platform info with per-case AIV core count and UB size (reference
// pattern from scale_custom_package/tests/tiling).
static void InitPlatformInfo(gert::TilingContext *ctx, int64_t coreNum, int64_t ubSize)
{
    fe::PlatFormInfos *pi = ctx->GetPlatformInfo();
    if (pi == nullptr) {
        return;
    }

    std::map<std::string, std::string> socInfos;
    socInfos["ai_core_cnt"] = std::to_string(coreNum);
    socInfos["cube_core_cnt"] = std::to_string(coreNum / 2);
    socInfos["vector_core_cnt"] = std::to_string(coreNum);
    socInfos["core_type_list"] = "AICore";
    socInfos["l2_size"] = "33554432";
    pi->SetPlatformRes("SoCInfo", socInfos);

    std::map<std::string, std::string> aicoreSpec;
    aicoreSpec["ub_size"] = std::to_string(ubSize);
    aicoreSpec["l0_a_size"] = "65536";
    aicoreSpec["l0_b_size"] = "65536";
    aicoreSpec["l0_c_size"] = "131072";
    aicoreSpec["l1_size"] = "524288";
    pi->SetPlatformRes("AICoreSpec", aicoreSpec);
    pi->SetCoreNumByCoreType("AICore");

    std::map<std::string, std::string> socVersion;
    socVersion["Short_SoC_version"] = "Ascend950";
    socVersion["NpuArch"] = "3510";
    pi->SetPlatformRes("version", socVersion);
}
```

**Fixture 骨架：SetUp 一次加载 + 每用例平台参数（CaseIn）**：

```cpp
// ---------------------------------------------------------------------------
// Fixture: common per-test work.
// ---------------------------------------------------------------------------
class ThnnFusedGruCellGradTilingTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        static auto func = LoadTilingKernel();
        tilingFunc_ = func;
        ASSERT_NE(tilingFunc_, nullptr) << "registered TilingFunc not loaded";
    }

    struct CaseIn {
        int64_t B = 0;
        int64_t H = 0;
        ge::DataType dtype = ge::DT_FLOAT;   // applied to grad_hy AND storage
        bool dtypeMismatch = false;          // reject case: storage gets storageDtype
        ge::DataType storageDtype = ge::DT_FLOAT16;
        bool hasBias = false;
        int64_t coreNum = 48;                // doc worked-example platform (§4.6)
        int64_t ubSize = 262144;             // 256KB (§4.6)
        // raw shape overrides (reject cases); empty -> derived from B/H
        std::vector<int64_t> gradHyShape;
        std::vector<int64_t> storageShape;
    };
```

**调用与快照（InvokeTiling，完整函数；输入/输出张量布局为示例算子语义，迁移时替换）**：

```cpp
    // Invoke the registered TilingFunc for the case; captures raw tiling data,
    // blockDim, tilingKey, workspace[0] and the sys workspace size queried from
    // the same mocked platform info (HostTiling.md §6.3 mirrors this query).
    // Oracle checks live in the TEST_F bodies / Expect* helpers below.
    void InvokeTiling(const CaseIn &c)
    {
        gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());

        const bool rawShapes = !c.gradHyShape.empty() || !c.storageShape.empty();
        const std::vector<int64_t> ghShape = rawShapes ? c.gradHyShape : std::vector<int64_t>{c.B, c.H};
        const std::vector<int64_t> stShape =
            rawShapes ? c.storageShape : std::vector<int64_t>{c.B, 5 * c.H};
        const ge::DataType ghDtype = c.dtype;
        const ge::DataType stDtype = c.dtypeMismatch ? c.storageDtype : c.dtype;

        // inputs: grad_hy (B,H) + storage (B,5H)（声明序，thnn_fused_gru_cell_grad_def.cpp）
        std::vector<gert::Tensor> ins;
        ins.emplace_back(MakeShape(ghShape), fmt, gert::kOnHost, ghDtype, nullptr);
        ins.emplace_back(MakeShape(stShape), fmt, gert::kOnHost, stDtype, nullptr);
        std::vector<gert::Tensor *> inPtrs;
        for (auto &t : ins) {
            inPtrs.push_back(&t);
        }

        // outputs (声明序 ×5): 门梯度×2 (B,3H)、grad_hx (B,H)、双路 bias (3H,)/(0,)
        std::vector<std::vector<int64_t>> outShapes;
        if (rawShapes) {
            outShapes = {{0}, {0}, {0}, {0}, {0}};
        } else {
            const std::vector<int64_t> biasShape = c.hasBias ? std::vector<int64_t>{3 * c.H}
                                                             : std::vector<int64_t>{0};
            outShapes = {{c.B, 3 * c.H}, {c.B, 3 * c.H}, {c.B, c.H}, biasShape, biasShape};
        }
        std::vector<gert::Tensor> outs;
        for (const auto &sh : outShapes) {
            outs.emplace_back(MakeShape(sh), fmt, gert::kOnHost, ghDtype, nullptr);
        }
        std::vector<gert::Tensor *> outPtrs;
        for (auto &t : outs) {
            outPtrs.push_back(&t);
        }

        auto tilingDataBuf = gert::TilingData::CreateCap(kTilingDatumSize);
        ASSERT_NE(tilingDataBuf, nullptr);
        // Deterministic zeroed tiling data region (fields the tiling function
        // does not write must read back 0; header lives at offset 0).
        std::memset(tilingDataBuf.get() + sizeof(gert::TilingData), 0, kTilingDatumSize);
        auto wsBuf = gert::ContinuousVector::Create<size_t>(16);

        fe::PlatFormInfos platformInfo;
        platformInfo.Init();

        gert::OpTilingContextBuilder builder;
        struct DummyCompileInfo {} compileInfo;
        builder.OpType(ge::AscendString(kOpType))
            .OpName(ge::AscendString(kOpType))
            .IONum(ins.size(), outs.size())
            .InputTensors(inPtrs)
            .OutputTensors(outPtrs)
            .CompileInfo(&compileInfo)
            .PlatformInfo(&platformInfo)
            .TilingData(reinterpret_cast<gert::TilingData *>(tilingDataBuf.get()))
            .Workspace(reinterpret_cast<gert::ContinuousVector *>(wsBuf.get()));

        // Attributes in registration order: has_bias (Bool, index 0).
        builder.AppendAttr(c.hasBias);

        auto holder = builder.Build();
        gert::TilingContext *ctx = holder.GetContext();
        ASSERT_NE(ctx, nullptr) << "built TilingContext is null";

        InitPlatformInfo(ctx, c.coreNum, c.ubSize);

        // sysWorkspaceSize from the same mocked platform info (HostTiling.md
        // §6.3: ws[0] = sysWorkspaceSize + usrWorkspaceBytes).
        fe::PlatFormInfos *pi = ctx->GetPlatformInfo();
        ASSERT_NE(pi, nullptr);
        platform_ascendc::PlatformAscendC plat(pi);
        sysWs_ = static_cast<int64_t>(plat.GetLibApiWorkSpaceSize());

        auto ret = tilingFunc_(ctx);
        retOk_ = (static_cast<uint32_t>(ret) == static_cast<uint32_t>(ge::GRAPH_SUCCESS));
        blockDim_ = ctx->GetBlockDim();
        tilingKey_ = ctx->GetTilingKey();
        const auto *rawTd = ctx->GetRawTilingData();
        const uint8_t *tdData = (rawTd != nullptr) ? reinterpret_cast<const uint8_t *>(rawTd->GetData()) : nullptr;

        // Snapshot the raw tiling data into fixture-owned storage: the CreateCap
        // buffer is a local unique_ptr that dies with this function, and later
        // heap reuse (e.g. gtest failure-message formatting) would corrupt the
        // dangling pointer before the Expect* comparisons read it.
        if (tdData != nullptr) {
            rawTdBytes_.assign(tdData, tdData + kTilingDatumSize);
            rawTd_ = rawTdBytes_.data();
        } else {
            rawTdBytes_.clear();
            rawTd_ = nullptr;
        }

        size_t *wsSizes = ctx->GetWorkspaceSizes(1);
        ASSERT_NE(wsSizes, nullptr);
        ws0_ = wsSizes[0];
    }
```

**Fixture 成员（快照与结果暂存）**：

```cpp
    gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc_ = nullptr;
    bool retOk_ = false;
    std::vector<uint8_t> rawTdBytes_;   // fixture-owned snapshot of the raw tiling data
    const uint8_t *rawTd_ = nullptr;    // points into rawTdBytes_ after InvokeTiling
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;
    size_t ws0_ = 0;
    int64_t sysWs_ = 0;
};
```

迁移到其他算子时替换：`kOpType`、TilingData 结构体类型、输入/输出张量构造（示例算子为 grad_hy (B,H) + storage (B,5H) → 5 个输出，属该算子语义）、attr 追加序；加载 / mock / 调用 / 快照骨架不变。oracle 与逐字段断言层（本卡未内联，属源算子设计常量）按算子各自的 TilingData 定义重写，且必须保持独立性——判别命令见 ## 解决配方第 7 步。

## 性能影响

不适用——本卡为测试脚手架范式，无算子性能主张。

## 预防

- 写 tiling UT 前先确认走"构建 → CPack → 安装 → 对安装树二次构建"闭环，gtest 只见安装树。
- oracle 不得 include 被测头 / 调用被测函数 / 引用 `optiling::`——用第 7 步三条 grep 自检。
- 跑完 UT 后 grep 一次 `do not registe tiling struct|FAILED`，确认无隐藏警告（本次实测 grep_exit=1）。
- 每个用例独立指定 coreNum / ubSize（写进 CaseIn），勿在全局写死平台参数。
- 断言不只比 TilingData 字段：tilingKey / blockDim / ws0（= sysWs + 期望 usrWs）三件套一起比。
- `CreateCap` 的 tiling data 缓冲在调用返回后即失效——先快照再断言。

## 验证信号

本卡验证信号来自一次算子仓开发中对"公共 Tiling 实现"任务的验收执行：在 x86_64 构建机上对 ascend950 目标的自定义算子 run 包跑公共 TilingUT（构建 → 安装 → gtest 全流程），并复核 oracle 独立性与 aclnn 回归。

- `[setup] loading op_host .so: <install>/vendors/<Op>/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so` 与 `[setup] tiling kernel for '<Op>' resolved`——被测 TilingFunc 从安装 run 包 .so 经注册表解析成功（来源平台：950（ascend950 包，x86 host 侧执行）；证据等级：tool_observed）。
- `[  PASSED  ] 33 tests.` / `EXIT_CODE=0`——公共 TilingUT 33/33 全通过（来源平台：950；证据等级：tool_observed）。
- `grep -iE 'do not registe tiling struct|FAILED' <ut_log>` → `grep_exit=1`——无隐藏注册警告、无 FAILED（来源平台：950；证据等级：tool_observed）。
- oracle 独立性三条 grep：UT 全文件无 `optiling::|OPTILING_SYS|IMPL_OP_OPTILING` 调用命中；`grep -E 'Oracle(Compute|ChooseTilingKey)' -A 40 ... | grep -E 'optiling::[A-Za-z]'` → `oracle_indirect_exit=1`（来源平台：950；证据等级：tool_observed）。
- aclnn 回归（同次验收的独立步骤）：`tests/aclnn ./run.sh` 退出码 0、无 FAILED、PASSED==1（`[smoke] <op> fp32 PASSED`、`1/1 cases OK`）——安装包端到端通路未因 tiling 实现引入回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖 host 侧 tiling 单测脚手架；kernel 执行 / 数值精度 / 性能不在此卡。
- mock 平台参数 ≠ 物理机：示例默认 48 核 / 256KB 为设计文档 worked example 值；设计文档转述物理机 Ascend950PR_9589 为 64 核（转述、未实测）。随核数漂移的分支判定只在 mock 值下验证过。
- 仅在 CANN 9.0.0-beta.2 + ascend950(arch35) run 包形态实测；其他 CANN 版本 / 包形态（kernel 直调工程等）未验证，推断可迁移。
- `gert::` / `fe::` 属 CANN 内部运行时接口，跨版本签名可能变化（未验证）。
- 断言正确性依赖 gtest 编译时包含的 TilingData 头与安装 .so 内的结构体二进制一致；两者不同步会误判。

## 坏实践

- 对源码 build 目录链接/加载 .so 做单测——run.sh 注释明令 "never from the source build directory"（与部署形态不一致）。
- oracle 复用被测实现（include 被测头 / 调用被测函数 / 抄被测代码路径）——测试退化为自证；独立性三条 grep 就是防这个。
- 只看 gtest 退出码，不 grep 隐藏警告与 FAILED 文本。
- `InvokeTiling` 返回后继续使用 `GetRawTilingData()` 原指针——悬垂（代码注释原文警示）。

## 误诊

无证据——本卡来源会话是一次全通过的验收执行，未出现误诊-纠偏过程。




## 后续增量（2026-09-17，并行策展批：多分支 tilingKey UT 家族与 branch-agnostic 零实现交叉验证，CANN 9.1.0）

euclidean_norm 算子（多分支 tilingKey 位组合 `(isGroup<<2)|(isTailA<<1)|isTailR`，key=0/1/2/4/5）把本卡 harness 扩展为多分支 UT 家族，并补三个机制点：

**1. 多分支 UT 家族与独立构建入口**：每分支一个 UT（`test_branch{0,1,2,4,5}_tiling.cpp`），`tests/tiling/CMakeLists.txt` 注册每个分支 UT target、`tests/tiling/run.sh` 顺序执行全部——注意算子包 `operators/build.sh` 只打包算子本体，**不构建 UT**（UT 有独立构建目录与入口）。

**2. branch-agnostic 切分 → 新分支 tiling 零实现交叉验证**：group 切分实现为 branch-agnostic（ShouldUseGroup/ComputeGroupSplit/SetWorkspaceSize 只按 isGroup 分支、不按 isTailR），Group Tail-R 分支（key=5）tiling 无需新实现即被独立 oracle UT 交叉验证（8/8 首跑通过，同轮对 op_host tiling 代码零变更）。**"测试先行预期 0 通过"被 8/8 通过推翻时的正确判读**：判据是 oracle 独立性（grep 校验 oracle 体内无 `optiling::` 调用，命中只允许出现在注释）+ 全字段断言强度，而非通过数本身——"实现已被前一分支覆盖"与 oracle 独立性互证，不是测试失效。

**3. 触发条件反推用例 shape**：设计文档字面 shape（如 R=4096）在真实 UB 容量下单块可装下（rLoopCntTotal=1）不触发 group 而路由到 normal 分支；用例 shape 必须按触发条件反推——group 触发需 `aLoopCntTotal ≤ coreNum/2 && rLoopCntTotal > 1`，选 totalA ≤ 2 且 R 足够深（R > r_i_max）的 shape 才真正命中目标 key。

**4. workspace 下界断言变体**：本卡原方案断言 `ws[0] == sysWs + 期望 usrWs`（sysWs 用同一 mock 平台现算）；该算子 UT 改为下界断言 `ws[0] ≥ usrSize`（usrSize = rGroupCnt × aTotal × sizeof(float) 由 oracle 精确复算）——sysWorkspaceSize 随平台/版本变化，下界断言对平台相关部分稳健且仍锁住布局公式；两种口径按稳定性需求选择。另注：该算子 UT 被测 .so 取构建树产物（可执行文件同目录解析 `kOpHostSoName`），与本卡"只对安装树二次构建"的更严纪律不同——迁移时优先遵循本卡原方案（与部署形态一致）。

### 复现增量验证信号

- UT 执行：`==== Branch-5 Tiling UT summary: 8/8 cases PASSED (0 failed) ====`、run.sh EXIT=0（含 `[setup] loading op_host .so ... tiling kernel for 'EuclideanNorm' resolved` 加载链路）（来源平台：950（ascend950 构建产物，UT 纯宿主 x86 执行）；证据等级：tool_observed）。
- 回归：同轮 public 19/19、Branch-0 9/9、Branch-1 9/9、Branch-2 13/13、Branch-4 8/8 全 PASSED（来源平台：950；证据等级：tool_observed）。
- oracle 独立性：grep 校验 oracle 体内无 `optiling::Compute*` 调用（54 处命中全在注释/用例描述）（来源平台：950；证据等级：tool_observed）。
- 分支确认：同轮 tiling 实现任务对 op_host tiling 代码零变更，Branch-5 tiling 仅靠本 UT 验收通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- harness 机制依赖 `OpImplSpaceRegistryV2` 注册表加载（CANN 9.1.0 验证）；其他 CANN 版本的 host 注册机制变化需复验。
- oracle 平台常量与目标 SoC 绑定（coreNum/ubSize/blockSize/cacheLineSize 按 DESIGN 手抄）；换 SoC 必须重取常量，否则 ShouldUseGroup 等分支判据失真。
- "branch-agnostic 判断"来自 op_host 源码阅读（agent_reported），与 UT 8/8 + 零代码变更的 tool_observed 事实互相印证，不单独作为验证依据。
