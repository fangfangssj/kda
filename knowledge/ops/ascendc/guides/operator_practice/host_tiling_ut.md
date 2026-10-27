---
type: Guide
title: Host Tiling 单元测试开发
description: 面向 ACLNN 等注册式算子包的 Host Tiling 单元测试，涵盖独立 oracle、TilingContext 构造、平台资源 mock、注册调用、隔离构建，以及测试先行和已有实现的回归验证。
tags: [task.testing, task.implementation, topic.compilation, topic.implementation.tiling, topic.implementation.operator_registration]
status: stable
sources:
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/workflow/op-develop.yaml
    role: primary
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/agents/tiling-developer.md
    role: supporting
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/operators/add_custom_package/tests/tiling/test_tiling_gtest.cpp
    role: example
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/operators/scale_custom_package/tests/tiling/test_tiling_gtest.cpp
    role: example
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/operators/scale_custom_package/tests/tiling/CMakeLists.txt
    role: example
  - resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/Tiling_tuning/ContextBuilder/PlatformInfo.md
    role: supporting
  - resource: https://gitcode.com/cann/asc-devkit/blob/b6f4717e39266934b77e116bb892c8a2885f0a1b/impl/utils/context/context_builder_impl.cpp
    role: supporting
created_at: "2026-09-10T10:08:59Z"
updated_at: "2026-09-11T02:11:00Z"
platforms: [agnostic]
aliases:
  - ACLNN Host Tiling UT
  - 注册式算子包 Tiling 单元测试
  - Host Tiling unit testing
  - Host Tiling UT
  - Tiling gtest
  - Host Tiling red baseline
  - PlatFormInfos GetCoreNumAiv mock
  - PlatformAscendC link closure
---

# Host Tiling 单元测试开发

## 目标与适用范围

本指南适用于 Host Tiling 的测试先行开发、已有实现的功能回归，以及测试基础设施的适配。测试以算子规格和切分设计为依据，通过以下链路验证真实算子包中的 Host Tiling 行为：

```text
设计契约
  → 独立 oracle
  → 测试输入和 Host Tiling 上下文
  → 平台资源 mock
  → 已加载算子包中的注册 TilingFunc
  → 返回码、tilingKey、blockDim、workspace、TilingData 断言
  → 可解释的测试结果
```

具体算子的 shape、dtype、属性、分支、公式和异常语义由规格与设计确定。测试先行开发可使用 stub 检查断言的判别力；对已有实现进行回归时，则验证合法输入的输出和非法输入的拒绝行为。

工程示例面向采用 Registry、TilingContext 和注册 TilingFunc 的算子包，适用于 ACLNN 算子开发中的 Host Tiling 层测试。适用性由这套内部调用链决定，而非上层调用入口名称。独立 oracle、边界覆盖和分层验证的方法也可用于直调算子；直调工程应按自身的 host 切分函数接口组织测试，Registry 加载与 ContextBuilder 示例仅适用于使用这些设施的实现。

## 最小调用链与覆盖扩展

最小探针用于确认构建、加载和调用链是否成立。基础设施首次接入时，可按以下顺序逐步扩大验证范围；已有可信调用链时，可直接复用它并补充当前变更涉及的测试：

1. 从采用相同接口族的公开样例取得 Registry、`gert::OpTilingContextBuilder`、`fe::PlatFormInfos` 和 CMake 骨架。
2. 只替换当前算子的 op type、最小合法 IO、必需属性、TilingData 容量和隔离安装路径。
3. 在工程测试目录中准备测试源文件、构建配置和运行入口，编译并调用一个最小合法 case；文件命名和组织沿用工程约定。
4. 接线探针闭合后，再按设计扩展单 route oracle、平台差分、完整 route、边界和异常矩阵。

公开样例和本指南给出的基础链接闭包，应直接作为第一次编译探针的起点。不要在第一次编译前用 `nm`、`strings`、`objdump`、`readelf` 或 `ar` 验证这些公开组件内部是否含有预期符号；编译器、链接器和加载器返回的首个准确错误才是扩大查询的依据。

这种分层验证将基础设施故障与 oracle、覆盖问题分开。探针能编译和运行只说明调用链成立，完整测试仍需覆盖设计要求的字段、分支、边界和异常行为。

## 测试依据与工程输入

测试设计需要以下信息；它们可以来自设计文档、接口契约或其他受维护的工程资料，不依赖固定的文档文件名：

- 算子规格：输入输出、属性、dtype 组合、shape 约束、边界条件和错误语义。
- Host Tiling 设计：公共切分公式、核数选择、对齐方式、UB 或其他资源预算、异常出口和短路路径。
- TilingData 设计：完整结构、每个字段的语义、生产公式和 kernel 消费点。
- TilingKey 与分支设计：可达 key 集、每个 key 的进入条件和分支内公式。
- Kernel 设计：退化路径实际读取哪些 TilingData 字段，以及确定性要求。
- 已搭好的算子包：OpDef、Host Tiling 源文件、测试目录和构建入口。

这些文件不是可以互相替代的重复信息。规格决定合法输入域和失败语义；Host Tiling 设计决定公共公式；分支文件决定各 route 的具体公式；TilingData 决定输出布局；Kernel 设计决定短路路径仍必须固定的字段。

如果上述输入彼此矛盾，应先报告设计缺口，不在 UT 中发明一套新的算子语义。测试代码可以注释尚未定义的断言边界，但不能用当前被测实现反推并覆盖设计。

## 先画清被测调用链

写断言前，先确认测试实际调用的是哪一个函数、由哪个共享库完成注册、Context 使用哪套接口。建议先形成一张最小接线表：

| 接线项 | 需要确认的事实 |
|---|---|
| 被测算子类型 | Registry 查找使用的准确 op type |
| 被测函数 | TilingFunc 的注册入口和函数签名 |
| 动态库 | 构建并加载的 op_host `.so` 的绝对路径 |
| Context 接口族 | 当前包实际采用的 builder 类型、头文件和 `Build` 返回类型 |
| 输入输出 | IO 数量、顺序、StorageShape、dtype 与可选输入表达方式 |
| 属性 | 属性顺序、类型、缺省值和所有权 |
| 输出缓冲 | TilingData 容量、workspace 容器、blockDim 与 tilingKey 的承载位置 |
| 平台数据 | TilingFunc 实际读取的核数、UB 等资源由谁生产、如何注入 |

接线表没有闭合时，不要先批量生成测试用例。此时大量用例只会把同一个基础设施错误放大成“全部失败”。

## ContextBuilder 接口族必须保持一致

CANN 环境中可能同时出现名称相近但对象模型不同的 ContextBuilder。选择原则不是“哪个示例看起来更完整”，而是沿当前算子包的真实构建链确定：

1. 查看测试目录已有的 include、CMake 查找路径和相邻样例采用的命名空间。
2. 查看当前 CANN 头文件中该 builder 的声明和返回类型。
3. 确认 Registry 返回的 TilingFunc 期待同一类 TilingContext。
4. 用一个最小合法 case 编译并调用，再扩展完整矩阵。

不要把 `context_ascendc::ContextBuilder` 的链式 API 与 `gert::OpTilingContextBuilder` 的成员、持有者或 `Build` 结果混用。相似的方法名只说明它们解决相近问题，不证明 ABI、内部 slot 排列或对象生命周期兼容。

已有 API 说明可从 [ContextBuilder](../../apis/utils_api/tiling_tuning/context_builder/index.md) 开始，并重点核对 [BuildTilingContext](../../apis/utils_api/tiling_tuning/context_builder/build_tiling_context.md)、[PlatformInfo](../../apis/utils_api/tiling_tuning/context_builder/platform_info.md)、[TilingData](../../apis/utils_api/tiling_tuning/context_builder/tiling_data.md) 与 [Workspace](../../apis/utils_api/tiling_tuning/context_builder/outputs.md)。这些 API 卡用于确认契约；最终采用的接口族仍必须和目标工程的真实头文件、构建入口一致。

## 通用 Fixture 的关系骨架

通用骨架表达的是对象关系和生命周期，不是可直接粘贴的完整实现：

```cpp
class HostTilingTest : public TestFixture {
 protected:
  void SetUpSuiteOrCase() {
    // 1. 建立 Registry owner。
    // 2. 从隔离安装目录加载本次构建的 op_host .so。
    // 3. 将 Registry 安装到测试进程使用的默认入口。
    // 4. 按准确 op type 取得 TilingFunc，并确认非空。
  }

  OwnedCaseStorage MakeCase(const CaseSpec &spec) {
    // owner 持有 shape、tensor desc、attrs、platform storage、
    // tiling-data buffer、workspace 和 builder 所引用的一切内存。
    // builder 只负责把这些对象接入同一 TilingContext。
  }

  TilingResult Invoke(OwnedCaseStorage &owned) {
    // 从 owned 构造 context；通过 Registry 取得的函数调用；
    // 调用完成后再读取返回码和输出字段。
  }
};
```

关键约束是“owner 比 context 活得久”。下列对象如果由局部临时变量提供地址，builder 成功不代表调用时仍有效：

- 输入输出的 shape、StorageShape 和 TensorDesc；
- 属性值以及保存属性指针的容器；
- 平台信息的存储对象；
- TilingData 原始缓冲区；
- workspace size 容器；
- 编译信息和其他可选上下文数据。

因此，一个 case 最好返回同时拥有数据和 context 的聚合对象，或者让 Fixture 在整个调用期间持有底层对象。不要只返回裸 `TilingContext*`，把其依赖留在已经退出的构造函数栈帧中。

## 注册调用的公开接口示例

以下示例适用于采用 `gert::OpImplSpaceRegistryV2`、`gert::OpTilingContextBuilder` 和 `fe::PlatFormInfos` 的工程。四段共同构成加载、注册、上下文和平台注入的调用链；使用前核对当前 SDK 的公开声明与链接依赖。算子类型、IO、属性和输出容量按当前算子设计替换。

### 1. 加载隔离安装包并取得注册函数

```cpp
gert::OppSoDesc soDesc(
    {ge::AscendString(opHostSoAbs.c_str())},
    ge::AscendString("op_host_so"));
auto registryOwner = std::make_shared<gert::OpImplSpaceRegistryV2>();
if (registryOwner->AddSoToRegistry(soDesc) != ge::GRAPH_SUCCESS) {
  return nullptr;
}
gert::DefaultOpImplSpaceRegistryV2::GetInstance()
    .SetSpaceRegistry(registryOwner);

auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                    .GetSpaceRegistry();
const auto *impl = registry == nullptr ? nullptr : registry->GetOpImpl(opType);
auto tilingFunc = impl == nullptr ? nullptr : impl->tiling;
```

`registryOwner` 必须覆盖全部调用的生命周期；`.so` 使用本轮 run package 安装到测试专用目录后的绝对路径，并在日志中打印。不要直接编译或调用被测 tiling 源文件来绕过 Registry。

### 2. 用同一接口族构造 TilingContext

```cpp
auto tilingData = gert::TilingData::CreateCap(tilingDataCapacity);
auto workspace = gert::ContinuousVector::Create<size_t>(workspaceSlots);
fe::PlatFormInfos platformInfo;
platformInfo.Init();

gert::OpTilingContextBuilder builder;
builder.OpType(ge::AscendString(opType))
    .OpName(ge::AscendString(opType))
    .IONum(inputPtrs.size(), outputPtrs.size())
    .InputTensors(inputPtrs)
    .OutputTensors(outputPtrs)
    .CompileInfo(&compileInfo)
    .PlatformInfo(&platformInfo)
    .TilingData(reinterpret_cast<gert::TilingData *>(tilingData.get()))
    .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace.get()));
// 按 OpDef 注册顺序 AppendAttr；ListInt 等复合属性使用当前 builder
// 接受的公开表示，不与另一套 ContextBuilder API 混用。
auto holder = builder.Build();
auto *context = holder.GetContext();
```

`inputPtrs`、`outputPtrs`、属性底层存储、`platformInfo`、`tilingData`、`workspace`、`holder` 均须活到 TilingFunc 返回以后。`TilingData::CreateCap` 已维护自身头部；不要对整个对象做额外 `memset`，否则可能破坏容量和数据指针。

属性必须严格按 OpDef 注册顺序追加；公开 `OpContextBuilderBase` 接口已经覆盖常见标量与列表类型，不需要先逆向内部 Attr 布局：

```cpp
bool keepDims = false;
int64_t dim = 0;
float epsilon = 1.0e-5F;
std::vector<int64_t> axes{-1};

builder.AppendAttr(axes)       // ListInt
    .AppendAttr(keepDims)      // Bool
    .AppendAttr(dim)           // Int
    .AppendAttr(epsilon);      // Float
```

还支持 `std::vector<bool>`、`std::vector<float>`、`std::vector<ge::AscendString>` 和 `std::vector<std::vector<int64_t>>`。属性名称本身不参与 `AppendAttr` 定位，顺序才决定 `GetAttrs()` 的下标；可选属性是否省略、是否补默认值应按当前 OpDef/Interface 契约处理。列表和字符串的底层对象仍应由 case storage 持有到 TilingFunc 返回。

### 3. 通过公开 PlatFormInfos 路线注入核数和 UB

来源中的公开样例给出了 `PlatFormInfos` 到 `PlatformAscendC` 的资源映射。对于采用该接口族且消费 `GetCoreNumAiv()` 和 `GetCoreMemSize(UB)` 的 tiling，可在 `Build()` 后、调用前写入当前 Context 持有的平台对象：

```cpp
auto *pi = context->GetPlatformInfo();

std::map<std::string, std::string> socInfo{
    {"ai_core_cnt", std::to_string(coreNum)},
    {"cube_core_cnt", std::to_string(cubeCoreNum)},
    {"vector_core_cnt", std::to_string(coreNum)},
    {"core_type_list", "AICore"},
    {"l2_size", std::to_string(l2Bytes)},
};
pi->SetPlatformRes("SoCInfo", socInfo);

std::map<std::string, std::string> aiCoreSpec{
    {"ub_size", std::to_string(ubBytes)},
    {"l0_a_size", std::to_string(l0ABytes)},
    {"l0_b_size", std::to_string(l0BBytes)},
    {"l0_c_size", std::to_string(l0CBytes)},
    {"l1_size", std::to_string(l1Bytes)},
};
pi->SetPlatformRes("AICoreSpec", aiCoreSpec);
pi->SetCoreNumByCoreType("AICore");

std::map<std::string, std::string> version{
    {"Short_SoC_version", shortSocVersion},
    {"NpuArch", npuArch},
};
pi->SetPlatformRes("version", version);
```

只填目标 TilingFunc 及平台读取 API 真正需要的资源，并让具体值来自当前任务的平台输入或设计，不复制样例的芯片数值。这里的 label/key 是公开样例的契约，不需要通过 `strings`、`objdump` 或解包 `libtiling_api.a` 再发现一次。写入后先做一组差分探针：只改变 `coreNum` 或只改变 `ubBytes`，检查设计相关输出是否随之变化。

### 4. 复用公开构建闭包

下表给出上述接口族的基础依赖归属；只有测试可执行文件直接调用 `PlatformAscendC` 高阶 Tiling API 时，才在测试目标中加入 `libtiling_api.a`。

| 测试侧能力 | 基础依赖 |
|---|---|
| Registry owner、加载与函数查询 | `libopp_registry.so` |
| gert/ge 元数据与 Context 对象 | `libmetadef.so` |
| `fe::PlatFormInfos` | `libplatform.so` |
| 测试侧直接调用 `PlatformAscendC` 高阶 Tiling API | `libtiling_api.a` |
| CANN 日志落地 | `libunified_dlog.so`，仅在当前链接/日志路径需要时加入 |
| 安全 C 库调用 | `libc_sec.so` |
| 动态加载与线程 | `dl`、`pthread` |

测试与 CANN 库使用一致的 `_GLIBCXX_USE_CXX11_ABI=0`，并把隔离安装的 op_host 目录和 CANN `lib64` 写入测试目标 RPATH；运行时同样将两者置于 `LD_LIBRARY_PATH` 前部。若缺少准确符号，先按上表与当前公开链接命令缩小到单个候选组件，不遍历整个 SDK。

同族工程可先用下面的最小编译闭包建立探针，再根据首个准确错误增减；路径中的 host arch 和安装根由当前环境得到，vendor/op type 由当前算子包得到：

```cmake
set(CANN_INC     ${CANN_HOME}/${ASCEND_HOST_ARCH}-linux/include)
set(CANN_PKG_INC ${CANN_HOME}/${ASCEND_HOST_ARCH}-linux/pkg_inc)
set(CANN_LIB     ${CANN_HOME}/lib64)

target_include_directories(test_tiling_gtest PRIVATE
    ${CANN_INC}
    ${CANN_PKG_INC}
    ${CANN_PKG_INC}/base)
target_compile_definitions(test_tiling_gtest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    OP_HOST_SO_ABS="${OP_HOST_SO}")
target_link_libraries(test_tiling_gtest PRIVATE
    gtest_oldabi
    ${CANN_LIB}/libopp_registry.so
    ${CANN_LIB}/libmetadef.so
    ${CANN_LIB}/libplatform.so
    ${CANN_LIB}/libc_sec.so
    dl pthread)
set_target_properties(test_tiling_gtest PROPERTIES
    INSTALL_RPATH "$ORIGIN;${OP_HOST_SO_DIR};${CANN_LIB}"
    BUILD_WITH_INSTALL_RPATH TRUE)
```

`gtest_oldabi` 应由当前环境的 googletest 源码以 `_GLIBCXX_USE_CXX11_ABI=0` 构建。只有测试 TU 自己调用 `PlatformAscendC` 高阶接口时，才把 `${CANN_LIB}/libtiling_api.a` 和对应 `asc/include` 加入闭包；仅仅是被加载的 op_host 内部使用该接口，不构成测试可执行文件直接链接它的理由。

## Platform mock 是生产者—注入—消费者闭环

平台 mock 的目标不是“构造出一个能传给 builder 的非空指针”，而是让 TilingFunc 通过正式消费路径读到测试指定的资源值。

通用关系如下：

```text
平台资源 owner
  ├─ 设置一个明确的非默认核数
  ├─ 设置一个明确的非默认 UB 或其他资源值
  └─ 保持序列化/内部存储有效
          ↓
与当前接口族匹配的 ContextBuilder 注入入口
          ↓
TilingContext
          ↓
TilingFunc 内部实际使用的平台读取对象
          ↓
blockDim、切分长度、循环次数、buffer 预算等可观察结果
```

至少选择一个会影响公式的非默认核数和一个会影响预算的非默认内存值。然后用设计公式计算期望结果，证明 TilingFunc 读取的不是环境默认值或零值。若改变 mock 输入不改变任何相关输出，应先判定为平台数据链未接通，而不是立即修改算子公式。

不同平台资源 API 的构造方式可能随接口族或版本变化。上述 `PlatFormInfos` label/key 来自固定提交的公开样例，适用于采用同一公开接口族的工程；其他接口族先查它自己的公开样例与当前头文件，不能把该映射写入私有对象或自行猜测二进制布局。API 语义可从 [PlatformInfo](../../apis/utils_api/platform_info/index.md) 和 [PlatformAscendC](../../apis/utils_api/platform_info/platform_ascend_c/index.md) 下钻。

还要区分两类平台量：

- 核数、UB、SoC/架构等属于 case 可控输入，应通过公开 `PlatFormInfos` 路线注入，并用差分 case 证明生效。
- block size、cache line、vector register size、系统 workspace 等若由设计明确为当前平台常量，独立 oracle 应采用设计给出的值或公开 API 结果；不要为了重新发现常量去扫描 SDK 二进制。仅在编译签名不确定时读取准确声明头文件。

这种区分可以避免为了一个不会随 case 改变的 oracle 常量，误把任务扩展成平台实现逆向。

## Registry 与动态库加载必须可观察

Host Tiling UT 通常通过加载 op_host 共享库触发注册，再从 Registry 取得函数。测试必须保留下列证据：

- 被加载 `.so` 的绝对路径；
- 路径位于本次测试的隔离安装树，而非共享安装目录；
- 加载动作成功；
- 当前进程使用的 Registry owner 已安装；
- 以准确 op type 查找到目标 TilingFunc；
- 函数指针非空并被真实调用。

只看到 `.so` 文件存在不等于加载成功；加载成功也不等于函数注册到了测试实际查询的 Registry；查到同名函数也不等于加载的是本轮构建产物。三者必须分别取证。

当加载阶段出现平台符号缺失时，可参考 [直调共享库加载未定义平台符号—补齐CANN链接依赖](../../runbooks/compilation/platform_manager_shared_library_link_dependencies.md)，但应依据当前未定义符号收敛依赖，不机械复制某个算子的完整链接列表。

## 日志锚点优先走公开输出通道

失败用例需要稳定日志锚点时，先使用 CANN 的公开运行期输出通道：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

在 gtest 中围绕单次 TilingFunc 调用使用 `CaptureStdout()`/`GetCapturedStdout()`，并确保当前目标所需的日志库已链接。只有公开输出通道在当前组件中不可用、且任务确实要求进程内日志断言时，才根据当前公开声明采用测试侧日志 seam；不得通过扫描日志 `.so` 猜 `DlogRecord`/`CheckLogLevel` 签名。日志捕获失败不应阻塞 Registry、Context 和平台差分探针的先行闭环。

## 从设计文档生成独立 oracle

### 不以被测实现作为期望值来源

独立 oracle 应由规格与设计公式重新表达。以下做法会让错误实现和错误期望同步通过：

- include 被测 TilingData 或 Tiling 实现头并直接复用其结构或 helper；
- 调用实现中的切分函数生成 expected；
- 读取一次实现输出，把该输出固化成“黄金值”；
- 为了让测试通过而复制实现中的整除、对齐或边界错误。

测试可以在自身文件中重述 TilingData 布局，但字段顺序和类型应从设计契约人工转写。oracle 公式应短小、可审阅，并在注释中标出它对应的设计规则。

### 建立规则清单再展开 case

先把设计规则编号，再为规则分配 case：

| 规则类别 | 最低覆盖方式 |
|---|---|
| dtype route | 每个支持的 dtype 组合至少一个成功 case |
| tilingKey route | 每个可达 key 至少一个明确路由 case |
| 阈值 | 精确阈值、阈值减一、阈值加一 |
| 对齐公式 | 已对齐、最小非对齐、尾块 |
| 核数切分 | 小于核数、等于核数、大于核数、不能整除 |
| buffer 预算 | 临界可容纳、刚好越界、多循环 |
| 短路路径 | 空输入、退化维度等规格允许的路径 |
| 异常出口 | 每个设计声明的失败条件一个期望失败 case |
| 确定性 | 固定的切分或归并顺序在输出字段中可验证 |

一个 case 可以覆盖多个规则，但映射必须显式。不要仅按输入尺寸堆 case；应能回答每个 case 在证明哪条公式、哪个 route 或哪个失败出口。

### 成功路径断言

普通成功 case 至少检查：

- 返回状态；
- tilingKey；
- blockDim；
- workspace 数量和每项大小；
- TilingData 中每个由该路径定义的字段。

短路或退化路径只检查设计明确钉住的字段。如果设计没有定义某字段，不要用零初始化的偶然结果制造契约。

### 失败路径断言

失败 case 应验证规格规定的失败状态，并在工程支持时验证稳定的日志锚点。异常是在 infer、tiling 还是平台层被拦截，必须与当前任务的异常覆盖定义一致。不要把“进程崩溃”“Registry 没找到函数”或“Context 构造失败”算成算子正确拒绝输入。

## 测试先行开发中的可信红基线

红阶段用于检查断言能否区分“未实现”与“正确实现”。它适用于测试先行开发；已有实现的常规回归以规格定义的通过条件为准，无需替换实现或要求功能用例失败。需要构造红基线时，使用隔离构建中的最小 stub，推荐顺序：

1. 在隔离测试构建中准备行为明确的 TilingFunc stub，并记录它已满足的契约。
2. 保留必要的空指针保护或任务明确豁免的固定行为。
3. 构建并安装到测试专用目录。
4. 打印并核对实际加载的 `.so` 路径。
5. 运行完整 UT，并保存未截断输出。
6. 确认只有明确记录的豁免用例通过，其他功能性 case 均因断言不满足而失败。

可信的红色与基础设施错误不同：

| 结果 | 是否是有效红基线 |
|---|---|
| TilingFunc 被调用，返回 stub 结果，oracle 断言失败 | 是 |
| 允许的空指针 guard 在 stub 下通过，且已记录豁免 | 是 |
| 测试未编译、`.so` 未加载或函数未注册 | 否 |
| Context 悬空导致崩溃 | 否 |
| 所有 case 意外通过 | 否，测试没有区分未实现状态 |
| 用例失败原因来自找不到 SDK、文件或权限 | 否 |

如果除豁免外仍有 case 通过，应增强该 case 对真正 Tiling 输出的断言；不能通过人为制造无关失败来满足“红色”。

## 推荐的开发节奏

为避免在错误接线之上一次性生成大量代码，按以下顺序推进：

1. **接线探针**：只构造一个最小合法 case，闭合加载、Registry、Context 和调用。
2. **平台探针**：用非默认资源值，确认 TilingFunc 的可观察输出随平台输入变化。
3. **单 route oracle**：为一个普通 route 写完整字段断言。
4. **route 扩展**：覆盖所有可达 tilingKey 和 dtype 组合。
5. **边界扩展**：覆盖阈值 `-1/0/+1`、尾块和多循环。
6. **异常扩展**：按规格及工程覆盖要求逐项落 case 或给出有规格证据的 N/A。
7. **结果审计**：回归测试核对预期输出与拒绝行为；测试先行开发核对红基线的失败来自 oracle。

每完成一层就编译或运行一次。这样可以把“接口接线错误”“平台 mock 无效”“oracle 错误”和“覆盖不完整”分开，而不是在一个巨大测试文件中同时排查。

## 知识库与 SDK 的使用顺序

遇到未知接口时，按信息成本从低到高查询：

1. 先查询本指南，明确当前缺口属于 Context、Platform、Registry、构建还是 oracle。
2. 查询对应的稳定 API 卡，确认 API 语义、约束与头文件线索。
3. 沿 API 卡的 `source_headers` 读取当前环境的声明或内联实现。
4. 查看当前算子包或采用同一接口族的公开样例的接线方式。
5. 若采用 `gert::OpTilingContextBuilder`，先对照本指南给出的公开 Registry、平台资源和链接闭包，核对当前工程的适用条件。
6. 只有在出现确定的未定义符号、ABI 或加载问题且头文件、链接命令和加载路径仍不足以解释时，才对单个候选库做有界的符号检查。

不要从整个 SDK 的 `.so` 扫描开始，也不要解包、反汇编 `libtiling_api.a` 来重新发现公开样例已经提供的资源键。宽泛的 `strings`、`nm -D`、`objdump` 和逐库试链会产生大量相似符号，容易把模型带到与任务无关的 ABI 猜测中。出现故障时使用 [Host Tiling UT 基础设施故障诊断](../../runbooks/compilation/host_tiling_ut_infrastructure_failures.md) 收敛所处层级。

## 完成检查表

- 测试只有任务允许的源文件和构建文件，没有意外旁文件。
- 被测 `.so` 来自本轮隔离安装目录，加载路径已打印。
- Registry、op type、TilingFunc 和 Context 接口族一致。
- Fixture 持有 Context 引用对象的完整生命周期。
- 平台 mock 使用至少一个影响公式的非默认资源值，并有输出侧证明。
- oracle 独立于被测实现，并能追溯到设计规则。
- 支持的 dtype、可达 tilingKey、阈值边界、尾块、多循环和异常出口均有明确处置。
- 成功 case 检查返回码、key、blockDim、workspace 与已定义 TilingData 字段。
- 回归阶段满足规格断言；采用红阶段时，通过集合符合 stub 已满足的契约，预期失败来自功能断言。
- 完整日志可证明加载对象、执行对象、通过集合和失败集合。

## 失效边界

- 本指南不固定某个 CANN 版本的私有 mock 数据布局；实际声明发生变化时以当前头文件为准。
- 本指南不替代算子设计，不决定某个具体算子的 route、公式或异常语义。
- 本指南不把任意测试失败视为有效红基线，也不允许用测试基础设施故障代替 TDD 证据。
- 具体工程的异常覆盖范围和验收要求，以其规格及测试规范为准。
