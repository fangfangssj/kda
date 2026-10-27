---
type: "Runbook"
title: Tiling UT compileInfo 传 nullptr 段错误
description: "op_host UT 里 gert::TilingContextPara 的 compileInfo 槽位传 nullptr 时 ExecuteTiling 无声段错误（成功用例与预期失败用例无差别崩溃、无 gtest 输出）；传静态空 CompileInfo 结构体的非空地址即可恢复，与 add_example 参考写法一致。构造函数的 nullptr 默认值不等于可以真传 nullptr。复现增量：第二条独立 trace 与 GetPlatformInfo() 空指针机制（CANN 9.2.0 头文件原文）。复现增量：CANN 9.1.0 三条新 trace——GetContext() 返回空与 GetPlatformInfo() 内部崩两种崩点形态、DO_TILING 宏展开逐步打印的二分定位法、executor 的 platformInfo.Init() 不可省。复现增量：CANN 9.1.0 两条新 trace——nullptr↔非空空结构体的单变量对照探针（GetContext() 0x0↔有效指针）与 add_example 逐字段对照定位法；一条 trace 首跑 InferShape 用例亦崩（是否同槽位未隔离）。"
tags: [task.testing, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# Tiling UT compileInfo 传 nullptr 段错误

## 现象

为自定义算子 BenchNoHarnessDepthwiseConv2d（目标 Ascend950PR/arch35，CANN 9.2.0，x86_64 主机）从参考模板（ascendc-registry-invoke-template 的 add_example）搭建 op_host UT 工程（tests/ut/，gtest + tiling_context_faker + tiling_case_executor，编译 op_host 源码为 `libbench_…_op_host_ut_lib.so` 动态加载）。前两轮编译错误修完后，UT 首次运行：InferShape 测试 10/10 全过，但 Tiling 测试在第一个用例（L0_001，fp16 MobileNet 典型 shape）执行 `ExecuteTiling(para, tilingInfo)` 时进程直接 **Segmentation fault**，没有任何 gtest 失败输出；用 `--gtest_filter` 单跑预期失败的校验用例 F001（groups≠C，本应在参数校验处返回 GRAPH_FAILED）也同样崩溃。gdb 未安装，无法直接取栈。

当时测试代码里 `BuildTilingContextPara` 构造 `gert::TilingContextPara` 时把 compileInfo 位置写成了 `nullptr`。

## 适用场景

- 工程形态：op_host Tiling/InferShape 单元测试（UT），基于 tiling_context_faker.h / tiling_case_executor.h 框架在 x86_64 主机上动态加载 op_host .so。
- API 路径：`gert::TilingContextPara`（exe_graph/runtime）构造参数中的 compileInfo 槽位；`ExecuteTiling`（tiling_case_executor）；`IMPL_OP_OPTILING(...).TilingParse<CompileInfo>(...)` 注册的编译信息结构体。
- 验证过的平台与 CANN 版本：Ascend950（BUILD_SOC_VERSION=Ascend950 的 UT，宿主 x86_64，CANN 9.2.0）；与 frontmatter `platforms` 一致（UT 级验证，未上板）。
- 适用粒度：platform_wide——约束来自 UT 框架对 TilingContextPara 的使用约定，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：InferShape UT（InfershapeContextPara 无 compileInfo 槽位，本会话 InferShape 测试始终通过）；aarch64 主机未验证。

## 根因

UT 框架执行链路（tiling_case_executor 的 DO_TILING 宏 → OpTilingContextBuilder → tiling 注册函数）内部会访问 TilingContextPara 携带的 compileInfo 指针，传 `nullptr` 即段错误；compileInfo 允许指向"空的"编译信息结构体（本算子的 `BenchNoHarnessDepthwiseConv2dCompileInfo` 本来就是空结构体），但不允许为空指针。参考样例 add_example 的写法就是定义空结构体实例并传其地址，而不是传 nullptr。

判别特征：**无论用例的 tiling 逻辑走成功路径还是参数校验失败路径都在 ExecuteTiling 内崩溃**（F001 的 groups≠C 校验在任何 tiling 计算之前就该返回），说明崩溃点在框架层而非算子 tiling 代码——这与"某个具体 shape/参数触发算子内 bug"的模式完全不同。

## 解决配方

修复前（tests/ut/op_host/test_…_tiling.cpp 的 BuildTilingContextPara，compileInfo 槽位传 nullptr）：

```cpp
    // 通过 placement new 重新构造 para（引用参数无法赋值，用析构+构造）
    para.~TilingContextPara();
    new (&para) gert::TilingContextPara(
        OP_NAME,
        inputTensorDesc,
        outputTensorDesc,
        attrs,
        nullptr,                          // compileInfo
        param.coreNum,
        param.ubSize,
        TILING_DATA_MAX_SIZE);
```

修复后（先定义与 op_host tiling.cpp 中一致（此处为空）的 CompileInfo 结构体和静态实例，再传非空地址）：

```cpp
// 空的 CompileInfo 结构体（与 op_host/tiling.cpp 中定义一致，入图场景依赖）
// 传非空地址避免框架内部访问 nullptr
struct BenchNoHarnessDepthwiseConv2dCompileInfo {};
static BenchNoHarnessDepthwiseConv2dCompileInfo g_compileInfo;
```

```cpp
    // 通过 placement new 重新构造 para（引用参数无法赋值，用析构+构造）
    para.~TilingContextPara();
    new (&para) gert::TilingContextPara(
        OP_NAME,
        inputTensorDesc,
        outputTensorDesc,
        attrs,
        &g_compileInfo,                   // compileInfo（非空空结构体地址）
        param.coreNum,
        param.ubSize,
        TILING_DATA_MAX_SIZE);
```

定位方法（gdb 不可用时的二分）：用 `--gtest_filter` 分别单跑成功用例与预期失败用例——失败用例也崩即可证明崩溃在算子逻辑执行之前的框架公共路径，从而把怀疑面收敛到 TilingContextPara 的公共入参（compileInfo/platformInfo/coreNum）。

## 设计决策

- 选择"测试侧定义同名空结构体并传地址"而不是"修改 tiling.cpp 把 CompileInfo 结构体导出到公共头"：add_example 参考框架就是这么做的（测试自带空结构体实例），改动最小且不污染算子侧接口；结构体为空时二者布局等价。
- 没有继续深挖框架源码中确切的解引用行：传非空地址后成功/失败用例全部恢复，已足够闭环；框架内部访问点记入遗留问题。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 抄 add_example 的 UT 框架时，注意它传的是**空结构体实例的地址**而不是 nullptr——`TilingContextPara(..., void* compileInfo = nullptr, ...)` 有默认值不等于可以真传 nullptr。
- Tiling UT 出现"所有用例（含预期失败用例）无差别段错误、无 gtest 输出"时，先怀疑框架级公共入参（compileInfo/platformInfo/coreNum），不要先查算子 tiling 逻辑。

## 验证信号

来源说明：该算子迭代一 A2 UT 联调：修复前后同一套 27 条用例对照。

- 修复前：InferShape 10/10 PASS；Tiling 首用例 L0_001 在 ExecuteTiling 段错误（进程崩溃、无 gtest 失败详情）；`--gtest_filter=*TilingFailureTest*` 单跑 F001 亦崩溃（来源平台：950；证据等级：tool_observed）
- 修复后：单跑 L0_001 PASS（`tilingKey=1 blockNum(usedCoreNum)=32 hTile=56`，EXIT=0）；全量 UT 27/27 通过（10 InferShape + 6 TilingSuccess + 11 TilingFailure）（来源平台：950；证据等级：tool_observed）
- 修复后（回归佐证）：迭代二在同一 UT 工程上扩到 37 条用例仍全过，含依赖 ExecuteTiling 的独立 TEST_F（来源平台：950；证据等级：tool_observed）

## 失效边界

- compileInfo 非空但指向错误布局（与 op_host 注册的 TilingParse<CompileInfo> 类型不一致）时的行为未验证——本例 CompileInfo 为空结构体，无布局风险。
- 框架内部具体在哪一行解引用 compileInfo 未读到源码证据；其他 CANN 版本的 tiling_case_executor 行为未验证（推断可迁移）。

**未决项**：

- 优先项：读 tiling_case_executor.cpp / OpTilingContextBuilder 源码，把 compileInfo 的内部访问点定位到行级（本卡根因止步于对照实验级证据）。
## 坏实践

- 环境（本会话 UT 主机）没有 gdb 时直接放弃定位：可先用 `--gtest_filter` 逐类用例二分（成功用例 vs 预期失败用例），失败用例也崩即可证明崩溃在算子逻辑执行之前的框架公共路径。
- 把 `TilingContextPara` 构造函数的 `compileInfo = nullptr` 默认值当成"可以不传"的许可。

## 误诊

- 无（成功/失败用例无差别崩溃的形态直接指向框架公共路径，未在算子 tiling 逻辑上浪费时间）。

## 增量（第二条独立 trace 复现与 GetPlatformInfo 空指针机制）

另一独立轨迹并入第二条独立 trace（MlaProlog 类算子，Ascend950PR + CANN 9.2.0，faker 框架 `tests/ut/common/tiling_case_executor.cpp` 的 `EXECUTE_TILING_CASE` 宏族），同形态复现（InferShape 用例全过、Tiling 首条用例段错误），并把崩溃点推进到头文件机制级：

**1. 崩溃点定位（插桩日志）**：`[DBG] Got tiling context` 之后、下一条 `tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos)` 之内崩溃——faker 宏对 `GetPlatformInfo()` 返回值直接调 `SetPlatformRes`，无空检查。

**2. `GetPlatformInfo()` 的空指针路径（CANN 头文件原文，`exe_graph/runtime/tiling_context.h:443`）**：

```cpp
fe::PlatFormInfos *GetPlatformInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
        return nullptr;
    }
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index + 1U);
    if (av == nullptr) {
        return nullptr;
    }
    return av->GetValue<fe::PlatFormInfos *>();
}
```

compileInfo 缺失（`TilingContextPara` 的 compileInfo 槽位传 `nullptr`）时该路径返回 `nullptr`，faker 宏随即对空指针调 `SetPlatformRes("SoCInfo", socInfos)` 直接段错误。修复与本卡主配方一致：传**非空空结构体**的地址。

**3. 修复形态变体（文件域 dummy struct，verbatim）**：

```cpp
// Dummy compile info (must be non-null for GetPlatformInfo to work)
struct MlaPrologCompileInfoDummy {} compileInfoDummy;

// Build TilingContextPara for <Op>
static TilingContextPara BuildTilingContext(const <Op>TilingParam& param)
{
    ...（inputs/outputs/attrs 构造同本卡主配方）...

    return TilingContextPara(
        OP_NAME,
        inputs,
        outputs,
        attrs,
        &compileInfoDummy,      // compileInfo (must be non-null)
        AIV_CORE_NUM,           // coreNum (AIV count)
        UB_SIZE,                // ubSize (248KB)
        TILING_DATA_MAX_SIZE);  // tilingDataSize
}
```

选择空 struct 而非伪造完整 compile info：`GetPlatformInfo()` 只依赖 compile info **非空**这一事实；UT 的 tiling 实现不读 compile info 内容时空 struct 即可走通，与 add_example 用法一致。

### 复现增量验证信号

- 修复前：首条 tiling 用例 `[DBG] Got tiling context` 后 `Segmentation fault (core dumped)`，6 条 tiling 用例 0 条可跑（来源平台：950；证据等级：tool_observed）。
- 修复后：`[DBG] SetPlatformRes SoCInfo done ... [DBG] Tiling func returned: 0`，6/6 tiling 用例 PASS（TilingKey/BlockDim/WsSize 正常输出）（来源平台：950；证据等级：tool_observed）。
- 清理调试打印后干净重建（`rm -rf build && cmake .. && make`）复跑：12/12 全部通过（6 InferShape + 6 Tiling），`run.sh` 汇总 `Result: PASS`（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- `GetPlatformInfo()` 空指针路径为 CANN 9.2.0 头文件原文；faker 宏（tiling_case_executor）无空检查是该参考模板的实现事实。
- 若 tiling 实现确实需要读 compile info 内容（TilingParse 填充的 CompileInfo），空 dummy 不够，需构造真实 compile info——本例 tiling 走 `GetPlatformInfo` 但不读 compile info 内容。
- 该轨迹另发现 faker 的 `socToUpper` 小写键表与 `BUILD_SOC_VERSION` 大小写不匹配致 socVersion 空串——对 tiling 结果无影响（修复 compileInfo 后测试全过），未修；若后续平台逻辑依赖 socVersion 精确值需另行处理。

## 增量（CANN 9.1.0 三条新 trace：两种崩点形态与定位/恢复边界）

另一独立轨迹并入三个独立 trace（argmax / 指数族 / SwiGlu 类算子的 op_host UT，均 CANN 9.1.0、faker 框架 `TilingContextPara` + `DO_TILING`），同形态复现（InferShape 用例全过、Tiling 首用例即段错误），并补充：

**1. 崩点观测的两种形态**（同一 nullptr 根因在不同观测点呈现）：

- `contextHolder.GetContext()` 返回 nullptr：插桩输出 `[DEBUG-MACRO] tilingContext=(nil)`，下一条 `tilingContext->GetPlatformInfo()->SetPlatformRes(...)` 解引用空指针即崩（argmax trace，EXIT_CODE=139）。
- 崩在 `GetPlatformInfo()` 调用内部：`[DBG] 11: got tilingContext` 打印后、函数返回前崩——测试里加 `auto* pi = tilingContext->GetPlatformInfo(); if (pi == nullptr) ...` 外层空指针保护拦不住（SwiGlu trace；与本卡先前增量 在 9.2.0 头文件读到的 null-return 路径并存，9.1.0 观测到的是函数内部越界形态）。

两种形态的修复一致：compileInfo 传非空。

**2. gdb 不可用（容器 `ptrace: Operation not permitted`）时的定位法**：在 `ExecuteTiling` 前后打点确认崩在 executor 内部；把 `tiling_case_executor.cpp` 的 `DO_TILING` 宏手工展开为带 `[DBG] n:` 步骤打印的等价实现（faker 创建 → 张量 → 属性 → `platformInfo.Init()` → tilingData/workspace → `Build()` → `GetContext` → `GetPlatformInfo` → `SetPlatformRes` → 取 tilingFunc → 调用），逐轮二分——崩溃落在哪两条打印之间，根因就在那一步。辅助排除：最小程序打印 `sizeof(fe::PlatFormInfos)`（正常 24 字节）排除头文件/库 ABI 不一致假设；补 `ASCEND_OPP_PATH`/`FMK_PLATFORM_INFO_PATH` 环境变量无效，排除环境缺失假设。

**3. 修复形态（文件域静态实例，与主卡 placement-new 变体等价）**：

```cpp
// Dummy compile info struct (matches <Op>CompileInfo in tiling.cpp)
struct DummyCompileInfo {};
static DummyCompileInfo g_compileInfo;
```

```cpp
    gert::TilingContextPara tilingContextPara(
        OP_NAME, inputDesc, outputDesc, attrs,
        &g_compileInfo,              // compileInfo（必须非空；空 struct 也行）
        CORE_NUM, UB_SIZE, TILING_DATA_SIZE);
```

要点：传 static 存储期对象的地址，不是临时对象的地址；结构体内容与算子 tiling 侧的 CompileInfo 定义匹配即可为空。

**4. executor 的 `platformInfo.Init()` 不能省**：调试中试验性注释掉 `fe::PlatFormInfos platformInfo; platformInfo.Init();` 后段错误同样消失（形成"已修复"假象），但 tilingFunc 返回 4294967295（GRAPH_FAILED）；恢复 `Init()` 后返回 0 且 tiling 字段与手算期望一致——`Init()` 负责 SoCInfo/version 等平台资源加载，跳过它则 tiling 内的 `PlatformAscendC` 平台查询失败。

### 复现增量验证信号

来源说明：三个独立算子的 op_host UT 首次运行联调，段错误复现后经插桩定位，改为非空 compileInfo 后逐级验证。

- argmax trace：修复前 `tiling_test/0` 在 `[DEBUG-MACRO] tilingContext=(nil)` 后 Segmentation fault，EXIT_CODE=139；修复后同一用例 `[DEBUG-MACRO] tilingContext=0x5600496dc490`、`tilingRet=0`，TilingFunc 正常执行；全量 op_host UT 27/27 + op_api 10/10 共 37/37 通过（来源平台：950；证据等级：tool_observed）
- 指数族 trace：修复前 tiling_test/0 段错误（core dumped）、infershape 4 用例正常、gdb 报 ptrace: Operation not permitted；修复后单用例 `[PASS] tiling_basic_1024x1024_fp16 | totalLen=1048576 blockFormer=16384 blockNum=64 ubFormer=15872 key=1`，移除调试输出后全量 8/8（4 infershape + 4 tiling），后续迭代 32/32、53/53 无回归（来源平台：950 环境主机侧 UT；证据等级：tool_observed）
- SwiGlu trace：compileInfo 换非空占位后（`Init()` 仍被注释的中间态）`[DBG] 11b: GetPlatformInfo() = 0x7ffdd6a4a530`、`[DBG] 12: SetPlatformRes SoCInfo done`——段错误消失，流程推进到 tilingFunc 调用；恢复 `platformInfo.Init()` 后 `[DBG] 17: tilingFunc returned 0`、totalOutputLength=128、blockDim=1 与手算期望一致（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- 两种崩点形态分别取自 CANN 9.1.0 的独立 trace；先前增量 的 null-return 头文件路径为 9.2.0。跨版本时崩点可能变化，判别以"compileInfo 槽位是否非空"为准，不依赖具体崩溃行。
- dummy compileInfo 只满足"非空"；被测 tiling 真实消费 compile info 内容（如解析 json 串）时不适用，需按真实结构填充（未验证非空场景）。
- `GetLibApiWorkSpaceSize = 4294967295`（0xFFFFFFFF 哨兵值）在 UT 运行中出现但 tilingRet=0 且用例通过，该值语义未追查。
- 崩溃点随 DO_TILING 宏版本变化，插桩输出仅为本版本的阶段序列。

## 增量（CANN 9.1.0 两条新 trace：单变量对照探针与 InferShape 侧边界）

另一独立轨迹并入两个独立 trace（TopK 类 / ApplyAdamW 类算子的 op_host UT 首次联调，均 CANN 9.1.0、faker 框架 `TilingContextPara` 构造函数 compileInfo 默认 `nullptr`），补决定性对照探针与一条边界：

**1. 单变量对照探针（决定性因果隔离）**：写一个独立 gtest 调试程序（DbgEnv 注册链加载 UT .so + 一个 `TilingContextFaker` 全流程用例），仅在 `CompileInfo(nullptr)` 与 `CompileInfo(&dummyCompileInfo)`（空结构体）之间切换：

```cpp
// Empty compile info struct (like add_example)
struct DummyCompileInfo {} compileInfo;

    auto holder = faker.CompileInfo(&compileInfo)      // 换回 CompileInfo(nullptr) 即复现 GetContext()==0
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto* ctx = holder.GetContext();
    cerr << "[DBG] ctx: " << ctx << endl;
```

实测：`CompileInfo(nullptr)` → `GetContext()==0`（Build "成功返回"但上下文为空，不抛异常、不返回错误码）；仅换成 `CompileInfo(&dummyCompileInfo)` → `GetContext()` 立即返回有效指针（`0x55e0df78fa00`）。配合 add_example 参考工程的交叉对照（同环境 `-DSOC_VERSION=Ascend910B` 下其 tiling UT 3/3 PASS，其 `TilingContextPara(..., &compileInfo, ...)` 传非空空 struct 地址——唯一差异即此槽位），单变量闭环。

**2. 定位链（gdb 不可用，退出码 139 形态）**：TopK trace 的 `[ RUN ]` 后立即 `Segmentation fault (core dumped)`、`真实退出码: 139`、无任何报错输出；容器 ptrace 受限（`Error disabling address space randomization: Operation not permitted`）时按三步收窄：① `ExecuteTiling` 前后打点（`Before ExecuteTiling` 可见、`ExecuteTiling returned` 不可见）；② DO_TILING 宏各步骤间插 `fprintf(stderr, "[DBG] ...")`——崩溃前最后一条为 `GetPlatFormInfos done, NpuArch=3510`（平台解析成功，排除 SOC/平台配置）；③ `GetContext()` 后打印指针，观测 `tilingContext=(nil)`。

**3. 恢复的伴生坑**：TopK trace 修复 compileInfo 后随即抛 `C++ exception "not found <OpName>"`——UT 可执行与被测 op_host .so 的构建目标分离（make 测试 target 不连带构建 dlopen 的 .so），须显式构建 .so 目标，相关经验：dlopen 型 TilingUT 须显式构建被加载的 .so 目标。

**4. InferShape 侧边界（与既有"InferShape 用例全过"模式不一致的观测）**：本批 TopK trace 首跑用 `--gtest_filter` 单跑一条 **InferShape** 用例也在 `[ RUN ]` 后段错误（修复后该套件全过）；而 ApplyAdamW trace 同批实测 InferShape 4/4 PASS（仅 Tiling 首用例崩）。InferShapeContextPara 无 compileInfo 槽位（独立 trace 已证），TopK 的 infershape 崩溃是否同根因未隔离（修复 tiling 侧 compileInfo 后该套件恢复，未做单变量复现）——记录为未定界观测，勿据此外推"InferShape 也受 compileInfo 影响"。

### 复现增量验证信号

来源说明：两个独立算子的 op_host UT 首次联调（CANN 9.1.0，faker 框架），修复前后同套用例对照。

- TopK trace：修复前 `[ RUN ]` 后立即 `Segmentation fault (core dumped)`、真实退出码 139（单跑 infershape 用例与 tiling 用例均崩）；分步日志停在 `GetPlatFormInfos done, NpuArch=3510`、`tilingContext=(nil)`；修复后同探针输出 `tilingContext=0x55ac11a38060` 非空、DO_TILING 走到 `GetOpImpl done`，连同 .so 目标补建后 op_host UT 27/27、op_api UT 13/13 通过（来源平台：950 host 侧 UT；证据等级：tool_observed）。
- ApplyAdamW trace：修复前 InferShape 4/4 PASS、首个 Tiling 用例 `Segmentation fault (core dumped)`；逐行复刻 DO_TILING 的独立探针打印到 "Step 8: SetPlatformRes" 后崩溃、最小化探针 `holder.GetContext()` 恒为 0；决定性对照 `CompileInfo(&dummyCompileInfo)` 后 `GetContext()` 返回 `0x55e0df78fa00`；同环境 add_example tiling UT 3/3 PASS（compileInfo 非空）；修复后 `./<op>_op_host_ut` 8/8 通过（4 Tiling + 4 InferShape），`bash run.sh` 输出 `[PASSED] 8 tests.`（来源平台：950；证据等级：tool_observed）。
- 头文件面锚点：`x86_64-linux/include/base/context_builder/op_tiling_context_builder.h` 确认 `CompileInfo(const void*)`/`Build()` 接口（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 行为证据取自 CANN 9.1.0 的 gert/metadef 实现（闭源 .so，未读到内部代码，仅行为级证据）；与本卡先前增量 在 9.2.0 头文件读到的 GetPlatformInfo() null-return 路径并存。
- `OpTilingContextBuilder::Build()` 对 null compileInfo 返回空 holder 的内部机制（是否应报错）未读到源码，仅行为级结论。
- TopK trace 的 InferShape 首跑崩溃未隔离是否同槽位（见增量正文第 4 点），不定界外推。
- gdb 在该环境因 ptrace 权限不可用属环境约束，本卡诊断方法因此设计为不依赖 gdb。

### 复现增量坏实践

- 依赖 `TilingContextPara` 构造函数的 `compileInfo = nullptr` 默认值写 tiling UT——`Build()` 不报错但返回空上下文，段错误出现在离根因很远的 `SetPlatformRes` 处。
- 段错误后直接怀疑被测 tiling 函数（attrs 顺序、平台信息、memset_s 等）——同 trace InferShape 4/4 PASS 且 tiling 函数在修复后完全正常，问题始终在测试脚手架。
- 在容器里先耗时间搞 gdb/ulimit：ptrace 与 core dump 均被限制时应当立即转向插桩日志/单变量对照探针。
- `nm ... 2>/dev/null` 这类吞错探针在目标文件不存在时静默无输出，容易误判为"符号不存在"——探针前先 `ls` 确认文件存在。

### 复现增量遗留问题

- TopK trace 首跑 InferShape 用例崩溃是否与 compileInfo 槽位同根因未隔离（优先项：单变量复现定界）。
- `Build()` 静默失败不给任何错误码/日志，属 faker 可用性问题，可向上游反馈；有 SDK 源码的环境可追一层内部访问点。
