---
type: "Runbook"
title: "PlatformAscendC 默认构造删除，须传平台指针"
description: "op_host TilingFunc 里 platform_ascendc::PlatformAscendC 默认构造函数被 = delete，直接声明对象报 use of deleted function；必须由 context->GetPlatformInfo() 返回的 fe::PlatFormInfos* 就地构造。"
tags: [task.debugging, topic.compilation, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# PlatformAscendC 默认构造删除，须传平台指针

## 现象

在为 MoE 分组矩阵乘算子（Ascend950PR / arch35 / CANN 9.1.0）的 op_host/arch35 TilingFunc 里获取平台信息（CubeCore 数、LibApiWorkSpaceSize）时，最初在 TilingFunc 入口声明了一个默认构造的 `PlatformAscendC`，再传引用给 helper 填充：

```cpp
    platform_ascendc::PlatformAscendC ascendcPlatform;   // ← TilingFunc 入口
    ...
    // helper 内：
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
```

`bash build.sh --soc=ascend950 -j1` 报错原文：

```text
op_host/arch35/bench_no_harness_grouped_matmul_tiling.cpp:237:39:
error: use of deleted function 'platform_ascendc::PlatformAscendC::PlatformAscendC()'
  237 |     platform_ascendc::PlatformAscendC ascendcPlatform;
      |                                       ^~~~~~~~~~~~~~~
```

## 适用场景

- 工程形态：registry-invoke 自定义算子 op_host TilingFunc / TilingParse 需要平台信息（核数、UB 大小、workspace 尺寸）的场合。
- API 路径：`platform_ascendc::PlatformAscendC`（`asc/include/tiling/platform/platform_ascendc.h`），构造自 `gert::TilingContext::GetPlatformInfo()` 返回的 `fe::PlatFormInfos*`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 9.1.0 平台信息类的构造契约）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：UT 等离线场景如何构造 PlatformAscendC（无真实 TilingContext）未在本闭环内验证。

## 根因

`platform_ascendc::PlatformAscendC` 把默认构造函数显式 `= delete`——平台信息对象必须绑定真实的平台描述（`fe::PlatFormInfos*`）才有意义。`PlatformAscendC ascendcPlatform;` 触发被删除的默认构造；"先默认构造再赋值"的写法同样在第一步就编译失败。

## 解决配方

修复前（TilingFunc 入口默认构造 + helper 引用传出）：

```cpp
// ===== 平台信息获取 =====
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                        platform_ascendc::PlatformAscendC& ascendcPlatform,
                                        int64_t& coreNumAic)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNumAic = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(coreNumAic <= 0,
                OP_LOGE(context, "GetCoreNumAic returned %ld", coreNumAic),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ===== Tiling 分发入口 =====
static ge::graphStatus BenchNoHarnessGroupedMatmulTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    platform_ascendc::PlatformAscendC ascendcPlatform;
    int64_t coreNumAic = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ascendcPlatform, coreNumAic) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);
```

修复后（helper 输出 `fe::PlatFormInfos*`，调用点直接从指针构造）：

```cpp
// ===== 平台信息获取 =====
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                        fe::PlatFormInfos*& platformInfoPtr,
                                        int64_t& coreNumAic)
{
    platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNumAic = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(coreNumAic <= 0,
                OP_LOGE(context, "GetCoreNumAic returned %ld", coreNumAic),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ===== Tiling 分发入口 =====
static ge::graphStatus BenchNoHarnessGroupedMatmulTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    fe::PlatFormInfos* platformInfoPtr = nullptr;
    int64_t coreNumAic = 0;
    OP_CHECK_IF(GetPlatformInfo(context, platformInfoPtr, coreNumAic) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
```

（后续在同一 TilingFunc 内以 `ascendcPlatform.GetLibApiWorkSpaceSize()` 取 Matmul API 系统 workspace 大小、以 `GetCoreNumAic()` 做多核分配。）

## 设计决策

- helper 只输出 `fe::PlatFormInfos*` 而不输出 `PlatformAscendC` 对象：平台信息指针是轻量原始事实，`PlatformAscendC` 是不可默认构造的包装，让调用点就地构造避免拷贝/赋值语义问题。
- `auto ascendcPlatform = PlatformAscendC(platformInfoPtr);` 直接初始化而非先声明后赋值——前者只需拷贝/移动构造可用，后者额外要求默认构造。

## 预防

- op_host 里所有 `platform_ascendc::PlatformAscendC` 一律从 `context->GetPlatformInfo()` 的指针就地构造，不要声明未初始化的对象成员或局部变量。
- 遇到 "use of deleted function" 类编译错，先查该类是否删除了默认构造，而不是绕道聚合初始化。

## 验证信号

以下信号来自该算子骨架的编译闭环：构造方式改写后重新编译，deleted function 报错消失；最终全量编译通过。

- 修复前：`error: use of deleted function 'platform_ascendc::PlatformAscendC::PlatformAscendC()'`（tiling.cpp:237）（来源平台：950；证据等级：tool_observed）。
- 修复后：该报错不再出现，host 侧（cust_optiling / cust_op_proto）最终 `Built target` 通过（来源平台：950；证据等级：tool_observed）。
- 全量回归：`build.sh --soc=ascend950 -j8` 生成 custom_opp_ubuntu_x86_64.run（694K）与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 基于 CANN 9.1.0 的类定义；若后续版本放开默认构造，本卡降级为风格建议（未验证，推断可迁移）。
- 本卡不覆盖 UT/Mock 场景下无 TilingContext 时的平台信息构造（faker 走 TilingContextPara 注入路径，未在本闭环验证，未验证，推断可迁移）。

**未决项**：

- 无。
## 坏实践

- `PlatformAscendC ascendcPlatform;` 再赋值——默认构造已被删除，编译直接失败。

## 误诊

- 无。
