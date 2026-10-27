---
type: "Runbook"
title: "RmsNormTiling 与 CANN 内置类型冲突 — 改名隔离"
description: "CANN kernel_tiling.h 已导出 RmsNormTiling，自定义同名结构体会触发声明冲突与引用歧义；给本地类型增加区分后缀可恢复编译。"
tags: ["task.debugging", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# RmsNormTiling 与 CANN 内置类型冲突 — 改名隔离

## 现象

在为 Ascend950 Flash 直调算子编译单文件 `operators/rms_norm/rms_norm.asc` 时，host tiling 区域自行声明了全局 `struct RmsNormTiling`。远程 CANN 构建在处理该声明和 helper 返回类型时失败，原始诊断包含：

```text
error: declaration conflicts with target of using declaration already in scope
struct RmsNormTiling {
^
.../include/kernel_tiling/kernel_tiling.h:565:8: note: target of using declaration
struct RmsNormTiling {
       ^
.../include/kernel_tiling/kernel_tiling.h:704:24: note: using declaration
using AscendC::tiling::RmsNormTiling;
                       ^
error: reference to 'RmsNormTiling' is ambiguous
```

同一诊断随后把 helper 的返回类型和局部对象声明都标为 ambiguous，并产生 `res` 未声明的级联错误。此时 dtype 派发导致的上一类模板错误已经消失，故可以把当前失败单独定位到 tiling 类型名。

## 适用场景

- 工程形态：Ascend C kernel 直调工程的 host tiling 编译；
- API 路径：`.asc` 同时包含 Torch host 入口、`kernel_operator.h`/`kernel_tiling.h` 与自定义 host tiling helper；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957c / dav-3510）+ CANN 9.0.1；
- 适用粒度：`platform_wide`，约束来自公共 CANN 头文件导出的类型名，不依赖具体算子公式；
- 路线优先级：`recommended_fastpath`，自定义类型使用 op 前缀或 `Params`/`Args` 后缀，避免占用 CANN 已导出的 `*Tiling` 名；
- 明确不适用或未验证场景：未引入 `kernel_tiling.h` 的纯 device 文件不适用；a2、a3 及其他 CANN 版本未验证。

## 根因

CANN 9.0.1 的 `kernel_tiling.h` 已把 `AscendC::tiling::RmsNormTiling` 通过 `using` 导入当前作用域，自定义全局同名结构体因而同时触发声明冲突与名称查找歧义。

## 解决配方

不要直接用算子 PascalCase 名拼接 `Tiling` 作为全局自定义结构体名；给本地承载对象增加 `Params`/`Args` 后缀或 op 前缀，并同步修改 helper 的返回类型和局部对象类型。

### before

代码证据：revision=`trace-state/pre-rename`；path=`operators/rms_norm/rms_norm.asc`；symbols=`RmsNormTiling`, `calc_rms_norm_tiling_params`；lines=`308-347`。以下结构体和完整函数由初版完整写入及修复前 Edit 状态逐字恢复：

```cpp
struct RmsNormTiling {
    int64_t numBlocks;
    int64_t blockRowLength;
    uint32_t tileRows;
    uint32_t rowStride;
    float avgFactor;
};

static RmsNormTiling calc_rms_norm_tiling_params(int64_t numRow, int64_t numCol, int32_t sizeofT)
{
    RmsNormTiling res;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();

    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) ubSize = 256ull * 1024;

    int64_t rowStride = (numCol + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN; // elements
    int64_t gammaBytes = rowStride * sizeofT;
    int64_t denom = 4 * rowStride * sizeofT; // xQ(d2)+yQ(d2) per row
    int64_t tileRows = (static_cast<int64_t>(ubSize) - UB_HEADROOM - gammaBytes) / denom;
    if (tileRows > MAX_TILE_ROWS) tileRows = MAX_TILE_ROWS;
    if (tileRows < 1) tileRows = 1;

    // ACC-006 (adapt_identifiers), SHA 673c790bb3b485fa61a0d4462df83d41f00be7a520734045bda3408be6eb8c60
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, numRow);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockRowLength = (numRow + numBlocks - 1) / numBlocks;
    if (tileRows > blockRowLength) tileRows = blockRowLength;
    if (tileRows < 1) tileRows = 1;

    res.numBlocks = numBlocks;
    res.blockRowLength = blockRowLength;
    res.tileRows = static_cast<uint32_t>(tileRows);
    res.rowStride = static_cast<uint32_t>(rowStride);
    res.avgFactor = (numCol == 0) ? 0.0f : 1.0f / static_cast<float>(numCol);
    return res;
}
```

### after

代码证据：revision=`trace-state/post-rename`；path=`operators/rms_norm/rms_norm.asc`；symbols=`RmsNormTileParams`, `calc_rms_norm_tiling_params`；lines=`308-347`。以下结构体和完整函数由精确替换命令后的工具输出逐字重建；除三个类型名引用外没有改动：

```cpp
struct RmsNormTileParams {
    int64_t numBlocks;
    int64_t blockRowLength;
    uint32_t tileRows;
    uint32_t rowStride;
    float avgFactor;
};

static RmsNormTileParams calc_rms_norm_tiling_params(int64_t numRow, int64_t numCol, int32_t sizeofT)
{
    RmsNormTileParams res;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();

    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) ubSize = 256ull * 1024;

    int64_t rowStride = (numCol + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN; // elements
    int64_t gammaBytes = rowStride * sizeofT;
    int64_t denom = 4 * rowStride * sizeofT; // xQ(d2)+yQ(d2) per row
    int64_t tileRows = (static_cast<int64_t>(ubSize) - UB_HEADROOM - gammaBytes) / denom;
    if (tileRows > MAX_TILE_ROWS) tileRows = MAX_TILE_ROWS;
    if (tileRows < 1) tileRows = 1;

    // ACC-006 (adapt_identifiers), SHA 673c790bb3b485fa61a0d4462df83d41f00be7a520734045bda3408be6eb8c60
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, numRow);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockRowLength = (numRow + numBlocks - 1) / numBlocks;
    if (tileRows > blockRowLength) tileRows = blockRowLength;
    if (tileRows < 1) tileRows = 1;

    res.numBlocks = numBlocks;
    res.blockRowLength = blockRowLength;
    res.tileRows = static_cast<uint32_t>(tileRows);
    res.rowStride = static_cast<uint32_t>(rowStride);
    res.avgFactor = (numCol == 0) ? 0.0f : 1.0f / static_cast<float>(numCol);
    return res;
}
```

## 设计决策

- 选择 `RmsNormTileParams`：保留与算子相关的可读前缀，同时用 `Params` 明确它只是本文件 helper 的返回值，不宣称是 CANN schema 类型；
- 放弃继续使用 `RmsNormTiling`：编译器已经给出公共头文件中的同名声明和 `using` 路径，限定局部引用也不能消除全局重复声明；
- 当前路线为 `recommended_fastpath`：只改变 C++ 类型标识符，不改变 tiling 数值、UB 预算、多核切分或 kernel ABI。

## 系统信息

- toolchain：CANN 9.0.1，编译输出路径显示 bisheng/clang 15.0.5 头文件集合；
- 目标：Ascend950PR_957c / dav-3510，平台注册值 950；
- 工程：Torch 自定义算子与 Ascend C device/host 代码共置于 `operators/rms_norm/rms_norm.asc`；
- 相关文件：仅修改 `operators/rms_norm/rms_norm.asc`，未改测试、golden 或评测 harness。

## 预防

- 声明全局 host tiling 类型前，检查 CANN 公共头是否已导出同名 `*Tiling`；
- 自定义仅供 helper 返回的结构体优先命名为 `<Op>TileParams`、`<Op>TilingArgs` 或放入独立命名空间；
- 看到 `target of using declaration` 与 `reference ... is ambiguous` 同时出现时，先沿诊断中的头文件和 `using` 路径核对符号冲突，不要继续修改 tiling 算法。

## 验证信号

来源说明：这条知识来自一次 Ascend950 rms_norm 直调工程的远程编译修复；第二次构建明确暴露 CANN 同名类型，精确重命名后立即重建并继续执行真实 NPU 验证。

- 重命名后的远程构建输出 `[100%] Built target op_rms_norm`，生成 `libop_rms_norm.so`（来源平台：950；证据等级：tool_observed）；
- 同一修复源码随后执行的 pytest 为 29 passed（来源平台：950；证据等级：tool_observed）；
- 同一修复源码随后执行的 cann-bench 正确性为 20/20 通过（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 只在 CANN 9.0.1、dav-3510 头文件集合中实测；其他 CANN 版本是否仍导出同名类型未验证；
- 如果自定义类型位于不会与该 `using` 相遇的隔离命名空间，可能不触发同一诊断，但仍应避免与公共 CANN 类型同名；
- a2、a3 未验证，不写入 `platforms`。

## 坏实践

- 把级联的 `res is undeclared` 当成 helper 内变量生命周期问题：该错误是返回类型和局部类型声明歧义后的次生结果；
- 只修改 helper 的局部变量名而保留全局 `struct RmsNormTiling`：公共头文件的声明冲突仍然存在；
- 为了让构建通过而改测试、改 golden 或跳过 dtype：这些操作不触及编译期名称冲突，本轨迹也没有采用。

## 误诊

第一次失败日志还同时包含局部变量 `t` 与模板 dtype 分支错误；必须在这些问题修掉后的下一次构建中读取剩余原始诊断，不能把三类独立根因合并成一张卡。
