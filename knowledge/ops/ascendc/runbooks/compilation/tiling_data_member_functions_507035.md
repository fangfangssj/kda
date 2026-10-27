---
type: "Runbook"
title: "TilingData加成员函数致507035"
description: TilingData 结构体加入 SetSpatialScale/GetSpatialScale 成员函数与头文件包含后 kernel 全量 507035 流同步失败；改纯 int64_t 字段 + union bitcast 传 float 恢复。
tags: [task.debugging, topic.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# TilingData加成员函数致507035

## 现象

BenchNoHarnessRoiAlign kernel 联调中排查"属性疑似不生效"时，把 TilingData 结构体从"float 字段 + 显式 int32_t 填充"改为"`spatialScaleBits`（int64_t）+ `SetSpatialScale/GetSpatialScale` 成员函数（内部 std::memcpy 做 float↔bits 转换）+ `#include <cstdint>/<cstring>`"，并在 tiling 侧调用 `tiling->SetSpatialScale(spatialScale)`、kernel 侧调用 `tilingData->GetSpatialScale()`。重建重装后跑 ST Real，golden 自测正常，但全部 8 个用例统一失败：

```
[FAIL] 流同步失败: ret=507035
```

即 kernel 执行即崩溃（AIV 异常）。两个中间尝试均无效：把成员函数里的 `std::memcpy` 换成 union 位转换（仍然保留成员函数）→ 仍 507035；去掉 `<cstring>` include → 仍 507035。最终把结构体改回**纯 int64_t 字段、无成员函数、无 include**（host/kernel 两侧各自用 union bitcast 读写 `spatialScaleBits`）后，507035 消失，kernel 恢复正常执行。

## 适用场景

- 工程形态：AscendC 自定义算子开发，手写 C++ struct 作为 TilingData（经 `GET_TILING_DATA_WITH_STRUCT` 在 kernel 侧还原）。
- API 路径：`GET_TILING_DATA_WITH_STRUCT`（由 `asc_op_compile_base/asc_op_compiler/get_op_tiling.py` 生成：`uint8_t data[sizeof(tiling_struct)]` mocker + `InitTilingData<tiling_struct>` + `reinterpret_cast`）、`context->GetTilingData<T>()`、错误码 507035（流同步失败/AICore 异常）。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有手写 TilingData struct 的 AscendC 算子。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：旧版 `BEGIN_TILING_DATA_DEF` 宏体系（本仓未用，未验证）；类内默认初始化器在该平台已有独立实证兼容（见库内 `tilingdata_dual_struct_nsmdi_registration`）；崩溃的确切触发条件（成员函数本身？非 POD 化？include 拉入的宿主头？）未定位，见失效边界。

## 根因

TilingData 结构体非纯 POD 化（加入成员函数与 `<cstdint>/<cstring>` 等 include）后，经 `GET_TILING_DATA_WITH_STRUCT` 的字节 mocker 还原路径在 kernel 侧触发 AIV 异常（507035）。机制层根因未验证——成员函数理论上不改变 sizeof/布局，实际崩溃的确切原因（编译器对该 struct 在 kernel 侧的处理差异、REGISTER_TILINGDATA_SIZE 注册路径等）没有追到；行为层结论由"加成员函数→崩、去成员函数→恢复"的对照支撑。

## 解决配方

TilingData 保持**纯 int64_t 字段的 POD struct**：无成员函数、无 STL include；float 型属性以位模式（bit pattern）存进 int64_t 字段，host 侧写入、kernel 侧读出时各自用 union 做 bitcast。

修复前（崩溃版 `op_kernel/arch35/bench_no_harness_roi_align_tiling_data.h`）：

```cpp
#ifndef _BENCH_NO_HARNESS_ROI_ALIGN_TILING_DATA_H_
#define _BENCH_NO_HARNESS_ROI_ALIGN_TILING_DATA_H_

#include <cstdint>
#include <cstring>

struct BenchNoHarnessRoiAlignTilingData {
    // === 平台与多核切分 ===
    int64_t numBoxes = 0;       // ROI 总数
    int64_t blockDim = 0;       // 多核核数
    int64_t boxesPerCore = 0;   // 每核处理的 ROI 数（尾核可能少）

    // === 特征图维度 ===
    int64_t B = 0;              // batch size
    int64_t C = 0;              // 通道数
    int64_t H = 0;              // 特征图高
    int64_t W = 0;              // 特征图宽

    // === 输出维度 ===
    int64_t outputHeight = 0;   // 输出高 (oh)
    int64_t outputWidth = 0;    // 输出宽 (ow)

    // === 属性参数 ===
    // spatialScale 以 int64_t 存储 float 的 bit pattern, 避免混合类型 padding 问题
    // 使用 union 确保_host_和_kernel_编译器布局一致
    int64_t spatialScaleBits = 0;  // float spatialScale 的 bit pattern
    int64_t samplingRatio = 0;     // 采样比率 (-1/0=自动, >0=固定)
    int64_t aligned = 0;           // 是否对齐 (0=false, 1=true)

    // === UB 切分参数 ===
    int64_t cTile = 0;          // 通道块大小
    int64_t ohTile = 0;         // oh 块大小

    // 辅助方法: 设置/获取 spatialScale (float)
    void SetSpatialScale(float val) {
        int32_t bits = 0;
        std::memcpy(&bits, &val, sizeof(float));
        spatialScaleBits = static_cast<int64_t>(bits);
    }
    float GetSpatialScale() const {
        int32_t bits = static_cast<int32_t>(spatialScaleBits);
        float val = 0.0f;
        std::memcpy(&val, &bits, sizeof(float));
        return val;
    }
};

#endif // _BENCH_NO_HARNESS_ROI_ALIGN_TILING_DATA_H_
```

修复后（纯 POD，`op_kernel/arch35/bench_no_harness_roi_align_tiling_data.h`）：

```cpp
#ifndef _BENCH_NO_HARNESS_ROI_ALIGN_TILING_DATA_H_
#define _BENCH_NO_HARNESS_ROI_ALIGN_TILING_DATA_H_

struct BenchNoHarnessRoiAlignTilingData {
    // 全部 int64_t, 避免 float/int32_t 混合 padding 不一致
    int64_t numBoxes = 0;
    int64_t blockDim = 0;
    int64_t boxesPerCore = 0;
    int64_t B = 0;
    int64_t C = 0;
    int64_t H = 0;
    int64_t W = 0;
    int64_t outputHeight = 0;
    int64_t outputWidth = 0;
    int64_t spatialScaleBits = 0;  // float spatialScale 的 bit pattern (低32位)
    int64_t samplingRatio = 0;
    int64_t aligned = 0;           // 0=false, 1=true
    int64_t cTile = 0;
    int64_t ohTile = 0;
};

#endif // _BENCH_NO_HARNESS_ROI_ALIGN_TILING_DATA_H_
```

host 侧 tiling 写入（`op_host/arch35/bench_no_harness_roi_align_tiling.cpp`）：

```cpp
    tiling->numBoxes = numBoxes;
    tiling->blockDim = blockDim;
    tiling->boxesPerCore = boxesPerCore;
    tiling->B = B;
    tiling->C = C;
    tiling->H = H;
    tiling->W = W;
    tiling->outputHeight = outputHeight;
    tiling->outputWidth = outputWidth;
    // spatialScale 以 bit pattern 存储 (避免 float/int64_t 混合 padding)
    {
        union { int32_t i; float f; } u;
        u.f = spatialScale;
        tiling->spatialScaleBits = static_cast<int64_t>(u.i);
    }
    tiling->samplingRatio = samplingRatio;
    tiling->aligned = aligned ? 1 : 0;
    tiling->cTile = cTile;
    tiling->ohTile = ohTile;
```

kernel 侧读出（`op_kernel/arch35/bench_no_harness_roi_align.h` 的 Init）：

```cpp
    // spatialScale 从 bit pattern 还原
    {
        union { int32_t i; float f; } u;
        u.i = static_cast<int32_t>(tilingData->spatialScaleBits);
        spatialScale_ = u.f;
    }
```

## 判别方法

- 信号特征：改动 TilingData 结构体后，全部用例统一 `流同步失败: ret=507035`（kernel 启动即崩，无任何精度输出），且与 shape/attrs 无关。
- 二分定位：把"成员函数 / include / 混合字段类型"逐个减掉重建——本例减到纯 int64_t POD 即恢复；保留成员函数仅换 memcpy→union、或仅去 `<cstring>` 都不恢复，说明不是单个转换语句的问题。
- 回归确认：恢复后同批用例回到改动前的 MERE 水平（无崩溃），即结构体改动是唯一变量。

## 设计决策

- 为什么 float 不直接存 float 字段而要 bitcast：本算子早前把 `bool aligned` 改 int64_t、给 float 加显式填充是在"host/kernel 编译器 padding 可能不一致"的假设下做的防御；统一全 int64_t + bitcast 一劳永逸消除混合宽度风险，代价只是两行 union。
- 为什么不在 struct 里提供 accessor：成员函数是非 POD 化的第一嫌疑（虽未证实机制），而 host/kernel 两侧各两行 union 的成本可忽略。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0。
- 相关文件：`op_kernel/arch35/bench_no_harness_roi_align_tiling_data.h`；`op_host/arch35/bench_no_harness_roi_align_tiling.cpp`（写入）；`op_kernel/arch35/bench_no_harness_roi_align.h`（Init 读出）；宏生成器 `python/site-packages/asc_op_compile_base/asc_op_compiler/get_op_tiling.py`（GET_TILING_DATA_WITH_STRUCT 定义，只读）。

## 预防

- TilingData struct 只放纯数据字段（推荐全 int64_t），不写成员函数、不 include STL 头；float 属性走 bitcast。
- 对 TilingData 的任何"语法糖"改动（accessor、构造函数、继承）都先跑一遍最小 ST 冒烟再继续。

## 验证信号

来源说明：以下信号来自该算子 kernel 精度调试会话（ST Real 全量 8 用例，每次改动均重建+重装后运行）：

- 崩溃版（成员函数 + `<cstdint>/<cstring>`，memcpy 实现）：8/8 `流同步失败: ret=507035`（来源平台：950；证据等级：tool_observed）。
- 中间版（成员函数保留，memcpy→union，去 `<cstring>`）：仍 8/8 `ret=507035`（来源平台：950；证据等级：tool_observed）。
- 修复版（纯 int64_t POD + 两侧 union bitcast）：507035 消失，8 用例恢复 MERE 输出（批量模式下 MERE≈1.76e-05 量级；当时残余的 aligned 用例大误差属同进程 TilingData 缓存问题，见库内 `tilingdata_shape`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 崩溃的确切触发因子未定位：成员函数、非平凡构造语义、include 链三者中谁是必要条件未知；"纯 POD 即安全"是充分经验，不是机制结论。
- 仅在 Ascend950PR + CANN 9.1.0 实测；其他版本未验证（推断可迁移，未验证）。
- 全 int64_t 会放大 tiling buffer 占用（每字段 8B）；字段极多时需评估 tiling 区容量（本例 14 字段 112B，无压力）。

**未决项**：

- 【优先】复现并追根因到机制层：用最小算子逐项加回成员函数/include，确认 507035 的必要触发条件（成员函数 vs include vs 默认成员初始化器）。
- `GET_TILING_DATA_WITH_STRUCT` 的 `REGISTER_TILINGDATA_SIZE` 注册路径对非 POD struct 的处理是否是崩溃点，未验证。
## 坏实践

- 给 TilingData struct 加便利 accessor/STL 依赖追求代码整洁——该 struct 是 host/kernel 间的裸字节契约，不是普通 C++ 类。
- 用 `std::memcpy` 在 kernel 头文件里做类型双关——即使编译通过，也把宿主库依赖带进 kernel 编译单元。

## 误诊

- 起初怀疑 `GetSpatialScale()` 里的 `std::memcpy` 在 kernel 不可用导致崩溃，改成 union 后仍崩——排除了单一语句归因；真因是结构体整体非 POD 化（机制未定）。
