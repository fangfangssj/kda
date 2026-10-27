---
type: "Runbook"
title: arch35 Cube 内联函数误入 Vector 通道编译失败
description: "arch35 Cube:Vector=1:2，构建为 Cube/Vector 通道分别生成编译目标；Cube 内联函数（set_padding 等）在 Vector 通道编译报 \"does not support the given target feature\"，须用 #if defined(SPLIT_CORE_CUBE) 预处理守卫。"
tags: [task.debugging, topic.compilation, topic.architecture]
status: stable
sources: []
created_at: '2026-09-16T11:15:37Z'
updated_at: '2026-09-16T11:15:37Z'
platforms: ["950"]
---

# arch35 Cube 内联函数误入 Vector 通道编译失败

## 现象

在 BenchNoHarnessConv2d 算子的 arch35 kernel 里加入 Cube 路径（LoadData3DParamsV1/Load3D、MMAD、Fixpipe，以及 A1/B1/L0A/L0B/L0C 的 TBuf 与 InitBuffer）后，执行 `bash build.sh --soc=ascend950 -j8`，kernel_0 目标报出 CANN 内部头文件的编译错误（报错位置在框架头文件而非业务代码，容易误判为环境问题）：

```
<CANN_HOME>/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_mm_impl.h:839:5: error: function type 'void (unsigned long) noexcept' of 'set_padding' does not support the given target feature
<CANN_HOME>/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_mm_impl.h:839:5: error: function type 'void (short) noexcept' of 'set_padding' does not support the given target feature
```

同时另外两个编译目标报 tilingkey 模板实例化错误（那是另一个根因，另行闭环）。当时代码状态：ComputeMBlock 函数内 Cube 操作（Load3D/MMAD/Fixpipe）与 Vector 操作（Add/DataCopyPad）混写在同一个无守卫的函数体里。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel，arch35（DAV_3510）上同时包含 Cube 操作（Load3D/Load2D/MMAD/Fixpipe）与 Vector 操作（Add/DataCopyPad）的 mixed kernel。
- API 路径：`SPLIT_CORE_CUBE`/`SPLIT_CORE_VEC` 预处理宏、`ASCEND_IS_AIC`/`ASCEND_IS_AIV`、`g_coreType`（`kernel_operator_data_copy_intf_impl.h`、`sys_macros.h` 等 CANN 头文件）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.1.0。
- 适用粒度：platform_wide（arch35 所有含 Cube 内联函数的 kernel）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：纯 Vector 算子（不触发）；非 arch35 架构的 Cube/Vector 划分机制（未验证，机制推断不迁移）。

## 根因

arch35 上 Cube:Vector = 1:2，构建系统按代码通道分别生成编译目标；Cube 专属内联函数（Load3D 的 `set_padding` 等，位于 `dav_3510/kernel_operator_mm_impl.h`）在 Vector 通道目标里编译时，后端无法为其选择指令，报 "does not support the given target feature"。

机制链（本会话实测证据）：

1. `ASCEND_IS_AIC` 不是编译期开关，而是运行时常量判断——头文件中为 `extern int32_t g_coreType;` 与 `#define ASCEND_IS_AIV ...` 形式（`constexpr(g_coreType == AscendC::AIC)` 语义）；对**编译目标选择**无效，只对运行时分支有效。
2. 通道由预处理宏决定：`__DAV_CUBE__` 定义 → `SPLIT_CORE_CUBE` → 该目标按 Cube 编译；`__DAV_VEC__` → `SPLIT_CORE_VEC` → 按 Vector 编译。
3. 因此把 Cube 内联函数排除出 Vector 目标必须在**预处理级**做：`#if defined(SPLIT_CORE_CUBE)`。

诊断探针（可复用）：在 kernel 源文件顶部临时插入 `#error`/`#warning` 输出当前宏状态再构建，本会话实测输出 `=== SPLIT_CORE_VEC defined (Vector channel) ===` 出现两次（kernel_1/kernel_2），证实多目标按通道区分、Cube 内联函数确实落进了 Vector 目标。

## 解决配方

修复前（无守卫，Cube/Vector 混写；引发 set_padding 报错的形态）：

```cpp
template <typename D_T_X, int TILING_MODE>
__aicore__ inline void Conv2d<D_T_X, TILING_MODE>::ComputeMBlock(
    int64_t mBlockIdx, int64_t batchIdx, int64_t cOutTileIdx)
{
    // Cube 操作与 Vector 操作混写，无任何通道守卫
    LocalTensor<D_T_X> a1Tensor = l1FeatureMapBuf_.Get<D_T_X>();
    ... LoadData3DParamsV1 / DataCopy(a2Tensor, a1Tensor, load3dParams) / Mmad / Fixpipe ...
    Add(resultUB, fixpipeUB, biasUB, ...);
    DataCopyPad(...);
}
```

第一次尝试（运行时守卫，不够）：用 `if ASCEND_IS_AIC { ... Cube 操作 ... }` 包住 Cube 代码——set_padding 错误消失，但引出连锁问题：跨通道使用的变量（如 `fixpipeUB`）在守卫内外重复声明/未声明（`use of undeclared identifier 'fixpipeUB'` / `redefinition of 'fixpipeUB'`）。

最终修复（预处理守卫 + 声明布局，逐层应用到 TBuf 声明、InitBuffer、LoadFeatureMap/LoadFilter、MTE2_MTE1 同步）：

```cpp
    // arch35 (DAV_3510): Cube:Vector=1:2，Cube 操作需在 Cube 通道编译
    // 使用 #if defined(SPLIT_CORE_CUBE) 在预处理器级别排除 Vector 通道的 Cube 内联函数
    // Cube 核: Load3D + Load2D + MMAD + Fixpipe
    // Vector 核: Add + DataCopyPad

    LocalTensor<D_T_X> fixpipeUB = ubFixpipeBuf_.Get<D_T_X>();   // 跨通道使用 → 声明放守卫外

#if defined(SPLIT_CORE_CUBE)
    // ===== 1. Load3D: L1/A1 → L0A/A2 (硬件 im2col) =====
    LocalTensor<D_T_X> a1Tensor = l1FeatureMapBuf_.Get<D_T_X>();
    LocalTensor<D_T_X> a2Tensor = l0aBuf_.Get<D_T_X>();
    ...
#endif
```

Cube 专属 TBuf 的 InitBuffer 同样守卫（UB buffer 保持两通道共用）：

```cpp
    // 分配 Cube 专用 buffer（仅 Cube 通道）
#if defined(SPLIT_CORE_CUBE)
    // A1: [C1, H, W, C0] NC1HWC0
    uint32_t a1Size = static_cast<uint32_t>(c1_ * H_ * W_ * c0_ * sizeof(D_T_X));
    pipe_.InitBuffer(l1FeatureMapBuf_, a1Size);
    ...B1/L0A/L0B/L0C 同理...
#endif

    // UB buffer（Cube 和 Vector 通道共用）
    uint32_t ubFixSize = static_cast<uint32_t>(M_BLOCK_SIZE * nTile_ * sizeof(D_T_X));
    pipe_.InitBuffer(ubFixpipeBuf_, ubFixSize);
```

使用 L1 buffer 的搬运函数整函数守卫：

```cpp
template <typename D_T_X, int TILING_MODE>
__aicore__ inline void Conv2d<D_T_X, TILING_MODE>::LoadFeatureMap(int64_t batchIdx)
{
#if defined(SPLIT_CORE_CUBE)
    // DataCopy(Nd2Nz): GM(x[batch]) → L1/A1 [C1, H, W, C0]
    LocalTensor<D_T_X> a1Tensor = l1FeatureMapBuf_.Get<D_T_X>();

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = static_cast<uint16_t>(CIn_);
    nd2nzParams.dValue = static_cast<uint32_t>(H_ * W_);
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = static_cast<uint64_t>(H_ * W_);
    nd2nzParams.dstNzC0Stride = static_cast<uint16_t>(H_ * W_);
    nd2nzParams.dstNzNStride = static_cast<uint16_t>(c0_ * H_ * W_);
    nd2nzParams.dstNzMatrixStride = 0;

    uint64_t gmOffset = static_cast<uint64_t>(batchIdx) * CIn_ * H_ * W_;
    DataCopy(a1Tensor, xGM_[gmOffset], nd2nzParams);
#endif
}
```

Cube 专属同步（MTE2→MTE1）守卫：

```cpp
    // 1. 加载特征图 GM → L1/A1
    LoadFeatureMap(batchIdx);

    // 2. 加载 filter workspace → L1/B1
    LoadFilter(cOutTileIdx);

#if defined(SPLIT_CORE_CUBE)
    // MTE2 → MTE1 同步（等待 L1 数据就绪）
    event_t mte2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    SetFlag<HardEvent::MTE2_MTE1>(mte2Event);
    WaitFlag<HardEvent::MTE2_MTE1>(mte2Event);
#endif
```

## 设计决策

- 为什么用 `#if defined(SPLIT_CORE_CUBE)` 而不是 `if ASCEND_IS_AIC`：后者是运行时判断（`g_coreType`），不能阻止 Cube 内联函数进入 Vector 目标的编译单元；只有预处理守卫能把 Cube 代码从 Vector 目标里物理排除。运行时守卫可作为语义兜底但解决不了本编译错误。
- 变量声明布局：跨通道使用的变量（UB 相关）放守卫外，Cube 专属变量放守卫内——第一次尝试把 `fixpipeUB` 同时声明在守卫内外导致 redefinition，收敛为"声明外提、使用点留在各通道分支"。
- 当前方案路线档位：recommended_fastpath（mixed kernel 在 arch35 编译的必要结构）。

## 预防

- arch35 上写含 Cube 操作的 kernel，第一版就把 Load3D/Load2D/MMAD/Fixpipe、L1/L0 的 TBuf 与 InitBuffer、MTE2_MTE1 同步全部包进 `#if defined(SPLIT_CORE_CUBE)`；Vector 操作（Add/DataCopyPad 等）留在守卫外。
- 报 "does not support the given target feature" 且位置在 dav_3510 框架头文件时，优先怀疑"Cube 内联函数进了 Vector 通道"，用 `#error` 探针确认当前目标的 SPLIT_CORE_* 宏。

## 验证信号

以下信号来自该算子 arch35 kernel 的 SPLIT_CORE 排障会话：加守卫后逐轮构建对照。

- 修复前：kernel_0 报 `kernel_operator_mm_impl.h:839:5: error: function type ... of 'set_padding' does not support the given target feature`（来源平台：950；证据等级：tool_observed）。
- 运行时守卫方案后同轮构建：set_padding 错误消失，错误变为 `fixpipeUB` 未声明/重复定义（守卫边界问题的连锁，非本根因）（来源平台：950；证据等级：tool_observed）。
- 预处理守卫 + 声明外提后：`redefinition of 'fixpipeUB'` 消失，构建错误只剩 tilingkey 模板实例化（另一根因，另行闭环）——本根因闭环（来源平台：950；证据等级：tool_observed）。
- 探针输出：kernel_1/kernel_2 两个目标各打印一次 `=== SPLIT_CORE_VEC defined (Vector channel) ===`，证实通道划分（来源平台：950；证据等级：tool_observed）。
- CANN 头文件锚点：`kernel_operator_data_copy_intf_impl.h` 中 `#if (__NPU_ARCH__ != 5102) if ASCEND_IS_AIC { return; }`（DataCopyPad 的 AIC 早退）；头文件 `extern int32_t g_coreType;` 与 `ASCEND_IS_AIV` 宏定义（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `__DAV_CUBE__`/`__DAV_VEC__` → `SPLIT_CORE_CUBE`/`SPLIT_CORE_VEC` 的映射与 Cube:Vector=1:2 均为 arch35（DAV_3510）机制；其他架构未验证（未验证，机制推断不迁移）。
- 本卡只闭合"编译失败"根因；被守卫的 Cube+Vector 协同（Fixpipe→UB→Add 的跨通道同步正确性）在该骨架阶段未上板验证。
- 纯 Vector 骨架方案（后续版本采用）不触发本问题，但恢复 Cube 路径时本守卫结构仍是必需。

**未决项**：

- 无（本根因已闭环到机制层；被守卫代码的功能正确性属后续 Cube 路径范围）。
## 坏实践

- 混合 kernel 里 Cube/Vector 操作不加任何通道守卫直接写在一个函数体里。
- 看到 "does not support the given target feature" 报在 CANN 框架头文件就怀疑 CANN 安装损坏去重装。
- 用 `if ASCEND_IS_AIC` 试图解决编译目标选择问题（它是运行时判断）。

## 误诊

- 曾把同轮构建的 tilingkey 错误与本错误归为同一根因（"SPLIT_CORE 模式 MIX/VEC/VEC 通道问题"）——后续实测 3 个编译目标实际是 3 种 dtype 的实例化目标，tilingkey 错误另有根因（ASCENDC_TPL dtype 声明不完整），两卡分开。
