---
type: "Runbook"
title: "507035 全量超时：GetInputStride 全 0 误判"
description: "CANN 9.1.0 tiling 上下文 gert::Stride 未填充（dim_num=0 全 0），stride 比对把所有输入误判为非连续触发 TilingKey=2，SyncAll 死锁报 507035；须加 stride 有效性回退。"
tags: [task.debugging, topic.tiling, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# 507035 全量超时：GetInputStride 全 0 误判

## 现象

在 Ascend950PR（CANN 9.1.0）上对 BenchNoHarnessSwiGlu 做迭代三联调验证：算子实现了双 TilingKey 路径——Host Tiling 新增 `DetectContiguity()`，用 `context->InputIsView(0)` + `GetInputStride(0)` 做 C-order stride 比对动态设置 tilingPath（连续=1 走原路径，非连续=2 走 kernel 内 CopyNonContiguous+SyncAll 拷贝到 workspace）。构建安装后在 tests/st 跑 Real（NPU）模式全量 115 条用例，**从最简单的连续输入 `[4,8] fp16 dim=-1` 开始全部失败**，报错原文：

```
[Real] builtin_fp16_2d_dim-1_[4,8]
  shape=[4,8], dtype=float16, dim=-1
  [FAIL] 流同步失败: 507035
```

部分 bf16 大 shape 用例报错前还输出全 0/错值（`期望=-0.256608, 实际=0.000000` 一类），CPU golden 自测 9/9 正常；Mock 模式此前 113/113 全过。CPU 侧 stdbuf/超时排查后确认不是测试挂死，而是每条 kernel 调用都流同步超时（507035 = kernel 执行超时）。

## 适用场景

- 工程形态：aclnn 自定义算子仓开发 + C++ ST Real（NPU）测试；Host Tiling 双路径（TilingKey 按输入连续性分流）设计。
- API 路径：`gert::TilingContext` 的 `InputIsView(index)` / `GetInputStride(index)` / `GetInputOffset(index)`（`exe_graph/runtime/tiling_context.h`）；kernel 侧 `SyncAll<true>(syncGM, syncBuf, blockDim)`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 tiling 上下文 stride 语义 + "全量 507035"这一类 tiling 误判故障画像。
- 路线优先级：recommended_fastpath（作为防御性修复；非连续输入的最终处置见独立卡：L2 API 层 Contiguous()）。
- 明确不适用 / 未验证场景：CANN 更新版本若已为 view tensor 填充 stride（未验证）；Mock/CPU golden 路径（不经过 tiling/kernel，不受影响）。

## 判别方法

1. **强制单路径隔离**：在 TilingFunc 里临时硬编码 `tilingPath = 1; isContiguous = 1;` 跳过检测，重编安装后重跑——若全量转 PASS，即可锁定问题在 TilingKey=2 路径被错误触发，而非 kernel 计算本身。
2. **Tiling 日志探针**：在 `DetectContiguity()` 的 stride 比对处加 `OP_LOGI`，输出每维 actual/expected stride 与最终 isView/isContiguous/tilingPath。
3. **plog 取证**：host tiling 的 OP_LOGI 落在**运行用户**家目录 `~/ascend/log/debug/plog/plog-<pid>_*.log`（本轨迹先查 /var/log/npu 与错误用户目录均扑空，最终在运行用户家目录命中；完整排障步骤见独立方法卡 [507035_tiling_path_bisection_plog](../runtime/507035_tiling_path_bisection_plog.md)）。

本轨迹实测日志（原文）：

```
DEBUG stride mismatch: dim=1, actual=0, expected=1, dimSize=8
DEBUG cOrderContiguous=0, rank=2
DEBUG DetectContiguity: isView=1, isContiguous=0, tilingPath=2, totalInput=32
```

判别要点：`actual=0` 且对**每个**输入（包括纯连续输入）都出现同样的 mismatch——说明 `GetStride(i)` 返回的不是真实逻辑 stride，而是默认构造的 `Stride()`（`dim_num_ = 0, strides_{0}`，见 `exe_graph/runtime/stride.h`）。

## 根因

CANN 9.1.0 中移除 op_def 输入的 `AutoContiguous()` 后，框架对 `aclCreateTensor` 创建的所有输入都生成 view tensor（`InputIsView(0)` 恒 true），但 **`GetInputStride(0)` 返回的 `gert::Stride` 未被填充**：`GetDimNum()==0`、`GetStride(i)` 恒 0。于是 C-order 比对 `strides->GetStride(i) != expectedStride` 对任何输入都在最内层维失配（0 != 1）→ 所有输入被判非连续 → `tilingPath=2` 恒触发 → kernel 走 `CopyNonContiguous()`，其中 `SyncAll<true>(syncGM, syncBuf.Get<int32_t>(), blockDim_)` 跨核屏障死锁（机制未单点复现）→ kernel 超时 → host 侧流同步报 507035。

注意这是**误判方向**问题（把连续判成非连续）；同一 API 缺陷的另一个方向（真非连续也拿不到 stride，被误判为连续致结果错）见独立非连续输入卡。

## 解决配方

修复前（`op_host/arch35/bench_no_harness_swi_glu_tiling.cpp` `DetectContiguity()`，verbatim，无 stride 有效性检查）：

```cpp
// 检测输入连续性并收集非连续信息
static NonContiguousInfo DetectContiguity(gert::TilingContext* context, uint64_t totalInput)
{
    NonContiguousInfo info;

    // 如果输入不携带非连续描述信息, 则为连续
    if (!context->InputIsView(0)) {
        info.isContiguous = true;
        return info;
    }

    // 获取 view shape
    auto inputShapeNode = context->GetInputShape(0);
    const auto& viewShape = inputShapeNode->GetOriginShape();
    int64_t rank = static_cast<int64_t>(viewShape.GetDimNum());
    info.inputRank = static_cast<uint32_t>(rank);

    // 获取 strides 和 offset
    const gert::Stride* strides = context->GetInputStride(0);
    int64_t offset = context->GetInputOffset(0);

    if (strides == nullptr) {
        // InputIsView 返回 true 但 strides 为空, 视为连续
        info.isContiguous = true;
        return info;
    }

    info.inputOffset = offset;
    for (int64_t i = 0; i < rank; i++) {
        info.inputShape[i] = viewShape.GetDim(i);
        info.inputStrides[i] = strides->GetStride(i);
    }
    // ...（其后为 C-order stride 比对，0 != 1 恒失配）
```

修复后（verbatim，新增 stride 有效性检查，`GetDimNum()==0` 或全 0 时回退为连续）：

```cpp
// 检测输入连续性并收集非连续信息
static NonContiguousInfo DetectContiguity(gert::TilingContext* context, uint64_t totalInput)
{
    NonContiguousInfo info;

    // 如果输入不携带非连续描述信息, 则为连续
    if (!context->InputIsView(0)) {
        info.isContiguous = true;
        return info;
    }

    // 获取 view shape
    auto inputShapeNode = context->GetInputShape(0);
    const auto& viewShape = inputShapeNode->GetOriginShape();
    int64_t rank = static_cast<int64_t>(viewShape.GetDimNum());
    info.inputRank = static_cast<uint32_t>(rank);

    // 获取 strides 和 offset
    const gert::Stride* strides = context->GetInputStride(0);
    int64_t offset = context->GetInputOffset(0);

    if (strides == nullptr) {
        // InputIsView 返回 true 但 strides 为空, 视为连续
        info.isContiguous = true;
        return info;
    }

    // 检查 stride 信息是否有效 (dim_num > 0 且不全为 0)
    // 当 AutoContiguous 被移除后, 框架可能为所有输入创建 view tensor,
    // 但 stride 信息可能未正确填充 (dim_num=0 或全 0)
    size_t strideDimNum = strides->GetDimNum();
    bool hasValidStride = false;
    if (strideDimNum > 0) {
        for (size_t i = 0; i < strideDimNum; i++) {
            if (strides->GetStride(i) != 0) {
                hasValidStride = true;
                break;
            }
        }
    }
    if (!hasValidStride) {
        // stride 信息不可用, 回退为连续判断
        info.isContiguous = true;
        return info;
    }

    info.inputOffset = offset;
    for (int64_t i = 0; i < rank; i++) {
        info.inputShape[i] = viewShape.GetDim(i);
        info.inputStrides[i] = strides->GetStride(i);
    }
    // ...（C-order 比对保留，仅在 stride 有效时到达）
```

同时删除临时强制路径与 OP_LOGI 调试探针。

## 设计决策

- 选"stride 无效即回退连续"而非继续信任比对结果：`dim_num=0/全 0` 是默认构造 `Stride()` 的确定性指纹，出现即说明框架没给数据，任何基于它的分支都不可靠。
- 保留 `DetectContiguity()` 与 TilingKey=2 kernel 代码路径不删：为将来 CANN 版本填充 stride 后启用留路（最终非连续处置改为 L2 API 层 Contiguous()，见独立卡）。
- 死锁点（SyncAll）未在本卡修复：修方向是"不让错误路径被触发"，SyncAll 本身的正确用法/死锁条件属遗留问题。

## 预防

- Tiling 层消费 `GetInputStride()` 前必做有效性检查（nullptr / `GetDimNum()==0` / 全 0 三态），无效时按保守语义（连续）回退。
- 上板前先想清楚 `InputIsView()` 的语义是"携带 view 描述"而非"逻辑上非连续"——两者在本版本解耦。
- 排查 507035 类全量超时：先强制 tiling 单路径隔离（见判别方法 1），再上 plog 探针。

## 坏实践

- 见到 `InputIsView(0)==true` 就认定输入非连续并触发特殊路径——本版本该标志对连续输入也为 true。
- 在 tiling 日志探针还没确认 stride 数值前，先入为主怀疑 kernel 算法或测试数据。

## 误诊

- 初始怀疑"大 shape 用例太慢导致超时"：300s→600s 加时后依然 Exit 124 且日志显示逐条 507035，排除（来源平台：950；证据等级：tool_observed）。
- 初始怀疑测试二进制无输出是挂死：`stdbuf -o0` 解决 stdout 缓冲后看到正常输出流，锁定问题在 kernel 执行而非 harness（来源平台：950；证据等级：tool_observed）。

## 验证信号

来源：迭代三联调验证会话，507035 全量失败 → 强制路径隔离 → plog 探针 → stride 回退修复 → 重跑。

- 修复前：Real 115 条全 `[FAIL] 流同步失败: 507035`（含最简连续用例），首条即失败（来源平台：950；证据等级：tool_observed）。
- 隔离探针：强制 `tilingPath=1` 后同套 Real 用例转 PASS（首条 `builtin_fp16_2d_dim-1_[4,8]` MERE=2.22e-05 通过）（来源平台：950；证据等级：tool_observed）。
- plog 实测 `DEBUG stride mismatch: dim=1, actual=0, expected=1` 且对所有输入复现，`DetectContiguity: isView=1, isContiguous=0, tilingPath=2`（来源平台：950；证据等级：tool_observed）。
- 修复后重跑：507035 全部消失，前 71 条结果 0 FAIL（大 shape 耗时长分批跑完）；全量 110 PASS + 12 FAIL——剩余 12 条失败全部是**非连续用例**的 MERE 超阈（0.09~0.20），属另一根因（见独立非连续输入卡），本卡的超时闭环已消除（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡修复的是"误判为非连续导致死锁"；真非连续输入在 stride 不可用时同样被判连续而**结果错误**，本卡不解决（独立 Contiguous() 卡处置）。
- stride 有效性指纹（dim_num=0/全 0）取自 CANN 9.1.0 实测；若合法输入存在"某维 stride 真为 0"（广播语义）会被本检查误回退为连续，本轨迹未涉及广播输入（未验证场景：广播/扩展维输入）。
- SyncAll 死锁机制未复现到指令级，"barrier 等不到对端"是推断。
- 仅 Ascend950PR + CANN 9.1.0 实测；其他平台/版本未验证（推断可迁移）。

**未决项**：

- 优先项：在能拿到有效 stride 的 CANN 版本上复现 `CopyNonContiguous()` + `SyncAll<true>` 组合，把死锁条件（blockDim 不齐/同步区初始化时序/MTE 与 barrier 交错）追到机制层。
- 框架为何创建 view tensor 却不填 stride，未定位到 CANN 内部实现层。
