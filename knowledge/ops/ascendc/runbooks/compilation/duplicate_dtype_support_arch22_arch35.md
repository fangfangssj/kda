---
type: "Runbook"
title: "Duplicate 支持类型分档：arch22 无 int8/int64"
description: "Duplicate 的支持类型按架构分档（arch22/dav_c220 不支持 int8_t/int64_t，arch35/dav_3510 支持）；跨架构共用 kernel 时 int8 广播走 Cast 1 元素→half→Duplicate<half>、int64 走标量循环（读 1 次广播值）；dtype 分派按两架构支持集交集设计。"
tags: [task.implementation, topic.implementation.api_layer.basic_api, topic.architecture]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["a2", "950"]
---

# Duplicate 支持类型分档：arch22 无 int8/int64

## 现象

逐元素比较算子（y = max(x1, x2)）迭代二要同时支持 arch22（ascend910b）与 arch35（ascend950），且设计要求两架构 kernel 代码完全一致（仅 UB 预算常量不同）。广播路径的实现方案是"把广播侧的 1 个标量 Duplicate 成整段，再走向量 Max"——在为 6 dtype（bf16/fp16/fp32/int8/int32/int64）落实该方案前，直接查 CANN 安装内 Duplicate 的实现头确认各架构支持类型：

- arch35（`asc/impl/basic_api/dav_3510/kernel_operator_vec_duplicate_impl.h`）：支持 int8_t、half、bfloat16_t、int16_t、int32_t、float、int64_t 等——覆盖全部 6 dtype。
- arch22（`asc/impl/basic_api/dav_c220/kernel_operator_vec_duplicate_impl.h`）：仅 half、bfloat16_t、int16_t、uint16_t、int32_t、uint32_t、float——**不含 int8_t、不含 int64_t**。

若两架构共用同一份 kernel 直接对 int8/int64 实例化 `Duplicate<T>`，arch22 编译会因类型不被支持而失败。

## 适用场景

- 工程形态：AscendC kernel 广播/标量扩展路径开发，一份 kernel 源码多架构编译（arch22 + arch35）。
- API 路径：`AscendC::Duplicate`（Level 2 向量填充 API，`__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)`）；配套 `Cast`、`Max`、`LocalTensor::GetValue/SetValue`。
- 验证过的平台与 CANN 版本：arch22（ascend910b）与 arch35（ascend950）均编译产出 6 dtype × 4 实例（24 TilingKey）；NPU（Ascend950PR）ST 13/13 通过。CANN 9.1.0。
- 适用粒度：platform_wide——API 支持矩阵是平台事实，任何用到 Duplicate 标量填充的算子通用。
- 路线优先级：recommended_fastpath（跨架构一致代码的既定分派方案）。
- 明确不适用 / 未验证场景：单架构目标（arch35 上 int8/int64 可直接 Duplicate，无需绕行）；`Broadcast<T, dim, axis>` 向量广播 API（静态 dim/axis 模板参数，另一套机制）；int8 广播路径的上板 ST 验证（本会话广播 ST 用例为 fp16/fp32，int8 广播仅编译级验证，未上板）。

## 判别方法（规则）

查证方式（可复用）：

1. 定位实现头：`find $ASCEND_HOME_PATH/x86_64-linux/asc/impl/basic_api -name "kernel_operator_vec_duplicate_impl.h"`（按 dav_3510 / dav_c220 等架构子目录分档）。
2. 读各档头文件的重载/模板特化列表，得到该架构支持的 T 集合。
3. `CheckDuplicateSupportedType` 等检查符号在 include 层 grep 不到时，直接看 `asc/impl/basic_api/dav_*/` 下的实现（本例 include 层无命中、impl 层有）。

跨架构一致的 dtype 分派方案（落地版）：

- fp16 / fp32 / int32：`Duplicate<T>` 广播值 → `Max`（两架构都直接支持）。
- bf16：`Duplicate<bfloat16_t>` 广播值 → `Cast` 到 fp32 → `Max` → `Cast` 回 bf16（Duplicate 支持 bf16，Max 不直接支持故走 fp32 中转）。
- int8：**不** `Duplicate<int8_t>`（arch22 不支持）——把广播侧 1 个 int8 先 `Cast` 成 1 个 half，`Duplicate<half>` 扩到整段；另一侧 int8 整段 `Cast` 成 half；`Max(half)` 后 `Cast` 回 int8。两架构一致可用。
- int64：**不** `Duplicate<int64_t>`（arch22 不支持）——广播值 `GetValue(0)` 读一次，标量循环逐元素比较写回。两架构一致可用。

## 系统信息

- 硬件/软件：CANN 9.1.0（`$ASCEND_HOME_PATH` 指向 CANN 安装根）；目标 soc ascend910b（arch22/dav_c220）+ ascend950（arch35/dav_3510）；NPU 实机 Ascend950PR。
- 相关文件：`op_kernel/arch35/<op>.h` 与 `op_kernel/arch22/<op>.h`（ComputeBcastDup 的 dtype 分派）；CANN 头 `x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_vec_duplicate_impl.h`、`.../dav_c220/kernel_operator_vec_duplicate_impl.h`。
- 构建/验证命令：`bash build.sh --soc=ascend950 -j8` / `bash build.sh --soc=ascend910b -j8`；ST `bash run.sh --real`（NPU）。

## 完整代码

`ComputeBcastDup`（kernel 头，arch22/arch35 同一份；广播侧标量扩展的完整 dtype 分派实现，verbatim）：

```cpp
template <typename T, int BUFFER_MODE, int BROADCAST_MODE>
__aicore__ inline void BenchNoHarnessMaximum<T, BUFFER_MODE, BROADCAST_MODE>::ComputeBcastDup(
    int64_t currentNum, bool x1Bcast, bool x2Bcast)
{
    AscendC::LocalTensor<T> x1Local = inputQueueX1.template DeQue<T>();
    AscendC::LocalTensor<T> x2Local = inputQueueX2.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    if constexpr (IsDirectMax<T>::value) {
        // float16 / float32 / int32: Duplicate 广播值后 Max
        if (x1Bcast) {
            T scalarVal = x1Local.GetValue(0);
            AscendC::Duplicate<T>(x1Local, scalarVal, currentNum);
        }
        if (x2Bcast) {
            T scalarVal = x2Local.GetValue(0);
            AscendC::Duplicate<T>(x2Local, scalarVal, currentNum);
        }
        AscendC::Max(yLocal, x1Local, x2Local, currentNum);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        // bfloat16: Duplicate bf16 → Cast→fp32→Max→Cast
        auto x1Fp32 = TmpBuf1.Get<float>();
        auto x2Fp32 = TmpBuf2.Get<float>();
        auto yFp32 = TmpBuf3.Get<float>();
        if (x1Bcast) {
            T scalarVal = x1Local.GetValue(0);
            AscendC::Duplicate<T>(x1Local, scalarVal, currentNum);
        }
        if (x2Bcast) {
            T scalarVal = x2Local.GetValue(0);
            AscendC::Duplicate<T>(x2Local, scalarVal, currentNum);
        }
        AscendC::Cast(x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::Cast(x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::Max(yFp32, x1Fp32, x2Fp32, currentNum);
        AscendC::Cast(yLocal, yFp32, AscendC::RoundMode::CAST_TRUNC, currentNum);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        // int8: Cast 1 元素→half→Duplicate half, Cast 另一输入→half, Max, Cast→int8
        auto x1Half = TmpBuf1.Get<half>();
        auto x2Half = TmpBuf2.Get<half>();
        auto yHalf = TmpBuf3.Get<half>();
        if (x1Bcast && x2Bcast) {
            // 两边都广播：都 Cast 1 元素 + Duplicate
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, 1);
            half sv1 = x1Half.GetValue(0);
            AscendC::Duplicate<half>(x1Half, sv1, currentNum);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, 1);
            half sv2 = x2Half.GetValue(0);
            AscendC::Duplicate<half>(x2Half, sv2, currentNum);
        } else if (x1Bcast) {
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, 1);
            half sv1 = x1Half.GetValue(0);
            AscendC::Duplicate<half>(x1Half, sv1, currentNum);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, currentNum);
        } else {
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, currentNum);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, 1);
            half sv2 = x2Half.GetValue(0);
            AscendC::Duplicate<half>(x2Half, sv2, currentNum);
        }
        AscendC::Max(yHalf, x1Half, x2Half, currentNum);
        AscendC::Cast(yLocal, yHalf, AscendC::RoundMode::CAST_TRUNC, currentNum);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        // int64: 标量比较，广播值读取 1 次
        if (x1Bcast && x2Bcast) {
            T v0 = x1Local.GetValue(0);
            T v1 = x2Local.GetValue(0);
            T result = (v0 > v1) ? v0 : v1;
            for (int64_t i = 0; i < currentNum; i++) {
                yLocal.SetValue(i, result);
            }
        } else if (x1Bcast) {
            T v0 = x1Local.GetValue(0);
            for (int64_t i = 0; i < currentNum; i++) {
                T v1 = x2Local.GetValue(i);
                yLocal.SetValue(i, (v0 > v1) ? v0 : v1);
            }
        } else {
            T v1 = x2Local.GetValue(0);
            for (int64_t i = 0; i < currentNum; i++) {
                T v0 = x1Local.GetValue(i);
                yLocal.SetValue(i, (v0 > v1) ? v0 : v1);
            }
        }
    }

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX1.FreeTensor(x1Local);
    inputQueueX2.FreeTensor(x2Local);
}
```

## 设计决策

- 为什么 int8 绕行 half 而不是按架构 `if constexpr` 分派（arch35 直接 Duplicate<int8_t>）：设计约束是两架构 kernel 代码逐字节一致，架构分派会引入 `__NPU_ARCH__` 条件编译；half 中转在两架构都合法，代价是 int8 广播段多两次 Cast（compute 非瓶颈时可接受）。
- 为什么 int64 用标量循环而不是 Cast 中转：int64 无法安全压到 half（值域溢出），fp32 也只有 24bit 尾数；int64 本就走标量路径，广播值读一次的标量循环语义最稳。
- 为什么 bf16 先 Duplicate 再 Cast 而不是 Cast 后 Duplicate fp32：Duplicate 支持 bfloat16_t（两架构均是），先扩再转只需扩 1 个元素再整段 Cast，buffer 占用与后者相当且少一次判断。
- 被放弃的方案：`Broadcast<T, dim, axis>` 向量 API（需静态 dim/axis 模板参数 + 对齐约束，对动态 shape 广播不适用）。
- 当前方案路线档位：recommended_fastpath（在"代码跨架构一致"约束下）。

## 预防

- 用 Duplicate 做标量填充前，先查目标架构的 `dav_*/kernel_operator_vec_duplicate_impl.h` 支持列表；int8/int64 在 arch22 明确缺席。
- 多架构共码算子的 dtype 分派表按"两架构支持集的交集"设计，交集外的 dtype 走 Cast 中转或标量路径。

## 验证信号

以下信号来自该算子迭代二广播路径开发会话：先查证两架构 Duplicate 支持矩阵，再按交集分派实现，双架构编译 + 上板回归。

- API 事实：`asc/impl/basic_api/dav_c220/kernel_operator_vec_duplicate_impl.h` 支持类型为 half/bfloat16_t/int16_t/uint16_t/int32_t/uint32_t/float（无 int8_t/int64_t）；`dav_3510/` 版本含 int8_t/int64_t（来源平台：a2, 950；证据等级：tool_observed）。
- 实现落地：按上述分派写 kernel 后，ascend950 与 ascend910b 均编译成功，各产出 6 个 kernel 二进制；安装后 tiling config 为 6 dtype × 4 tiling keys = 24 实例（来源平台：a2, 950；证据等级：tool_observed）。
- 上板回归：Ascend950PR ST 13/13 通过（含 fp16/fp32 广播用例）；UT 55/55（来源平台：950；证据等级：tool_observed）。
未验证：int8/int64 广播路径的上板 ST 用例未在本轮执行（仅编译级验证）。

## 失效边界

- 支持矩阵为 CANN 9.1.0 实测；后续版本若 arch22 补齐 int8/int64 的 Duplicate，绕行方案只剩兼容性意义（可简化但不必改）。
- int8 经 half 中转的值域：half 尾数 10bit，|int8| ≤ 127 可精确表示，无精度损失；该中转方案**不可**直接推广到 int16/int32（half 表示不下）。
- int64 标量循环的性能代价未量化（广播段逐元素 GetValue/SetValue）；大数据量 int64 广播若有性能要求须另行评估。
- `CheckDuplicateSupportedType` 类运行时检查接口在 include 层未检索到（grep 无命中），本卡依据 impl 头静态事实 + 编译验证。

**未决项**：

- int8/int64 广播路径的上板功能验证（含大 shape）待补；如失败优先检查 half 中转的 Cast 对齐（currentNum 非 32 倍数时的尾块处理）。
- arch22 上 int64 广播的标量循环性能（相对 Duplicate 路线）未测。
## 坏实践

- 假设 Duplicate 支持所有基础 dtype，在多架构共码的 kernel 里对 int8/int64 直接实例化 `Duplicate<T>`（arch22 编译失败）。
- 为单一架构的便利引入 `__NPU_ARCH__` 条件编译分支，破坏两架构代码一致性约束。

## 误诊

- 无。
