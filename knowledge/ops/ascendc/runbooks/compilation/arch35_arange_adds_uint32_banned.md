---
type: "Runbook"
title: arch35 Arange/Adds 拒 uint32，int32 生成重解释
description: arch35（dav_3510）的 Arange/Adds 设备实现按 SupportType 白名单实例化，uint32_t/unsigned int 不在白名单，Arange<uint32_t>/Adds<uint32_t> 触发 static_assert 编译失败并级联 vci/Get 报错；用 Arange<int32_t> 生成偏移表、调用处 Get<uint32_t>() 重解释同一 UB buffer；Gather 亦不支持 int64，用两步 Gather<int32_t> 取低/高 32 位再组合。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# arch35 Arange/Adds 拒 uint32，int32 生成重解释

## 现象

为一个 TopK 归约类自定义算子（目标芯片 Ascend950PR / arch35，CANN cann-9.1.0）扩展 int32/int64 数据类型支持时，kernel 里实现了"非末维数据提取"：先用 `DataCopyPad` 以 32 字节块把非连续 GM 数据搬入 UB，再用 `Gather` 按"每个 32B 块首元素"的等差偏移表（`[0, 32, 64, ...]`）提取成连续数据。因为 `Gather` 的偏移表形参类型是 `LocalTensor<uint32_t>`，最初直接用 `Arange<uint32_t>` 生成偏移表，并在 int64 分支里用 `Adds<uint32_t>` 给偏移表整体加 4（取高 32 位）。执行 `bash build.sh --soc=ascend950 -j8` 编译 kernel（设备侧编译）时，报出一串 static assertion 编译错误：

```text
<CANN>/x86_64-linux/asc/include/adv_api/index/../../../impl/adv_api/detail/index/arithprogression/arithprogression_3510_impl.h:73:5: error: static assertion failed due to requirement 'SupportType<unsigned int, short, int, half, float, long>()': current data type is not supported on current device!
<CANN>/x86_64-linux/asc/impl/basic_api/reg_compute/../../basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_arange_impl.h:47:5: error: static assertion failed due to requirement 'SupportType<unsigned int, signed char, short, int, half, float, long>()': current Arange data type is not supported on current device!
<CANN>/x86_64-linux/asc/impl/basic_api/reg_compute/../../basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_arange_impl.h:53:9: error: no matching function for call to 'vci'
operators/bench_no_harness_top_k/op_kernel/arch35/bench_no_harness_top_k.h:489:52: error: no member named 'Get' in 'AscendC::LocalTensor<long>'
<CANN>/x86_64-linux/asc/impl/basic_api/reg_compute/../../basic_api/reg_compute/dav_3510/kernel_operator_vec_binary_scalar_impl.h:235:5: error: static assertion failed due to requirement 'SupportType<unsigned int, unsigned char, signed char, half, __bf16, float, short, int, long, unsigned long, AscendC::Complex<half>, AscendC::Complex<float>>()': Adds not support current datatype!
```

同一批错误在后续重新编译中再次出现，并伴随 `no matching function for call to 'bench_no_harness_top_k_0_tilingkey'` 与 `ld.lld: error: cannot open ... .o: No such file or directory`（对象文件因编译失败未生成）。

## 适用场景

- 工程形态：自定义算子仓开发（kernel 直调/算子仓工程），kernel 设备侧代码编译。
- API 路径：AscendC `Arange`（reg_compute / arithprogression）、`Adds`（vec_binary_scalar）、`Gather`（vec）偏移表生成链。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510，SOC ascend950）+ cann-9.1.0（bisheng clang 15.0.5 设备编译）（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（arch35 一代的 reg_compute 类型白名单行为）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证：非 arch35 架构（dav_c220 / dav_l300 等）的 Arange 白名单未在本次开发会话验证；不同 CANN 版本的白名单可能变化。

## 根因

arch35（dav_3510）的 `Arange`/`Adds` 设备实现只对 `{signed char, short, int, half, float, long}`（Arange）与 `{unsigned char, signed char, half, __bf16, float, short, int, long, unsigned long, Complex<...>}`（Adds）等类型做 `SupportType` 白名单实例化，`uint32_t` / `unsigned int` 不在白名单内，模板实例化时 static_assert 直接失败。

## 解决配方

修复前（`InitGatherOffsets`，偏移表直接按 Gather 形参类型生成；int64 分支用 `Adds<uint32_t>`）：

```cpp
template <typename T, int INNER_MODE>
__aicore__ inline void BenchNoHarnessTopK<T, INNER_MODE>::InitGatherOffsets(int64_t count)
{
    LocalTensor<uint32_t> gatherOffset = gatherOffsetBuf.Get<uint32_t>();
    // Arange 生成等差数列：start=0, step=32, count=count
    Arange<uint32_t>(gatherOffset, static_cast<uint32_t>(0), static_cast<uint32_t>(32),
                     static_cast<uint32_t>(count));
    PipeBarrier<PIPE_V>();
}
```

修复后（用 `Arange<int32_t>` 生成，调用处再重解释为 `uint32_t`）：

```cpp
// 创建 Gather 偏移表：[0, 32, 64, ..., (count-1)*32]（字节偏移，每个元素在 32B 块的起始位置）
template <typename T, int INNER_MODE>
__aicore__ inline void BenchNoHarnessTopK<T, INNER_MODE>::InitGatherOffsets(int64_t count)
{
    // Arange 不支持 uint32_t（arch35 仅支持 short/int/half/float/long），用 int32_t 生成
    LocalTensor<int32_t> gatherOffsetI32 = gatherOffsetBuf.Get<int32_t>();
    Arange<int32_t>(gatherOffsetI32, static_cast<int32_t>(0), static_cast<int32_t>(32),
                    static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
}
```

配套的 `GatherFromPadded`（修复后终版；`Adds` 同步改为 `int32_t`，Gather 调用处用 `Get<uint32_t>()` 重解释同一 UB buffer）：

```cpp
// 从 32B-padded UB 数据中 Gather 提取元素到连续 UB
// 对于 int64_t：Gather 不支持 int64，用两步 Gather<int32_t> 提取低/高 32 位再组合
template <typename T, int INNER_MODE>
__aicore__ inline void BenchNoHarnessTopK<T, INNER_MODE>::GatherFromPadded(
    LocalTensor<T>& dst, LocalTensor<T>& src, int64_t count)
{
    // Gather API 需要 LocalTensor<uint32_t> 偏移表，用 Get<uint32_t>() 重解释 int32_t buffer
    LocalTensor<uint32_t> gatherOffsetU32 = gatherOffsetBuf.template Get<uint32_t>();
    if constexpr (std::is_same_v<T, int64_t>) {
        // int64 不被 Gather 直接支持，用 Gather<int32_t> 分别提取低/高 32 位
        LocalTensor<int32_t> srcI32 = src.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> dstLow = gatherLowBuf.template Get<int32_t>();
        LocalTensor<int32_t> dstHigh = gatherHighBuf.template Get<int32_t>();
        // 低 32 位：偏移 [0, 32, 64, ...]
        Gather<int32_t>(dstLow, srcI32, gatherOffsetU32, 0, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
        // 高 32 位：偏移 [4, 36, 68, ...] = 低偏移 + 4
        LocalTensor<int32_t> gatherOffsetI32 = gatherOffsetBuf.template Get<int32_t>();
        Adds<int32_t>(gatherOffsetI32, gatherOffsetI32, static_cast<int32_t>(4),
                      static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Gather<int32_t>(dstHigh, srcI32, gatherOffsetU32, 0, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
        // 组合低/高 32 位为 int64（SetValue，S pipe，但仅用于组合不涉及 GM 读写）
        for (int64_t i = 0; i < count; i++) {
            int32_t low = dstLow.GetValue(static_cast<uint32_t>(i));
            int32_t high = dstHigh.GetValue(static_cast<uint32_t>(i));
            int64_t val = (static_cast<int64_t>(static_cast<uint32_t>(high)) << 32) |
                          static_cast<uint32_t>(low);
            dst.SetValue(static_cast<uint32_t>(i), val);
        }
        PipeBarrier<PIPE_ALL>();
    } else {
        // 其他类型：直接 Gather<T>
        Gather<T>(dst, src, gatherOffsetU32, 0, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }
}
```

## 设计决策

- 为什么"生成用 int32、使用时重解释"：`Gather` 的偏移表形参固定为 `LocalTensor<uint32_t>`（编译约束），而 `Arange`/`Adds` 又不接收 uint32_t——两个 API 约束的交集只能在同一块 TBuf 上用 `Get<int32_t>()` 写、`Get<uint32_t>()` 读（或 `ReinterpretCast`）。并行会话里另一实现用 `gatherOffsetI32.template ReinterpretCast<uint32_t>()` 同样编译通过，两条重解释路线均可行。
- 备选方案：手写 for 循环 `SetValue` 逐个写偏移表（S pipe 标量写，慢且引入跨 pipe 同步点），被放弃。
- int64 分支额外决策：`Gather` 的 arch35 实现断言不含 int64 → 用两步 `Gather<int32_t>` 分别取低/高 32 位（高 32 位用 `Adds<int32_t>` 把偏移表整体 +4 字节），再移位组合；组合阶段只用 `SetValue` 操作 UB、不涉及 GM。
- 当前方案属 recommended_fastpath：不引入额外同步原语，纯 V pipe。

## 预防

- 因为某个 API 的形参类型是 `LocalTensor<uint32_t>` 就顺手把上游生成 API 也全用 `uint32_t` 实例化——两个 API 的类型白名单并不一致，需分别确认。
- `SupportType` 白名单未从官方文档核对时，直接阅读 `kernel_reg_compute_vec_arange_impl.h` 与 `kernel_operator_vec_binary_scalar_impl.h` 的 SupportType 列表确认。

## 验证信号

来源说明：该 TopK 算子在 Ascend950PR 上的一次迭代开发会话（含本修复前后的编译与测试记录）。

- 修复前 `bash build.sh --soc=ascend950 -j8` 报上述 `SupportType ... not supported on current device` / `Adds not support current datatype!` 原文（来源平台：950；证据等级：tool_observed）
- 修复后 `bash build.sh --soc=ascend950 -j8`（`--make_clean` 后全量重编）输出 `Build completed successfully!` 并生成自定义算子包（来源平台：950；证据等级：tool_observed）
- kernel 二进制目录出现 7 个 `.o`/`.json` 对（每个 json 对应一组 TilingKey 模板实例，共 14 组合），`aic-ascend950-ops-info.json` 注册 7 种 dtype（来源平台：950；证据等级：tool_observed）
- 含该代码路径的算子包在 ST Real 模式（NPU 实测）70/70 非跳过用例通过（来源平台：950；证据等级：tool_observed）
- Mock 模式（CPU Golden）129/129 通过，无回归（来源平台：950（Mock 为 CPU 侧）；证据等级：tool_observed）

## 失效边界

- 该修复只解决"编译通过"；`GatherFromPadded` 所属的非末维提取路径在 NPU 上功能性失败（多轮 mismatch≈100%），整体方案被回退——功能失败与偏移单位歧义另见非末维负结果记录，勿把本卡当作非末维方案可行的证据。
- `SupportType` 白名单来自 cann-9.1.0 的 dav_3510 实现头，未在其他 CANN 版本/架构上验证（未验证，推断不可迁移前请先读对应架构实现头）。
- `unsigned int` 对 `Adds` 同样不在白名单（错误原文列表里有 unsigned long 而无 unsigned int）。

**未决项**：

- Arange/Adds 在 dav_3510 的完整类型白名单未从官方文档核对（仅由错误信息反推）。
- `Gather` 偏移值的单位（字节 vs 元素）在 arch35 上的确切语义未定论（本次开发会话两种取值均未产出正确功能结果），复用 `GatherFromPadded` 前需先闭环该问题。
## 坏实践

- 报错后先怀疑 TilingKey 注册/宏定义问题并去改 `ASCENDC_TPL_DATATYPE_DECL`——`no matching function for call to '<op>_0_tilingkey'`、`no member named 'Get' in 'AscendC::LocalTensor<long>'`、`no matching function for call to 'vci'` 等错误是 `Arange<uint32_t>` 实例化失败的级联表现（佐证：仅修复 Arange/Adds 类型后这些错误随整体编译成功消失）。

## 误诊

- 并行会话一度把级联错误当 TilingKey 定义问题去删 dtype 声明，方向错误；级联噪声应回溯到首条 static_assert。
