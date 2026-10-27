---
type: "Runbook"
title: "Gather 偏移表形参固定 LocalTensor<uint32_t>"
description: "AscendC Gather Level-2 API 签名固定第 3 形参为 const LocalTensor<uint32_t>&、count 为 uint32_t；传 LocalTensor<int32_t> 偏移表或 int32_t count 报 no matching function for call to 'Gather'；用 TBuf::Get<uint32_t>() 或 ReinterpretCast<uint32_t>() 重解释既有 buffer，arch35 元素类型断言含 int32 不含 int64。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:51:56Z'
updated_at: '2026-09-17T00:51:56Z'
platforms: ["950"]
---

# Gather 偏移表形参固定 LocalTensor<uint32_t>

## 现象

自定义 TopK 算子（Ascend950PR / arch35，CANN 9.1.0）的非末维提取实现里，先用 `Arange<int32_t>` 生成 Gather 偏移表（`Arange<uint32_t>` 编译不过），随后把 `LocalTensor<int32_t>` 偏移表直接传给 `Gather<int32_t>` / `Gather<T>`，并把 `count` 写成 `static_cast<int32_t>(count)`。执行 `bash build.sh --soc=ascend950 -j8` 时设备编译报（原文）：

```text
<工程路径>/op_kernel/arch35/<op>.h:497:9: error: no matching function for call to 'Gather'
<工程路径>/op_kernel/arch35/<op>.h:503:9: error: no matching function for call to 'Gather'
gmake[2]: *** [CMakeFiles/Makefile2:471: op_kernel/CMakeFiles/<Op>_ascend950.dir/all] Error 2
```

## 适用场景

- 工程形态：自定义算子仓开发，kernel 设备侧代码编译（使用 AscendC Gather Level-2 API 做按偏移表提取）。
- API 路径：AscendC `Gather`（Level-2，`kernel_operator_vec_gather_intf.h`）及其 arch35 实现（`dav_3510/kernel_operator_vec_gather_impl.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510）+ CANN 9.1.0（与 frontmatter 一致）。
- 适用粒度：operator_family——Gather 接口签名跨架构一致，arch35 支持类型列表为架构相关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证：`GatherMask` 等其它重载的形参约束未逐一核对；非 arch35 架构的元素类型白名单未验证（未验证，推断可迁移前先读对应架构实现）；本卡只覆盖"偏移表/count 形参类型匹配"这一编译层约束，不含 Gather 偏移单位语义（见失效边界）。

## 根因

`Gather` Level-2 API 的签名固定为：

```cpp
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseOffset, const uint32_t count);
```

偏移表形参类型是 `LocalTensor<uint32_t>`、`count` 是 `uint32_t`；传 `LocalTensor<int32_t>` 或 `int32_t` count 时模板推导失败，报 `no matching function for call to 'Gather'`。arch35 实现内的元素类型断言（`SupportType<PrimType, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>`）含 `int32_t` 但不含 `int64_t`——int64 数据需要拆成两个 int32 视图分别 Gather。

## 判别方法

用 grep 定位签名并读 arch35 实现确认类型断言（CANN 9.1.0 实读）：

```bash
grep -rn "inline void Gather" $ASCEND_HOME_PATH/x86_64-linux/asc/include/ | head
```

```cpp
// kernel_operator_vec_gather_intf_impl.h（Level-2 实现，arch35 分支）
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseOffset, const uint32_t count)
{
    using PrimType = PrimT<T>;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) ||                       \
      (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
    ASCENDC_ASSERT((SupportType<PrimType, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, uint32_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Gather, current api support dtype combination is src and "
        "dst both: uint8 / int8 / half / bfloat16_t / uint16 / int16 / float / uint32 / int32");});
    GatherImpl((__ubuf__ PrimType *)dst.GetPhyAddr(), (__ubuf__ PrimType *)src.GetPhyAddr(),
        (__ubuf__ uint32_t *)srcOffset.GetPhyAddr(), srcBaseOffset, count);
#endif
}
```

## 解决配方

修复前（偏移表与 count 都是 int32，直接传参）→ 修复后（同一 UB buffer 用 `Get<uint32_t>()` 重解释出偏移表视图，count 统一 `uint32_t`；verbatim，含 int64 拆分路径）：

```cpp
template <typename T, int INNER_MODE>
__aicore__ inline void <Op><T, INNER_MODE>::GatherFromPadded(
    LocalTensor<T>& dst, LocalTensor<T>& src, int64_t count)
{
    // Gather API 需要 LocalTensor<uint32_t> 偏移表，用 Get<uint32_t>() 重解释 int32_t buffer
    LocalTensor<uint32_t> gatherOffsetU32 = gatherOffsetBuf.template Get<uint32_t>();
    if constexpr (std::is_same_v<T, int64_t>) {
        // int64 不被 Gather 直接支持（arch35 SupportType 不含 int64），
        // 用 Gather<int32_t> 分别提取低/高 32 位
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
        // 组合低/高 32 位为 int64（SetValue，S pipe，仅用于组合不涉及 GM 读写）
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

重解释的等价写法（并行会话实测同样编译通过）：`LocalTensor<uint32_t> gatherOffsetU32 = gatherOffsetI32.template ReinterpretCast<uint32_t>();`。注意 `Adds` 仍需 int32 视图（按原生成类型做算术），Gather 用 uint32 视图。

## 设计决策

- 重解释两条路线（`TBuf::Get<uint32_t>()` vs `LocalTensor<int32_t>::ReinterpretCast<uint32_t>()`）均编译通过；选 `Get<uint32_t>()` 是不引入中间局部变量、直接从 TBuf 取目标类型视图。
- int64 拆分决策来自 arch35 类型断言（实现层 SupportType 列表不含 int64），不是文档结论——若上游版本放开 int64 支持可直接换回 `Gather<int64_t>`。

## 预防

- 传参前看签名而不是调用示例：偏移表生成侧的 tensor 类型（本例 Arange 只能产 int32）不决定 Gather 形参类型。
- `count` 顺手写 `static_cast<int32_t>` 是同类错误——该重载的 count 是 `uint32_t`，混用同样落入 no matching function。
- 报错行号可能因并行修改漂移，先读对应片段确认真实调用点再改，不要按报错行号盲改。

## 验证信号

来源说明：以下信号来自该 TopK 算子开发会话中本修复前后的编译与测试记录（Ascend950PR，CANN 9.1.0）。

- 修复前：设备编译报 `no matching function for call to 'Gather'`（.h:497/503 两处）原文（来源平台：950；证据等级：tool_observed）。
- 修复后：`bash build.sh --soc=ascend950 -j8` 编译成功、生成自定义算子包并安装（`SUCCESS`）（来源平台：950；证据等级：tool_observed）。
- 含该代码的算子包 ST Real 模式（NPU）70/70 非跳过用例通过（来源平台：950；证据等级：tool_observed）。
- 签名事实锚点：`kernel_operator_vec_gather_intf.h`/`_impl.h` 的 `Gather(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<uint32_t>&, const uint32_t, const uint32_t)` 与 arch35 `SupportType` 断言列表（含 int32、不含 int64）为 CANN 9.1.0 头文件直读（来源平台：CANN 9.1.0 安装树；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"偏移表/count 形参类型匹配"这一编译层约束；`srcOffset` 偏移值的单位（字节偏移 vs 元素偏移）在 arch35 硬件实现层未确定（本会话两种取值的功能结果都不正确），复用前需先用最小用例闭环——不要把本卡当作 Gather 功能用法已验证的证明。
- arch35 元素类型白名单来自 CANN 9.1.0 实现头，其他版本/架构未验证（未验证，推断可迁移前先读对应架构实现）。
- `srcBaseOffset` 在调试构建中要求能被 `sizeof(PrimType)` 整除（实现内断言），传非对齐基址偏移的风险未覆盖。

**未决项**：

- `srcOffset` 中偏移值的单位（字节偏移 vs 元素偏移）未在 arch35 实现层确定，复用前需最小用例闭环（优先项）。
## 坏实践

- 只看调用示例不看签名，把生成侧 tensor 类型直接传给 API。
- `count` 顺手写 `static_cast<int32_t>`。

## 误诊

- 报错行号定位偏差：错误指向的源行因并行修改发生过行号漂移，先读对应片段确认真实调用点再改。
