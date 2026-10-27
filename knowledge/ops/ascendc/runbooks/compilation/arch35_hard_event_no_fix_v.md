---
type: "Runbook"
title: arch35 HardEvent 无 FIX_V，显式同步编译失败
description: "AscendC HardEvent 的 FIX_V/V_FIX 仅 __NPU_ARCH__ 5102/3003 定义；arch35(DAV_3510) kernel 写 SetFlag<HardEvent::FIX_V> 报 no member named 'FIX_V'，需移除显式同步改用隐式/TQue。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# arch35 HardEvent 无 FIX_V，显式同步编译失败

## 现象

自定义 Conv2d 基准算子（ascend950，arch35/DAV_3510）的 kernel 实现采用 Cube 通路：Load3D 进图 → Mmad 矩阵乘 → Fixpipe（L0C→UB）→ Vector 侧 Add 加 bias → DataCopyPad 出 GM。在 Fixpipe 之后、Add 之前，为做 Fixpipe→Vector 的跨核同步，代码里写了：

```cpp
    event_t fixEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_V));
    SetFlag<HardEvent::FIX_V>(fixEvent);
    WaitFlag<HardEvent::FIX_V>(fixEvent);
```

kernel 入口名修复后重新编译，kernel 头文件（op_kernel/arch35/bench_no_harness_conv_2d.h）第 339/340 行报：

```text
error: no member named 'FIX_V' in 'AscendC::HardEvent'
    event_t fixEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_V));
    SetFlag<HardEvent::FIX_V>(fixEvent);
```

## 适用场景

- 工程形态：算子仓自定义算子 kernel 编写（arch35 目录，Cube+Vector 混合通路，Fixpipe 输出接 Vector 计算）。
- API 路径：`AscendC::HardEvent` 枚举（ascendc/include/basic_api/impl/kernel_event.h）、`SetFlag<HardEvent::...>/WaitFlag<HardEvent::...>`、`GetTPipePtr()->FetchEventID(...)`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950，arch35/DAV_3510，`__NPU_ARCH__` 3510）+ CANN 9.1.0，编译期验证。
- 适用粒度：operator_family（arch35 上所有用 Fixpipe→Vector 通路的 kernel 都适用；结论按架构代际而非单算子成立）。
- 路线优先级：recommended_fastpath（编译层）。
- 明确不适用/未验证场景：`__NPU_ARCH__` 5102/3003 等有 FIX_V 的架构不受此卡约束；移除显式同步后的运行时正确性（数据竞争）未在本阶段验证。

## 根因

CANN 9.1.0 `ascendc/include/basic_api/impl/kernel_event.h` 中 `HardEvent` 枚举的 `FIX_V`、`V_FIX` 成员被条件编译包裹：

```cpp
enum class HardEvent : uint8_t {
    // src_dst
    MTE2_MTE1,
    MTE1_MTE2,
    MTE1_M,
    M_MTE1,
    MTE2_V,
    V_MTE2,
    MTE3_V,
    V_MTE3,
    M_V,
    V_M,
    V_V,
    MTE3_MTE1,
    MTE1_MTE3,
    MTE1_V,
    MTE2_M,
    M_MTE2,
    V_MTE1,
    M_FIX,
    FIX_M,
    MTE3_MTE2,
    MTE2_MTE3,
    S_V,
    V_S,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MTE2_FIX,
    FIX_MTE2,
    FIX_S,
    M_S,
    FIX_MTE3,
    MTE1_FIX,
    FIX_MTE1,
    FIX_FIX,
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003))
    FIX_V,
    V_FIX,
#endif
    MAX,
};
```

即 FIX_V/V_FIX 仅在 `__NPU_ARCH__ == 5102 || 3003` 下存在；arch35（DAV_3510，3510）编译单元里根本没有这个枚举成员，`SetFlag<HardEvent::FIX_V>` 直接编译失败。arch35 上可用的 FIX 相关事件只有 M_FIX、FIX_M、MTE2_FIX、FIX_MTE2、FIX_MTE3、FIX_MTE1、FIX_FIX、FIX_S 等，没有 FIX→V 方向的事件。

## 解决配方

骨架阶段移除显式 FIX_V 同步，依赖硬件隐式同步（Fixpipe 输出落 UB 后继续执行），并在后续迭代改用 TQue（EnQue/DeQue）实现 Cube→Vector 的自动同步。修复前后（kernel 头文件中 Fixpipe 调用之后的段落）：

修复前：

```cpp
    Fixpipe<D_T_X, float, CFG_ROW_MAJOR_TO_UB>(fixpipeUB, co1Tensor, fixpipeParams);

    // FIX → V 同步
    event_t fixEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_V));
    SetFlag<HardEvent::FIX_V>(fixEvent);
    WaitFlag<HardEvent::FIX_V>(fixEvent);

    // ===== 5. Add: UB(fixpipe) + UB(bias) → UB(result) =====
```

修复后：

```cpp
    Fixpipe<D_T_X, float, CFG_ROW_MAJOR_TO_UB>(fixpipeUB, co1Tensor, fixpipeParams);

    // arch35: FIX_V 不可用，Fixpipe→UB→Add 依赖硬件隐式同步
    // 后续迭代将通过 TQue(EnQue/DeQue) 实现 Cube→Vector 自动同步

    // ===== 5. Add: UB(fixpipe) + UB(bias) → UB(result) =====
```

## 设计决策

- 为什么不换成别的显式事件：arch35 的 HardEvent 没有 FIX→V 方向成员，MTE3_V/V_MTE3 语义不匹配（MTE3 是 UB→GM 搬运管道），强行借用语义错误。
- 为什么选"移除显式同步"而非"上 TQue"：当前是骨架迭代，目标是先打通编译与整体链路；TQue（EnQue/DeQue 队列语义）改动面大，留给下一迭代做 Cube→Vector 自动同步。该取舍的代价是同步正确性未验证（见失效边界）。

## 预防

- 写跨管道事件同步前，先看目标架构的 `kernel_event.h` 枚举哪些成员被 `#if __NPU_ARCH__` 包裹，不要从其他架构的样例代码照抄事件名。
- arch35 上 Cube（Fixpipe）与 Vector 的同步优先考虑 TQue 队列机制，而不是手写 SetFlag/WaitFlag。

## 验证信号

来源说明：本次修复来自同一算子 kernel 编译排障：移除显式 FIX_V 同步后重新全量编译。

- 后续全量重建（make_clean 后 build.sh --soc=ascend950）并用 `grep -E "error:|Error|INFERCHANNEL|FATBIN|not implement|fatal"` 过滤的输出中不再出现 `no member named 'FIX_V'`，剩余错误为独立的 tilingkey 包装函数与 DataCopyPad 重载问题，证明本处修复生效（来源平台：950；证据等级：tool_observed）。
- `kernel_event.h` 枚举源码确认 FIX_V/V_FIX 仅在 `__NPU_ARCH__ == 5102 || 3003` 分支下定义（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "移除显式同步"只保证编译通过，不构成同步正确性证明：Fixpipe→Add 的数据依赖在本阶段依赖硬件隐式同步，未经上板验证；若存在竞态，需按设计改用 TQue(EnQue/DeQue) 方案。
- 结论仅对 arch35（DAV_3510/3510，Ascend950PR）成立；5102/3003 架构上 FIX_V 存在，可直接使用（未验证，仅由头文件条件编译推断）。

**未决项**：

- 优先项：上板验证移除显式同步后 Fixpipe→Add 是否存在数据竞态；若存在，按迭代计划改用 TQue(EnQue/DeQue) 实现 Cube→Vector 自动同步（未尝试）。
- arch35 上 Cube→Vector 同步的推荐机制（隐式同步是否够用、TQue 是否官方推荐路径）未在本阶段确认。
## 误诊

- 初见 `no member named 'FIX_V'` 时可能以为头文件没包含全或枚举名拼错；实际枚举确实存在但被架构条件编译排除，换拼写/补 include 都无效。
