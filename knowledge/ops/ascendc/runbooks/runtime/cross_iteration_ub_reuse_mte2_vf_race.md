---
type: "Runbook"
title: "循环复用 UB buffer 跨迭代 MTE2/V 竞态"
description: "L 循环复用同一 UB buffer 时首迭代正确、后续迭代中间量变垃圾/inf 并间歇 507035；PRINTF 逐迭代探针判别跨迭代数据冒险，已试的三条同步路线（预置 SetFlag、条件 WaitFlag、PipeBarrier 全替换）均未闭合。"
tags: [task.debugging, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# 循环复用 UB buffer 跨迭代 MTE2/V 竞态

## 现象

kernel 以 `for (int64_t l = 0; l < L_; l++)` 逐行处理：每轮 CopyIn（MTE2，DataCopyPad GM→UB）把新一行写入复用的 UB buffer B0，Vector 侧 `asc_vf_call`（`__simd_vf__` 函数）从 B0 读入做 Cast/归约/归一化。运行现象：`aclrtSynchronizeStream` 间歇返回 ACL err 507035（向量核异常），几乎全部 ST 用例 bitwise mismatch。

在循环内插入 kernel `PRINTF` 探针逐迭代打印中间标量（V→S 事件同步后 `GetValue` 读归约结果）后，拿到第一手原始观测：

```text
[AIV Block 0/2] DBG: before CastBf16 b=0 hc=0 l=0
[AIV Block 0/2] DBG: statK=0.028437 invRmsK=16.749222
[AIV Block 0/2] DBG: before CastBf16 b=0 hc=0 l=1
[AIV Block 0/2] DBG: statK=239069803195275788943360000.000000 invRmsK=0.000000
[AIV Block 0/2] DBG: before CastBf16 b=0 hc=0 l=2
[AIV Block 0/2] DBG: statK=inf invRmsK=0.000000
[AIV Block 0/2] DBG: before CastBf16 b=0 hc=0 l=3
[AIV Block 0/2] DBG: statK=0.002375 invRmsK=57.082325
```

判别特征：**首迭代中间量正确（0.028437），后续迭代系统性变成垃圾值（2.39e23）/ inf 并交替**——同一份代码、同一份输入分布，仅迭代序号不同结果不同。

## 适用场景

- 工程形态：算子仓 kernel 开发，TBuf 直管 UB buffer + `asc_vf_call` VF 主链（无 TQue 队列化）的行/块循环 kernel。
- API 路径：`AscendC::SetFlag/WaitFlag`（`HardEvent::MTE3_MTE2` / `MTE2_V` / `V_S` / `S_V`）、`GetTPipePtr()->FetchEventID`、`AscendC::PipeBarrier<PIPE_ALL>`（注意 `PIPE_ALL` 枚举在全局命名空间，不在 `AscendC` 内）、`asc_vf_call`。
- 验证过的平台与 CANN 版本：Ascend 950PR + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——一切"循环 + 复用同一 UB buffer"的 kernel。
- 路线优先级：`safety_fallback`。fallback 声明：本卡提供的是判别方法与已证失败的同步路线；在找到可靠原语前，用双 buffer 交替（或 TQue 队列化让 EnQue/DeQue 承担同步）以 UB 空间换正确性——该替代路线在本轨迹未实现验证，采用前需评估 UB 容量代价。
- 明确不适用 / 未验证场景：迭代间不复用 buffer 的 kernel；`PipeBarrier` 在非 3510 后端的行为未验证。

## 判别方法

`diagnostic_rule`，四要素：

1. **触发**：循环复用同一 UB buffer + 间歇 507035 + 大面积 bitwise mismatch。
2. **探针（可执行）**：循环体内、V→S 同步点之后 `AscendC::PRINTF` 打印每迭代的归约中间标量（如上输出）；对比 l=0 与 l≥1。
3. **原始观测**：l=0 正确、l=1 垃圾、l=2 inf（逐字输出见现象节）。
4. **解释**：计算逻辑本身正确；这是跨迭代数据冒险——上一迭代 V 流水线尚未消费完 B0 时，下一迭代的 CopyIn（MTE2）已开始覆写 B0。迭代内虽有 MTE2_V 配对同步，但缺少"上一迭代 V 读完 → 下一迭代 MTE2 才能写"的跨迭代反向依赖。

拓扑区分（与库内相邻卡划界）：

- 消费者是 **MTE3** 的跨调用槽复用，双向 HardEvent 有可行配方，见 [MTE2 跨流水未建依赖卡](../precision/direct_mte_slot_reuse_needs_bidirectional_hard_events.md)——选用条件：回写走 MTE3 且能成对建事件；本卡消费者是 VF，实测 `MTE3_MTE2` 事件对 V→MTE2 覆盖不足。
- **迭代内** MTE2→V 的 RAW（同迭代先搬后算）用 TQue 队列交接即可闭合，见 [TBuf 跨 MTE2 与 Vector 须队列交接卡](../precision/tbuf_mte2_to_vector_requires_queue_handoff.md)——本卡是跨迭代 WAR（下一轮写覆盖上一轮读），方向相反，迭代内配对同步救不了。
- MTE3 消费者的跨迭代 WAR 在 TQue 路径下用 fence 闭合的先例，见 [TQue MTE2→MTE3 缺 MTE3_MTE2 fence 卡](../compilation/tque_mte2_mte3_mte3_mte2_fence.md)——本卡证明同一招式换成 V 消费者后连事件对都不足，不能照搬。

## 解决配方

本轨迹内尝试过的三条同步路线及实测结果（均未完全闭合，供下一轮接续）：

**尝试 1：MTE3_MTE2 事件同步 + 循环前预置 SetFlag + 循环内无条件 WaitFlag（verbatim）**：

```cpp
    // Cross-iteration sync: MTE3_MTE2 ensures previous iteration's CopyOut (MTE3)
    // is done before next iteration's CopyIn (MTE2) overwrites B0.
    // Set initial flag so first iteration's WaitFlag doesn't block.
    int32_t evMTE3MTE2 = (int32_t)GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2);

    for (int64_t l = 0; l < L_; l++) {
        // Wait for previous iteration's MTE3 (CopyOut) to complete before overwriting B0
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2);

        // ===== S1: CopyIn keys bf16 → B0, Cast → B1 fp32 =====
        CopyInRowBf16((b * L_ + l) * HC_ * D_ + hc * D_, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2V);
        asc_vf_call<CastBf16ToF32RowVF>(
            (__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),
            (__ubuf__ bfloat16_t*)bufB0_.Get<T>().GetPhyAddr(),
            (uint16_t)dAlignElems_, dRep_);
```

并在循环尾 CopyOut 之后补 `SetFlag<MTE3_MTE2>(evMTE3MTE2)`。**结果：死锁**——测试进程 120 秒超时被 shell 终止。循环外预置的 `SetFlag` 不能作为"首轮不阻塞"的占位信号，会破坏事件配对导致 WaitFlag 永久阻塞。

**尝试 2：条件 WaitFlag（仅 l>0 时等待，verbatim）**：

```cpp
    // Cross-iteration sync: MTE3_MTE2 ensures previous iteration's CopyOut (MTE3)
    // is done before next iteration's CopyIn (MTE2) overwrites B0.
    int32_t evMTE3MTE2 = (int32_t)GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);

    for (int64_t l = 0; l < L_; l++) {
        // Wait for previous iteration's MTE3 (CopyOut) to complete before overwriting B0
        if (l > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2);
        }
```

**结果：死锁消除，但 507035 仍复现**（该事件对 V→MTE2 的覆盖不足）。

**尝试 3：用 `AscendC::PipeBarrier<PIPE_ALL>()` 替换全部事件同步**。**结果：全零输入用例主输出转正确，但 l≥1 精度仍不达标**——`PipeBarrier` 不能等待异步 `asc_vf_call` 的 VF 流水线完成。`PIPE_ALL`/`PIPE_V` 是全局枚举、不属 `AscendC` 命名空间的编译坑另见 [PipeBarrier 的 PIPE_V 是全局枚举卡](../compilation/pipebarrier_pipe_v_uses_global_enum.md)。

接续方向（本轨迹未实现）：TQue 队列化（EnQue/DeQue 自带同步语义）或双 buffer 交替。迭代内 MTE2→V 同步的正确顺序（先同步再 Cast 消费）另见 [DataCopyPad 后未先同步 Cast 卡](../precision/mte2_barrier_must_precede_vector_cast.md)——排查本竞态时先确认迭代内顺序没有欠账，再追跨迭代反向依赖。

## 坏实践

- 循环前预置 `SetFlag<MTE3_MTE2>` 给首轮 `WaitFlag` "放行"——直接死锁（120s 超时），事件必须由真实指令流配对发出。
- 用 `PipeBarrier<PIPE_ALL>` 一把梭覆盖所有同步点——对 `asc_vf_call` 异步 VF 流水线无效，l≥1 仍错。
- 把"l=0 对、l≥1 错"当作数值精度问题去查公式/容差——方向完全错误，这是同步问题。

## 误诊

同会话早期曾把整体崩溃归因为"标量与 Vector 混用同一 UB buffer 导致 S/V 同步断裂"，并据此整体重写为 `__simd_vf__` + `asc_vf_call` 主链；重写后 PRINTF 探针才把问题精化到"跨迭代 MTE2/V 竞态"这一层。整体重写是必要的（主链合规要求），但对本竞态本身不是充分修复。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0。
- 相关文件：operators/<pkg>/op_kernel/arch35/bench_harness_engram_gate_fusion_kernel.h（ProcessOneBhc 的 L 循环：CopyIn MTE2 写 B0 与 asc_vf_call 消费 B0 交替）（路径按语料脱敏策略泛化）。

## 验证信号

本卡验证来自该算子一次崩溃修复开发会话的 kernel 调试循环实测（PRINTF 探针 + 逐轮重编上板运行）。

- 探针观测：l=0 `statK=0.028437` 正确，l=1 `statK=239069803195275788943360000.000000`、l=2 `statK=inf`（来源平台：950；证据等级：tool_observed）。
- 预置 SetFlag 方案：测试进程 `timeout 120000 ms` 超时被终止，判死锁（来源平台：950；证据等级：tool_observed）。
- 条件 WaitFlag 方案：不再超时，但 `ACL err 507035` 仍复现（来源平台：950；证据等级：tool_observed）。
- PipeBarrier 方案：全零输入用例主输出比对 0/64 不匹配（数值正确），其余用例仍 bitwise mismatch（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判别规则基于单算子单轨迹；"l=0 对、l≥1 错"也可能由 UB 容量越界等其他跨迭代状态引起，需用 PRINTF 探针确认具体中间量后再定论。
- `MTE3_MTE2` 事件语义（FetchEventID 配额、SetFlag 必须由哪个指令流发出）未从官方文档逐条验证，死锁机制解释为行为层推断。
- 本卡不含已验证修复；三条路线的失败结论绑定 CANN 9.2.0 + 3510 后端。

**未决项**：

- 优先项：为 `asc_vf_call` 异步 VF 找到可靠完成等待原语并验证（TQue 队列化改造是首选候选），本轨迹终态精度未达标即卡在此。
- 机制层：V 流水线对 UB 写入的完成可见性语义（PipeBarrier 覆盖范围、事件对 V→MTE2 的支持）未从文档/微架构确认。
