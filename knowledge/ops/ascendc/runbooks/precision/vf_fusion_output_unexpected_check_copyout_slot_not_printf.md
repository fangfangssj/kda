---
type: "Runbook"
title: "双 VF 融合输出非预期 先排 CopyOut slot 勿用 PRINTF"
description: "VF 融合(asc_vf_call/__simd_vf__)kernel 输出非预期(全零/错误)时，先排除 CopyOut buffer slot/参数顺序/buffer 映射等非 VF 因素，勿在 __simd_vf__ 函数内用 PRINTF(语义不兼容致观测假象)，用控制变量实验+参考实现交叉验证。"
tags: ["task.debugging", "topic.implementation.debugging", "topic.programming_model.execution_unit.vector", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 双 VF 融合输出非预期 先排 CopyOut slot 勿用 PRINTF

## 现象

Ascend950PR（3510 arch）上 SoftplusV2Grad arch35 kernel 的 DESIGN.md 设计了 VF 融合方案：将 `Muls→Exp→Adds→Div→Mul→Compare→Select` 7 步计算链融合为单条 `asc_vf_call<SoftplusGradVF<float>>(...)`，中间值全留寄存器，P=3（比同步向量 API 的 P=4 少一个 UB buffer）。编译通过、Kernel 二进制生成成功，但 NPU 运行 fp32 路径**输出非预期（全零）**。同期 subagent 拆成两个 VF（VFSigmoid + VFGradSelect）中间无同步的版本同样输出非预期。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发，arch35/950 kernel 用 `asc_vf_call` + `__simd_vf__`/`__simt_vf__` VF 融合，输出非预期（全零或错误值）的精度定位。
- **API 路径**：`asc_vf_call<VFFunc>(args...)` / `asc_vf_call<VFFunc>(dim3{...}, args...)` + `MicroAPI::` / `Reg::` 命名空间 VF 函数体 + `CopyOut`/`SetFlag`/`WaitFlag`/`PipeBarrier`。
- **验证平台与版本**：Ascend950PR（3510/DAV_3510，平台 token `950`），CANN cann-9.0.0-beta.2。仅 950 实测；2201(Ascend910B) 的 asc_sync_vec 实现存在，行为不同，未在本 trace 验证。
- **适用粒度**：platform_wide（3510/950 上所有用 VF 融合的 kernel）。
- **路线优先级**：recommended_fastpath（先排非 VF 因素，再做控制变量实验与参考实现交叉验证）。
- **不适用或未验证场景**：非 VF 融合的同步向量 API kernel 输出错误不套本判别；2201 的 asc_sync_vec 可用，同步假说可被排除的路径不同，未验证。

## 判别方法

**触发**：VF 融合 kernel（`asc_vf_call` + `__simd_vf__`/`__simt_vf__`）编译通过、Kernel 二进制生成成功，但 NPU 输出非预期（全零或错误值）。注意：不要把"全零"当唯一现象——输出 GM 上若恰好无脏数据，非预期结果也会读成全零；测试框架可能先往输出 GM 写脏数据；应框定为"输出非预期"而非"输出全零"。

**可执行探针（按序）**：

1. **先排除非 VF 因素**（最常见真因）：
   - `CopyOut` 的 buffer slot 指向是否正确？本 trace 的真因就是 `CopyOutOne(coord, OUT0, B2, a_i_seg)` 误指向 B2，应为 B1（见 `## 判别方法` 末 verbatim edit）。
   - VF 函数参数顺序、Load/Store 的 `LoadAlign`/`StoreAlign` 与 stride 配置是否正确。
   - 输出 GM 是否在 CopyIn 前被脏数据污染。
2. **勿在 `__simd_vf__` 函数内用 PRINTF**：PRINTF 是 `__aicore__` 语义，VF 函数是 `__simd_vf__` 语义，混用导致时序/寄存器状态异常，可能得到"UB 全零"的**假象**（本 trace 的错误路径正是 PRINTF 显示全零后误判"编译器优化掉了 StoreAlign"）。改用 `DumpTensor` 或在 VF 外部、CopyOut 后观测。
3. **控制变量实验**：固定其它变量，每次只改一个——Load/Store vs LoadAlign/StoreAlign、直接调用 `__simd_vf__` vs `asc_vf_call`、有/无 `PipeBarrier<PIPE_V>` 或全同步。本 trace 复盘中 5 个控制变量实验全部通过，证明双 VF 完全可行，之前的"必须合并为单 VF"结论是错的。
4. **参考实现交叉验证**：在 CANN 内置算子仓（ops-math/ops-cv/ops-transformer/cann-samples）搜索相同 VF 模式的参考实现；若参考实现能工作，问题一定在具体代码而非通用机制。
5. **若 2 个 VF 分开实现**：在两 VF 之间加一次全同步（`PIPE_BARRIER_ALL` 或 `SetFlag`/`WaitFlag` 对应 pipe event）验证是否同步问题——这样下次不会停在"猜测 2 个 VF 的同步问题"上；若加全同步仍非预期，排除同步假说，回到步骤 1。

**原始观测（tool_observed）**：
- CANN SDK 源码 `common_functions.h:35-42`（arch3510/5102 分支）：

```cpp
#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)
template <auto funcPtr, typename... Args>
__aicore__ inline void asc_vf_call(dim3 threadNums, Args &&...args)
{
    cce::async_invoke<funcPtr>(threadNums, args...);  // 异步
}
#endif
```

- `asc_sync_vec` 实现存在于 `npu_arch_2201/sync_impl.h`，`npu_arch_3510/sync_impl.h` 无此实现（link 阶段 undefined）。
- issue 记录（commit `09f5f57`, `issue_20260704_regbase_vf_async_sync_1.md`）对比失败 vs 成功实现：

| 维度 | 失败实现 | 成功实现 |
|------|---------|---------|
| 命名空间 | `Reg::`（DESIGN.md 字面） | `MicroAPI::`（SDK 实际） |
| 调用方式 | `asc_vf_call<SoftplusGradVF>(dim3{1}, args...)` | `asc_vf_call<SoftplusGradVF>(args...)`（无 dim3） |
| 函数修饰 | 不确定 | `__simd_vf__` |

**解释**：`asc_vf_call` 在 3510 上底层调 `cce::async_invoke`（异步），且 3510 无 `asc_sync_vec` 实现——这是真实存在的架构事实，agent 第一反应据此归因为"async VF + 同步原语缺失导致 CopyOut 抢跑读零"。但这是**误诊**：后续 5 个控制变量实验证明双 VF 可行，真因是 CopyOut slot 指向错误 buffer（B2 而非 B1）+ 非 VF 因素。架构级的 async/sync 缺失是真实约束（值得记入 `## 失效边界`），但不是本例输出非预期的根因——把架构约束当根因会跳过非 VF 因素的排查。

**真因修复（verbatim edit）**：

before:
```cpp
        if constexpr (!NEED_CAST) {
            CopyOutOne(coord, OUT0, B2, a_i_seg);
        } else {
```
after:
```cpp
        if constexpr (!NEED_CAST) {
            CopyOutOne(coord, OUT0, B1, a_i_seg);
        } else {
```

## 系统信息

- toolchain：CANN cann-9.0.0-beta.2，arch35（3510/DAV_3510，Ascend950PR）。
- 目标平台：Ascend950PR（平台 token `950`）。
- 目录布局：`operators/softplus_v2_grad/op_kernel/arch35/softplus_v2_grad_kernel.h`（kernel）、`operators/softplus_v2_grad/docs/DESIGN.md`（VF 融合设计）、`operators/softplus_v2_grad/issues/issue_20260704_regbase_vf_async_sync_1.md`（issue 记录，commit `09f5f57`）。
- 相关文件：`/home/developer/Ascend/cann-9.0.0-beta.2/x86_64-linux/asc/include/simt_api/common_functions.h`（asc_vf_call 定义）、`.../impl/c_api/instr_impl/npu_arch_3510/sync_impl.h` 与 `.../npu_arch_2201/sync_impl.h`（asc_sync_vec 实现）。

## 设计决策

- **为何选 VF 融合**：DESIGN.md 设计 VF 融合以省一个 UB buffer（P=3 vs P=4），中间值留寄存器。数学正确，但未在 DESIGN_PREP 验证目标架构（3510）的 asc_vf_call async 行为与 asc_sync_vec 支持情况。
- **被放弃的误诊路径**：① "async VF + V_MTE3 flag 无法捕获 async VF 完成"——架构事实但非本例根因；② "PRINTF 显示 UB 全零→编译器优化掉 StoreAlign"——PRINTF 在 `__simd_vf__` 内观测不可靠导致的假象；③ "双 VF 必须合并为单 VF"——5 个控制变量实验全部通过证伪。
- **当前方案属哪档路线**：recommended_fastpath = 先排非 VF 因素 + 控制变量实验 + 参考实现交叉验证。

## 验证信号

- 失败实现：VF 融合编译通过、Kernel 二进制生成成功，NPU fp32 路径输出非预期/全零（来源平台：950；证据等级: tool_observed）
- SDK 源码 grep/read：`asc_vf_call`→`cce::async_invoke`（async），3510 无 `asc_sync_vec`、2201 有（来源平台：950；证据等级: tool_observed）
- issue 记录 git show（commit `09f5f57`）：失败 vs 成功实现对比表（来源平台：950；证据等级: artifact_verified）
- 5 个控制变量实验全部通过、双 VF 可行（来源平台：950；证据等级: tool_observed）
- CopyOut slot B2→B1 修复后输出正确（来源平台：950；证据等级: tool_observed）

## 性能影响

VF 融合（P=3，中间值留寄存器）相对同步向量 API（P=4，中间值落 UB）省一个 UB buffer 槽位；UB 预算 248KB 仍够用。无绝对性能分数落正文（参照路线：同步向量 API P=4，来源平台：950）。

## 预防

- DESIGN_PREP 的 API 验证阶段，对 `asc_vf_call`/`__simd_vf__` 这类跨架构语义 API，必须查目标架构的 sync 原语支持情况（3510 无 asc_sync_vec），不能只验证 VF 函数体内部 API。
- DESIGN.md 写 `Reg::` 命名空间但 SDK 实际是 `MicroAPI::`——DESIGN 文档的命名空间/API 须与 SDK 头文件核对后再交给 developer；`dim3` 是 SIMT-VF 特有的线程数参数，SIMD VF 不传 dim3，DESIGN 不可混用。

## 失效边界

- 950PR（3510）已实测；950DT（同 DAV_3510）推断可迁移，未验证——标注"未验证，推断可迁移"。
- 2201（Ascend910B / A2 / A3）的 `asc_sync_vec` 实现存在，async/sync 假说的排除路径不同，本判别方法在 2201 上的适用性未验证。
- `asc_vf_call` 在 3510 上的 async 行为是真实架构约束：若真需要 async VF + 后续依赖，须用 `PIPE_BARRIER_ALL` 或对应 pipe event 显式同步，不能依赖 `V_MTE3` flag 捕获 async VF 完成。

## 误诊

agent 第一反应归因"asc_vf_call async + 3510 无 asc_sync_vec + V_MTE3 flag 无法捕获 async VF 完成"——这是把真实存在的架构约束误当成本例根因，跳过了非 VF 因素（CopyOut slot）的排查。错误路径：PRINTF 在 `__simd_vf__` 内显示 UB 全零 → 误推"编译器优化掉 StoreAlign" → 尝试 PipeBarrier/noinline"无效" → 误断"双 VF 必须合并为单 VF"，全程未做控制变量实验、未排除 CopyOut slot。复盘（5 个控制变量实验全通过）证伪了"必须合并"结论。