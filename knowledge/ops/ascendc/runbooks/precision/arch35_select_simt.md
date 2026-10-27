---
type: "Runbook"
title: "arch35 Select 异步SIMT同步失效致分支错选"
description: "arch35 (DAV_3510, Ascend950) 上 Select 的 VSEL_CMPMASK_SPR 模式内部 set_cmpmask 为异步 SIMT，而 arch35 无 asc_sync_vec 实现、PipeBarrier<PIPE_V> 无法保证 vsel 前 CMPMASK 就绪，导致分支选择非确定地取到上一轮 stale CMPMASK 值。诊断方法与可用/失效同步手段的实测结论。"
tags: ["task.debugging", "topic.precision", "topic.implementation.synchronization", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-17T03:44:13Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["async_SIMT"]
---

# arch35 Select 异步SIMT同步失效致分支错选

## 现象

在一次 Ascend C `softplus_v2_grad` 算子精度调试会话中，对 arch35（DAV_3510，目标 SoC=ascend950，CANN 9.0.0-beta.2）的 `op_kernel/arch35/softplus_v2_grad_kernel.h` 走 `bash build.sh --soc=ascend950` 编译并通过 `custom_opp_*.run` 安装、torch adapter 构建完成后，批跑 ST 用例 578 个中 115 个失败，全部失败集中在 L1（极端 beta/threshold）用例，L0（正常 beta/threshold）用例全 PASS。

失败的形态不是全零或 NaN，而是**分支选择错乱**：当 `bx = beta*self > threshold` 时本应取线性分支（返回 `grad_output`），kernel 却产出 sigmoid 分支值（`sigmoid(bx_clamped)*grad_output`，例如 `bx=0` 时返回 `0.5*grad_output` 而非 `grad_output`）；反之亦然。更关键的是：**同一组输入在不同执行顺序下结果不同**——把某用例单独跑能 PASS，紧跟在其它用例后跑就 FAIL。kernel 的分支选择走 `Compare<float,uint8_t>(bMask, bBx, bThr, GT, count_padded)` 产出 mask，再由 `Select<float, uint8_t>(bBx, bMask, bG, bThr, VSEL_CMPMASK_SPR, count_padded)` 在线性分支（src0=bG）与 sigmoid 分支（src1=bThr/grad_sig）间选择，其中 `count_padded` 取 64 元素对齐、而 `Muls` 等只填 `count` 个有效元素，padding 区为 stale 数据。

kernel 代码注释已写明：`arch35 上 asc_vf_call 底层为 async SIMT，且无 asc_sync_vec 同步原语，传统 V_MTE3 flag 无法捕获 async SIMT VF 完成`。本次调试验证了这一限制在 `Select` 路径上的具体后果。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发，kernel 直调（多核切分 arch35 专用 kernel，CopyIn→ComputeFp32Core→CopyOut 流水线，V 管线承载 Muls/Compare/Select/Exp/Div 等向量运算）。
- **API 路径**：`AscendC::Select<T, uint8_t>(dst, mask, src0, src1, VSEL_CMPMASK_SPR, count)` 与 `Compare<T,uint8_t>(..., GT, count_padded)`；底层为 CANN `include/pto/npu/a2a3/TSel.hpp` 的 `set_cmpmask` + `vsel` 实现。
- **验证过的平台与 CANN 版本**：Ascend950（arch=DAV_3510/NPU_ARCH 3510），CANN 9.0.0-beta.2（`$ASCEND_HOME_PATH = /home/developer/Ascend/cann-9.0.0-beta.2`）。
- **适用粒度**：`platform_wide`——结论是 arch35 平台的同步原语缺失与 Select 实现机制，与具体算子无关，凡在 arch35 上用 `VSEL_CMPMASK_SPR` 做依赖前序向量运算结果的分支选择都受影响。
- **路线优先级**：`banned`——在 arch35 上用 `Compare`+`Select(VSEL_CMPMASK_SPR)` 做依赖性敏感的逐元素分支选择是禁用路线（无有效生产级同步手段保证 CMPMASK 在 vsel 前就绪，见 `## 失效边界`）。
- **不适用/未验证场景**：arch 2201（`asc_sync_vec` 有实现，本结论不适用）；非 arch35 平台未验证；本 trace 未找到一个能在 arch35 上通过生产同步手段替代 `Compare+Select` 的 validated 修复（算术替换尝试均失败，见 `## 失败路线`），故 `recommended_fastpath` 在本卡内留空、待后续轨迹补全。

## 根因

arch35 上 `Select` 的 `VSEL_CMPMASK_SPR` 模式内部流程为：先把 mask 张量 `vector_dup` 到临时 buffer，再 `set_cmpmask(cmpMaskPtr)` 加载 CMPMASK 寄存器，然后 `vsel` 依据 CMPMASK 在 src0/src1 间选择。`set_cmpmask` 与 `vsel` 之间仅靠 `pipe_barrier(PIPE_V)` 排序。但 arch35 的向量指令底层是异步 SIMT（`asc_vf_call`），`pipe_barrier(PIPE_V)` 无法捕获 async SIMT VF 的完成；而 arch35 头文件里**声明了 `asc_sync_vec` 却只在 `npu_arch_2201` 有实现、`npu_arch_3510` 无实现**，`asc_sync` 同样未实现——arch35 没有向量管线粒度的同步原语。因此 `vsel` 可能在 `set_cmpmask` 尚未把新 mask 写入 CMPMASK 寄存器前就执行，读到上一轮调用残留的 stale CMPMASK 值，分支选择非确定地错乱。同时 `Compare` 用 `count_padded`（64 元素对齐）但有效输入只填了 `count` 个，padding 区的 stale 数据被 `set_cmpmask` 一并加载，进一步污染 CMPMASK 位。机制链：`set_cmpmask(async) → pipe_barrier(PIPE_V) 不阻塞 async SIMT → vsel 读 stale CMPMASK → 分支错选 → 非确定性数值错`。

**TSel.hpp 中 `VSEL_CMPMASK_SPR` 模式 Select 实现**（agent 在 reasoning 中逐字引用自 `include/pto/npu/a2a3/TSel.hpp`，这是本卡根因的机制证据）：

```cpp
set_mask_count();
for (unsigned i = 0; i < validRow; i++) {
    set_vector_mask(0, cmpmaskLen);
    maskAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(maskPtr + i * maskRowStride));
    vector_dup(cmpMaskPtr, maskAddr, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    set_cmpmask(cmpMaskPtr);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, validCol);
    vsel(dstPtr + i * dstRowStride, src0Ptr + i * src0RowStride,
         src1Ptr + i * src1RowStride, 1, 1, 1, 1, 8, 8, 8, SELMODE::VSEL_TENSOR_TENSOR_MODE);
}
set_mask_norm();
set_vector_mask(-1, -1);
```

可见 `set_cmpmask` 与 `vsel` 之间仅靠 `pipe_barrier(PIPE_V)`；arch35 上该 barrier 不阻塞 async SIMT，CMPMASK 可能尚未就绪即被 `vsel` 读取。

## 判别方法

**触发**：逐元素分支选择算子（`where(mask, A, B)` 语义）在 arch35 上出现"部分 case 错、同一 case 不同执行顺序结果不同、单独跑 PASS 批跑 FAIL"的精度失败，且 L0（常规参数）全过、L1（极端参数）大批失败时排查同步路径。

**探针**（两步，均在本 trace 内实测）：

1. **GetValue/PRINTF 排空探针**：在可疑的 `Select`（或任何依赖前序向量运算结果的指令）之后，对**多个**相关缓冲区同时调用 `GetValue`，例如：

```cpp
AscendC::PRINTF("D2 mask[0]=%f mask[1]=%f\n",
                bMaskF.GetValue(0), bMaskF.GetValue(1));
AscendC::PRINTF("D2 gs[0]=%f gs[1]=%f res[0]=%f res[1]=%f\n",
                bThr.GetValue(0), bThr.GetValue(1),
                bBx.GetValue(0), bBx.GetValue(1));
```

PRINTF 在 arch35 由 `PrintfImpl` 实现，其参数（GetValue 调用）必须在函数调用前求值，编译器无法把带副作用的硬件读优化掉，从而强制 V 管线排空。注意：`AscendC::PRINTF` 在 arch35 上仅当构建定义 `ASCENDC_DUMP` 时才真正输出，但**参数侧的 GetValue 无论如何都会执行**，排空效果不依赖打印是否落地。

2. **隔离执行探针**：把可疑用例单独跑一次，再把它紧跟在其它用例后跑一次，对比结果。若"单独跑 PASS、批跑 FAIL"或"同输入两次结果不同"，即同步/状态依赖。

**原始观测**：

- 插入 PRINTF（GetValue 多缓冲区）后，全部原失败用例 PASS、中间值（mask、grad_sig、result）均正确；移除 PRINTF 后同一批用例恢复失败（非确定性）。
- `pos_thr`（threshold=20, bx=[0,100]）单独跑 PASS，紧跟其它 case 后 FAIL；`veryneg_thr`（threshold=-1e38）即使单独跑也 FAIL——前者印证 stale 状态依赖，后者还混入了算术溢出问题（见 `## 失败路线`）。
- `Compare` 产出的 mask 经 GetValue 读到 `1`、`2`、`255` 等**位值**（bit-packed，而非 0x00/0xFF 字节），印证 padding 位被一并加载进 CMPMASK。

**解释**：arch35 上 `set_cmpmask` 为 async SIMT，`pipe_barrier(PIPE_V)` 不阻塞其完成，arch35 又无 `asc_sync_vec` 向量管线同步原语（仅在 `npu_arch_2201` 有实现），`vsel` 因此可能读到 stale CMPMASK。判别要点是"PRINTF/GetValue 排空后即正确"这一正反对照——它把"公式/数值逻辑错"与"同步错"区分开：若加 GetValue 后中间值正确且结果通过，则问题在同步而非算式。

## 验证信号

本卡验证信号来自上述 softplus_v2_grad arch35 调试会话（同一 trace 内实测，无外部 hearsay）。

- 批跑 578 用例 115 失败，失败全集中在 L1 极端参数 case、L0 常规参数用例全 PASS（来源平台：950；证据等级: tool_observed）。
- 同一输入（如 `pos_thr` beta=1 threshold=20, shape=(2,)）单独跑 PASS、紧跟其它 case 后跑 FAIL，结果随执行顺序变化（来源平台：950；证据等级: tool_observed）。
- 在 Select 后插入 PRINTF(GetValue 多缓冲区) 后全部用例 PASS、中间值正确；移除 PRINTF 后恢复失败（来源平台：950；证据等级: tool_observed）。
- 单独 `PipeBarrier<PIPE_V>()`、`PipeBarrier<PIPE_ALL>()`、`SetFlag/WaitFlag<V_MTE3>`、以及对单一结果缓冲区 `GetValue(0)` 均不能使失败用例通过；仅对多个缓冲区同时 GetValue（PRINTF 副作用）有效（来源平台：950；证据等级: tool_observed）。
- CANN `include/pto/npu/a2a3/TSel.hpp` 实现确认 `VSEL_CMPMASK_SPR` 流程为 `set_cmpmask`+`pipe_barrier(PIPE_V)`+`vsel(VSEL_TENSOR_TENSOR_MODE)`（来源平台：950；证据等级: artifact_verified）。
- `grep -rn asc_sync_vec` 在 `impl/c_api/instr_impl/npu_arch_3510/` 无命中，仅在 `npu_arch_2201` 有实现；`asc_sync` 同样无 arch35 实现；仅 `asc_sync_pipe(=pipe_barrier)` 与 `asc_sync_notify/wait(=set_flag/wait_flag)` 可用（来源平台：950；证据等级: tool_observed）。

## 失败路线

- **路线 A（banned）**：`Compare`+`Select(VSEL_CMPMASK_SPR)` 在 arch35 做依赖前序向量运算结果的逐元素分支选择——非确定地取 stale CMPMASK，禁用。
- **路线 B（未成功）**：算术掩码 `max(min(diff*LARGE,1.0),0.0)` 绕开 CMPMASK——在常规 threshold 下把非确定性降为确定性（进步），但极端 threshold 下因溢出路径仍错；未找到能同时满足常规与极端 threshold 的算式。
- **路线 C（未成功）**：在 ComputeFp32Core 末尾或 Process 内加 `PipeBarrier<PIPE_V>`/`PIPE_ALL`/`SetFlag V_MTE3`/单缓冲 `GetValue(0)`——实测均无效，arch35 无向量管线粒度同步原语是根因。
- **相对代价**：本卡不涉及两条可工作路线间的性能权衡（trace 内无可工作生产路线）；唯一"能工作"的手段是 GetValue/PRINTF 多缓冲区排空，但它是 debug-only、带打印副作用、不可作生产同步。

## 坏实践

- 在 arch35 上假设 `PipeBarrier<PIPE_V>()` 能保证前序 async SIMT 向量指令对后继指令可见——实测不成立。
- 用 `Compare(..., count_padded)` 但上游只填 `count` 个有效元素、不初始化 padding——stale padding 被 `set_cmpmask` 一并加载，污染 CMPMASK 位。
- 看到"加 PRINTF 后就过了"就以为问题已修、直接交付——PRINTF 的 GetValue 是 debug 同步副作用，不是生产修复，移除即复现。

## 误诊

- 早期一度把失败归因于"float16/bfloat16 的 NEED_CAST 路径特有"（因为第一批样本里 fp32 似乎通过）——后被多 dtype 复跑推翻：失败与 dtype 无关、与执行顺序有关，所有 dtype 在 `veryneg_thr`/`pos_thr` 上都失败。该误诊源于早期测试样本的执行顺序巧合（agent_reported，非实测结论，仅作误诊记录）。
- 早期一度怀疑 threshold 属性被读为 0——被 L1_010 等用例的反例推翻。
- 关于 `Mul(1e38, -50)→0` 的 arch35 溢出行为：是 agent 由"纸面公式得 [1,2]、实测得 [0.5,0.238]"反推出的假设，**未通过 PRINTF 直接观测 Mul 中间值确证**；agent 同时枚举了 `Adds(大标量)→0`、`Div(1,1)→0` 等多种假设均未确证。此机制不进 `## 验证信号`，仅作未确证推断记录。

## 预防

- 在 arch35 写依赖前序向量运算结果的逐元素分支选择前，先确认是否真有向量管线同步原语可用（`grep asc_sync_vec` 在 `npu_arch_3510` 是否有实现）；若无，避免 `VSEL_CMPMASK_SPR`，改用不依赖跨指令中间态的算式或上位框架的 `aclnn` 选择算子。
- 用 `Compare` 时，要么把 padding 区也用合法数据填满（如填 threshold 使 mask=False），要么只用 `count` 而非 `count_padded`，避免 stale padding 进 CMPMASK。
- 把"加 PRINTF/GetValue 后通过"当作诊断信号而非修复：它定位的是同步缺口，交付前必须找到生产级同步手段或绕开路径。

## 失效边界

- 本结论在 arch35（DAV_3510, Ascend950, CANN 9.0.0-beta.2）实测；arch 2201 有 `asc_sync_vec` 实现，本结论不适用（未验证，推断不适用）。
- 其它 NPU 架构（a2/a3 等注册平台）未在 trace 内实测，**未验证，推断可迁移性需逐平台核对 `asc_sync_vec` 是否有实现**，不得直接套用。
- `asc_sync_vec`/`asc_sync` 在 arch35 的"声明但无实现"是 CANN 9.0.0-beta.2 的当前状态；后续 CANN 版本若补实现，本卡 banned 结论需复审。
- 本卡不含源算子常量（softplus_v2_grad 的具体 beta/threshold/shape 仅作判别信号示例，不构成知识结论）；结论是平台级同步原语缺失。

## 系统信息

- 目标平台：Ascend950（AICore arch=DAV_3510，`__NPU_ARCH__==3510`），SoC=ascend950。
- CANN 版本：9.0.0-beta.2（`$ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0-beta.2`）。
- 工具链入口：`bash build.sh --soc=ascend950` → `bash build/custom_opp_*.run --quiet` → torch adapter `cmake`/`make` → `python3 test.py --lib ./build/libcustom_*.so`。
- 相关 CANN 头文件（与事件流 read/grep 一一对应）：
  - `include/pto/npu/a2a3/TSel.hpp`（`VSEL_CMPMASK_SPR` Select 实现）
  - `include/pto/npu/a2a3/TSels.hpp`（`TENSOR_SCALAR_MODE` 亦用 `set_cmpmask`）
  - `x86_64-linux/asc/include/c_api/sync/sync.h`（`asc_sync_vec`/`asc_sync` 声明）
  - `x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510/sync_impl.h` 与 `.../sync_impl/asc_sync_pipe_impl.h`（arch35 仅 `asc_sync_pipe=pipe_barrier`、`asc_sync_notify/wait=set/wait_flag`；无 `asc_sync_vec`/`asc_sync` 实现）
  - `x86_64-linux/asc/impl/basic_api/kernel_operator_dump_tensor_intf_impl.h`（PRINTF/PrintfImpl，arch35 需 `ASCENDC_DUMP`）
- 算子工程文件（arch35 专用）：`op_kernel/arch35/softplus_v2_grad_kernel.h`、`op_kernel/arch35/softplus_v2_grad_struct.h`、`op_kernel/arch35/softplus_v2_grad_tiling_struct.h`、`op_host/arch35/softplus_v2_grad_tiling_arch35.cpp`。

## 设计决策

- **为何用 `VSEL_CMPMASK_SPR`**：算子需要按 `bx > threshold` 在线性分支与 sigmoid 分支间逐元素选择，`Compare`+`Select(VSEL_CMPMASK_SPR)` 是 AscendC 表达 `where(mask, A, B)` 的标准向量通路。本 trace 证明该通路在 arch35 因缺少向量管线同步原语而不可靠。
- **被放弃的候选与原因**：
  1. 算术替换 `mask_f = max(min(diff*LARGE, 1.0), 0.0)` + `result = grad_sig + mask_f*(grad_output - grad_sig)`——绕开 CMPMASK，但 `Sub`/`Add` 张量-张量运算在 arch35 同样是异步 SIMT，仍需同步；且 `Mul(positive_diff, bx-50)` 在极端 threshold（如 -1e38）下 fp32 溢出，疑似 arch35 产出 0 而非 ±inf（未直接观测确证），导致 `veryneg_thr` 即便单独跑仍 FAIL。
  2. 除法优先重排 `ratio = positive_diff / max(positive_diff, eps)` 再乘——仍失败，疑似 `Adds(标量 1e38)` 或 `Mul` 大数路径产出 0（未确证）。
  3. `PipeBarrier<PIPE_V>`/`PipeBarrier<PIPE_ALL>`/`SetFlag/WaitFlag<V_MTE3>`/单缓冲 `GetValue(0)`——均不能使失败用例通过；仅多缓冲区同时 GetValue（PRINTF 副作用）有效。
- **当前方案属哪档路线**：`banned`——`VSEL_CMPMASK_SPR` 在 arch35 做依赖性选择已实测失败；生产级替代路线在本 trace 内未 validated。